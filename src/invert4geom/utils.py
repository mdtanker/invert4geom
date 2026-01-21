import copy  # pylint: disable=too-many-lines
import os
import typing
import warnings
from contextlib import contextmanager

import dask
import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import sklearn
import verde as vd
import xarray as xr
import xrft
from numpy.typing import NDArray
from polartoolkit import fetch
from pykdtree.kdtree import KDTree  # pylint: disable=no-name-in-module

from invert4geom import logger, plotting


@contextmanager
def _log_level(level):  # type: ignore[no-untyped-def]
    "Run body with logger at a different level"
    saved_logger_level = logger.level
    logger.setLevel(level)
    try:
        yield saved_logger_level
    finally:
        logger.setLevel(saved_logger_level)


@contextmanager
def _environ(**env):  # type: ignore[no-untyped-def] # pylint: disable=missing-function-docstring
    """temporarily set/reset an environment variable"""
    originals = {k: os.environ.get(k) for k in env}
    for k, val in env.items():
        if val is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = val
    try:
        yield
    finally:
        for k, val in originals.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val


class DuplicateFilter:
    """
    Filters away duplicate log messages.
    Adapted from https://stackoverflow.com/a/60462619/18686384
    """

    def __init__(self, log):  # type: ignore[no-untyped-def]
        self.msgs = set()
        self.log = log

    def filter(self, record):  # type: ignore[no-untyped-def] # pylint: disable=missing-function-docstring
        msg = str(record.msg)
        is_duplicate = msg in self.msgs
        if not is_duplicate:
            self.msgs.add(msg)
        return not is_duplicate

    def __enter__(self):  # type: ignore[no-untyped-def]
        self.log.addFilter(self)

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[no-untyped-def]
        self.log.removeFilter(self)


def _check_constraints_inside_gravity_region(
    constraints_df: pd.DataFrame,
    grav_df: pd.DataFrame,
) -> None:
    """check that all constraints are inside the region of the gravity data"""
    grav_region = vd.get_region((grav_df.easting, grav_df.northing))
    inside = vd.inside(
        (constraints_df.easting, constraints_df.northing), region=grav_region
    )
    if not inside.all():
        msg = (
            "Some constraints are outside the region of the gravity data. "
            "This may result in unexpected behavior."
        )
        logger.warning(msg)


def rmse(data: NDArray, as_median: bool = False) -> float:
    """
    function to give the root mean/median squared error (RMSE) of data

    Parameters
    ----------
    data : numpy.ndarray
        input data
    as_median : bool, optional
        choose to give root median squared error instead, by default False

    Returns
    -------
    float
        RMSE value
    """
    if as_median:
        value: float = np.sqrt(np.nanmedian(data**2).item())
    else:
        value = np.sqrt(np.nanmean(data**2).item())

    return value


def _nearest_grid_fill(
    grid: xr.DataArray,
    method: str = "verde",
    crs: str | None = None,
) -> xr.DataArray:
    """
    fill missing values in a grid with the nearest value.

    Parameters
    ----------
    grid : xarray.DataArray
        grid with missing values
    method : str, optional
        choose method of filling, by default "verde"
    crs : str | None, optional
        if method is 'rioxarray', provide the crs of the grid, in format 'epsg:xxxx',
        by default None
    Returns
    -------
    xarray.DataArray
        filled grid
    """

    # TODO: also check out rasterio fillnodata() https://rasterio.readthedocs.io/en/latest/api/rasterio.fill.html#rasterio.fill.fillnodata
    # uses https://gdal.org/en/stable/api/gdal_alg.html#_CPPv414GDALFillNodata15GDALRasterBandH15GDALRasterBandHdiiPPc16GDALProgressFuncPv
    # can fill with nearest neighbor or inverse distance weighting

    # get coordinate names
    original_dims = list(grid.sizes.keys())

    # get original grid name
    original_name = grid.name

    if method == "rioxarray":
        filled: xr.DataArray = (
            grid.rio.write_crs(crs)
            .rio.set_spatial_dims(original_dims[1], original_dims[0])
            .rio.write_nodata(np.nan)
            .rio.interpolate_na(method="nearest")
            .rename(original_name)
        )
    elif method == "verde":
        df = vd.grid_to_table(grid)
        df_dropped = df[df[grid.name].notna()]
        coords = (df_dropped[grid.dims[1]], df_dropped[grid.dims[0]])
        region = vd.get_region((df[grid.dims[1]], df[grid.dims[0]]))
        filled = (
            vd.KNeighbors()
            .fit(coords, df_dropped[grid.name])
            .grid(region=region, shape=grid.shape, data_names=original_name)[
                original_name
            ]
        )
    # elif method == "pygmt":
    #     filled = pygmt.grdfill(grid, mode="n", verbose="q").rename(original_name)
    else:
        msg = "method must be 'rioxarray', or 'verde'"
        raise ValueError(msg)

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        return filled.rename(
            {
                next(iter(filled.dims)): original_dims[0],
                list(filled.dims)[1]: original_dims[1],
            }
        )


def region_mask(
    grid: xr.DataArray,
    region: tuple[float, float, float, float],
) -> xr.DataArray:
    """
    Make a mask with values of 1 inside a region and 0 outside the region.
    """

    mask_easting = (grid.easting >= region[0]) & (grid.easting <= region[1])
    mask_northing = (grid.northing >= region[2]) & (grid.northing <= region[3])

    return xr.where(mask_easting & mask_northing, 1, 0)


def filter_grid(
    grid: xr.DataArray,
    filter_width: float | None = None,
    height_displacement: float | None = None,
    filt_type: str = "lowpass",
    pad_width_factor: int = 3,
    pad_mode: str = "linear_ramp",
    pad_constant: float | None = None,
    pad_end_values: float | None = None,
) -> xr.DataArray:
    """
    Apply a spatial filter to a grid.

    Parameters
    ----------
    grid : xarray.DataArray
        grid to filter the values of
    filter_width : float, optional
        width of the filter in meters, by default None
    height_displacement : float, optional
        height displacement for upward continuation, relative to observation height, by
        default None
    filt_type : str, optional
        type of filter to use from 'lowpass', 'highpass' 'up_deriv', 'easting_deriv',
        'northing_deriv', 'up_continue', or 'total_gradient', by default "lowpass"
    pad_width_factor : int, optional
        factor of grid width to pad the grid by, by default 3, which equates to a pad
        with a width of 1/3 of the grid width.
    pad_mode : str, optional
        mode of padding, can be "linear", by default "linear_ramp"
    pad_constant : float | None, optional
        constant value to use for padding, by default None
    pad_end_values : float | None, optional
        value to use for end of padding if pad_mode is "linear_ramp", by default None

    Returns
    -------
    xarray.DataArray
        a filtered grid
    """
    # get coordinate names
    original_dims = list(grid.sizes.keys())

    # get original grid name
    original_name = grid.name

    # if there are nan's, fill them with nearest neighbor
    if grid.isnull().any():  # noqa: PD003
        filled = _nearest_grid_fill(grid, method="verde")
    else:
        filled = grid.copy()

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        filled = filled.rename(
            {
                next(iter(filled.dims)): original_dims[0],
                list(filled.dims)[1]: original_dims[1],
            }
        )

    # define width of padding in each direction
    pad_width = {
        original_dims[1]: grid[original_dims[1]].size // pad_width_factor,
        original_dims[0]: grid[original_dims[0]].size // pad_width_factor,
    }

    if pad_mode == "constant":
        if pad_constant is None:
            pad_constant = filled.median()
        pad_end_values = None

    if (pad_mode == "linear_ramp") and (pad_end_values is None):
        pad_end_values = filled.median()

    if pad_mode != "constant":
        pad_constant = (
            None  # needed until https://github.com/xgcm/xrft/issues/211 is fixed
        )

    # apply padding
    pad_kwargs = {
        **pad_width,
        "mode": pad_mode,
        "constant_values": pad_constant,
        "end_values": pad_end_values,
    }

    padded = xrft.pad(
        filled,
        **pad_kwargs,
    )

    if filt_type == "lowpass":
        if filter_width is None:
            msg = "filter_width must be provided if filt_type is 'lowpass'"
            raise ValueError(msg)
        filt = hm.gaussian_lowpass(padded, wavelength=filter_width).rename("filt")
    elif filt_type == "highpass":
        if filter_width is None:
            msg = "filter_width must be provided if filt_type is 'highpass'"
            raise ValueError(msg)
        filt = hm.gaussian_highpass(padded, wavelength=filter_width).rename("filt")
    elif filt_type == "up_deriv":
        filt = hm.derivative_upward(padded).rename("filt")
    elif filt_type == "easting_deriv":
        filt = hm.derivative_easting(padded).rename("filt")
    elif filt_type == "northing_deriv":
        filt = hm.derivative_northing(padded).rename("filt")
    elif filt_type == "up_continue":
        if height_displacement is None:
            msg = "height_displacement must be provided if filt_type is 'up_continue'"
            raise ValueError(msg)
        filt = hm.upward_continuation(
            padded, height_displacement=height_displacement
        ).rename("filt")
    elif filt_type == "total_gradient":
        filt = hm.total_gradient_amplitude(padded).rename("filt")
    else:
        msg = (
            "filt_type must be 'lowpass', 'highpass' 'up_deriv', 'easting_deriv', "
            "'northing_deriv', 'up_continue', or 'total_gradient'"
        )
        raise ValueError(msg)

    unpadded = xrft.unpad(filt, pad_width)

    # reset coordinate values to original (avoid rounding errors)
    unpadded = unpadded.assign_coords(
        {
            original_dims[0]: grid[original_dims[0]].to_numpy(),
            original_dims[1]: grid[original_dims[1]].to_numpy(),
        }
    )

    if grid.isnull().any():  # noqa: PD003
        result: xr.DataArray = xr.where(grid.notnull(), unpadded, grid)  # noqa: PD004
    else:
        result = unpadded.copy()

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        result = result.rename(
            {
                next(iter(result.dims)): original_dims[0],
                # list(result.dims)[0]: original_dims[0],
                list(result.dims)[1]: original_dims[1],
            }
        )

    return result.rename(original_name)


def dist_nearest_points(
    targets: pd.DataFrame,
    data: pd.DataFrame | xr.DataArray | xr.Dataset,
    coord_names: tuple[str, str] | str | None = None,
) -> typing.Any:
    """
    for all points in `data` calculate to the distance to the nearest `target`. Can be
    any dimension (1D, 2D, 3D, etc) as long as the length of the provided
    `coord_names` matches the dimension of the data.

    Parameters
    ----------
    targets : pandas.DataFrame
        contains the coordinates of the targets
    data : pandas.DataFrame | xarray.DataArray | xarray.Dataset
        the grid data, in either gridded or tabular form
    coord_names : tuple[str, str] | str | None, optional
        the names of the coordinate(s) for both the targets and the data, by default None

    Returns
    -------
    typing.Any
        the distance to the nearest target for each gridcell, in the same format as the
        input for `data`.
    """

    if coord_names is None:
        coord_names = ("easting", "northing")

    if isinstance(coord_names, str):
        coords = coord_names
    else:
        coords = list(coord_names)

    targets = targets[coords].to_numpy()

    original_data = data.copy()

    data: pd.DataFrame | xr.DataArray | xr.Dataset
    if isinstance(data, pd.DataFrame):
        data_df = data.copy()
        coords_df = data_df[coords].copy()
    elif isinstance(data, xr.DataArray):
        df_grid = vd.grid_to_table(data).dropna()
        data_df = df_grid.copy()
        coords_df = data_df[coords].copy()
    elif isinstance(data, xr.Dataset):
        try:
            df_grid = vd.grid_to_table(data[next(iter(data.variables))]).dropna()
        except IndexError:
            df_grid = vd.grid_to_table(data).dropna()
        data_df = df_grid.copy()
        coords_df = data_df[coords].copy()

    coords_np = coords_df.to_numpy()

    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if coords_np.ndim == 1:
        coords_np = coords_np.reshape(-1, 1)

    min_dist, _ = KDTree(targets).query(coords_np, k=1)

    data_df["min_dist"] = min_dist

    if isinstance(original_data, pd.DataFrame):
        return data_df
    if isinstance(original_data, xr.Dataset):
        return data_df.set_index(coords[::-1]).to_xarray()
    if isinstance(original_data, xr.DataArray):
        return data_df.set_index(coords[::-1]).to_xarray().min_dist
    msg = "data must be pandas.DataFrame, xarray.DataArray, or xarray.Dataset"
    raise ValueError(msg)


def normalize(
    x: NDArray,
    low: float = 0,
    high: float = 1,
) -> NDArray:
    """
    Normalize a list of numbers between provided values

    Parameters
    ----------
    x : NDArray
        numbers to normalize
    low : float, optional
        lower value for normalization, by default 0
    high : float, optional
        higher value for normalization, by default 1

    Returns
    -------
    NDArray
        a normalized list of numbers
    """
    min_val = np.min(x)
    max_val = np.max(x)

    norm = (x - min_val) / (max_val - min_val)

    return norm * (high - low) + low


def normalize_xarray(
    da: xr.DataArray,
    low: float = 0,
    high: float = 1,
) -> xr.DataArray:
    """
    Normalize a grid between provided values

    Parameters
    ----------
    da : xarray.DataArray
        grid to normalize
    low : float, optional
        lower value for normalization, by default 0
    high : float, optional
        higher value for normalization, by default 1

    Returns
    -------
    xarray.DataArray
        a normalized grid
    """
    # min_val = da.to_numpy().min()
    # max_val = da.to_numpy().max()

    da = da.copy()

    min_val = da.quantile(0)
    max_val = da.quantile(1)

    da2: xr.DataArray = (high - low) * (
        ((da - min_val) / (max_val - min_val)).clip(0, 1)
    ) + low

    return da2.drop_vars("quantile")


def scale_normalized(
    sample: NDArray,
    bounds: tuple[float, float],
) -> NDArray:
    """
    Rescales the sample space into the unit hypercube, bounds = [0,1]

    Parameters
    ----------
    sample : NDArray
        sampled values
    bounds : tuple
        bounds of the sampling

    Returns
    -------
    NDArray
        sampled values normalized from 0 to 1
    """
    print(sample.shape)
    scaled_sample = np.zeros(sample.shape)
    print(scaled_sample)
    for j in range(sample.shape[1]):
        scaled_sample[:, j] = (sample[:, j] - bounds[0]) / (bounds[1] - bounds[0])

    return scaled_sample


def normalized_mindist(
    points: pd.DataFrame,
    grid: xr.DataArray,
    low: float | None = None,
    high: float | None = None,
    mindist: float | None = None,
    region: list[float] | None = None,
) -> xr.DataArray:
    """
    Find the minimum distance between each grid cell and the nearest point. If low and
    high are provided, normalize the min dists grid between these values. If region is
    provided, all grid cells outside region are set to a distance of 0.

    Parameters
    ----------
    points : pandas.DataFrame
        coordinates of the points
    grid : xarray.DataArray
        gridded data to find min dists for each grid cell
    low : float | None, optional
        lower value for normalization, by default None
    high : float | None, optional
        higher value for normalization, by default None
    mindist : float | None, optional
        the minimum allowed distance, all values below are set equal to, by default None
    region : list[float] | None, optional
        bounding region for which all grid cells outside will be set to low, by default
        None

    Returns
    -------
    xarray.DataArray
        grid of normalized minimum distances
    """

    grid = copy.deepcopy(grid)

    # get coordinate names
    original_dims = list(grid.sizes.keys())

    constraints_df = points.copy()

    min_dist: xr.DataArray = dist_nearest_points(
        targets=constraints_df,
        data=grid,
        coord_names=(str(original_dims[1]), str(original_dims[0])),
    )

    # set points < mindist to low
    if mindist is not None:
        min_dist = xr.where(min_dist < mindist, 0, min_dist)

    # set points outside of region to low
    if region is not None:
        df = vd.grid_to_table(min_dist)
        df["are_inside"] = vd.inside(
            (df[original_dims[1]], df[original_dims[0]]),
            region=region,
        )
        new_min_dist = df.set_index([original_dims[0], original_dims[1]]).to_xarray()
        new_min_dist = xr.where(new_min_dist.are_inside, new_min_dist, 0)

        # add nans back
        new_min_dist = xr.where(min_dist.isnull(), np.nan, new_min_dist)  # noqa: PD003

        min_dist = new_min_dist.min_dist

    # normalize from low to high
    if (low is None) & (high is None):
        pass
    else:
        assert low is not None
        assert high is not None
        min_dist = normalize_xarray(min_dist, low=low, high=high)

    return min_dist


def sample_grids(
    df: pd.DataFrame,
    grid: str | xr.DataArray,
    sampled_name: str,
    **kwargs: typing.Any,
) -> pd.DataFrame:
    """
    Sample data at every point along a line

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing columns 'x', 'y', or columns with names defined by kwarg
        "coord_names".
    grid : str or xarray.DataArray
        Grid to sample, either file name or xarray.DataArray
    sampled_name : str,
        Name for sampled column

    Returns
    -------
    pandas.DataFrame
        Dataframe with new column (sampled_name) of sample values from (grid)
    """

    # drop name column if it already exists
    try:
        df1 = df.drop(columns=sampled_name)
    except KeyError:
        df1 = df.copy()

    if "index" in df1.columns:
        msg = "index column must be removed or renamed before sampling"
        raise ValueError(msg)

    df2 = df1.copy()

    # reset the index
    df3 = df2.reset_index()

    # get x and y column names
    x, y = kwargs.get("coord_names", ("easting", "northing"))

    # check column names exist, if not, use other common names
    if (x in df3.columns) and (y in df3.columns):
        pass
    elif ("x" in df3.columns) and ("y" in df3.columns):
        x, y = ("x", "y")

    # get points to sample at
    points = df3[[x, y]].copy()

    # sample the grid at all x,y points
    sampled = pygmt.grdtrack(
        points=points,
        grid=grid,
        newcolname=sampled_name,
        # radius=kwargs.get("radius", None),
        no_skip=True,  # if false causes issues
        verbose=kwargs.get("verbose", "w"),
        interpolation=kwargs.get("interpolation", "c"),
    )

    df3[sampled_name] = sampled[sampled_name]

    # reset index to previous
    df4 = df3.set_index("index")

    # reset index name to be same as originals
    df4.index.name = df1.index.name

    # check that dataframe is identical to original except for new column
    pd.testing.assert_frame_equal(df4.drop(columns=sampled_name), df1)

    return df4


def get_spacing(prisms_df: pd.DataFrame) -> None:  # noqa: ARG001
    """
    DEPRECATED: function has been removed
    """
    # pylint: disable=W0613
    msg = "Function `get_spacing` deprecated"
    raise DeprecationWarning(msg)


def sample_bounding_surfaces(
    prisms_df: pd.DataFrame,
    upper_confining_layer: xr.DataArray | None = None,
    lower_confining_layer: xr.DataArray | None = None,
) -> pd.DataFrame:
    """
    sample upper and/or lower confining layers into prisms dataframe

    Parameters
    ----------
    prisms_df : pandas.DataFrame
        dataframe of prism properties
    upper_confining_layer : xarray.DataArray | None, optional
        layer which the inverted topography should always be below, by default None
    lower_confining_layer : xarray.DataArray | None, optional
        layer which the inverted topography should always be above, by default None

    Returns
    -------
    pandas.DataFrame
        a dataframe with added columns 'upper_bounds' and 'lower_bounds', which are the
        sampled values of the supplied confining grids.
    """
    df = prisms_df.copy()

    if upper_confining_layer is not None:
        df = sample_grids(
            df=df,
            grid=upper_confining_layer,
            sampled_name="upper_bounds",
        )
        assert len(df.upper_bounds) != 0
    if lower_confining_layer is not None:
        df = sample_grids(
            df=df,
            grid=lower_confining_layer,
            sampled_name="lower_bounds",
        )
        assert len(df.lower_bounds) != 0
    return df


def create_topography(
    method: str,
    region: tuple[float, float, float, float],
    spacing: float,
    dampings: list[float] | None = None,
    registration: str = "g",
    upwards: float | None = None,
    constraints_df: pd.DataFrame | None = None,
    weights: pd.Series | NDArray | None = None,
    weights_col: str | None = None,
    block_size: float | None = None,
    block_reduction: str = "median",
    upper_confining_layer: xr.DataArray | None = None,
    lower_confining_layer: xr.DataArray | None = None,
) -> xr.Dataset:
    """
    Create a grid of topography data from either the interpolation (with splines) of
    point data or creating a grid of constant value. Optionally, a subset of point data
    can be interpolated and then merged with an existing grid. To do this,
    ``constraints_df`` must contain two additional columns of booleans, ``inside`` which
    is True for points inside the region of interest, and False otherwise, and
    ``buffer`` which is True for points within a buffer region around the region of
    interest, and False otherwise. Inside and Buffer points are used to interpolated the
    data, and then the interpolated data (without the buffer zone) is merged with the
    points outside the region of interest. For interpolations, ``block_size`` can be
    supplied to perform a block-median filtering of the points before fitting the
    spline, reducing the computational cost.

    Parameters
    ----------
    method : str
        method to use, either ``flat`` or ``splines``
    region : tuple[float, float, float, float]
        region of the grid
    spacing : float
        spacing of the grid
    dampings : list[float] | None, optional
        damping values to use in spline cross validation for method ``spline``, by default
        None
    registration : str, optional
        choose between gridline ``g`` or pixel ``p`` registration, by default ``g``
    upwards : float | None, optional
        constant value to use for method ``flat``, by default None
    constraints_df : pandas.DataFrame | None, optional
        dataframe with column 'upwards' to use for method ``splines``, and optionally
        columns ``inside`` and ``buffer``, by default None
    weights : pandas.Series | numpy.ndarray | None, optional
        weight to use for fitting the spline. Typically, this should be 1 over the data
        uncertainty squared, by default None
    weights_col : str | None, optional
        instead of passing the weights, pass the name of the column containing the
        weights, by default None
    block_size : float | None, optional
        block size to use for block-reduction of constraint points before fitting
        splines. If None, no block-reduction is applied, by default None
    block_reduction: str, optional
        type of block reduction to apply, if ``median``, weights will be ignored, of
        ``mean``, and weights are provided, they will be used in the block reduction.
        Defaults to ``median``.
    upper_confining_layer : xarray.DataArray | None, optional
        layer which the inverted topography should always be below, by default None
    lower_confining_layer : xarray.DataArray | None, optional
        layer which the inverted topography should always be above, by default None

    Returns
    -------
    xarray.Dataset
        a topography grid
    """
    if method == "flat":
        if registration == "g":
            pixel_register = False
        elif registration == "p":
            pixel_register = True
        else:
            msg = "registration must be 'g' or 'p'"
            raise ValueError(msg)

        if upwards is None:
            msg = "upwards must be provided if method is `flat`"
            raise ValueError(msg)

        # create grid of coordinates
        (x, y) = vd.grid_coordinates(  # pylint: disable=unbalanced-tuple-unpacking
            region=region,
            spacing=spacing,
            pixel_register=pixel_register,
        )
        # make flat topography of value = upwards
        grid = vd.make_xarray_grid(
            (x, y),
            np.ones_like(x) * upwards,
            data_names="upward",
            dims=("northing", "easting"),
        ).upward

    elif method == "splines":
        # get coordinates of the constraint points
        if constraints_df is None:
            msg = "constraints_df must be provided if method is `splines`"
            raise ValueError(msg)

        df = constraints_df.copy()

        # if only 1 point, return a flat topography
        if len(df) == 1:
            # create grid of coordinates
            (x, y) = vd.grid_coordinates(  # pylint: disable=unbalanced-tuple-unpacking
                region=region,
                spacing=spacing,
            )
            # make flat topography of value = upwards
            grid = vd.make_xarray_grid(
                (x, y),
                np.ones_like(x) * df.upward.to_numpy(),
                data_names="upward",
                dims=("northing", "easting"),
            ).upward

        if pd.Series(["inside", "buffer"]).isin(df.columns).all():
            df_to_interpolate = df[df.inside | df.buffer]
            df_outside_buffer = df[(df.inside == False) & (df.buffer == False)]  # noqa: E712 # pylint: disable=singleton-comparison

            coords = (df_to_interpolate.easting, df_to_interpolate.northing)
            data = df_to_interpolate.upward
            if weights_col is not None:
                weights = df_to_interpolate[weights_col]

            if block_size is not None:
                if block_reduction == "mean":
                    reduction = np.mean
                elif block_reduction == "median":
                    reduction = np.median
                    if weights is not None:
                        msg = "weights are ignored when block_reduction is 'median'"
                        logger.warning(msg)
                else:
                    msg = "block_reduction must be 'mean' or 'median'"
                    raise ValueError(msg)

                reducer = vd.BlockReduce(
                    reduction=reduction,
                    spacing=block_size,
                    region=region,
                    center_coordinates=True,
                )

                coords, data = reducer.filter(
                    coordinates=coords,
                    data=data,
                    weights=weights,
                )

            # run CV for fitting a spline to the data
            spline = optimal_spline_damping(
                coordinates=coords,
                data=data,
                weights=weights,
                dampings=dampings,
            )

            # grid the fitted spline at desired spacing and region
            inside_grid = spline.grid(
                region=region,
                spacing=spacing,
            ).scalars

            # merge interpolation of inner / buffer points with outside grid
            # outside_grid = df_outside_buffer.set_index(
            #   ["northing", "easting"]).to_xarray().upward
            # outside_grid = vd.make_xarray_grid(
            #     (df_outside_buffer.easting, df_outside_buffer.northing),
            #     df_outside_buffer.upward,
            #     data_names="upward",
            # )
            outside_grid = pygmt.xyz2grd(
                x=df_outside_buffer.easting,
                y=df_outside_buffer.northing,
                z=df_outside_buffer.upward,
                region=region,
                spacing=spacing,
            ).rename({"x": "easting", "y": "northing"})

            grid = inside_grid.where(
                outside_grid.isnull(),  # noqa: PD003
                outside_grid,
            )

        else:
            coords = (df.easting, df.northing)
            data = df.upward
            if weights_col is not None:
                weights = df[weights_col]

            if block_size is not None:
                if block_reduction == "mean":
                    reduction = np.mean
                elif block_reduction == "median":
                    reduction = np.median
                    if weights is not None:
                        msg = "weights are ignored when block_reduction is 'median'"
                        logger.warning(msg)
                else:
                    msg = "block_reduction must be 'mean' or 'median'"
                    raise ValueError(msg)

                reducer = vd.BlockReduce(
                    reduction=reduction,
                    spacing=block_size,
                    region=region,
                    center_coordinates=True,
                )

                coords, data = reducer.filter(
                    coordinates=coords,
                    data=data,
                    weights=weights,
                )

            # run CV for fitting a spline to the data
            spline = optimal_spline_damping(
                coordinates=coords,
                data=data,
                weights=weights,
                dampings=dampings,
            )
            # grid the fitted spline at desired spacing and region
            grid = spline.grid(
                region=region,
                spacing=spacing,
            ).scalars

        try:
            grid = grid.assign_attrs(damping=spline.damping_)
        except AttributeError:
            grid = grid.assign_attrs(damping=None)

    else:
        msg = "method must be 'flat' or 'splines'"
        raise ValueError(msg)

    # ensure grid doesn't cross supplied confining layers
    if upper_confining_layer is not None:
        da = fetch.fetch.resample_grid(
            upper_confining_layer,
            spacing=spacing,
            region=region,
            registration=registration,
        )
        grid = grid.where(grid <= da, da)
    if lower_confining_layer is not None:
        da = fetch.fetch.resample_grid(
            lower_confining_layer,
            spacing=spacing,
            region=region,
            registration=registration,
        )
        grid = grid.where(grid >= da, da)

    return grid.to_dataset(name="upward")


def grid_to_model(
    surface: xr.DataArray,
    reference: float | xr.DataArray,
    density: float | int | xr.DataArray,
    model_type: str,
) -> xr.Dataset:
    """
    create a Harmonica layer of prisms or tesseroids with assigned densities.

    Parameters
    ----------
    surface : xarray.DataArray
        data to use for model surface
    reference : float | xarray.DataArray
        data or constant to use for model reference, if value is below surface,
        prism/tesseroid will be inverted
    density : float | int | xarray.DataArray
        data or constant to use for model densities, should be in the form of a density
        contrast across a surface (i.e. between air and rock).
    model_type : str
        type of model to create, either 'prisms' or 'tesseroids'

    Returns
    -------
    xarray.Dataset
       a prism or tesseroid layer with assigned densities
    """

    # if density provided as a single number, use it for all prisms/tesseroids
    if isinstance(density, (float, int)):
        dens = density * np.ones_like(surface)
    # if density provided as a dataarray, map each density to the correct prisms/tesseroids
    elif isinstance(density, xr.DataArray):
        dens = density
    else:
        msg = "invalid density type, should be a number or DataArray"
        raise ValueError(msg)

    # create layer of prisms/tesseroids based off input dataarrays
    if model_type == "tesseroids":
        model = hm.tesseroid_layer(
            coordinates=(
                surface["longitude"].to_numpy(),
                surface["latitude"].to_numpy(),
            ),
            surface=surface,
            reference=reference,
            properties={
                "density": dens,
            },
        )
    elif model_type == "prisms":
        model = hm.prism_layer(
            coordinates=(
                surface["easting"].to_numpy(),
                surface["northing"].to_numpy(),
            ),
            surface=surface,
            reference=reference,
            properties={
                "density": dens,
            },
        )
    else:
        msg = "model_type must be 'prisms' or 'tesseroids'"
        raise ValueError(msg)

    model["thickness"] = model.top - model.bottom

    # add zref as an attribute
    return model.assign_attrs(zref=reference, model_type=model_type)


def grids_to_prisms(
    surface: xr.DataArray,  # noqa: ARG001
    reference: float | xr.DataArray,  # noqa: ARG001
    density: float | int | xr.DataArray,  # noqa: ARG001
    input_coord_names: tuple[str, str] = ("easting", "northing"),  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the function `grid_to_model` instead
    """
    # pylint: disable=W0613
    msg = "Function `grids_to_prisms` deprecated, use `grid_to_model` instead"
    raise DeprecationWarning(msg)


def best_spline_cv(
    coordinates: tuple[pd.Series | NDArray, pd.Series | NDArray],  # noqa: ARG001 # pylint: disable=unused-argument
    data: pd.Series | NDArray,  # noqa: ARG001 # pylint: disable=unused-argument
    weights: pd.Series | NDArray | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    **kwargs: typing.Any,  # noqa: ARG001 # pylint: disable=unused-argument
) -> None:
    """
    DEPRECATED: use the function `optimal_spline_damping` instead
    """
    msg = "Function `best_spline_cv` deprecated, use `optimal_spline_damping` instead"
    raise DeprecationWarning(msg)


def optimal_spline_damping(
    coordinates: tuple[pd.Series | NDArray, pd.Series | NDArray],
    data: pd.Series | NDArray,
    weights: pd.Series | NDArray | None = None,
    **kwargs: typing.Any,
) -> vd.Spline:
    """
    Find the best damping parameter for a verde.SplineCV() fit. All kwargs are passed to
    the verde.SplineCV class.

    Parameters
    ----------
    coordinates : tuple[pandas.Series  |  numpy.ndarray, pandas.Series  |  \
            numpy.ndarray]
        easting and northing coordinates of the data
    data : pandas.Series | numpy.ndarray
        data for fitting the spline to
    weights : pandas.Series | numpy.ndarray | None, optional
        if not None, then the weights assigned to each data point. Typically, this
        should be 1 over the data uncertainty squared, by default None

    Keyword Arguments
    -----------------
    dampings : float | None
        The positive damping regularization parameter. Controls how much smoothness is
        imposed on the estimated forces. If None, no regularization is used, by default
        None
    force_coords : bool
        The easting and northing coordinates of the point forces. If None (default),
        then will be set to the data coordinates.
    cv : None | cross-validation generator
        Any scikit-learn cross-validation generator. If not given, will use the
        default set by :func:`verde.cross_val_score`.
    delayed : bool
        If True, will use :func:`dask.delayed.delayed` to dispatch computations and
        allow :mod:`dask` to execute the grid search in parallel (see note
        above).
    scoring : None | str | Callable
        The scoring function (or name of a function) used for cross-validation.
        Must be known to scikit-learn. See the description of *scoring* in
        :func:`sklearn.model_selection.cross_val_score` for details. If None,
        will fall back to the :meth:`verde.Spline.score` method.

    Returns
    -------
    verde.Spline
        the spline which best fits the data
    """
    kwargs = copy.deepcopy(kwargs)

    dampings = kwargs.pop("dampings", None)

    # if single damping value provided, convert to list
    if isinstance(dampings, typing.Iterable):
        pass
    else:
        dampings = [dampings]

    n_splits = 5
    while n_splits > 0:
        try:
            spline = vd.SplineCV(
                dampings=dampings,
                cv=sklearn.model_selection.KFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=0,
                ),
                **kwargs,
            )
            spline.fit(
                coordinates,
                data,
                weights=weights,
            )
            break
        except ValueError as e:
            logger.error(e)
            msg = "decreasing number of splits by 1 until ValueError is resolved"
            logger.warning(msg)
        if n_splits == 1:
            msg = "ValueError not resolved, fitting spline with no damping"
            logger.warning(msg)
            spline = vd.Spline(
                damping=None,
                **kwargs,
            )
            spline.fit(
                coordinates,
                data,
                weights=weights,
            )
        n_splits -= 1

    if len(dampings) > 1:
        try:
            logger.info("Best SplineCV score: %s", spline.scores_.max())
        except AttributeError:
            logger.info("Best SplineCV score: %s", max(dask.compute(spline.scores_)[0]))

        logger.info("Best damping: %s", spline.damping_)

    dampings_without_none = [i for i in dampings if i is not None]

    try:
        if spline.damping_ is None:
            pass
        elif len(dampings) > 2 and spline.damping_ in [
            np.min(dampings_without_none),
            np.max(dampings_without_none),
        ]:
            logger.warning(
                "Best damping value (%s) is at the limit of provided values (%s, %s) "
                "and thus is likely not a global minimum, expand the range of values "
                "test to ensure the best parameter value value is found.",
                spline.damping_,
                np.nanmin(dampings_without_none),
                np.nanmax(dampings_without_none),
            )
    except AttributeError:
        pass

    return spline


def best_equivalent_source_damping(
    coordinates: tuple[pd.Series | NDArray, pd.Series | NDArray, pd.Series | NDArray],
    data: pd.Series | NDArray,
    delayed: bool = False,
    weights: pd.Series | NDArray | None = None,
    **kwargs: typing.Any,
) -> hm.EquivalentSources:
    """
    Find the best damping parameter for a harmonica.EquivalentSource() fit. All kwargs
    are passed to the harmonica.EquivalentSource class.

    Parameters
    ----------
    coordinates : tuple[pandas.Series | numpy.ndarray, pandas.Series | numpy.ndarray, \
            pandas.Series | numpy.ndarray]
        tuple of easting, northing, and upward coordinates of the gravity data
    data : pandas.Series | numpy.ndarray
        the gravity data
    delayed : bool, optional
        compute the scores in parallel if True, by default False
    weights : numpy.ndarray | None, optional
        optional weight values for each gravity data point, by default None

    Keyword Arguments
    -----------------
    damping : float | None
        The positive damping regularization parameter. Controls how much
        smoothness is imposed on the estimated coefficients.
        If None, no regularization is used.
    points : list[numpy.ndarray] | None
        List containing the coordinates of the equivalent point sources.
        Coordinates are assumed to be in the following order:
        (``easting``, ``northing``, ``upward``).
        If None, will place one point source below each observation point at
        a fixed relative depth below the observation point.
        Defaults to None.
    depth : float or str
        Parameter used to control the depth at which the point sources will be
        located.
        If a value is provided, each source is located beneath each data point
        (or block-averaged location) at a depth equal to its elevation minus
        the ``depth`` value.
        If set to ``"default"``, the depth of the sources will be estimated as
        4.5 times the mean distance between first neighboring sources.
        This parameter is ignored if *points* is specified.
        Defaults to ``"default"``.
    block_size: float | tuple[float, float] | None
        Size of the blocks used on block-averaged equivalent sources.
        If a single value is passed, the blocks will have a square shape.
        Alternatively, the dimensions of the blocks in the South-North and
        West-East directions can be specified by passing a tuple.
        If None, no block-averaging is applied.
        This parameter is ignored if *points* are specified.
        Default to None.
    parallel : bool
        If True any predictions and Jacobian building is carried out in
        parallel through Numba's ``jit.prange``, reducing the computation time.
        If False, these tasks will be run on a single CPU. Default to True.
    dtype : str
        The desired data-type for the predictions and the Jacobian matrix.
        Default to ``"float64"``.

    Returns
    -------
    harmonica.EquivalentSources
        the best fitted equivalent sources
    """
    kwargs = copy.deepcopy(kwargs)
    dampings = kwargs.pop("dampings", None)
    kwargs.pop("damping", None)
    # if single damping value provided, convert to list
    if isinstance(dampings, typing.Iterable):
        pass
    else:
        dampings = [dampings]
    # pylint: disable=duplicate-code
    if np.isnan(coordinates).any():
        msg = "coordinates contain NaN"
        raise ValueError(msg)
    if np.isnan(data).any():
        msg = "data contains is NaN"
        raise ValueError(msg)

    scores = []
    for d in dampings:
        eqs = hm.EquivalentSources(
            damping=d,
            **kwargs,
        )

        score = np.mean(
            vd.cross_val_score(
                eqs, coordinates, data, delayed=delayed, weights=weights, scoring="r2"
            )
        )
        # pylint: enable=duplicate-code
        scores.append(score)

    if delayed:
        scores = dask.compute(scores)[0]
    else:
        pass

    best = np.argmax(scores)
    logger.info("Best EqSources score: %s", scores[best])
    logger.info("Best damping: %s", dampings[best])

    dampings_without_none = [i for i in dampings if i is not None]

    if dampings[best] is None:
        pass
    elif len(dampings) > 2 and dampings[best] in [
        np.min(dampings_without_none),
        np.max(dampings_without_none),
    ]:
        logger.warning(
            "Best damping value (%s) is at the limit of provided values (%s, %s) and "
            "thus is likely not a global minimum, expand the range of values test to "
            "ensure the best parameter value value is found.",
            dampings[best],
            np.nanmin(dampings_without_none),
            np.nanmax(dampings_without_none),
        )

    return hm.EquivalentSources(damping=dampings[best]).fit(
        coordinates, data, weights=weights
    )


def gravity_decay_buffer(
    buffer_perc: float,
    spacing: float,
    inner_region: tuple[float, float, float, float],
    top: float,
    zref: float,
    obs_height: float,
    density: float,
    model_type: str = "prisms",
    amplitude: float | None = None,
    wavelength: float | None = None,
    checkerboard: bool = False,
    as_density_contrast: bool = False,
    plot: bool = True,
    plot_profile: bool = True,
    progressbar: bool = False,
) -> tuple[float, float, int, xr.Dataset]:
    """
    For a given buffer zone width (as percentage of x or y range) and domain parameters,
    calculate the max percent decay of the gravity anomaly within the region of
    interest.

    Parameters
    ----------
    buffer_perc : float
        percentage of the widest dimension of inner_region to use as buffer zone
    spacing : float
        spacing of the prism layer and gravity observation points
    inner_region : tuple[float, float, float, float]
        region boundaries for the region of interest
    top : float
        height for the top of the prisms
    zref : float
        reference level for the prisms
    obs_height : float
        gravity observation height
    density : float
        density value for the prisms
    model_type : str, optional
        type of model to create, either 'prisms' or 'tesseroids', by default 'prisms'
    amplitude : float | None, optional
        if using `checkerboard`, this is the amplitude of each undulation, by default
        None
    wavelength : float | None, optional
        if using `checkerboard`, this is the wavelength of each undulation, by default
        None
    checkerboard : bool, optional
        use an undulating checkerboard for the topography instead of a flat surface, by
        default False
    as_density_contrast : bool, optional
        discretize the topography as a density contrast, resulting in no edge effects,
        by default False
    plot : bool, optional
        plot the results, by default True
    plot_profile : bool, optional
        plot a profile across the prism layer, by default True
    progressbar : bool, optional
        show a progressbar for the forward gravity calculation, by default False

    Returns
    -------
    max_decay : float
        the maximum percentage decay of the gravity anomaly within the region of
        interest
    buffer_width : float
        width of the buffer zone
    buffer_cells : int
        number of cells in the buffer zone
    grav_ds : xarray.Dataset
        dataset of the forward gravity calculations
    """

    if (checkerboard is False) & (top == zref):
        msg = "top and zref must be different if checkerboard is False"
        raise ValueError(msg)

    # get x and y range of interest region
    x_diff = np.abs(inner_region[0] - inner_region[1])
    y_diff = np.abs(inner_region[2] - inner_region[3])

    # pick the bigger range
    max_diff = max(x_diff, y_diff)

    # calc buffer as percentage of width
    buffer_width = max_diff * (buffer_perc / 100)

    # round to nearest multiple of spacing
    def round_to_input(num: float, multiple: float) -> float:
        return round(num / multiple) * multiple

    # round buffer width to nearest spacing interval
    buffer_width = round_to_input(buffer_width, spacing)

    # define buffer region
    buffer_region = vd.pad_region(inner_region, buffer_width)

    # calculate buffer width in terms of number of cells
    buffer_cells = buffer_width / spacing

    # create topography
    if checkerboard:
        synth = vd.synthetic.CheckerBoard(
            amplitude=amplitude,
            region=buffer_region,
            w_east=wavelength,
            w_north=wavelength,
        )

        surface = synth.grid(
            spacing=spacing, data_names="upward", dims=("northing", "easting")
        ).upward

        surface += top

    else:
        surface = create_topography(
            method="flat",
            upwards=top,
            region=buffer_region,
            spacing=spacing,
        ).upward

    # create prism layer
    if as_density_contrast:
        # create prisms around mean value to compare to to calculate decay
        zref = surface.to_numpy().mean()

        # positive densities above, negative below
        dens = surface.copy()
        dens.to_numpy()[:] = density
        dens = dens.where(surface >= zref, -density)

        # create prism layer with a mean zref
        model = grid_to_model(
            surface,
            zref,
            density=dens,
            model_type=model_type,
        )
    else:
        model = grid_to_model(
            surface,
            zref,
            density=density,
            model_type=model_type,
        )

    # create prisms around mean value to compare to to calculate decay
    zref = surface.to_numpy().mean()

    # positive densities above, negative below
    dens = surface.copy()
    dens.to_numpy()[:] = density
    dens = dens.where(surface >= zref, -density)

    # create prism layer with a mean zref
    model_mean_zref = grid_to_model(
        surface,
        zref,
        density=dens,
        model_type=model_type,
    )

    # create set of observation points
    data = vd.grid_coordinates(
        inner_region,
        spacing=spacing,
        extra_coords=obs_height,
    )

    forward_df = pd.DataFrame(
        {
            "easting": data[0].ravel(),
            "northing": data[1].ravel(),
            "upward": data[2].ravel(),
        }
    )
    # calculate forward gravity of layer
    if model_type == "prisms":
        forward_df["forward"] = model.prism_layer.gravity(
            coordinates=(
                forward_df.easting,
                forward_df.northing,
                forward_df.upward,
            ),
            field="g_z",
            progressbar=progressbar,
        )
    elif model_type == "tesseroids":
        forward_df["forward"] = model.tesseroid_layer.gravity(
            coordinates=(
                forward_df.easting,
                forward_df.northing,
                forward_df.upward,
            ),
            field="g_z",
            progressbar=progressbar,
        )

    # if checkerboard:
    # calculate forward gravity of layer
    if model_type == "prisms":
        forward_df["forward_no_edge_effects"] = model_mean_zref.prism_layer.gravity(
            coordinates=(
                forward_df.easting,
                forward_df.northing,
                forward_df.upward,
            ),
            field="g_z",
            progressbar=progressbar,
        )
    elif model_type == "tesseroids":
        forward_df["forward_no_edge_effects"] = model_mean_zref.tesseroid_layer.gravity(
            coordinates=(
                forward_df.easting,
                forward_df.northing,
                forward_df.upward,
            ),
            field="g_z",
            progressbar=progressbar,
        )

    grav_ds = forward_df.set_index(["northing", "easting"]).to_xarray()

    # shift forward gravity with and without edge effects so max value are equal
    shift = grav_ds.forward_no_edge_effects.max() - grav_ds.forward.max()

    grav_ds["forward_no_edge_effects"] -= shift

    dif = grav_ds.forward - grav_ds.forward_no_edge_effects

    max_grav = grav_ds.forward.to_numpy().max()
    max_decay = 100 * (max_grav - (max_grav + dif.to_numpy().min())) / max_grav

    if plot:
        try:
            plotting.plot_edge_effects(
                grav_ds=grav_ds,
                layer=model,
                inner_region=inner_region,
                plot_profile=plot_profile,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("plotting failed with error: %s", e)

    return max_decay, buffer_width, int(buffer_cells), grav_ds
