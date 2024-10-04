from __future__ import annotations  # pylint: disable=too-many-lines

import copy
import os
import typing
import warnings
from contextlib import contextmanager

import dask
import deprecation
import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import sklearn
import verde as vd
import xarray as xr
import xrft
from numpy.typing import NDArray
from pykdtree.kdtree import KDTree  # pylint: disable=no-name-in-module

import invert4geom
from invert4geom import cross_validation, log


@contextmanager
def _log_level(level):  # type: ignore[no-untyped-def]
    "Run body with logger at a different level"
    saved_logger_level = log.level
    log.setLevel(level)
    try:
        yield saved_logger_level
    finally:
        log.setLevel(saved_logger_level)


@contextmanager
def environ(**env):  # type: ignore[no-untyped-def] # pylint: disable=missing-function-docstring
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

    def __init__(self, logger):  # type: ignore[no-untyped-def]
        self.msgs = set()
        self.logger = logger

    def filter(self, record):  # type: ignore[no-untyped-def] # pylint: disable=missing-function-docstring
        msg = str(record.msg)
        is_duplicate = msg in self.msgs
        if not is_duplicate:
            self.msgs.add(msg)
        return not is_duplicate

    def __enter__(self):  # type: ignore[no-untyped-def]
        self.logger.addFilter(self)

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[no-untyped-def]
        self.logger.removeFilter(self)


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
        log.warning(msg)


def _check_gravity_inside_topography_region(
    grav_df: pd.DataFrame,
    topography: xr.DataArray,
) -> None:
    """check that all gravity data is inside the region of the topography grid"""
    topo_region = vd.get_region((topography.easting.values, topography.northing.values))
    inside = vd.inside((grav_df.easting, grav_df.northing), region=topo_region)
    if not inside.all():
        msg = (
            "Some gravity data are outside the region of the topography grid. "
            "This may result in unexpected behavior."
        )
        raise ValueError(msg)


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


def nearest_grid_fill(
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
        df_dropped = df[df[grid.name].notnull()]
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


def filter_grid(
    grid: xr.DataArray,
    filter_width: float | None = None,
    filt_type: str = "lowpass",
) -> xr.DataArray:
    """
    Apply a spatial filter to a grid.

    Parameters
    ----------
    grid : xarray.DataArray
        grid to filter the values of
    filter_width : float, optional
        width of the filter in meters, by default None
    filt_type : str, optional
        type of filter to use, by default "lowpass"

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
    if grid.isnull().any():
        filled = nearest_grid_fill(grid, method="verde")
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
        original_dims[1]: grid[original_dims[1]].size // 3,
        original_dims[0]: grid[original_dims[0]].size // 3,
    }

    # apply padding
    padded = xrft.pad(filled, pad_width)

    if filt_type == "lowpass":
        filt = hm.gaussian_lowpass(padded, wavelength=filter_width).rename("filt")
    elif filt_type == "highpass":
        filt = hm.gaussian_highpass(padded, wavelength=filter_width).rename("filt")
    elif filt_type == "up_deriv":
        filt = hm.derivative_upward(padded).rename("filt")
    elif filt_type == "easting_deriv":
        filt = hm.derivative_easting(padded).rename("filt")
    elif filt_type == "northing_deriv":
        filt = hm.derivative_northing(padded).rename("filt")
    else:
        msg = "filt_type must be 'lowpass' or 'highpass'"
        raise ValueError(msg)

    unpadded = xrft.unpad(filt, pad_width)

    # reset coordinate values to original (avoid rounding errors)
    unpadded = unpadded.assign_coords(
        {
            original_dims[0]: grid[original_dims[0]].to_numpy(),
            original_dims[1]: grid[original_dims[1]].to_numpy(),
        }
    )

    if grid.isnull().any():
        result: xr.DataArray = xr.where(grid.notnull(), unpadded, grid)
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
    coord_names: tuple[str, str] | None = None,
) -> typing.Any:
    """
    for all gridcells calculate to the distance to the nearest target.

    Parameters
    ----------
    targets : pandas.DataFrame
        contains the coordinates of the targets
    data : pandas.DataFrame | xarray.DataArray | xarray.Dataset
        the grid data, in either gridded or tabular form
    coord_names : tuple[str, str] | None, optional
        the names of the coordinates for both the targets and the data, by default None

    Returns
    -------
    typing.Any
        the distance to the nearest target for each gridcell, in the same format as the
        input for `data`.
    """

    if coord_names is None:
        coord_names = ("easting", "northing")

    df_targets = targets[[coord_names[0], coord_names[1]]].copy()

    df_data: pd.DataFrame | xr.DataArray | xr.Dataset
    if isinstance(data, pd.DataFrame):
        df_data = data[list(coord_names)].copy()
    elif isinstance(data, xr.DataArray):
        df_grid = vd.grid_to_table(data).dropna()
        df_data = df_grid[[coord_names[0], coord_names[1]]].copy()  # pylint: disable=unsubscriptable-object
    elif isinstance(data, xr.Dataset):
        try:
            df_grid = vd.grid_to_table(data[next(iter(data.variables))]).dropna()
            # df_grid = vd.grid_to_table(data[list(data.variables)[0]]).dropna()
        except IndexError:
            df_grid = vd.grid_to_table(data).dropna()
        df_data = df_grid[[coord_names[0], coord_names[1]]].copy()  # pylint: disable=unsubscriptable-object

    min_dist, _ = KDTree(df_targets.values).query(df_data.values, 1)

    df_data["min_dist"] = min_dist

    if isinstance(data, pd.DataFrame):
        return df_data
    return df_data.set_index([coord_names[0], coord_names[1]][::-1]).to_xarray()


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
    # min_val = da.values.min()
    # max_val = da.values.max()

    da = da.copy()

    min_val = da.quantile(0)
    max_val = da.quantile(1)

    da2: xr.DataArray = (high - low) * (
        ((da - min_val) / (max_val - min_val)).clip(0, 1)
    ) + low

    return da2.drop("quantile")


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
    ).min_dist

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
        new_min_dist = xr.where(min_dist.isnull(), np.nan, new_min_dist)

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

    x, y = kwargs.get("coord_names", ("easting", "northing"))
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


def extract_prism_data(
    prism_layer: xr.Dataset,
) -> tuple[
    pd.DataFrame,
    xr.Dataset,
    float,
    xr.DataArray,
]:
    """
    extract the grid spacing from the starting prism layer and adds variables 'topo' and
    'starting_topo', which are the both the starting topography elevation.
    'starting_topo' remains unchanged, while 'topo' is updated at each iteration.

    Parameters
    ----------
    prism_layer : xarray.Dataset
       starting model prism layer

    Returns
    -------
    prisms_df : pandas.DataFrame
        dataframe of prism layer
    prisms_ds : xarray.Dataset
        prism layer with added variables 'topo' and 'starting_topo'
    spacing : float
        spacing of prisms
    topo_grid : xarray.DataArray
        grid of starting topography
    """

    prisms_ds = copy.deepcopy(prism_layer.load())

    # add starting topo to dataset
    topo_grid = xr.where(prisms_ds.density > 0, prisms_ds.top, prisms_ds.bottom)
    prisms_ds["topo"] = topo_grid
    prisms_ds["starting_topo"] = topo_grid

    # turn dataset into dataframe
    prisms_df = prisms_ds.to_dataframe().reset_index().dropna().astype(float)

    spacing = get_spacing(prisms_df)

    return prisms_df, prisms_ds, spacing, topo_grid


def get_spacing(prisms_df: pd.DataFrame) -> float:
    """
    Extract spacing of harmonica prism layer using a dataframe representation.

    Parameters
    ----------
    prisms_df : pandas.DataFrame
        dataframe of harmonica prism layer

    Returns
    -------
    float
        spacing of prisms
    """
    return float(abs(prisms_df.northing.unique()[1] - prisms_df.northing.unique()[0]))


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
            coord_names=["easting", "northing"],
        )
        assert len(df.upper_bounds) != 0
    if lower_confining_layer is not None:
        df = sample_grids(
            df=df,
            grid=lower_confining_layer,
            sampled_name="lower_bounds",
            coord_names=["easting", "northing"],
        )
        assert len(df.lower_bounds) != 0
    return df


def enforce_confining_surface(
    prisms_df: pd.DataFrame,
    iteration_number: int,
) -> pd.DataFrame:
    """
    alter the surface correction values to ensure when added to the current iteration's
    topography it doesn't intersect optional confining layers.

    Parameters
    ----------
    prisms_df : pandas.DataFrame
        prism layer dataframe with optional 'upper_bounds' or 'lower_bounds' columns,
        and current iteration's topography.
    iteration_number : int
        number of the current iteration, starting at 1 not 0

    Returns
    -------
    pandas.DataFrame
        a dataframe with added column 'iter_{iteration_number}_correction
    """

    df = prisms_df.copy()

    if "upper_bounds" in df:
        # get max upward change allowed for each prism
        # positive values indicate max allowed upward change
        # negative values indicate topography is already too far above upper bound
        df["max_change_above"] = df.upper_bounds - df.topo
        number_enforced = 0
        for i, j in enumerate(df[f"iter_{iteration_number}_correction"]):
            if j > df.max_change_above[i]:
                number_enforced += 1
                df.loc[i, f"iter_{iteration_number}_correction"] = df.max_change_above[
                    i
                ]
        log.info("enforced upper confining surface at %s prisms", number_enforced)
    if "lower_bounds" in df:
        # get max downward change allowed for each prism
        # negative values indicate max allowed downward change
        # positive values indicate topography is already too far below lower bound
        df["max_change_below"] = df.lower_bounds - df.topo
        number_enforced = 0
        for i, j in enumerate(df[f"iter_{iteration_number}_correction"]):
            if j < df.max_change_below[i]:
                number_enforced += 1
                df.loc[i, f"iter_{iteration_number}_correction"] = df.max_change_below[
                    i
                ]

        log.info("enforced lower confining surface at %s prisms", number_enforced)

    # check that when constrained correction is added to topo it doesn't intersect
    # either bounding layer
    updated_topo: pd.Series[float] = df[f"iter_{iteration_number}_correction"] + df.topo
    if "upper_bounds" in df and np.any((df.upper_bounds - updated_topo) < -0.001):
        msg = (
            "Constraining didn't work and updated topography intersects upper "
            "constraining surface"
        )
        raise ValueError(msg)
    if "lower_bounds" in df and np.any((updated_topo - df.lower_bounds) < -0.001):
        msg = (
            "Constraining didn't work and updated topography intersects lower "
            "constraining surface"
        )
        raise ValueError(msg)
    return df


def apply_surface_correction(
    prisms_df: pd.DataFrame,
    iteration_number: int,
) -> tuple[pd.DataFrame, xr.DataArray]:
    """
    update the prisms dataframe and dataset with the surface correction. Ensure that
    the updated surface doesn't intersect the optional confining surfaces.

    Parameters
    ----------
    prisms_df : pandas.DataFrame
        dataframe of prism properties
    iteration_number : int
        the iteration number, starting at 1 not 0

    Returns
    -------
    tuple[pandas.DataFrame, xarray.DataArray]
        updated prisms dataframe and correction grid
    """

    df = prisms_df.copy()

    # for negative densities, negate the correction
    df.loc[df.density < 0, f"iter_{iteration_number}_correction"] *= -1

    # optionally constrain the surface correction with bounding surfaces
    df = enforce_confining_surface(df, iteration_number)

    # grid the corrections
    correction_grid = (
        df.rename(columns={f"iter_{iteration_number}_correction": "z"})
        .set_index(["northing", "easting"])
        .to_xarray()
        .z
    )

    return df, correction_grid


def update_prisms_ds(
    prisms_ds: xr.Dataset,
    correction_grid: xr.DataArray,
) -> xr.Dataset:
    """
    apply the corrections grid and update the prism tops, bottoms, topo, and
    densities.

    Parameters
    ----------
    prisms_ds : xarray.Dataset
        harmonica prism layer
    correction_grid : xarray.DataArray
        grid of corrections to apply to the prism layer

    Returns
    -------
    xarray.Dataset
        updated prism layer with new tops, bottoms, topo, and densities
    """

    ds = prisms_ds.copy()

    # extract the reference value used to create the prisms
    zref = ds.attrs.get("zref")

    # extract the element-wise absolute value of the density contrast
    density_contrast = np.fabs(ds.density)

    # create topo from top and bottom
    topo_grid = xr.where(ds.density > 0, ds.top, ds.bottom)

    # apply correction to topo
    topo_grid += correction_grid

    # update the prism layer
    ds.prism_layer.update_top_bottom(surface=topo_grid, reference=zref)

    # update the density
    ds["density"] = xr.where(ds.top > zref, density_contrast, -density_contrast)

    # update the topo
    ds["topo"] = topo_grid

    return ds


def add_updated_prism_properties(
    prisms_df: pd.DataFrame,
    prisms_ds: xr.Dataset,
    iteration_number: int,
) -> pd.DataFrame:
    """
    update the prisms dataframe the the new prism tops, bottoms, topo, and densities
    the iteration number, starting at 1 not 0

    Parameters
    ----------
    prisms_df : pandas.DataFrame
        dataframe of prism properties
    prisms_ds : xarray.Dataset
        dataset of prism properties
    iteration_number : int
        the iteration number, starting at 1 not 0

    Returns
    -------
    pandas.DataFrame
        updated prism dataframe with new tops, bottoms, topo, and densities
    """

    df = prisms_df.copy()
    ds = prisms_ds.copy()

    # turn back into dataframe
    prisms_iter = ds.to_dataframe().reset_index().dropna().astype(float)

    # add new cols to dict
    dict_of_cols = {
        f"iter_{iteration_number}_top": prisms_iter.top,
        f"iter_{iteration_number}_bottom": prisms_iter.bottom,
        f"iter_{iteration_number}_density": prisms_iter.density,
        f"iter_{iteration_number}_layer": prisms_iter.topo,
    }

    df = pd.concat([df, pd.DataFrame(dict_of_cols)], axis=1)
    df["topo"] = prisms_iter.topo

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
) -> xr.DataArray:
    """
    Create a grid of topography data from either the interpolation of point data or
    creating a grid of constant value.

    Parameters
    ----------
    method : str
        method to use, either 'flat' or 'splines'
    region : tuple[float, float, float, float]
        region of the grid
    spacing : float
        spacing of the grid
    dampings : list[float] | None, optional
        damping values to use in spline cross validation for method "spline", by default
        None
    registration : str, optional
        choose between gridline "g" or pixel "p" registration, by default "g"
    upwards : float | None, optional
        constant value to use for method "flat", by default None
    constraints_df : pandas.DataFrame | None, optional
        dataframe with column 'upwards' to use for method "splines", by default None
    weights : pandas.Series | numpy.ndarray | None, optional
        weight to use for fitting the spline. Typically, this should be 1 over the data
        uncertainty squared, by default None
    weights_col : str | None, optional
        instead of passing the weights, pass the name of the column containing the
        weights, by default None

    Returns
    -------
    xarray.DataArray
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
        return vd.make_xarray_grid(
            (x, y),
            np.ones_like(x) * upwards,
            data_names="upward",
            dims=("northing", "easting"),
        ).upward

    if method == "splines":
        # get coordinates of the constraint points
        if constraints_df is None:
            msg = "constraints_df must be provided if method is `splines`"
            raise ValueError(msg)
        coords = (constraints_df.easting, constraints_df.northing)

        if len(constraints_df) == 1:
            # create grid of coordinates
            (x, y) = vd.grid_coordinates(  # pylint: disable=unbalanced-tuple-unpacking
                region=region,
                spacing=spacing,
            )
            # make flat topography of value = upwards
            return vd.make_xarray_grid(
                (x, y),
                np.ones_like(x) * constraints_df.upward.values,
                data_names="upward",
                dims=("northing", "easting"),
            ).upward

        if weights_col is not None:
            weights = constraints_df[weights_col]

        # run CV for fitting a spline to the data
        spline = best_spline_cv(
            coordinates=coords,
            data=constraints_df.upward,
            weights=weights,
            dampings=dampings,
        )
        # grid the fitted spline at desired spacing and region
        grid = spline.grid(
            region=region,
            spacing=spacing,
        ).scalars

        try:
            return grid.assign_attrs(damping=spline.damping_)
        except AttributeError:
            return grid.assign_attrs(damping=None)

    msg = "method must be 'flat' or 'splines'"
    raise ValueError(msg)


def grids_to_prisms(
    surface: xr.DataArray,
    reference: float | xr.DataArray,
    density: float | int | xr.DataArray,
    input_coord_names: tuple[str, str] = ("easting", "northing"),
) -> xr.Dataset:
    """
    create a Harmonica layer of prisms with assigned densities.

    Parameters
    ----------
    surface : xarray.DataArray
        data to use for prism surface
    reference : float | xarray.DataArray
        data or constant to use for prism reference, if value is below surface, prism
        will be inverted
    density : float | int | xarray.DataArray
        data or constant to use for prism densities, should be in the form of a density
        contrast across a surface (i.e. between air and rock).
    input_coord_names : tuple[str, str], optional
        names of the coordinates in the input dataarray, by default
        ["easting", "northing"]
    Returns
    -------
    xarray.Dataset
       a prisms layer with assigned densities
    """

    # if density provided as a single number, use it for all prisms
    if isinstance(density, (float, int)):
        dens = density * np.ones_like(surface)
    # if density provided as a dataarray, map each density to the correct prisms
    elif isinstance(density, xr.DataArray):
        dens = density
    else:
        msg = "invalid density type, should be a number or DataArray"
        raise ValueError(msg)

    # create layer of prisms based off input dataarrays
    prisms = hm.prism_layer(
        coordinates=(
            surface[input_coord_names[0]].values,
            surface[input_coord_names[1]].values,
        ),
        surface=surface,
        reference=reference,
        properties={
            "density": dens,
        },
    )

    prisms["thickness"] = prisms.top - prisms.bottom

    # add zref as an attribute
    return prisms.assign_attrs(zref=reference)


def best_spline_cv(
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
            log.error(e)
            msg = "decreasing number of splits by 1 until ValueError is resolved"
            log.warning(msg)
        if n_splits == 1:
            msg = "ValueError not resolved, fitting spline with no damping"
            log.warning(msg)
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
            log.info("Best SplineCV score: %s", spline.scores_.max())
        except AttributeError:
            log.info("Best SplineCV score: %s", max(dask.compute(spline.scores_)[0]))

        log.info("Best damping: %s", spline.damping_)

    dampings_without_none = [i for i in dampings if i is not None]

    try:
        if spline.damping_ is None:
            pass
        elif len(dampings) > 2 and spline.damping_ in [
            np.min(dampings_without_none),
            np.max(dampings_without_none),
        ]:
            log.warning(
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
                eqs,
                coordinates,
                data,
                delayed=delayed,
                weights=weights,
            )
        )
        # pylint: enable=duplicate-code
        scores.append(score)

    if delayed:
        scores = dask.compute(scores)[0]
    else:
        pass

    best = np.argmax(scores)
    log.info("Best EqSources score: %s", scores[best])
    log.info("Best damping: %s", dampings[best])

    dampings_without_none = [i for i in dampings if i is not None]

    if dampings[best] is None:
        pass
    elif len(dampings) > 2 and dampings[best] in [
        np.min(dampings_without_none),
        np.max(dampings_without_none),
    ]:
        log.warning(
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


@deprecation.deprecated(  # type: ignore[misc]
    deprecated_in="0.8.0",
    removed_in="0.14.0",
    current_version=invert4geom.__version__,
    details="function eq_sources_score has been moved to the cross_validation model.",
)
def eq_sources_score(kwargs: typing.Any) -> float:
    """
    deprecated function, use cross_validation.eq_sources_score instead.
    """
    return cross_validation.eq_sources_score(**kwargs)
