import copy  # pylint: disable=too-many-lines
import os
import typing
import warnings
from contextlib import contextmanager

import bordado as bd
import dask
import harmonica as hm
import numpy as np
import pandas as pd
import pyproj
import rioxarray  # noqa: F401 # pylint: disable=unused-import # registers the `.rio` xarray accessor
import sklearn
import verde as vd
import xarray as xr
import xrft
from numpy.typing import NDArray
from pykdtree.kdtree import KDTree  # pylint: disable=no-name-in-module
from scipy import ndimage

from invert4geom import logger, maps, plotting


def _import_pygmt() -> typing.Any:
    """
    Import PyGMT, which is only needed for the ``pygmt`` gridding methods, and raise an
    informative error if it isn't installed.
    """
    try:
        import pygmt  # noqa: PLC0415 # pylint: disable=import-outside-toplevel
    except ImportError as e:
        msg = (
            "PyGMT is needed for the 'pygmt' gridding method but isn't installed. "
            "Either install it (`conda install pygmt`) or use one of the other "
            "methods, such as 'splines'."
        )
        raise ImportError(msg) from e
    return pygmt


@contextmanager
def _log_level(level, log_instance=logger):  # type: ignore[no-untyped-def, has-type, unused-ignore]
    "Run body with logger at a different level"
    saved_logger_level = log_instance.level
    log_instance.setLevel(level)
    try:
        yield saved_logger_level
    finally:
        log_instance.setLevel(saved_logger_level)


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


def buffer_edges(
    grid: xr.DataArray,
    cells: int = 3,
    grow: bool = False,
    fill_value: float | None = None,
) -> xr.DataArray:
    """
    Shrink or grow the non-NaN region of a 2D grid by `cells` cells.

    Parameters
    ----------
    grid : xarray.DataArray
        2D grid of values and NaNs.
    cells : int
        Number of cell rings to remove (or add if `grow`).
    grow : bool
        If True, dilate the valid region instead of eroding it.
    fill_value : float, optional
        Value given to newly added cells. Defaults to the mean of the
        existing non-NaN cells. Ignored if `grow` is False.

    Returns
    -------
    xarray.DataArray
        Grid with the same coords/attrs and an adjusted valid region.
    """
    if cells < 1:
        return grid.copy()

    mask = ~np.isnan(grid.to_numpy())
    structure = np.ones((3, 3))  # 8-connectivity, includes diagonal corners

    if grow:
        new_mask = ndimage.binary_dilation(
            mask,
            structure=structure,
            iterations=cells,
            border_value=0,
        )
        if fill_value is None:
            fill_value = float(grid.mean(skipna=True))
        return grid.where(mask, fill_value).where(new_mask)

    new_mask = ndimage.binary_erosion(
        mask,
        structure=structure,
        iterations=cells,
        border_value=1,  # don't erode inward from the array edges
    )
    return grid.where(new_mask)


def get_epsg(coast: bool) -> tuple[str, bool]:
    """
    Get the EPSG code for plotting coastlines. Read from the environment variable
    'INVERT4GEOM_EPSG', falling back to 'POLARTOOLKIT_EPSG'. If neither is set, default
    to EPSG of 3857 and skip plotting coastlines.

    Parameters
    ----------
    coast : bool
        whether to plot coastlines, if True, will attempt to get EPSG code for plotting
        coastlines, if False, will skip plotting coastlines and return EPSG of 3857

    Returns
    -------
    tuple[str, bool]
        epsg code and whether to plot coastlines
    """
    epsg = os.environ.get("INVERT4GEOM_EPSG") or os.environ.get("POLARTOOLKIT_EPSG")

    if epsg is None:
        if coast:
            msg = (
                "couldn't find environment variable 'INVERT4GEOM_EPSG' with the EPSG "
                "code needed for plotting elements with geographic coordinates, such as"
                "coastlines. You can set the environment variable "
                "temporarily with `import os; os.environ['INVERT4GEOM_EPSG'] = '3031' "
                "where 3031 is the EPSG projection of your region. For now defaulting to "
                "EPSG of 3857 and skipping plotting coastlines."
            )
            warnings.warn(
                msg,
                UserWarning,
                stacklevel=2,
            )
            logger.warning(msg)
            coast = False
        epsg = "3857"
    elif not epsg.isdigit():
        msg = f"the EPSG environment variable must be a string of digits, not {epsg}"
        raise ValueError(msg)

    return epsg, coast


def _check_constraints_inside_gravity_region(
    constraints_df: pd.DataFrame,
    grav_ds: xr.Dataset,
) -> None:
    """check that all constraints are inside the region of the gravity data"""
    coord_names = grav_ds.coord_names

    grav_df = grav_ds.inv.df

    grav_region = bd.get_region((grav_df[coord_names[0]], grav_df[coord_names[1]]))
    inside = bd.inside(
        (constraints_df[coord_names[0]], constraints_df[coord_names[1]]),
        region=grav_region,
    )
    if not inside.all():
        msg = (
            "Some constraints are outside the region of the gravity data. "
            "This may result in unexpected behavior."
        )
        logger.warning(msg)


def points_to_grid(
    x: NDArray | pd.Series,
    y: NDArray | pd.Series,
    z: NDArray | pd.Series,
    region: tuple[float, float, float, float],
    spacing: float,
    registration: str = "g",
    coord_names: tuple[str, str] = ("easting", "northing"),
) -> xr.DataArray:
    """
    Place point data onto the nodes of a regular grid, without interpolating. Each
    point is snapped to its nearest node, points sharing a node are averaged, and nodes
    without a point are left as missing values (NaNs).

    Parameters
    ----------
    x, y, z : numpy.ndarray | pandas.Series
        the coordinates and values of the points
    region : tuple[float, float, float, float]
        region of the grid in format (xmin, xmax, ymin, ymax)
    spacing : float
        spacing of the grid
    registration : str, optional
        registration of the grid, ``g`` for gridline or ``p`` for pixel, by default
        ``g``
    coord_names : tuple[str, str], optional
        names to give the grid's x and y dimensions, by default ("easting", "northing")

    Returns
    -------
    xarray.DataArray
        the gridded points
    """
    easting, northing = bd.grid_coordinates(
        region=region,
        spacing=spacing,
        pixel_register=registration == "p",
    )
    eastings, northings = easting[0, :], northing[:, 0]

    values = np.full((northings.size, eastings.size), np.nan)

    # snap each point to its nearest node
    east_index = np.rint(
        (np.asarray(x, dtype=float) - eastings[0]) / (eastings[1] - eastings[0])
    ).astype(int)
    north_index = np.rint(
        (np.asarray(y, dtype=float) - northings[0]) / (northings[1] - northings[0])
    ).astype(int)

    inside = (
        (east_index >= 0)
        & (east_index < eastings.size)
        & (north_index >= 0)
        & (north_index < northings.size)
    )

    # average any points which snapped to the same node
    means = (
        pd.DataFrame(
            {
                "north_index": north_index[inside],
                "east_index": east_index[inside],
                "z": np.asarray(z, dtype=float)[inside],
            }
        )
        .groupby(["north_index", "east_index"])
        .z.mean()
    )
    values[
        means.index.get_level_values("north_index"),
        means.index.get_level_values("east_index"),
    ] = means.to_numpy()

    return xr.DataArray(
        values,
        coords={coord_names[1]: northings, coord_names[0]: eastings},
        dims=(coord_names[1], coord_names[0]),
    )


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


def _restore_dims(
    grid: xr.DataArray,
    original_dims: list[typing.Any],
) -> xr.DataArray:
    """rename a grid's dims back to their original (northing, easting) names"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        dims = list(grid.dims)
        return grid.rename(
            {
                dims[0]: original_dims[0],
                dims[1]: original_dims[1],
            }
        )


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
        region = bd.get_region((df[grid.dims[1]], df[grid.dims[0]]))
        filled = (
            vd.KNeighbors()
            .fit(coords, df_dropped[grid.name])
            .grid(
                region=region,
                shape=grid.shape,
                data_names=original_name,
                dims=(original_dims[1], original_dims[0]),
            )[original_name]
        )
    # elif method == "pygmt":
    #     filled = pygmt.grdfill(grid, mode="n", verbose="q").rename(original_name)
    else:
        msg = "method must be 'rioxarray', or 'verde'"
        raise ValueError(msg)

    # reset coordinate names if changed
    return _restore_dims(filled, original_dims)


def _interpolation_points(
    grid: xr.DataArray,
    data_name: typing.Any,
    dims: list[typing.Any],
    constraints_df: pd.DataFrame | None,
    data_col: str,
    maxdist: float | None,
) -> tuple[pd.DataFrame, tuple[NDArray, NDArray], NDArray]:
    """
    Turn a grid with missing values into a table, and collect the points used to
    interpolate the missing values; the grid's non-NaN nodes and, optionally, extra
    point data. See :func:`fill_missing`.
    """
    df = vd.grid_to_table(grid)
    nans = df[data_name].isna().to_numpy()

    # the grid's non-NaN nodes
    eastings = [df[dims[1]].to_numpy()[~nans]]
    northings = [df[dims[0]].to_numpy()[~nans]]
    data = [df[data_name].to_numpy()[~nans]]

    # and any supplied point data
    if constraints_df is not None:
        missing_cols = {dims[0], dims[1], data_col} - set(constraints_df.columns)
        if missing_cols:
            msg = (
                f"`constraints_df` is missing the columns {sorted(missing_cols)} needed "
                "to fill the grid's missing values"
            )
            raise ValueError(msg)
        points = constraints_df[constraints_df[data_col].notna()]
        eastings.append(points[dims[1]].to_numpy())
        northings.append(points[dims[0]].to_numpy())
        data.append(points[data_col].to_numpy())

    coords = (np.concatenate(eastings), np.concatenate(northings))
    values = np.concatenate(data)

    if maxdist is not None:
        # only keep points which are within maxdist of a missing value
        tree = KDTree(
            np.column_stack(
                (df[dims[1]].to_numpy()[nans], df[dims[0]].to_numpy()[nans])
            ).astype(np.float64)
        )
        distances, _ = tree.query(
            np.column_stack(coords).astype(np.float64),
            k=1,
        )
        keep = distances <= maxdist
        if not np.any(keep):
            msg = (
                "maxdist resulted in no points to interpolate with, increase the value"
            )
            raise ValueError(msg)
        coords = (coords[0][keep], coords[1][keep])
        values = values[keep]

    return df, coords, values


def fill_missing(
    grid: xr.DataArray,
    method: str = "pygmt",
    constraints_df: pd.DataFrame | None = None,
    data_col: str = "upward",
    tension: float = 1,
    dampings: list[float] | None = None,
    maxdist: float | None = None,
) -> xr.DataArray:
    """
    Fill the missing values (NaNs) of a grid by interpolating the grid's non-NaN values
    with a choice of method. Optionally, additional point data, which doesn't need to
    align with the grid, can be included in the interpolation with ``constraints_df``.
    Only the missing values are altered; the grid's existing values are unchanged.

    Parameters
    ----------
    grid : xarray.DataArray
        a 2D grid with missing values (NaNs)
    method : str, optional
        method used to interpolate the missing values, either ``pygmt`` for tensioned
        minimum curvature with :func:`pygmt.surface`, ``splines`` for cross-validated
        bi-harmonic splines with :class:`verde.Spline`, ``nearest`` for the nearest
        neighbor with :class:`verde.KNeighbors`, or ``linear`` / ``cubic`` for
        :class:`verde.Linear` / :class:`verde.Cubic`, by default ``pygmt``
    constraints_df : pandas.DataFrame | None, optional
        additional point data to include in the interpolation, with columns matching the
        grid's dimension names and a data column, by default None
    data_col : str, optional
        name of the data column of ``constraints_df``, by default ``upward``
    tension : float, optional
        tension factor used if ``method`` is ``pygmt``, between 0 (minimum curvature)
        and 1 (harmonic surface), by default 1
    dampings : list[float] | None, optional
        damping values to use in the spline cross validation if ``method`` is
        ``splines``, by default None
    maxdist : float | None, optional
        only interpolate with points which are within this distance of a missing value.
        For expensive methods, such as ``splines``, this can significantly speed up the
        interpolation, by default None

    Returns
    -------
    xarray.DataArray
        the grid with its missing values filled
    """
    if method not in ("pygmt", "splines", "nearest", "linear", "cubic"):
        msg = "method must be 'pygmt', 'splines', 'nearest', 'linear' or 'cubic'"
        raise ValueError(msg)

    if not np.any(grid.isnull()):  # noqa: PD003
        logger.info("grid has no missing values, returning it unaltered")
        return grid

    original_dims = list(grid.sizes.keys())
    # grid_to_table needs a named variable
    named_grid = grid if grid.name is not None else grid.rename("data")
    data_name = named_grid.name

    df, coords, data = _interpolation_points(
        named_grid,
        data_name,
        original_dims,
        constraints_df,
        data_col,
        maxdist,
    )

    if method == "pygmt":
        pygmt = _import_pygmt()
        spacing, region, _, _, registration = maps.get_grid_info(grid)
        filled = pygmt.surface(
            x=coords[0],
            y=coords[1],
            z=data,
            region=region,
            spacing=spacing,
            registration=registration,
            tension=tension,
            verbose="quiet",
        )
        filled = _restore_dims(filled, original_dims)
        if filled.shape != grid.shape:
            msg = (
                "the filled grid doesn't match the shape of the supplied grid, "
                "please raise an issue on GitHub"
            )
            raise ValueError(msg)
        # pygmt returns the coordinates at a lower precision, so reuse the originals
        for dim in original_dims:
            if not np.allclose(filled[dim], grid[dim], atol=float(spacing) / 100):
                msg = (
                    "the filled grid's coordinates don't match those of the supplied "
                    "grid, please raise an issue on GitHub"
                )
                raise ValueError(msg)
        filled = filled.assign_coords({dim: grid[dim] for dim in original_dims})
    else:
        if method == "nearest":
            interpolator: typing.Any = vd.KNeighbors(k=1)
        elif method == "linear":
            interpolator = vd.Linear()
        elif method == "cubic":
            interpolator = vd.Cubic()
        else:
            interpolator = _fit_spline_cv(coords, data, None, dampings)

        if method != "splines":
            interpolator.fit(coords, data)

        # predict only at the missing values
        nans = df[data_name].isna().to_numpy()
        # as a float64 array, since the grid may be a lower precision dtype
        values = df[data_name].to_numpy(dtype=np.float64, copy=True)
        values[nans] = interpolator.predict(
            (
                df[original_dims[1]].to_numpy()[nans],
                df[original_dims[0]].to_numpy()[nans],
            )
        )
        df[data_name] = values
        filled = df.set_index([original_dims[0], original_dims[1]]).to_xarray()[
            data_name
        ]

    result: xr.DataArray = grid.where(grid.notnull(), filled)  # noqa: PD004

    if np.any(result.isnull()):  # noqa: PD003
        msg = (
            "missing values are still present in the grid, this may be due to the "
            "choice of method; `linear` and `cubic` don't extrapolate outside of the "
            "supplied points. Re-run with method `pygmt`, `splines` or `nearest` to "
            "fill the remaining values."
        )
        logger.warning(msg)

    return result.rename(grid.name)


def region_mask(
    grid: xr.DataArray,
    region: tuple[float, float, float, float] | list[float],
) -> xr.DataArray:
    """
    Make a mask with values of 1 inside a region and 0 outside the region.
    """

    # get coordinate names; dims are ordered (northing, easting)
    coord_names = list(grid.sizes.keys())

    mask_easting = (grid[coord_names[1]] >= region[0]) & (
        grid[coord_names[1]] <= region[1]
    )
    mask_northing = (grid[coord_names[0]] >= region[2]) & (
        grid[coord_names[0]] <= region[3]
    )

    return xr.where(mask_easting & mask_northing, 1, 0)


def filter_grid(
    grid: xr.DataArray,
    filter_width: float | None = None,
    height_displacement: float | None = None,
    filter_type: str = "lowpass",
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
    filter_type : str, optional
        type of filter to use from 'lowpass', 'highpass' 'up_deriv', 'easting_deriv',
        'northing_deriv', 'up_continue', or 'total_gradient', by default "lowpass"
    pad_width_factor : int, optional
        factor of grid width to pad the grid by, by default 3, which equates to a pad
        with a width of 1/3 of the grid width.
    pad_mode : str, optional
        mode of padding, one of the modes accepted by :func:`xrft.pad`, such as
        "constant" or "linear_ramp", by default "linear_ramp"
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
    filled = _restore_dims(filled, original_dims)

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

    if filter_type == "lowpass":
        if filter_width is None:
            msg = "filter_width must be provided if filter_type is 'lowpass'"
            raise ValueError(msg)
        filt = hm.gaussian_lowpass(padded, wavelength=filter_width).rename("filt")
    elif filter_type == "highpass":
        if filter_width is None:
            msg = "filter_width must be provided if filter_type is 'highpass'"
            raise ValueError(msg)
        filt = hm.gaussian_highpass(padded, wavelength=filter_width).rename("filt")
    elif filter_type == "up_deriv":
        filt = hm.derivative_upward(padded).rename("filt")
    elif filter_type == "easting_deriv":
        filt = hm.derivative_easting(padded).rename("filt")
    elif filter_type == "northing_deriv":
        filt = hm.derivative_northing(padded).rename("filt")
    elif filter_type == "up_continue":
        if height_displacement is None:
            msg = "height_displacement must be provided if filter_type is 'up_continue'"
            raise ValueError(msg)
        filt = hm.upward_continuation(
            padded, height_displacement=height_displacement
        ).rename("filt")
    elif filter_type == "total_gradient":
        filt = hm.total_gradient_amplitude(padded).rename("filt")
    else:
        msg = (
            "filter_type must be 'lowpass', 'highpass' 'up_deriv', 'easting_deriv', "
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
    result = _restore_dims(result, original_dims)

    return result.rename(original_name)


def dist_nearest_points(
    targets: pd.DataFrame,
    data: pd.DataFrame | xr.DataArray | xr.Dataset,
    coord_names: tuple[str, str] | None = None,
) -> typing.Any:
    """
    for all grid cells calculate to the distance to the nearest target.

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

    min_dist, _ = KDTree(df_targets.to_numpy()).query(df_data.to_numpy(), 1)

    df_data["min_dist"] = min_dist

    if isinstance(data, pd.DataFrame):
        return df_data
    return df_data.set_index([coord_names[0], coord_names[1]][::-1]).to_xarray()


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
    low_percentile: float = 0,
    high_percentile: float = 100,
) -> xr.DataArray:
    """
    Normalize a grid between provided values. Be default normalizes the grids minimum
    values to 0 and maximum value to 1, alternatively, low_percentile and
    high_percentile can be passed for more robust normilization. For example, values of
    2% and 98% would set the grids 2nd percentile to 0, and all data below the 2nd
    percentile would be clipped to 0, and same for the high values of the grid.

    Parameters
    ----------
    da : xarray.DataArray
        grid to normalize
    low : float, optional
        lower value for normalization, by default 0
    high : float, optional
        higher value for normalization, by default 1
    low_percentile : float, optional
        which grid value as percentile to normalize to the low value, by default 0, the
        grids true minimum
    high_percentile : float, optional
        which grid value as percentile to normalize to the high value, by default 100,
        the grids true maximum

    Returns
    -------
    xarray.DataArray
        a normalized grid
    """
    da = da.copy()

    min_val = da.quantile(low_percentile / 100, skipna=True)
    max_val = da.quantile(high_percentile / 100, skipna=True)

    if float(max_val - min_val) == 0:
        logger.warning(
            "grid has no range between the %s and %s percentiles, setting all values "
            "to `low` (%s)",
            low_percentile,
            high_percentile,
            low,
        )
        # multiply by 0 instead of zeros_like to keep the grid's nans
        da2: xr.DataArray = (da * 0) + low
    else:
        da2 = (high - low) * (((da - min_val) / (max_val - min_val)).clip(0, 1)) + low

    return da2.drop_vars("quantile", errors="ignore")


def scale_normalized(
    sample: NDArray,
    bounds: NDArray,
) -> NDArray:
    """
    Rescales the sample space into the unit hypercube, bounds = [0,1]

    Parameters
    ----------
    sample : NDArray
        sampled values
    bounds : NDArray
        bounds of the sampling

    Returns
    -------
    NDArray
        sampled values normalized from 0 to 1
    """

    scaled_sample = np.zeros(sample.shape)

    for j in range(sample.shape[1]):
        scaled_sample[:, j] = (sample[:, j] - bounds[j][0]) / (
            bounds[j][1] - bounds[j][0]
        )

    return scaled_sample


def normalized_mindist(
    points: pd.DataFrame,
    grid: xr.DataArray,
    low: float | None = None,
    high: float | None = None,
    mindist: float | None = None,
    maxdist: float | None = None,
    region: list[float] | None = None,
    outside_region_value: float | None = None,
    low_percentile: float = 0,
    high_percentile: float = 100,
) -> xr.DataArray:
    """
    Find the minimum distance between each grid cell and the nearest point. If low and
    high are provided, normalize the min dists grid between these values. Grid cells
    closer than `mindist` to a point are set to `low`, and grid cells outside of
    `region` are set to `outside_region_value` (`low` by default). These fixed-value
    cells are excluded when calculating the percentiles used for the normalization.

    There are two ways of choosing which distances map to `low` and `high`: either with
    percentiles of the min dists grid (`low_percentile` and `high_percentile`), or with
    actual distance values (`mindist` and `maxdist`). Providing `maxdist` switches to
    the latter, where distances at or below `mindist` (0 by default) are set to `low`,
    distances at or above `maxdist` are set to `high`, and distances in between are
    scaled linearly. The two ways are mutually exclusive.

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
        the minimum allowed distance, all grid cells with distances below this are set
        to `low`, by default None
    maxdist : float | None, optional
        the distance which is mapped to `high`, with `mindist` (0 if not provided)
        mapped to `low` and distances in between scaled linearly. Requires `low` and
        `high`, and can't be used together with `low_percentile` or `high_percentile`.
        By default None, meaning percentiles are used instead.
    region : list[float] | None, optional
        bounding region, in the format (min_easting, max_easting, min_northing,
        max_northing), for which all grid cells outside will be set to
        `outside_region_value`, by default None
    outside_region_value : float | None, optional
        the value to set for areas outside of the region, by default is `low`. Takes
        precedence over `mindist` where both apply.
    low_percentile : float, optional
        which grid value as percentile to normalize to the low value, by default 0, the
        grids true minimum
    high_percentile : float, optional
        which grid value as percentile to normalize to the high value, by default 100,
        the grids true maximum

    Returns
    -------
    xarray.DataArray
        grid of normalized minimum distances
    """

    if (low is None) != (high is None):
        msg = "must provide both `low` and `high`, or neither"
        raise ValueError(msg)

    if not 0 <= low_percentile < high_percentile <= 100:
        msg = (
            "must have 0 <= `low_percentile` < `high_percentile` <= 100, got "
            f"{low_percentile} and {high_percentile}"
        )
        raise ValueError(msg)

    if outside_region_value is not None and region is None:
        msg = "must provide `region` to use `outside_region_value`"
        raise ValueError(msg)

    if maxdist is not None:
        if low is None or high is None:
            msg = "must provide `low` and `high` to use `maxdist`"
            raise ValueError(msg)
        if (low_percentile, high_percentile) != (0, 100):
            msg = (
                "can't use `maxdist` together with `low_percentile` or "
                "`high_percentile`, normalize either with distance values or with "
                "percentiles"
            )
            raise ValueError(msg)
        if maxdist <= (0 if mindist is None else mindist):
            msg = (
                f"`maxdist` ({maxdist}) must be greater than `mindist` "
                f"({0 if mindist is None else mindist})"
            )
            raise ValueError(msg)

    grid = copy.deepcopy(grid)

    # get coordinate names
    original_dims = list(grid.sizes.keys())

    constraints_df = points.copy()

    min_dist: xr.DataArray = dist_nearest_points(
        targets=constraints_df,
        data=grid,
        coord_names=(str(original_dims[1]), str(original_dims[0])),
    ).min_dist

    # without normalizing, distances are in their original units and `low_value` is
    # just the value given to the fixed-value cells
    low_value = 0.0 if low is None else low
    if outside_region_value is None:
        outside_region_value = low_value

    # cells which get a fixed value instead of a distance
    below_mindist = None if mindist is None else min_dist < mindist
    outside_region = None if region is None else region_mask(min_dist, region) == 0

    # nans of the supplied grid, to be restored at the end
    nan_mask = min_dist.isnull()  # noqa: PD003

    if maxdist is not None:
        # normalize with distance values instead of percentiles
        assert low is not None
        assert high is not None
        low_dist = 0.0 if mindist is None else mindist
        min_dist = (high - low) * (
            ((min_dist - low_dist) / (maxdist - low_dist)).clip(0, 1)
        ) + low
    elif low is not None and high is not None:
        # exclude fixed-value cells so they don't influence the percentiles
        to_normalize = min_dist
        if below_mindist is not None:
            to_normalize = to_normalize.where(~below_mindist)
        if outside_region is not None:
            to_normalize = to_normalize.where(~outside_region)

        if bool(to_normalize.isnull().all()):  # noqa: PD003
            msg = (
                "no grid cells remain to normalize, all are either nans, below "
                "`mindist`, or outside of `region`"
            )
            raise ValueError(msg)

        min_dist = normalize_xarray(
            to_normalize,
            low=low,
            high=high,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
        )

    # set the fixed values, region takes precedence where both apply
    if below_mindist is not None:
        min_dist = min_dist.where(~below_mindist, low_value)
    if outside_region is not None:
        min_dist = min_dist.where(~outside_region, outside_region_value)

    # add nans back
    return min_dist.where(~nan_mask)


def sample_grids(
    df: pd.DataFrame,
    grid: str | xr.DataArray,
    sampled_name: str,
    coord_names: tuple[str, str] | None = None,
    method: str = "cubic",
) -> pd.DataFrame:
    """
    Sample a grid at the coordinates of each row of a dataframe. Points which fall on
    a grid node take that node's value exactly, the rest are interpolated.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing columns 'easting', 'northing', 'longitude', 'latitude' or
        columns with names given by ``coord_names``.
    grid : str or xarray.DataArray
        Grid to sample, either file name or xarray.DataArray
    sampled_name : str
        Name for sampled column
    coord_names : tuple[str, str] | None, optional
        names of the x and y coordinate columns of ``df``, by default None which uses
        'easting'/'northing', 'x'/'y' or 'longitude'/'latitude'
    method : str, optional
        interpolation method used away from the grid's nodes, either ``cubic``,
        ``linear`` or ``nearest``, by default ``cubic``

    Returns
    -------
    pandas.DataFrame
        Dataframe with new column (sampled_name) of sample values from (grid)
    """
    if isinstance(grid, str):
        grid = xr.load_dataarray(grid)

    if len(grid.dims) > 2:
        grid = grid.squeeze()

    # get the x and y column names of the dataframe
    if coord_names is None:
        for names in (("easting", "northing"), ("x", "y"), ("longitude", "latitude")):
            if all(n in df.columns for n in names):
                coord_names = names
                break
        else:
            msg = (
                "dataframe must have coordinate columns 'easting'/'northing', 'x'/'y' "
                "or 'longitude'/'latitude', or `coord_names` must be supplied"
            )
            raise ValueError(msg)
    elif not all(n in df.columns for n in coord_names):
        msg = f"dataframe must have columns {coord_names}"
        raise ValueError(msg)

    # grid dims are ordered (northing, easting)
    grid_northing, grid_easting = (str(d) for d in grid.dims)

    easting = xr.DataArray(
        df[coord_names[0]].to_numpy(dtype=float), dims="sample_points"
    )
    northing = xr.DataArray(
        df[coord_names[1]].to_numpy(dtype=float), dims="sample_points"
    )
    indexers = {grid_easting: easting, grid_northing: northing}

    def interp(da: xr.DataArray, interp_method: str) -> NDArray:
        return np.asarray(
            da.interp(
                indexers,
                method=interp_method,
                kwargs={"bounds_error": False},
            ),
            dtype=float,
        )

    if method == "cubic" and bool(np.isnan(grid.to_numpy()).any()):
        # scipy's cubic interpolation fails on grids containing NaNs, so fit it to a
        # filled grid, then set back to NaN any point with a NaN among its neighboring
        # nodes, which is where linear interpolation gives NaN
        values = interp(grid.fillna(float(grid.mean())), method)
        values[np.isnan(interp(grid, "linear"))] = np.nan
    else:
        values = interp(grid, method)

    # interpolation introduces floating point error even at the grid's own nodes, so
    # take the value directly wherever a point falls on a node
    nearest = grid.sel(indexers, method="nearest")
    spacing = maps.get_grid_info(grid)[0]
    on_node = (
        np.abs(np.asarray(nearest[grid_easting], dtype=float) - easting.to_numpy())
        <= spacing * 1e-6
    ) & (
        np.abs(np.asarray(nearest[grid_northing], dtype=float) - northing.to_numpy())
        <= spacing * 1e-6
    )
    values[on_node] = np.asarray(nearest, dtype=float)[on_node]

    result = df.copy()
    result[sampled_name] = values

    return result


def resample_grid(
    grid: xr.DataArray,
    spacing: float | None = None,
    region: tuple[float, float, float, float] | None = None,
    registration: str | None = None,
    method: str = "linear",
) -> xr.DataArray:
    """
    Resample a grid onto a new spacing, region and/or registration by interpolating it
    at the new grid's coordinates.

    Parameters
    ----------
    grid : xarray.DataArray
        grid to resample
    spacing : float | None, optional
        new spacing for the grid, by default None which keeps the current spacing
    region : tuple[float, float, float, float] | None, optional
        new region for the grid in format (xmin, xmax, ymin, ymax), by default None
        which keeps the current region
    registration : str | None, optional
        new registration, ``g`` for gridline or ``p`` for pixel, by default None which
        keeps the current registration
    method : str, optional
        interpolation method, either ``linear``, ``nearest`` or ``cubic``, by default
        ``linear``

    Returns
    -------
    xarray.DataArray
        the resampled grid
    """
    initial_spacing, initial_region, _, _, initial_registration = maps.get_grid_info(
        grid
    )

    spacing = initial_spacing if spacing is None else spacing
    region = initial_region if region is None else region
    registration = initial_registration if registration is None else registration

    if (
        np.allclose(spacing, initial_spacing)
        and np.allclose(region, initial_region)
        and registration == initial_registration
    ):
        logger.info("returning original grid")
        return grid

    if spacing > initial_spacing:
        logger.info(
            "requested spacing (%s) is larger than the original (%s), the grid will be "
            "sub-sampled without filtering",
            spacing,
            initial_spacing,
        )

    # dims are ordered (northing, easting)
    northing_name, easting_name = (str(d) for d in grid.dims)

    easting, northing = bd.grid_coordinates(
        region=region,
        spacing=spacing,
        pixel_register=registration == "p",
    )

    resampled: xr.DataArray = grid.interp(
        {
            easting_name: easting[0, :],
            northing_name: northing[:, 0],
        },
        method=method,
        kwargs={"bounds_error": False},
    )

    return resampled


def get_spacing(prisms_df: pd.DataFrame) -> None:  # noqa: ARG001
    """
    DEPRECATED: function has been removed
    """
    # pylint: disable=W0613
    msg = "Function `get_spacing` deprecated"
    raise DeprecationWarning(msg)


def _deprecated_parameter(
    old_name: str,
    old_value: typing.Any,
    new_name: str,
    new_value: typing.Any,
) -> typing.Any:
    """
    Map a deprecated parameter onto its new name, warning if the old name was used.
    """
    if old_value is None:
        return new_value
    warnings.warn(
        f"`{old_name}` is deprecated, please use `{new_name}` instead.",
        UserWarning,
        stacklevel=3,
    )
    return old_value


def _normalize_mask(
    mask: xr.DataArray,
    reference: xr.DataArray,
) -> xr.DataArray:
    """
    Convert a mask into values of 1 (include) and 0 (exclude), where the excluded cells
    are those with values of 0, False, or NaN. This allows both the ``mask`` convention
    of :func:`invert4geom.create_model` (NaNs are excluded) and of ``mask_column``
    (zeros are excluded) to be used for the gravity data. The mask must be on the same
    grid as ``reference``.
    """
    if dict(mask.sizes) != dict(reference.sizes):
        msg = (
            f"mask has dimensions {dict(mask.sizes)} but the gravity data has "
            f"{dict(reference.sizes)}, they must be on the same grid"
        )
        raise ValueError(msg)
    for dim in reference.dims:
        if (dim not in mask.coords) or not np.allclose(mask[dim], reference[dim]):
            msg = (
                f"mask coordinate '{dim}' doesn't match that of the gravity data, the "
                "mask must be on the same grid as the gravity data"
            )
            raise ValueError(msg)

    values = mask.astype("float64")
    normalized: xr.DataArray = xr.where(np.isfinite(values) & (values != 0), 1.0, 0.0)
    return normalized.rename("mask")


def _subset_data(values: typing.Any, keep: NDArray) -> typing.Any:
    """subset an array, a tuple of arrays, or None, with a boolean mask"""
    if values is None:
        return None
    if isinstance(values, tuple):
        return tuple(np.asarray(v)[keep] for v in values)
    return np.asarray(values)[keep]


def _block_reduce_points(
    coords: tuple[typing.Any, typing.Any],
    data: typing.Any,
    weights: typing.Any,
    block_size: float,
    block_reduction: str,
    region: tuple[float, float, float, float] | None,
    mask: NDArray | None = None,
    uncertainty: bool = False,
    center_coordinates: bool = True,
) -> tuple[typing.Any, typing.Any, typing.Any]:
    """
    Block-reduce point data before interpolating. Means are computed with
    :class:`verde.BlockMean`, which also reduces the weights; medians are computed with
    :class:`verde.BlockReduce` and any weights are dropped. ``data`` can be a tuple of
    arrays to reduce several data components together, in which case the reduced weights
    of the first component are returned. If ``mask`` is provided, only the points where
    it is True are block-reduced, and the remaining points are appended, unaltered, to
    the reduced points.
    """
    if mask is None:
        return _block_reduce_all_points(
            coords,
            data,
            weights,
            block_size,
            block_reduction,
            region,
            uncertainty=uncertainty,
            center_coordinates=center_coordinates,
        )

    mask = np.asarray(mask, dtype=bool)
    first_component = data[0] if isinstance(data, tuple) else data
    if mask.shape != np.asarray(first_component).shape:
        msg = "block reduction mask must be the same length as the data"
        raise ValueError(msg)

    if not mask.any():
        logger.warning("no points selected for block-reduction, skipping it")
        return coords, data, weights

    reduced_coords, reduced_data, reduced_weights = _block_reduce_all_points(
        tuple(_subset_data(c, mask) for c in coords),
        _subset_data(data, mask),
        _subset_data(weights, mask),
        block_size,
        block_reduction,
        region,
        uncertainty=uncertainty,
        center_coordinates=center_coordinates,
    )

    # add back the points which weren't block-reduced
    coords = tuple(
        np.concatenate([np.asarray(r), _subset_data(c, ~mask)])
        for r, c in zip(reduced_coords, coords, strict=True)
    )
    if isinstance(data, tuple):
        data = tuple(
            np.concatenate([np.asarray(r), np.asarray(d)[~mask]])
            for r, d in zip(reduced_data, data, strict=True)
        )
    else:
        data = np.concatenate([np.asarray(reduced_data), _subset_data(data, ~mask)])
    if weights is None or reduced_weights is None:
        # weights are dropped by verde.BlockReduce, so they can't be reused
        weights = None
    else:
        weights = np.concatenate(
            [np.asarray(reduced_weights), _subset_data(weights, ~mask)]
        )

    return coords, data, weights


def _block_reduce_all_points(
    coords: tuple[typing.Any, typing.Any],
    data: typing.Any,
    weights: typing.Any,
    block_size: float,
    block_reduction: str,
    region: tuple[float, float, float, float] | None,
    uncertainty: bool = False,
    center_coordinates: bool = True,
) -> tuple[typing.Any, typing.Any, typing.Any]:
    """
    Block-reduce all the supplied point data, see :func:`_block_reduce_points`.
    """
    multiple_components = isinstance(data, tuple)

    if block_reduction == "mean":
        reducer = vd.BlockMean(
            spacing=block_size,
            region=region,
            center_coordinates=center_coordinates,
            uncertainty=uncertainty,
        )
        if multiple_components and weights is not None:
            # verde expects one weights array per data component
            weights = tuple(weights for _ in data)
        coords, data, weights = reducer.filter(
            coordinates=coords,
            data=data,
            weights=weights,
        )
        if multiple_components and isinstance(weights, tuple):
            weights = weights[0]
    elif block_reduction == "median":
        if weights is not None:
            msg = (
                "weights are ignored when block_reduction is 'median', to use "
                "weights, set block_reduction to 'mean'"
            )
            logger.warning(msg)

        reducer = vd.BlockReduce(
            reduction=np.median,
            spacing=block_size,
            region=region,
            center_coordinates=center_coordinates,
        )
        coords, data = reducer.filter(
            coordinates=coords,
            data=data,
        )
        # BlockReduce doesn't reduce the weights, so they can't be reused
        weights = None
    else:
        msg = "block_reduction must be 'mean' or 'median'"
        raise ValueError(msg)

    return coords, data, weights


def _fit_spline_cv(
    coords: tuple[typing.Any, typing.Any],
    data: typing.Any,
    weights: typing.Any,
    dampings: list[float] | None,
) -> vd.Spline:
    """fit a cross-validated spline to the data, suppressing known verde warnings"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The mindist parameter of verde.Spline"
        )
        warnings.filterwarnings("ignore", message="The default scoring will change")
        return optimal_spline_damping(
            coordinates=coords,
            data=data,
            weights=weights,
            dampings=dampings,
        )


def _check_grid_to_fill(
    grid_to_fill: xr.DataArray,
    region: tuple[float, float, float, float],
    spacing: float,
    registration: str,
    coord_names: tuple[str, str],
) -> xr.DataArray:
    """
    Ensure the supplied grid matches the requested grid and restore its dimensions.
    See :func:`create_topography`.
    """
    grid = _restore_dims(grid_to_fill, [coord_names[1], coord_names[0]])

    grid_spacing, grid_region, _, _, grid_registration = maps.get_grid_info(grid)
    if (
        not np.allclose(float(grid_spacing), spacing)
        or not np.allclose(grid_region, region)
        or grid_registration != registration
    ):
        msg = (
            f"`grid_to_fill` (spacing: {grid_spacing}, region: {grid_region}, "
            f"registration: {grid_registration}) doesn't match the requested grid "
            f"(spacing: {spacing}, region: {region}, registration: {registration}), "
            "please resample it before passing it to `create_topography`"
        )
        raise ValueError(msg)

    return grid


def _fill_topography_grid(
    grid_to_fill: xr.DataArray,
    method: str,
    region: tuple[float, float, float, float],
    spacing: float,
    registration: str,
    coord_names: tuple[str, str],
    constraints_df: pd.DataFrame | None,
    tension: float,
    dampings: list[float] | None,
    maxdist: float | None,
    block_size: float | None,
    block_reduction: str,
    block_reduction_col: str | None,
    grid_points_col: str,
) -> xr.DataArray:
    """
    Fill the missing values of a supplied topography grid with the constraint points
    which aren't already included in the grid. See :func:`create_topography`.
    """
    grid = _check_grid_to_fill(
        grid_to_fill,
        region,
        spacing,
        registration,
        coord_names,
    )

    fill_points = None
    if constraints_df is not None:
        if grid_points_col in constraints_df.columns:
            # points which are already included in the grid shouldn't be reused
            fill_points = constraints_df[constraints_df[grid_points_col] == False]  # noqa: E712 # pylint: disable=singleton-comparison
        else:
            msg = (
                f"`constraints_df` has no column `{grid_points_col}`, using all the "
                "points to fill the missing values of `grid_to_fill`"
            )
            logger.info(msg)
            fill_points = constraints_df

        if block_size is not None:
            coords = (fill_points[coord_names[0]], fill_points[coord_names[1]])
            if block_reduction_col is None:
                mask = None
            elif block_reduction_col not in fill_points.columns:
                msg = (
                    f"`block_reduction_col` '{block_reduction_col}' not in "
                    "`constraints_df` columns"
                )
                raise ValueError(msg)
            else:
                mask = fill_points[block_reduction_col].to_numpy(dtype=bool)
            coords, data, _ = _block_reduce_points(
                coords,
                fill_points.upward,
                None,
                block_size,
                block_reduction,
                region,
                mask=mask,
            )
            fill_points = pd.DataFrame(
                {
                    coord_names[0]: coords[0],
                    coord_names[1]: coords[1],
                    "upward": data,
                }
            )

    return fill_missing(
        grid,
        method=method,
        constraints_df=fill_points,
        data_col="upward",
        tension=tension,
        dampings=dampings,
        maxdist=maxdist,
    )


def create_topography(
    method: str,
    region: tuple[float, float, float, float],
    spacing: float,
    dampings: list[float] | None = None,
    registration: str = "g",
    upward: float | None = None,
    upwards: float | None = None,
    coord_names: tuple[str, str] = ("easting", "northing"),
    projection: pyproj.Proj | None = None,
    constraints_df: pd.DataFrame | None = None,
    weights: pd.Series | NDArray | None = None,
    weights_col: str | None = None,
    block_size: float | None = None,
    block_reduction: str = "median",
    block_reduction_col: str | None = None,
    tension: float = 1,
    grid_to_fill: xr.DataArray | None = None,
    maxdist: float | None = None,
    grid_points_col: str = "grid_points",
    upper_confining_layer: xr.DataArray | None = None,
    lower_confining_layer: xr.DataArray | None = None,
    dataset_to_add: xr.Dataset | None = None,
) -> xr.Dataset:
    """
    Create a grid of topography data from either the interpolation of point data, with
    bi-harmonic splines (:class:`verde.Spline`) or tensioned minimum curvature
    (:func:`pygmt.surface`), or creating a grid of constant value. Optionally, a subset
    of point data can be interpolated and then merged with an existing grid. To do this,
    ``constraints_df`` must contain two additional columns of booleans, ``inside`` which
    is True for points inside the region of interest, and False otherwise, and
    ``buffer`` which is True for points within a buffer region around the region of
    interest, and False otherwise. Inside and Buffer points are used to interpolated the
    data, and then the interpolated data (without the buffer zone) is merged with the
    points outside the region of interest. For interpolations, ``block_size`` can be
    supplied to perform a block-median filtering of the points before interpolating,
    reducing the computational cost. This block-reduction can be limited to a subset of
    the points with ``block_reduction_col``. Alternatively, an existing grid with
    missing values (NaNs) can be supplied with ``grid_to_fill``, and only its missing
    values will be interpolated, using the constraint points which aren't already
    included in the grid (where the column named by ``grid_points_col`` is False), or
    filled with a constant value if ``method`` is ``flat``.

    Parameters
    ----------
    method : str
        method to use, either ``flat``, ``splines`` or ``pygmt``. If using
        ``grid_to_fill``, ``nearest``, ``linear`` and ``cubic`` are also available, see
        :func:`fill_missing`
    region : tuple[float, float, float, float]
        bounding region to use for the output grid. If you are using projected
        coordinates (meters), this is in the format (min_easting, max_easting,
        min_northing, max_northing). If using geographic coordinates (latitude and
        longitude), this is in the format (min_longitude, max_longitude, min_latitude,
        max_latitude)
    spacing : float
        spacing of the grid in meters if using projected coordinates, or in decimal
        degrees if using geographic coordinates
    dampings : list[float] | None, optional
        damping values to use in spline cross validation for method ``spline``, by default
        None
    registration : str, optional
        choose between gridline ``g`` or pixel ``p`` registration, by default ``g``
    upward : float | None, optional
        constant elevation in meters to use for method ``flat``. If ``grid_to_fill`` is
        also provided, its missing values are filled with this value, by default None
    upwards : float | None, optional
        deprecated, use `upward` instead, by default None
    coord_names : tuple[str, str], optional
        names to use for coordinates, should be either (easting, northing) or
        (longitude, latitude), by default (easting, northing)
    projection : pyproj.Proj | None, optional
        if using geographic coordinates (latitude and longitude) and the splines method,
        provide a pyproj Proj object to convert to cartesian coordinates, by default
        None. Not needed for method ``pygmt``, which grids in the supplied coordinates.
    constraints_df : pandas.DataFrame | None, optional
        dataframe with column 'upward' to use for methods ``splines`` and ``pygmt``, and
        optionally columns ``inside`` and ``buffer``, by default None
    weights : pandas.Series | numpy.ndarray | None, optional
        weight to use for fitting the spline. Typically, this should be 1 over the data
        uncertainty squared, by default None
    weights_col : str | None, optional
        instead of passing the weights, pass the name of the column containing the
        weights, by default None
    block_size : float | None, optional
        block size to use for block-reduction of constraint points before
        interpolating. In meters if using projected coordinates, or in decimal degrees
        if using geographic coordinates. If None, no block-reduction is applied, by
        default None
    block_reduction: str, optional
        type of block reduction to apply. If ``median``, :class:`verde.BlockReduce` is
        used and any weights are ignored and dropped. If ``mean``,
        :class:`verde.BlockMean` is used and the reduced weights are kept for the
        interpolation. Defaults to ``median``.
    block_reduction_col : str | None, optional
        name of a boolean column in ``constraints_df``, where True indicates points
        which should be block-reduced. Points which are False are left unaltered and
        added back to the block-reduced points before interpolating. If None, all points
        are block-reduced, by default None
    tension : float, optional
        tension factor used if ``method`` is ``pygmt``, between 0 (minimum curvature)
        and 1 (harmonic surface), by default 1
    grid_to_fill : xarray.DataArray | None, optional
        instead of interpolating all the constraint points, supply an existing grid,
        with the same region, spacing and registration as requested, whose missing
        values (NaNs) should be filled with :func:`fill_missing`. The constraint points
        which aren't already included in the grid, indicated by the boolean column
        named by ``grid_points_col`` being False, are used in the interpolation. If
        ``method`` is ``flat``, the missing values are instead filled with the constant
        value ``upward``, by default None
    maxdist : float | None, optional
        if using ``grid_to_fill``, only interpolate with points within this distance of
        a missing value. Not used with method ``flat``, by default None
    grid_points_col : str, optional
        name of the boolean column of ``constraints_df``, where True indicates points
        which are already included in ``grid_to_fill`` and shouldn't be reused in the
        interpolation, by default ``grid_points``
    upper_confining_layer : xarray.DataArray | None, optional
        layer which the inverted topography should always be below, by default None
    lower_confining_layer : xarray.DataArray | None, optional
        layer which the inverted topography should always be above, by default None
    dataset_to_add : xarray.Dataset | None, optional
        additional variables to add to the topography dataset, such as variables `mask`
        or `geocentric_radius`, by default None

    Returns
    -------
    xarray.Dataset
        a topography grid
    """
    if upwards is not None:
        warnings.warn(
            "`upwards` is deprecated, please use `upward` instead.",
            UserWarning,
            stacklevel=2,
        )
        upward = upwards

    if (method != "pygmt") & (tension != 1):
        msg = "`tension` is only used if `method` is 'pygmt'"
        raise ValueError(msg)

    if (grid_to_fill is None) & (maxdist is not None):
        msg = "`maxdist` is only used if `grid_to_fill` is provided"
        raise ValueError(msg)

    if (method == "flat") & (maxdist is not None):
        msg = "`maxdist` can't be used with method 'flat'"
        raise ValueError(msg)

    if method == "flat":
        if registration == "g":
            pixel_register = False
        elif registration == "p":
            pixel_register = True
        else:
            msg = "registration must be 'g' or 'p'"
            raise ValueError(msg)

        if upward is None:
            msg = "upward must be provided if method is `flat`"
            raise ValueError(msg)

        if grid_to_fill is not None:
            # fill the missing values of the supplied grid with the constant value
            grid = _check_grid_to_fill(
                grid_to_fill,
                region,
                spacing,
                registration,
                coord_names,
            ).fillna(upward)
        else:
            # create grid of coordinates
            (x, y) = bd.grid_coordinates(  # pylint: disable=unbalanced-tuple-unpacking
                region=region,
                spacing=spacing,
                pixel_register=pixel_register,
            )
            # make flat topography of value = upward
            grid = vd.make_xarray_grid(
                (x, y),
                np.ones_like(x) * upward,
                data_names="upward",
                dims=(coord_names[1], coord_names[0]),
            ).upward

    elif grid_to_fill is not None:
        # only interpolate the missing values of the supplied grid
        grid = _fill_topography_grid(
            grid_to_fill,
            method,
            region,
            spacing,
            registration,
            coord_names,
            constraints_df,
            tension,
            dampings,
            maxdist,
            block_size,
            block_reduction,
            block_reduction_col,
            grid_points_col,
        )
        grid = grid.assign_attrs(damping=None)

    elif method in ("splines", "pygmt"):  # pylint: disable=too-many-nested-blocks
        # get coordinates of the constraint points
        if constraints_df is None:
            msg = f"constraints_df must be provided if method is `{method}`"
            raise ValueError(msg)

        df = constraints_df.copy()

        # if only 1 point, return a flat topography
        if len(df) == 1:
            # create grid of coordinates
            (x, y) = bd.grid_coordinates(  # pylint: disable=unbalanced-tuple-unpacking
                region=region,
                spacing=spacing,
            )
            # make flat topography of value = upward
            grid = vd.make_xarray_grid(
                (x, y),
                np.ones_like(x) * df.upward.to_numpy(),
                data_names="upward",
                dims=(coord_names[1], coord_names[0]),
            ).upward
        else:
            if (
                method == "splines"
                and coord_names == ("longitude", "latitude")
                and projection is None
            ):
                msg = (
                    "If using geographic coordinates (latitude and longitude) and the "
                    "spline method, a pyproj.Proj projection must be provided to "
                    "convert to projected coordinates."
                )
                raise ValueError(msg)

            # convert to projected coordinates if needed
            if method == "splines" and projection is not None:
                projected_points = [
                    projection(lon, lat)
                    for (lon, lat) in zip(df.longitude, df.latitude, strict=True)
                ]
                eastings, northings = zip(*projected_points, strict=True)
                df["easting"] = eastings
                df["northing"] = northings

                # convert region in lat/lon to projected easting/northing
                corners_lonlat = [
                    (region[0], region[2]),
                    (region[1], region[2]),
                    (region[1], region[3]),
                    (region[0], region[3]),
                ]
                projected_corners = [
                    projection(lon, lat) for (lon, lat) in corners_lonlat
                ]
                eastings, northings = zip(*projected_corners, strict=True)
                cartesian_region = (
                    min(eastings),
                    max(eastings),
                    min(northings),
                    max(northings),
                )
                if block_size is None:
                    cartesian_block_size = None
                else:
                    cartesian_block_size = (
                        block_size * 111139
                    )  # approx conversion from degrees latitude to meters
            else:
                cartesian_region = region
                cartesian_block_size = block_size

            if method == "pygmt":
                # pygmt grids in the supplied coordinates, no projection needed
                fit_coord_names = coord_names
                fit_region = region
                fit_block_size = block_size
            else:
                # verde splines are fit in projected (cartesian) coordinates
                fit_coord_names = ("easting", "northing")
                fit_region = cartesian_region
                fit_block_size = cartesian_block_size

            # optionally only interpolate the points inside the region and its buffer
            subset_points = {"inside", "buffer"} <= set(df.columns)
            if subset_points:
                df_to_interpolate = df[df.inside | df.buffer]
                df_outside_buffer = df[(df.inside == False) & (df.buffer == False)]  # noqa: E712 # pylint: disable=singleton-comparison
            else:
                df_to_interpolate = df

            coords = (
                df_to_interpolate[fit_coord_names[0]],
                df_to_interpolate[fit_coord_names[1]],
            )
            data = df_to_interpolate.upward
            if weights_col is not None:
                weights = df_to_interpolate[weights_col]

            if fit_block_size is not None:
                if block_reduction_col is None:
                    block_reduction_mask = None
                elif block_reduction_col not in df_to_interpolate.columns:
                    msg = (
                        f"`block_reduction_col` '{block_reduction_col}' not in "
                        "`constraints_df` columns"
                    )
                    raise ValueError(msg)
                else:
                    block_reduction_mask = df_to_interpolate[
                        block_reduction_col
                    ].to_numpy(dtype=bool)

                coords, data, weights = _block_reduce_points(
                    coords,
                    data,
                    weights,
                    fit_block_size,
                    block_reduction,
                    fit_region,
                    mask=block_reduction_mask,
                )

            if method == "pygmt":
                # grid the points with tensioned minimum curvature
                grid = (
                    _import_pygmt()
                    .surface(
                        x=coords[0],
                        y=coords[1],
                        z=data,
                        region=region,
                        spacing=spacing,
                        registration=registration,
                        tension=tension,
                        verbose="quiet",
                    )
                    .rename({"x": coord_names[0], "y": coord_names[1]})
                )
            else:
                # run CV for fitting a spline to the data
                spline = _fit_spline_cv(coords, data, weights, dampings)

                # grid the fitted spline at desired spacing and region
                grid = spline.grid(
                    region=region,
                    spacing=spacing,
                    projection=projection,
                    dims=(coord_names[1], coord_names[0]),
                ).scalars

            if subset_points:
                # merge interpolation of inner / buffer points with outside grid
                outside_grid = points_to_grid(
                    x=df_outside_buffer[coord_names[0]],
                    y=df_outside_buffer[coord_names[1]],
                    z=df_outside_buffer.upward,
                    region=region,
                    spacing=spacing,
                    registration=registration if method == "pygmt" else "g",
                    coord_names=coord_names,
                )

                grid = grid.where(
                    outside_grid.isnull(),  # noqa: PD003
                    outside_grid,
                )

        try:
            grid = grid.assign_attrs(damping=spline.damping_)
        except UnboundLocalError:
            grid = grid.assign_attrs(damping=None)

    else:
        msg = "method must be 'flat', 'splines' or 'pygmt'"
        raise ValueError(msg)

    # ensure grid doesn't cross supplied confining layers
    if upper_confining_layer is None or np.all(upper_confining_layer.isnull()):  # noqa: PD003
        pass
    else:
        da = resample_grid(
            upper_confining_layer,
            spacing=spacing,
            region=region,
            registration=registration,
        )
        original_dims = list(da.sizes.keys())
        da = da.rename(
            {original_dims[1]: coord_names[0], original_dims[0]: coord_names[1]}
        )
        grid = xr.where(grid > da, da, grid)

    if lower_confining_layer is None or np.all(lower_confining_layer.isnull()):  # noqa: PD003
        pass
    else:
        da = resample_grid(
            lower_confining_layer,
            spacing=spacing,
            region=region,
            registration=registration,
        )
        original_dims = list(da.sizes.keys())
        da = da.rename(
            {original_dims[1]: coord_names[0], original_dims[0]: coord_names[1]}
        )
        grid = xr.where(grid < da, da, grid)

    ds = grid.to_dataset(name="upward")

    if dataset_to_add is not None:
        # check datasets are aligned
        try:
            _ = xr.align(ds, dataset_to_add, join="exact")
        except xr.AlignmentError as err:
            msg = "`dataset_to_add` should be exactly aligned with the create topography dataset"
            raise xr.AlignmentError(msg) from err
        ds = ds.merge(dataset_to_add)

    return ds


def create_window_regions(
    region: tuple[float, float, float, float],
    window_width: float,
    window_overlap: float = 0.0,
    spacing: float | None = None,
) -> list[tuple[float, float, float, float]]:
    """
    Tile a region with square windows of a set width, optionally overlapping. Window
    placement uses :func:`bordado.line_coordinates` (the same placement as
    :func:`bordado.rolling_window`); the first and last windows in each dimension
    align with the region edges, the entire region is covered, and the actual overlap
    is adjusted (up or down) from the requested value so the windows are evenly
    distributed. If the region is smaller than the window width in a dimension, a
    single window spanning that dimension is used.

    Parameters
    ----------
    region : tuple[float, float, float, float]
        bounding region to tile, in the format (min_easting, max_easting,
        min_northing, max_northing).
    window_width : float
        width of the square windows, in the same units as the region.
    window_overlap : float, optional
        fraction (0 to <1) of the window width which adjacent windows should overlap
        by, by default 0.0
    spacing : float | None, optional
        if provided, window edges are snapped to multiples of this grid spacing, by
        default None

    Returns
    -------
    list[tuple[float, float, float, float]]
        list of window regions in the same format as the input region.
    """
    if not 0 <= window_overlap < 1:
        msg = "window_overlap must be greater or equal to 0 and less than 1"
        raise ValueError(msg)
    if window_width <= 0:
        msg = "window_width must be greater than 0"
        raise ValueError(msg)

    step = window_width * (1 - window_overlap)

    def _starts(minc: float, maxc: float) -> NDArray:
        if maxc - minc <= window_width:
            return np.array([minc])
        centers = bd.line_coordinates(
            minc + window_width / 2,
            maxc - window_width / 2,
            spacing=step,
            adjust="spacing",
        )
        starts = centers - window_width / 2
        if spacing is not None:
            starts = minc + np.round((starts - minc) / spacing) * spacing
        return starts

    windows = []
    for n_start in _starts(region[2], region[3]):
        for e_start in _starts(region[0], region[1]):
            windows.append(  # noqa: PERF401
                (
                    e_start,
                    min(e_start + window_width, region[1]),
                    n_start,
                    min(n_start + window_width, region[3]),
                )
            )
    return windows


def grid_to_model(
    surface: xr.DataArray,
    reference: float | xr.DataArray,
    density: float | xr.DataArray,
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
        raise ValueError(msg)  # noqa: TRY004

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
    density: float | xr.DataArray,  # noqa: ARG001
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

    spline = None
    for n_splits in range(5, 1, -1):
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
            spline = None

    if spline is None:
        msg = "ValueError not resolved, fitting spline with no damping"
        logger.warning(msg)
        # drop kwargs only valid for SplineCV, not Spline
        spline_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("cv", "delayed", "scoring", "client")
        }
        spline = vd.Spline(
            damping=None,
            **spline_kwargs,
        )
        spline.fit(
            coordinates,
            data,
            weights=weights,
        )

    if len(dampings) > 1 and hasattr(spline, "scores_"):
        try:
            logger.info("Best SplineCV score: %s", spline.scores_.max())
        except AttributeError:
            # scores are dask delayed objects
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
                "tested to ensure the best parameter value is found.",
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
        msg = "data contains NaN"
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
            "thus is likely not a global minimum, expand the range of values tested to "
            "ensure the best parameter value is found.",
            dampings[best],
            np.nanmin(dampings_without_none),
            np.nanmax(dampings_without_none),
        )

    # refit with the same kwargs used during scoring so the returned model matches
    # the model which was cross-validated
    return hm.EquivalentSources(damping=dampings[best], **kwargs).fit(
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
        spacing of the prism layer and gravity observation points in meters
    inner_region : tuple[float, float, float, float]
        region boundaries for the region of interest in meters
    top : float
        height in meters for the top of the prisms
    zref : float
        reference level in meters for the prisms
    obs_height : float
        gravity observation height in meters
    density : float
        density value for the prisms in kg/m^3
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
        width of the buffer zone in meters
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
    buffer_region = bd.pad_region(inner_region, buffer_width)

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
            upward=top,
            region=buffer_region,
            spacing=spacing,
        ).upward

    # create prisms around the mean surface value with signed densities (positive
    # above, negative below); this layer has no edge effects and is the baseline for
    # calculating the decay
    mean_zref = surface.to_numpy().mean()
    dens = xr.where(surface >= mean_zref, density, -density)
    model_mean_zref = grid_to_model(
        surface,
        mean_zref,
        density=dens,
        model_type="prisms",
    )

    # create prism layer
    if as_density_contrast:
        model = model_mean_zref
    else:
        model = grid_to_model(
            surface,
            zref,
            density=density,
            model_type="prisms",
        )

    # create set of observation points
    data = bd.grid_coordinates(
        inner_region,
        spacing=spacing,
        non_dimensional_coords=obs_height,
    )

    forward_df = pd.DataFrame(
        {
            "easting": data[0].ravel(),
            "northing": data[1].ravel(),
            "upward": data[2].ravel(),
        }
    )
    # calculate forward gravity of layer
    forward_df["forward"] = model.prism_layer.gravity(
        coordinates=(
            forward_df.easting,
            forward_df.northing,
            forward_df.upward,
        ),
        field="g_z",
        progressbar=progressbar,
    )

    # calculate forward gravity of the layer without edge effects
    forward_df["forward_no_edge_effects"] = model_mean_zref.prism_layer.gravity(
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
        except Exception as e:  # pylint: disable=broad-exception-caught # noqa: BLE001
            logger.error("plotting failed with error: %s", e)

    return max_decay, buffer_width, int(buffer_cells), grav_ds
