# import contextlib
from __future__ import annotations

import copy
import logging

# import itertools
# import os
# import pathlib
# import string
# import sys
# from getpass import getpass
import warnings

import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
import xrft
from antarctic_plots import profile
from pykdtree.kdtree import KDTree  # pylint: disable=no-name-in-module

# from antarctic_plots import utils as ap_utils


# import scipy as sp
# import dask
# import geopandas as gpd

# from requests import get
# from sklearn.metrics import mean_squared_error
# from tqdm.autonotebook import tqdm


def rmse(data: np.array, as_median: bool = False) -> float:
    """
    function to give the root mean/median squared error (RMSE) of data

    Parameters
    ----------
    data : np.array
        input data
    as_median : bool, optional
        choose to give root median squared error instead, by default False

    Returns
    -------
    float
        RMSE value
    """
    if as_median:
        value = np.sqrt(np.nanmedian(data**2).item())
    else:
        value = np.sqrt(np.nanmean(data**2).item())

    return value


def nearest_grid_fill(
    grid: xr.DataArray,
    method: str = "verde",
) -> xr.DataArray:
    """
    fill missing values in a grid with the nearest value.

    Parameters
    ----------
    grid : xr.DataArray
        grid with missing values
    method : str, optional
        choose method of filling, by default "verde"

    Returns
    -------
    xr.DataArray
        filled grid
    """

    # get coordinate names
    original_dims = list(grid.sizes.keys())

    # get original grid name
    original_name = grid.name

    if method == "rioxarray":
        filled = (
            grid.rio.write_crs("epsg:3031")
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
    # change_spacing:bool=False,
):
    """
    _summary_

    Parameters
    ----------
    grid : xr.DataArray
        _description_
    filter_width : float, optional
        _description_, by default None
    filt_type : str, optional
        _description_, by default "lowpass"
    change_spacing : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_

    """
    # get coordinate names
    original_dims = list(grid.sizes.keys())

    # get original grid name
    original_name = grid.name

    # if there are nan's, fill them with nearest neighbor
    if grid.isnull().any():
        filled = nearest_grid_fill(grid, method="verde")
        # print("filling NaN's with nearest neighbor")
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

    # if change_spacing is True:
    #     region = ap_utils.get_grid_info(grid)[1]
    #     grid = fetch.resample_grid(
    #         grid,
    #         spacing=filter_width,
    #         verbose="q",
    #         region = region,
    #     )
    #     unpadded = fetch.resample_grid(
    #         unpadded,
    #         spacing=filter_width,
    #         verbose="q",
    #         region = region,
    #     )
    # else:
    #     pass
    if grid.isnull().any():
        result = xr.where(grid.notnull(), unpadded, grid)
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
    coord_names: list | None = None,
) -> pd.DataFrame | xr.DataArray | xr.Dataset:
    """
    for all gridcells calculate to the distance to the nearest target.
    """

    if coord_names is None:
        coord_names = ["easting", "northing"]
    df_targets = targets[[coord_names[0], coord_names[1]]].copy()

    if isinstance(data, pd.DataFrame):
        df_data = data[coord_names].copy()
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


def normalize_xarray(da, low=0, high=1):
    # min_val = da.values.min()
    # max_val = da.values.max()

    da = da.copy()

    min_val = da.quantile(0)
    max_val = da.quantile(1)

    da2 = (high - low) * (((da - min_val) / (max_val - min_val)).clip(0, 1)) + low

    return da2.drop("quantile")


def normalized_mindist(
    points: pd.DataFrame,
    grid: xr.DataArray,  # Union[xr.DataArray, xr.Dataset],
    low: float | None = None,
    high: float | None = None,
    mindist: float | None = None,
    region: list | None = None,
):
    """
    Find the minimum distance between each grid cell and the nearest point. If low and
    high are provided, normalize the min dists grid between these values. If region is
    provided, all grid cells outside region are set to a distance of 0.
    """
    grid = copy.deepcopy(grid)

    # if a dataset supplied, use first variable as a dataarray
    # if isinstance(grid, xr.Dataset):
    #     grid = grid[list(grid.variables.keys())[0]]

    # get coordinate names
    original_dims = list(grid.sizes.keys())

    constraint_points = points.copy()

    min_dist = dist_nearest_points(
        targets=constraint_points,
        data=grid,
        coord_names=(original_dims[1], original_dims[0]),
    ).min_dist

    # set points < mindist to 0
    if mindist is not None:
        min_dist = xr.where(min_dist < mindist, 0, min_dist)

    # set points outside of region to 0
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
        min_dist = normalize_xarray(min_dist, low=low, high=high)

    return min_dist


def sample_grids(
    df: pd.DataFrame,
    grid: str | xr.DataArray,
    name: str,
    **kwargs,
):
    """
    Sample data at every point along a line

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing columns 'x', 'y', or columns with names defined by kwarg
        "coor_names".
    grid : str or xr.DataArray
        Grid to sample, either file name or xr.DataArray
    name : str,
        Name for sampled column

    Returns
    -------
    pd.DataFrame
        Dataframe with new column (name) of sample values from (grid)
    """

    # drop name column if it already exists
    try:
        df1 = df.drop(columns=name)
    except KeyError:
        df1 = df.copy()

    df2 = df1.copy()

    # reset the index
    df3 = df2.reset_index()

    x, y = kwargs.get("coord_names", ("x", "y"))
    # get points to sample at
    points = df3[[x, y]].copy()

    # sample the grid at all x,y points
    sampled = pygmt.grdtrack(
        points=points,
        grid=grid,
        newcolname=name,
        # radius=kwargs.get("radius", None),
        no_skip=True,  # if false causes issues
        verbose=kwargs.get("verbose", "w"),
        interpolation=kwargs.get("interpolation", "c"),
    )

    df3[name] = sampled[name]

    # reset index to previous
    df4 = df3.set_index("index")

    # reset index name to be same as originals
    df4.index.name = df1.index.name

    # check that dataframe is identical to original except for new column
    pd.testing.assert_frame_equal(df4.drop(columns=name), df1)

    return df4


def extract_prism_data(
    prism_layer: xr.Dataset,
) -> tuple[
    pd.DataFrame,
    xr.Dataset,
    float,
    float,
    float,
    xr.DataArray,
]:
    """
    extract necessary info from starting prism layer, adds variables 'topo' and
    'starting_topo' to prism layer dataset (prisms_ds), converts it into dataframe
    (prisms_df), gets the density contrast value (density) from the max density value in
     prisms_df, gets the reference level (zref) from the min value of the prims tops,
     gets the prism spacing (spacing) from prisms_ds, and creates a grid of the starting
     topography (topo_grid) from the tops and bottoms of the prism layer.

    Parameters
    ----------
    prism_layer : xr.Dataset
       starting model prism layer

    Returns
    -------
    tuple
        prisms_df, prisms_ds, density_contrast, zref, spacing, topo_grid)
    """

    prisms_ds = copy.deepcopy(prism_layer.load())

    # check that minimum elevation of prism tops is equal to max elevation of prism
    # bottoms
    # if not prisms_ds.top.to_numpy().min() == prisms_ds.bottom.to_numpy().max():
    #     msg = "reference for prism layer is outside limits of tops and bottoms"
    #     raise ValueError(msg)

    # check prisms above reference have densities of opposite sign to prisms below
    # try:
    #     if not prisms_ds.density.to_numpy().max() == -prisms_ds.density.to_numpy().min(): # noqa: E501
    #         msg = "densities should be represented as contrasts not absolutes."
    #         raise ValueError(msg)
    # # if not, they should at least be equal (if starting topo model is flat)
    # except:
    #     if not prisms_ds.density.to_numpy().max() == prisms_ds.density.to_numpy().min(): # noqa: E501
    #         msg = "densities should be represented as contrasts not absolutes."
    #         raise ValueError(msg)

    density_contrast = prisms_ds.density.to_numpy().max()
    zref = prisms_ds.top.to_numpy().min()

    # add starting topo to dataset
    topo_grid = xr.where(prisms_ds.density > 0, prisms_ds.top, prisms_ds.bottom)
    prisms_ds["topo"] = topo_grid
    prisms_ds["starting_topo"] = topo_grid

    # turn dataset into dataframe
    prisms_df = prisms_ds.to_dataframe().reset_index().dropna().astype(float)

    spacing = get_spacing(prisms_df)

    return prisms_df, prisms_ds, density_contrast, zref, spacing, topo_grid


def get_spacing(prisms_df: pd.DataFrame) -> float:
    """
    Extract spacing of harmonica prism layer using a dataframe representation.

    Parameters
    ----------
    prisms_df : pd.DataFrame
        dataframe of harmonica prism layer

    Returns
    -------
    float
        spacing of prisms
    """
    return abs(prisms_df.northing.unique()[1] - prisms_df.northing.unique()[0])


def sample_bounding_surfaces(
    prisms_df: pd.DataFrame,
    upper_confining_layer: xr.DataArray | None = None,
    lower_confining_layer: xr.DataArray | None = None,
) -> pd.DataFrame:
    """
    sample upper and/or lower confining layers into prisms dataframe

    Parameters
    ----------
    prisms_df : pd.DataFrame
        dataframe of prism properties
    upper_confining_layer : xr.DataArray | None, optional
        layer which the inverted topography should always be below, by default None
    lower_confining_layer : xr.DataArray | None, optional
        layer which the inverted topography should always be above, by default None

    Returns
    -------
    pd.DataFrame
        a dataframe with added columns 'upper_bounds' and 'lower_bounds', which are the
        sampled values of the supplied confining grids.
    """
    df = prisms_df.copy()

    if upper_confining_layer is not None:
        df = profile.sample_grids(
            df=df,
            grid=upper_confining_layer,
            name="upper_bounds",
            coord_names=["easting", "northing"],
        )
        assert len(df.upper_bounds) != 0
    if lower_confining_layer is not None:
        df = profile.sample_grids(
            df=df,
            grid=lower_confining_layer,
            name="lower_bounds",
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
    prisms_df : pd.DataFrame
        prism layer dataframe with optional 'upper_bounds' or 'lower_bounds' columns,
        and current iteration's topography.
    iteration_number : int
        number of the current iteration

    Returns
    -------
    pd.DataFrame
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
        logging.info("enforced upper confining surface at %s prisms", number_enforced)
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
        logging.info("enforced lower confining surface at %s prisms", number_enforced)

    # check that when constrained correction is added to topo it doesn't intersect
    # either bounding layer
    updated_topo = df[f"iter_{iteration_number}_correction"] + df.topo
    if "upper_bounds" in df and np.any((df.upper_bounds - updated_topo) < 0):
        msg = (
            "Constraining didn't work and updated topography intersects upper "
            "constraining surface"
        )
        raise ValueError(msg)
    if "lower_bounds" in df and np.any((updated_topo - df.lower_bounds) < 0):
        msg = (
            "Constraining didn't work and updated topography intersects lower "
            "constraining surface"
        )
        raise ValueError(msg)
    return df


def apply_surface_correction(
    prisms_df: pd.DataFrame,
    iteration_number: int,
) -> tuple:
    """
    update the prisms dataframe and dataset with the surface correction. Ensure that
    the updated surface doesn't intersect the optional confining surfaces.
    """
    df = prisms_df.copy()

    # for negative densities, negate the correction
    df.loc[df.density < 0, f"iter_{iteration_number}_correction"] *= -1

    # grid the corrections
    # correction_grid_before = (
    #     df.rename(columns={f"iter_{iteration_number}_correction": "z"})
    #     .set_index(["northing", "easting"])
    #     .to_xarray()
    #     .z
    # )

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
    zref: float,
):
    """
    apply the corrections grid and update the prism tops, bottoms, topo, and
    densities.
    """
    ds = prisms_ds.copy()

    density_contrast = ds.density.values.max()

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
):
    """
    update the prisms dataframe the the new prism tops, bottoms, topo, and densities
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
