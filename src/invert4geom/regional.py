from __future__ import annotations

import typing

import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
from nptyping import NDArray

from invert4geom import optimization, utils


def regional_dc_shift(
    grav_df: pd.DataFrame,
    dc_shift: float | None = None,
    grav_grid: xr.DataArray | None = None,
    constraint_points: pd.DataFrame | None = None,
    coord_names: tuple[str, str] = ("easting", "northing"),
    regional_col_name: str = "reg",
) -> pd.DataFrame:
    """
    separate the regional field by applying a constant shift (DC-shift) to the gravity
    data. If constraint points of the layer of interested are supplied, the DC shift
    will minimize the residual misfit at these constraint points.

    Parameters
    ----------
    grav_df : pd.DataFrame
        gravity data with columns defined by coord_names and input_grav_name.
    dc_shift : float
        shift to apply to the data
    grav_grid : xr.DataArray
        gridded gravity misfit data
    constraint_points : pd.DataFrame
        a dataframe of constraint points with columns X and Y columns defined by the
        coord_names parameter.
    coord_names : tuple
        names of the X and Y column names in constraint points dataframe
    regional_col_name : str
        name for the new column in grav_df for the regional field.

    Returns
    -------
    pd.DataFrame
        grav_df with new regional column
    """
    if constraint_points is not None:
        # get the gravity values at the constraint points
        constraints_df = constraint_points.copy()

        # sample gravity at constraint points
        constraints_df = utils.sample_grids(
            df=constraints_df,
            grid=grav_grid,
            sampled_name="sampled_grav",
            coord_names=coord_names,
        )

        # use RMS of sampled value for DC shift
        dc_shift = utils.rmse(constraints_df.sampled_grav)

    grav_df[regional_col_name] = dc_shift

    # return the new dataframe
    return grav_df


def regional_filter(
    filter_width: float,
    grav_grid: xr.DataArray,
    grav_df: pd.DataFrame,
    regional_col_name: str = "reg",
) -> pd.DataFrame:
    """
    separate the regional field with a low-pass filter
    """
    # get coordinate names
    original_dims = grav_grid.dims

    # filter the gravity grid with the provided filter in meters
    regional_grid = utils.filter_grid(
        grav_grid,
        filter_width,
        filt_type="lowpass",
    )

    return utils.sample_grids(
        grav_df,
        regional_grid,
        sampled_name=regional_col_name,
        coord_names=(original_dims[1], original_dims[0]),
    )


def regional_trend(
    trend: int,
    grav_grid: xr.DataArray,
    grav_df: pd.DataFrame,
    fill_method: str = "verde",
    regional_col_name: str = "reg",
) -> pd.DataFrame:
    """
    separate the regional field with a trend
    """
    # get coordinate names
    original_dims = grav_grid.dims

    grav_filled = utils.nearest_grid_fill(grav_grid, method=fill_method)

    df = vd.grid_to_table(grav_filled).astype("float64")
    vdtrend = vd.Trend(degree=trend).fit(
        (df[original_dims[1]], df[original_dims[0]].values),
        df[grav_filled.name],
    )
    grav_df[regional_col_name] = vdtrend.predict(
        (grav_df[original_dims[1]], grav_df[original_dims[0]])
    )

    return grav_df


def regional_eq_sources(
    source_depth: float,
    grav_df: pd.DataFrame,
    input_grav_name: str,
    eq_damping: float | None = None,
    block_size: float | None = None,
    depth_type: str = "relative",
    input_coord_names: tuple[str, str] = ("easting", "northing"),
    regional_col_name: str = "reg",
) -> pd.DataFrame:
    """
    separate the regional field by estimating deep equivalent sources

    eq_damping : float: smoothness to impose on estimated coefficients
    block_size : float: block reduce the data to speed up
    depth_type : str: constant depths, not relative to observation heights
    """

    df = grav_df[grav_df[input_grav_name].notna()]

    # create set of deep sources
    equivalent_sources = hm.EquivalentSources(
        depth=source_depth,
        damping=eq_damping,
        block_size=block_size,
        depth_type=depth_type,
    )

    # fit the source coefficients to the data
    coordinates = (df[input_coord_names[0]], df[input_coord_names[1]], df.upward)
    equivalent_sources.fit(coordinates, df[input_grav_name])

    # use sources to predict the regional field at the observation points
    df[regional_col_name] = equivalent_sources.predict(coordinates)

    return df


def regional_constraints(
    constraint_points: pd.DataFrame,
    grav_grid: xr.DataArray,
    grav_df: pd.DataFrame,
    region: tuple[float, float, float, float],
    spacing: float,
    tension_factor: float = 1,
    registration: str = "g",
    constraint_block_size: float | None = None,
    grid_method: str = "pygmt",
    dampings: typing.Any | None = None,
    delayed: bool = False,
    constraint_weights_col: str | None = None,
    eqs_gridding_trials: int = 10,
    eqs_gridding_damping_lims: tuple[float, float] = (0.1, 100),
    eqs_gridding_depth_lims: tuple[float, float] = (1e3, 100e3),
    force_coords: tuple[pd.Series | NDArray, pd.Series | NDArray] | None = None,
    regional_col_name: str = "reg",
) -> pd.DataFrame:
    """
    separate the regional field by sampling and regridding at the constraint points
    """
    # get coordinate names
    original_dims = grav_grid.dims

    constraints_df = constraint_points.copy()

    # sample gravity at constraint points
    constraints_df = utils.sample_grids(
        df=constraints_df,
        grid=grav_grid,
        sampled_name="sampled_grav",
        coord_names=(original_dims[1], original_dims[0]),
        no_skip=True,
        verbose="q",
    )

    constraints_df = constraints_df[constraints_df.sampled_grav.notna()]

    if constraint_block_size is not None:
        # get weighted mean gravity value of constraint points in each cell
        if constraint_weights_col is None:
            weights = None
            uncertainty = False
        else:
            weights = constraints_df[constraint_weights_col]
            uncertainty = True

        blockmean = vd.BlockMean(
            spacing=constraint_block_size,
            uncertainty=uncertainty,
        )

        coordinates, data, weights = blockmean.filter(
            coordinates=(
                constraints_df[original_dims[1]],
                constraints_df[original_dims[0]],
            ),
            data=constraints_df.sampled_grav,
            weights=weights,
        )
        # add reduced coordinates to a dictionary
        coord_cols = dict(zip([original_dims[1], original_dims[0]], coordinates))

        # add reduced data to a dictionary
        if constraint_weights_col is None:
            data_cols = {"sampled_grav": data}
        else:
            data_cols = {"sampled_grav": data, constraint_weights_col: weights}
        # merge dicts and create dataframe
        constraints_df = pd.DataFrame(data=coord_cols | data_cols)

    # grid the entire regional gravity based just on the values at the constraints
    if grid_method == "pygmt":
        regional_grav = pygmt.surface(
            data=constraints_df[[original_dims[1], original_dims[0], "sampled_grav"]],
            region=region,
            spacing=spacing,
            registration=registration,
            tension=tension_factor,
            verbose="q",
        )

    elif grid_method == "verde":
        if dampings is None:
            dampings = list(np.logspace(-10, -2, num=9))
            dampings.append(None)

        if constraint_weights_col is None:
            weights = None
        else:
            weights = constraints_df[constraint_weights_col]

        spline = utils.best_spline_cv(
            coordinates=(
                constraints_df[original_dims[1]],
                constraints_df[original_dims[0]],
            ),
            data=constraints_df.sampled_grav,
            weights=weights,
            dampings=dampings,
            delayed=delayed,
            force_coords=force_coords,
        )

        regional_grav = spline.grid(region=region, spacing=spacing).scalars

    elif grid_method == "eq_sources":
        # pass
        coords = (
            constraints_df[original_dims[1]],
            constraints_df[original_dims[0]],
            np.ones_like(constraints_df[original_dims[1]]) * 1e3,  # grav obs height
        )
        if constraint_weights_col is None:
            weights = None
        else:
            weights = constraints_df[constraint_weights_col]

        # eqs = hm.EquivalentSources(depth=100e3, damping=1e2)
        # eqs.fit(coords, constraints_df.sampled_grav, weights=weights,)

        _study_df, eqs = optimization.optimize_eq_source_params(
            coords,
            constraints_df.sampled_grav,
            n_trials=eqs_gridding_trials,
            damping_limits=eqs_gridding_damping_lims,
            depth_limits=eqs_gridding_depth_lims,
            plot=False,
            parallel=True,
            weights=weights,
        )

        # Define grid coordinates
        grid_coords = vd.grid_coordinates(
            region=region,
            spacing=spacing,
            extra_coords=coords[2].max(),
        )
        # predict sources onto grid to get regional
        regional_grav = eqs.grid(grid_coords, data_names="pred").pred

    else:
        msg = "invalid string for grid_method"
        raise ValueError(msg)

    # sample the resulting grid and add to grav_df dataframe
    return utils.sample_grids(
        df=grav_df,
        grid=regional_grav,
        sampled_name=regional_col_name,
        coord_names=(original_dims[1], original_dims[0]),
        verbose="q",
    )
