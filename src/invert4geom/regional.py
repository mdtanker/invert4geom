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
    grav_data_column: str,
    dc_shift: float | None = None,
    constraints_df: pd.DataFrame | None = None,
    regional_column: str = "reg",
) -> pd.DataFrame:
    """
    separate the regional field by applying a constant shift (DC-shift) to the gravity
    data. If constraint points of the layer of interested are supplied, the DC
    shift will minimize the residual gravity at these constraint points.

    Parameters
    ----------
    grav_df : pd.DataFrame
        gravity data with columns "easting", "northing" and set by grav_data_column.
    grav_data_column: str,
        column name for the gravity data
    dc_shift : float
        shift to apply to the data
    constraints_df : pd.DataFrame
        a dataframe of constraint points with columns easting and northing.
    regional_column : str
        name for the new column in grav_df for the regional field.

    Returns
    -------
    pd.DataFrame
        grav_df with new regional column
    """

    grav_df = grav_df.copy()

    # grid the grav_df data
    grav_grid = grav_df.set_index(["northing", "easting"]).to_xarray()[grav_data_column]

        # get the gravity values at the constraint points
        constraints_df = constraints_df.copy()

        # sample gravity at constraint points
        constraints_df = utils.sample_grids(
            df=constraints_df,
            grid=grav_grid,
            sampled_name="sampled_grav",
            coord_names=coord_names,
        )

        # use RMS of sampled value for DC shift
        dc_shift = utils.rmse(constraints_df.sampled_grav)

    grav_df[regional_column] = dc_shift

    # return the new dataframe
    return grav_df


def regional_filter(
    grav_df: pd.DataFrame,
    grav_data_column: str,
    regional_column: str = "reg",
) -> pd.DataFrame:
    """
    separate the regional field with a low-pass filter

    grav_df = grav_df.copy()

    # grid the grav_df data
    grav_grid = grav_df.set_index(["northing", "easting"]).to_xarray()[grav_data_column]

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
        sampled_name=regional_column,
        coord_names=(original_dims[1], original_dims[0]),
    )


def regional_trend(
    grav_df: pd.DataFrame,
    grav_data_column: str,
    regional_column: str = "reg",
) -> pd.DataFrame:
    """
    separate the regional field with a trend

    grav_df = grav_df.copy()

    vdtrend = vd.Trend(degree=trend).fit(
        (df[original_dims[1]], df[original_dims[0]].values),
        df[grav_filled.name],
    )
    grav_df[regional_column] = vdtrend.predict(
        (grav_df[original_dims[1]], grav_df[original_dims[0]])
    )

    return grav_df


def regional_eq_sources(
    source_depth: float,
    grav_df: pd.DataFrame,
    grav_data_column: str,
    eq_damping: float | None = None,
    block_size: float | None = None,
    depth_type: str = "relative",
    regional_column: str = "reg",
) -> pd.DataFrame:
    """
    separate the regional field by estimating deep equivalent sources

    eq_damping : float: smoothness to impose on estimated coefficients
    block_size : float: block reduce the data to speed up
    depth_type : str: constant depths, not relative to observation heights
    """

    grav_df = grav_df[grav_df[grav_data_column].notna()].copy()

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
    df[regional_column] = equivalent_sources.predict(coordinates)

    return df


def regional_constraints(
    grav_df: pd.DataFrame,
    grav_data_column: str,
    constraints_df: pd.DataFrame,
    tension_factor: float = 1,
    registration: str = "g",
    constraint_block_size: float | None = None,
    grid_method: str = "verde",
    dampings: typing.Any | None = None,
    delayed: bool = False,
    constraint_weights_col: str | None = None,
    eqs_gridding_trials: int = 10,
    eqs_gridding_damping_lims: tuple[float, float] = (0.1, 100),
    eqs_gridding_depth_lims: tuple[float, float] = (1e3, 100e3),
    eqs_gridding_parallel: bool = False,
    eqs_gridding_plot: bool = False,
    force_coords: tuple[pd.Series | NDArray, pd.Series | NDArray] | None = None,
    grav_obs_height: float | None = None,
    regional_column: str = "reg",
) -> pd.DataFrame:
    """
    separate the regional field by sampling and regridding at the constraint points
    grav_df = grav_df.copy()
    constraints_df = constraints_df.copy()
    # grid the grav_df data
    grav_grid = grav_df.set_index(["northing", "easting"]).to_xarray()[grav_data_column]

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
            np.ones_like(constraints_df.easting) * grav_obs_height,
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
            plot=eqs_gridding_plot,
            parallel=eqs_gridding_parallel,
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
        sampled_name=regional_column,
        coord_names=(original_dims[1], original_dims[0]),
        verbose="q",
    )
