from __future__ import annotations

import logging
import typing

import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import verde as vd
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

    if (constraints_df is None) and (dc_shift is None):
        msg = "need to provide either constraints_df of dc_shift"
        raise ValueError(msg)

    if constraints_df is not None:
        if dc_shift is not None:
            msg = (
                "`dc_shift` parameter provide but not used since `constraints_df`"
                "were provided."
            )
            logging.warning(msg)
        # get the gravity values at the constraint points
        constraints_df = constraints_df.copy()

        # sample gravity at constraint points
        constraints_df = utils.sample_grids(
            df=constraints_df,
            grid=grav_grid,
            sampled_name="sampled_grav",
            coord_names=("easting", "northing"),
        )

        # use median of sampled value for DC shift
        dc_shift = np.nanmedian(constraints_df.sampled_grav)

    grav_df[regional_column] = dc_shift

    # return the new dataframe
    return grav_df


def regional_filter(
    grav_df: pd.DataFrame,
    grav_data_column: str,
    filter_width: float,
    regional_column: str = "reg",
) -> pd.DataFrame:
    """
    separate the regional field with a low-pass filter

    Parameters
    ----------
    grav_df : pd.DataFrame
        gravity data with columns "easting", "northing" and set by grav_data_column.
    grav_data_column: str,
        column name for the gravity data
    filter_width : float
        width in meters to use for the low-pass filter
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
    trend: int,
    regional_column: str = "reg",
) -> pd.DataFrame:
    """
    separate the regional field with a trend

    Parameters
    ----------
    grav_df : pd.DataFrame
        gravity data with columns "easting", "northing" and set by grav_data_column.
    grav_data_column: str,
        column name for the gravity data
    trend : int
        order of the polynomial trend to fit to the data
    regional_column : str
        name for the new column in grav_df for the regional field.

    Returns
    -------
    pd.DataFrame
        grav_df with new regional column
    """

    grav_df = grav_df.copy()

    vdtrend = vd.Trend(degree=trend).fit(
        (grav_df.easting, grav_df.northing),
        grav_df[grav_data_column],
    )
    grav_df[regional_column] = vdtrend.predict(
        (grav_df.easting, grav_df.northing),
    )

    return grav_df


def regional_eq_sources(
    grav_df: pd.DataFrame,
    grav_data_column: str,
    source_depth: float,
    eq_damping: float | None = None,
    block_size: float | None = None,
    regional_column: str = "reg",
) -> pd.DataFrame:
    """
    separate the regional field by estimating deep equivalent sources

    Parameters
    ----------
    grav_df : pd.DataFrame
        gravity data with columns "easting", "northing", "upward" and set by
        grav_data_column.
    grav_data_column: str,
        column name for the gravity data
    source_depth : float
        depth of each source relative to the data elevation
    eq_damping : float | None, optional
        smoothness to impose on estimated coefficients, by default None
    block_size : float | None, optional
        block reduce the data to speed up, by default None
    regional_column : str
        name for the new column in grav_df for the regional field.

    Returns
    -------
    pd.DataFrame
        grav_df with new regional column
    """

    grav_df = grav_df[grav_df[grav_data_column].notna()].copy()

    # create set of deep sources
    equivalent_sources = hm.EquivalentSources(
        depth=source_depth,
        damping=eq_damping,
        block_size=block_size,
        # depth_type="relative",
    )

    # fit the source coefficients to the data
    coordinates = (grav_df.easting, grav_df.northing, grav_df.upward)
    equivalent_sources.fit(coordinates, grav_df[grav_data_column])

    # use sources to predict the regional field at the observation points
    grav_df[regional_column] = equivalent_sources.predict(coordinates)

    return grav_df


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

    Parameters
    ----------
    grav_df : pd.DataFrame
        gravity data with coordinate columns "easting" and "northing" and gravity data
        column set by grav_data_column.
    grav_data_column: str,
        column name for the gravity data
    constraints_df : pd.DataFrame
        dataframe of constraints with columns "easting", "northing", and "upward".
    tension_factor : float, optional
        Tension factor used if `grid_method` is "pygmt", by default 1
    registration : str, optional
       grid registration used if `grid_method` is "pygmt",, by default "g"
    constraint_block_size : float | None, optional
        size of block used in a block-mean reduction of the constraints points, by
        default None
    grid_method : str, optional
        method used to grid the sampled gravity data at the constraint points. Choose
        between "verde", "pygmt", or "eq_sources", by default "verde"
    dampings : typing.Any | None, optional
        damping values used if `grid_method` is "verde", by default None
    delayed : bool, optional
        whether to parallelize the gridding if `grid_method` is "verde", by default
        False
    constraint_weights_col : str | None, optional
       column name for weighting values of each constraint point. Used if
       `constraint_block_size` is not None or if `grid_method` is "verde", by default
       None
    eqs_gridding_trials : int, optional
        Number of trials to be performed if `grid_method` is "eq_sources", by default 10
    eqs_gridding_damping_lims : tuple[float, float], optional
        Damping limits to be used if `grid_method` is "eq_sources", by default
        (0.1, 100)
    eqs_gridding_depth_lims : tuple[float, float], optional
       Depth limits to be used if `grid_method` is "eq_sources", by default (1e3, 100e3)
    grav_obs_height : float, optional
        Observation height to use if `grid_method` is "eq_sources", by default None
    force_coords : tuple[pd.Series  |  NDArray, pd.Series  |  NDArray] | None, optional
        Optionally forced coordinates to use if `grid_method` is "eq_sources", by
        default None
    regional_column : str
        name for the new column in grav_df for the regional field, by default "reg"

    Returns
    -------
    pd.DataFrame
        grav_df with new regional column
    """

    if constraints_df is None:
        msg = "need to provide constraints_df"
        raise ValueError(msg)

    grav_df = grav_df.copy()
    constraints_df = constraints_df.copy()

    region = vd.get_region((grav_df.easting, grav_df.northing))
    spacing = utils.get_spacing(grav_df)

    # grid the grav_df data
    grav_grid = grav_df.set_index(["northing", "easting"]).to_xarray()[grav_data_column]

    # sample gravity at constraint points
    constraints_df = utils.sample_grids(
        df=constraints_df,
        grid=grav_grid,
        sampled_name="sampled_grav",
        coord_names=("easting", "northing"),
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
                constraints_df["easting"],
                constraints_df["northing"],
            ),
            data=constraints_df.sampled_grav,
            weights=weights,
        )
        # add reduced coordinates to a dictionary
        coord_cols = dict(zip(["easting", "northing"], coordinates))

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
            data=constraints_df[["easting", "northing", "sampled_grav"]],
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
                constraints_df.easting,
                constraints_df.northing,
            ),
            data=constraints_df.sampled_grav,
            weights=weights,
            dampings=dampings,
            delayed=delayed,
            force_coords=force_coords,
        )

        regional_grav = spline.grid(region=region, spacing=spacing).scalars

    elif grid_method == "eq_sources":
        coords = (
            constraints_df.easting,
            constraints_df.northing,
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
        coord_names=("easting", "northing"),
        verbose="q",
    )


def regional_separation(
    method: str,
    grav_df: pd.DataFrame,
    grav_data_column: str,
    regional_column: str = "reg",
    **kwargs: typing.Any,
) -> pd.DataFrame:
    """
    Separate the regional field from the gravity data using the specified method
    and return the dataframe with a new column for the regional field.

    Parameters
    ----------
    method : str
        choose method to apply; one of "constant", "dc_shift", "filter", "trend",
        "eq_sources", "constraints".
    grav_df : pd.DataFrame
        gravity data with columns "easting", "northing" and set by grav_data_column.
    grav_data_column: str,
        column name for the gravity data
    regional_column : str, optional
        name to use for new regional gravity column, by default "reg"
    **kwargs : typing.Any
        additional keyword arguments for the specified method.

    Returns
    -------
    pd.DataFrame
        updated dataframe with new regional gravity column
    """
    grav_df = grav_df.copy()

    kwargs = kwargs.copy()

    if method == "constant":
        constant = kwargs.get("constant", None)
        if constant is None:
            msg = "constant value not provided"
            raise ValueError(msg)

        grav_df[regional_column] = constant
        return grav_df
    if method == "dc_shift":
        return regional_dc_shift(
            grav_df=grav_df,
            grav_data_column=grav_data_column,
            regional_column=regional_column,
            **kwargs,
        )
    if method == "filter":
        return regional_filter(
            grav_df=grav_df,
            grav_data_column=grav_data_column,
            regional_column=regional_column,
            **kwargs,
        )
    if method == "trend":
        return regional_trend(
            grav_df=grav_df,
            grav_data_column=grav_data_column,
            regional_column=regional_column,
            **kwargs,
        )
    if method == "eq_sources":
        return regional_eq_sources(
            grav_df=grav_df,
            grav_data_column=grav_data_column,
            regional_column=regional_column,
            **kwargs,
        )
    if method == "constraints":
        return regional_constraints(
            grav_data_column=grav_data_column,
            grav_df=grav_df,
            regional_column=regional_column,
            **kwargs,
        )
    msg = "invalid string for regional method"
    raise ValueError(msg)
