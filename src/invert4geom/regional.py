from __future__ import annotations

import typing

import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import verde as vd
from nptyping import NDArray

from invert4geom import cross_validation, log, optimization, utils


def log_filter(record: typing.Any) -> bool:  # noqa: ARG001 # pylint: disable=unused-argument
    """Used to filter logging."""
    return False


def _check_grav_cols(grav_df: pd.DataFrame) -> None:
    """
    ensure gravity dataframe has the necessary columns

    Parameters
    ----------
    grav_df : pd.DataFrame
        gravity dataframe
    """
    cols = [
        "gravity_anomaly",
        "starting_gravity",
    ]
    if all(i in grav_df.columns for i in cols) is False:
        msg = f"`grav_df` needs all the following columns: {cols}"
        raise ValueError(msg)


def regional_constant(
    grav_df: pd.DataFrame,
    constant: float | None = None,
    constraints_df: pd.DataFrame | None = None,
    regional_shift: float = 0,
) -> pd.DataFrame:
    """
    approximate the regional field with a constant value. If constraint points of the
    layer of interested are supplied, the constant value will be the median misfit value
    at the constraint points.

    Parameters
    ----------
    grav_df : pd.DataFrame
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "starting_gravity".
    constant : float
        shift to apply to the data
    constraints_df : pd.DataFrame
        a dataframe of constraint points with columns easting and northing.
    regional_shift : float, optional
        shift to add to the regional field, by default 0

    Returns
    -------
    pd.DataFrame
        grav_df with new columns 'misfit', 'reg', and 'res'.
    """

    grav_df = grav_df.copy()

    _check_grav_cols(grav_df)
    # Gobs_shift = f"{gravity_anomaly}_shift"

    # # add optional dc shift
    # if regional_shift is not None:
    #     grav_df[Gobs_shift] = grav_df[gravity_anomaly] + regional_shift
    # else:
    #     grav_df[Gobs_shift] = grav_df[gravity_anomaly]

    # grav_df["misfit"] = grav_df[Gobs_shift] - grav_df[starting_gravity]

    grav_df["misfit"] = grav_df.gravity_anomaly - grav_df.starting_gravity

    # grid the grav_df data
    grav_grid = grav_df.set_index(["northing", "easting"]).to_xarray().misfit

    if (constraints_df is None) and (constant is None):
        msg = "need to provide either `constraints_df` of `constant`"
        raise ValueError(msg)

    if constraints_df is not None:
        if constant is not None:
            msg = (
                "`constant` parameter provide but not used since `constraints_df`"
                "were provided."
            )
            log.warning(msg)
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
        constant = np.nanmedian(constraints_df.sampled_grav)

        msg = (
            "using median gravity misfit of constraint points for regional field: "
            f"{constant} mGal"
        )
        log.info(msg)

    grav_df["reg"] = constant + regional_shift  # type: ignore[operator]

    grav_df["res"] = grav_df.misfit - grav_df.reg

    # return the new dataframe
    return grav_df


def regional_filter(
    grav_df: pd.DataFrame,
    filter_width: float,
    regional_shift: float = 0,
) -> pd.DataFrame:
    """
    separate the regional field with a low-pass filter

    Parameters
    ----------
    grav_df : pd.DataFrame
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "starting_gravity".
    filter_width : float
        width in meters to use for the low-pass filter
    regional_shift : float, optional
        shift to add to the regional field, by default 0

    Returns
    -------
    pd.DataFrame
        grav_df with new columns 'misfit', 'reg', and 'res'.
    """

    grav_df = grav_df.copy()
    _check_grav_cols(grav_df)

    # grav_df["misfit"] = grav_df[Gobs_shift] - grav_df[starting_gravity]
    grav_df["misfit"] = grav_df.gravity_anomaly - grav_df.starting_gravity

    # grid the grav_df data
    grav_grid = grav_df.set_index(["northing", "easting"]).to_xarray().misfit

    # remove the mean from the data
    data_mean = grav_grid.mean()
    grav_grid -= data_mean

    # get coordinate names
    original_dims = grav_grid.dims

    # filter the gravity grid with the provided filter in meters
    regional_grid = utils.filter_grid(
        grav_grid,
        filter_width,
        filt_type="lowpass",
    )

    # add the mean back to the data
    regional_grid += data_mean

    regional_grid += regional_shift

    grav_df = utils.sample_grids(
        grav_df,
        regional_grid,
        sampled_name="reg",
        coord_names=(original_dims[1], original_dims[0]),
    )
    grav_df["res"] = grav_df.misfit - grav_df.reg

    return grav_df


def regional_trend(
    grav_df: pd.DataFrame,
    trend: int,
    regional_shift: float = 0,
) -> pd.DataFrame:
    """
    separate the regional field with a trend

    Parameters
    ----------
    grav_df : pd.DataFrame
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "starting_gravity".
    trend : int
        order of the polynomial trend to fit to the data
    regional_shift : float, optional
        shift to add to the regional field, by default 0

    Returns
    -------
    pd.DataFrame
        grav_df with new columns 'misfit', 'reg', and 'res'.
    """

    grav_df = grav_df.copy()
    _check_grav_cols(grav_df)

    grav_df["misfit"] = grav_df.gravity_anomaly - grav_df.starting_gravity

    vdtrend = vd.Trend(degree=trend).fit(
        (grav_df.easting, grav_df.northing),
        grav_df.misfit,
    )
    grav_df["reg"] = (
        vdtrend.predict(
            (grav_df.easting, grav_df.northing),
        )
        + regional_shift
    )

    grav_df["res"] = grav_df.misfit - grav_df.reg

    return grav_df


def regional_eq_sources(
    grav_df: pd.DataFrame,
    source_depth: float | str = "default",
    eq_damping: float | None = None,
    block_size: float | None = None,
    grav_obs_height: float | None = None,
    regional_shift: float = 0,
    eq_cv: bool = False,
    weights_column: str | None = None,
) -> pd.DataFrame:
    """
    separate the regional field by estimating deep equivalent sources

    Parameters
    ----------
    grav_df : pd.DataFrame
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "starting_gravity".
    source_depth : float
        depth of each source relative to the data elevation
    eq_damping : float | None, optional
        smoothness to impose on estimated coefficients, by default None
    block_size : float | None, optional
        block reduce the data to speed up, by default None
    grav_obs_height: float, optional
        Observation height to use predicting the eq sources, by default None and will
        use the data height from grav_df.
    regional_shift : float, optional
        shift to add to the regional field, by default 0
    eq_cv : bool, optional
        use cross-validation to find the best equivalent source parameters, by default
        False
    weights_column: str | None, optional
        column name for weighting values of each gravity point.
    Returns
    -------
    pd.DataFrame
        grav_df with new columns 'misfit', 'reg', and 'res'.
    """

    grav_df = grav_df.copy()
    _check_grav_cols(grav_df)

    grav_df["misfit"] = grav_df.gravity_anomaly - grav_df.starting_gravity

    coords = (grav_df.easting, grav_df.northing, grav_df.upward)

    weights = None if weights_column is None else grav_df[weights_column]

    if eq_cv is True:
        _, eqs = optimization.optimize_eq_source_params(
            coordinates=coords,
            data=grav_df.misfit,
            weights=weights,
            progressbar=True,
            n_trials=10,
            eq_damping_limits=(1e-3, 1e3),
        )
    else:
        # create set of deep sources
        eqs = hm.EquivalentSources(
            depth=source_depth,
            damping=eq_damping,
            block_size=block_size,
        )

        # fit the source coefficients to the data
        eqs.fit(
            coords,
            grav_df.misfit,
            weights=weights,
        )

    # use sources to predict the regional field at the observation points
    # set observation height
    if grav_obs_height is None:
        upward_continuation_height = grav_df.upward
    else:
        upward_continuation_height = np.ones_like(grav_df.upward) * grav_obs_height

    coords = (grav_df.easting, grav_df.northing, upward_continuation_height)

    grav_df["res"] = grav_df.misfit - grav_df.reg

    return grav_df


def regional_constraints(
    grav_df: pd.DataFrame,
    constraints_df: pd.DataFrame,
    tension_factor: float = 1,
    registration: str = "g",
    constraints_block_size: float | None = None,
    grid_method: str = "verde",
    spline_damping: float | None = None,
    source_depth: float | None = None,
    eq_damping: float | None = None,
    block_size: float | None = None,
    eq_points: list[NDArray] | None = None,
    constraints_weights_column: str | None = None,
    grav_obs_height: float | None = None,
    regional_shift: float = 0,
) -> pd.DataFrame:
    """
    separate the regional field by sampling and regridding at the constraint points

    Parameters
    ----------
    grav_df : pd.DataFrame
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "starting_gravity".
    constraints_df : pd.DataFrame
        dataframe of constraints with columns "easting", "northing", and "upward".
    tension_factor : float, optional
        Tension factor used if `grid_method` is "pygmt", by default 1
    registration : str, optional
       grid registration used if `grid_method` is "pygmt",, by default "g"
    constraints_block_size : float | None, optional
        size of block used in a block-mean reduction of the constraints points, by
        default None
    grid_method : str, optional
        method used to grid the sampled gravity data at the constraint points. Choose
        between "verde", "pygmt", or "eq_sources", by default "verde"
    spline_damping : typing.Any | None, optional
        damping values used if `grid_method` is "verde", by default None
    source_depth : float | None, optional
        depth of each source relative to the data elevation, positive downwards in
        meters, by default None
    eq_damping : float | None, optional
        damping values used if `grid_method` is "eq_sources", by default None
    block_size : float | None, optional
        block size used if `grid_method` is "eq_sources", by default None
    eq_points : list[NDArray] | None, optional
        specify source locations for equivalent source fitting, by default None
    constraints_weights_column : str | None, optional
       column name for weighting values of each constraint point. Used if
       `constraint_block_size` is not None or if `grid_method` is "verde", by default
       None
    grav_obs_height : float, optional
        Observation height to use if `grid_method` is "eq_sources", by default None
    regional_shift : float, optional
        shift to add to the regional field, by default 0

    Returns
    -------
    pd.DataFrame
        grav_df with new columns 'misfit', 'reg', and 'res'.
    """

    if constraints_df is None:
        msg = "need to provide constraints_df"
        raise ValueError(msg)

    grav_df = grav_df.copy()
    _check_grav_cols(grav_df)
    constraints_df = constraints_df.copy()

    region = vd.get_region((grav_df.easting, grav_df.northing))
    spacing = utils.get_spacing(grav_df)

    grav_df["misfit"] = grav_df.gravity_anomaly - grav_df.starting_gravity

    # grid the grav_df data
    grav_grid = grav_df.set_index(["northing", "easting"]).to_xarray().misfit

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

    if constraints_block_size is not None:
        # get weighted mean gravity value of constraint points in each cell
        if constraints_weights_column is None:
            weights = None
            uncertainty = False
        else:
            weights = constraints_df[constraints_weights_column]
            uncertainty = True

        blockmean = vd.BlockMean(
            spacing=constraints_block_size,
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
        if constraints_weights_column is None:
            data_cols = {"sampled_grav": data}
        else:
            data_cols = {"sampled_grav": data, constraints_weights_column: weights}
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
        # sample the resulting grid and add to grav_df dataframe
        grav_df = utils.sample_grids(
            df=grav_df,
            grid=regional_grav + regional_shift,
            sampled_name="reg",
            coord_names=("easting", "northing"),
            verbose="q",
        )
    elif grid_method == "verde":
        if constraints_weights_column is None:
            weights = None
        else:
            weights = constraints_df[constraints_weights_column]

        spline = vd.Spline(
            damping=spline_damping,
        )
        spline.fit(
            coordinates=(
                constraints_df.easting,
                constraints_df.northing,
            ),
            data=constraints_df.sampled_grav,
            weights=weights,
        )
        # predict fitted grid at gravity points
        grav_df["reg"] = (
            spline.predict(
                (grav_df.easting, grav_df.northing),
            )
            + regional_shift
        )
    elif grid_method == "eq_sources":
        if grav_obs_height is None:
            msg = "if grid_method is 'eq_sources`, must provide grav_obs_height"
            raise ValueError(msg)
        coords = (
            constraints_df.easting,
            constraints_df.northing,
            np.ones_like(constraints_df.easting) * grav_obs_height,
        )
        if constraints_weights_column is None:
            weights = None
        else:
            weights = constraints_df[constraints_weights_column]

        # create set of deep sources
        eqs = hm.EquivalentSources(
            depth=source_depth,
            damping=eq_damping,
            block_size=block_size,
            points=eq_points,
        )

        # fit the source coefficients to the data
        eqs.fit(
            coords,
            constraints_df.sampled_grav,
            weights=weights,
        )

        # predict sources at gravity points
        grav_df["reg"] = (
            eqs.predict(
                (
                    grav_df.easting,
                    grav_df.northing,
                    np.ones_like(grav_df.northing) * grav_obs_height,
                ),
            )
            + regional_shift
        )
    else:
        msg = "invalid string for grid_method"
        raise ValueError(msg)

    grav_df["res"] = grav_df.misfit - grav_df.reg
    return grav_df


def regional_separation(
    method: str,
    grav_df: pd.DataFrame,
    remove_starting_grav_mean: bool = False,
    **kwargs: typing.Any,
) -> pd.DataFrame:
    """
    Separate the regional field from the gravity data using the specified method
    and return the dataframe with a new column for the regional field.

    Parameters
    ----------
    method : str
        choose method to apply; one of "constant", "filter", "trend",
        "eq_sources", "constraints".
    grav_df : pd.DataFrame
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "starting_gravity".
    remove_starting_grav_mean : bool, optional
        add the mean of the starting gravity to the regional gravity field, by default
        False.
    **kwargs : typing.Any
        additional keyword arguments for the specified method.

    Returns
    -------
    pd.DataFrame
        grav_df with new columns 'misfit', 'reg', and 'res'.
    """
    grav_df = grav_df.copy()
    _check_grav_cols(grav_df)

    kwargs = kwargs.copy()

    if remove_starting_grav_mean is True:
        regional_shift = np.nanmean(grav_df.starting_gravity)
        msg = f"adding {regional_shift} to the observed gravity data"
        log.info(msg)
        if "regional_shift" in kwargs:
            msg = (
                "if remove_starting_grav_mean is True, do not provide"
                "`regional_shift` in kwargs"
            )
            raise ValueError(msg)
    else:
        regional_shift = kwargs.pop("regional_shift", 0)

    if method == "constant":
        return regional_constant(
            grav_df=grav_df,
            regional_shift=regional_shift,
            **kwargs,
        )
    if method == "filter":
        return regional_filter(
            grav_df=grav_df,
            regional_shift=regional_shift,
            **kwargs,
        )
    if method == "trend":
        return regional_trend(
            grav_df=grav_df,
            regional_shift=regional_shift,
            **kwargs,
        )
    if method == "eq_sources":
        return regional_eq_sources(
            grav_df=grav_df,
            regional_shift=regional_shift,
            **kwargs,
        )
    if method == "constraints":
        return regional_constraints(
            grav_df=grav_df,
            regional_shift=regional_shift,
            **kwargs,
        )
    msg = "invalid string for regional method"
    raise ValueError(msg)
