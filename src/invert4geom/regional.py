from __future__ import annotations

import copy
import typing

import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import verde as vd
from numpy.typing import NDArray

from invert4geom import cross_validation, log, optimization, utils


def _check_grav_cols(grav_df: pd.DataFrame) -> None:
    """
    ensure gravity dataframe has the necessary columns

    Parameters
    ----------
    grav_df : pandas.DataFrame
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
    grav_df : pandas.DataFrame
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "starting_gravity".
    constant : float
        shift to apply to the data
    constraints_df : pandas.DataFrame
        a dataframe of constraint points with columns easting and northing.
    regional_shift : float, optional
        shift to add to the regional field, by default 0

    Returns
    -------
    pandas.DataFrame
        grav_df with new columns 'misfit', 'reg', and 'res'.
    """
    log.debug("starting regional_constant")
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

        utils._check_constraints_inside_gravity_region(constraints_df, grav_df)  # pylint: disable=protected-access

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
    grav_df : pandas.DataFrame
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "starting_gravity".
    filter_width : float
        width in meters to use for the low-pass filter
    regional_shift : float, optional
        shift to add to the regional field, by default 0

    Returns
    -------
    pandas.DataFrame
        grav_df with new columns 'misfit', 'reg', and 'res'.
    """
    log.debug("starting regional_filter")

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
    grav_df : pandas.DataFrame
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "starting_gravity".
    trend : int
        order of the polynomial trend to fit to the data
    regional_shift : float, optional
        shift to add to the regional field, by default 0

    Returns
    -------
    pandas.DataFrame
        grav_df with new columns 'misfit', 'reg', and 'res'.
    """
    log.debug("starting regional_trend")

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
    depth: float | str = "default",
    damping: float | None = None,
    block_size: float | None = None,
    points: list[NDArray] | None = None,
    grav_obs_height: float | None = None,
    regional_shift: float = 0,
    cv: bool = False,
    weights_column: str | None = None,
    cv_kwargs: dict[str, typing.Any] | None = None,
) -> pd.DataFrame:
    """
    separate the regional field by estimating deep equivalent sources

    Parameters
    ----------
    grav_df : pandas.DataFrame
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "starting_gravity".
    depth : float
        depth of each source relative to the data elevation
    damping : float | None, optional
        smoothness to impose on estimated coefficients, by default None
    block_size : float | None, optional
        block reduce the data to speed up, by default None
    points : list[numpy.ndarray] | None, optional
        specify source locations for equivalent source fitting, by default None
    grav_obs_height: float, optional
        Observation height to use predicting the eq sources, by default None and will
        use the data height from grav_df.
    regional_shift : float, optional
        shift to add to the regional field, by default 0
    cv : bool, optional
        use cross-validation to find the best equivalent source parameters, by default
        False, provide dictionary `cv_kwargs` which is passed to
        `optimize_eq_source_params` and can contain: "n_trials", "damping_limits",
        "depth_limits", "block_size_limits", "sampler", "plot", "progressbar",
        "parallel", "dtype", or "delayed".
    weights_column: str | None, optional
        column name for weighting values of each gravity point.
    Returns
    -------
    pandas.DataFrame
        grav_df with new columns 'misfit', 'reg', and 'res'.
    """
    log.debug("starting regional_eq_sources")

    grav_df = grav_df.copy()
    _check_grav_cols(grav_df)

    grav_df["misfit"] = grav_df.gravity_anomaly - grav_df.starting_gravity

    coords = (grav_df.easting, grav_df.northing, grav_df.upward)

    weights = None if weights_column is None else grav_df[weights_column]

    if cv is True:
        _, eqs = optimization.optimize_eq_source_params(
            coordinates=coords,
            data=grav_df.misfit,
            points=points,
            weights=weights,
            depth=depth,
            damping=damping,
            block_size=block_size,
            **cv_kwargs,  # type: ignore[arg-type]
        )
    else:
        # create set of deep sources
        eqs = hm.EquivalentSources(
            depth=depth,
            damping=damping,
            block_size=block_size,
            points=points,
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
    grav_df["reg"] = eqs.predict(coords) + regional_shift

    grav_df["res"] = grav_df.misfit - grav_df.reg

    return grav_df


def regional_constraints(
    grav_df: pd.DataFrame,
    constraints_df: pd.DataFrame,
    grid_method: str = "eq_sources",
    constraints_block_size: float | None = None,
    constraints_weights_column: str | None = None,
    tension_factor: float = 1,
    registration: str = "g",
    spline_dampings: float | list[float] | None = None,
    depth: float | str | None = None,
    damping: float | None = None,
    cv: bool = False,
    block_size: float | None = None,
    points: list[NDArray] | None = None,
    grav_obs_height: float | None = None,
    cv_kwargs: dict[str, typing.Any] | None = None,
    regional_shift: float = 0,
) -> pd.DataFrame:
    """
    Separate the regional field by sampling and re-gridding the gravity misfit at
    points of known topography (constraint points). The re-gridding can be done with:
    1. Tensioned minimum curvature with PyGMT, using `grid_method` "pygmt", 2.
    Bi-Harmonica splines with Verde, using `grid_method` "verde", or 3. Equivalent
    Sources with Harmonica, using `grid_method` "eq_sources". Optionally, a dc-shift can
    be added to the calculated regional with `regional_shift`.

    Parameters
    ----------
    grav_df : pandas.DataFrame
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "starting_gravity".
    constraints_df : pandas.DataFrame
        dataframe of constraints with columns "easting", "northing", and "upward".
    grid_method : str, optional
        method used to grid the sampled gravity data at the constraint points. Choose
        between "verde", "pygmt", or "eq_sources", by default "eq_sources"
    constraints_block_size : float | None, optional
        size of block used in a block-mean reduction of the constraints points, by
        default None
    constraints_weights_column : str | None, optional
       column name for weighting values of each constraint point. Used if
       `constraint_block_size` is not None or if `grid_method` is "verde" or
       "eq_sources", by default None
    tension_factor : float, optional
        Tension factor used if `grid_method` is "pygmt", by default 1
    registration : str, optional
       grid registration used if `grid_method` is "pygmt",, by default "g"
    spline_dampings : float | list[float] | None, optional
        damping values used if `grid_method` is "verde", by default None
    depth : float | str | None, optional
        depth of each source relative to the data elevation, positive downwards in
        meters, by default None
    damping : float | None, optional
        damping values used if `grid_method` is "eq_sources", by default None
    cv : bool, optional
        use cross-validation to find the best equivalent source parameters, by
        default False, provide dictionary `cv_kwargs` which is passed to
        `optimization.optimize_eq_source_params` and can contain: "n_trials",
        "damping_limits", "depth_limits", "block_size_limits", and "progressbar".
    block_size : float | None, optional
        block size used if `grid_method` is "eq_sources", by default None
    points : list[numpy.ndarray] | None, optional
        specify source locations for equivalent source fitting, by default None
    grav_obs_height : float, optional
        Observation height to use if `grid_method` is "eq_sources", by default None
    cv_kwargs : dict[str, typing.Any] | None, optional
        additional keyword arguments for the cross-validation optimization of
        equivalent source parameters, by default None. Can contain: "n_trials",
        "damping_limits", "depth_limits", "block_size_limits", "points", "sampler",
        "plot", "progressbar", "parallel", "fname", "dtype", or "delayed".
    regional_shift : float, optional
        shift to add to the regional field, by default 0

    Returns
    -------
    pandas.DataFrame
        grav_df with new columns 'misfit', 'reg', and 'res'.
    """
    log.debug("starting regional_constraints")

    if constraints_df is None:
        msg = "need to provide constraints_df"
        raise ValueError(msg)

    utils._check_constraints_inside_gravity_region(constraints_df, grav_df)  # pylint: disable=protected-access

    grav_df = grav_df.copy()
    _check_grav_cols(grav_df)
    constraints_df = constraints_df.copy()

    grav_df["misfit"] = grav_df.gravity_anomaly - grav_df.starting_gravity

    # grid the grav_df data
    grav_grid = grav_df.set_index(["northing", "easting"]).to_xarray()

    # sample gravity at constraint points
    constraints_df = utils.sample_grids(
        df=constraints_df,
        grid=grav_grid.misfit,
        sampled_name="sampled_grav",
        coord_names=("easting", "northing"),
        no_skip=True,
        verbose="q",
    )

    # drop rows with NaN values
    constraints_df = constraints_df[constraints_df.sampled_grav.notna()]

    # get weights for each constraint point
    if constraints_weights_column is None:
        weights = None
        uncertainty = False
    else:
        weights = constraints_df[constraints_weights_column]
        uncertainty = True

    # get weighted mean gravity value of constraint points in each cell
    if constraints_block_size is not None:
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

        constraints_df = constraints_df.dropna(how="any")
    ###
    ###
    # Tensioned minimum curvature with PyGMT
    ###
    ###
    # grid the entire regional gravity based just on the values at the constraints
    if grid_method == "pygmt":
        regional_grav = pygmt.surface(
            data=constraints_df[["easting", "northing", "sampled_grav"]],
            region=vd.get_region((grav_df.easting, grav_df.northing)),
            spacing=utils.get_spacing(grav_df),
            registration=registration,
            tension=tension_factor,
            verbose="q",
        )
        # sample the resulting grid and add to grav_df dataframe
        grav_df = utils.sample_grids(
            df=grav_df,
            grid=regional_grav,
            sampled_name="reg",
            coord_names=("easting", "northing"),
            verbose="q",
        )
    ###
    ###
    # Bi-Harmonica splines with Verde
    ###
    ###
    elif grid_method == "verde":
        spline = utils.best_spline_cv(
            coordinates=(
                constraints_df.easting,
                constraints_df.northing,
            ),
            data=constraints_df.sampled_grav,
            weights=weights,
            dampings=spline_dampings,
        )
        # predict fitted grid at gravity points
        grav_df["reg"] = spline.predict(
            (grav_df.easting, grav_df.northing),
        )
    ###
    ###
    # Equivalent Sources with Harmonica
    ###
    ###
    elif grid_method == "eq_sources":
        if grav_obs_height is None:
            grav_obs_height = grav_df.upward
        else:
            grav_obs_height = np.ones_like(grav_df.easting) * grav_obs_height

        # sample gravity observation height at constraint points
        constraints_df = utils.sample_grids(
            df=constraints_df,
            grid=grav_grid.upward,
            sampled_name="sampled_grav_height",
            coord_names=("easting", "northing"),
            no_skip=True,
            verbose="q",
        )
        coords = (
            constraints_df.easting,
            constraints_df.northing,
            constraints_df.sampled_grav_height,
        )

        if depth == "default":
            depth = 4.5 * np.mean(
                vd.median_distance(
                    (coords[0], coords[1]),
                    k_nearest=1,
                )
            )

        if cv is True:
            # eqs = utils.best_equivalent_source_damping(
            _, eqs = optimization.optimize_eq_source_params(
                coordinates=coords,
                data=constraints_df.sampled_grav,
                # kwargs
                weights=weights,
                depth=depth,
                damping=damping,
                block_size=block_size,
                **cv_kwargs,  # type: ignore[arg-type]
            )
        else:
            # create set of deep sources
            eqs = hm.EquivalentSources(
                depth=depth,
                damping=damping,
                block_size=block_size,
                points=points,
            )
            # fit the source coefficients to the data
            eqs.fit(
                coords,
                constraints_df.sampled_grav,
                weights=weights,
            )
        msg = "depth: %s, damping: %s"
        log.debug(msg, eqs.depth, eqs.damping)

        # predict sources at gravity points and chosen height for upward continuation
        grav_df["reg"] = eqs.predict(
            (
                grav_df.easting,
                grav_df.northing,
                grav_obs_height,  # either grav_df.upward or user-set constant value
            ),
        )
    else:
        msg = "invalid string for grid_method"
        raise ValueError(msg)

    grav_df["reg"] += regional_shift
    grav_df["res"] = grav_df.misfit - grav_df.reg
    return grav_df


def regional_constraints_cv(
    constraints_df: pd.DataFrame,
    split_kwargs: dict[str, typing.Any] | None = None,
    **kwargs: typing.Any,
) -> pd.DataFrame:
    """
    This is a convenience function to wrap
    `optimization.optimize_regional_constraint_point_minimization`. It takes a full
    constraints dataframe and dictionary `split_kwargs`, to split the constraints into
    testing and training sets (with K-folds), uses these folds in a K-Folds
    hyperparameter optimization to find the set of parameter values which estimates the
    best regional field. It then uses the optimal parameter values and all of the
    constraint points to re-calculate the best regional field. All kwargs are passed to
    the function :func:`.optimize_regional_constraint_point_minimization`

    Parameters
    ----------
    constraints_df : pandas.DataFrame
        dataframe of un-separated constraints
    split_kwargs : dict[str, typing.Any] | None, optional
        kwargs to be passed to `split_test_train`, by default None
    **kwargs : typing.Any
        kwargs to be passed to `optimize_regional_constraint_point_minimization`

    Returns
    -------
    pandas.DataFrame
        a gravity dataframe with new columns 'misfit', 'reg', and 'res'.
    """

    log.debug("starting regional_constraints_cv")

    utils._check_constraints_inside_gravity_region(  # pylint: disable=protected-access
        constraints_df, kwargs.get("grav_df")
    )

    df = constraints_df.copy()
    df = df[df.columns.drop(list(df.filter(regex="fold_")))]

    if split_kwargs is None:
        msg = "need to provide split_kwargs"
        raise ValueError(msg)

    testing_training_df = cross_validation.split_test_train(
        df,
        **split_kwargs,
    )

    _, grav_df, _ = optimization.optimize_regional_constraint_point_minimization(
        testing_training_df=testing_training_df,
        **kwargs,
    )

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
        "eq_sources", "constraints" or "constraints_cv".
    grav_df : pandas.DataFrame
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "starting_gravity".
    remove_starting_grav_mean : bool, optional
        add the mean of the starting gravity to the regional gravity field, by default
        False.
    **kwargs : typing.Any
        additional keyword arguments for the specified method.

    Returns
    -------
    pandas.DataFrame
        grav_df with new columns 'misfit', 'reg', and 'res'.
    """
    grav_df = grav_df.copy()
    _check_grav_cols(grav_df)

    kwargs = copy.deepcopy(kwargs)

    if remove_starting_grav_mean is True:
        regional_shift = np.nanmean(grav_df.starting_gravity)
        msg = f"adding {regional_shift} to the regional gravity data"
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
    if method == "constraints_cv":
        return regional_constraints_cv(
            grav_df=grav_df,
            remove_starting_grav_mean=remove_starting_grav_mean,
            **kwargs,
        )

    msg = "invalid string for regional method"
    raise ValueError(msg)
