import typing

import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
from polartoolkit import utils as polar_utils

from invert4geom import cross_validation, logger, optimization, utils


def regional_constant(
    grav_ds: xr.Dataset,
    constant: float | None = None,
    constraints_df: pd.DataFrame | None = None,
    regional_shift: float = 0,
    mask_column: str | None = None,
) -> None:
    """
    approximate the regional field with a constant value. If constraint points of the
    layer of interested are supplied, the constant value will be the median misfit value
    at the constraint points.

    Parameters
    ----------
    grav_ds : xr.Dataset
        gravity data with variables "easting", "northing", "gravity_anomaly", and
        "forward_gravity".
    constant : float
        shift to apply to the data
    constraints_df : pandas.DataFrame
        a dataframe of constraint points with columns easting and northing.
    regional_shift : float, optional
        shift to add to the regional field, by default 0
    mask_column : str | None, optional
        Name of optional column with values to multiply estimated regional field by,
        should have values of 1 or 0, by default None.
    """

    if isinstance(grav_ds, xr.Dataset) is False:
        msg = "Function `regional_constant` has been changed, data must be provided as an xarray dataset initialized through function `create_data`"
        raise DeprecationWarning(msg)

    logger.debug("starting regional_constant")
    grav_ds.inv._check_grav_vars_for_regional()  # pylint: disable=protected-access

    grav_ds["misfit"] = grav_ds.gravity_anomaly - grav_ds.forward_gravity

    if (constraints_df is None) and (constant is None):
        msg = "need to provide either `constraints_df` of `constant`"
        raise ValueError(msg)

    if constraints_df is not None:
        if constant is not None:
            msg = (
                "`constant` parameter provide but not used since `constraints_df`"
                "were provided."
            )
            raise ValueError(msg)

        utils._check_constraints_inside_gravity_region(constraints_df, grav_ds.inv.df)  # pylint: disable=protected-access

        # get the gravity values at the constraint points
        constraints_df = constraints_df.copy()

        # sample gravity at constraint points
        constraints_df = utils.sample_grids(
            df=constraints_df,
            grid=grav_ds.misfit,
            sampled_name="sampled_grav",
        )

        # use median of sampled value for DC shift
        constant = np.nanmedian(constraints_df.sampled_grav)

        msg = (
            "using median gravity misfit of constraint points for regional field: "
            f"{constant} mGal"
        )
        logger.info(msg)

    grav_ds["reg"] = xr.full_like(grav_ds.misfit, constant + regional_shift)  # type: ignore[operator]

    grav_ds["res"] = grav_ds.misfit - grav_ds.reg

    if mask_column is not None:
        grav_ds["res"] *= grav_ds[mask_column]
        grav_ds["reg"] = grav_ds.misfit - grav_ds.res


def regional_filter(
    grav_ds: xr.Dataset,
    filter_width: float,
    regional_shift: float = 0,
    mask_column: str | None = None,
) -> None:
    """
    separate the regional field with a low-pass filter

    Parameters
    ----------
    grav_ds : xr.Dataset
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "forward_gravity".
    filter_width : float
        width in meters to use for the low-pass filter
    regional_shift : float, optional
        shift to add to the regional field, by default 0
    mask_column : str | None, optional
        Name of optional column with values to multiply estimated regional field by,
        should have values of 1 or 0, by default None.
    """
    if isinstance(grav_ds, xr.Dataset) is False:
        msg = "Function `regional_filter` has been changed, data must be provided as an xarray dataset initialized through function `create_data`"
        raise DeprecationWarning(msg)

    logger.debug("starting regional_filter")

    grav_ds.inv._check_grav_vars_for_regional()  # pylint: disable=protected-access

    grav_ds["misfit"] = grav_ds.gravity_anomaly - grav_ds.forward_gravity

    # remove the mean from the data
    data_mean = grav_ds.misfit.mean()
    misfit = grav_ds.misfit - data_mean

    # filter the gravity grid with the provided filter in meters
    regional_grid = utils.filter_grid(
        misfit,
        filter_width,
        filt_type="lowpass",
    )

    # add the mean back to the data
    regional_grid += data_mean

    regional_grid += regional_shift

    grav_ds["reg"] = regional_grid
    grav_ds["res"] = grav_ds.misfit - grav_ds.reg

    if mask_column is not None:
        grav_ds["res"] *= grav_ds[mask_column]
        grav_ds["reg"] = grav_ds.misfit - grav_ds.res


def regional_trend(
    grav_ds: xr.Dataset,
    trend: int,
    regional_shift: float = 0,
    mask_column: str | None = None,
) -> None:
    """
    separate the regional field with a trend

    Parameters
    ----------
    grav_ds : xr.Dataset
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "forward_gravity".
    trend : int
        order of the polynomial trend to fit to the data
    regional_shift : float, optional
        shift to add to the regional field, by default 0
    mask_column : str | None, optional
        Name of optional column with values to multiply estimated regional field by,
        should have values of 1 or 0, by default None.
    """
    if isinstance(grav_ds, xr.Dataset) is False:
        msg = "Function `regional_trend` has been changed, data must be provided as an xarray dataset initialized through function `create_data`"
        raise DeprecationWarning(msg)

    logger.debug("starting regional_trend")

    grav_ds.inv._check_grav_vars_for_regional()  # pylint: disable=protected-access

    grav_ds["misfit"] = grav_ds.gravity_anomaly - grav_ds.forward_gravity

    vdtrend = vd.Trend(degree=trend).fit(
        (grav_ds.inv.df.easting, grav_ds.inv.df.northing),
        grav_ds.inv.df.misfit,
    )

    grav_ds["reg"] = (
        vdtrend.grid(
            coordinates=(grav_ds.easting, grav_ds.northing),
            data_names=["reg"],
        ).reg
        + regional_shift
    )

    grav_ds["res"] = grav_ds.misfit - grav_ds.reg

    if mask_column is not None:
        grav_ds["res"] *= grav_ds[mask_column]
        grav_ds["reg"] = grav_ds.misfit - grav_ds.res


def regional_eq_sources(
    grav_ds: xr.Dataset,
    depth: float | str = "default",
    damping: float | None = None,
    block_size: float | None = None,
    grav_obs_height: float | None = None,
    regional_shift: float = 0,
    cv: bool = False,
    weights_column: str | None = None,
    cv_kwargs: dict[str, typing.Any] | None = None,
    mask_column: str | None = None,
) -> None:
    """
    separate the regional field by estimating deep equivalent sources

    Parameters
    ----------
    grav_ds : xr.Dataset
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "forward_gravity".
    depth : float
        depth of each source relative to the data elevation
    damping : float | None, optional
        smoothness to impose on estimated coefficients, by default None
    block_size : float | None, optional
        block reduce the data to speed up, by default None
    grav_obs_height: float, optional
        Observation height to use predicting the eq sources, by default None and will
        use the data height from grav_ds.
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
    mask_column : str | None, optional
        Name of optional column with values to multiply estimated regional field by,
        should have values of 1 or 0, by default None.
    """
    if isinstance(grav_ds, xr.Dataset) is False:
        msg = "Function `regional_eq_sources` has been changed, data must be provided as an xarray dataset initialized through function `create_data`"
        raise DeprecationWarning(msg)

    logger.debug("starting regional_eq_sources")

    grav_ds.inv._check_grav_vars_for_regional()  # pylint: disable=protected-access

    grav_ds["misfit"] = grav_ds.gravity_anomaly - grav_ds.forward_gravity

    grav_df = grav_ds.inv.df

    coords = (grav_df.easting, grav_df.northing, grav_df.upward)

    weights = None if weights_column is None else grav_df[weights_column]

    if cv is True:
        _, eqs = optimization.optimize_eq_source_params(
            coordinates=coords,
            data=grav_df.misfit,
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
    df = grav_ds.inv.df
    df["reg"] = eqs.predict(coords) + regional_shift
    grav_ds["reg"] = df.set_index(["northing", "easting"]).to_xarray().reg

    grav_ds["res"] = grav_ds.misfit - grav_ds.reg

    if mask_column is not None:
        grav_ds["res"] *= grav_ds[mask_column]
        grav_ds["reg"] = grav_ds.misfit - grav_ds.res


def regional_constraints(
    grav_ds: xr.Dataset,
    constraints_df: pd.DataFrame,
    grid_method: str = "eq_sources",
    constraints_block_size: float | None = None,
    constraints_weights_column: str | None = None,
    tension_factor: float = 1,
    spline_dampings: float | list[float] | None = None,
    depth: float | str | None = None,
    damping: float | None = None,
    cv: bool = False,
    block_size: float | None = None,
    grav_obs_height: float | None = None,
    cv_kwargs: dict[str, typing.Any] | None = None,
    regional_shift: float = 0,
    mask_column: str | None = None,
) -> None:
    """
    Separate the regional field by sampling and re-gridding the gravity misfit at
    points of known topography (constraint points). The re-gridding can be done with:
    1. Tensioned minimum curvature with PyGMT, using `grid_method` "pygmt", 2.
    Bi-Harmonica splines with Verde, using `grid_method` "verde", or 3. Equivalent
    Sources with Harmonica, using `grid_method` "eq_sources". Optionally, a dc-shift can
    be added to the calculated regional with `regional_shift`.

    Parameters
    ----------
    grav_ds : xarray.Dataset
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "forward_gravity".
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
    grav_obs_height : float, optional
        Observation height to use if `grid_method` is "eq_sources", by default None
    cv_kwargs : dict[str, typing.Any] | None, optional
        additional keyword arguments for the cross-validation optimization of
        equivalent source parameters, by default None. Can contain: "n_trials",
        "damping_limits", "depth_limits", "block_size_limits", "sampler",
        "plot", "progressbar", "parallel", "fname", "dtype", or "delayed".
    regional_shift : float, optional
        shift to add to the regional field, by default 0
    mask_column : str | None, optional
        Name of optional column with values to multiply estimated residual field by,
        should have values of 1 or 0, by default None.
    """

    if isinstance(grav_ds, xr.Dataset) is False:
        msg = "Function `regional_constraints` has been changed, data must be provided as an xarray dataset initialized through function `create_data`"
        raise DeprecationWarning(msg)

    logger.debug("starting regional_constraints")

    if constraints_df is None:
        msg = "need to provide constraints_df"
        raise ValueError(msg)

    grav_ds.inv._check_grav_vars_for_regional()  # pylint: disable=protected-access

    constraints_df = constraints_df.copy()

    grav_ds["misfit"] = grav_ds.gravity_anomaly - grav_ds.forward_gravity

    # sample gravity at constraint points
    constraints_df = utils.sample_grids(
        df=constraints_df,
        grid=grav_ds.misfit,
        sampled_name="sampled_grav",
        no_skip=True,
        verbose="q",
    )

    # drop rows with NaN values
    constraints_df = constraints_df[constraints_df.sampled_grav.notna()]

    if grid_method == "eq_sources":
        # sample gravity observation height at constraint points
        constraints_df = utils.sample_grids(
            df=constraints_df,
            grid=grav_ds.upward,
            sampled_name="sampled_grav_height",
            no_skip=True,
            verbose="q",
        )
        # drop rows with NaN values
        constraints_df = constraints_df[constraints_df.sampled_grav_height.notna()]

    # get weights for each constraint point
    if constraints_weights_column is None:
        weights = None
        uncertainty = False
    else:
        weights = constraints_df[constraints_weights_column]
        uncertainty = True

    # get weighted mean gravity value of constraint points in each cell
    if constraints_block_size is not None:
        if grid_method == "eq_sources":
            msg = "blockmean reduction not supported for eq_sources grid method yet"
            raise ValueError(msg)

        blockmean = vd.BlockMean(
            spacing=constraints_block_size,
            uncertainty=uncertainty,
        )

        data = constraints_df.sampled_grav
        weight_values = weights

        coordinates, data, weights = blockmean.filter(
            coordinates=(
                constraints_df["easting"],
                constraints_df["northing"],
            ),
            data=data,
            weights=weight_values,
        )

        # add reduced coordinates to a dictionary
        coord_cols = dict(zip(["easting", "northing"], coordinates, strict=False))

        # add reduced data to a dictionary
        if constraints_weights_column is None:
            data_cols = {"sampled_grav": data}
        else:
            data_cols = {
                "sampled_grav": data,
                constraints_weights_column: weights,
            }

        # merge dicts and create dataframe
        constraints_df = pd.DataFrame(data=coord_cols | data_cols)

        constraints_df = constraints_df.dropna(how="any")
    ###
    ###
    # Tensioned splines with PyGMT
    ###
    ###
    # grid the entire regional gravity based just on the values at the constraints
    if grid_method == "pygmt":
        registration = polar_utils.get_grid_info(grav_ds.forward_gravity)[-1]
        da = pygmt.surface(
            data=constraints_df[["easting", "northing", "sampled_grav"]],
            region=grav_ds.region,
            spacing=grav_ds.spacing,
            registration=registration,
            tension=tension_factor,
            verbose="q",
        ).rename({"x": "easting", "y": "northing"})
        grav_ds["reg"] = da
    ###
    ###
    # Bi-Harmonica splines with Verde
    ###
    ###
    elif grid_method == "verde":
        spline = utils.optimal_spline_damping(
            coordinates=(
                constraints_df.easting,
                constraints_df.northing,
            ),
            data=constraints_df.sampled_grav,
            weights=weights,
            dampings=spline_dampings,
        )
        # predict fitted grid at gravity points
        grav_ds["reg"] = spline.grid(
            coordinates=(grav_ds.easting.to_numpy(), grav_ds.northing.to_numpy()),
            data_names=["reg"],
        ).reg
    ###
    ###
    # Equivalent Sources with Harmonica
    ###
    ###
    elif grid_method == "eq_sources":
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
            try:
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
            except ValueError as e:
                logger.error(e)
                msg = (
                    "eq sources optimization failed, using damping=None and "
                    "depth='default'"
                )
                logger.error(msg)
                eqs = hm.EquivalentSources(
                    depth="default",
                    damping=None,
                    block_size=block_size,
                )
                eqs.fit(
                    coords,
                    constraints_df.sampled_grav,
                    weights=weights,
                )

        else:
            if depth is None:
                depth = "default"

            # create set of deep sources
            eqs = hm.EquivalentSources(
                depth=depth,
                damping=damping,
                block_size=block_size,
            )

            # fit the source coefficients to the data
            eqs.fit(
                coords,
                constraints_df.sampled_grav,
                weights=weights,
            )
        msg = "depth: %s, damping: %s"
        logger.debug(msg, eqs.depth, eqs.damping)

        # predict sources at gravity points and chosen height for upward continuation
        df = grav_ds.inv.df
        if grav_obs_height is None:
            grav_obs_height = df.upward
        else:
            grav_obs_height = np.ones_like(df.easting.values) * grav_obs_height
        coords = (df.easting, df.northing, grav_obs_height)

        df["reg"] = eqs.predict(coords)
        grav_ds["reg"] = df.set_index(["northing", "easting"]).to_xarray().reg
    else:
        msg = "invalid string for grid_method"
        raise ValueError(msg)

    grav_ds["reg"] += regional_shift
    grav_ds["res"] = grav_ds.misfit - grav_ds.reg

    if mask_column is not None:
        grav_ds["res"] *= grav_ds[mask_column]
        grav_ds["reg"] = grav_ds.misfit - grav_ds.res


def regional_constraints_cv(
    grav_ds: xr.Dataset,
    constraints_df: pd.DataFrame,
    split_kwargs: dict[str, typing.Any] | None = None,
    **kwargs: typing.Any,
) -> None:
    """
    This is a convenience function to wrap
    `optimization.optimize_regional_constraint_point_minimization`. It takes a full
    constraints dataframe and dictionary `split_kwargs`, to split the constraints into
    testing and training sets (with K-folds), uses these folds in a K-Folds
    hyperparameter optimization to find the set of parameter values which estimates the
    best regional field. It then uses the optimal parameter values and all of the
    constraint points to re-calculate the best regional field. All kwargs are passed to
    the function :func:`optimize_regional_constraint_point_minimization`

    Parameters
    ----------
    grav_ds : xr.Dataset
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "forward_gravity".
    constraints_df : pandas.DataFrame
        dataframe of un-separated constraints
    split_kwargs : dict[str, typing.Any] | None, optional
        kwargs to be passed to `split_test_train`, by default None
    **kwargs : typing.Any
        kwargs to be passed to `optimize_regional_constraint_point_minimization`
    """
    logger.debug("starting regional_constraints_cv")

    utils._check_constraints_inside_gravity_region(  # pylint: disable=protected-access
        constraints_df,
        grav_ds.inv.df,
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

    _, grav_ds, _ = optimization.optimize_regional_constraint_point_minimization(
        grav_ds=grav_ds,
        testing_training_df=testing_training_df,
        **kwargs,
    )


def regional_separation(
    method: str,  # noqa: ARG001 # pylint: disable=unused-argument
    grav_ds: xr.Dataset,  # noqa: ARG001 # pylint: disable=unused-argument
    remove_starting_grav_mean: bool = False,  # noqa: ARG001 # pylint: disable=unused-argument
    **kwargs: typing.Any,  # noqa: ARG001 # pylint: disable=unused-argument
) -> xr.Dataset:
    """
    DEPRECATED: use :meth:`DatasetAccessorInvert4Geom.regional_separation` instead.
    """
    msg = (
        "Function `regional_separation` deprecated, use the `DatasetAccessorInvert4Geom.regional_separation` method"
        "instead"
    )
    raise DeprecationWarning(msg)
