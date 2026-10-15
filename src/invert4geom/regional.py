import typing
import warnings

import bordado as bd
import harmonica as hm
import numpy as np
import pandas as pd
import verde as vd
import xarray as xr

from invert4geom import cross_validation, logger, maps, optimization, utils


def _initialize_separation(grav_ds: xr.Dataset, function_name: str) -> None:
    """
    Validate the gravity dataset and (re)calculate the gravity misfit as the
    difference between the observed and forward gravity.
    """
    if not isinstance(grav_ds, xr.Dataset):
        msg = (
            f"Function `{function_name}` has been changed, data must be provided as "
            "an xarray dataset initialized through function `create_data`"
        )
        raise DeprecationWarning(msg)

    logger.debug("starting %s", function_name)

    grav_ds.inv._check_grav_vars_for_regional()  # pylint: disable=protected-access

    grav_ds["misfit"] = grav_ds.gravity_anomaly - grav_ds.forward_gravity


def _check_projected_units(grav_ds: xr.Dataset, function_name: str) -> None:
    """raise an error for methods which only support projected coordinates"""
    if grav_ds.model_type == "tesseroids":
        msg = (
            f"function `{function_name}` only supports gravity data with projected "
            "units, not geographic units (i.e. lat/lon)"
        )
        raise NotImplementedError(msg)


def _finalize_separation(
    grav_ds: xr.Dataset,
    regional_shift: float,
    mask_column: str | None,
    reverse_regional_residual: bool,
) -> None:
    """
    Apply the optional shift to the estimated regional field, calculate the residual
    field, optionally mask the residual, and optionally swap the two fields.
    """
    # the mask is now a `mask` variable of the gravity dataset, added by `create_data`
    if mask_column is not None:
        warnings.warn(
            "`mask_column` is deprecated, instead supply the mask as a `mask` variable "
            "of the gravity dataset passed to `create_data`.",
            UserWarning,
            stacklevel=3,
        )
    elif "mask" in grav_ds:
        mask_column = "mask"
        logger.info("masking the residual with the gravity dataset's `mask` variable")

    grav_ds["reg"] = grav_ds.reg + regional_shift
    grav_ds["res"] = grav_ds.misfit - grav_ds.reg

    mask = None
    if mask_column is not None:
        # normalize so that cells marked with either 0 or NaN are zeroed, otherwise a
        # mask made for a model (where NaNs are excluded) would give NaN residuals
        mask = utils._normalize_mask(grav_ds[mask_column], grav_ds.misfit)  # pylint: disable=protected-access
        grav_ds["res"] = grav_ds.res * mask
        grav_ds["reg"] = grav_ds.misfit - grav_ds.res

    if reverse_regional_residual:
        grav_ds["reg"], grav_ds["res"] = grav_ds["res"], grav_ds["reg"]

    # keep a copy of the mask so that the zeroed-out residual values can be excluded
    # from the inversion convergence statistics, see the `stats_res` accessor property.
    # after reversing, the mask applies to the regional field instead of the residual
    if (mask is not None) and not reverse_regional_residual:
        grav_ds["res_mask"] = mask
    elif "res_mask" in grav_ds:
        # remove a stale mask left by a previous separation of the same dataset
        del grav_ds["res_mask"]


def regional_constant(
    grav_ds: xr.Dataset,
    constant: float | None = None,
    constraints_df: pd.DataFrame | None = None,
    regional_shift: float = 0,
    mask_column: str | None = None,
    reverse_regional_residual: bool = False,
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
        deprecated, instead supply the mask to :func:`create_data`, by default None
    reverse_regional_residual : bool, optional
        if True, reverse the regional and residual fields after calculation, by default
        False
    """

    _initialize_separation(grav_ds, "regional_constant")

    if (constraints_df is None) and (constant is None):
        msg = "need to provide either `constraints_df` or `constant`"
        raise ValueError(msg)

    if constraints_df is not None:
        if constant is not None:
            msg = (
                "`constant` parameter provide but not used since `constraints_df`"
                "were provided."
            )
            raise ValueError(msg)

        utils._check_constraints_inside_gravity_region(constraints_df, grav_ds)  # pylint: disable=protected-access

        # get the gravity values at the constraint points
        constraints_df = constraints_df.copy()

        # sample gravity at constraint points
        constraints_df = utils.sample_grids(
            df=constraints_df,
            grid=grav_ds.misfit,
            sampled_name="sampled_grav",
            coord_names=grav_ds.coord_names,
        )

        # use median of sampled value for DC shift
        constant = np.nanmedian(constraints_df.sampled_grav)

        msg = (
            "using median gravity misfit of constraint points for regional field: "
            f"{constant} mGal"
        )
        logger.info(msg)

    grav_ds["reg"] = xr.full_like(grav_ds.misfit, constant)

    _finalize_separation(
        grav_ds, regional_shift, mask_column, reverse_regional_residual
    )


def regional_filter(
    grav_ds: xr.Dataset,
    filter_width: float | None = None,
    filter_type: str = "lowpass",
    height_displacement: float | None = None,
    regional_shift: float = 0,
    mask_column: str | None = None,
    reverse_regional_residual: bool = False,
) -> None:
    """
    separate the regional field with a filter

    Parameters
    ----------
    grav_ds : xr.Dataset
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "forward_gravity".
    filter_width : float | None, optional
        width in grid units to use for low-pass / high-pass filters, by default None
    filter_type : str, optional
        type of filter to use, by default "lowpass"
    height_displacement : float | None, optional
        height displacement in meters for ``filter_type="up_continue"``, by default None
    regional_shift : float, optional
        shift to add to the regional field, by default 0
    mask_column : str | None, optional
        deprecated, instead supply the mask to :func:`create_data`, by default None
    reverse_regional_residual : bool, optional
        if True, reverse the regional and residual fields after calculation, by default
        False
    """
    _initialize_separation(grav_ds, "regional_filter")

    # remove the mean from the data
    data_mean = grav_ds.misfit.mean()
    misfit = grav_ds.misfit - data_mean

    # filter the gravity grid with the provided filter in meters
    regional_grid = utils.filter_grid(
        misfit,
        filter_width,
        height_displacement=height_displacement,
        filter_type=filter_type,
    )

    # add the mean back to the data
    regional_grid += data_mean

    grav_ds["reg"] = regional_grid

    _finalize_separation(
        grav_ds, regional_shift, mask_column, reverse_regional_residual
    )


def regional_trend(
    grav_ds: xr.Dataset,
    trend: int,
    regional_shift: float = 0,
    mask_column: str | None = None,
    reverse_regional_residual: bool = False,
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
        deprecated, instead supply the mask to :func:`create_data`, by default None
    reverse_regional_residual : bool, optional
        if True, reverse the regional and residual fields after calculation, by default
        False
    """
    _initialize_separation(grav_ds, "regional_trend")

    coord_names = grav_ds.coord_names

    vdtrend = vd.Trend(degree=trend).fit(
        (grav_ds.inv.df[coord_names[0]], grav_ds.inv.df[coord_names[1]]),
        grav_ds.inv.df.misfit,
    )

    grav_ds["reg"] = vdtrend.grid(
        coordinates=(grav_ds[coord_names[0]], grav_ds[coord_names[1]]),
        data_names=["reg"],
        dims=(coord_names[1], coord_names[0]),
    ).reg

    _finalize_separation(
        grav_ds, regional_shift, mask_column, reverse_regional_residual
    )


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
    reverse_regional_residual: bool = False,
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
        deprecated, instead supply the mask to :func:`create_data`, by default None
    reverse_regional_residual : bool, optional
        if True, reverse the regional and residual fields after calculation, by default
        False
    """
    _initialize_separation(grav_ds, "regional_eq_sources")
    _check_projected_units(grav_ds, "regional_eq_sources")

    coord_names = ("easting", "northing")

    grav_df = grav_ds.inv.df

    coords = (grav_df[coord_names[0]], grav_df[coord_names[1]], grav_df.upward)

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

    coords = (
        grav_df[coord_names[0]],
        grav_df[coord_names[1]],
        upward_continuation_height,
    )
    df = grav_ds.inv.df
    df["reg"] = eqs.predict(coords)
    grav_ds["reg"] = df.set_index([coord_names[1], coord_names[0]]).to_xarray().reg

    _finalize_separation(
        grav_ds, regional_shift, mask_column, reverse_regional_residual
    )


def _sampled_misfit_grid(
    grav_ds: xr.Dataset,
    grid_constraints: pd.DataFrame,
    coord_names: tuple[str, str],
) -> xr.DataArray:
    """
    Grid the misfit sampled at the constraint points which fall on the gravity grid's
    nodes, leaving the cells without a constraint point as missing values (NaNs). See
    :func:`regional_constraints`.
    """
    registration = maps.get_grid_info(grav_ds.forward_gravity)[-1]
    da: xr.DataArray = utils.points_to_grid(
        x=grid_constraints[coord_names[0]],
        y=grid_constraints[coord_names[1]],
        z=grid_constraints.sampled_grav,
        region=grav_ds.region,
        spacing=grav_ds.spacing,
        registration=registration,
        coord_names=coord_names,
    )
    return da.rename("reg")


# methods which grid the sampled misfit at all of the constraint points
_CONSTRAINT_GRID_METHODS = (
    "pygmt",
    "splines",
    "eq_sources",
    "nearest",
    "linear",
    "cubic",
)
# methods which are only available when filling the cells without a constraint point
_FILL_ONLY_GRID_METHODS = ("nearest", "linear", "cubic")


def _block_reduce_constraints(
    constraints_df: pd.DataFrame,
    weights: pd.Series | None,
    weights_col: str | None,
    coord_names: tuple[str, str],
    region: tuple[float, float, float, float],
    block_size: float,
    block_reduction: str,
    block_reduction_col: str | None,
    with_height: bool,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Reduce the sampled constraint points to one value per block, optionally reducing the
    sampled gravity observation height alongside the sampled misfit. See
    :func:`regional_constraints`.
    """
    if block_reduction_col is None:
        mask = None
    elif block_reduction_col not in constraints_df.columns:
        msg = (
            f"`block_reduction_col` '{block_reduction_col}' not in `constraints_df` "
            "columns"
        )
        raise ValueError(msg)
    else:
        mask = constraints_df[block_reduction_col].to_numpy(dtype=bool)

    data: typing.Any = constraints_df.sampled_grav
    if with_height:
        data = (data, constraints_df.sampled_grav_height)

    coords, data, weights = utils._block_reduce_points(  # pylint: disable=protected-access
        (constraints_df[coord_names[0]], constraints_df[coord_names[1]]),
        data,
        weights,
        block_size,
        block_reduction,
        region,
        mask=mask,
        # weights are 1 over the data uncertainty squared
        uncertainty=weights is not None,
        # use the mean coordinates of the points in each block
        center_coordinates=False,
    )

    cols = {
        coord_names[0]: coords[0],
        coord_names[1]: coords[1],
    }
    if with_height:
        cols["sampled_grav"], cols["sampled_grav_height"] = data
    else:
        cols["sampled_grav"] = data
    if (weights_col is not None) and (weights is not None):
        cols[weights_col] = weights

    constraints_df = pd.DataFrame(data=cols).dropna(how="any")

    # rows may have been dropped, so take the weights from the dataframe
    if (weights_col is not None) and (weights_col in constraints_df.columns):
        weights = constraints_df[weights_col]
    else:
        weights = None

    return constraints_df, weights


def regional_constraints(
    grav_ds: xr.Dataset,
    constraints_df: pd.DataFrame,
    grid_method: str = "eq_sources",
    block_size: float | None = None,
    block_reduction: str = "mean",
    block_reduction_col: str | None = None,
    weights_col: str | None = None,
    grid_points_col: str | None = None,
    tension: float = 1,
    dampings: float | list[float] | None = None,
    maxdist: float | None = None,
    depth: float | str | None = None,
    damping: float | None = None,
    cv: bool = False,
    eq_source_block_size: float | None = None,
    grav_obs_height: float | None = None,
    cv_kwargs: dict[str, typing.Any] | None = None,
    regional_shift: float = 0,
    mask_column: str | None = None,
    reverse_regional_residual: bool = False,
    constraints_block_size: float | None = None,
    constraints_weights_column: str | None = None,
    tension_factor: float | None = None,
    spline_dampings: float | list[float] | None = None,
) -> None:
    """
    Separate the regional field by sampling and re-gridding the gravity misfit at
    points of known topography (constraint points). The re-gridding can be done with:
    1. Tensioned minimum curvature with PyGMT, using `grid_method` "pygmt", 2.
    Bi-Harmonic splines with Verde, using `grid_method` "splines", or 3. Equivalent
    Sources with Harmonica, using `grid_method` "eq_sources". Optionally, a dc-shift can
    be added to the calculated regional with `regional_shift`.

    If some of the constraint points fall on the gravity grid's nodes, indicated by a
    boolean column named by `grid_points_col`, the misfit sampled at those points is
    used directly as the regional field there, and only the remaining cells are
    interpolated, with :func:`invert4geom.fill_missing`, from the other constraint
    points. This makes the residual field exactly zero at those points, instead of
    approximately zero as with an interpolation of all the points. In this case, the
    additional `grid_method` values "nearest", "linear" and "cubic" are also available.

    If there are many constraint points, they can be block-reduced before gridding with
    `block_size`, optionally weighted with `weights_col`, and optionally limited to a
    subset of the points with `block_reduction_col`. This mirrors the block reduction of
    :func:`invert4geom.create_topography`.

    Parameters
    ----------
    grav_ds : xarray.Dataset
        gravity data with columns "easting", "northing", "gravity_anomaly", and
        "forward_gravity".
    constraints_df : pandas.DataFrame
        dataframe of constraints with columns "easting", "northing", and "upward".
    grid_method : str, optional
        method used to grid the sampled gravity data at the constraint points. Choose
        between "splines", "pygmt", or "eq_sources", or, if `grid_points_col` is
        provided, "nearest", "linear" or "cubic", by default "eq_sources"
    block_size : float | None, optional
        size of block used in a block reduction of the constraint points before
        gridding, by default None
    block_reduction : str, optional
        type of block reduction to apply. If "mean", :class:`verde.BlockMean` is used
        and the reduced weights are kept for the gridding. If "median",
        :class:`verde.BlockReduce` is used and any weights are ignored and dropped, by
        default "mean"
    block_reduction_col : str | None, optional
        name of a boolean column of `constraints_df`, where True indicates points which
        should be block-reduced. Points which are False are left unaltered and added
        back to the block-reduced points. If None, all points are block-reduced, by
        default None
    weights_col : str | None, optional
       column name for weighting values of each constraint point. Used if `block_size`
       is not None or if `grid_method` is "splines" or "eq_sources", by default None
    grid_points_col : str | None, optional
        name of a boolean column of `constraints_df`, where True indicates points which
        fall on the nodes of the gravity grid. The misfit at these points is used
        directly as the regional field, and only the remaining cells are interpolated
        from the other constraint points. Not available if `grid_method` is
        "eq_sources", by default None
    tension : float, optional
        tension factor used if `grid_method` is "pygmt", by default 1
    dampings : float | list[float] | None, optional
        damping values used if `grid_method` is "splines". If more than one is
        provided, the best is chosen by cross-validation, by default None
    maxdist : float | None, optional
        if `grid_points_col` is provided, only interpolate the remaining cells with
        constraint points within this distance of them, by default None
    depth : float | str | None, optional
        depth of each source relative to the data elevation, positive downwards in
        meters, by default is 4.5 times the mean nearest-neighbor
        distance
    damping : float | None, optional
        damping values used if `grid_method` is "eq_sources", by default None
    cv : bool, optional
        use cross-validation to find the best equivalent source parameters, by
        default False, provide dictionary `cv_kwargs` which is passed to
        `optimization.optimize_eq_source_params` and can contain: "n_trials",
        "damping_limits", "depth_limits", "block_size_limits", and "progressbar".
    eq_source_block_size : float | None, optional
        block size for defining equivalent sources used if `grid_method` is
        "eq_sources". This block reduction is done after the block reduction of the
        constraint points set by `block_size`, by default None
    grav_obs_height : float, optional
        Observation height to use for predictions if `grid_method` is "eq_sources". Note
        that this is separate from the observation height of the gravity data used in
        the fitting of equivalent sources. Those heights are determined from the
        observation heights in the gravity dataset.
        by default None
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
    reverse_regional_residual : bool, optional
        if True, reverse the regional and residual fields after calculation, by default
        False
    constraints_block_size : float | None, optional
        deprecated, use `block_size` instead, by default None
    constraints_weights_column : str | None, optional
        deprecated, use `weights_col` instead, by default None
    tension_factor : float | None, optional
        deprecated, use `tension` instead, by default None
    spline_dampings : float | list[float] | None, optional
        deprecated, use `dampings` instead, by default None
    """

    _initialize_separation(grav_ds, "regional_constraints")
    _check_projected_units(grav_ds, "regional_constraints")

    if constraints_df is None:
        msg = "need to provide constraints_df"
        raise ValueError(msg)

    # map the deprecated parameter names onto the new ones
    if (
        (grid_method == "eq_sources")
        and (block_size is not None)
        and (eq_source_block_size is None)
        and (constraints_block_size is None)
    ):
        msg = (
            "`block_size` now block-reduces the constraint points before fitting the "
            "equivalent sources; use `eq_source_block_size` to set the block size of "
            "the sources themselves"
        )
        logger.warning(msg)
    block_size = utils._deprecated_parameter(  # pylint: disable=protected-access
        "constraints_block_size", constraints_block_size, "block_size", block_size
    )
    weights_col = utils._deprecated_parameter(  # pylint: disable=protected-access
        "constraints_weights_column",
        constraints_weights_column,
        "weights_col",
        weights_col,
    )
    tension = utils._deprecated_parameter(  # pylint: disable=protected-access
        "tension_factor", tension_factor, "tension", tension
    )
    dampings = utils._deprecated_parameter(  # pylint: disable=protected-access
        "spline_dampings", spline_dampings, "dampings", dampings
    )
    if grid_method == "verde":
        warnings.warn(
            "`grid_method` value 'verde' is deprecated, please use 'splines' instead.",
            UserWarning,
            stacklevel=2,
        )
        grid_method = "splines"

    if grid_method not in _CONSTRAINT_GRID_METHODS:
        msg = (
            "invalid string for grid_method, choose from "
            f"{sorted(_CONSTRAINT_GRID_METHODS)}"
        )
        raise ValueError(msg)

    # a single damping value is still passed as a list to the spline cross-validation
    if isinstance(dampings, float | int):
        dampings = [float(dampings)]

    # warn about unused args
    if (grid_method != "pygmt") & (tension != 1):
        msg = "`tension` only used if `grid_method` is 'pygmt'"
        logger.warning(msg)
    if (grid_method != "splines") & (dampings is not None):
        msg = "`dampings` only used if `grid_method` is 'splines'"
        logger.warning(msg)
    if grid_points_col is None:
        if grid_method in _FILL_ONLY_GRID_METHODS:
            msg = (
                f"`grid_method` '{grid_method}' can only be used if `grid_points_col` "
                "is provided"
            )
            raise ValueError(msg)
        if maxdist is not None:
            msg = "`maxdist` only used if `grid_points_col` is provided"
            logger.warning(msg)
    else:
        if grid_method == "eq_sources":
            msg = "`grid_points_col` can't be used if `grid_method` is 'eq_sources'"
            raise ValueError(msg)
        if grid_points_col not in constraints_df.columns:
            msg = f"`grid_points_col` '{grid_points_col}' not in `constraints_df`"
            raise ValueError(msg)

    if grid_method != "eq_sources":
        if depth is not None:
            msg = "`depth` only used if `grid_method` is 'eq_sources'"
            logger.warning(msg)
        if damping is not None:
            msg = "`damping` only used if `grid_method` is 'eq_sources'"
            logger.warning(msg)
        if cv is True:
            msg = "`cv` only used if `grid_method` is 'eq_sources'"
            logger.warning(msg)
        if eq_source_block_size is not None:
            msg = "`eq_source_block_size` only used if `grid_method` is 'eq_sources'"
            logger.warning(msg)
        if grav_obs_height is not None:
            msg = "`grav_obs_height` only used if `grid_method` is 'eq_sources'"
            logger.warning(msg)
        if cv_kwargs is not None:
            msg = "`cv_kwargs` only used if `grid_method` is 'eq_sources'"
            logger.warning(msg)

    constraints_df = constraints_df.copy()

    coord_names = grav_ds.coord_names

    # sample gravity at constraint points
    constraints_df = utils.sample_grids(
        df=constraints_df,
        grid=grav_ds.misfit,
        sampled_name="sampled_grav",
        coord_names=coord_names,
    )

    # drop rows with NaN values
    constraints_df = constraints_df[constraints_df.sampled_grav.notna()]

    # split off the points which fall on the gravity grid's nodes; the misfit at these
    # points is used as-is, so only the remaining points are gridded
    grid_constraints = None
    if grid_points_col is not None:
        on_grid = constraints_df[grid_points_col].to_numpy(dtype=bool)
        grid_constraints = constraints_df[on_grid]
        constraints_df = constraints_df[~on_grid]
        if grid_constraints.empty:
            msg = f"no constraint points have `{grid_points_col}` set to True"
            raise ValueError(msg)
        logger.info(
            "using %s constraint points as the regional field directly, and %s points "
            "to interpolate the remaining cells",
            len(grid_constraints),
            len(constraints_df),
        )

    if grid_method == "eq_sources":
        # sample gravity observation height at constraint points
        constraints_df = utils.sample_grids(
            df=constraints_df,
            grid=grav_ds.upward,
            sampled_name="sampled_grav_height",
            coord_names=coord_names,
        )
        # drop rows with NaN values
        constraints_df = constraints_df[constraints_df.sampled_grav_height.notna()]

    # get weights for each constraint point
    weights = None if weights_col is None else constraints_df[weights_col]

    # reduce the constraint points to one value per block
    if (block_size is not None) & (not constraints_df.empty):
        constraints_df, weights = _block_reduce_constraints(
            constraints_df,
            weights,
            weights_col,
            coord_names,
            grav_ds.region,
            block_size,
            block_reduction,
            block_reduction_col,
            with_height=grid_method == "eq_sources",
        )
    ###
    ###
    # Interpolation of only the cells without a constraint point
    ###
    ###
    # the misfit at the points which fall on the grid's nodes is used as-is, and the
    # remaining cells are interpolated from the other constraint points
    if grid_constraints is not None:
        grav_ds["reg"] = utils.fill_missing(
            _sampled_misfit_grid(grav_ds, grid_constraints, coord_names),
            method=grid_method,
            constraints_df=None if constraints_df.empty else constraints_df,
            data_col="sampled_grav",
            tension=tension,
            dampings=dampings,
            maxdist=maxdist,
        )
    ###
    ###
    # Tensioned splines with PyGMT
    ###
    ###
    # grid the entire regional gravity based just on the values at the constraints
    elif grid_method == "pygmt":
        pygmt = utils._import_pygmt()  # pylint: disable=protected-access
        registration = maps.get_grid_info(grav_ds.forward_gravity)[-1]
        grav_ds["reg"] = pygmt.surface(
            data=constraints_df[[coord_names[0], coord_names[1], "sampled_grav"]],
            region=grav_ds.region,
            spacing=grav_ds.spacing,
            registration=registration,
            tension=tension,
            verbose="quiet",
        ).rename({"x": coord_names[0], "y": coord_names[1]})
    ###
    ###
    # Bi-Harmonic splines with Verde
    ###
    ###
    elif grid_method == "splines":
        spline = utils._fit_spline_cv(  # pylint: disable=protected-access
            (
                constraints_df[coord_names[0]],
                constraints_df[coord_names[1]],
            ),
            constraints_df.sampled_grav,
            weights,
            dampings,
        )
        # predict fitted grid at gravity points
        grav_ds["reg"] = spline.grid(
            coordinates=(
                grav_ds[coord_names[0]].to_numpy(),
                grav_ds[coord_names[1]].to_numpy(),
            ),
            data_names=["reg"],
            dims=(coord_names[1], coord_names[0]),
        ).reg
    ###
    ###
    # Equivalent Sources with Harmonica
    ###
    ###
    elif grid_method == "eq_sources":
        coords = (
            constraints_df[coord_names[0]],
            constraints_df[coord_names[1]],
            constraints_df.sampled_grav_height,
        )
        data = constraints_df.sampled_grav
        if depth is None:
            depth = "default"
        if depth == "default":
            depth = 4.5 * np.mean(
                bd.neighbor_distance_statistics(
                    (coords[0], coords[1]),
                    "median",
                    k=1,
                )
            )

        if cv is True:
            # eqs = utils.best_equivalent_source_damping(
            try:
                _, eqs = optimization.optimize_eq_source_params(
                    coordinates=coords,
                    data=data,
                    # kwargs
                    weights=weights,
                    depth=depth,
                    damping=damping,
                    block_size=eq_source_block_size,
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
                    block_size=eq_source_block_size,
                )
                eqs.fit(
                    coords,
                    data,
                    weights=weights,
                )

        else:
            # create set of deep sources
            eqs = hm.EquivalentSources(
                depth=depth,
                damping=damping,
                block_size=eq_source_block_size,
            )

            # fit the source coefficients to the data
            eqs.fit(
                coords,
                data,
                weights=weights,
            )
        msg = "depth: %s, damping: %s"
        logger.debug(msg, eqs.depth, eqs.damping)

        # predict sources at gravity points and chosen height for upward continuation
        df = grav_ds.inv.df
        if grav_obs_height is None:
            grav_obs_height = df.upward
        else:
            grav_obs_height = np.ones_like(df[coord_names[0]].values) * grav_obs_height
        coords = (df[coord_names[0]], df[coord_names[1]], grav_obs_height)

        df["reg"] = eqs.predict(coords)
        grav_ds["reg"] = df.set_index([coord_names[1], coord_names[0]]).to_xarray().reg

    _finalize_separation(
        grav_ds, regional_shift, mask_column, reverse_regional_residual
    )


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

    _check_projected_units(grav_ds, "regional_constraints_cv")

    utils._check_constraints_inside_gravity_region(  # pylint: disable=protected-access
        constraints_df,
        grav_ds,
    )

    df = constraints_df.copy()
    df = df[df.columns.drop(list(df.filter(regex="fold_")))]

    if split_kwargs is None:
        msg = "need to provide split_kwargs"
        raise ValueError(msg)

    testing_training_df = cross_validation.split_test_train(
        df,
        coord_names=grav_ds.coord_names,
        **split_kwargs,
    )

    # this updates grav_ds in place with the best regional separation; the returned
    # dataset is the same object so the return values are not needed here
    optimization.optimize_regional_constraint_point_minimization(
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
