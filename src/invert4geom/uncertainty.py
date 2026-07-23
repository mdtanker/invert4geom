import copy  # pylint: disable=too-many-lines
import logging
import pathlib
import pickle
import random
import typing

import harmonica as hm
import numpy as np
import pandas as pd
import polartoolkit as ptk
import scipy as sp
import verde as vd
import xarray as xr
from numpy.typing import NDArray
from tqdm.autonotebook import tqdm

from invert4geom import inversion, logger, plotting, utils

if typing.TYPE_CHECKING:
    from invert4geom.inversion import Inversion


def _mean_and_stdev(stats_ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """
    return the weighted mean and standard deviation of an ensemble statistics
    dataset if present, otherwise the unweighted versions
    """
    if "weighted_mean" in stats_ds:
        return stats_ds.weighted_mean, stats_ds.weighted_stdev
    return stats_ds.z_mean, stats_ds.z_stdev


def create_lhc(
    n_samples: int,
    parameter_dict: dict[str, dict[str, typing.Any]],
    random_state: int = 1,
    criterion: str = "centered",
) -> dict[str, dict[str, typing.Any]]:
    """
    Given some parameter values and their expected distributions, create a Latin
    Hypercube with a given number of samples.

    Parameters
    ----------
    n_samples : int
        how many samples to make for each parameter
    parameter_dict : dict
        nested dictionary, with a dictionary of 'distribution', 'loc', 'scale' and
        optionally 'log' for each parameter to be sampled. Distributions can be
        'uniform' or 'normal'. For 'uniform', 'loc' is the lower bound and 'scale' is
        the range of the distribution. 'loc' + 'scale' = upper bound. For 'normal',
        'loc' is the center (mean) of the distribution and 'scale' is the standard
        deviation. If 'log' is True, the provided 'loc' and 'scale' values are the base
        10 exponents. For example, a uniform distribution with loc=-4, scale=6 and
        log=True would sample values between 1e-4 and 1e2.
        An optional 'clip' entry of (min, max) bounds the sampled values (applied
        after any log transform, in real units) - use it to keep physically-positive
        parameters positive, e.g. a normal distribution for an equivalent-source depth.
    random_state : int, optional
        random state to use for sampling, by default 1
    criterion : str, optional
        criterion to use for sampling, by default "centered", options are "centered",
        "random", "maximin", or "mincorrelation". These control how the Latin
        Hypercube's unit-interval strata are sampled via
        :class:`scipy.stats.qmc.LatinHypercube`: "centered" takes the midpoint of each
        stratum, "random" takes a uniformly random point within each stratum, and
        "maximin" / "mincorrelation" both use scipy's "random-cd" optimization, which
        iteratively improves the sample's space-filling properties.
    Returns
    -------
    dict[dict[typing.Any]]
        nested dictionary with parameter names, distribution specifics, and sampled
        values
    """
    param_dict = copy.deepcopy(parameter_dict)

    if criterion == "centered":
        scramble = False
        optimization = None
    elif criterion == "random":
        scramble = True
        optimization = None
    elif criterion in ("maximin", "mincorrelation"):
        scramble = True
        optimization = "random-cd"
    else:
        msg = f"Unknown criterion type: {criterion}"
        raise ValueError(msg)

    # sample the unit hypercube, 1 dimension per parameter
    sampler = sp.stats.qmc.LatinHypercube(
        d=len(param_dict),
        scramble=scramble,
        optimization=optimization,
        seed=random_state,
    )
    unit_samples = sampler.random(n=n_samples)

    # transform unit samples into each parameter's distribution via its inverse CDF
    for j, (k, v) in enumerate(param_dict.items()):
        unit_values = unit_samples[:, j]
        if v["distribution"] == "uniform":
            values = sp.stats.uniform(loc=v["loc"], scale=v["scale"]).ppf(unit_values)
        elif v["distribution"] == "normal":
            values = sp.stats.norm(loc=v["loc"], scale=v["scale"]).ppf(unit_values)
        elif v["distribution"] == "uniform_discrete":
            values = sp.stats.randint(low=v["loc"], high=v["loc"] + v["scale"] + 1).ppf(
                unit_values
            )
        else:
            msg = f"Unknown distribution type: {v['distribution']}"
            raise ValueError(msg)
        values = np.asarray(values, dtype=float)
        if v.get("norm_limits", None) is not None:
            norm_limits = v["norm_limits"]
            values = utils.normalize(
                values,
                low=norm_limits[0],
                high=norm_limits[1],
            )
        if v.get("log", False) is True:
            values = 10**values
        if v.get("clip", None) is not None:
            values = np.clip(values, v["clip"][0], v["clip"][1])
        if v.get("dtype", None) is int:
            values = values.round().astype(int)
        v["sampled_values"] = values

        logger.info(
            "Sampled '%s' parameter values; mean: %s, min: %s, max: %s",
            k,
            v["sampled_values"].mean(),
            v["sampled_values"].min(),
            v["sampled_values"].max(),
        )

        # ax = pd.Series(v["sampled_values"]).hist()
        # ax = np.log(pd.Series(v["sampled_values"])).plot.hist(bins=8)
        # import matplotlib.pyplot as plt
        # plt.show()

    return param_dict


def randomly_sample_data(
    seed: int,
    data_df: pd.DataFrame,
    data_col: str,
    uncert_col: str,
) -> pd.DataFrame:
    """
    Given a dataframe with a data column and an uncertainty column, sample the data with
    a normal distribution within the uncertainty range. Note that this overwrites the
    data column with the newly sampled data.

    Parameters
    ----------
    seed : int
        random number generator seed
    data_df : pandas.DataFrame
        dataframe with columns `data_col` and `uncert_col`
    data_col : str
        name of data column to sample
    uncert_col : str
        name of uncertainty column to sample within

    Returns
    -------
    pandas.DataFrame
        dataframe with data column updated with sampled values
    """
    # create random generator
    rand = np.random.default_rng(seed=seed)

    # sample data within uncertainty range with normal distribution
    sampled_data = data_df.copy()
    sampled_data[data_col] = rand.normal(
        sampled_data[data_col],
        sampled_data[uncert_col],
    )

    return sampled_data


def starting_topography_uncertainty(
    runs: int,
    sample_constraints: bool = False,
    parameter_dict: dict[str, typing.Any] | None = None,
    plot: bool = True,
    plot_region: tuple[float, float, float, float] | None = None,
    true_topography: xr.DataArray | None = None,
    coast: bool = False,
    coord_names: tuple[str, str] = ("easting", "northing"),
    **kwargs: typing.Any,
) -> tuple[xr.Dataset, dict[str, typing.Any]]:
    """
    Create a stochastic ensemble of starting topographies by sampling the constraints or
    parameters within their respective distributions and find the cell-wise (weighted)
    statistics of the ensemble.

    Parameters
    ----------
    runs : int
        number of runs to perform
    sample_constraints : bool, optional
        choose to sample the constraints from a normal distribution with a mean of each
        constraints depth and a standard deviation set by the `uncert` column, by
        default False
    parameter_dict : dict[str, typing.Any] | None, optional
        dictionary of parameters passes to `create_topography` with the uncertainty
        distributions defined, by default None
    plot : bool, optional
        show the results, by default True
    plot_region : tuple[float, float, float, float] | None, optional
        clip the plot to a region, by default None
    true_topography : xarray.DataArray | None, optional
        if the true topography is known, will make a plot comparing the results, by
        default None
    coast : bool, optional
        whether to plot coastlines, by default False
    coord_names : tuple[str, str], optional
        names of the coordinate columns in the constraints dataframe, by default
        ("easting", "northing")

    Returns
    -------
    stats_ds: xarray.Dataset
        a dataset with the cell-wise statistics of the ensemble of topographies.
    sampled_param_dict : dict[str, typing.Any]
        dictionary of sampled parameter values.
    """
    new_kwargs = copy.deepcopy(kwargs)
    constraints_df = new_kwargs.pop("constraints_df", None)

    if constraints_df is None:
        msg = "constraints_df must be provided"
        raise ValueError(msg)

    if parameter_dict is not None:
        sampled_param_dict = create_lhc(
            n_samples=runs,
            parameter_dict=parameter_dict,
            random_state=0,
            criterion="centered",
        )
    else:
        sampled_param_dict = None

    sampled_constraints = copy.deepcopy(constraints_df)

    topos = []
    weight_vals = []
    for i in tqdm(range(runs), desc="starting topography ensemble"):
        # create random generator
        rand = np.random.default_rng(seed=i)

        if sample_constraints:
            # always sample from the original constraint values, not the previously
            # sampled values, to avoid a random walk of accumulating noise
            sampled_constraints["upward"] = rand.normal(
                constraints_df.upward, constraints_df.uncert
            )

        if sampled_param_dict is not None:
            for k, v in sampled_param_dict.items():
                new_kwargs[k] = v["sampled_values"][i]

        with utils._log_level(logging.WARNING):  # pylint: disable=protected-access
            starting_topography = utils.create_topography(
                constraints_df=sampled_constraints,
                **new_kwargs,
            ).upward
        topos.append(starting_topography)

        # sample the topography at the constraint points
        sampled_constraints = utils.sample_grids(
            df=sampled_constraints,
            grid=starting_topography,
            sampled_name="sampled",
            coord_names=coord_names,
        )
        # get weights of rmse between constraints and results
        weight_vals.append(
            utils.rmse(sampled_constraints.upward - sampled_constraints.sampled)
        )

    # convert residuals into weights
    weights = [1 / (x**2) for x in weight_vals]

    # merge all topos into 1 dataset
    merged = merge_simulation_results(topos)
    # pylint: disable=duplicate-code
    # get stats and weighted stats on the merged dataset
    stats_ds = model_ensemble_stats(
        merged,
        weights=weights,
    )
    if plot is True:
        epsg, coast = utils.get_epsg(coast=coast)
        try:
            plotting.plot_stochastic_results(
                stats_ds=stats_ds,
                points=sampled_constraints,
                cmap="rain",
                reverse_cpt=True,
                label="topography",
                points_label="Topography constraints",
                region=plot_region,
            )
            if true_topography is not None:
                mean, stdev = _mean_and_stdev(stats_ds)

                _ = ptk.grid_compare(
                    np.abs(true_topography - mean),
                    stdev,
                    fig_height=12,
                    region=plot_region,
                    grid1_name="True error",
                    grid2_name="Stochastic uncertainty",
                    robust=True,
                    hist=True,
                    inset=False,
                    verbose="q",
                    title="difference",
                    coast=coast,
                    cmap="thermal",
                    points=sampled_constraints.rename(
                        columns={"easting": "x", "northing": "y"}
                    ),
                    points_style="x.3c",
                    epsg=epsg,
                )
                _ = ptk.grid_compare(
                    true_topography,
                    mean,
                    fig_height=12,
                    region=plot_region,
                    grid1_name="True topography",
                    grid2_name="Mean topography",
                    robust=True,
                    hist=True,
                    inset=False,
                    verbose="q",
                    title="difference",
                    coast=coast,
                    cmap="rain",
                    reverse_cpt=True,
                    points=sampled_constraints.rename(
                        columns={"easting": "x", "northing": "y"}
                    ),
                    points_style="x.3c",
                    epsg=epsg,
                )
        except Exception as e:  # pylint: disable=broad-exception-caught # noqa: BLE001
            logger.error("plotting failed with error: %s", e)

    return stats_ds, sampled_param_dict  # type: ignore[return-value]
    # pylint: enable=duplicate-code


"""
CHANGES
1. sample_data / data_uncertainty: the data are resampled within their measurement
   uncertainty each run. Before, only the fit parameters varied, so observational noise
   never entered the ensemble.
2. demean: the data mean is removed before fitting and added back after prediction.
   harmonica's EquivalentSources doesn't do this, so damping shrinks predictions toward
   0 mGal. With anomalies around -200 mGal, varying damping without demeaning produces a
   huge spread that is just the datum offset, not real uncertainty.
3. weight_by="cv_rmse": members can be weighted by blocked cross-validation RMSE
   (with the new blocked_cv_rmse helper). The old "score" option used the training R2,
   which is ~1 for equivalent sources no matter the parameters, so it weights nothing.
4. grid_points is no longer mutated in place, and one random_state seeds both the Latin
   hypercube and the per-run noise.
"""


def blocked_cv_rmse(
    eqs: typing.Any,
    coords: tuple[NDArray, NDArray, NDArray],
    data: NDArray,
    weights: NDArray | None = None,
    spacing: float | None = None,
    n_splits: int = 5,
    random_state: int = 0,
    cv: typing.Any | None = None,
) -> float:
    """
    RMSE of held-out predictions from spatially blocked K-fold cross-validation.

    Unlike the training score, this measures how well a gridder predicts data it was not
    fit to. Blocked (rather than random) folds avoid rewarding overfit models via
    spatially correlated neighboring points.

    Parameters
    ----------
    eqs : harmonica.EquivalentSources
        an unfitted gridder instance to cross-validate
    coords : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        coordinates of the data points in the order (easting, northing, upward)
    data : numpy.ndarray
        data values to fit and predict
    weights : numpy.ndarray | None, optional
        fit weights for the data points, by default None
    spacing : float | None, optional
        size of the spatial blocks passed to :class:`verde.BlockKFold`, by default None
    n_splits : int, optional
        number of folds, by default 5
    random_state : int, optional
        random state for shuffling the blocks, by default 0

    Returns
    -------
    float
        RMSE of the held-out predictions, pooled over all folds
    """
    if cv is None:
        cv = vd.BlockKFold(
            spacing=spacing,
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    residuals = []
    for train, test in cv.split(np.transpose(coords[:2])):
        model = copy.deepcopy(eqs)
        model.fit(
            tuple(c[train] for c in coords),
            data[train],
            weights=None if weights is None else weights[train],
        )
        residuals.append(data[test] - model.predict(tuple(c[test] for c in coords)))
    return float(np.sqrt(np.mean(np.concatenate(residuals) ** 2)))


def equivalent_sources_uncertainty(
    runs: int,
    data: NDArray,
    coords: tuple[NDArray, NDArray, NDArray],
    grid_points: pd.DataFrame,
    parameter_dict: dict[str, typing.Any] | None = None,
    region: tuple[float, float, float, float] | None = None,
    plot: bool = True,
    plot_region: tuple[float, float, float, float] | None = None,
    true_gravity: xr.DataArray | None = None,
    deterministic_error: xr.DataArray | None = None,
    weight_by: str | None = None,
    coast: bool = False,
    data_uncertainty: NDArray | None = None,
    sample_data: bool = False,
    demean: bool = True,
    source_surface: NDArray | float | None = None,
    cv: typing.Any | None = None,
    cv_spacing: float | None = None,
    cv_n_splits: int = 5,
    random_state: int = 0,
    criterion: str = "centered",
    **kwargs: typing.Any,
) -> tuple[xr.Dataset, dict[str, typing.Any]]:
    """
    Create a stochastic ensemble of interpolated gravity grids by sampling the gravity
    data and/or the equivalent source interpolation parameters within their respective
    distributions and calculate the cell-wise (weighted) statistics of the ensemble.

    Parameters
    ----------
    runs : int
        number of runs to perform
    data : numpy.ndarray
        The gravity data to fit the equivalent sources to.
    coords: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        The coordinates of the gravity data points in the order (easting, northing,
        upward).
    parameter_dict : dict[str, typing.Any] | None, optional
        dictionary of parameters passed to `harmonica.EquivalentSources` with the
        uncertainty distributions defined, by default None
    region: tuple[float, float, float, float] | None = None,
        region to calculate statistics within, by default None
    plot : bool, optional
        show the results, by default True
    plot_region : tuple[float, float, float, float] | None, optional
        clip the plot to a region, by default None
    true_gravity : xarray.DataArray | None, optional
        if the true gravity is known, will make a plot comparing the results, by
        default None
    deterministic_error : xarray.DataArray | None, optional
        if the deterministic error is known, will make a plot comparing the results, by
        default None
    weight_by : str | None, optional
        how to weight the ensemble members, by default None (equal weights). Options:
        "cv_rmse" weights by 1/RMSE² from spatially blocked K-fold cross-validation of
        each member (requires `cv_spacing`); "rmse" weights by 1/RMSE² between a
        `gravity_anomaly` column of `grid_points` and each predicted grid; "score"
        weights by the training R² (note equivalent sources fit their training data
        almost exactly regardless of parameters, so training R² is ~1 for every member
        and weights little — prefer "cv_rmse").
    coast : bool, optional
        whether to plot coastlines, by default False
    data_uncertainty : numpy.ndarray | None, optional
        per-point 1σ uncertainty of `data`, required if `sample_data` is True, by
        default None
    sample_data : bool, optional
        resample the data each run from a normal distribution with a standard deviation
        of `data_uncertainty`, so observational noise is included in the ensemble
        spread, by default False
    demean : bool, optional
        remove the data mean before fitting and restore it after prediction, by default
        True. `harmonica.EquivalentSources` does not do this internally, so damping
        shrinks predictions toward 0; for data with a large offset (e.g. Bouguer
        anomalies of ~-200 mGal) varying the damping without demeaning produces a large
        ensemble spread which reflects the offset, not interpolation uncertainty.
    source_surface : numpy.ndarray | float | None, optional
        reference elevation(s) the sources hang below instead of the observation
        points. By default (None) harmonica places each source `depth` meters below
        its own observation, so the source layer copies the station topography -
        rough terrain then imprints artifacts. Provide a scalar (flat source layer)
        or a per-station array (e.g. a low-passed topography sampled at the
        stations) and each run's sources are built as
        ``(easting, northing, source_surface - depth)`` and passed via ``points``,
        keeping the sampled ``depth`` parameter meaningful.
    cv : typing.Any | None, optional
        a cross-validation splitter for `weight_by="cv_rmse"` (any object with
        sklearn's split interface), overriding `cv_spacing`/`cv_n_splits`. Use e.g. a
        region-scored splitter so members are weighted by their skill inside the area
        of interest, matching an AOI-scored parameter optimization. By default None
    cv_spacing : float | None, optional
        block size (in meters) for `weight_by="cv_rmse"`, by default None
    cv_n_splits : int, optional
        number of folds for `weight_by="cv_rmse"`, by default 5
    random_state : int, optional
        seeds both the Latin Hypercube sampling and the per-run data noise, by default 0
    criterion : str, optional
        Latin Hypercube criterion passed to `create_lhc`, by default "centered"

    Returns
    -------
    stats_ds: xarray.Dataset,
        a dataset with the cell-wise statistics of the ensemble of predicted gravity.
        If `weight_by="cv_rmse"`, the per-run RMSE values are stored in
        `stats_ds.attrs["cv_rmse"]`.
    sampled_parms_dict: dict[str, typing.Any]
        a dictionary of sampled parameter values.
    """
    new_kwargs = copy.deepcopy(kwargs)

    if sample_data and data_uncertainty is None:
        msg = "data_uncertainty must be provided when sample_data=True"
        raise ValueError(msg)
    if weight_by not in (None, "score", "rmse", "cv_rmse"):
        msg = f"unknown weight_by: {weight_by!r}"
        raise ValueError(msg)
    if weight_by == "cv_rmse" and cv_spacing is None and cv is None:
        msg = "cv_spacing or cv must be provided when weight_by='cv_rmse'"
        raise ValueError(msg)

    data = np.asarray(data, dtype=float)
    coords = tuple(np.asarray(c, dtype=float) for c in coords)  # type: ignore[assignment]
    if data_uncertainty is not None:
        data_uncertainty = np.asarray(data_uncertainty, dtype=float)

    if parameter_dict is not None:
        sampled_param_dict = create_lhc(
            n_samples=runs,
            parameter_dict=parameter_dict,
            random_state=random_state,
            criterion=criterion,
        )
    else:
        sampled_param_dict = None

    # sources at or above the observations produce singular predictions, so refuse
    # non-positive depths outright (a normal distribution with a wide scale can sample
    # them silently - use a log-space distribution or a 'clip' bound instead)
    sampled_depths = (
        np.asarray(sampled_param_dict["depth"]["sampled_values"])
        if sampled_param_dict is not None and "depth" in sampled_param_dict
        else None
    )
    fixed_depth = new_kwargs.get("depth")
    if (sampled_depths is not None and np.any(sampled_depths <= 0)) or (
        isinstance(fixed_depth, (int, float)) and fixed_depth <= 0
    ):
        msg = (
            "equivalent-source depths must be positive; the sampled or provided values "
            "include depths <= 0 (sources at or above the observations). Use a "
            "log-space distribution or a 'clip' bound in the parameter_dict."
        )
        raise ValueError(msg)

    # weights are passed to fit and score, not the EquivalentSources constructor
    data_weights = new_kwargs.pop("weights", None)

    grav_grids = []
    scores = []
    cv_rmses = []
    for i in tqdm(range(runs), desc="equivalent sources ensemble"):
        run_kwargs = dict(new_kwargs)
        if sampled_param_dict is not None:
            for k, v in sampled_param_dict.items():
                run_kwargs[k] = v["sampled_values"][i]

        run_data = data
        if sample_data:
            # each run gets an independent, reproducible noise realization
            rng = np.random.default_rng(seed=[random_state, i])
            run_data = rng.normal(data, data_uncertainty)

        if source_surface is not None:
            # hang the sources below the reference surface instead of the stations,
            # preserving the sampled depth via the points argument
            run_depth = run_kwargs.pop("depth", None)
            if run_depth is None:
                msg = "source_surface requires a depth (sampled or fixed)"
                raise ValueError(msg)
            source_elev = (
                np.broadcast_to(
                    np.asarray(source_surface, dtype=float), coords[0].shape
                )
                - run_depth
            )
            if np.any(source_elev >= coords[2]):
                msg = (
                    "sources at or above their observation points; increase depth or "
                    "lower the source_surface"
                )
                raise ValueError(msg)
            run_kwargs["points"] = (coords[0], coords[1], source_elev)

        shift = run_data.mean() if demean else 0.0

        with utils._log_level(logging.WARNING):  # pylint: disable=protected-access
            eqs = hm.EquivalentSources(
                **run_kwargs,
            )
            eqs.fit(coords, run_data - shift, weights=data_weights)

        if weight_by == "score":
            scores.append(eqs.score(coords, run_data - shift, weights=data_weights))
        elif weight_by == "cv_rmse":
            cv_rmses.append(
                blocked_cv_rmse(
                    hm.EquivalentSources(**run_kwargs),
                    coords,
                    run_data - shift,
                    weights=data_weights,
                    spacing=cv_spacing,
                    n_splits=cv_n_splits,
                    random_state=random_state,
                    cv=cv,
                )
            )

        # predict sources onto grid, restoring the mean
        predicted = shift + eqs.predict(
            (
                grid_points.easting,
                grid_points.northing,
                grid_points.upward,
            ),
        )

        grav_grids.append(
            grid_points.assign(predicted_grav=predicted)
            .set_index(["northing", "easting"])
            .to_xarray()
            .predicted_grav
        )

    # merge all grids into 1 dataset
    merged = merge_simulation_results(grav_grids)

    # get weights for each ensemble member
    if weight_by == "score":
        # scores are R² values where higher is better, so use them directly as
        # weights (1/x² would incorrectly give the best-fitting models the
        # lowest weights)
        weights = scores
    elif weight_by == "cv_rmse":
        weights = [1 / (x**2) for x in cv_rmses]
    elif weight_by == "rmse":
        weight_vals = []
        for g in grav_grids:
            points = utils.sample_grids(
                grid_points,
                g,
                sampled_name="sampled",
            )
            weight_vals.append(utils.rmse(points.gravity_anomaly - points.sampled))
        # convert residuals into weights
        weights = [1 / (x**2) for x in weight_vals]
    else:
        weights = None

    # get stats and weighted stats on the merged dataset
    stats_ds = model_ensemble_stats(
        merged,
        weights=weights,
        region=region,
    )

    if weight_by == "cv_rmse":
        stats_ds.attrs["cv_rmse"] = cv_rmses

    if plot is True:
        epsg, coast = utils.get_epsg(coast=coast)
        try:
            plotting.plot_stochastic_results(
                stats_ds=stats_ds,
                cmap="viridis",
                reverse_cpt=False,
                label="Predicted gravity",
                unit="mGal",
                region=plot_region,
            )
            if true_gravity is not None:
                mean, stdev = _mean_and_stdev(stats_ds)

                # pylint: disable=duplicate-code
                _ = ptk.grid_compare(
                    np.abs(true_gravity - mean),
                    stdev,
                    fig_height=12,
                    region=plot_region,
                    grid1_name="Stochastic error",
                    grid2_name="Stochastic uncertainty",
                    robust=True,
                    hist=True,
                    inset=False,
                    verbose="q",
                    title="difference",
                    coast=coast,
                    cmap="thermal",
                    epsg=epsg,
                )
                if deterministic_error is not None:
                    _ = ptk.grid_compare(
                        np.abs(deterministic_error),
                        stdev,
                        fig_height=12,
                        region=plot_region,
                        grid1_name="Deterministic error",
                        grid2_name="Stochastic uncertainty",
                        robust=True,
                        hist=True,
                        inset=False,
                        verbose="q",
                        title="difference",
                        coast=coast,
                        cmap="thermal",
                        epsg=epsg,
                    )
                _ = ptk.grid_compare(
                    true_gravity,
                    mean,
                    fig_height=12,
                    region=plot_region,
                    grid1_name="True gravity",
                    grid2_name="Mean gravity",
                    robust=True,
                    hist=True,
                    inset=False,
                    verbose="q",
                    title="difference",
                    coast=coast,
                    cmap="viridis",
                    epsg=epsg,
                )
                # pylint: enable=duplicate-code
        except Exception as e:  # pylint: disable=broad-exception-caught # noqa: BLE001
            logger.error("plotting failed with error: %s", e)

    return stats_ds, sampled_param_dict  # type: ignore[return-value]


def regional_misfit_uncertainty(
    runs: int,
    sample_gravity: bool = False,
    parameter_dict: dict[str, typing.Any] | None = None,
    region: tuple[float, float, float, float] | None = None,
    plot: bool = True,
    plot_region: tuple[float, float, float, float] | None = None,
    true_regional: xr.DataArray | None = None,
    coast: bool = False,
    weight_by: str | None = None,
    points_style="x.3c",
    **kwargs: typing.Any,
) -> tuple[xr.Dataset, dict[str, typing.Any]]:
    """
    Create a stochastic ensemble of regional gravity anomalies by sampling the
    constraints, gravity, or parameters within their respective distributions and
    calculate the cell-wise (weighted) statistics of the ensemble.

    Parameters
    ----------
    runs : int
        number of runs to perform
    sample_gravity : bool, optional
        choose to sample the gravity data from a normal distribution with a mean of each
        points value and a standard deviation set by the `uncert` column, by
        default False
    parameter_dict : dict[str, typing.Any] | None, optional
        dictionary of parameters passes to `regional_separation` with the uncertainty
        distributions defined, by default None
    region: tuple[float, float, float, float] | None = None,
        region to calculate statistics within, by default None
    plot : bool, optional
        show the results, by default True
    plot_region : tuple[float, float, float, float] | None, optional
        clip the plot to a region, by default None
    true_regional : xarray.DataArray | None, optional
        if the true regional misfit is known, will make a plot comparing the results, by
        default None
    coast : bool, optional
        whether to plot coastlines, by default False
    weight_by : str | None, optional
        how to weight the models, by default None

    Returns
    -------
    stats_ds: xarray.Dataset
        a dataset with the cell-wise statistics of the ensemble of regional gravity
    sampled_param_dict : dict[str, typing.Any]
        a dictionary of sampled parameter values.
    """
    new_kwargs = copy.deepcopy(kwargs)
    constraints_df = new_kwargs.pop("constraints_df", None)
    grav_ds = new_kwargs.pop("grav_ds", None)
    if isinstance(grav_ds, pd.DataFrame):
        msg = "DataFrame representation of gravity data is deprecated, use a dataset created through function `create_data`"
        raise DeprecationWarning(msg)
    if grav_ds is None:
        msg = "grav_ds must be provided"
        raise ValueError(msg)

    if parameter_dict is not None:
        sampled_param_dict = create_lhc(
            n_samples=runs,
            parameter_dict=parameter_dict,
            random_state=0,
            criterion="centered",
        )
    else:
        sampled_param_dict = None

    original_gravity = grav_ds.gravity_anomaly.copy()

    # only constraint-based regional separation methods accept constraints
    if constraints_df is not None:
        new_kwargs["constraints_df"] = constraints_df

    regional_grids = []
    for i in tqdm(range(runs), desc="starting regional ensemble"):
        # create random generator
        rand = np.random.default_rng(seed=i)
        if sample_gravity is True:
            # always sample from the original gravity values, not the previously
            # sampled values, to avoid a random walk of accumulating noise
            sampled_values = rand.normal(original_gravity, grav_ds.uncert)
            grav_ds["gravity_anomaly"] = original_gravity.copy(data=sampled_values)

        if sampled_param_dict is not None:
            for k, v in sampled_param_dict.items():
                new_kwargs[k] = v["sampled_values"][i]

        with utils._log_level(logging.WARNING):  # pylint: disable=protected-access
            grav_ds.inv.regional_separation(**new_kwargs)

        regional_grids.append(grav_ds.reg)

    # merge all topos into 1 dataset
    merged = merge_simulation_results(regional_grids)

    # get constraint point RMSE of each model
    if weight_by == "constraints":
        weight_vals = []
        for g in regional_grids:
            points = utils.sample_grids(
                constraints_df,
                g,
                sampled_name="sampled_regional",
            )
            weight_vals.append(utils.rmse(points.sampled_regional))
        # convert residuals into weights
        weights = [1 / (x**2) for x in weight_vals]
    else:
        weights = None

    # get stats and weighted stats on the merged dataset
    stats_ds = model_ensemble_stats(
        merged,
        weights=weights,
        region=region,
    )

    if plot is True:
        epsg, coast = utils.get_epsg(coast=coast)
        try:
            plotting.plot_stochastic_results(
                stats_ds=stats_ds,
                points=constraints_df,
                cmap="viridis",
                reverse_cpt=False,
                label="Regional gravity",
                unit="mGal",
                points_label="Topography constraints",
                region=plot_region,
                points_style=points_style,
            )
            if true_regional is not None:
                mean, stdev = _mean_and_stdev(stats_ds)
                # pylint: disable=duplicate-code
                _ = ptk.grid_compare(
                    np.abs(true_regional - mean),
                    stdev,
                    fig_height=12,
                    region=plot_region,
                    grid1_name="True error",
                    grid2_name="Stochastic uncertainty",
                    robust=True,
                    hist=True,
                    inset=False,
                    verbose="q",
                    title="difference",
                    coast=coast,
                    epsg=epsg,
                    cmap="thermal",
                    points=constraints_df.rename(
                        columns={"easting": "x", "northing": "y"}
                    ),
                    points_style=points_style,
                )
                _ = ptk.grid_compare(
                    true_regional,
                    mean,
                    fig_height=12,
                    region=plot_region,
                    grid1_name="True regional",
                    grid2_name="Mean regional",
                    robust=True,
                    hist=True,
                    inset=False,
                    verbose="q",
                    title="difference",
                    coast=coast,
                    epsg=epsg,
                    cmap="viridis",
                    points=constraints_df.rename(
                        columns={"easting": "x", "northing": "y"}
                    ),
                    points_style=points_style,
                )
                # pylint: enable=duplicate-code
        except Exception as e:  # pylint: disable=broad-exception-caught # noqa: BLE001
            logger.error("plotting failed with error: %s", e)

    return stats_ds, sampled_param_dict  # type: ignore[return-value]


def full_workflow_uncertainty_loop(
    inversion_object: "Inversion",
    runs: int,
    fname: str | None = None,
    sample_gravity: bool = False,
    gravity_filter_width: float | None = None,
    constraints_df: pd.DataFrame | None = None,
    sample_constraints: bool = False,
    starting_topography_parameter_dict: dict[str, typing.Any] | None = None,
    regional_misfit_parameter_dict: dict[str, typing.Any] | None = None,
    parameter_dict: dict[str, typing.Any] | None = None,
    create_starting_topography: bool = False,
    calculate_starting_gravity: bool = False,
    calculate_regional_misfit: bool = False,
    regional_grav_kwargs: dict[str, typing.Any] | None = None,
    starting_topography_kwargs: dict[str, typing.Any] | None = None,
) -> tuple[
    dict[str, typing.Any], list[pd.DataFrame], list[pd.DataFrame], dict[str, typing.Any]
]:
    """
    Run a series of inversions (N=runs), and save results of
    each inversion to pickle files starting with `fname`. If files already
    exist, just return the loaded results instead of re-running the inversion.
    Choose which variables to include in the sampling and whether or not to
    run a damping value cross-validation for each inversion.

    Feed returned values into function `merged_stats` to compute
    cell-wise stats on the resulting ensemble of starting topography models,
    inverted topography models, and gravity anomalies.

    Sampling of data (gravity and constraints) uses the columns "uncert" in the
    dataframes and randomly samples the data from a normal distribution with the
    uncertainty value as the standard deviation and the data value as the mean. The
    randomness is controlled by a seed which is equal to the run number, so it changes
    at every run, and the same run will always produce the same sampling. This allows
    the run number to be increased and this function run again with the same filename
    to continue the stochastic ensemble. This only works with data sampling, not
    parameter sampling.

    Sampling of parameter values are determined by 3 supplied dictionaries:
    `parameter_dict` which can contain parameters density_contrast, zref, and
    solver_damping. The other two dictionaries are `starting_topography_parameter_dict`
    and `regional_misfit_parameter_dict` which can contain any parameters that are used
    in :func:`create_topography` and :meth:`DatasetAccessorInvert4Geom.regional_separation` respectively. Any
    parameters in these 3 dictionaries will be sampled with a Latin Hypercube sampling
    technique and the sampled values will be past to `inversion.run_inversion`. These
    dictionaries should be formatted as follows: `{"parameter_name": {"distribution":
    "normal", "loc": 0, "scale": 1, "log": True}}` where for a "distribution" of
    "normal", "loc" is the center of the distribution and "scale" is the standard
    deviation, and for a "distribution" of "uniform", "loc" is the lower bound and
    "scale" is the range of the distribution. If "log" is True, "loc" and "scale"
    refer to the base 10 exponent of the values. For example, a uniform distribution
    with loc=-4, scale=6 and log=True will sample values between 1e-4 and 1e2. The
    Latin Hypercube sampling takes the parameter distributions and the number of runs
    and creates evenly spaced samples within the distribution bounds. Therefore, unlike
    the sampled of data, the same run number will only reproduce the same sampling
    results if the total run numbers are the same. This means you should not reuse the
    filename to add more iterations to the stochastic ensemble but increasing the run
    number if you are using parameter sampling.

    Parameters
    ----------
    inversion_object : Inversion
        an Inversion object created through
    runs : int
        number of inversion workflows to run
    fname : str | None, optional
        file name to use as root to save each inversions results to, by default None and
        is set to "tmp_{random.randint(0,999)}_stochastic_ensemble".
    sample_gravity : bool, optional
        choose to randomly sample the gravity data from a normal distribution with a
        mean of each data value and a standard deviation given by the column "uncert",
        by default False
    gravity_filter_width : float | None, optional
        the width in meters of a low-pass filter to apply to the gravity data after
        sampling, by default None
    constraints_df : pandas.DataFrame | None, optional
        dataframe of constraints with columns "easting", "northing", and "upward", by
        default None
    sample_constraints : bool, optional
        choose to randomly sample the constraint elevations from a normal distribution
        with a mean of each data value and a standard deviation given by the column
        "uncert", by default False
    starting_topography_parameter_dict : dict[str, typing.Any] | None, optional
        parameters with their uncertainty distributions used for creating the starting
        topography model, by default None
    regional_misfit_parameter_dict : dict[str, typing.Any] | None, optional
        parameters with their uncertainty distributions used for estimating the regional
        component of the gravity misfit, by default None
    parameter_dict : dict[str, typing.Any] | None, optional
        parameters with their uncertainty distributions used in the inversion workflow,
        by default None
    create_starting_topography : bool, optional
        choose to recreate the starting topography model, by default False
    calculate_starting_gravity : bool, optional
        choose to recalculate the starting gravity, by default False
    calculate_regional_misfit : bool, optional
        choose to recalculate the regional gravity, by default False
    regional_grav_kwargs : dict[str, typing.Any] | None, optional
        kwargs passed to :meth:`DatasetAccessorInvert4Geom.regional_separation`, by default None
    starting_topography_kwargs : dict[str, typing.Any] | None, optional
        kwargs passed to :func:`create_topography`, by default None

    Returns
    -------
    params : list[dict[str, typing.Any]]
        list of inversion parameters dictionaries with added key for the run number
    grav_datasets : list[xr.Dataset]
        list of gravity datasets from each inversion run
    prism_dfs : list[pandas.DataFrame]
        list of prism dataframes from each inversion run
    sampled_params : dict[str, typing.Any]
        dictionary of sampled parameter values from the Latin Hypercube sampling
    """

    if isinstance(inversion_object, int):
        msg = "`full_workflow_uncertainty_loop` function has been updated, first parameter must be an Inversion object created through the `Inversion` class"
        raise DeprecationWarning(msg)

    inv = copy.deepcopy(inversion_object)
    original_constraints_df = (
        constraints_df.copy() if constraints_df is not None else None
    )
    original_grav_df = inv.data.inv.df.copy()

    if sample_constraints is True:
        if original_constraints_df is None:
            msg = "constraints_df must be provided if sample_constraints is True"
            raise ValueError(msg)
        sampled_constraints = original_constraints_df.copy()
        test_constraint_value = copy.deepcopy(constraints_df.upward.iloc[0])  # type: ignore[union-attr]

    # ensure kwargs are not altered by making copies before sampling values
    if regional_grav_kwargs is not None:
        new_regional_grav_kwargs = copy.deepcopy(regional_grav_kwargs)
    else:
        new_regional_grav_kwargs = None

    if starting_topography_kwargs is not None:
        starting_topography_kwargs = copy.deepcopy(starting_topography_kwargs)

        upper_confining_layer = inv.model.upper_confining_layer
        lower_confining_layer = inv.model.lower_confining_layer

        # copy over kwargs from model instance
        starting_topography_kwargs["region"] = inv.model.region
        starting_topography_kwargs["spacing"] = inv.model.spacing
        starting_topography_kwargs["coord_names"] = inv.model.coord_names

        if inv.model.model_type == "tesseroids":
            dataset_to_add = inv.model[["mask", "geocentric_radius"]].drop_vars(
                ["top", "bottom"]
            )
        else:
            dataset_to_add = inv.model[["mask"]].drop_vars(["top", "bottom"])

        starting_topography_kwargs["dataset_to_add"] = dataset_to_add
        starting_topography_kwargs["upper_confining_layer"] = upper_confining_layer
        starting_topography_kwargs["lower_confining_layer"] = lower_confining_layer

    if parameter_dict is not None:
        sampled_param_dict = create_lhc(
            n_samples=runs,
            parameter_dict=parameter_dict,
            random_state=0,
            criterion="centered",
        )
    else:
        sampled_param_dict = None

    if starting_topography_parameter_dict is not None:
        sampled_starting_topography_parameter_dict = create_lhc(
            n_samples=runs,
            parameter_dict=starting_topography_parameter_dict,
            random_state=0,
            criterion="centered",
        )
    else:
        sampled_starting_topography_parameter_dict = None

    if regional_misfit_parameter_dict is not None:
        sampled_regional_misfit_parameter_dict = create_lhc(
            n_samples=runs,
            parameter_dict=regional_misfit_parameter_dict,
            random_state=0,
            criterion="centered",
        )
    else:
        sampled_regional_misfit_parameter_dict = None

    # set file name for saving results with random number between 0 and 999
    if fname is None:
        fname = f"tmp_{random.randint(0, 999)}_stochastic_ensemble"

    # if file exists, start and next run, else start at 0
    try:
        # load pickle files
        params = []
        with pathlib.Path(f"{fname}_params.pickle").open("rb") as file:
            while 1:
                try:
                    params.append(pickle.load(file))
                except EOFError:
                    break
        starting_run = len(params)
    except FileNotFoundError:
        logger.info(
            "No pickle files starting with '%s' found, creating new files\n", fname
        )

        # create / overwrite pickle files
        with pathlib.Path(f"{fname}_params.pickle").open("wb") as _:
            pass
        with pathlib.Path(f"{fname}_grav_datasets.pickle").open("wb") as _:
            pass
        with pathlib.Path(f"{fname}_prism_dfs.pickle").open("wb") as _:
            pass
        starting_run = 0
    if starting_run == runs:
        logger.info("all %s runs already complete, loading results from files.", runs)

    if sample_gravity is True:
        sampled_grav = original_grav_df.copy()
        test_grav_value = copy.deepcopy(inv.data.inv.df.gravity_anomaly.iloc[0])

    for i in tqdm(range(starting_run, runs), desc="stochastic ensemble"):
        if i == starting_run:
            logger.info(
                "starting stochastic uncertainty analysis at run %s of %s\n"
                "saving results to pickle files with prefix: '%s'\n",
                starting_run,
                runs,
                fname,
            )

        # sample grav and constraints with random sampling
        # create random generator
        rand = np.random.default_rng(seed=i)

        if sample_gravity is True:
            # assert original gravity values are unaltered
            assert test_grav_value == original_grav_df.gravity_anomaly.iloc[0], (
                "original gravity values have been altered by sampling!"
            )
            # always sample from the original gravity values, not the previously
            # sampled values in inv.data, to avoid a random walk of accumulating noise
            sampled_grav["gravity_anomaly"] = rand.normal(
                original_grav_df.gravity_anomaly, original_grav_df.uncert
            )
            # low-pass filter the sampled gravity data
            if gravity_filter_width is not None:
                filtered_grav = utils.filter_grid(
                    sampled_grav.set_index(["northing", "easting"])
                    .to_xarray()
                    .gravity_anomaly,
                    gravity_filter_width,
                    filter_type="lowpass",
                    pad_mode="linear_ramp",
                )
                # df = inv.data.inv.df.copy()
                sampled_grav["gravity_anomaly"] = filtered_grav.to_numpy().ravel()
                # sampled_grav["gravity_anomaly"] = df.set_index(["northing", "easting"]).to_xarray().gravity_anomaly
            # update the inversion object with the sampled gravity data
            ds = sampled_grav.set_index(["northing", "easting"]).to_xarray()
            inv.data["gravity_anomaly"] = ds.gravity_anomaly
            # sampled_grav.attrs.update(inv.data.attrs)  # type: ignore[union-attr]
            # inv.data = sampled_grav

        if sample_constraints is True:
            if original_constraints_df is None:
                msg = "constraints_df must be provided if sample_constraints is True"
                raise ValueError(msg)

            # assert original constraint values are unaltered
            assert test_constraint_value == original_constraints_df.upward.iloc[0], (
                "original constraint values have been altered by sampling!"
            )

            sampled_constraints = randomly_sample_data(
                seed=i,
                data_df=original_constraints_df,
                data_col="upward",
                uncert_col="uncert",
            )
            if (starting_topography_kwargs is not None) and (
                starting_topography_kwargs.get("constraints_df", None) is not None
            ):
                starting_topography_kwargs["constraints_df"] = sampled_constraints
            if (new_regional_grav_kwargs is not None) and (
                new_regional_grav_kwargs.get("constraints_df", None) is not None
            ):
                new_regional_grav_kwargs["constraints_df"] = sampled_constraints
            constraints_df = sampled_constraints

        # if parameters provided, sampled and add back to kwargs
        if sampled_param_dict is not None:
            if "solver_damping" in sampled_param_dict:
                sampled_solver_damping = sampled_param_dict["solver_damping"][
                    "sampled_values"
                ][i]
                inv.solver_damping = sampled_solver_damping
                assert inv.solver_damping == sampled_solver_damping, (
                    "sampled damping hasn't been correctly set"
                )
            if "density_contrast" in sampled_param_dict:
                sampled_density_contrast = sampled_param_dict["density_contrast"][
                    "sampled_values"
                ][i]
                inv.model = inv.model.assign_attrs(
                    {"density_contrast": sampled_density_contrast}
                )
                assert inv.model.density_contrast == sampled_density_contrast, (
                    "sampled density contrast hasn't been correctly set"
                )
                calculate_starting_gravity = True
            if "zref" in sampled_param_dict:
                sampled_zref = sampled_param_dict["zref"]["sampled_values"][i]
                inv.model = inv.model.assign_attrs({"zref": sampled_zref})
                assert inv.model.zref == sampled_zref, (
                    "sampled zref hasn't been correctly set"
                )
                calculate_starting_gravity = True
        if sampled_starting_topography_parameter_dict is not None:
            for k, v in sampled_starting_topography_parameter_dict.items():
                starting_topography_kwargs[k] = v["sampled_values"][i]  # type: ignore[index]
        if sampled_regional_misfit_parameter_dict is not None:
            for k, v in sampled_regional_misfit_parameter_dict.items():
                new_regional_grav_kwargs[k] = v["sampled_values"][i]  # type: ignore[index]

        # define what needs to be done depending on what parameters are sampled
        if sample_gravity is True:
            calculate_starting_gravity = True
        if sample_constraints is True:
            create_starting_topography = True
        if sampled_starting_topography_parameter_dict is not None:
            create_starting_topography = True
        if sampled_regional_misfit_parameter_dict is not None:
            calculate_regional_misfit = True
        # if certain things are recalculated, other must be as well
        if create_starting_topography is True:
            calculate_starting_gravity = True
        if calculate_starting_gravity is True:
            calculate_regional_misfit = True

        inversion_kwargs = {
            k: inv.__dict__[k]
            for k in (
                "max_iterations",
                "l2_norm_tolerance",
                "delta_l2_norm_tolerance",
                "perc_increase_limit",
                "deriv_type",
                "jacobian_finite_step_size",
                "model_properties_method",
                "solver_type",
                "solver_damping",
                "sharpness_weight",
                "sharpness_norm",
                "irls_epsilon",
                "irls_iterations",
                "apply_weighting_grid",
                "weighting_grid",
                "apply_residual_weighting_grid",
                "residual_weighting_grid",
            )
        }
        # run inversion
        with utils._log_level(logging.ERROR):  # pylint: disable=protected-access
            inv_results = inversion.run_inversion_workflow(
                grav_ds=inv.data,
                create_starting_topography=create_starting_topography,
                calculate_starting_gravity=calculate_starting_gravity,
                calculate_regional_misfit=calculate_regional_misfit,  # pylint: disable=possibly-used-before-assignment
                run_damping_cv=False,
                run_zref_or_density_optimization=False,
                fname=f"{fname}_{i}",
                starting_topography=inv.model.starting_topography.to_dataset(
                    name="upward"
                ),
                starting_topography_kwargs=starting_topography_kwargs,
                upper_confining_layer=inv.model.upper_confining_layer,
                lower_confining_layer=inv.model.lower_confining_layer,
                buffer_width=inv.model.buffer_width,
                density_contrast=inv.model.density_contrast,
                zref=inv.model.zref,
                regional_grav_kwargs=new_regional_grav_kwargs,
                constraints_df=constraints_df,
                inversion_kwargs=inversion_kwargs,
            )
        # add run number to the parameter values
        inv_results.params["run_num"] = i  # type: ignore[index]

        # save results
        with pathlib.Path(f"{fname}_params.pickle").open("ab") as file:
            pickle.dump(inv_results.params, file, protocol=pickle.HIGHEST_PROTOCOL)
        with pathlib.Path(f"{fname}_grav_datasets.pickle").open("ab") as file:
            pickle.dump(inv_results.data, file, protocol=pickle.HIGHEST_PROTOCOL)
        with pathlib.Path(f"{fname}_prism_dfs.pickle").open("ab") as file:
            pickle.dump(inv_results.model, file, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug("Finished inversion %s of %s for stochastic ensemble", i + 1, runs)

    # load pickle files
    params = []
    with pathlib.Path(f"{fname}_params.pickle").open("rb") as file:
        while 1:
            try:
                params.append(pickle.load(file))
            except EOFError:
                break
    grav_datasets = []
    with pathlib.Path(f"{fname}_grav_datasets.pickle").open("rb") as file:
        while 1:
            try:
                grav_datasets.append(pickle.load(file))
            except EOFError:
                break
    prism_dfs = []
    with pathlib.Path(f"{fname}_prism_dfs.pickle").open("rb") as file:
        while 1:
            try:
                prism_dfs.append(pickle.load(file))
            except EOFError:
                break

    return (
        params,
        grav_datasets,
        prism_dfs,
        sampled_param_dict,
    )  # type: ignore[return-value]


def topography_gradient_uncertainty(
    merged: xr.Dataset,
    threshold: float,
) -> xr.Dataset:
    """
    Convert an ensemble of inverted topographies into fault-location statistics.

    For each ensemble member the horizontal gradient magnitude of the topography is
    computed. Steep gradients in an inverted surface are candidate fault scarps, so the
    cell-wise fraction of members whose gradient exceeds ``threshold`` is a probability
    map of fault locations: cells where every member is steep are robust scarp
    positions, and the width of a high-probability band maps the horizontal
    uncertainty of that fault's position.

    Parameters
    ----------
    merged : xarray.Dataset
        ensemble of topographies as one variable per run, in the format returned by
        :func:`merge_simulation_results`
    threshold : float
        gradient magnitude (in grid units, e.g. m/m) above which a cell is counted as
        a scarp

    Returns
    -------
    xarray.Dataset
        dataset with variables ``gradient_mean``, ``gradient_stdev`` (cell-wise
        statistics of the ensemble's gradient magnitudes) and ``scarp_probability``
        (fraction of members exceeding ``threshold``)
    """
    original_dims = list(merged[next(iter(merged))].sizes.keys())

    gradients = []
    for name in merged:
        topo = merged[name]
        grad = np.sqrt(
            topo.differentiate(original_dims[1]) ** 2
            + topo.differentiate(original_dims[0]) ** 2
        )
        gradients.append(grad.rename(name))

    stacked = xr.concat(gradients, dim="runs")
    return xr.Dataset(
        {
            "gradient_mean": stacked.mean("runs"),
            "gradient_stdev": stacked.std("runs"),
            "scarp_probability": (stacked > threshold).mean("runs"),
        }
    )


def model_ensemble_stats(
    dataset: xr.Dataset,
    weights: list[float] | NDArray | None = None,
    region: tuple[float, float, float, float] | None = None,
) -> xr.Dataset:
    """
    Given a dataset, calculate the cell-wise mean, standard deviation, and weighted mean
    and standard deviation of the variables.

    Parameters
    ----------
    dataset : xarray.Dataset
        dataset to perform cell-wise statistics on
    weights : list | numpy.ndarray, optional
        weights to use in statistic calculations for each inversion topography, by
        default None
    region : tuple[float, float, float, float], optional
        regions to calculate statistics within, by default None

    Returns
    -------
    xarray.Dataset
        Dataset with variables for the mean, standard deviation, weighted mean, and
        weighted standard deviation of the ensemble of inverted topographies.
    """
    if region is None:
        da_list = [dataset[i] for i in list(dataset)]
    else:
        da_list = [
            dataset[i].sel(
                easting=slice(region[0], region[1]),
                northing=slice(region[2], region[3]),
            )
            for i in list(dataset)
        ]
    merged = (
        xr.concat(da_list, dim="runs")
        .assign_coords({"runs": list(dataset)})
        .rename("run_num")
        .to_dataset()
    )

    z_mean = merged["run_num"].mean("runs").rename("z_mean")
    z_min = merged["run_num"].min("runs").rename("z_min")
    z_max = merged["run_num"].max("runs").rename("z_max")
    z_stdev = merged["run_num"].std("runs").rename("z_stdev")
    # z_var = merged["run_num"].var("runs").rename("z_var")

    if weights is not None:
        assert len(da_list) == len(weights)

        weighted_mean = sum(
            g * w for g, w in zip(da_list, weights, strict=False)
        ) / sum(weights)
        weighted_mean = weighted_mean.rename("weighted_mean")

        # from https://stackoverflow.com/questions/30383270/how-do-i-calculate-the-standard-deviation-between-weighted-measurements
        weighted_var = (
            sum(
                w * (g - weighted_mean) ** 2
                for g, w in zip(da_list, weights, strict=False)
            )
        ) / sum(weights)
        weighted_stdev = np.sqrt(weighted_var)
        weighted_stdev = weighted_stdev.rename("weighted_stdev")

    else:
        weighted_mean = None
        weighted_stdev = None
        # weighted_var = None

    grids = [
        merged,
        z_mean,
        z_stdev,
        weighted_mean,
        weighted_stdev,
        z_min,
        z_max,
        # z_var, weighted_var,
    ]
    stats = [g for g in grids if g is not None]

    return xr.merge(stats)


def merge_simulation_results(
    grids: list[xr.DataArray],
) -> xr.Dataset:
    """
    Merge a list of grids into a single dataset with variable names "run_<number>"
    where x is the run number.

    Parameters
    ----------
    grids : list[xarray.DataArray]
        list of xarray grids

    Returns
    -------
    xarray.Dataset
        dataset with a variable for each grid, with the variable
        name in the format "run_<number>".
    """
    renamed_grids = []
    for i, j in enumerate(grids):
        da = j.rename(f"run_{i}")
        renamed_grids.append(da)
    return xr.merge(renamed_grids, compat="override")


def merged_stats(
    results: tuple[typing.Any],
    plot: bool = True,
    constraints_df: pd.DataFrame | None = None,
    weight_by: str = "residual",
    region: tuple[float, float, float, float] | None = None,
    points_style: str = "x.3c",
) -> xr.Dataset:
    """
    Use the outputs of the function `uncertainty.full_workflow_uncertainty_loop` to
    calculate the cell-wise statistics of the inversion ensemble and plot the resulting
    mean and standard deviation of the ensemble.

    Parameters
    ----------
    results : tuple[typing.Any]
        list of lists of inversion results output from the function
        `uncertainty.full_workflow_uncertainty_loop`
    plot : bool, optional
        show the resulting weighted mean and weighted standard deviation of the
        inversion ensemble, by default True
    constraints_df : pandas.DataFrame, optional
        dataframe of constraint points to use for weighting the cell-wise statistics and
        for plotting , by default None
    weight_by : str, optional
        choose to weight the cell-wise stats by either the RMS of the final residual
        gravity misfit of each inversion, or by the RMS between a priori topography
        measurements supplied by constraints_df and the inverted topography of each
        inversion, by default "residual"
    region : tuple[float, float, float, float], optional
        region to calculate statistics within, by default None

    Returns
    -------
    xarray.Dataset
        Dataset with variables for the mean, standard deviation, weighted mean, and
        weighted standard deviation of the ensemble of inverted topographies.
    """
    # unpack results
    _params, grav_datasets, prism_dss, _ = results  # type: ignore[misc]

    # get merged dataset
    merged = merge_simulation_results([ds.topography for ds in prism_dss])
    # get final gravity residual RMS of each model
    if weight_by == "residual":
        # get the RMS of the final gravity residual of each model; "res" is updated
        # after every inversion iteration so holds the final residual
        weight_vals = [utils.rmse(ds.res) for ds in grav_datasets]
        # convert residuals into weights
        weights = [1 / (x**2) for x in weight_vals]
    # get constraint point RMSE of each model
    elif weight_by == "constraints":
        weight_vals = []
        for ds in prism_dss:
            bed = ds["topography"]
            points = utils.sample_grids(
                constraints_df,
                bed,
                sampled_name="sampled_topo",
            )
            points["dif"] = points.upward - points.sampled_topo
            weight_vals.append(utils.rmse(points.dif))
        # convert residuals into weights
        weights = [1 / (x**2) for x in weight_vals]
    else:
        weights = None
    # get stats and weighted stats on the merged dataset
    stats_ds = model_ensemble_stats(
        merged,
        weights=weights,
        region=region,
    )

    if plot is True:
        try:
            plotting.plot_stochastic_results(
                stats_ds=stats_ds,
                points=constraints_df,
                cmap="rain",
                reverse_cpt=True,
                label="inverted topography",
                points_label="Topography constraints",
                points_style=points_style,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught # noqa: BLE001
            logger.error("plotting failed with error: %s", e)

    return stats_ds
