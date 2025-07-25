from __future__ import annotations  # pylint: disable=too-many-lines

import copy
import logging
import pathlib
import pickle
import random
import typing

import harmonica as hm
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr
from numpy.typing import NDArray
from polartoolkit import utils as polar_utils
from tqdm.autonotebook import tqdm

try:
    import UQpy

    class DiscreteUniform(UQpy.distributions.DistributionDiscrete1D):  # type: ignore[misc]
        """
        Discrete uniform distribution.
        """

        def __init__(
            self,
            loc: float | int = 0.0,
            scale: float | int = 1.0,
        ):
            super().__init__(
                low=loc, high=loc + scale + 1, ordered_parameters=("low", "high")
            )
            self._construct_from_scipy(scipy_name=sp.stats.randint)
except ImportError:
    UQpy = None

from invert4geom import inversion, log, plotting, regional, utils


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
    random_state : int, optional
        random state to use for sampling, by default 1
    criterion : str, optional
        criterion to use for sampling, by default "centered", options are "centered",
        "random", "maximin", or "mincorrelation", which each relate to a criterion from
        the Python package UQpy.
    Returns
    -------
    dict[dict[typing.Any]]
        nested dictionary with parameter names, distribution specifics, and sampled
        values
    """
    if UQpy is None:
        msg = "Missing optional dependency 'UQpy' required for uncertainty analysis."
        raise ImportError(msg)

    param_dict = copy.deepcopy(parameter_dict)

    # create distributions for parameters
    dists = {}
    for k, v in param_dict.items():
        if v["distribution"] == "uniform":
            dists[k] = UQpy.distributions.Uniform(loc=v["loc"], scale=v["scale"])
        elif v["distribution"] == "normal":
            dists[k] = UQpy.distributions.Normal(loc=v["loc"], scale=v["scale"])
        elif v["distribution"] == "uniform_discrete":
            dists[k] = DiscreteUniform(loc=v["loc"], scale=v["scale"])
        else:
            msg = "Unknown distribution type: %s"
            raise ValueError(msg, v["distribution"])

    if criterion == "centered":
        criterion = (
            UQpy.sampling.stratified_sampling.latin_hypercube_criteria.Centered()
        )
    elif criterion == "random":
        criterion = UQpy.sampling.stratified_sampling.latin_hypercube_criteria.Random()
    elif criterion == "maximin":
        criterion = UQpy.sampling.stratified_sampling.latin_hypercube_criteria.MaxiMin()
    elif criterion == "mincorrelation":
        criterion = (
            UQpy.sampling.stratified_sampling.latin_hypercube_criteria.MinCorrelation()
        )
    else:
        msg = "Unknown criterion type: %s"
        raise ValueError(msg, criterion)

    # make latin hyper cube
    lhc = UQpy.sampling.LatinHypercubeSampling(
        distributions=[v for k, v in dists.items()],
        criterion=criterion,
        random_state=np.random.RandomState(random_state),  # pylint: disable=no-member
        nsamples=n_samples,
    )

    # add sampled values to parameters dict
    for j, (k, v) in enumerate(param_dict.items()):
        if v.get("norm_limits", None) is not None:
            norm_limits = v["norm_limits"]
            lhc.samples[:, j] = utils.normalize(
                lhc.samples[:, j],
                low=norm_limits[0],
                high=norm_limits[1],
            )
        if v.get("log", False) is True:
            v["sampled_values"] = 10 ** lhc._samples[:, j]  # pylint: disable=protected-access
        else:
            v["sampled_values"] = lhc.samples[:, j]  # pylint: disable=protected-access
        if v.get("dtype", None) is int:
            v["sampled_values"] = v["sampled_values"].round().astype(int)

        log.info(
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
            sampled_constraints["upward"] = rand.normal(
                sampled_constraints.upward, sampled_constraints.uncert
            )

        if sampled_param_dict is not None:
            for k, v in sampled_param_dict.items():
                new_kwargs[k] = v["sampled_values"][i]

        with utils._log_level(logging.WARN):  # pylint: disable=protected-access
            starting_topography = utils.create_topography(
                constraints_df=sampled_constraints,
                **new_kwargs,
            )
        topos.append(starting_topography)

        # sample the topography at the constraint points
        sampled_constraints = utils.sample_grids(
            sampled_constraints,
            starting_topography,
            "sampled",
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
                try:
                    mean = stats_ds.weighted_mean
                    stdev = stats_ds.weighted_stdev
                except AttributeError:
                    mean = stats_ds.z_mean
                    stdev = stats_ds.z_stdev

                _ = polar_utils.grd_compare(
                    np.abs(true_topography - mean),
                    stdev,
                    fig_height=12,
                    region=plot_region,
                    plot=True,
                    grid1_name="True error",
                    grid2_name="Stochastic uncertainty",
                    robust=True,
                    hist=True,
                    inset=False,
                    verbose="q",
                    title="difference",
                    grounding_line=False,
                    cmap="thermal",
                    points=sampled_constraints.rename(
                        columns={"easting": "x", "northing": "y"}
                    ),
                    points_style="x.3c",
                )
                _ = polar_utils.grd_compare(
                    true_topography,
                    mean,
                    fig_height=12,
                    region=plot_region,
                    plot=True,
                    grid1_name="True topography",
                    grid2_name="Mean topography",
                    robust=True,
                    hist=True,
                    inset=False,
                    verbose="q",
                    title="difference",
                    grounding_line=False,
                    cmap="rain",
                    reverse_cpt=True,
                    points=sampled_constraints.rename(
                        columns={"easting": "x", "northing": "y"}
                    ),
                    points_style="x.3c",
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error("plotting failed with error: %s", e)

    return stats_ds, sampled_param_dict  # type: ignore[return-value]
    # pylint: enable=duplicate-code


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
    deterministic_error : xarray.DataArray | None, optional
        if the deterministic error is known, will make a plot comparing the results, by
        default None
    weight_by : str | None, optional
        how to weight the models, by default None

    Returns
    -------
    stats_ds: xarray.Dataset,
        a dataset with the cell-wise statistics of the ensemble of regional gravity
    sampled_parms_dict: dict[str, typing.Any]
        a dictionary of sampled parameter values.
    """
    new_kwargs = copy.deepcopy(kwargs)

    if parameter_dict is not None:
        sampled_param_dict = create_lhc(
            n_samples=runs,
            parameter_dict=parameter_dict,
            random_state=0,
            criterion="centered",
        )
    else:
        sampled_param_dict = None

    grav_grids = []
    scores = []
    for i in tqdm(range(runs), desc="starting equivalent sources ensemble"):
        if sampled_param_dict is not None:
            for k, v in sampled_param_dict.items():
                new_kwargs[k] = v["sampled_values"][i]

        with utils._log_level(logging.WARN):  # pylint: disable=protected-access
            # refit EqSources with best parameters
            eqs = hm.EquivalentSources(
                **new_kwargs,
            )
            eqs.fit(coords, data, weights=new_kwargs.get("weights", None))

        if weight_by == "score":
            scores.append(
                eqs.score(coords, data, weights=new_kwargs.get("weights", None))
            )

        # predict sources onto grid
        grid_points["predicted_grav"] = eqs.predict(
            (
                grid_points.easting,
                grid_points.northing,
                grid_points.upward,
            ),
        )

        grav_grids.append(
            grid_points.set_index(["northing", "easting"]).to_xarray().predicted_grav
        )

    # merge all topos into 1 dataset
    merged = merge_simulation_results(grav_grids)

    # get constraint point RMSE of each model
    if weight_by == "score":
        # convert residuals into weights
        weights = [1 / (x**2) for x in scores]
        # weights = scores
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

    if plot is True:
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
                try:
                    mean = stats_ds.weighted_mean
                    stdev = stats_ds.weighted_stdev
                except AttributeError:
                    mean = stats_ds.z_mean
                    stdev = stats_ds.z_stdev

                # pylint: disable=duplicate-code
                _ = polar_utils.grd_compare(
                    np.abs(true_gravity - mean),
                    stdev,
                    fig_height=12,
                    region=plot_region,
                    plot=True,
                    grid1_name="Stochastic error",
                    grid2_name="Stochastic uncertainty",
                    robust=True,
                    hist=True,
                    inset=False,
                    verbose="q",
                    title="difference",
                    grounding_line=False,
                    cmap="thermal",
                )
                if deterministic_error is not None:
                    _ = polar_utils.grd_compare(
                        np.abs(deterministic_error),
                        stdev,
                        fig_height=12,
                        region=plot_region,
                        plot=True,
                        grid1_name="Deterministic error",
                        grid2_name="Stochastic uncertainty",
                        robust=True,
                        hist=True,
                        inset=False,
                        verbose="q",
                        title="difference",
                        grounding_line=False,
                        cmap="thermal",
                    )
                _ = polar_utils.grd_compare(
                    true_gravity,
                    mean,
                    fig_height=12,
                    region=plot_region,
                    plot=True,
                    grid1_name="True gravity",
                    grid2_name="Mean gravity",
                    robust=True,
                    hist=True,
                    inset=False,
                    verbose="q",
                    title="difference",
                    grounding_line=False,
                    cmap="viridis",
                )
                # pylint: enable=duplicate-code
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error("plotting failed with error: %s", e)

    return stats_ds, sampled_param_dict  # type: ignore[return-value]


def regional_misfit_uncertainty(
    runs: int,
    sample_gravity: bool = False,
    parameter_dict: dict[str, typing.Any] | None = None,
    region: tuple[float, float, float, float] | None = None,
    plot: bool = True,
    plot_region: tuple[float, float, float, float] | None = None,
    true_regional: xr.DataArray | None = None,
    weight_by: str | None = None,
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

    Returns
    -------
    stats_ds: xarray.Dataset
        a dataset with the cell-wise statistics of the ensemble of regional gravity
    sampled_param_dict : dict[str, typing.Any]
        a dictionary of sampled parameter values.
    """
    new_kwargs = copy.deepcopy(kwargs)
    constraints_df = new_kwargs.pop("constraints_df", None)
    grav_df = new_kwargs.pop("grav_df", None)

    if parameter_dict is not None:
        sampled_param_dict = create_lhc(
            n_samples=runs,
            parameter_dict=parameter_dict,
            random_state=0,
            criterion="centered",
        )
    else:
        sampled_param_dict = None

    # print(sampled_param_dict)
    regional_grids = []
    for i in tqdm(range(runs), desc="starting regional ensemble"):
        # create random generator
        rand = np.random.default_rng(seed=i)
        if sample_gravity is True:
            sampled_grav = grav_df.copy()
            gobs_sampled = rand.normal(
                sampled_grav.gravity_anomaly, sampled_grav.uncert
            )
            sampled_grav["gravity_anomaly"] = gobs_sampled
            grav_df = sampled_grav.copy()

        if sampled_param_dict is not None:
            for k, v in sampled_param_dict.items():
                new_kwargs[k] = v["sampled_values"][i]

        with utils._log_level(logging.WARN):  # pylint: disable=protected-access
            grav_df = regional.regional_separation(
                constraints_df=constraints_df,
                grav_df=grav_df,
                **new_kwargs,
            )

        regional_grids.append(
            grav_df.set_index(["northing", "easting"]).to_xarray().reg
        )

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
            )
            if true_regional is not None:
                try:
                    mean = stats_ds.weighted_mean
                    stdev = stats_ds.weighted_stdev
                except AttributeError:
                    mean = stats_ds.z_mean
                    stdev = stats_ds.z_stdev
                # pylint: disable=duplicate-code
                _ = polar_utils.grd_compare(
                    np.abs(true_regional - mean),
                    stdev,
                    fig_height=12,
                    region=plot_region,
                    plot=True,
                    grid1_name="True error",
                    grid2_name="Stochastic uncertainty",
                    robust=True,
                    hist=True,
                    inset=False,
                    verbose="q",
                    title="difference",
                    grounding_line=False,
                    cmap="thermal",
                    points=constraints_df.rename(
                        columns={"easting": "x", "northing": "y"}
                    ),
                    points_style="x.3c",
                )
                _ = polar_utils.grd_compare(
                    true_regional,
                    mean,
                    fig_height=12,
                    region=plot_region,
                    plot=True,
                    grid1_name="True regional",
                    grid2_name="Mean regional",
                    robust=True,
                    hist=True,
                    inset=False,
                    verbose="q",
                    title="difference",
                    grounding_line=False,
                    cmap="viridis",
                    points=constraints_df.rename(
                        columns={"easting": "x", "northing": "y"}
                    ),
                    points_style="x.3c",
                )
                # pylint: enable=duplicate-code
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error("plotting failed with error: %s", e)

    return stats_ds, sampled_param_dict  # type: ignore[return-value]


def full_workflow_uncertainty_loop(
    runs: int,
    fname: str | None = None,
    sample_gravity: bool = False,
    gravity_filter_width: float | None = None,
    sample_constraints: bool = False,
    starting_topography_parameter_dict: dict[str, typing.Any] | None = None,
    regional_misfit_parameter_dict: dict[str, typing.Any] | None = None,
    parameter_dict: dict[str, typing.Any] | None = None,
    create_starting_topography: bool = False,
    create_starting_prisms: bool = False,
    calculate_starting_gravity: bool = False,
    calculate_regional_misfit: bool = False,
    regional_grav_kwargs: dict[str, typing.Any] | None = None,
    starting_topography_kwargs: dict[str, typing.Any] | None = None,
    **kwargs: typing.Any,
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
    in `utils.create_topography` and `regional.regional_separation` respectively. Any
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
    create_starting_prisms : bool, optional
        choose to recreate the starting prism model, by default False
    calculate_starting_gravity : bool, optional
        choose to recalculate the starting gravity, by default False
    calculate_regional_misfit : bool, optional
        choose to recalculate the regional gravity, by default False
    regional_grav_kwargs : dict[str, typing.Any] | None, optional
        kwargs passed to :func:`.regional.regional_separation`, by default None
    starting_topography_kwargs : dict[str, typing.Any] | None, optional
        kwargs passed to :func:`.utils.create_topography`, by default None

    Returns
    -------
    params : list[dict[str, typing.Any]]
        list of inversion parameters dictionaries with added key for the run number
    grav_dfs : list[pandas.DataFrame]
        list of gravity dataframes from each inversion run
    prism_dfs : list[pandas.DataFrame]
        list of prism dataframes from each inversion run
    sampled_params : dict[str, typing.Any]
        dictionary of sampled parameter values from the Latin Hypercube sampling
    """
    # ensure kwargs are not altered by making copies before sampling values
    new_kwargs = copy.deepcopy(kwargs)
    if regional_grav_kwargs is not None:
        new_regional_grav_kwargs = copy.deepcopy(regional_grav_kwargs)
    else:
        new_regional_grav_kwargs = None
    if starting_topography_kwargs is not None:
        new_starting_topography_kwargs = copy.deepcopy(starting_topography_kwargs)
    else:
        new_starting_topography_kwargs = None

    # print("before loop")
    # print(new_regional_grav_kwargs["constraints_df"].upward.mean())
    # print(new_starting_topography_kwargs["constraints_df"].upward.mean())
    # print(new_kwargs["constraints_df"].upward.mean())

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
        log.info(
            "No pickle files starting with '%s' found, creating new files\n", fname
        )

        # create / overwrite pickle files
        with pathlib.Path(f"{fname}_params.pickle").open("wb") as _:
            pass
        with pathlib.Path(f"{fname}_grav_dfs.pickle").open("wb") as _:
            pass
        with pathlib.Path(f"{fname}_prism_dfs.pickle").open("wb") as _:
            pass
        starting_run = 0
    if starting_run == runs:
        log.info("all %s runs already complete, loading results from files.", runs)

    if sample_constraints is True:
        constraints_df = new_kwargs.get("constraints_df", None)
        sampled_constraints = copy.deepcopy(constraints_df)
        test_constraint_value = copy.deepcopy(constraints_df.upward.iloc[0])  # type: ignore[union-attr]

    if sample_gravity is True:
        grav_df = new_kwargs.get("grav_df", None)
        sampled_grav = copy.deepcopy(grav_df)
        test_grav_value = copy.deepcopy(grav_df.gravity_anomaly.iloc[0])  # type: ignore[union-attr]

    for i in tqdm(range(starting_run, runs), desc="stochastic ensemble"):
        if i == starting_run:
            log.info(
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
            new_kwargs.pop("grav_df", None)
            if grav_df is None:
                msg = "grav_df must be provided"
                raise ValueError(msg)

            # assert original gravity values are unaltered
            assert test_grav_value == grav_df.gravity_anomaly.iloc[0]

            sampled_grav["gravity_anomaly"] = rand.normal(  # type: ignore[index]
                grav_df.gravity_anomaly, grav_df.uncert
            )

            # low-pass filter the sampled gravity data
            if gravity_filter_width is not None:
                filtered_grav = utils.filter_grid(
                    sampled_grav.set_index(["northing", "easting"])  # type: ignore[union-attr]
                    .to_xarray()
                    .gravity_anomaly,
                    gravity_filter_width,
                    filt_type="lowpass",
                    pad_mode="linear_ramp",
                )
                sampled_grav["gravity_anomaly"] = filtered_grav.values.ravel()  # type: ignore[index]

            new_kwargs["grav_df"] = sampled_grav
        if sample_constraints is True:
            new_kwargs.pop("constraints_df")
            if constraints_df is None:
                msg = "constraints_df must be provided if sample_constraints is True"
                raise ValueError(msg)

            # assert original constraint values are unaltered
            assert test_constraint_value == constraints_df.upward.iloc[0]

            sampled_constraints = randomly_sample_data(
                seed=i,
                data_df=constraints_df,
                data_col="upward",
                uncert_col="uncert",
            )
            if (new_starting_topography_kwargs is not None) and (
                new_starting_topography_kwargs.get("constraints_df", None) is not None
            ):
                new_starting_topography_kwargs["constraints_df"] = sampled_constraints
            if (new_regional_grav_kwargs is not None) and (
                new_regional_grav_kwargs.get("constraints_df", None) is not None
            ):
                new_regional_grav_kwargs["constraints_df"] = sampled_constraints
            new_kwargs["constraints_df"] = sampled_constraints

        # if parameters provided, sampled and add back to kwargs
        if sampled_param_dict is not None:
            for k, v in sampled_param_dict.items():
                new_kwargs[k] = v["sampled_values"][i]
        if sampled_starting_topography_parameter_dict is not None:
            for k, v in sampled_starting_topography_parameter_dict.items():
                new_starting_topography_kwargs[k] = v["sampled_values"][i]  # type: ignore[index]
        if sampled_regional_misfit_parameter_dict is not None:
            for k, v in sampled_regional_misfit_parameter_dict.items():
                new_regional_grav_kwargs[k] = v["sampled_values"][i]  # type: ignore[index]

        # define what needs to be done depending on what parameters are sampled
        if sample_gravity is True:
            calculate_starting_gravity = True
        if sample_constraints is True:
            create_starting_topography = True
        if sampled_param_dict is not None:  # noqa: SIM102
            if ("density_contrast" in sampled_param_dict) or (
                "zref" in sampled_param_dict
            ):
                create_starting_prisms = True
        if sampled_starting_topography_parameter_dict is not None:
            create_starting_topography = True
        if sampled_regional_misfit_parameter_dict is not None:
            calculate_regional_misfit = True
        # if certain things are recalculated, other must be as well
        if create_starting_topography is True:
            create_starting_prisms = True
        if create_starting_prisms is True:
            calculate_starting_gravity = True
        if calculate_starting_gravity is True:
            calculate_regional_misfit = True

        # run inversion
        with utils._log_level(logging.ERROR):  # pylint: disable=protected-access
            inv_results = inversion.run_inversion_workflow(
                create_starting_topography=create_starting_topography,
                create_starting_prisms=create_starting_prisms,
                calculate_starting_gravity=calculate_starting_gravity,
                calculate_regional_misfit=calculate_regional_misfit,  # pylint: disable=possibly-used-before-assignment
                regional_grav_kwargs=new_regional_grav_kwargs,
                starting_topography_kwargs=new_starting_topography_kwargs,
                fname=f"{fname}_{i}",
                **new_kwargs,
            )

        # get results
        prism_df, final_grav_df, params, _ = inv_results  # type: ignore[assignment]

        # add run number to the parameter values
        params["run_num"] = i  # type: ignore[call-overload]

        # save results
        with pathlib.Path(f"{fname}_params.pickle").open("ab") as file:
            pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)
        with pathlib.Path(f"{fname}_grav_dfs.pickle").open("ab") as file:
            pickle.dump(final_grav_df, file, protocol=pickle.HIGHEST_PROTOCOL)
        with pathlib.Path(f"{fname}_prism_dfs.pickle").open("ab") as file:
            pickle.dump(prism_df, file, protocol=pickle.HIGHEST_PROTOCOL)

        log.debug("Finished inversion %s of %s for stochastic ensemble", i + 1, runs)

    # load pickle files
    params = []
    with pathlib.Path(f"{fname}_params.pickle").open("rb") as file:
        while 1:
            try:
                params.append(pickle.load(file))
            except EOFError:
                break
    grav_dfs = []
    with pathlib.Path(f"{fname}_grav_dfs.pickle").open("rb") as file:
        while 1:
            try:
                grav_dfs.append(pickle.load(file))
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
        grav_dfs,
        prism_dfs,
        sampled_param_dict,
    )  # type: ignore[return-value]


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

        weighted_mean = sum(g * w for g, w in zip(da_list, weights)) / sum(weights)
        weighted_mean = weighted_mean.rename("weighted_mean")

        # from https://stackoverflow.com/questions/30383270/how-do-i-calculate-the-standard-deviation-between-weighted-measurements
        weighted_var = (
            sum(w * (g - weighted_mean) ** 2 for g, w in zip(da_list, weights))
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
    stats = []
    for g in grids:
        if g is not None:
            stats.append(g)

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
    return xr.merge(renamed_grids)


def merged_stats(
    results: tuple[typing.Any],
    plot: bool = True,
    constraints_df: pd.DataFrame | None = None,
    weight_by: str = "residual",
    region: tuple[float, float, float, float] | None = None,
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
    _, grav_dfs, prism_dfs, _ = results  # type: ignore[misc]

    # get merged dataset
    merged = merge_simulation_results(
        [df.set_index(["northing", "easting"]).to_xarray()["topo"] for df in prism_dfs]
    )

    # get final gravity residual RMS of each model
    if weight_by == "residual":
        # get the RMS of the final gravity residual of each model
        weight_vals = [utils.rmse(df[list(df.columns)[-1]]) for df in grav_dfs]
        # convert residuals into weights
        weights = [1 / (x**2) for x in weight_vals]
    # get constraint point RMSE of each model
    elif weight_by == "constraints":
        weight_vals = []
        for df in prism_dfs:
            ds = df.set_index(["northing", "easting"]).to_xarray()
            bed = ds["topo"]
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
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error("plotting failed with error: %s", e)

    return stats_ds
