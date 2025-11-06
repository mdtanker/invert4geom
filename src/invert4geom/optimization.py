import copy  # pylint: disable=too-many-lines
import logging
import math
import multiprocessing
import os
import pathlib
import pickle
import random
import re
import subprocess
import typing
import warnings

import harmonica as hm
import joblib
import numpy as np
import optuna
import pandas as pd
import psutil
import verde as vd
import xarray as xr
from numpy.typing import NDArray
from tqdm.autonotebook import tqdm

from invert4geom import (
    cross_validation,
    inversion,
    logger,
    plotting,
    regional,
    utils,
)

warnings.simplefilter(
    "ignore",
    category=optuna.exceptions.ExperimentalWarning,
)


def available_cpu_count() -> typing.Any:
    """
    Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program

    Adapted from https://stackoverflow.com/a/1006301/18686384
    """

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        # m = re.search(r"(?m)^Cpus_allowed:\s*(.*)$", open("/proc/self/status").read())
        with pathlib.Path("/proc/self/status").open(encoding="utf8") as f:
            m = re.search(r"(?m)^Cpus_allowed:\s*(.*)$", f.read())
        if m:
            res = bin(int(m.group(1).replace(",", ""), 16)).count("1")
            if res > 0:
                return res
    except OSError:
        pass

    # Python 2.6+
    try:
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        return psutil.cpu_count()  # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf("SC_NPROCESSORS_ONLN"))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ["NUMBER_OF_PROCESSORS"])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # BSD
    try:
        with subprocess.Popen(
            ["sysctl", "-n", "hw.ncpu"], stdout=subprocess.PIPE
        ) as sysctl:
            # sysctl = subprocess.Popen(["sysctl", "-n", "hw.ncpu"],
            # stdout=subprocess.PIPE)
            sc_std_out = sysctl.communicate()[0]
            res = int(sc_std_out)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        # res = open("/proc/cpuinfo").read().count("processor\t:")
        with pathlib.Path("/proc/cpuinfo").open(encoding="utf8") as f:
            res = f.read().count("processor\t:")

        if res > 0:
            return res
    except OSError:
        pass

    # Solaris
    try:
        pseudo_devices = os.listdir("/devices/pseudo/")  # noqa: PTH208
        res = 0
        for pds in pseudo_devices:
            if re.match(r"^cpuid@[0-9]+$", pds):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            # dmesg = open("/var/run/dmesg.boot").read()
            with pathlib.Path("/var/run/dmesg.boot").open(encoding="utf8") as f:
                dmesg = f.read()
            # dmesg = pathlib.Path("/var/run/dmesg.boot").open().read()
        except OSError:
            with subprocess.Popen(["dmesg"], stdout=subprocess.PIPE) as dmesg_process:
                # dmesg_process = subprocess.Popen(["dmesg"], stdout=subprocess.PIPE)
                dmesg = dmesg_process.communicate()[0]  # type: ignore[assignment]

        res = 0
        while "\ncpu" + str(res) + ":" in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    msg = "Can not determine number of CPUs on this system"
    raise Exception(msg)  # pylint: disable=broad-exception-raised


def run_optuna(
    study: optuna.study.Study,
    objective: typing.Callable[..., float],
    n_trials: int,
    storage: optuna.storages.BaseStorage | None = None,
    maximize_cpus: bool = True,
    parallel: bool = False,
    progressbar: bool | None = None,
    callbacks: typing.Any | None = None,
) -> optuna.study.Study:
    """
    Run optuna optimization, optionally in parallel. Pre-define the study, and objective
    function, and if parallel is True, the storage (preferably with JournalStorage) and
    study name.
    """
    optuna.logging.set_verbosity(optuna.logging.WARN)

    # set up parallel processing and run optimization
    if parallel is True:
        if progressbar is None:
            progressbar = False
        if storage is None:
            msg = "if running in parallel, must provide an Optuna storage object"
            raise ValueError(msg)
        study_name = study.study_name

        def optimize_study(
            study_name: str,
            storage: optuna.storages.BaseStorage,
            objective: typing.Callable[..., float],
            n_trials: int,
        ) -> None:
            study = optuna.load_study(study_name=study_name, storage=storage)
            optuna.logging.set_verbosity(optuna.logging.WARN)
            study.optimize(
                objective,
                n_trials=n_trials,
            )

        _optuna_set_cores(
            n_trials=n_trials,
            optimize_study=optimize_study,
            study_name=study_name,
            storage=storage,
            objective=objective,
            max_cores=maximize_cpus,
        )

        # reload the study
        return optuna.load_study(
            study_name=study_name,
            storage=storage,
        )

    # run in normal, non-parallel mode
    if progressbar is None:
        progressbar = True
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks,
        show_progress_bar=progressbar,
    )

    return study


def _optuna_set_cores(
    n_trials: int,
    optimize_study: typing.Callable[..., None],
    study_name: str,
    storage: typing.Any,
    objective: typing.Callable[..., float],
    max_cores: bool = True,
) -> None:
    """
    Set up optuna optimization in parallel splitting up the number of trials over either
    all available cores or giving each available core 1 trial.
    """
    if max_cores:
        # get available cores (UNIX and Windows)
        # num_cores = len(psutil.Process().cpu_affinity())
        num_cores = available_cpu_count()

        # set trials per job
        trials_per_job = math.ceil(n_trials / num_cores)

        # set number of jobs
        n_jobs = num_cores if n_trials >= num_cores else n_trials
    else:
        trials_per_job = 1
        n_jobs = int(n_trials / trials_per_job)
    logger.info(
        "Running %s trials with %s jobs with up to %s trials per job",
        n_trials,
        n_jobs,
        trials_per_job,
    )
    try:
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(optimize_study)(
                study_name,
                storage,
                objective,
                n_trials=trials_per_job,
            )
            for i in range(n_trials)
        )
    except FileNotFoundError:
        logger.exception("FileNotFoundError occurred in parallel optimization")
        pathlib.Path(f"{study_name}.log.lock").unlink(missing_ok=True)
        pathlib.Path(f"{study_name}.lock").unlink(missing_ok=True)


def warn_parameter_at_limits(
    trial: optuna.trial.FrozenTrial,
) -> None:
    """
    Warn if any best parameter values are at their limits

    Parameters
    ----------
    trial : optuna.trial.FrozenTrial
       optuna trial, most likely should be the best trial.
    """
    for k, v in trial.params.items():
        dist = trial.distributions.get(k)
        lims = (dist.high, dist.low)
        if v in lims:
            logger.warning(
                "Best %s value (%s) is at the limit of provided values "
                "%s and thus is likely not a global minimum, expand the range of "
                "values tested to ensure the best parameter value is found.",
                k,
                v,
                lims,
            )


def log_optuna_results(
    trial: optuna.trial.FrozenTrial,
) -> None:
    """
    Log the results of an optuna trial

    Parameters
    ----------
    trial : optuna.trial.FrozenTrial
        optuna trial
    """

    logger.info("Trial with best score: ")
    logger.info("\ttrial number: %s", trial.number)
    logger.info("\tparameter: %s", trial.params)
    logger.info("\tscores: %s", trial.values)


def _create_regional_separation_study(
    optimize_on_true_regional_misfit: bool,
    separate_metrics: bool,
    sampler: optuna.samplers.BaseSampler,
    true_regional: xr.DataArray | None = None,
    parallel: bool = True,
    fname: str | None = None,
) -> tuple[optuna.study.Study, optuna.storages.BaseStorage | None]:
    """
    Creates a study, sets directions and metric names based on the input parameters.

    Parameters
    ----------
    optimize_on_true_regional_misfit : bool
        choose to optimize on the true regional misfit instead of the residual misfit at
        constraints and the residual misfit amplitude.
    separate_metrics : bool
        choose to optimize on the residual misfit at constraints and the residual misfit
        amplitude as separate metrics, as opposed to them as a ratio.
    sampler : optuna.samplers.BaseSampler
        sampler object
    true_regional : xarray.DataArray | None, optional
        grid of true regional values, by default None
    parallel : bool, optional
        inform whether the study should be run in run in parallel, by default True. If
        True, uses file storage, which slows down the optimization, but allows for
        running in parallel.
    fname : str | None, optional
        file name to save the study to, by default None

    Returns
    -------
    study : optuna.study.Study
        return a study object with direction, sampler, and metric names set
    storage : optuna.storages.BaseStorage | None
        return an optuna storage object if parallel is True, otherwise None
    """
    direction = None
    directions = None

    if fname is None:
        fname = f"tmp_{random.randint(0, 999)}"
    if parallel:
        pathlib.Path(f"{fname}.log").unlink(missing_ok=True)
        pathlib.Path(f"{fname}.lock").unlink(missing_ok=True)
        pathlib.Path(f"{fname}.log.lock").unlink(missing_ok=True)
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(f"{fname}.log"),
        )
    else:
        storage = None

    if optimize_on_true_regional_misfit is True:
        if true_regional is None:
            msg = (
                "if optimizing on true regional misfit, must provide true_regional grid"
            )
            raise ValueError(msg)
        direction = "minimize"
        metric_names = ["difference with true regional"]
        logger.info("optimizing on minimizing the true regional misfit")
    elif separate_metrics is True:
        directions = ["minimize", "maximize"]
        metric_names = ["residual at constraints", "amplitude of residual"]
    else:
        direction = "minimize"
        metric_names = ["combined scores"]

    study = optuna.create_study(
        direction=direction,
        directions=directions,
        sampler=sampler,
        study_name=fname,
        storage=storage,
        load_if_exists=False,
        pruner=DuplicateIterationPruner,
    )
    study.set_metric_names(metric_names)

    return study, storage


def _logging_callback(
    study: optuna.study.Study,
    frozen_trial: optuna.trial.FrozenTrial,
) -> None:
    """
    custom optuna callback, only log trial info if it's the best value yet.

    Parameters
    ----------
    study : optuna.study.Study
        optuna study
    frozen_trial : optuna.trial.FrozenTrial
        current trial
    """
    optuna.logging.set_verbosity(optuna.logging.WARN)
    if study._is_multi_objective() is False:  # pylint: disable=protected-access
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        if previous_best_value != study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            logger.info(
                "Trial %s finished with best value: %s and parameters: %s.",
                frozen_trial.number,
                frozen_trial.value,
                frozen_trial.params,
            )
    elif frozen_trial.number in [n.number for n in study.best_trials]:
        msg = (
            "Trial %s is on the Pareto front with value 1: %s, value 2: %s and "
            "parameters: %s."
        )
        logger.info(
            msg,
            frozen_trial.number,
            frozen_trial.values[0],  # noqa: PD011
            frozen_trial.values[1],  # noqa: PD011
            frozen_trial.params,
        )


def _warn_limits_better_than_trial_1_param(
    study: optuna.study.Study,
    trial: optuna.trial.FrozenTrial,
) -> None:
    """
    custom optuna callback, warn if limits provide better score than current trial

    Parameters
    ----------
    study : optuna.study.Study
        optuna study
    trial : optuna.trial.FrozenTrial
        current trial
    """
    # exit if one of first 2 trials (lower and upper limits)
    if trial.number < 2:
        return

    # get scores of lower and upper limits
    # this assumes that the first two trials are the lower and upper limits set by
    # study.enqueue_trial()
    lower_limit_score = study.trials[0].value
    upper_limit_score = study.trials[1].value
    msg = (
        "Current trial (#%s, %s) has a worse score (%s) than either of the lower "
        "(%s) or upper (%s) parameter value limits, it might be best to stop the "
        "study and expand the limits."
    )
    if lower_limit_score is None:
        return
    if upper_limit_score is None:
        return
    # if study direction is minimize
    if study.direction == optuna.study.StudyDirection.MINIMIZE:
        # if current trial is worse than either limit, log a warning
        if trial.values[0] > max(lower_limit_score, upper_limit_score):  # noqa: PD011
            logger.info(
                msg,
                trial.number,
                trial.params,
                trial.values[0],  # noqa: PD011
                lower_limit_score,
                upper_limit_score,
            )
        else:
            pass

    # if study direction is maximize
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        # if current trial is worse than either limit, log a warning
        if trial.values[0] < min(lower_limit_score, upper_limit_score):  # noqa: PD011
            logger.info(
                msg,
                trial.number,
                trial.params,
                trial.values[0],  # noqa: PD011
                lower_limit_score,
                upper_limit_score,
            )
        else:
            pass


def _warn_limits_better_than_trial_multi_params(
    study: optuna.study.Study,
    trial: optuna.trial.FrozenTrial,
) -> None:
    """
    custom optuna callback, warn if limits provide better score than current trial for
    multiple parameter optimization

    Parameters
    ----------
    study : optuna.study.Study
        optuna study
    trial : optuna.trial.FrozenTrial
        current trial
    """

    # number of parameters in the study
    num_params = len(trial.params)

    # get number of combos (2 params->4 trials, 3 params->8 trials etc.)
    num_combos = 2**num_params

    # exit if one of enqueued trials
    if trial.number < num_combos:
        return

    # get scores of combos of upper and lower limits of both parameters
    # this assumes that the first four trials are set by study.enqueue_trial()
    scores = []
    for i in range(num_combos):
        scores.append(study.trials[i].value)

    msg = (
        "Current trial (#%s, %s) has a worse score (%s) than any of the combinations "
        "of parameter value limits, it might be best to stop the study and expand the "
        "limits."
    )
    # if study direction is minimize
    if study.direction == optuna.study.StudyDirection.MINIMIZE:
        # if current trial is worse than either limit, log a warning
        if trial.values[0] > max(scores):  # noqa: PD011
            logger.info(
                msg,
                trial.number,
                trial.params,
                trial.values[0],  # noqa: PD011
            )
        else:
            pass

    # if study direction is maximize
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        # if current trial is worse than either limit, log a warning
        try:
            if trial.values[0] < min(scores):  # noqa: PD011
                logger.info(
                    msg,
                    trial.number,
                    trial.params,
                    trial.values[0],  # noqa: PD011
                )
        except TypeError:
            pass
        else:
            pass


class OptimalInversionDamping:
    """
    Objective function to use in an Optuna optimization for finding the optimal damping
    regularization value for a gravity inversion. Used within function
    :func:`optimize_inversion_damping`.
    """

    def __init__(
        self,
        inversion_obj: inversion.Inversion,
        fname: str,
        damping_limits: tuple[float, float],
        rmse_as_median: bool = False,
        plot_grids: bool = False,
    ) -> None:
        self.inversion_obj = inversion_obj
        self.fname = fname
        self.rmse_as_median = rmse_as_median
        self.damping_limits = damping_limits
        self.plot_grids = plot_grids

    def __call__(self, trial: optuna.trial) -> float:
        """
        Parameters
        ----------
        trial : optuna.trial
            the trial to run

        Returns
        -------
        float
            the score of the eq_sources fit
        """
        self.inversion_obj.solver_damping = trial.suggest_float(
            "damping",
            self.damping_limits[0],
            self.damping_limits[1],
            log=True,
        )

        trial.set_user_attr("fname", f"{self.fname}_trial_{trial.number}")

        _new_inversion_obj = self.inversion_obj.grav_cv_score(
            rmse_as_median=self.rmse_as_median,
            plot=self.plot_grids,
            results_fname=trial.user_attrs.get("fname"),
        )

        # trial.set_user_attr("results", new_inversion_obj.damping_cv_results)

        return self.inversion_obj.gravity_cv_best_score  # type: ignore[return-value]


def optimize_inversion_damping(
    training_df: pd.DataFrame,  # noqa: ARG001
    testing_df: pd.DataFrame,  # noqa: ARG001
    n_trials: int,  # noqa: ARG001
    damping_limits: tuple[float, float],  # noqa: ARG001
    n_startup_trials: int | None = None,  # noqa: ARG001
    score_as_median: bool = False,  # noqa: ARG001
    sampler: optuna.samplers.BaseSampler | None = None,  # noqa: ARG001
    grid_search: bool = False,  # noqa: ARG001
    fname: str | None = None,  # noqa: ARG001
    plot_cv: bool = True,  # noqa: ARG001
    plot_grids: bool = False,  # noqa: ARG001
    logx: bool = True,  # noqa: ARG001
    logy: bool = True,  # noqa: ARG001
    progressbar: bool = True,  # noqa: ARG001
    parallel: bool = False,  # noqa: ARG001
    seed: int = 0,  # noqa: ARG001
    **kwargs: typing.Any,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `Inversion` class method `optimize_inversion_damping` instead
    """
    # pylint: disable=W0613
    msg = (
        "Function `optimize_inversion_damping` deprecated, use the `Inversion` class method "
        "`optimize_inversion_damping` instead"
    )
    raise DeprecationWarning(msg)


class OptimalInversionZrefDensity:
    """
    Objective function to use in an Optuna optimization for finding the optimal values
    for zref and or density contrast values for a gravity inversion. This class is used
    within the function `optimize_inversion_zref_density_contrast`. If using constraint
    point minimization for the regional separation, split constraints into testing and
    training sets and provide the testing set to argument `constraints_df` and the
    training set to the `constraints_df` argument of `regional_grav_kwargs`. To perform
    K-folds cross-validation, provide lists of constraints dataframes to the parameters
    where each dataframe in each list corresponds to fold.
    """

    def __init__(
        self,
        inversion_obj: inversion.Inversion,
        constraints_df: pd.DataFrame,
        fname: str,
        regional_grav_kwargs: dict[str, typing.Any],
        starting_topography: xr.Dataset | None = None,
        starting_topography_kwargs: dict[str, typing.Any] | None = None,
        zref_limits: tuple[float, float] | None = None,
        density_contrast_limits: tuple[float, float] | None = None,
        rmse_as_median: bool = False,
        progressbar: bool = True,
    ) -> None:
        self.inversion_obj = inversion_obj
        self.constraints_df = constraints_df.copy()
        self.fname = fname
        self.regional_grav_kwargs = copy.deepcopy(regional_grav_kwargs)
        self.zref_limits = zref_limits
        self.density_contrast_limits = density_contrast_limits
        self.starting_topography = starting_topography
        self.starting_topography_kwargs = copy.deepcopy(starting_topography_kwargs)
        self.rmse_as_median = rmse_as_median
        self.progressbar = progressbar

    def __call__(self, trial: optuna.trial) -> float:
        """
        Parameters
        ----------
        trial : optuna.trial
            the trial to run

        Returns
        -------
        float
            the score of the eq_sources fit
        """
        if self.inversion_obj.apply_weighting_grid is True:
            msg = (
                "Using the weighting grid within the inversion for regularization. "
                "This makes the inversion not update the topography at constraint "
                "points. Since constraint points are used to determine the scores "
                "within this cross validation, the scores will not be useful, giving "
                "biased results. Set `apply_weighting_grid` to False to continue."
            )
            raise ValueError(msg)

        if (self.zref_limits is None) & (self.density_contrast_limits is None):
            msg = "must provide either or both zref_limits and density_contrast_limits"
            raise ValueError(msg)

        if self.zref_limits is not None:
            self.inversion_obj.model = self.inversion_obj.model.assign_attrs(
                {
                    "zref": trial.suggest_float(
                        "zref",
                        self.zref_limits[0],
                        self.zref_limits[1],
                    )
                }
            )
        if self.density_contrast_limits is not None:
            self.inversion_obj.model = self.inversion_obj.model.assign_attrs(
                {
                    "density_contrast": trial.suggest_int(
                        "density_contrast",
                        self.density_contrast_limits[0],
                        self.density_contrast_limits[1],
                    )
                }
            )
        starting_topography_kwargs = copy.deepcopy(self.starting_topography_kwargs)

        reg_kwargs = copy.deepcopy(self.regional_grav_kwargs)

        constraints_warning = (
            "Using constraint point minimization technique for regional field "
            "estimation. This is not recommended as the constraint points are used "
            "for the density / reference level cross-validation scoring, which "
            "biases the scoring. Consider using a different method for regional "
            "field estimation, or separate constraints in training and testing "
            "sets and provide the training set to `regional_grav_kwargs` and the "
            "testing set to `constraints_df` to use for scoring."
        )
        logger.debug("prism model created and forward gravity calculated")
        ###
        ###
        # Single optimization
        ###
        ###
        if isinstance(self.constraints_df, pd.DataFrame):
            logger.debug("running single optimization")

            # raise warning about using constraint point minimization for regional
            # estimation
            if (reg_kwargs.get("method") in ["constraints", "constraints_cv"]) and (
                len(reg_kwargs.get("constraints_df")) == len(self.constraints_df)  # type: ignore[arg-type]
            ):
                assert isinstance(reg_kwargs.get("constraints_df"), pd.DataFrame)
                logger.warning(constraints_warning)
            # create starting topography model if not provided
            if self.starting_topography is None:
                msg = (
                    "starting_topography not provided, creating a starting topography "
                    "model with the supplied starting_topography_kwargs"
                )
                logger.info(msg)
                if starting_topography_kwargs is None:
                    msg = (
                        "must provide `starting_topography_kwargs` to be passed to the "
                        "function `utils.create_topography`."
                    )
                    raise ValueError(msg)
                if starting_topography_kwargs["method"] == "flat":
                    msg = "using zref to create a flat starting topography model"
                    logger.info(msg)
                    starting_topography_kwargs["upwards"] = (
                        self.inversion_obj.model.zref
                    )

            # create starting topography model if not provided
            if self.starting_topography is None:
                msg = (
                    "starting_topography not provided, creating a starting topography "
                    "model with the supplied starting_topography_kwargs"
                )
                logger.info(msg)
                if starting_topography_kwargs is None:
                    msg = (
                        "must provide `starting_topography_kwargs` to be passed to the "
                        "function `utils.create_topography`."
                    )
                    raise ValueError(msg)
                if starting_topography_kwargs["method"] == "flat":
                    msg = "using zref to create a flat starting topography model"
                    logger.info(msg)
                    starting_topography_kwargs["upwards"] = (
                        self.inversion_obj.model.zref
                    )

                starting_topo = utils.create_topography(
                    **starting_topography_kwargs,
                ).to_dataset(name="upward")
            else:
                if starting_topography_kwargs is not None:
                    msg = (
                        "starting_topography and starting_topography_kwargs provided, "
                        "please only provide one or the other."
                    )
                    raise ValueError(msg)
                starting_topo = self.starting_topography.copy()

            # update model with new topography
            self.inversion_obj.model = inversion.create_model(
                zref=self.inversion_obj.model.zref,
                density_contrast=self.inversion_obj.model.density_contrast,
                model_type=self.inversion_obj.model.model_type,
                starting_topography=starting_topo,
                upper_confining_layer=self.inversion_obj.model.upper_confining_layer,
                lower_confining_layer=self.inversion_obj.model.lower_confining_layer,
            )

            # calculate forward gravity of starting prism layer
            self.inversion_obj.data.inv.forward_gravity(self.inversion_obj.model)

            # calculate regional field
            with utils._log_level(logging.WARN):  # pylint: disable=protected-access
                self.inversion_obj.data.inv.regional_separation(**reg_kwargs)

            trial.set_user_attr("fname", f"{self.fname}_trial_{trial.number}")

            # run cross validation
            constraints_cv_object = self.inversion_obj.constraints_cv_score(
                results_fname=trial.user_attrs.get("fname"),
                constraints_df=self.constraints_df,
            )
            # log the termination reason
            logger.debug(
                "Trial %s termination reason: %s",
                trial.number,
                constraints_cv_object.params.get("Termination reason"),  # type: ignore[attr-defined]
            )

        ###
        ###
        # K-Folds optimization
        ###
        ###
        else:
            logger.debug("running k-folds optimization")

            training_constraints = reg_kwargs.pop("constraints_df", None)

            if starting_topography_kwargs is None:
                msg = (
                    "must provide `starting_topography_kwargs` to be passed to the "
                    "function `utils.create_topography`."
                )
                raise ValueError(msg)

            starting_topography_kwargs.pop("constraints_df", None)

            testing_constraints = self.constraints_df

            if training_constraints is None:
                msg = (
                    "must provide training constraints dataframes for regional "
                    "separation"
                )
                raise ValueError(msg)
            if isinstance(training_constraints, pd.DataFrame):
                msg = (
                    "must provide a list of training constraints dataframes for "
                    "cross-validation to parameter `constraints_df` of "
                    "`regional_grav_kwargs`."
                )
                raise ValueError(msg)

            assert len(training_constraints) == len(testing_constraints)

            # get list of folds
            folds = testing_constraints

            # progressbar for folds
            if self.progressbar is True:
                pbar = tqdm(
                    folds,
                    desc="Regional Estimation CV folds",
                )
            elif self.progressbar is False:
                pbar = folds
            else:
                msg = "progressbar must be a boolean"  # type: ignore[unreachable]
                raise ValueError(msg)

            logger.debug("Running %s folds", len(folds))
            # logger.debug(testing_constraints)
            # logger.debug(training_constraints)
            # for each fold, run CV
            scores: list[float] = []
            for i, _ in enumerate(pbar):
                logger.debug(training_constraints[i])

                # create starting topography model if not provided
                if self.starting_topography is None:
                    msg = (
                        "starting_topography not provided, creating a starting "
                        "topography model with the supplied starting_topography_kwargs"
                    )
                    logger.info(msg)
                    if starting_topography_kwargs["method"] == "flat":
                        msg = "using zref to create a flat starting topography model"
                        logger.info(msg)
                        starting_topography_kwargs["upwards"] = (
                            self.inversion_obj.model.zref
                        )
                    elif starting_topography_kwargs["method"] == "splines":
                        starting_topography_kwargs["constraints_df"] = (
                            training_constraints[i]
                        )

                    with utils.DuplicateFilter(logger):  # type: ignore[no-untyped-call]
                        starting_topo = utils.create_topography(
                            **starting_topography_kwargs,
                        ).to_dataset(name="upward")
                else:
                    starting_topo = self.starting_topography.copy()

                # update model with new topography
                self.inversion_obj.model = inversion.create_model(
                    zref=self.inversion_obj.model.zref,
                    density_contrast=self.inversion_obj.model.density_contrast,
                    model_type=self.inversion_obj.model.model_type,
                    starting_topography=starting_topo,
                    upper_confining_layer=self.inversion_obj.model.upper_confining_layer,
                    lower_confining_layer=self.inversion_obj.model.lower_confining_layer,
                )

                # calculate forward gravity of starting prism layer
                self.inversion_obj.data.inv.forward_gravity(self.inversion_obj.model)

                # calculate regional field
                with utils._log_level(logging.WARN):  # pylint: disable=protected-access
                    self.inversion_obj.data.inv.regional_separation(
                        constraints_df=training_constraints[i],
                        **reg_kwargs,
                    )

                trial.set_user_attr("fname", f"{self.fname}_trial_{trial.number}")

                # run cross validation
                constraints_cv_object = self.inversion_obj.constraints_cv_score(
                    constraints_df=testing_constraints[i],
                    results_fname=trial.user_attrs.get("fname"),
                )
                scores.append(self.inversion_obj.constraints_cv_best_score)  # type: ignore[arg-type]

                # log the termination reason
                logger.debug(
                    "Trial %s termination reason: %s",
                    trial.number,
                    constraints_cv_object.params.get("Termination reason"),  # type: ignore[attr-defined]
                )
            # get mean of scores of all folds
            self.inversion_obj.constraints_cv_best_score = np.mean(scores)

        return self.inversion_obj.constraints_cv_best_score  # type: ignore[return-value]


def optimize_inversion_zref_density_contrast(
    grav_df: pd.DataFrame,  # noqa: ARG001
    constraints_df: pd.DataFrame | list[pd.DataFrame],  # noqa: ARG001
    n_trials: int,  # noqa: ARG001
    n_startup_trials: int | None = None,  # noqa: ARG001
    starting_topography: xr.DataArray | None = None,  # noqa: ARG001
    zref_limits: tuple[float, float] | None = None,  # noqa: ARG001
    density_contrast_limits: tuple[float, float] | None = None,  # noqa: ARG001
    zref: float | None = None,  # noqa: ARG001
    density_contrast: float | None = None,  # noqa: ARG001
    starting_topography_kwargs: dict[str, typing.Any] | None = None,  # noqa: ARG001
    regional_grav_kwargs: dict[str, typing.Any] | None = None,  # noqa: ARG001
    score_as_median: bool = False,  # noqa: ARG001
    sampler: optuna.samplers.BaseSampler | None = None,  # noqa: ARG001
    grid_search: bool = False,  # noqa: ARG001
    fname: str | None = None,  # noqa: ARG001
    plot_cv: bool = True,  # noqa: ARG001
    logx: bool = False,  # noqa: ARG001
    logy: bool = False,  # noqa: ARG001
    progressbar: bool = True,  # noqa: ARG001
    parallel: bool = False,  # noqa: ARG001
    fold_progressbar: bool = True,  # noqa: ARG001
    seed: int = 0,  # noqa: ARG001
    **kwargs: typing.Any,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `Inversion` class method `optimize_inversion_zref_density_contrast` instead
    """
    # pylint: disable=W0613
    msg = (
        "Function `optimize_inversion_zref_density_contrast` deprecated, use the `Inversion` class method "
        "`optimize_inversion_zref_density_contrast` instead"
    )
    raise DeprecationWarning(msg)


class OptimalEqSourceParams:
    """
    Objective function to use in an Optuna optimization for finding the optimal
    equivalent source parameters for fitting to gravity data.
    """

    def __init__(
        self,
        depth_limits: tuple[float, float] | None = None,
        block_size_limits: tuple[float, float] | None = None,
        damping_limits: tuple[float, float] | None = None,
        **kwargs: typing.Any,
    ) -> None:
        self.depth_limits = depth_limits
        self.block_size_limits = block_size_limits
        self.damping_limits = damping_limits
        self.kwargs = kwargs

    def __call__(self, trial: optuna.trial) -> float:
        """
        Parameters
        ----------
        trial : optuna.trial
            the trial to run

        Returns
        -------
        float
            the score of the eq_sources fit
        """
        kwargs = copy.deepcopy(self.kwargs)
        # get parameters provided not as limits
        depth = kwargs.pop("depth", "default")
        # calculate 4.5 times the mean distance between points
        if depth == "default":
            depth = 4.5 * np.mean(
                vd.median_distance(
                    (kwargs.get("coordinates")[0], kwargs.get("coordinates")[1]),  # type: ignore[unused-ignore, index]
                    k_nearest=1,
                )
            )
        block_size = kwargs.pop("block_size", None)
        damping = kwargs.pop("damping", None)

        # replace with suggested values if limits provided
        if self.depth_limits is not None:
            depth = trial.suggest_float(
                "depth",
                self.depth_limits[0],
                self.depth_limits[1],
            )
        if self.block_size_limits is not None:
            block_size = trial.suggest_float(
                "block_size",
                self.block_size_limits[0],
                self.block_size_limits[1],
            )
        if self.damping_limits is not None:
            damping = trial.suggest_float(
                "damping",
                self.damping_limits[0],
                self.damping_limits[1],
                log=True,
            )

        try:
            score = cross_validation.eq_sources_score(
                damping=damping,
                depth=depth,
                block_size=block_size,
                **kwargs,
            )
        except ValueError as e:
            logger.error(e)
            msg = "score could not be calculated, returning NaN"
            logger.warning(msg)
            score = np.nan
        return score


def optimize_eq_source_params(
    coordinates: tuple[pd.Series | NDArray, pd.Series | NDArray, pd.Series | NDArray],
    data: pd.Series | NDArray,
    n_trials: int = 100,
    damping_limits: tuple[float, float] | None = None,
    depth_limits: tuple[float, float] | None = None,
    block_size_limits: tuple[float, float] | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    plot: bool = False,
    progressbar: bool = True,
    parallel: bool = False,
    fname: str | None = None,
    seed: int = 0,
    **kwargs: typing.Any,
) -> tuple[optuna.study, hm.EquivalentSources]:
    """
    Use Optuna to find the optimal parameters for fitting equivalent sources to gravity
    data. The 3 parameters are damping, depth, and block size. Any or all of these can
    be optimized at the same time. Provide upper and lower limits for each parameter,
    or if you don't want to optimize a parameter, provide a constant value of the
    parameter in the kwargs.

    Parameters
    ----------
    coordinates : tuple[pandas.Series | numpy.ndarray, pandas.Series | numpy.ndarray, \
            pandas.Series | numpy.ndarray]
        tuple of coordinates in the order (easting, northing, upward) for the gravity
        observation locations.
    data : pandas.Series | numpy.ndarray
        gravity data values
    n_trials : int, optional
        number of trials to run, by default 100
    damping_limits : tuple[float, float], optional
        damping parameter limits, by default (0, 10**3)
    depth_limits : tuple[float, float], optional
        source depth limits (positive downwards) in meters, by default (0, 10e6)
    block_size_limits : tuple[float, float] | None, optional
        block size limits in meters, by default None
    sampler : optuna.samplers.BaseSampler | None, optional
        specify which Optuna sampler to use, by default GPsampler
    plot : bool, optional
        plot the resulting optimization figures, by default False
    progressbar : bool, optional
        add a progressbar, by default True
    parallel : bool, optional
        run the optimization in parallel, by default False
    fname : str | None, optional
        file name to save the study to, by default None
    seed : int, optional
        random seed for the samplers, by default 0
    kwargs : typing.Any
        additional keyword arguments to pass to `OptimalEqSourceParams`, which are
        passed to `eq_sources_score`. These can include parameters to pass to
        `harmonica.EquivalentSources`; "damping", "depth", "block_size",
        "parallel", and "dtype", or parameters to pass to `vd.cross_val_score`;
        "delayed", or "weights".

    Returns
    -------
    study : optuna.study.Study
        the completed optuna study
    eqs : harmonica.EquivalentSources
        the fitted equivalent sources model
    """
    optuna.logging.set_verbosity(optuna.logging.WARN)

    kwargs = copy.deepcopy(kwargs)

    study_fname = f"tmp_{random.randint(0, 999)}" if fname is None else fname

    if parallel:
        pathlib.Path(f"{study_fname}.log").unlink(missing_ok=True)
        pathlib.Path(f"{study_fname}.lock").unlink(missing_ok=True)
        pathlib.Path(f"{study_fname}.log.lock").unlink(missing_ok=True)
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(f"{study_fname}.log"),
        )
    else:
        storage = None

    # get number of parameters included in optimization
    num_params = sum(
        x is not None for x in [depth_limits, block_size_limits, damping_limits]
    )

    if num_params == 0:
        msg = (
            "No parameters to optimize, must provide at least one set of limits for "
            "damping, depth, or block size."
        )
        raise ValueError(msg)

    # define number of startup trials, whichever is bigger between 1/4 of trials, or
    # 4 x the number of parameters
    n_startup_trials = max(num_params * 4, int(n_trials / 4))

    logger.info("using %s startup trials", n_startup_trials)
    if n_startup_trials >= n_trials:
        logger.warning(
            "n_startup_trials is >= n_trials resulting in all trials sampled from "
            "a QMC sampler instead of the GP sampler",
        )

    # QMC's Sobol' sequence is best with n_trials as a power of 2
    # e.g. 2, 4, 8, 16, ...
    # def next_power_of_2(x):
    #     return 1 if x == 0 else 2**(x - 1).bit_length()
    # n_startup_trials = next_power_of_2(n_startup_trials)

    # create study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.QMCSampler(
            seed=seed,
            qmc_type="halton",
            scramble=True,
        ),
        load_if_exists=False,
        study_name=study_fname,
        storage=storage,
        pruner=DuplicateIterationPruner,
    )
    # explicitly add the limits as trials
    to_enqueue = []
    if block_size_limits is not None:
        block_size_trials = [
            {"block_size": block_size_limits[0]},
            {"block_size": block_size_limits[1]},
        ]
        to_enqueue.append(block_size_trials)
    if damping_limits is not None:
        damping_trials = [
            {"damping": damping_limits[0]},
            {"damping": damping_limits[1]},
        ]
        to_enqueue.append(damping_trials)
    if depth_limits is not None:
        depth_trials = [{"depth": depth_limits[0]}, {"depth": depth_limits[1]}]
        to_enqueue.append(depth_trials)

    # get 2 lists of lists of dicts to enqueue (2 trials)
    to_enqueue = np.array(to_enqueue).transpose()

    for i in to_enqueue:
        # turn list of dicts into single dict
        x = {k: v for d in i for k, v in d.items()}
        study.enqueue_trial(x, skip_if_exists=True)

    # define the objective function
    objective = OptimalEqSourceParams(
        depth_limits=depth_limits,
        block_size_limits=block_size_limits,
        damping_limits=damping_limits,
        coordinates=coordinates,
        data=data,
        **kwargs,
    )

    logger.debug("starting eq_source parameter optimization")
    # pylint: enable=duplicate-code
    # ignore skLearn LinAlg warnings
    with (utils._environ(PYTHONWARNINGS="ignore")) and (utils.DuplicateFilter(logger)):  # type: ignore[no-untyped-call, truthy-bool] # pylint: disable=protected-access
        # run startup trials with QMC low-discrepancy sampling
        study = run_optuna(
            study=study,
            storage=storage,
            objective=objective,
            n_trials=n_startup_trials,
            # callbacks=callbacks,
            maximize_cpus=True,
            parallel=parallel,
            progressbar=progressbar,
        )

    # continue with remaining trials with user-defined sampler
    # if sampler not provided, used GPsampler as default
    if sampler is None:
        sampler = optuna.samplers.GPSampler(
            n_startup_trials=0,
            seed=seed,
            deterministic_objective=True,
        )
    study.sampler = sampler
    with utils.DuplicateFilter(logger):  # type: ignore[no-untyped-call]
        study = run_optuna(
            study=study,
            storage=storage,
            objective=objective,
            n_trials=n_trials - n_startup_trials,
            # callbacks=callbacks,
            maximize_cpus=True,
            parallel=parallel,
            progressbar=progressbar,
        )
    # pylint: disable=duplicate-code
    best_trial = study.best_trial

    # warn if any best parameter values are at their limits
    warn_parameter_at_limits(best_trial)

    # log the results of the best trial
    log_optuna_results(best_trial)

    best_damping = best_trial.params.get("damping", None)
    best_depth = best_trial.params.get("depth", None)
    best_block_size = best_trial.params.get("block_size", None)

    if best_damping is None:
        try:
            best_damping = kwargs["damping"]
        except KeyError:
            msg = (
                "No damping parameter value found in best params or kwargs, setting to "
                "'None'"
            )
            logger.warning(msg)
            best_damping = None
    if best_depth is None:
        try:
            best_depth = kwargs["depth"]
        except KeyError:
            msg = (
                "No depth parameter value found in best params or kwargs, setting to "
                "'default' (4.5 times mean distance between points)"
            )
            logger.warning(msg)
            best_depth = "default"
    if best_depth == "default":
        best_depth = 4.5 * np.mean(
            vd.median_distance((coordinates[0], coordinates[1]), k_nearest=1)
        )
    if best_block_size is None:
        try:
            best_block_size = kwargs["block_size"]
        except KeyError:
            msg = (
                "No block size parameter value found in best params or kwargs, setting "
                "to 'None'"
            )
            logger.warning(msg)
            best_block_size = None

    # refit EqSources with best parameters
    eqs = hm.EquivalentSources(
        damping=best_damping,
        depth=best_depth,
        block_size=best_block_size,
        points=kwargs.pop("points", None),
        parallel=kwargs.pop("parallel", True),
        dtype=kwargs.pop("dtype", "float64"),
    )
    eqs.fit(coordinates, data, weights=kwargs.pop("weights", None))

    # save study
    if study_fname is not None:
        # remove if exists
        pathlib.Path(f"{study_fname}.pickle").unlink(missing_ok=True)

        # save study to pickle
        with pathlib.Path(f"{study_fname}.pickle").open("wb") as f:
            pickle.dump(study, f)

    if plot is True:
        try:
            plotting.plot_optuna_figures(
                study,
                target_names=["score"],
                plot_history=False,
                plot_slice=True,
                plot_importance=True,
                include_duration=False,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("plotting failed with error: %s", e)

    return study, eqs


class OptimizeRegionalTrend:
    """
    Objective function to use in an Optuna optimization for finding the optimal trend
    order for estimation the regional component of gravity misfit.
    """

    def __init__(
        self,
        trend_limits: tuple[int, int],
        optimize_on_true_regional_misfit: bool = False,
        separate_metrics: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        self.trend_limits = trend_limits
        self.optimize_on_true_regional_misfit = optimize_on_true_regional_misfit
        self.separate_metrics = separate_metrics
        self.kwargs = kwargs

    def __call__(self, trial: optuna.trial) -> tuple[float, float] | float:
        """
        Parameters
        ----------
        trial : optuna.trial
            the trial to run

        Returns
        -------
        float
            the scores
        """

        trend = trial.suggest_int(
            "trend",
            self.trend_limits[0],
            self.trend_limits[1],
        )

        with utils._log_level(logging.WARN):  # pylint: disable=protected-access
            residual_constraint_score, residual_amplitude_score, true_reg_score, _ = (
                cross_validation.regional_separation_score(
                    method="trend",
                    trend=trend,
                    **self.kwargs,
                )
            )

        trial.set_user_attr("true_reg_score", true_reg_score)

        if self.optimize_on_true_regional_misfit is True:
            trial.set_user_attr("residual constraint score", residual_constraint_score)
            trial.set_user_attr("residual amplitude score", residual_amplitude_score)
            return true_reg_score  # type: ignore[return-value]

        if self.separate_metrics is True:
            return residual_constraint_score, residual_amplitude_score

        # combine the two metrics into one
        return residual_constraint_score / residual_amplitude_score


class OptimizeRegionalFilter:
    """
    Objective function to use in an Optuna optimization for finding the optimal filter
    width for estimation the regional component of gravity misfit.
    """

    def __init__(
        self,
        filter_width_limits: tuple[float, float],
        optimize_on_true_regional_misfit: bool = False,
        separate_metrics: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        self.filter_width_limits = filter_width_limits
        self.optimize_on_true_regional_misfit = optimize_on_true_regional_misfit
        self.separate_metrics = separate_metrics
        self.kwargs = kwargs

    def __call__(self, trial: optuna.trial) -> float:
        """
        Parameters
        ----------
        trial : optuna.trial
            the trial to run

        Returns
        -------
        float
            the scores
        """

        filter_width = trial.suggest_float(
            "filter_width",
            self.filter_width_limits[0],
            self.filter_width_limits[1],
        )

        with utils._log_level(logging.WARN):  # pylint: disable=protected-access
            residual_constraint_score, residual_amplitude_score, true_reg_score, _ = (
                cross_validation.regional_separation_score(
                    method="filter",
                    filter_width=filter_width,
                    **self.kwargs,
                )
            )

        trial.set_user_attr("true_reg_score", true_reg_score)

        if self.optimize_on_true_regional_misfit is True:
            trial.set_user_attr("residual constraint score", residual_constraint_score)
            trial.set_user_attr("residual amplitude score", residual_amplitude_score)
            return true_reg_score  # type: ignore[return-value]

        if self.separate_metrics is True:
            return residual_constraint_score, residual_amplitude_score  # type: ignore[return-value]

        # combine the two metrics into one
        return residual_constraint_score / residual_amplitude_score


class OptimizeRegionalEqSources:
    """
    Objective function to use in an Optuna optimization for finding the optimal
    equivalent source parameters for estimation the regional component of gravity
    misfit.
    """

    def __init__(
        self,
        depth_limits: tuple[float, float] | None = None,
        block_size_limits: tuple[float, float] | None = None,
        damping_limits: tuple[float, float] | None = None,
        grav_obs_height_limits: tuple[float, float] | None = None,
        optimize_on_true_regional_misfit: bool = False,
        separate_metrics: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        self.depth_limits = depth_limits
        self.block_size_limits = block_size_limits
        self.damping_limits = damping_limits
        self.grav_obs_height_limits = grav_obs_height_limits
        self.optimize_on_true_regional_misfit = optimize_on_true_regional_misfit
        self.separate_metrics = separate_metrics
        self.kwargs = kwargs

    def __call__(self, trial: optuna.trial) -> float:
        """
        Parameters
        ----------
        trial : optuna.trial
            the trial to run

        Returns
        -------
        float
            the scores
        """

        if self.depth_limits is not None:
            depth = trial.suggest_float(
                "depth",
                self.depth_limits[0],
                self.depth_limits[1],
            )
        else:
            depth = self.kwargs.get("depth", None)
            if depth is None:
                msg = "must provide depth if depth_limits not provided"
                raise ValueError(msg)

        if self.block_size_limits is not None:
            block_size = trial.suggest_float(
                "block_size",
                self.block_size_limits[0],
                self.block_size_limits[1],
            )
        else:
            block_size = self.kwargs.get("block_size", None)

        if self.damping_limits is not None:
            damping = trial.suggest_float(
                "damping",
                self.damping_limits[0],
                self.damping_limits[1],
                log=True,
            )
        else:
            damping = self.kwargs.get("damping", None)

        if self.grav_obs_height_limits is not None:
            grav_obs_height = trial.suggest_float(
                "grav_obs_height",
                self.grav_obs_height_limits[0],
                self.grav_obs_height_limits[1],
            )
        else:
            grav_obs_height = self.kwargs.get("grav_obs_height", None)

        new_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key not in ["depth", "block_size", "damping", "grav_obs_height"]
        }

        with utils._log_level(logging.WARN):  # pylint: disable=protected-access
            residual_constraint_score, residual_amplitude_score, true_reg_score, _ = (
                cross_validation.regional_separation_score(
                    method="eq_sources",
                    depth=depth,
                    block_size=block_size,
                    damping=damping,
                    grav_obs_height=grav_obs_height,
                    **new_kwargs,
                )
            )

        trial.set_user_attr("true_reg_score", true_reg_score)

        if self.optimize_on_true_regional_misfit is True:
            trial.set_user_attr("residual constraint score", residual_constraint_score)
            trial.set_user_attr("residual amplitude score", residual_amplitude_score)
            return true_reg_score  # type: ignore[return-value]

        if self.separate_metrics is True:
            return residual_constraint_score, residual_amplitude_score  # type: ignore[return-value]

        # combine the two metrics into one
        return residual_constraint_score / residual_amplitude_score


class OptimizeRegionalConstraintsPointMinimization:
    """
    Objective function to use in an Optuna optimization for finding the optimal
    hyperparameter values the Constraint Point Minimization technique for estimation the
    regional component of gravity misfit. If single dataframes are supplied to
    `training_df` and `testing_df`, for each parameter value a regional field will be
    estimated using the `training_df`, and a score calculated used the `testing_df`. If
    lists of dataframes are supplied, a score will be calculated for each item in the
    list and the mean of the scores will be the metric returned. This class is used with
    the function `optimize_regional_constraint_point_minimization`.
    """

    def __init__(
        self,
        training_df: pd.DataFrame | list[pd.DataFrame],
        testing_df: pd.DataFrame | list[pd.DataFrame],
        grid_method: str,
        # for tensioned minimum curvature gridding
        tension_factor_limits: tuple[float, float] = (0, 1),
        # for bi-harmonic spline gridding
        spline_damping_limits: tuple[float, float] | None = None,
        # for eq source gridding
        depth_limits: tuple[float, float] | None = None,
        block_size_limits: tuple[float, float] | None = None,
        damping_limits: tuple[float, float] | None = None,
        grav_obs_height_limits: tuple[float, float] | None = None,
        # other args
        optimize_on_true_regional_misfit: bool = False,
        separate_metrics: bool = True,
        progressbar: bool = False,
        **kwargs: typing.Any,
    ) -> None:
        self.training_df = training_df
        self.testing_df = testing_df
        self.grid_method = grid_method
        self.tension_factor_limits = tension_factor_limits
        self.spline_damping_limits = spline_damping_limits
        self.depth_limits = depth_limits
        self.block_size_limits = block_size_limits
        self.damping_limits = damping_limits
        self.grav_obs_height_limits = grav_obs_height_limits
        self.optimize_on_true_regional_misfit = optimize_on_true_regional_misfit
        self.separate_metrics = separate_metrics
        self.progressbar = progressbar
        self.kwargs = kwargs

    def __call__(self, trial: optuna.trial) -> float:
        """
        Parameters
        ----------
        trial : optuna.trial
            the trial to run

        Returns
        -------
        float
            the scores
        """

        new_kwargs = copy.deepcopy(self.kwargs)

        if self.grid_method == "pygmt":
            new_kwargs["tension_factor"] = trial.suggest_float(
                "tension_factor",
                self.tension_factor_limits[0],
                self.tension_factor_limits[1],
            )
        elif self.grid_method == "verde":
            if self.spline_damping_limits is None:
                msg = "if grid_method is 'verde' must provide spline_damping_limits"
                raise ValueError(msg)
            new_kwargs["spline_dampings"] = trial.suggest_float(
                "spline_dampings",
                self.spline_damping_limits[0],
                self.spline_damping_limits[1],
                log=True,
            )

        elif self.grid_method == "eq_sources":
            if self.block_size_limits is not None:
                new_kwargs["block_size"] = trial.suggest_float(
                    "block_size",
                    self.block_size_limits[0],
                    self.block_size_limits[1],
                )
            else:
                new_kwargs["block_size"] = self.kwargs.get("block_size", None)

            if self.damping_limits is not None:
                new_kwargs["damping"] = trial.suggest_float(
                    "damping",
                    self.damping_limits[0],
                    self.damping_limits[1],
                    log=True,
                )
            else:
                new_kwargs["damping"] = self.kwargs.get("damping", None)

            if self.grav_obs_height_limits is not None:
                new_kwargs["grav_obs_height"] = trial.suggest_float(
                    "grav_obs_height",
                    self.grav_obs_height_limits[0],
                    self.grav_obs_height_limits[1],
                )
            else:
                new_kwargs["grav_obs_height"] = self.kwargs.get("grav_obs_height", None)

        else:
            msg = "invalid gridding method"
            raise ValueError(msg)

        if isinstance(self.training_df, pd.DataFrame):
            if self.depth_limits is not None:
                new_kwargs["depth"] = trial.suggest_float(
                    "depth",
                    self.depth_limits[0],
                    self.depth_limits[1],
                )
            else:
                eq_depth = self.kwargs.get("depth", "default")
                if eq_depth == "default":
                    # calculate 4.5 times the mean distance between points
                    eq_depth = 4.5 * np.mean(
                        vd.median_distance(
                            (self.training_df.easting, self.training_df.northing),
                            k_nearest=1,
                        )
                    )
                new_kwargs["depth"] = eq_depth

            with utils._log_level(logging.WARN):  # pylint: disable=protected-access
                (
                    residual_constraint_score,
                    residual_amplitude_score,
                    true_reg_score,
                    _df,
                ) = cross_validation.regional_separation_score(
                    constraints_df=self.training_df,
                    testing_df=self.testing_df,
                    method="constraints",
                    grid_method=self.grid_method,
                    **new_kwargs,
                )
        else:
            # get list of folds
            folds = [
                list(df.columns[df.columns.str.startswith("fold_")])[0]  # noqa: RUF015
                for df in self.testing_df
            ]
            # progressbar for folds
            if self.progressbar is True:
                pbar = tqdm(
                    folds,
                    desc="Cross-validation folds",
                )
            elif self.progressbar is False:
                pbar = folds
            else:
                msg = "progressbar must be a boolean"  # type: ignore[unreachable]
                raise ValueError(msg)

            with utils._log_level(logging.WARN):  # pylint: disable=protected-access
                # for each fold, run CV
                results = []
                for i, _ in enumerate(pbar):
                    if self.depth_limits is not None:
                        new_kwargs["depth"] = trial.suggest_float(
                            "depth",
                            self.depth_limits[0],
                            self.depth_limits[1],
                        )
                    else:
                        eq_depth = self.kwargs.get("depth", "default")
                        if eq_depth == "default":
                            # calculate 4.5 times the mean distance between points
                            eq_depth = 4.5 * np.mean(
                                vd.median_distance(
                                    (
                                        self.training_df[i].easting,
                                        self.training_df[i].northing,
                                    ),
                                    k_nearest=1,
                                )
                            )
                        new_kwargs["depth"] = eq_depth

                    fold_results = cross_validation.regional_separation_score(
                        constraints_df=self.training_df[i],
                        testing_df=self.testing_df[i],
                        method="constraints",
                        grid_method=self.grid_method,
                        **new_kwargs,
                    )
                    results.append(fold_results)

            # get mean of scores of all folds
            residual_constraint_score = np.mean([r[0] for r in results])
            residual_amplitude_score = np.mean([r[1] for r in results])
            try:
                true_reg_score = np.mean([r[2] for r in results])
            except TypeError:
                true_reg_score = None

        logger.debug("separate_metrics: %s", self.separate_metrics)
        logger.debug(
            "optimize_on_true_regional_misfit: %s",
            self.optimize_on_true_regional_misfit,
        )

        trial.set_user_attr("true_reg_score", true_reg_score)

        if self.optimize_on_true_regional_misfit is True:
            trial.set_user_attr("residual constraint score", residual_constraint_score)
            trial.set_user_attr("residual amplitude score", residual_amplitude_score)
            return true_reg_score  # type: ignore[return-value]

        if self.separate_metrics is True:
            return residual_constraint_score, residual_amplitude_score  # type: ignore[return-value]

        # combine the two metrics into one
        return residual_constraint_score / residual_amplitude_score


def optimize_regional_filter(
    testing_df: pd.DataFrame,
    grav_ds: xr.Dataset,
    filter_width_limits: tuple[float, float],
    score_as_median: bool = False,
    remove_starting_grav_mean: bool = False,
    true_regional: xr.DataArray | None = None,
    n_trials: int = 100,
    sampler: optuna.samplers.BaseSampler | None = None,
    plot: bool = False,
    plot_grid: bool = False,
    optimize_on_true_regional_misfit: bool = False,
    separate_metrics: bool = True,
    progressbar: bool = True,
    parallel: bool = False,
    fname: str | None = None,
    seed: int = 0,
) -> tuple[optuna.study, xr.Dataset, optuna.trial.FrozenTrial]:
    """
    Run an Optuna optimization to find the optimal filter width for estimating the
    regional component of gravity misfit. For synthetic testing, if the true regional
    grid is provided, the optimization can be set to optimize on the RMSE of the
    predicted and true regional gravity, by setting
    `optimize_on_true_regional_misfit=True`. By default this will perform a
    multi-objective optimization to find the best trade-off between the lowest RMSE of
    the residual at the constraints and the highest RMSE of the residual at all
    locations.

    Parameters
    ----------
    testing_df : pandas.DataFrame
        constraint points to use for calculating the score with columns "easting",
        "northing" and "upward".
    grav_ds : xarray.Dataset
        gravity dataset with coordinates "easting", "northing", and variables "reg" and
        `gravity_anomaly`.
    filter_width_limits : tuple[float, float]
        limits to use for the filter width in meters.
    score_as_median : bool, optional
        use the root median square instead of the root mean square for the scoring
        metric, by default False
    remove_starting_grav_mean : bool, optional
        remove the mean of the starting gravity data before estimating the regional.
        Useful to mitigate effects of poorly-chosen zref value. By default False
    true_regional : xarray.DataArray | None, optional
        if the true regional gravity is known (in synthetic models), supply this as a
        grid to include a user_attr of the RMSE between this and the estimated regional
        for each trial, or set `optimize_on_true_regional_misfit=True` to have the
        optimization optimize on the RMSE, by default None
    n_trials : int, optional
        number of trials to run, by default 100
    sampler : optuna.samplers.BaseSampler | None, optional
        customize the optuna sampler, by default TPE sampler
    plot : bool, optional
        plot the resulting optimization figures, by default False
    plot_grid : bool, optional
        plot the resulting regional gravity grid, by default False
    optimize_on_true_regional_misfit : bool, optional
        if true_regional grid is provide, choose to perform optimization on the RMSE
        between the true regional and the estimated region, by default False
    separate_metrics : bool, optional
        if False, returns the scores combined with the formula
        residual_constraints_score / residual_amplitude_score, by default is True and
        returns both the residual and regional scores separately.
    progressbar : bool, optional
        add a progressbar, by default True
    parallel : bool, optional
        run the optimization in parallel, by default False
    fname : str | None, optional
        file name to save the study to, by default None
    seed : int, optional
        random seed for the samplers, by default 0

    Returns
    -------
    study : optuna.study.Study,
        the completed Optuna study
    resulting_grav_ds : xarray.Dataset
        the resulting gravity dataset of the best trial
    best_trial : optuna.trial.FrozenTrial
        the best trial
    """
    if isinstance(grav_ds, xr.Dataset) is False:
        msg = "Function `optimize_regional_filter` has been changed, data must be provided to parameter `grav_ds` as an xarray dataset initialized through function `create_data`"
        raise DeprecationWarning(msg)

    optuna.logging.set_verbosity(optuna.logging.WARN)

    # if sampler not provided, use TPE as default
    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(n_trials / 4),
            seed=seed,
        )

    results_fname = f"tmp_{random.randint(0, 999)}" if fname is None else fname

    # create study and set directions / metric names depending on optimization type
    study, storage = _create_regional_separation_study(
        optimize_on_true_regional_misfit=optimize_on_true_regional_misfit,
        separate_metrics=separate_metrics,
        sampler=sampler,
        true_regional=true_regional,
        fname=results_fname,
    )

    # run optimization
    study = run_optuna(
        study=study,
        storage=storage,
        objective=OptimizeRegionalFilter(
            filter_width_limits=filter_width_limits,
            testing_df=testing_df,
            grav_ds=grav_ds,
            true_regional=true_regional,
            score_as_median=score_as_median,
            optimize_on_true_regional_misfit=optimize_on_true_regional_misfit,
            separate_metrics=separate_metrics,
            remove_starting_grav_mean=remove_starting_grav_mean,
        ),
        n_trials=n_trials,
        progressbar=progressbar,
        parallel=parallel,
    )

    if study._is_multi_objective() is False:  # pylint: disable=protected-access
        best_trial = study.best_trial
    else:
        best_trial = min(study.best_trials, key=lambda t: t.values[0])  # noqa: PD011
        # best_trial = max(study.best_trials, key=lambda t: t.values[1])

        logger.info("Number of trials on the Pareto front: %s", len(study.best_trials))

    # warn if any best parameter values are at their limits
    warn_parameter_at_limits(best_trial)

    # log the results of the best trial
    log_optuna_results(best_trial)

    # redo the regional separation with ALL constraint points
    resulting_grav_ds = regional.regional_separation(
        method="filter",
        filter_width=best_trial.params["filter_width"],
        grav_ds=grav_ds,
        remove_starting_grav_mean=remove_starting_grav_mean,
    )

    if plot is True:
        try:
            if study._is_multi_objective() is False:  # pylint: disable=protected-access
                if optimize_on_true_regional_misfit is True:
                    plotting.plot_optimization_combined_slice(
                        study,
                        attribute_names=[
                            "residual constraint score",
                            "residual amplitude score",
                        ],
                    ).show()
                else:
                    optuna.visualization.plot_slice(study).show()
            else:
                optuna.visualization.plot_pareto_front(study).show()
                for i, j in enumerate(study.metric_names):
                    optuna.visualization.plot_slice(
                        study,
                        target=lambda t: t.values[i],  # noqa: B023 PD011 # pylint: disable=cell-var-from-loop
                        target_name=j,
                    ).show()
            if plot_grid is True:
                resulting_grav_ds.reg.plot()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("plotting failed with error: %s", e)

    return study, resulting_grav_ds, best_trial


def optimize_regional_trend(
    testing_df: pd.DataFrame,
    grav_ds: xr.Dataset,
    trend_limits: tuple[int, int],
    score_as_median: bool = False,
    remove_starting_grav_mean: bool = False,
    true_regional: xr.DataArray | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    plot: bool = False,
    plot_grid: bool = False,
    optimize_on_true_regional_misfit: bool = False,
    separate_metrics: bool = True,
    progressbar: bool = True,
    parallel: bool = False,
    fname: str | None = None,
    seed: int = 0,
) -> tuple[optuna.study, xr.Dataset, optuna.trial.FrozenTrial]:
    """
    Run an Optuna optimization to find the optimal trend order for estimating the
    regional component of gravity misfit. For synthetic testing, if the true regional
    grid is provided, the optimization can be set to optimize on the RMSE of the
    predicted and true regional gravity, by setting
    `optimize_on_true_regional_misfit=True`. By default this will perform a
    multi-objective optimization to find the best trade-off between the lowest RMSE of
    the residual at the constraints and the highest RMSE of the residual at all
    locations.

    Parameters
    ----------
    testing_df : pandas.DataFrame
        constraint points to use for calculating the score with columns "easting",
        "northing" and "upward".
    grav_ds : xarray.Dataset
        gravity dataset with coordinates "easting", "northing", and variables "reg" and
        `gravity_anomaly`.
    trend_limits : tuple[int, int]
        limits to use for the trend order in degrees.
    score_as_median : bool, optional
        use the root median square instead of the root mean square for the scoring
        metric, by default False
    remove_starting_grav_mean : bool, optional
        remove the mean of the starting gravity data before estimating the regional.
        Useful to mitigate effects of poorly-chosen zref value. By default False
    true_regional : xarray.DataArray | None, optional
        if the true regional gravity is known (in synthetic models), supply this as a
        grid to include a user_attr of the RMSE between this and the estimated regional
        for each trial, or set `optimize_on_true_regional_misfit=True` to have the
        optimization optimize on the RMSE, by default None
    sampler : optuna.samplers.BaseSampler | None, optional
        customize the optuna sampler, by default GridSampler
    plot : bool, optional
        plot the resulting optimization figures, by default False
    plot_grid : bool, optional
        plot the resulting regional gravity grid, by default False
    optimize_on_true_regional_misfit : bool, optional
        if true_regional grid is provide, choose to perform optimization on the RMSE
        between the true regional and the estimated region, by default False
    separate_metrics : bool, optional
        if False, returns the scores combined with the formula
        residual_constraints_score / residual_amplitude_score, by default is True and
        returns both the residual and regional scores separately.
    progressbar : bool, optional
        add a progressbar, by default True
    parallel : bool, optional
        run the optimization in parallel, by default False
    fname : str | None, optional
        file name to save the study to, by default None
    seed : int, optional
        random seed for the samplers, by default 0

    Returns
    -------
    study : optuna.study.Study,
        the completed Optuna study
    resulting_grav_ds : xarray.Dataset
        the resulting gravity dataset of the best trial
    best_trial : optuna.trial.FrozenTrial
        the best trial
    """
    if isinstance(grav_ds, xr.Dataset) is False:
        msg = "Function `optimize_regional_trend` has been changed, data must be provided to parameter `grav_ds` as an xarray dataset initialized through function `create_data`"
        raise DeprecationWarning(msg)

    optuna.logging.set_verbosity(optuna.logging.WARN)

    optuna.logging.set_verbosity(optuna.logging.WARN)

    # if sampler not provided, use GridSampler as default
    if sampler is None:
        sampler = optuna.samplers.GridSampler(
            search_space={"trend": list(range(trend_limits[0], trend_limits[1] + 1))},
            seed=seed,
        )

    results_fname = f"tmp_{random.randint(0, 999)}" if fname is None else fname

    # create study and set directions / metric names depending on optimization type
    study, storage = _create_regional_separation_study(
        optimize_on_true_regional_misfit=optimize_on_true_regional_misfit,
        separate_metrics=separate_metrics,
        sampler=sampler,
        true_regional=true_regional,
        fname=results_fname,
    )

    # run optimization
    study = run_optuna(
        study=study,
        storage=storage,
        objective=OptimizeRegionalTrend(  # type: ignore[arg-type]
            trend_limits=trend_limits,
            testing_df=testing_df,
            grav_ds=grav_ds,
            true_regional=true_regional,
            score_as_median=score_as_median,
            optimize_on_true_regional_misfit=optimize_on_true_regional_misfit,
            separate_metrics=separate_metrics,
            remove_starting_grav_mean=remove_starting_grav_mean,
        ),
        n_trials=len(list(range(trend_limits[0], trend_limits[1] + 1))),
        maximize_cpus=True,
        parallel=parallel,
        progressbar=progressbar,
    )

    if study._is_multi_objective() is False:  # pylint: disable=protected-access
        best_trial = study.best_trial
    else:
        best_trial = min(study.best_trials, key=lambda t: t.values[0])  # noqa: PD011
        # best_trial = max(study.best_trials, key=lambda t: t.values[1])

        logger.info("Number of trials on the Pareto front: %s", len(study.best_trials))

    # warn if any best parameter values are at their limits
    warn_parameter_at_limits(best_trial)

    # log the results of the best trial
    log_optuna_results(best_trial)

    # redo the regional separation with ALL constraint points
    resulting_grav_ds = regional.regional_separation(
        method="trend",
        trend=best_trial.params["trend"],
        grav_ds=grav_ds,
        remove_starting_grav_mean=remove_starting_grav_mean,
    )

    if plot is True:
        try:
            if study._is_multi_objective() is False:  # pylint: disable=protected-access
                if optimize_on_true_regional_misfit is True:
                    plotting.plot_optimization_combined_slice(
                        study,
                        attribute_names=[
                            "residual constraint score",
                            "residual amplitude score",
                        ],
                    ).show()
                else:
                    optuna.visualization.plot_slice(study).show()
            else:
                optuna.visualization.plot_pareto_front(study).show()
                for i, j in enumerate(study.metric_names):
                    optuna.visualization.plot_slice(
                        study,
                        target=lambda t: t.values[i],  # noqa: B023 PD011 # pylint: disable=cell-var-from-loop
                        target_name=j,
                    ).show()
            if plot_grid is True:
                resulting_grav_ds.reg.plot()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("plotting failed with error: %s", e)

    return study, resulting_grav_ds, best_trial


def optimize_regional_eq_sources(
    testing_df: pd.DataFrame,
    grav_ds: xr.Dataset,
    score_as_median: bool = False,
    true_regional: xr.DataArray | None = None,
    n_trials: int = 100,
    depth_limits: tuple[float, float] | None = None,
    block_size_limits: tuple[float, float] | None = None,
    damping_limits: tuple[float, float] | None = None,
    grav_obs_height_limits: tuple[float, float] | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    plot: bool = False,
    plot_grid: bool = False,
    optimize_on_true_regional_misfit: bool = False,
    separate_metrics: bool = True,
    progressbar: bool = True,
    parallel: bool = False,
    fname: str | None = None,
    seed: int = 0,
    **kwargs: typing.Any,
) -> tuple[optuna.study, xr.Dataset, optuna.trial.FrozenTrial]:
    """
    Run an Optuna optimization to find the optimal equivalent source parameters for
    estimating the regional component of gravity misfit. For synthetic testing, if the
    true regional grid is provided, the optimization can be set to optimize on the
    RMSE of the predicted and true regional gravity, by setting
    `optimize_on_true_regional_misfit=True`. By default this will perform a
    multi-objective optimization to find the best trade-off between the lowest RMSE of
    the residual at the constraints and the highest RMSE of the residual at all
    locations.

    Parameters
    ----------
    testing_df : pandas.DataFrame
        constraint points to use for calculating the score with columns "easting",
        "northing" and "upward".
    grav_ds : xarray.Dataset
        gravity dataset with coordinates "easting", "northing", and variables "reg" and
        `gravity_anomaly`.
    score_as_median : bool, optional
        use the root median square instead of the root mean square for the scoring
        metric, by default False
    true_regional : xarray.DataArray | None, optional
        if the true regional gravity is known (in synthetic models), supply this as a
        grid to include a user_attr of the RMSE between this and the estimated regional
        for each trial, or set `optimize_on_true_regional_misfit=True` to have the
        optimization optimize on the RMSE, by default None
    n_trials : int, optional
        number of trials to run, by default 100
    depth_limits : tuple[float, float] | None, optional
        limits to use for source depths, positive down in meters, by default None
    block_size_limits : tuple[float, float] | None, optional
        limits to use for block size in meters, by default None
    damping_limits : tuple[float, float] | None, optional
        limits to use for the damping parameter, by default None
    grav_obs_height_limits : tuple[float, float] | None, optional
        limits to use for the gravity observation height in meters, by default None
    sampler : optuna.samplers.BaseSampler | None, optional
        customize the optuna sampler, by default TPE sampler
    plot : bool, optional
        plot the resulting optimization figures, by default False
    plot_grid : bool, optional
        plot the resulting regional gravity grid, by default False
    optimize_on_true_regional_misfit : bool, optional
        if true_regional grid is provide, choose to perform optimization on the RMSE
        between the true regional and the estimated region, by default False
    separate_metrics : bool, optional
        if False, returns the scores combined with the formula
        residual_constraints_score / residual_amplitude_score, by default is True and
        returns both the residual and regional scores separately.
    progressbar : bool, optional
        add a progressbar, by default True
    parallel : bool, optional
        run the optimization in parallel, by default False
    fname : str | None, optional
        file name to save the study to, by default None
    seed : int, optional
        random seed for the samplers, by default 0
    kwargs : typing.Any
        additional keyword arguments to pass to the :meth:`DatasetAccessorInvert4Geom.regional_separation`

    Returns
    -------
    study : optuna.study.Study
        the completed Optuna study
    resulting_grav_ds : xarray.Dataset
        the resulting gravity dataset of the best trial
    best_trial : optuna.trial.FrozenTrial
        the best trial
    """
    if isinstance(grav_ds, xr.Dataset) is False:
        msg = "Function `optimize_regional_eq_sources` has been changed, data must be provided to parameter `grav_ds` as an xarray dataset initialized through function `create_data`"
        raise DeprecationWarning(msg)

    optuna.logging.set_verbosity(optuna.logging.WARN)

    optuna.logging.set_verbosity(optuna.logging.WARN)

    kwargs = copy.deepcopy(kwargs)

    # if sampler not provided, use TPE as default
    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(n_trials / 4),
            seed=seed,
        )

    results_fname = f"tmp_{random.randint(0, 999)}" if fname is None else fname

    # create study and set directions / metric names depending on optimization type
    study, storage = _create_regional_separation_study(
        optimize_on_true_regional_misfit=optimize_on_true_regional_misfit,
        separate_metrics=separate_metrics,
        sampler=sampler,
        true_regional=true_regional,
        fname=results_fname,
    )

    # run optimization
    study = run_optuna(
        study=study,
        storage=storage,
        objective=OptimizeRegionalEqSources(
            depth_limits=depth_limits,
            block_size_limits=block_size_limits,
            damping_limits=damping_limits,
            grav_obs_height_limits=grav_obs_height_limits,
            testing_df=testing_df,
            grav_ds=grav_ds,
            true_regional=true_regional,
            score_as_median=score_as_median,
            optimize_on_true_regional_misfit=optimize_on_true_regional_misfit,
            separate_metrics=separate_metrics,
            **kwargs,
        ),
        n_trials=n_trials,
        maximize_cpus=True,
        parallel=parallel,
        progressbar=progressbar,
    )

    if study._is_multi_objective() is False:  # pylint: disable=protected-access
        best_trial = study.best_trial
    else:
        best_trial = min(study.best_trials, key=lambda t: t.values[0])  # noqa: PD011
        # best_trial = max(study.best_trials, key=lambda t: t.values[1])

        logger.info("Number of trials on the Pareto front: %s", len(study.best_trials))

    # warn if any best parameter values are at their limits
    warn_parameter_at_limits(best_trial)

    # log the results of the best trial
    log_optuna_results(best_trial)

    # get optimal hyperparameter values
    depth = best_trial.params.get("depth", kwargs.pop("depth", "default"))
    if depth == "default":
        # calculate 4.5 times the mean distance between points
        depth = 4.5 * np.mean(
            vd.median_distance((grav_ds.easting, grav_ds.northing), k_nearest=1)
        )
    damping = best_trial.params.get("damping", kwargs.pop("damping", None))
    block_size = best_trial.params.get("block_size", kwargs.pop("block_size", None))
    grav_obs_height = best_trial.params.get(
        "grav_obs_height",
        kwargs.pop("grav_obs_height", None),
    )
    # redo the regional separation with best parameters
    resulting_grav_ds = regional.regional_separation(
        method="eq_sources",
        depth=depth,
        damping=damping,
        block_size=block_size,
        grav_obs_height=grav_obs_height,
        grav_ds=grav_ds,
        **kwargs,
    )
    if plot is True:
        try:
            if study._is_multi_objective() is False:  # pylint: disable=protected-access
                if optimize_on_true_regional_misfit is True:
                    for p in best_trial.params:
                        plotting.plot_optimization_combined_slice(
                            study,
                            attribute_names=[
                                "residual constraint score",
                                "residual amplitude score",
                            ],
                            parameter_name=[p],  # type: ignore[arg-type]
                        ).show()
                else:
                    optuna.visualization.plot_slice(study).show()
            else:
                optuna.visualization.plot_pareto_front(study).show()
                for i, j in enumerate(study.metric_names):
                    optuna.visualization.plot_slice(
                        study,
                        target=lambda t: t.values[i],  # noqa: B023 PD011 # pylint: disable=cell-var-from-loop
                        target_name=j,
                    ).show()
            if plot_grid is True:
                resulting_grav_ds.reg.plot()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("plotting failed with error: %s", e)

    return study, resulting_grav_ds, best_trial


def optimize_regional_constraint_point_minimization(
    testing_training_df: pd.DataFrame,
    grid_method: str,
    grav_ds: xr.Dataset,
    n_trials: int,
    tension_factor_limits: tuple[float, float] = (0, 1),
    spline_damping_limits: tuple[float, float] | None = None,
    depth_limits: tuple[float, float] | None = None,
    block_size_limits: tuple[float, float] | None = None,
    damping_limits: tuple[float, float] | None = None,
    grav_obs_height_limits: tuple[float, float] | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    plot: bool = False,
    plot_grid: bool = False,
    fold_progressbar: bool = False,
    optimize_on_true_regional_misfit: bool = False,
    separate_metrics: bool = True,
    score_as_median: bool = False,
    true_regional: xr.DataArray | None = None,
    progressbar: bool = True,
    parallel: bool = False,
    fname: str | None = None,
    seed: int = 0,
    **kwargs: typing.Any,
) -> tuple[optuna.study, xr.Dataset, optuna.trial.FrozenTrial]:
    """
    Run an Optuna optimization to find the optimal hyperparameters for the Constraint
    Point Minimization technique for estimating the regional component of gravity
    misfit. Since constraints are used both for determining the regional field, and for
    the scoring of the performance, we must split the constraints into testing and
    training sets. This function can perform both single and K-Folds cross validations,
    determined by the number of "fold_x" columns in testing_training_df. If using more
    than one fold, the score for each parameter set is the mean of the scores of each
    fold. The total number of regional separation this will perform is n_trials*K-folds.
    This function then uses the optimal parameter values to redo the regional
    estimation using all the constraints points, not just the training points, and
    returns the results.
    By default this will perform a multi-objective optimization to
    find the best trade-off between the lowest RMSE of the residual misfit at the
    constraints and the highest RMS amplitude of the residual at all locations.
    Choose which Constraint Point Minimization gridding method with the `grid_method`
    parameter, and supplied the associated method parameter limits via parameters
    <parameter>_limits. For grid method "eq_sources" which has multiple parameters, if
    limits aren't provided for one of the parameters, supply a constant value for the
    parameter in the keyword arguments, which are past direction to
    :meth:`DatasetAccessorInvert4Geom.regional_separation`. For synthetic testing, if the true regional grid is
    provided, the optimization can be set to optimize on the RMSE of the predicted and
    true regional gravity, by setting `optimize_on_true_regional_misfit=True`.

    Parameters
    ----------
    testing_training_df : pandas.DataFrame
        constraints dataframe with columns "easting", "northing", "upward", and a column
        for each fold in the format "fold_0", "fold_1", etc. This can be created with
        function `cross_validation.split_test_train()`. Each fold column should have
        strings of "test" or "train" to indicate which rows are testing or training
        points. If more than one fold is provided, this function will perform a K-Folds
        cross validation and the score for each set of parameters will be the mean of
        the K-scores.
    grid_method : str
        constraint point minimization method to use, choose between "verde" for
        bi-harmonic spline gridding, "pygmt" for tensioned minimum curvature gridding,
        or "eq_sources" for equivalent sources gridding.
    grav_ds : xarray.Dataset
        gravity dataset with coordinates "easting", "northing", and variables "reg" and
        `gravity_anomaly`.
    n_trials : int
        number of trials to run
    tension_factor_limits : tuple[float, float], optional
        limits to use for the PyGMT tension factor gridding, by default (0, 1)
    spline_damping_limits : tuple[float, float] | None, optional
        limits to use for the Verde bi-harmonic spline damping, by default None
    depth_limits : tuple[float, float] | None, optional
        limits to use for the equivalent sources' depths, by default None
    block_size_limits : tuple[float, float] | None, optional
        limits to use for the block size for fitting equivalent sources, by default None
    damping_limits : tuple[float, float] | None, optional
        limits to use for the damping value for fitting equivalent sources, by default
        None
    grav_obs_height_limits : tuple[float, float] | None, optional
        limits to use for the gravity observation height for fitting equivalent sources,
        by default None
    sampler : optuna.samplers.BaseSampler | None, optional
        customize the optuna sampler, by default TPE sampler
    plot : bool, optional
        plot the resulting optimization figures, by default False
    plot_grid : bool, optional
        plot the resulting regional gravity grid, by default False
    fold_progressbar : bool, optional
        turn on or off a progress bar for the optimization of each fold if performing
        a K-Folds cross-validation within the optimization, by default False
    optimize_on_true_regional_misfit : bool, optional
        if true_regional grid is provide, choose to perform optimization on the RMSE
        between the true regional and the estimated region, by default False
    separate_metrics : bool, optional
        if False, returns the scores combined with the formula
        residual_constraints_score / residual_amplitude_score, by default is True and
        returns both the residual and regional scores separately.
    score_as_median : bool, optional
        use the root median square instead of the root mean square for the scoring
        metric, by default False
    true_regional : xarray.DataArray | None, optional
        if the true regional gravity is known (in synthetic models), supply this as a
        grid to include a user_attr of the RMSE between this and the estimated regional
        for each trial, or set `optimize_on_true_regional_misfit=True` to have the
        optimization optimize on the RMSE, by default None
    progressbar : bool, optional
        add a progressbar, by default True
    parallel : bool, optional
        run the optimization in parallel, by default False
    fname : str | None, optional
        file name to save the study to, by default None
    seed : int, optional
        random seed for the samplers, by default 0
    kwargs : typing.Any
        additional keyword arguments to pass to the :meth:`DatasetAccessorInvert4Geom.regional_separation`

    Returns
    -------
    study : optuna.study.Study,
        the completed Optuna study
    resulting_grav_ds : xarray.Dataset
        the resulting gravity dataset of the best trial
    best_trial : optuna.trial.FrozenTrial
        the best trial
    """

    if isinstance(grav_ds, xr.Dataset) is False:
        msg = "Function `optimize_regional_constraint_point_minimization` has been changed, data must be provided to parameter `grav_ds` as an xarray dataset initialized through function `create_data`"
        raise DeprecationWarning(msg)
    optuna.logging.set_verbosity(optuna.logging.WARN)

    kwargs = copy.deepcopy(kwargs)

    # if sampler not provided, use TPE as default
    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(n_trials / 4),
            seed=seed,
        )

    results_fname = f"tmp_{random.randint(0, 999)}" if fname is None else fname

    # create study and set directions / metric names depending on optimization type
    study, storage = _create_regional_separation_study(
        optimize_on_true_regional_misfit=optimize_on_true_regional_misfit,
        separate_metrics=separate_metrics,
        sampler=sampler,
        true_regional=true_regional,
        parallel=parallel,
        fname=results_fname,
    )

    # get folds from constraints_df
    test_dfs, train_dfs = cross_validation.kfold_df_to_lists(testing_training_df)
    assert len(test_dfs) == len(train_dfs)

    logger.info("Number of folds: %s", len(test_dfs))

    # combine testing and training to get a full constraints dataframe
    constraints_df = (
        pd.concat(test_dfs + train_dfs)
        .drop_duplicates(subset=["easting", "northing", "upward"])
        .sort_index()
    )

    if len(test_dfs) == 1:
        test_dfs = test_dfs[0]
        train_dfs = train_dfs[0]

    logger.debug("separate_metrics: %s", separate_metrics)
    logger.debug(
        "optimize_on_true_regional_misfit: %s", optimize_on_true_regional_misfit
    )

    # enqueue limits as trials
    if grid_method == "pygmt":
        study.enqueue_trial(
            {"tension_factor": tension_factor_limits[0]}, skip_if_exists=True
        )
        study.enqueue_trial(
            {"tension_factor": tension_factor_limits[1]}, skip_if_exists=True
        )
    elif grid_method == "verde":
        study.enqueue_trial(
            {"spline_dampings": spline_damping_limits[0]},  # type: ignore[index]
            skip_if_exists=True,
        )
        study.enqueue_trial(
            {"spline_dampings": spline_damping_limits[1]},  # type: ignore[index]
            skip_if_exists=True,
        )
    elif grid_method == "eq_sources":
        if depth_limits is not None:
            study.enqueue_trial({"depth": depth_limits[0]}, skip_if_exists=True)
            study.enqueue_trial({"depth": depth_limits[1]}, skip_if_exists=True)
        if block_size_limits is not None:
            study.enqueue_trial(
                {"block_size": block_size_limits[0]}, skip_if_exists=True
            )
            study.enqueue_trial(
                {"block_size": block_size_limits[1]}, skip_if_exists=True
            )
        if damping_limits is not None:
            study.enqueue_trial({"damping": damping_limits[0]}, skip_if_exists=True)
            study.enqueue_trial({"damping": damping_limits[1]}, skip_if_exists=True)
        if grav_obs_height_limits is not None:
            study.enqueue_trial(
                {"grav_obs_height": grav_obs_height_limits[0]}, skip_if_exists=True
            )
            study.enqueue_trial(
                {"grav_obs_height": grav_obs_height_limits[1]}, skip_if_exists=True
            )

    # run optimization
    study = run_optuna(
        study=study,
        storage=storage,
        objective=OptimizeRegionalConstraintsPointMinimization(
            training_df=train_dfs,
            testing_df=test_dfs,
            # kwargs for regional.regional_constraints:
            grav_ds=grav_ds,
            grid_method=grid_method,
            tension_factor_limits=tension_factor_limits,
            spline_damping_limits=spline_damping_limits,
            depth_limits=depth_limits,
            block_size_limits=block_size_limits,
            damping_limits=damping_limits,
            grav_obs_height_limits=grav_obs_height_limits,
            # optimization kwargs
            true_regional=true_regional,
            score_as_median=score_as_median,
            optimize_on_true_regional_misfit=optimize_on_true_regional_misfit,
            separate_metrics=separate_metrics,
            progressbar=fold_progressbar,
            **kwargs,
        ),
        n_trials=n_trials,
        maximize_cpus=True,
        parallel=parallel,
        progressbar=progressbar,
    )

    if study._is_multi_objective() is False:  # pylint: disable=protected-access
        best_trial = study.best_trial
    else:
        best_trial = min(study.best_trials, key=lambda t: t.values[0])  # noqa: PD011
        # best_trial = max(study.best_trials, key=lambda t: t.values[1])

        logger.info("Number of trials on the Pareto front: %s", len(study.best_trials))

    # warn if any best parameter values are at their limits
    warn_parameter_at_limits(best_trial)

    # log the results of the best trial
    log_optuna_results(best_trial)

    logger.info(
        "re-running regional separation with best parameters and all constraints"
    )

    # get optimal hyperparameter values
    # if not included in optimization, get from kwargs
    tension_factor = best_trial.params.get("tension_factor", None)
    spline_dampings = best_trial.params.get("spline_dampings", None)
    depth = best_trial.params.get("depth", kwargs.pop("depth", "default"))
    if depth == "default":
        # calculate 4.5 times the mean distance between points
        depth = 4.5 * np.mean(
            vd.median_distance(
                (constraints_df.easting, constraints_df.northing), k_nearest=1
            )
        )
    damping = best_trial.params.get("damping", kwargs.pop("damping", None))
    block_size = best_trial.params.get("block_size", kwargs.pop("block_size", None))
    grav_obs_height = best_trial.params.get(
        "grav_obs_height", kwargs.pop("grav_obs_height", None)
    )

    # redo the regional separation with ALL constraint points
    resulting_grav_ds = regional.regional_separation(
        method="constraints",
        grav_ds=grav_ds,
        constraints_df=constraints_df,
        grid_method=grid_method,
        tension_factor=tension_factor,
        spline_dampings=spline_dampings,
        depth=depth,
        damping=damping,
        block_size=block_size,
        grav_obs_height=grav_obs_height,
        **kwargs,
    )

    # save study
    if results_fname is not None:
        # remove if exists
        pathlib.Path(f"{results_fname}_study.pickle").unlink(missing_ok=True)

        # save study to pickle
        with pathlib.Path(f"{results_fname}_study.pickle").open("wb") as f:
            pickle.dump(study, f)

    if plot is True:
        try:
            if study._is_multi_objective() is False:  # pylint: disable=protected-access
                if optimize_on_true_regional_misfit is True:
                    for p in best_trial.params:
                        plotting.plot_optimization_combined_slice(
                            study,
                            attribute_names=[
                                "residual constraint score",
                                "residual amplitude score",
                            ],
                            parameter_name=[p],  # type: ignore[arg-type]
                        ).show()
                else:
                    optuna.visualization.plot_slice(study).show()
            else:
                optuna.visualization.plot_pareto_front(study).show()
                for i, j in enumerate(study.metric_names):
                    optuna.visualization.plot_slice(
                        study,
                        target=lambda t: t.values[i],  # noqa: B023 PD011 # pylint: disable=cell-var-from-loop
                        target_name=j,
                    ).show()
            if plot_grid is True:
                resulting_grav_ds.reg.plot()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("plotting failed with error: %s", e)

    return study, resulting_grav_ds, best_trial


def optimal_buffer(
    target: float,
    buffer_perc_limits: tuple[float, float] = (1, 50),
    n_trials: int = 25,
    sampler: optuna.samplers.BaseSampler | None = None,
    grid_search: bool = False,
    fname: str | None = None,
    progressbar: bool = True,
    parallel: bool = False,
    plot: bool = True,
    seed: int = 0,
    **kwargs: typing.Any,
) -> tuple[optuna.study, tuple[float, float, int, xr.Dataset]]:
    """
    Run an optimization to find best buffer zone width.
    """

    optuna.logging.set_verbosity(optuna.logging.WARN)
    # pylint: enable=duplicate-code
    # if sampler not provided, use GPSampler as default unless grid_search is True
    if sampler is None:
        if grid_search is True:
            if n_trials < 4:
                msg = (
                    "if grid_search is True, n_trials must be at least 4, "
                    "resetting n_trials to 4 now."
                )
                logger.warning(msg)
                n_trials = 4
            space = np.linspace(buffer_perc_limits[0], buffer_perc_limits[1], n_trials)
            # omit first and last since they will be enqueued separately
            space = space[1:-1]
            sampler = optuna.samplers.GridSampler(
                search_space={"buffer_perc": space},
                seed=seed,
            )
        else:
            with warnings.catch_warnings():
                sampler = optuna.samplers.GPSampler(
                    n_startup_trials=int(n_trials / 4),
                    seed=seed,
                    deterministic_objective=True,
                )
    # pylint: disable=duplicate-code
    # set file name for saving results with random number between 0 and 999
    if fname is None:
        fname = f"tmp_{random.randint(0, 999)}"

    if parallel:
        pathlib.Path(f"{fname}.log").unlink(missing_ok=True)
        pathlib.Path(f"{fname}.lock").unlink(missing_ok=True)
        pathlib.Path(f"{fname}.log.lock").unlink(missing_ok=True)
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(f"{fname}.log"),
        )
    else:
        storage = None

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        load_if_exists=False,
        study_name=fname,
        storage=storage,
        pruner=DuplicateIterationPruner,
    )

    # explicitly add the limits as trials
    study.enqueue_trial({"damping": buffer_perc_limits[0]}, skip_if_exists=True)
    study.enqueue_trial({"damping": buffer_perc_limits[1]}, skip_if_exists=True)

    # run optimization
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="logei_candidates_func is experimental"
        )
        with utils.DuplicateFilter(logger):  # type: ignore[no-untyped-call]
            study = run_optuna(
                study=study,
                storage=storage,
                objective=OptimalBuffer(
                    buffer_perc_limits=buffer_perc_limits,
                    fname=fname,
                    target=target,
                    **kwargs,
                ),
                n_trials=n_trials,
                # callbacks=[_warn_limits_better_than_trial_1_param],
                maximize_cpus=True,
                parallel=parallel,
                progressbar=progressbar,
            )

    best_trial = study.best_trial

    # warn if any best parameter values are at their limits
    warn_parameter_at_limits(best_trial)

    # log the results of the best trial
    log_optuna_results(best_trial)

    # re-run decay calculation with optimal buffer
    results = utils.gravity_decay_buffer(
        buffer_perc=best_trial.params["buffer_perc"],
        plot=plot,
        **kwargs,
    )

    if plot:
        try:
            plot1 = optuna.visualization.plot_optimization_history(study)
            plot2 = optuna.visualization.plot_slice(study)

            plot1.show()
            plot2.show()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("plotting failed with error: %s", e)

    return study, results


class OptimalBuffer:
    """
    Objective function to use in an Optuna optimization for finding the buffer zone
    width as a percentage of region width which limits the gravity decay (edge effects)
    to a specified amount within a region of interest. Used within function
    func:`optimal_buffer`.
    """

    def __init__(
        self,
        buffer_perc_limits: tuple[float, float],
        target: float,
        fname: str,
        **kwargs: typing.Any,
    ) -> None:
        self.fname = fname
        self.buffer_perc_limits = buffer_perc_limits
        self.target = target
        self.kwargs = kwargs

    def __call__(self, trial: optuna.trial) -> float:
        """
        Parameters
        ----------
        trial : optuna.trial
            the trial to run

        Returns
        -------
        float
            the score of the eq_sources fit
        """
        buffer_perc = trial.suggest_float(
            "buffer_perc",
            self.buffer_perc_limits[0],
            self.buffer_perc_limits[1],
        )

        new_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key
            not in [
                "buffer_perc",
                "progressbar",
                "results_fname",
                "plot",
            ]
        }

        trial.set_user_attr("fname", f"{self.fname}_trial_{trial.number}")

        score = utils.gravity_decay_buffer(
            buffer_perc=buffer_perc,
            progressbar=False,
            plot=False,
            **new_kwargs,
        )[0]

        return np.abs((self.target) - score)  # type: ignore[no-any-return]


class DuplicateIterationPruner(optuna.pruners.BasePruner):  # type: ignore[misc]
    """
    DuplicatePruner

    Pruner to detect duplicate trials based on the parameters.

    This pruner is used to identify and prune trials that have the same set of
    parameters as a previously completed trial.
    """

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

        for completed_trial in completed_trials:
            if completed_trial.params == trial.params:
                return True

        return False
