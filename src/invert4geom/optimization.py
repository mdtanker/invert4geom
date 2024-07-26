from __future__ import annotations  # pylint: disable=too-many-lines

import itertools
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
import numpy as np
import optuna
import pandas as pd
import xarray as xr
from nptyping import NDArray
from tqdm.autonotebook import tqdm

from invert4geom import cross_validation, inversion, log, plotting, regional, utils

try:
    import joblib
except ImportError:
    joblib = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    from tqdm_joblib import tqdm_joblib
except ImportError:
    tqdm_joblib = None


def log_filter(record: typing.Any) -> bool:  # noqa: ARG001 # pylint: disable=unused-argument
    """Used to filter logging."""
    return False


def logging_callback(
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
            log.info(
                "Trial %s finished with best value: %s and parameters: %s.",
                frozen_trial.number,
                frozen_trial.value,
                frozen_trial.params,
            )
    else:
        if frozen_trial.number in [n.number for n in study.best_trials]:
            msg = (
                "Trial %s is on the Pareto front with value 1: %s, value 2: %s and "
                "parameters: %s."
            )
            log.info(
                msg,
                frozen_trial.number,
                frozen_trial.values[0],
                frozen_trial.values[1],
                frozen_trial.params,
            )


def warn_limits_better_than_trial_1_param(
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
    # if study direction is minimize
    if study.direction == optuna.study.StudyDirection.MINIMIZE:
        # if current trial is worse than either limit, log a warning
        if trial.values[0] > max(lower_limit_score, upper_limit_score):
            log.warning(
                msg,
                trial.number,
                trial.params,
                trial.values[0],
                lower_limit_score,
                upper_limit_score,
            )
        else:
            pass

    # if study direction is maximize
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        # if current trial is worse than either limit, log a warning
        if trial.values[0] < min(lower_limit_score, upper_limit_score):
            log.warning(
                msg,
                trial.number,
                trial.params,
                trial.values[0],
                lower_limit_score,
                upper_limit_score,
            )
        else:
            pass


def warn_limits_better_than_trial_multi_params(
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
        if trial.values[0] > max(scores):
            log.warning(
                msg,
                trial.number,
                trial.params,
                trial.values[0],
            )
        else:
            pass

    # if study direction is maximize
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        # if current trial is worse than either limit, log a warning
        if trial.values[0] < min(scores):
            log.warning(
                msg,
                trial.number,
                trial.params,
                trial.values[0],
            )
        else:
            pass


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
        if psutil is None:
            msg = "Missing optional dependency 'psutil' required for optimization."
            raise ImportError(msg)
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

    # # jython
    # try:
    #     runtime = Runtime.getRuntime()
    #     res = runtime.availableProcessors()
    #     if res > 0:
    #         return res
    # except ImportError:
    #     pass

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
        pseudo_devices = os.listdir("/devices/pseudo/")
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


def optuna_parallel(
    study_name: str,
    study_storage: optuna.storages.BaseStorage,
    objective: typing.Callable[..., float],
    n_trials: int = 100,
    maximize_cpus: bool = True,
    parallel: bool = True,
) -> tuple[typing.Any, pd.DataFrame]:
    """
    Run optuna optimization in parallel. Pre-define the study, storage, and objective
    function and input them here.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # load study metadata from storage
    study = optuna.load_study(storage=study_storage, study_name=study_name)

    # set up parallel processing and run optimization
    if parallel is True:

        def optimize_study(
            study_name: str,
            storage: optuna.storages.BaseStorage,
            objective: typing.Callable[..., float],
            n_trials: int,
        ) -> None:
            study = optuna.load_study(study_name=study_name, storage=storage)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(
                objective,
                n_trials=n_trials,
            )

        if maximize_cpus is True:
            optuna_max_cores(
                n_trials, optimize_study, study_name, study_storage, objective
            )
        elif maximize_cpus is False:
            optuna_1job_per_core(
                n_trials, optimize_study, study_name, study_storage, objective
            )

    # run in normal, non-parallel mode
    elif parallel is False:
        study = optuna.load_study(
            study_name=study_name,
            storage=study_storage,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Progress bar is experimental")
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=True,
            )

    # reload the study
    study = optuna.load_study(
        study_name=study_name,
        storage=study_storage,
    )

    # get dataframe from study and sort by objective value
    study_df = study.trials_dataframe()

    return study, study_df


def optuna_max_cores(
    n_trials: int,
    optimize_study: typing.Callable[..., None],
    study_name: str,
    study_storage: optuna.storages.BaseStorage,
    objective: typing.Callable[..., float],
) -> None:
    """
    Set up optuna optimization in parallel splitting up the number of trials over all
    available cores.
    """
    if joblib is None:
        msg = "Missing optional dependency 'joblib' required for optimization."
        raise ImportError(msg)

    if tqdm_joblib is None:
        msg = "Missing optional dependency 'tqdm_joblib' required for optimization."
        raise ImportError(msg)

    # get available cores (UNIX and Windows)
    # num_cores = len(psutil.Process().cpu_affinity())
    num_cores = available_cpu_count()

    # set trials per job
    trials_per_job = math.ceil(n_trials / num_cores)

    # set number of jobs
    n_jobs = num_cores if n_trials >= num_cores else n_trials

    with tqdm_joblib(desc="Optimizing", total=n_trials) as _:
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(optimize_study)(
                study_name,
                study_storage,
                objective,
                n_trials=trials_per_job,
            )
            for i in range(n_trials)
        )


def optuna_1job_per_core(
    n_trials: int,
    optimize_study: typing.Callable[..., None],
    study_name: str,
    study_storage: optuna.storages.BaseStorage,
    objective: typing.Callable[..., float],
) -> None:
    """
    Set up optuna optimization in parallel giving each available core 1 trial.
    """
    if joblib is None:
        msg = "Missing optional dependency 'joblib' required for optimization."
        raise ImportError(msg)

    if tqdm_joblib is None:
        msg = "Missing optional dependency 'tqdm_joblib' required for optimization."
        raise ImportError(msg)

    trials_per_job = 1
    with tqdm_joblib(desc="Optimizing", total=n_trials) as _:
        joblib.Parallel(n_jobs=int(n_trials / trials_per_job))(
            joblib.delayed(optimize_study)(
                study_name,
                study_storage,
                objective,
                n_trials=trials_per_job,
            )
            for i in range(int(n_trials / trials_per_job))
        )


class OptimalInversionDamping:
    """
    Objective function to use in an Optuna optimization for finding the optimal damping
    regularization value for a gravity inversion. Used within function
    `optimize_inversion_damping()`.
    """

    def __init__(
        self,
        damping_limits: tuple[float, float],
        fname: str,
        plot_grids: bool = False,
        **kwargs: typing.Any,
    ) -> None:
        self.fname = fname
        self.damping_limits = damping_limits
        self.kwargs = kwargs
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
        damping = trial.suggest_float(
            "damping",
            self.damping_limits[0],
            self.damping_limits[1],
            log=True,
        )

        new_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key
            not in [
                "solver_damping",
                "progressbar",
                "results_fname",
            ]
        }

        trial.set_user_attr("fname", f"{self.fname}_trial_{trial.number}")

        score, results = cross_validation.grav_cv_score(
            solver_damping=damping,
            progressbar=False,
            results_fname=trial.user_attrs.get("fname"),
            plot=self.plot_grids,
            **new_kwargs,
        )

        trial.set_user_attr("results", results)

        return score


def optimize_inversion_damping(
    training_df: pd.DataFrame,
    testing_df: pd.DataFrame,
    n_trials: int,
    damping_limits: tuple[float, float],
    score_as_median: bool = False,
    sampler: optuna.samplers.BaseSampler | None = None,
    grid_search: bool = False,
    fname: str | None = None,
    plot_cv: bool = True,
    plot_grids: bool = False,
    logx: bool = True,
    logy: bool = True,
    **kwargs: typing.Any,
) -> tuple[
    optuna.study, tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float]
]:
    """
    Use Optuna to find the optimal damping regularization parameter for a gravity
    inversion. The optimization aims to minimize the cross-validation score,
    represented by the root mean (or median) squared error (RMSE), between the testing
    gravity data, and the predict gravity data after and inversion. Follows methods of
    :footcite:t:`uiedafast2017`.

    Provide upper and low damping values, number of trials to run, and specify to let
    Optuna choose the best damping value for each trial or to use a grid search. The
    results are saved to a pickle file with the best inversion results and the study.

    Parameters
    ----------
    training_df : pd.DataFrame
        rows of the gravity data frame which are just the training data
    testing_df : pd.DataFrame
        rows of the gravity data frame which are just the testing data
    n_trials : int
        number of damping values to try
    damping_limits : tuple[float, float]
        upper and lower limits
    score_as_median : bool, optional
        if True, changes the scoring from the root mean square to the root median
        square, by default False
    sampler : optuna.samplers.BaseSampler | None, optional
        customize the optuna sampler, by default either BoTorch sampler or GridSampler
        depending on if grid_search is True or False
    grid_search : bool, optional
        search the entire parameter space between damping_limits in n_trial steps, by
        default False
    fname : str | None, optional
        file name to save both study and inversion results to as pickle files, by
        default in format `tmp_{random.randint(0,999)}`.
    plot_cv : bool, optional
        plot the cross-validation results, by default True
    plot_grids : bool, optional
        for each damping value, plot comparison of predicted and testing gravity data,
        by default False
    logx : bool, optional
        make x axis of CV result plot on log scale, by default True
    logy : bool, optional
        make y axis of CV result plot on log scale, by default True

    Returns
    -------
    tuple[optuna.study, tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float]]
        a tuple of the completed optuna study and a tuple of the inversion results:
        topography dataframe, gravity dataframe, parameter values and elapsed time.
    """

    optuna.logging.set_verbosity(optuna.logging.WARN)

    # if sampler not provided, use BoTorch as default unless grid_search is True
    if sampler is None:
        if grid_search is True:
            if n_trials < 4:
                msg = (
                    "if grid_search is True, n_trials must be at least 4, "
                    "resetting n_trials to 4 now."
                )
                log.warning(msg)
                n_trials = 4
            space = np.logspace(
                np.log10(damping_limits[0]), np.log10(damping_limits[1]), n_trials
            )
            # omit first and last since they will be enqueued separately
            space = space[1:-1]
            sampler = optuna.samplers.GridSampler(
                search_space={"damping": space},
                seed=10,
            )
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="BoTorch")
                sampler = optuna.integration.BoTorchSampler(
                    n_startup_trials=int(n_trials / 4),
                    seed=10,
                )

    # set file name for saving results with random number between 0 and 999
    if fname is None:
        fname = f"tmp_{random.randint(0,999)}"

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        load_if_exists=False,
    )

    # explicitly add the limits as trials
    # if grid_search is False:
    study.enqueue_trial({"damping": damping_limits[0]}, skip_if_exists=True)
    study.enqueue_trial({"damping": damping_limits[1]}, skip_if_exists=True)

    # run optimization
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="logei_candidates_func is experimental"
        )
        study.optimize(
            OptimalInversionDamping(
                damping_limits=damping_limits,
                rmse_as_median=score_as_median,
                training_data=training_df,
                testing_data=testing_df,
                fname=fname,
                plot_grids=plot_grids,
                **kwargs,
            ),
            n_trials=n_trials,
            callbacks=[warn_limits_better_than_trial_1_param],
            show_progress_bar=True,
        )

    best_trial = study.best_trial

    if best_trial.params.get("damping") in damping_limits:
        log.warning(
            "Best damping value (%s) is at the limit of provided values "
            "(%s) and thus is likely not a global minimum, expand the range of "
            "values tested to ensure the best parameter value is found.",
            best_trial.params.get("damping"),
            damping_limits,
        )

    log.info("Trial with lowest score: ")
    log.info("\ttrial number: %s", best_trial.number)
    log.info("\tparameter: %s", best_trial.params)
    log.info("\tscores: %s", best_trial.values)

    # get best inversion result of each set
    with pathlib.Path(f"{fname}_trial_{best_trial.number}.pickle").open("rb") as f:
        inv_results = pickle.load(f)

    # delete other inversion results
    for i in range(n_trials):
        if i == best_trial.number:
            pass
        else:
            pathlib.Path(f"{fname}_trial_{i}.pickle").unlink(missing_ok=True)

    # remove if exists
    pathlib.Path(fname).unlink(missing_ok=True)

    # save study to pickle
    with pathlib.Path(f"{fname}.pickle").open("wb") as f:
        pickle.dump(study, f)

    if plot_cv is True:
        plotting.plot_cv_scores(
            study.trials_dataframe().value.values,
            study.trials_dataframe().params_damping.values,
            param_name="Damping",
            logx=logx,
            logy=logy,
        )
    return study, inv_results


class OptimalInversionZrefDensity:
    """
    Objective function to use in an Optuna optimization for finding the optimal values
    for zref and or density contrast values for a gravity inversion. Used within
    function `optimize_inversion_zref_density_contrast()`.
    """

    def __init__(
        self,
        fname: str,
        grav_df: pd.DataFrame,
        constraints_df: pd.DataFrame | list[pd.DataFrame],
        zref: float | None = None,
        zref_limits: tuple[float, float] | None = None,
        density_contrast_limits: tuple[float, float] | None = None,
        density_contrast: float | None = None,
        starting_topography: xr.DataArray | None = None,
        starting_topography_kwargs: dict[str, typing.Any] | None = None,
        regional_grav_kwargs: dict[str, typing.Any] | None = None,
        progressbar: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        self.fname = fname
        self.grav_df = grav_df
        self.constraints_df = constraints_df
        self.zref_limits = zref_limits
        self.density_contrast_limits = density_contrast_limits
        self.zref = zref
        self.density_contrast = density_contrast
        self.starting_topography = starting_topography
        self.starting_topography_kwargs = starting_topography_kwargs
        self.regional_grav_kwargs = regional_grav_kwargs
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
            the score of the eq_sources fit
        """
        grav_df = self.grav_df.copy()

        cols = [
            "easting",
            "northing",
            "upward",
            "gravity_anomaly",
        ]
        if all(i in grav_df.columns for i in cols) is False:
            msg = f"`grav_df` needs all the following columns: {cols}"
            raise ValueError(msg)

        kwargs = self.kwargs.copy()

        if kwargs.get("apply_weighting_grid", None) is True:
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
            zref = trial.suggest_float(
                "zref",
                self.zref_limits[0],
                self.zref_limits[1],
            )
        else:
            zref = self.zref
            if zref is None:
                msg = "must provide zref if zref_limits not provided"
                raise ValueError(msg)

        if self.density_contrast_limits is not None:
            density_contrast = trial.suggest_float(
                "density_contrast",
                self.density_contrast_limits[0],
                self.density_contrast_limits[1],
            )
        else:
            density_contrast = self.density_contrast
            if density_contrast is None:
                msg = (
                    "must provide density_contrast if density_contrast_limits not "
                    "provided"
                )
                raise ValueError(msg)

        if self.starting_topography is None:
            msg = (
                "starting_topography not provided, will create a flat surface at each "
                "zref value to be the starting topography."
            )
            log.warning(msg)
            if self.starting_topography_kwargs is None:
                msg = (
                    "must provide `starting_topography_kwargs` with items `region` and "
                    "`spacing` to create the starting topography for each zref level."
                )
                raise ValueError(msg)

        # raise warning about using constraint point minimization for regional
        # estimation
        if (
            (self.regional_grav_kwargs is not None)
            and (self.regional_grav_kwargs.get("regional_method") == "constraints")
            and (
                len(self.regional_grav_kwargs.get("constraints_df"))  # type: ignore[arg-type]
                == len(self.constraints_df)
            )
        ):
            msg = (
                "Using constraint point minimization technique for regional field "
                "estimation. This is not recommended as the constraint points are used "
                "for the density / reference level cross-validation scoring, which "
                "biases the scoring. Consider using a different method for regional "
                "field estimation, or separate constraints in training and testing "
                "sets and provide the training set to `regional_grav_kwargs` and the "
                "testing set to `constraints_df` to use for scoring."
            )
            log.warning(msg)

        # raise warning about using constraint points for regional estimation
        # if self.regional_grav_kwargs is not None:
        #     if self.regional_grav_kwargs.get("regional_method") == "constant":
        #         if self.regional_grav_kwargs.get("constraints_df", None) is not None:
        #             if len(
        #                 self.regional_grav_kwargs.get("constraints_df", None)
        #             ) == len(self.constraints_df):
        #                 msg = (
        #                     "Using constraint points for estimating a constant "
        #                     "regional field.This is not recommended as the constraint"
        #                     "points are used for the density / reference level "
        #                     "cross-validation scoring, which biases the scoring. "
        #                     "Consider using a constant value not determined from the "
        #                     "constraints, a different method for regional field "
        #                     "estimation, or separate constraints in training and "
        #                     "testing sets and provide the training set to "
        #                     "`regional_grav_kwargs` and the testing set to "
        #                     "`constraints_df` to use for scoring."
        #                 )
        #                 log.warning(msg)

        # make flat starting topo at zref if not provided
        if self.starting_topography is None:
            starting_topo = utils.create_topography(
                method="flat",
                region=self.starting_topography_kwargs.get("region"),  # type: ignore[union-attr,arg-type]
                spacing=self.starting_topography_kwargs.get("spacing"),  # type: ignore[union-attr,arg-type]
                upwards=zref,
            )
        else:
            starting_topo = self.starting_topography.copy()

        # re-calculate density grid with new density contrast
        density_grid = xr.where(
            starting_topo >= zref,
            density_contrast,
            -density_contrast,  # pylint: disable=invalid-unary-operand-type
        )

        # create layer of prisms
        starting_prisms = utils.grids_to_prisms(
            starting_topo,
            reference=zref,
            density=density_grid,
        )

        # calculate forward gravity of starting prism layer
        grav_df["starting_gravity"] = starting_prisms.prism_layer.gravity(
            coordinates=(
                grav_df.easting,
                grav_df.northing,
                grav_df.upward,
            ),
            field="g_z",
            progressbar=False,
        )

        # calculate regional field
        reg_kwargs = self.regional_grav_kwargs.copy()  # type: ignore[union-attr]

        if isinstance(self.constraints_df, list):
            regional_method = reg_kwargs.pop("regional_method", None)
            training_constraints = reg_kwargs.pop("constraints_df", None)
            testing_constraints = self.constraints_df

            if training_constraints is None:
                pass
            elif isinstance(training_constraints, pd.DataFrame):
                msg = (
                    "must provide a list of training constraints dataframes for "
                    "cross-validation to parameter `constraints_df` of "
                    "`regional_grav_kwargs`."
                )
                raise ValueError(msg)

            # get list of folds
            folds = [
                list(df.columns[df.columns.str.startswith("fold_")])[0]  # noqa: RUF015
                for df in self.constraints_df
            ]

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

            # for each fold, run CV
            scores = []
            for i, _ in enumerate(pbar):
                log.addFilter(log_filter)
                grav_df = regional.regional_separation(
                    method=regional_method,
                    grav_df=grav_df,
                    constraints_df=training_constraints[i],
                    **reg_kwargs,
                )
                log.removeFilter(log_filter)

                new_kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key
                    not in [
                        "zref",
                        "density_contrast",
                        "progressbar",
                        "results_fname",
                        "prism_layer",
                    ]
                }

                trial.set_user_attr("fname", f"{self.fname}_trial_{trial.number}")

                # run cross validation
                score, _ = cross_validation.constraints_cv_score(
                    grav_df=grav_df,
                    constraints_df=testing_constraints[i],
                    results_fname=trial.user_attrs.get("fname"),
                    prism_layer=starting_prisms,
                    **new_kwargs,
                )
                scores.append(score)

            # get mean of scores of all folds
            score = np.mean(scores)

        else:
            assert isinstance(self.constraints_df, pd.DataFrame)

            log.addFilter(log_filter)
            grav_df = regional.regional_separation(
                method=reg_kwargs.pop("regional_method", None),
                grav_df=grav_df,
                **reg_kwargs,
            )
            log.removeFilter(log_filter)

            new_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key
                not in [
                    "zref",
                    "density_contrast",
                    "progressbar",
                    "results_fname",
                    "prism_layer",
                ]
            }

            trial.set_user_attr("fname", f"{self.fname}_trial_{trial.number}")

            # run cross validation
            score, _ = cross_validation.constraints_cv_score(
                grav_df=grav_df,
                constraints_df=self.constraints_df,
                results_fname=trial.user_attrs.get("fname"),
                prism_layer=starting_prisms,
                **new_kwargs,
            )

        return score


def optimize_inversion_zref_density_contrast(
    grav_df: pd.DataFrame,
    constraints_df: pd.DataFrame | list[pd.DataFrame],
    n_trials: int,
    starting_topography: xr.DataArray | None = None,
    zref_limits: tuple[float, float] | None = None,
    density_contrast_limits: tuple[float, float] | None = None,
    zref: float | None = None,
    density_contrast: float | None = None,
    starting_topography_kwargs: dict[str, typing.Any] | None = None,
    regional_grav_kwargs: dict[str, typing.Any] | None = None,
    score_as_median: bool = False,
    sampler: optuna.samplers.BaseSampler | None = None,
    grid_search: bool = False,
    fname: str | None = None,
    plot_cv: bool = True,
    logx: bool = False,
    logy: bool = False,
    fold_progressbar: bool = True,
    **kwargs: typing.Any,
) -> tuple[
    optuna.study, tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float]
]:
    """
    Use Optuna to find the optimal zref and or density contrast values for a gravity
    inversion. The optimization aims to minimize the cross-validation score, represented
    by the root mean (or median) squared error (RMSE), between the testing gravity data,
    and the predict gravity data after and inversion. Follows methods of
    :footcite:t:`uiedafast2017`. This can optimize for either zref, density contrast, or
    both at the same time. Provide upper and low limits for each parameter, number of
    trials and let Optuna choose the best parameter values for each trial or use a grid
    search to test all values between the limits in intervals of n_trials. The results
    are saved to a pickle file with the best inversion results and the study.

    Parameters
    ----------
    grav_df : pd.DataFrame
        gravity data frame with columns `easting`, `northing`, `upward`, and
        `gravity_anomaly`
    constraints_df : pd.DataFrame or list[pd.DataFrame]
        constraints data frame with columns `easting`, `northing`, and `upward`, or list
        of dataframes for each fold of a cross-validation
    n_trials : int
        number of trials, if grid_search is True, needs to be a perfect square and >=16.
    starting_topography : xr.DataArray | None, optional
        a starting topography grid used to create the prisms layers. If not provided,
        must provide region and spacing to starting_topography_kwargs, by default None
    zref_limits : tuple[float, float] | None, optional
        upper and lower limits for the reference level, in meters, by default None
    density_contrast_limits : tuple[float, float] | None, optional
        upper and lower limits for the density contrast, in kg/m^-3, by default None
    zref : float | None, optional
        if zref_limits not provided, must provide a constant zref value, by default None
    density_contrast : float | None, optional
        if density_contrast_limits not provided, must provide a constant density
        contrast value, by default None
    starting_topography_kwargs : dict[str, typing.Any] | None, optional
        dictionary with region and spacing arguments used to create a flat starting
        topography at each zref value if starting_topography not provided, by default
        None
    regional_grav_kwargs : dict[str, typing.Any] | None, optional
        dictionary with kwargs to supply to `regional.regional_separation()`, by default
        None
    score_as_median : bool, optional
        change scoring metric from root mean square to root median square, by default
        False
    sampler : optuna.samplers.BaseSampler | None, optional
        customize the optuna sampler, by default uses BoTorch sampler unless grid_search
        is True, then uses GridSampler.
    grid_search : bool, optional
        Switch the sampler to GridSampler and search entire parameter space between
        provided limits in intervals set by n_trials (for 1 parameter optimizations), or
        by the square root of n_trials (for 2 parameter optimizations), by default False
    fname : str | None, optional
        filename to save both the inversion results and study to as pickle files, by
        default None
    plot_cv : bool, optional
        plot the cross-validation results, by default True
    logx : bool, optional
        use a log scale for the cross-validation plot x-axis, by default False
    logy : bool, optional
        use a log scale for the cross-validation plot y-axis, by default False
    fold_progressbar : bool, optional
        show a progress bar for each fold of the constraint-point minimization
        cross-validation, by default True

    Returns
    -------
    tuple[
        optuna.study,
        tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float] ]
        a tuple of the completed optuna study and a tuple of the inversion results:
        topography dataframe, gravity dataframe, parameter values and elapsed time.
    """

    if "test" in grav_df.columns:
        assert (
            grav_df.test.any()
        ), "test column contains True value, not needed except for during damping CV"

    optuna.logging.set_verbosity(optuna.logging.WARN)

    # if sampler not provided, use BoTorch as default unless grid_search is True
    if sampler is None:
        if grid_search is True:
            if zref_limits is None:
                if n_trials < 4:
                    msg = (
                        "if grid_search is True, n_trials must be at least 4, "
                        "resetting n_trials to 4 now."
                    )
                    log.warning(msg)
                    n_trials = 4
                space = np.linspace(
                    density_contrast_limits[0],  # type: ignore[index]
                    density_contrast_limits[1],  # type: ignore[index]
                    n_trials,
                )
                # omit first and last since they will be enqueued separately
                space = space[1:-1]
                sampler = optuna.samplers.GridSampler(
                    search_space={"density_contrast": space},
                    seed=10,
                )
            elif density_contrast_limits is None:
                if n_trials < 4:
                    msg = (
                        "if grid_search is True, n_trials must be at least 4, "
                        "resetting n_trials to 4 now."
                    )
                    log.warning(msg)
                    n_trials = 4
                space = np.linspace(zref_limits[0], zref_limits[1], n_trials)
                # omit first and last since they will be enqueued separately
                space = space[1:-1]
                sampler = optuna.samplers.GridSampler(
                    search_space={"zref": space},
                    seed=10,
                )
            else:
                if n_trials < 16:
                    msg = (
                        "if grid_search is True, n_trials must be at least 16, "
                        "resetting n_trials to 16 now."
                    )
                    log.warning(msg)
                    n_trials = 16

                # n_trials needs to be square for 2 param grid search so each param has
                # sqrt(n_trials).
                if np.sqrt(n_trials).is_integer() is False:
                    # get next largest square number
                    old_n_trials = n_trials
                    n_trials = (math.floor(math.sqrt(n_trials)) + 1) ** 2
                    msg = (
                        "if grid_search is True with provided limits for both zref and "
                        "density contrast, n_trials (%s) must have an integer square "
                        "root. Resetting n_trials to to next largest compatible value "
                        "now (%s)"
                    )
                    log.warning(msg, old_n_trials, n_trials)

                zref_space = np.linspace(
                    zref_limits[0],
                    zref_limits[1],
                    int(np.sqrt(n_trials)),
                )

                density_contrast_space = np.linspace(
                    density_contrast_limits[0],
                    density_contrast_limits[1],
                    int(np.sqrt(n_trials)),
                )

                # omit first and last since they will be enqueued separately
                sampler = optuna.samplers.GridSampler(
                    search_space={
                        "zref": zref_space[1:-1],
                        "density_contrast": density_contrast_space[1:-1],
                    },
                    seed=10,
                )
        else:
            with warnings.catch_warnings():
                # if optimizing on both zref and density, do more startup trials to
                # cover param space
                if (zref_limits is not None) & (density_contrast_limits is not None):
                    n_startup_trials = int(n_trials / 3)
                else:
                    n_startup_trials = int(n_trials / 4)
                warnings.filterwarnings("ignore", message="BoTorch")
                sampler = optuna.integration.BoTorchSampler(
                    n_startup_trials=n_startup_trials,
                    seed=10,
                )

    # set file name for saving results with random number between 0 and 999
    if fname is None:
        fname = f"tmp_{random.randint(0,999)}"

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        load_if_exists=False,
    )

    # explicitly add the limits as trials
    if zref_limits is None:
        study.enqueue_trial({"density_contrast": density_contrast_limits[0]})  # type: ignore[index]
        study.enqueue_trial({"density_contrast": density_contrast_limits[1]})  # type: ignore[index]
    elif density_contrast_limits is None:
        study.enqueue_trial({"zref": zref_limits[0]})
        study.enqueue_trial({"zref": zref_limits[1]})
    else:
        if grid_search is True:
            a = range(int(np.sqrt(n_trials)))
            full_pairs = list(itertools.product(a, a))

            a = range(int(np.sqrt(n_trials)))[1:-1]
            inside_pairs = list(itertools.product(a, a))

            pairs_to_enqueue = [p for p in full_pairs if p not in inside_pairs]

            for pair in pairs_to_enqueue:
                study.enqueue_trial(
                    {
                        "zref": zref_space[pair[0]],
                        "density_contrast": density_contrast_space[pair[1]],
                    },
                )
        else:
            study.enqueue_trial(
                {
                    "zref": zref_limits[0],
                    "density_contrast": density_contrast_limits[0],
                },
            )
            study.enqueue_trial(
                {
                    "zref": zref_limits[0],
                    "density_contrast": density_contrast_limits[1],
                },
            )
            study.enqueue_trial(
                {
                    "zref": zref_limits[1],
                    "density_contrast": density_contrast_limits[0],
                },
            )
            study.enqueue_trial(
                {
                    "zref": zref_limits[1],
                    "density_contrast": density_contrast_limits[1],
                },
            )

    # run optimization
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="logei_candidates_func is experimental"
        )
        study.optimize(
            OptimalInversionZrefDensity(
                grav_df=grav_df,
                constraints_df=constraints_df,
                zref_limits=zref_limits,
                density_contrast_limits=density_contrast_limits,
                zref=zref,
                density_contrast=density_contrast,
                starting_topography=starting_topography,
                starting_topography_kwargs=starting_topography_kwargs,
                regional_grav_kwargs=regional_grav_kwargs,
                rmse_as_median=score_as_median,
                fname=fname,
                progressbar=fold_progressbar,
                **kwargs,
            ),
            n_trials=n_trials,
            callbacks=[warn_limits_better_than_trial_multi_params],
            show_progress_bar=True,
        )

    best_trial = study.best_trial

    if zref_limits is not None:  # noqa: SIM102
        if best_trial.params.get("zref") in zref_limits:
            log.warning(
                "Best zref value (%s) is at the limit of provided values (%s) and "
                "thus is likely not a global minimum, expand the range of values "
                "tested to ensure the best parameter value is found.",
                best_trial.params.get("zref"),
                zref_limits,
            )
    if density_contrast_limits is not None:  # noqa: SIM102
        if best_trial.params.get("density_contrast") in density_contrast_limits:
            log.warning(
                "Best density contrast value (%s) is at the limit of provided values "
                "(%s) and thus is likely not a global minimum, expand the range of "
                "values tested to ensure the best parameter value is found.",
                best_trial.params.get("density_contrast"),
                density_contrast_limits,
            )
    log.info("Trial with lowest score: ")
    log.info("\ttrial number: %s", best_trial.number)
    log.info("\tparameter: %s", best_trial.params)
    log.info("\tscores: %s", best_trial.values)

    # get best inversion result of each
    with pathlib.Path(f"{fname}_trial_{best_trial.number}.pickle").open("rb") as f:
        inv_results = pickle.load(f)

    # delete other inversion results
    for i in range(n_trials):
        if i == best_trial.number:
            pass
        else:
            pathlib.Path(f"{fname}_trial_{i}.pickle").unlink(missing_ok=True)

    # remove if exists
    pathlib.Path(fname).unlink(missing_ok=True)

    # save study to pickle
    with pathlib.Path(f"{fname}.pickle").open("wb") as f:
        pickle.dump(study, f)

    if plot_cv is True:
        if zref_limits is None:
            plotting.plot_cv_scores(
                study.trials_dataframe().value.values,
                study.trials_dataframe().params_density_contrast.values,
                param_name="Density contrast (kg/m$^3$)",
                plot_title="Density contrast Cross-validation",
                logx=logx,
                logy=logy,
            )
        elif density_contrast_limits is None:
            plotting.plot_cv_scores(
                study.trials_dataframe().value.values,
                study.trials_dataframe().params_zref.values,
                param_name="Reference level (m)",
                plot_title="Reference level Cross-validation",
                logx=logx,
                logy=logy,
            )
        else:
            if grid_search is True:
                parameter_pairs = list(
                    zip(
                        study.trials_dataframe().params_zref,
                        study.trials_dataframe().params_density_contrast,
                    )
                )
                plotting.plot_2_parameter_cv_scores(
                    study.trials_dataframe().value.values,
                    parameter_pairs,
                    param_names=("Reference level (m)", "Density contrast (kg/m$^3$)"),
                )
            else:
                plotting.plot_2_parameter_cv_scores_uneven(
                    study,
                    param_names=(
                        "params_zref",
                        "params_density_contrast",
                    ),
                    plot_param_names=(
                        "Reference level (m)",
                        "Density contrast (kg/m$^3$)",
                    ),
                )

    return study, inv_results


def optimize_inversion_zref_density_contrast_kfolds(
    testing_training_constraints_df: pd.DataFrame,
    plot_cv: bool = True,
    fold_progressbar: bool = True,
    **kwargs: typing.Any,
) -> tuple[
    optuna.study, tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float]
]:
    """
    Perform a cross validation for the optimal zref and density contrast values same as
    function `optimize_inversion_zref_density_contrast`, but pass a dataframe of
    constraint points which contains folds of testing and training data (generated with
    `cross_validation.split_test_train`) so for each iteration of the zref/density cross
    validation, the regional separation is performed with 1 training fold, and the
    scoring is performed with 1 testing fold. This is only useful if the regional
    separation technique you supply via `regional_grav_kwargs` uses constraints points
    for the estimations, such as constraint point minimization.

    Parameters
    ----------
    testing_training_constraints_df : pd.DataFrame
        dataframe of constraints with columns "fold_x" for x folds, with values of
        'test' or 'train'.
    plot_cv : bool, optional
        plot the density and/or zref parameters values and the resulting scores, by
        default True
    fold_progressbar : bool, optional
        add a progressbar for each fold, by default True

    Returns
    -------
    tuple[ optuna.study,
    tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float] ]
        returns the optuna study, and a tuple of the best inversion results.
    """
    test_dfs, train_dfs = cross_validation.kfold_df_to_lists(
        testing_training_constraints_df
    )

    kwargs.pop("plot_cv", False)

    regional_grav_kwargs_original = kwargs.pop("regional_grav_kwargs", None).copy()

    regional_grav_kwargs = regional_grav_kwargs_original.copy()

    regional_grav_kwargs.pop("constraints_df", None)

    regional_grav_kwargs["constraints_df"] = train_dfs

    study, _ = optimize_inversion_zref_density_contrast(
        constraints_df=test_dfs,
        fold_progressbar=fold_progressbar,
        regional_grav_kwargs=regional_grav_kwargs,
        plot_cv=False,
        **kwargs,
    )

    best_trial = study.best_trial

    # log.info("Trial with lowest score: ")
    # log.info("\ttrial number: %s", best_trial.number)
    # log.info("\tparameter: %s", best_trial.params)
    # log.info("\tscores: %s", best_trial.values)

    # redo inversion with best parameters
    zref = best_trial.params.get("zref", None)
    density_contrast = best_trial.params.get("density_contrast", None)

    if zref is None:
        zref = kwargs.get("zref")
    if density_contrast is None:
        density_contrast = kwargs.get("density_contrast")

    fname = kwargs.get("fname", None)
    if fname is not None:
        fname = f"{fname}_inversion_results"

    new_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key
        not in [
            "zref",
            "density_contrast",
            "progressbar",
            "results_fname",
            "prism_layer",
            "density_contrast_limits",
            "zref_limits",
            "n_trials",
            "fname",
            "grid_search",
        ]
    }
    # run the inversion workflow
    log.addFilter(log_filter)
    final_inversion_results = inversion.run_inversion_workflow(
        create_starting_prisms=True,
        starting_prisms_kwargs={
            "zref": zref,
            "density_contrast": density_contrast,
        },
        calculate_regional_misfit=True,
        regional_grav_kwargs=regional_grav_kwargs_original,
        fname=fname,
        plot_convergence=True,
        **new_kwargs,
    )
    log.removeFilter(log_filter)

    if plot_cv is True:
        if kwargs.get("zref_limits", None) is None:
            plotting.plot_cv_scores(
                study.trials_dataframe().value.values,
                study.trials_dataframe().params_density_contrast.values,
                param_name="Density contrast (kg/m$^3$)",
                plot_title="Density contrast Cross-validation",
            )
        elif kwargs.get("density_contrast_limits", None) is None:
            plotting.plot_cv_scores(
                study.trials_dataframe().value.values,
                study.trials_dataframe().params_zref.values,
                param_name="Reference level (m)",
                plot_title="Reference level Cross-validation",
            )
        else:
            if kwargs.get("grid_search", False) is True:
                parameter_pairs = list(
                    zip(
                        study.trials_dataframe().params_zref,
                        study.trials_dataframe().params_density_contrast,
                    )
                )
                plotting.plot_2_parameter_cv_scores(
                    study.trials_dataframe().value.values,
                    parameter_pairs,
                    param_names=("Reference level (m)", "Density contrast (kg/m$^3$)"),
                )
            else:
                plotting.plot_2_parameter_cv_scores_uneven(
                    study,
                    param_names=(
                        "params_zref",
                        "params_density_contrast",
                    ),
                    plot_param_names=(
                        "Reference level (m)",
                        "Density contrast (kg/m$^3$)",
                    ),
                )

    return study, final_inversion_results


class OptimalEqSourceParams:
    """
    Objective function to use in an Optuna optimization for finding the optimal
    equivalent source parameters for fitting to gravity data.
    """

    def __init__(
        self,
        source_depth_limits: tuple[float, float] | None = None,
        block_size_limits: tuple[float, float] | None = None,
        eq_damping_limits: tuple[float, float] | None = None,
        **kwargs: typing.Any,
    ) -> None:
        self.source_depth_limits = source_depth_limits
        self.block_size_limits = block_size_limits
        self.eq_damping_limits = eq_damping_limits
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
        kwargs_to_remove = []
        if self.source_depth_limits is not None:
            source_depth = trial.suggest_float(
                "source_depth",
                self.source_depth_limits[0],
                self.source_depth_limits[1],
            )
        else:
            source_depth = self.kwargs.get("source_depth", "default")
            kwargs_to_remove.append("source_depth")

        if self.block_size_limits is not None:
            block_size = trial.suggest_float(
                "block_size",
                self.block_size_limits[0],
                self.block_size_limits[1],
            )
        else:
            block_size = self.kwargs.get("block_size", None)
            kwargs_to_remove.append("block_size")

        if self.eq_damping_limits is not None:
            eq_damping = trial.suggest_float(
                "eq_damping",
                self.eq_damping_limits[0],
                self.eq_damping_limits[1],
                log=True,
            )
        else:
            eq_damping = self.kwargs.get("eq_damping", None)
            kwargs_to_remove.append("eq_damping")

        new_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key not in kwargs_to_remove
        }

        return cross_validation.eq_sources_score(
            damping=eq_damping,
            depth=source_depth,
            block_size=block_size,
            **new_kwargs,
        )


def optimize_eq_source_params(
    coordinates: tuple[pd.Series | NDArray, pd.Series | NDArray, pd.Series | NDArray],
    data: pd.Series | NDArray,
    points: NDArray | None = None,
    n_trials: int = 100,
    eq_damping_limits: tuple[float, float] | None = None,
    source_depth_limits: tuple[float, float] | None = None,
    block_size_limits: tuple[float, float] | None = None,
    weights: NDArray | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    plot: bool = False,
    progressbar: bool = True,
    **kwargs: typing.Any,
) -> tuple[optuna.study, hm.EquivalentSources]:
    """
    Use Optuna to find the optimal parameters for fitting equivalent sources to gravity
    data.

    Parameters
    ----------
    coordinates : tuple[pd.Series | NDArray, pd.Series | NDArray, pd.Series | NDArray]
        tuple of coordinates in the order (easting, northing, upward) for the gravity
        observation locations.
    data : pd.Series | NDArray
        gravity data values
    points : NDArray | None, optional
        specify the coordinates of source points, by default None
    n_trials : int, optional
        number of trials to run, by default 100
    eq_damping_limits : tuple[float, float], optional
        damping parameter limits, by default (0, 10**3)
    source_depth_limits : tuple[float, float], optional
        source depth limits (positive downwards) in meters, by default (0, 10e6)
    block_size_limits : tuple[float, float] | None, optional
        block size limits in meters, by default None
    weights : NDArray | None, optional
        weights for the gravity data, typically 1/(uncertainty**2), by default None
    sampler : optuna.samplers.BaseSampler | None, optional
        specify which Optuna sampler to use, by default None
    plot : bool, optional
        plot the resulting optimization figures, by default False
    progressbar : bool, optional
        add a progressbar, by default True
    Returns
    -------
    tuple[optuna., hm.EquivalentSources]
        a tuple of the resulting Optuna study and the fitted equivalent sources model
    """
    optuna.logging.set_verbosity(optuna.logging.WARN)

    # if sampler not provided, used TPE as default
    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(n_trials / 4),
            seed=10,
        )

    # create study
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        load_if_exists=False,
    )

    # explicitly add the limits as trials
    num_params = 0
    if source_depth_limits is not None:
        study.enqueue_trial(
            {"source_depth": source_depth_limits[0]}, skip_if_exists=True
        )
        study.enqueue_trial(
            {"source_depth": source_depth_limits[1]}, skip_if_exists=True
        )
        num_params += 1
    if block_size_limits is not None:
        study.enqueue_trial({"block_size": block_size_limits[0]}, skip_if_exists=True)
        study.enqueue_trial({"block_size": block_size_limits[1]}, skip_if_exists=True)
        num_params += 1
    if eq_damping_limits is not None:
        study.enqueue_trial({"eq_damping": eq_damping_limits[0]}, skip_if_exists=True)
        study.enqueue_trial({"eq_damping": eq_damping_limits[1]}, skip_if_exists=True)
        num_params += 1

    if num_params == 1:
        callbacks = [warn_limits_better_than_trial_1_param]
    else:
        callbacks = [warn_limits_better_than_trial_multi_params]

    study.optimize(
        OptimalEqSourceParams(
            source_depth_limits=source_depth_limits,
            block_size_limits=block_size_limits,
            eq_damping_limits=eq_damping_limits,
            coordinates=coordinates,
            data=data,
            points=points,
            **kwargs,
        ),
        n_trials=n_trials,
        callbacks=callbacks,
        show_progress_bar=progressbar,
    )

    log.info("Best params: %s", study.best_params)
    log.info("Best trial: %s", study.best_trial.number)
    log.info("Best score: %s", study.best_trial.value)

    best_damping = study.best_params.get("eq_damping", None)
    if best_damping is None:
        best_damping = kwargs.get("damping")

    best_source_depth = study.best_params.get("source_depth", None)
    if best_source_depth is None:
        best_source_depth = kwargs.get("source_depth", None)

    best_block_size = study.best_params.get("block_size", None)
    if best_block_size is None:
        best_block_size = kwargs.get("block_size", None)

    eqs = hm.EquivalentSources(
        damping=best_damping,
        depth=best_source_depth,
        block_size=best_block_size,
        points=points,
    )
    eqs.fit(coordinates, data, weights=weights)

    if plot is True:
        plotting.plot_optuna_figures(
            study,
            target_names=["score"],
            plot_history=False,
            plot_slice=True,
            plot_importance=True,
            include_duration=False,
        )

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
        **kwargs: typing.Any,
    ) -> None:
        self.trend_limits = trend_limits
        self.optimize_on_true_regional_misfit = optimize_on_true_regional_misfit
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

        log.addFilter(log_filter)

        residual_constraint_score, residual_amplitude_score, true_reg_score, df = (
            cross_validation.regional_separation_score(
                method="trend",
                trend=trend,
                **self.kwargs,
            )
        )
        log.removeFilter(log_filter)

        trial.set_user_attr("results", df)
        trial.set_user_attr("true_reg_score", true_reg_score)

        if self.optimize_on_true_regional_misfit is True:
            trial.set_user_attr("residual constraint score", residual_constraint_score)
            trial.set_user_attr("residual amplitude score", residual_amplitude_score)
            return true_reg_score  # type: ignore[return-value]
        return res_score, reg_score


class OptimizeRegionalFilter:
    """
    Objective function to use in an Optuna optimization for finding the optimal filter
    width for estimation the regional component of gravity misfit.
    """

    def __init__(
        self,
        filter_width_limits: tuple[float, float],
        optimize_on_true_regional_misfit: bool = False,
        **kwargs: typing.Any,
    ) -> None:
        self.filter_width_limits = filter_width_limits
        self.optimize_on_true_regional_misfit = optimize_on_true_regional_misfit
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

        log.addFilter(log_filter)

        residual_constraint_score, residual_amplitude_score, true_reg_score, df = (
            cross_validation.regional_separation_score(
                method="filter",
                filter_width=filter_width,
                **self.kwargs,
            )
        )
        log.removeFilter(log_filter)

        trial.set_user_attr("results", df)
        trial.set_user_attr("true_reg_score", true_reg_score)

        if self.optimize_on_true_regional_misfit is True:
            trial.set_user_attr("residual constraint score", residual_constraint_score)
            trial.set_user_attr("residual amplitude score", residual_amplitude_score)
            return true_reg_score  # type: ignore[return-value]
        return res_score, reg_score  # type: ignore[return-value]


class OptimizeRegionalEqSources:
    """
    Objective function to use in an Optuna optimization for finding the optimal
    equivalent source parameters for estimation the regional component of gravity
    misfit.
    """

    def __init__(
        self,
        source_depth_limits: tuple[float, float] | None = None,
        block_size_limits: tuple[float, float] | None = None,
        eq_damping_limits: tuple[float, float] | None = None,
        optimize_on_true_regional_misfit: bool = False,
        **kwargs: typing.Any,
    ) -> None:
        self.source_depth_limits = source_depth_limits
        self.block_size_limits = block_size_limits
        self.eq_damping_limits = eq_damping_limits
        self.optimize_on_true_regional_misfit = optimize_on_true_regional_misfit
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

        if self.source_depth_limits is not None:
            source_depth = trial.suggest_float(
                "source_depth",
                self.source_depth_limits[0],
                self.source_depth_limits[1],
            )
        else:
            source_depth = self.kwargs.get("source_depth", None)
            if source_depth is None:
                msg = "must provide source_depth if source_depth_limits not provided"
                raise ValueError(msg)

        if self.block_size_limits is not None:
            block_size = trial.suggest_float(
                "block_size",
                self.block_size_limits[0],
                self.block_size_limits[1],
            )
        else:
            block_size = self.kwargs.get("block_size", None)

        if self.eq_damping_limits is not None:
            eq_damping = trial.suggest_float(
                "eq_damping",
                self.eq_damping_limits[0],
                self.eq_damping_limits[1],
                log=True,
            )
        else:
            eq_damping = self.kwargs.get("eq_damping", None)

        new_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key not in ["source_depth", "block_size", "eq_damping"]
        }

        log.addFilter(log_filter)

        residual_constraint_score, residual_amplitude_score, true_reg_score, df = (
            cross_validation.regional_separation_score(
                method="eq_sources",
                source_depth=source_depth,
                block_size=block_size,
                eq_damping=eq_damping,
                **new_kwargs,
            )
        )
        log.removeFilter(log_filter)

        trial.set_user_attr("results", df)
        trial.set_user_attr("true_reg_score", true_reg_score)

        if self.optimize_on_true_regional_misfit is True:
            trial.set_user_attr("residual constraint score", residual_constraint_score)
            trial.set_user_attr("residual amplitude score", residual_amplitude_score)
            return true_reg_score  # type: ignore[return-value]
        return res_score, reg_score  # type: ignore[return-value]


class OptimizeRegionalConstraintsPointMinimization:
    """
    Objective function to use in an Optuna optimization for finding the optimal
    hyperparameter values the Constraint Point Minimization technique for estimation the
    regional component of gravity misfit.
    """

    def __init__(
        self,
        training_df: pd.DataFrame,
        testing_df: pd.DataFrame,
        grid_method: str,
        # for tensioned minimum curvature gridding
        tension_factor_limits: tuple[float, float] = (0, 1),
        # for bi-harmonic spline gridding
        spline_damping_limits: tuple[float, float] | None = None,
        # for eq source gridding
        source_depth_limits: tuple[float, float] | None = None,
        block_size_limits: tuple[float, float] | None = None,
        eq_damping_limits: tuple[float, float] | None = None,
        grav_obs_height_limits: tuple[float, float] | None = None,
        # other args
        optimize_on_true_regional_misfit: bool = False,
        progressbar: bool = False,
        **kwargs: typing.Any,
    ) -> None:
        self.training_df = training_df
        self.testing_df = testing_df
        self.grid_method = grid_method
        self.tension_factor_limits = tension_factor_limits
        self.spline_damping_limits = spline_damping_limits
        self.source_depth_limits = source_depth_limits
        self.block_size_limits = block_size_limits
        self.eq_damping_limits = eq_damping_limits
        self.grav_obs_height_limits = grav_obs_height_limits
        self.optimize_on_true_regional_misfit = optimize_on_true_regional_misfit
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

        new_kwargs = self.kwargs.copy()

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
            new_kwargs["spline_damping"] = trial.suggest_float(
                "spline_damping",
                self.spline_damping_limits[0],
                self.spline_damping_limits[1],
                log=True,
            )

        elif self.grid_method == "eq_sources":
            if self.source_depth_limits is not None:
                new_kwargs["source_depth"] = trial.suggest_float(
                    "source_depth",
                    self.source_depth_limits[0],
                    self.source_depth_limits[1],
                )
            else:
                new_kwargs["source_depth"] = self.kwargs.get("source_depth", "default")

            if self.block_size_limits is not None:
                new_kwargs["block_size"] = trial.suggest_float(
                    "block_size",
                    self.block_size_limits[0],
                    self.block_size_limits[1],
                )
            else:
                new_kwargs["block_size"] = self.kwargs.get("block_size", None)

            if self.eq_damping_limits is not None:
                new_kwargs["eq_damping"] = trial.suggest_float(
                    "eq_damping",
                    self.eq_damping_limits[0],
                    self.eq_damping_limits[1],
                    log=True,
                )
            else:
                new_kwargs["eq_damping"] = self.kwargs.get("eq_damping", None)

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

        if isinstance(self.training_df, list):
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

            log.addFilter(log_filter)

            # for each fold, run CV
            results = []
            for i, _ in enumerate(pbar):
                fold_results = cross_validation.regional_separation_score(
                    constraints_df=self.training_df[i],
                    testing_df=self.testing_df[i],
                    method="constraints",
                    grid_method=self.grid_method,
                    **new_kwargs,
                )
                results.append(fold_results)
            log.removeFilter(log_filter)

            # get mean of scores of all folds
            residual_constraint_score = np.mean([r[0] for r in results])
            residual_amplitude_score = np.mean([r[1] for r in results])
            try:
                true_reg_score = np.mean([r[2] for r in results])
            except TypeError:
                true_reg_score = None
            # df = pd.concat([r[3] for r in results])
            df = None

        else:
            assert isinstance(self.training_df, pd.DataFrame)
            assert isinstance(self.testing_df, pd.DataFrame)

            log.addFilter(log_filter)

            residual_constraint_score, residual_amplitude_score, true_reg_score, df = (
                cross_validation.regional_separation_score(
                    constraints_df=self.training_df,
                    testing_df=self.testing_df,
                    method="constraints",
                    grid_method=self.grid_method,
                    **new_kwargs,
                )
            )
            log.removeFilter(log_filter)

        trial.set_user_attr("results", df)
        trial.set_user_attr("true_reg_score", true_reg_score)

        if self.optimize_on_true_regional_misfit is True:
            trial.set_user_attr("residual constraint score", residual_constraint_score)
            trial.set_user_attr("residual amplitude score", residual_amplitude_score)
            return true_reg_score  # type: ignore[no-any-return]
        return res_score, reg_score  # type: ignore[return-value]


def optimize_regional_filter(
    testing_df: pd.DataFrame,
    grav_df: pd.DataFrame,
    filter_width_limits: tuple[float, float],
    score_as_median: bool = False,
    remove_starting_grav_mean: bool = False,
    true_regional: xr.DataArray | None = None,
    n_trials: int = 100,
    sampler: optuna.samplers.BaseSampler | None = None,
    plot: bool = False,
    plot_grid: bool = False,
    optimize_on_true_regional_misfit: bool = False,
) -> tuple[optuna.study, pd.DataFrame, optuna.trial.FrozenTrial]:
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
    testing_df : pd.DataFrame
        constraint points to use for calculating the score with columns "easting",
        "northing" and "upward".
    grav_df : pd.DataFrame
        gravity dataframe with columns "easting", "northing", "reg", and
        `gravity_anomaly`.
    filter_width_limits : tuple[float, float]
        limits to use for the filter width in meters.
    score_as_median : bool, optional
        use the root median square instead of the root mean square for the scoring
        metric, by default False
    remove_starting_grav_mean : bool, optional
        remove the mean of the starting gravity data before estimating the regional.
        Useful to mitigate effects of poorly-chosen zref value. By default False
    true_regional : xr.DataArray | None, optional
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

    Returns
    -------
    tuple[optuna.study, pd.DataFrame, optuna.trial.FrozenTrial]
        the completed Optuna study, the resulting gravity dataframe of the best trial,
        and the best trial itself.
    """

    optuna.logging.set_verbosity(optuna.logging.WARN)

    # if sampler not provided, use TPE as default
    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(n_trials / 4),
            seed=10,
        )

    if optimize_on_true_regional_misfit is True:
        if true_regional is None:
            msg = (
                "if optimizing on true regional misfit, must provide true_regional grid"
            )
            raise ValueError(msg)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            load_if_exists=False,
        )
    else:
        study = optuna.create_study(
            directions=[
                "minimize",
                "maximize",
            ],
            sampler=sampler,
            load_if_exists=False,
        )

    # run optimization
    study.optimize(
        OptimizeRegionalFilter(
            filter_width_limits=filter_width_limits,
            testing_df=testing_df,
            grav_df=grav_df,
            true_regional=true_regional,
            score_as_median=score_as_median,
            optimize_on_true_regional_misfit=optimize_on_true_regional_misfit,
            remove_starting_grav_mean=remove_starting_grav_mean,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    if optimize_on_true_regional_misfit is True:
        best_trial = study.best_trial
    else:
        best_trial = min(study.best_trials, key=lambda t: t.values[0])
        log.info("Number of trials on the Pareto front: %s", len(study.best_trials))

    log.info("Trial with lowest residual at constraint points: ")
    log.info("\ttrial number: %s", best_trial.number)
    log.info("\tparameter: %s", best_trial.params)
    log.info("\tscores: %s", best_trial.values)

    resulting_grav_df = best_trial.user_attrs.get("results")

    if plot is True:
        if study._is_multi_objective() is False:  # pylint: disable=protected-access
            if optimize_on_true_regional_misfit is True:
                plotting.combined_slice(
                    study,
                    attribute_names=[
                        "residual constraint score",
                        "residual amplitude score",
                    ],
                ).show()
            else:
                optuna.visualization.plot_slice(study).show()
        else:
            p = optuna.visualization.plot_pareto_front(study)
            plotting.remove_df_from_hoverdata(p).show()
            for i, j in enumerate(study.metric_names):
                optuna.visualization.plot_slice(
                    study,
                    target=lambda t: t.values[i],  # noqa: B023 # pylint: disable=cell-var-from-loop
                    target_name=j,
                ).show()
        if plot_grid is True:
            resulting_grav_df.set_index(["northing", "easting"]).to_xarray().reg.plot()

    return study, resulting_grav_df, best_trial


def optimize_regional_trend(
    testing_df: pd.DataFrame,
    grav_df: pd.DataFrame,
    trend_limits: tuple[int, int],
    score_as_median: bool = False,
    remove_starting_grav_mean: bool = False,
    true_regional: xr.DataArray | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    plot: bool = False,
    plot_grid: bool = False,
    optimize_on_true_regional_misfit: bool = False,
) -> tuple[optuna.study, pd.DataFrame, optuna.trial.FrozenTrial]:
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
    testing_df : pd.DataFrame
        constraint points to use for calculating the score with columns "easting",
        "northing" and "upward".
    grav_df : pd.DataFrame
        gravity dataframe with columns "easting", "northing", "reg" and
        `gravity_anomaly`.
    trend_limits : tuple[int, int]
        limits to use for the trend order in degrees.
    score_as_median : bool, optional
        use the root median square instead of the root mean square for the scoring
        metric, by default False
    remove_starting_grav_mean : bool, optional
        remove the mean of the starting gravity data before estimating the regional.
        Useful to mitigate effects of poorly-chosen zref value. By default False
    true_regional : xr.DataArray | None, optional
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

    Returns
    -------
    tuple[optuna.study, pd.DataFrame, optuna.trial.FrozenTrial]
        the completed Optuna study, the resulting gravity dataframe of the best trial,
        and the best trial itself.
    """
    optuna.logging.set_verbosity(optuna.logging.WARN)

    # if sampler not provided, use GridSampler as default
    if sampler is None:
        sampler = optuna.samplers.GridSampler(
            search_space={"trend": list(range(trend_limits[0], trend_limits[1] + 1))},
            seed=10,
        )

    if optimize_on_true_regional_misfit is True:
        if true_regional is None:
            msg = (
                "if optimizing on true regional misfit, must provide true_regional grid"
            )
            raise ValueError(msg)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            load_if_exists=False,
        )
    else:
        study = optuna.create_study(
            directions=[
                "minimize",
                "maximize",
            ],
            sampler=sampler,
            load_if_exists=False,
        )

    # run optimization
    study.optimize(
        OptimizeRegionalTrend(
            trend_limits=trend_limits,
            testing_df=testing_df,
            grav_df=grav_df,
            true_regional=true_regional,
            score_as_median=score_as_median,
            optimize_on_true_regional_misfit=optimize_on_true_regional_misfit,
            remove_starting_grav_mean=remove_starting_grav_mean,
        ),
        show_progress_bar=True,
    )

    if optimize_on_true_regional_misfit is True:
        best_trial = study.best_trial
    else:
        best_trial = min(study.best_trials, key=lambda t: t.values[0])
        log.info("Number of trials on the Pareto front: %s", len(study.best_trials))

    log.info("Trial with lowest residual at constraint points: ")
    log.info("\ttrial number: %s", best_trial.number)
    log.info("\tparameter: %s", best_trial.params)
    log.info("\tscores: %s", best_trial.values)

    resulting_grav_df = best_trial.user_attrs.get("results")

    if plot is True:
        if study._is_multi_objective() is False:  # pylint: disable=protected-access
            if optimize_on_true_regional_misfit is True:
                plotting.combined_slice(
                    study,
                    attribute_names=[
                        "residual constraint score",
                        "residual amplitude score",
                    ],
                ).show()
            else:
                optuna.visualization.plot_slice(study).show()
        else:
            p = optuna.visualization.plot_pareto_front(study)
            plotting.remove_df_from_hoverdata(p).show()
            for i, j in enumerate(study.metric_names):
                optuna.visualization.plot_slice(
                    study,
                    target=lambda t: t.values[i],  # noqa: B023 # pylint: disable=cell-var-from-loop
                    target_name=j,
                ).show()
        if plot_grid is True:
            resulting_grav_df.set_index(["northing", "easting"]).to_xarray().reg.plot()

    return study, resulting_grav_df, best_trial


def optimize_regional_eq_sources(
    testing_df: pd.DataFrame,
    grav_df: pd.DataFrame,
    score_as_median: bool = False,
    remove_starting_grav_mean: bool = False,
    true_regional: xr.DataArray | None = None,
    n_trials: int = 100,
    source_depth_limits: tuple[float, float] | None = None,
    source_depth: float | None = None,
    block_size_limits: tuple[float, float] | None = None,
    block_size: float | None = None,
    eq_damping_limits: tuple[float, float] | None = None,
    eq_damping: float | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    plot: bool = False,
    plot_grid: bool = False,
    optimize_on_true_regional_misfit: bool = False,
) -> tuple[optuna.study, pd.DataFrame, optuna.trial.FrozenTrial]:
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
    testing_df : pd.DataFrame
        constraint points to use for calculating the score with columns "easting",
        "northing" and "upward".
    grav_df : pd.DataFrame
        gravity dataframe with columns "easting", "northing", "reg", and
        `gravity_anomaly`.
    score_as_median : bool, optional
        use the root median square instead of the root mean square for the scoring
        metric, by default False
    remove_starting_grav_mean : bool, optional
        remove the mean of the starting gravity data before estimating the regional.
        Useful to mitigate effects of poorly-chosen zref value. By default False
    true_regional : xr.DataArray | None, optional
        if the true regional gravity is known (in synthetic models), supply this as a
        grid to include a user_attr of the RMSE between this and the estimated regional
        for each trial, or set `optimize_on_true_regional_misfit=True` to have the
        optimization optimize on the RMSE, by default None
    n_trials : int, optional
        number of trials to run, by default 100
    source_depth_limits : tuple[float, float] | None, optional
        limits to use for source depths, positive down in meters, by default None
    source_depth : float | None, optional
        if source_depth_limits not supplied, use this value, by default None
    block_size_limits : tuple[float, float] | None, optional
        limits to use for block size in meters, by default None
    block_size : float | None, optional
        if block_size_limits not supplied, use this value, by default None
    eq_damping_limits : tuple[float, float] | None, optional
        limits to use for the damping parameter, by default None
    eq_damping : float | None, optional
        if eq_damping_limits not provided, use this value, by default None
    sampler : optuna.samplers.BaseSampler | None, optional
        customize the optuna sampler, by default TPE sampler
    plot : bool, optional
        plot the resulting optimization figures, by default False
    plot_grid : bool, optional
        plot the resulting regional gravity grid, by default False
    optimize_on_true_regional_misfit : bool, optional
        if true_regional grid is provide, choose to perform optimization on the RMSE
        between the true regional and the estimated region, by default False

    Returns
    -------
    tuple[optuna.study, pd.DataFrame, optuna.trial.FrozenTrial]
        the completed Optuna study, the resulting gravity dataframe of the best trial,
        and the best trial itself.
    """

    optuna.logging.set_verbosity(optuna.logging.WARN)

    # if sampler not provided, use TPE as default
    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(n_trials / 4),
            seed=10,
        )

    if optimize_on_true_regional_misfit is True:
        if true_regional is None:
            msg = (
                "if optimizing on true regional misfit, must provide true_regional grid"
            )
            raise ValueError(msg)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            load_if_exists=False,
        )
    else:
        study = optuna.create_study(
            directions=[
                "minimize",
                "maximize",
            ],
            sampler=sampler,
            load_if_exists=False,
        )
    # run optimization
    study.optimize(
        OptimizeRegionalEqSources(
            source_depth_limits=source_depth_limits,
            block_size_limits=block_size_limits,
            eq_damping_limits=eq_damping_limits,
            testing_df=testing_df,
            grav_df=grav_df,
            true_regional=true_regional,
            score_as_median=score_as_median,
            source_depth=source_depth,
            block_size=block_size,
            eq_damping=eq_damping,
            optimize_on_true_regional_misfit=optimize_on_true_regional_misfit,
            remove_starting_grav_mean=remove_starting_grav_mean,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    if optimize_on_true_regional_misfit is True:
        best_trial = study.best_trial
    else:
        best_trial = min(study.best_trials, key=lambda t: t.values[0])
        log.info("Number of trials on the Pareto front: %s", len(study.best_trials))

    log.info("Trial with lowest residual at constraint points: ")
    log.info("\ttrial number: %s", best_trial.number)
    log.info("\tparameter: %s", best_trial.params)
    log.info("\tscores: %s", best_trial.values)

    resulting_grav_df = best_trial.user_attrs.get("results")

    if plot is True:
        if study._is_multi_objective() is False:  # pylint: disable=protected-access
            if optimize_on_true_regional_misfit is True:
                for p in best_trial.params:
                    plotting.combined_slice(
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
            p = optuna.visualization.plot_pareto_front(study)
            plotting.remove_df_from_hoverdata(p).show()
            for i, j in enumerate(study.metric_names):
                optuna.visualization.plot_slice(
                    study,
                    target=lambda t: t.values[i],  # noqa: B023 # pylint: disable=cell-var-from-loop
                    target_name=j,
                ).show()
        if plot_grid is True:
            resulting_grav_df.set_index(["northing", "easting"]).to_xarray().reg.plot()

    return study, resulting_grav_df, best_trial


def optimize_regional_constraint_point_minimization(
    training_df: pd.DataFrame | list[pd.DataFrame],
    testing_df: pd.DataFrame | list[pd.DataFrame],
    grid_method: str,
    grav_df: pd.DataFrame,
    constraints_weights_column: str | None = None,
    score_as_median: bool = False,
    remove_starting_grav_mean: bool = False,
    true_regional: xr.DataArray | None = None,
    n_trials: int = 100,
    tension_factor_limits: tuple[float, float] = (0, 1),
    spline_damping_limits: tuple[float, float] | None = None,
    source_depth_limits: tuple[float, float] | None = None,
    source_depth: float | None = None,
    block_size_limits: tuple[float, float] | None = None,
    block_size: float | None = None,
    eq_damping_limits: tuple[float, float] | None = None,
    eq_damping: float | None = None,
    grav_obs_height_limits: tuple[float, float] | None = None,
    grav_obs_height: float | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    plot: bool = False,
    plot_grid: bool = False,
    progressbar: bool = True,
    fold_progressbar: bool = False,
    optimize_on_true_regional_misfit: bool = False,
) -> tuple[optuna.study, pd.DataFrame, optuna.trial.FrozenTrial]:
    """
    Run an Optuna optimization to find the optimal hyperparameters for the Constraint
    Point Minimization technique for estimating the regional component of gravity
    misfit. This function can be used for both single and K-Folds cross validation,
    which is determined by the type of supplied training_df and testing_df. For
    synthetic testing, if the true regional grid is provided, the optimization can be
    set to optimize on the RMSE of the predicted and true regional gravity, by setting
    `optimize_on_true_regional_misfit=True`. By default this will perform a
    multi-objective optimization to find the best trade-off between the lowest RMSE of
    the residual at the constraints and the highest RMSE of the residual at all
    locations.

    Parameters
    ----------
    training_df : pd.DataFrame | list[pd.DataFrame]
        constraint points to use for training (estimating the regional field) with
        columns "easting", "northing" and "upward". If a list of dataframe are provided,
        each should represent 1 fold of a K-Folds cross-validation.
    testing_df : pd.DataFrame | list[pd.DataFrame]
        constraint points to use for testing (calculating the score) with columns
        "easting", "northing" and "upward". If a list of dataframe are provided, each
        should represent 1 fold of a K-Folds cross-validation.
    grid_method : str
        constraint point minimization method to use, choose between "verde" for
        bi-harmonic spline gridding, "pygmt" for tensioned minimum curvature gridding,
        or "eq_sources" for equivalent sources gridding.
    grav_df : pd.DataFrame
        gravity dataframe with columns "easting", "northing", "reg", and
        "gravity_anomaly".
    constraints_weights_column : str | None, optional
        column name containing the optional weight values for each constraint point, by
        default None
    score_as_median : bool, optional
        use the root median square instead of the root mean square for the scoring
        metric, by default False
    remove_starting_grav_mean : bool, optional
        remove the mean of the starting gravity data before estimating the regional.
        Useful to mitigate effects of poorly-chosen zref value. By default False
    true_regional : xr.DataArray | None, optional
        if the true regional gravity is known (in synthetic models), supply this as a
        grid to include a user_attr of the RMSE between this and the estimated regional
        for each trial, or set `optimize_on_true_regional_misfit=True` to have the
        optimization optimize on the RMSE, by default None
    n_trials : int, optional
        number of trials to run, by default 100
    tension_factor_limits : tuple[float, float], optional
        limits to use for the PyGMT tension factor gridding, by default (0, 1)
    spline_damping_limits : tuple[float, float] | None, optional
        limits to use for the Verde bi-harmonic spline damping, by default None
    source_depth_limits : tuple[float, float] | None, optional
        limits to use for the equivalent sources' depths, by default None
    source_depth : float | None, optional
        if source_depth_limits are not supplied, use this value, by default None
    block_size_limits : tuple[float, float] | None, optional
        limits to use for the block size for fitting equivalent sources, by default None
    block_size : float | None, optional
        if block_size_limits are not supplied, use this value, by default None
    eq_damping_limits : tuple[float, float] | None, optional
        limits to use for the damping value for fitting equivalent sources, by default
        None
    eq_damping : float | None, optional
        if eq_damping_limits are not provided, use this value, by default None
    grav_obs_height_limits : tuple[float, float] | None, optional
        limits to use for the gravity observation height for fitting equivalent sources,
        by default None
    grav_obs_height : float | None, optional
        if grav_obs_height_limits are not provided, use this value, by default None
    sampler : optuna.samplers.BaseSampler | None, optional
        customize the optuna sampler, by default TPE sampler
    plot : bool, optional
        plot the resulting optimization figures, by default False
    plot_grid : bool, optional
        plot the resulting regional gravity grid, by default False
    progressbar : bool, optional
        show a progressbar for the optimization, by default True
    fold_progressbar : bool, optional
        turn on or off a progress bar for the optimization of each fold if performing
        a K-Folds cross-validation within the optimization, by default False
    optimize_on_true_regional_misfit : bool, optional
        if true_regional grid is provide, choose to perform optimization on the RMSE
        between the true regional and the estimated region, by default False

    Returns
    -------
    tuple[optuna.study, pd.DataFrame, optuna.trial.FrozenTrial]
        the completed Optuna study, the resulting gravity dataframe of the best trial,
        and the best trial itself.
    """

    optuna.logging.set_verbosity(optuna.logging.WARN)

    # if sampler not provided, use TPE as default
    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(n_trials / 4),
            seed=10,
        )

    if optimize_on_true_regional_misfit is True:
        if true_regional is None:
            msg = (
                "if optimizing on true regional misfit, must provide true_regional grid"
            )
            raise ValueError(msg)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            load_if_exists=False,
        )
    else:
        study = optuna.create_study(
            directions=[
                "minimize",
                "maximize",
            ],
            sampler=sampler,
            load_if_exists=False,
        )

    if isinstance(training_df, list):
        msg = (
            "training and testing data supplied as lists of dataframe, using them "
            "for a K-Folds cross validation"
        )
        log.info(msg)

    # run optimization
    study.optimize(
        OptimizeRegionalConstraintsPointMinimization(
            training_df=training_df,
            testing_df=testing_df,
            grid_method=grid_method,
            grav_df=grav_df,
            constraints_weights_column=constraints_weights_column,
            true_regional=true_regional,
            score_as_median=score_as_median,
            tension_factor_limits=tension_factor_limits,
            spline_damping_limits=spline_damping_limits,
            source_depth_limits=source_depth_limits,
            block_size_limits=block_size_limits,
            eq_damping_limits=eq_damping_limits,
            grav_obs_height_limits=grav_obs_height_limits,
            source_depth=source_depth,
            block_size=block_size,
            eq_damping=eq_damping,
            grav_obs_height=grav_obs_height,
            optimize_on_true_regional_misfit=optimize_on_true_regional_misfit,
            remove_starting_grav_mean=remove_starting_grav_mean,
            progressbar=fold_progressbar,
        ),
        n_trials=n_trials,
        show_progress_bar=progressbar,
    )

    if optimize_on_true_regional_misfit is True:
        best_trial = study.best_trial
    else:
        best_trial = min(study.best_trials, key=lambda t: t.values[0])
        log.info("Number of trials on the Pareto front: %s", len(study.best_trials))

    log.info("Trial with lowest residual at constraint points: ")
    log.info("\ttrial number: %s", best_trial.number)
    log.info("\tparameter: %s", best_trial.params)
    log.info("\tscores: %s", best_trial.values)

    resulting_grav_df = best_trial.user_attrs.get("results")

    if plot is True:
        if study._is_multi_objective() is False:  # pylint: disable=protected-access
            if optimize_on_true_regional_misfit is True:
                for p in best_trial.params:
                    plotting.combined_slice(
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
            p = optuna.visualization.plot_pareto_front(study)
            plotting.remove_df_from_hoverdata(p).show()
            for i, j in enumerate(study.metric_names):
                optuna.visualization.plot_slice(
                    study,
                    target=lambda t: t.values[i],  # noqa: B023 # pylint: disable=cell-var-from-loop
                    target_name=j,
                ).show()
        if plot_grid is True:
            resulting_grav_df.set_index(["northing", "easting"]).to_xarray().reg.plot()

    return study, resulting_grav_df, best_trial


def optimize_regional_constraint_point_minimization_kfolds(
    testing_training_df: pd.DataFrame,
    plot: bool = False,
    plot_grid: bool = False,
    fold_progressbar: bool = True,
    **kwargs: typing.Any,
) -> tuple[optuna.study, pd.DataFrame, optuna.trial.FrozenTrial]:
    """
    see below links for ideas of a better method to do this.
    https://stackoverflow.com/questions/63224426/how-can-i-cross-validate-by-pytorch-and-optuna
    https://www.kaggle.com/code/muhammetgamal5/kfold-cross-validation-optuna-tuning

    Parameters
    ----------
    testing_training_df : pd.DataFrame
        constraints dataframe with columns "easting", "northing", "upward", and a column
        for each fold in the format "fold_0", "fold_1", etc. This can be created with
        function `cross_validation.split_test_train()`.
    plot : bool, optional
        plot optimization figures and optional the regional gravity grid, by default
        False
    plot_grid : bool, optional
        plot the resulting regional gravity grid, by default False
    fold_progressbar : bool, optional
        turn on or off a progress bar for the optimization of each fold, by default True

    Returns
    -------
    tuple[optuna.study, pd.DataFrame, optuna.trial.FrozenTrial]
        the completed Optuna study, the resulting gravity dataframe of the best trial,
        and the best trial itself.
    """

    test_dfs, train_dfs = cross_validation.kfold_df_to_lists(testing_training_df)

    kwargs.pop("plot", False)
    kwargs.pop("plot_grid", False)

    study, _, best_trial = (
        optimize_regional_constraint_point_minimization(
            training_df=train_dfs,
            testing_df=test_dfs,
            plot=False,
            plot_grid=False,
            fold_progressbar=fold_progressbar,
            **kwargs,
        )
    )

    # redo regional sep with all data (not 1 fold) to get resulting df
    resulting_grav_df = regional.regional_separation(
        method="constraints",
        grav_df=kwargs.get("grav_df"),
        constraints_df=testing_training_df,
        registration=kwargs.get("registration", "g"),
        constraints_block_size=kwargs.get("constraints_block_size", None),
        eq_points=kwargs.get("eq_points", None),
        constraints_weights_column=kwargs.get("constraints_weights_column", None),
        grid_method=kwargs.get("grid_method"),
        remove_starting_grav_mean=kwargs.get("remove_starting_grav_mean", False),
        # hyperparameters
        tension_factor=best_trial.params.get("tension_factor", None),
        spline_damping=best_trial.params.get("spline_damping", None),
        source_depth=best_trial.params.get(
            "source_depth", kwargs.get("source_depth", "default")
        ),
        eq_damping=best_trial.params.get("eq_damping", kwargs.get("eq_damping", None)),
        block_size=best_trial.params.get("block_size", kwargs.get("block_size", None)),
        grav_obs_height=best_trial.params.get(
            "grav_obs_height", kwargs.get("grav_obs_height", None)
        ),
    )

    if plot is True:
        if study._is_multi_objective() is False:  # pylint: disable=protected-access
            if kwargs.get("optimize_on_true_regional_misfit") is True:
                for p in best_trial.params:
                    plotting.combined_slice(
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
            p = optuna.visualization.plot_pareto_front(study)
            plotting.remove_df_from_hoverdata(p).show()
            for i, j in enumerate(study.metric_names):
                optuna.visualization.plot_slice(
                    study,
                    target=lambda t: t.values[i],  # noqa: B023 # pylint: disable=cell-var-from-loop
                    target_name=j,
                ).show()
        if plot_grid is True:
            resulting_grav_df.set_index(["northing", "easting"]).to_xarray().reg.plot()

    return study, resulting_grav_df, best_trial
