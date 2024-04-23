from __future__ import annotations

import logging
import math
import multiprocessing
import os
import pathlib
import re
import subprocess
import typing
import warnings

import harmonica as hm
import pandas as pd
from nptyping import NDArray

from invert4geom import utils

try:
    import optuna
except ImportError:
    optuna = None

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


def logging_callback(study: typing.Any, frozen_trial: optuna.trial.FrozenTrial) -> None:
    """
    custom optuna callback, only print trial if it's the best value yet.
    """
    if optuna is None:
        msg = "Missing optional dependency 'optuna' required for optimization."
        raise ImportError(msg)

    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        logging.info(
            "Trial %s finished with best value: %s and parameters: %s.",
            frozen_trial.number,
            frozen_trial.value,
            frozen_trial.params,
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
    study_storage: typing.Any,
    objective: typing.Callable[..., float],
    n_trials: int = 100,
    maximize_cpus: bool = True,
    parallel: bool = True,
) -> tuple[typing.Any, pd.DataFrame]:
    """
    Run optuna optimization in parallel. Pre-define the study, storage, and objective
    function and input them here.
    """
    if optuna is None:
        msg = "Missing optional dependency 'optuna' required for optimization."
        raise ImportError(msg)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # load study metadata from storage
    study = optuna.load_study(storage=study_storage, study_name=study_name)

    # set up parallel processing and run optimization
    if parallel is True:
        # @utils.supress_stdout
        def optimize_study(
            study_name: str,
            storage: typing.Any,
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
    study_storage: typing.Any,
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
    study_storage: typing.Any,
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


class OptimalEqSourceParams:
    """
    a class for finding the optimal depth and damping parameters to best fit a set of
    equivalent sources to the gravity data.
    """

    def __init__(
        self,
        coordinates: tuple[
            pd.Series | NDArray, pd.Series | NDArray, pd.Series | NDArray
        ],
        data: pd.Series | NDArray,
        damping_limits: tuple[float, float],
        depth_limits: tuple[float, float],
        **kwargs: typing.Any,
    ) -> None:
        """
        Parameters
        ----------
        coordinates : tuple[ pd.Series  |  NDArray, pd.Series  |  NDArray,
        pd.Series  |  NDArray ]
            easting, northing, and upward coordinates of the data
        data : pd.Series | NDArray
            gravity values
        damping_limits : tuple[float, float]
            lower and upper bounds for the damping parameter
        depth_limits : tuple[float, float]
            lower and upper bounds for the depth of the sources
        """
        self.coordinates = coordinates
        self.data = data
        self.damping_limits = damping_limits
        self.depth_limits = depth_limits
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
        if optuna is None:
            msg = "Missing optional dependency 'optuna' required for optimization."
            raise ImportError(msg)

        # define parameter space
        damping = trial.suggest_float(
            "damping",
            self.damping_limits[0],
            self.damping_limits[1],
        )
        depth = trial.suggest_float(
            "depth",
            self.depth_limits[0],
            self.depth_limits[1],
        )

        return utils.eq_sources_score(
            params={"damping": damping, "depth": depth},
            coordinates=self.coordinates,
            data=self.data,
            **self.kwargs,
        )


def optimize_eq_source_params(
    coordinates: tuple[pd.Series | NDArray, pd.Series | NDArray, pd.Series | NDArray],
    data: pd.Series | NDArray,
    n_trials: int = 0,
    damping_limits: tuple[float, float] = (0, 10**3),
    depth_limits: tuple[float, float] = (0, 10e6),
    sampler: optuna.samplers.BaseSampler | None = None,
    parallel: bool = False,
    fname: str = "tmp",
    use_existing: bool = False,
    # plot:bool=False,
    **eq_kwargs: typing.Any,
) -> tuple[pd.DataFrame, hm.EquivalentSources]:
    """
    find the best parameter values for fitting equivalent sources to a set of gravity
    data.

    Parameters
    ----------
    coordinates : tuple[pd.Series  |  NDArray, pd.Series  |  NDArray,
        pd.Series  |  NDArray]
       easting, northing, and upwards coordinates of gravity data
    data : pd.Series | NDArray
        gravity data values
    n_trials : int, optional
        number of trials to perform / set of parameters to test, by default 0
    damping_limits : tuple[float, float], optional
        lower and upper bounds of damping parameter, by default (0, 10**3)
    depth_limits : tuple[float, float], optional
        lower and upper bounds of depth parameter, by default (0, 10e6)
    sampler : optuna.samplers.BaseSampler | None, optional
        type of sampler to use, by default None
    parallel : bool, optional
        if True, will run the trials in parallel, by default False
    fname : str, optional
        path and filename to save the study results, by default "tmp"
    use_existing : bool, optional
        if True, will continue a previously starting optimization, by default False

    Returns
    -------
    tuple[pd.DataFrame, hm.EquivalentSources]
        gives a dataframe of the tested parameter sets and associated scores, and the
        best resulting fitted equivalent sources.
    """
    if optuna is None:
        msg = "Missing optional dependency 'optuna' required for optimization."
        raise ImportError(msg)

    # set name and storage for the optimization
    study_name = fname
    fname = f"{study_name}.log"

    # remove if exists
    if use_existing is True:
        pass
    else:
        pathlib.Path(fname).unlink(missing_ok=True)
        pathlib.Path(f"{fname}.lock").unlink(missing_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="JournalStorage is experimental")
        lock_obj = optuna.storages.JournalFileOpenLock(fname)
        file_storage = optuna.storages.JournalFileStorage(fname, lock_obj=lock_obj)
        storage = optuna.storages.JournalStorage(file_storage)

        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(fname)
        )

    # if sampler not provided, used BoTorch as default
    if sampler is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="BoTorch")
            sampler = (
                optuna.integration.BoTorchSampler(n_startup_trials=int(n_trials / 3)),
            )
            # sampler=optuna.samplers.TPESampler(n_startup_trials=int(n_trials/3)),
            # sampler=optuna.samplers.GridSampler(search_space),

    # create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    # define the objective function
    objective = OptimalEqSourceParams(
        coordinates=coordinates,
        data=data,
        damping_limits=damping_limits,
        depth_limits=depth_limits,
        parallel=True,
        **eq_kwargs,
    )

    if n_trials == 0:  # (use_existing is True) & (n_trials is None):
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
        study_df = study.trials_dataframe()
    else:
        # run the optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # with HiddenPrints():
            study, study_df = optuna_parallel(
                study_name=study_name,
                study_storage=storage,
                objective=objective,
                n_trials=n_trials,
                maximize_cpus=True,
                parallel=parallel,
            )

    logging.info("Best params: %s", study.best_params)
    logging.info("Best trial: %s", study.best_trial.number)
    logging.info("Best score: %s", study.best_trial.value)

    eqs = hm.EquivalentSources(
        damping=study.best_params.get("damping"),
        depth=study.best_params.get("depth"),
        **eq_kwargs,
    ).fit(coordinates, data, weights=eq_kwargs.get("weights"))

    # if plot is True:
    #     plotting.plot_optuna_inversion_figures(
    #         study,
    #         target_names=["score"],
    #         # include_duration=True,
    #     )

    return study_df.sort_values("value", ascending=False), eqs
