from __future__ import annotations

import logging
import math
import pathlib
import typing
import warnings

import harmonica as hm
import joblib
import optuna
import pandas as pd
import psutil
from nptyping import NDArray
from optuna.storages import JournalFileStorage, JournalStorage
from tqdm_joblib import tqdm_joblib

from invert4geom import utils


def logging_callback(study: typing.Any, frozen_trial: optuna.trial.FrozenTrial) -> None:
    """
    custom optuna callback, only print trial if it's the best value yet.
    """
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        logging.info(
            "Trial %s finished with best value: %s and parameters: %s.",
            frozen_trial.number,
            frozen_trial.value,
            frozen_trial.params,
        )


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
    # get available cores (UNIX and Windows)
    num_cores = len(psutil.Process().cpu_affinity())

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
        storage = JournalStorage(JournalFileStorage(fname))

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
