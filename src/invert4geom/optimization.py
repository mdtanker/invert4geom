from __future__ import annotations

import logging
import math
import multiprocessing
import os
import pathlib
import random
import re
import subprocess
import typing
import warnings

import harmonica as hm
import optuna
import optuna
import pandas as pd
from nptyping import NDArray

from invert4geom import plotting, utils


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
        # @utils.supress_stdout
        def optimize_study(
            study_name: str,
            storage: typing.Any,
            objective: typing.Callable[..., float],
            n_trials: int,
        ) -> None:
            storage: optuna.storages.BaseStorage,
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
    study_storage: optuna.storages.BaseStorage,
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

        return cross_validation.grav_cv_score(
            solver_damping=damping,
            progressbar=False,
            results_fname=trial.user_attrs.get("fname"),
            plot=self.plot_grids,
            **new_kwargs,
        )


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
                logging.warning(msg)
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
        logging.warning(
            "Best damping value (%s) is at the limit of provided values "
            "(%s) and thus is likely not a global minimum, expand the range of "
            "values tested to ensure the best parameter value is found.",
            best_trial.params.get("damping"),
            damping_limits,
        )

    logging.info("Trial with lowest score: ")
    logging.info("\ttrial number: %s", best_trial.number)
    logging.info("\tparameter: %s", best_trial.params)
    logging.info("\tscores: %s", best_trial.values)

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
        constraints_df: pd.DataFrame,
        zref: float | None = None,
        zref_limits: tuple[float, float] | None = None,
        density_contrast_limits: tuple[float, float] | None = None,
        density_contrast: float | None = None,
        starting_topography: xr.DataArray | None = None,
        starting_topography_kwargs: dict[str, typing.Any] | None = None,
        regional_grav_kwargs: dict[str, typing.Any] | None = None,
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
            logging.warning(msg)
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
                "field estimation, or set separate constraints in training and testing "
                "sets and provide the training set to `regional_grav_kwargs` and the "
                "testing set to `constraints_df` to use for scoring."
            )
            logging.warning(msg)

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
        grav_df["starting_grav"] = starting_prisms.prism_layer.gravity(
            coordinates=(
                grav_df.easting,
                grav_df.northing,
                grav_df.upward,
            ),
            field="g_z",
            progressbar=False,
        )

        # calculate misfit as observed - starting
        grav_data_column = kwargs.get("grav_data_column")
        grav_df["misfit"] = grav_df[grav_data_column] - grav_df.starting_grav

        # calculate regional field
        reg_kwargs = self.regional_grav_kwargs.copy()  # type: ignore[union-attr]

        grav_df = regional.regional_separation(
            method=reg_kwargs.pop("regional_method", None),
            grav_df=grav_df,
            regional_column="reg",
            grav_data_column="misfit",
            **reg_kwargs,
        )

        # remove the regional from the misfit to get the residual
        grav_df["res"] = grav_df.misfit - grav_df.reg

        # update starting model in kwargs
        kwargs["prism_layer"] = starting_prisms

        new_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            not in [
                "zref",
                "density_contrast",
                "progressbar",
                "results_fname",
            ]
        }

        trial.set_user_attr("fname", f"{self.fname}_trial_{trial.number}")

        # run cross validation
        return cross_validation.constraints_cv_score(
            grav_df=grav_df,
            constraints_df=self.constraints_df,
            results_fname=trial.user_attrs.get("fname"),
            **new_kwargs,
        )


def optimize_inversion_zref_density_contrast(
    grav_df: pd.DataFrame,
    constraints_df: pd.DataFrame,
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
        gravity data frame with columns `easting`, `northing`, `upward`, and a gravity
        column defined by kwarg `grav_data_column`
    constraints_df : pd.DataFrame
        constraints data frame with columns `easting`, `northing`, and `upward`.
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
                    logging.warning(msg)
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
                    logging.warning(msg)
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
                    logging.warning(msg)
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
                    logging.warning(msg, old_n_trials, n_trials)

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
                **kwargs,
            ),
            n_trials=n_trials,
            callbacks=[warn_limits_better_than_trial_multi_params],
            show_progress_bar=True,
        )

    best_trial = study.best_trial

    if zref_limits is not None:  # noqa: SIM102
        if best_trial.params.get("zref") in zref_limits:
            logging.warning(
                "Best zref value (%s) is at the limit of provided values (%s) and "
                "thus is likely not a global minimum, expand the range of values "
                "tested to ensure the best parameter value is found.",
                best_trial.params.get("zref"),
                zref_limits,
            )
    if density_contrast_limits is not None:  # noqa: SIM102
        if best_trial.params.get("density_contrast") in density_contrast_limits:
            logging.warning(
                "Best density contrast value (%s) is at the limit of provided values "
                "(%s) and thus is likely not a global minimum, expand the range of "
                "values tested to ensure the best parameter value is found.",
                best_trial.params.get("density_contrast"),
                density_contrast_limits,
            )
    logging.info("Trial with lowest score: ")
    logging.info("\ttrial number: %s", best_trial.number)
    logging.info("\tparameter: %s", best_trial.params)
    logging.info("\tscores: %s", best_trial.values)

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
                # plotting.plot_cv_scores(
                #     study.trials_dataframe().value.values,
                #     study.trials_dataframe().params_density_contrast.values,  # type: ignore[arg-type] # noqa: E501
                #     param_name="Density contrast (kg/m$^3$)",
                #     plot_title="Density contrast Cross-validation",
                #     logx=logx,
                #     logy=logy,
                # )
                # plotting.plot_cv_scores(
                #     study.trials_dataframe().value.values,
                #     study.trials_dataframe().params_zref.values,
                #     param_name="Reference level (m)",
                #     plot_title="Reference level Cross-validation",
                #     logx=logx,
                #     logy=logy,
                # )

    return study, inv_results

    logging.info("Best params: %s", study.best_params)
    logging.info("Best trial: %s", study.best_trial.number)
    logging.info("Best score: %s", study.best_trial.value)

    if study.best_params.get("damping") in [damping_limits[0], damping_limits[1]]:
        logging.warning(
            "Best damping value (%s) is at the limit of provided "
            "values (%s, %s) and thus is likely not a global minimum, expand the "
            "range "
            "of values tested to ensure the best parameter value is found.",
            study.best_params.get("damping"),
            damping_limits[0],
            damping_limits[1],
        )

    eqs = hm.EquivalentSources(
        damping=study.best_params.get("damping"),
        depth=study.best_params.get("depth"),
        **eq_kwargs,
    ).fit(coordinates, data, weights=eq_kwargs.get("weights"))

    if plot is True:
        plotting.plot_optuna_inversion_figures(
            study,
            target_names=["score"],
            plot_history=False,
            plot_slice=True,
            # include_duration=True,
        )

    return study_df.sort_values("value", ascending=False), eqs
