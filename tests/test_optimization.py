import logging
import pathlib
import tempfile
import typing
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest
import verde as vd
import xarray as xr

import invert4geom
from invert4geom import optimization


def test_run_optuna_parallel():
    """
    test that the optuna parallel optimization works
    Just tests that functions runs, doesn't test that it's properly running in parallel.
    """
    with tempfile.NamedTemporaryFile(delete=False) as file:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=optuna.exceptions.ExperimentalWarning
            )
            lock_obj = optuna.storages.journal.JournalFileOpenLock(file.name)
            file_storage = optuna.storages.journal.JournalFileBackend(
                file.name, lock_obj=lock_obj
            )
            storage = optuna.storages.journal.JournalStorage(file_storage)
            study_name = file.name

        # create study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(n_startup_trials=5),
            direction="minimize",
        )

        # create a dummy objective function
        def objective(trial):
            x = trial.suggest_int("x", 0, 10)
            return (x) ** 2

        # run the optimization
        study = optimization.run_optuna(
            study=study,
            storage=storage,
            objective=objective,
            n_trials=10,
            parallel=True,
            maximize_cpus=True,
        )

        # all trials may run simultaneously, in which case each worker samples
        # randomly (TPE has no completed trials to learn from), so only assert the
        # optimization ran and improved on the worst possible value
        assert len(study.trials) >= 10
        assert study.best_value < 100


def test_run_optuna_series():
    """
    test that the optuna optimization works
    """

    # create study
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),
        direction="minimize",
    )

    # create a dummy objective function
    def objective(trial):
        x = trial.suggest_int("x", 0, 10)
        return (x) ** 2

    # run the optimization
    study = optimization.run_optuna(
        study=study,
        objective=objective,
        n_trials=10,
        parallel=False,
    )

    assert study.best_value < 5


################
################
# run_optuna / _optuna_set_cores
################
################


def test_run_optuna_parallel_without_storage_raises():
    study = optuna.create_study()
    with pytest.raises(ValueError, match="must provide an Optuna storage"):
        optimization.run_optuna(
            study=study,
            objective=lambda _trial: 0.0,
            n_trials=2,
            parallel=True,
        )


def test_optuna_set_cores_dispatches_one_task_per_job(monkeypatch):
    """
    regression test: one task was dispatched per *trial* instead of per job,
    over-running the requested trial budget by a factor of trials-per-job
    """
    monkeypatch.setattr(optimization, "available_cpu_count", lambda: 4)

    captured = {}

    class FakeParallel:
        def __init__(self, n_jobs):
            captured["n_jobs"] = n_jobs

        def __call__(self, iterable):
            captured["tasks"] = list(iterable)

    monkeypatch.setattr(joblib, "Parallel", FakeParallel)

    optimization._optuna_set_cores(
        n_trials=10,
        optimize_study=lambda *_args, **_kwargs: None,
        study_name="test",
        storage=None,
        objective=lambda _trial: 0.0,
        max_cores=True,
    )

    # 4 cores and 10 trials -> 4 jobs with ceil(10/4)=3 trials each
    assert captured["n_jobs"] == 4
    assert len(captured["tasks"]) == 4
    # joblib.delayed returns (function, args, kwargs) tuples
    assert all(task[2]["n_trials"] == 3 for task in captured["tasks"])


def test_available_cpu_count_positive():
    assert optimization.available_cpu_count() >= 1


################
################
# warn_parameter_at_limits
################
################


def test_warn_parameter_at_limits_warns(caplog):
    study = optuna.create_study()
    study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=5)
    # build a trial whose parameter value is at the lower limit
    trial = optuna.trial.create_trial(
        params={"x": 0.0},
        distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
        value=0.0,
    )
    with caplog.at_level(logging.WARNING, logger="invert4geom"):
        optimization.warn_parameter_at_limits(trial)
    assert "at the limit" in caplog.text


def test_warn_parameter_at_limits_no_limits_distribution():
    """
    regression test: a parameter with a distribution without high/low limits
    (e.g. categorical) used to raise an AttributeError
    """
    trial = optuna.trial.create_trial(
        params={"x": "a"},
        distributions={"x": optuna.distributions.CategoricalDistribution(["a", "b"])},
        value=0.0,
    )
    optimization.warn_parameter_at_limits(trial)  # should not raise


################
################
# _get_best_trial
################
################


def test_get_best_trial_single_objective():
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: trial.suggest_float("x", 0, 10) ** 2, n_trials=5)
    best = optimization._get_best_trial(study)
    assert best.number == study.best_trial.number


def test_get_best_trial_multi_objective():
    """for multi-objective studies, pick the Pareto trial with lowest first value"""
    study = optuna.create_study(directions=["minimize", "maximize"])

    def objective(trial):
        x = trial.suggest_float("x", 0, 10)
        return x, x

    study.optimize(objective, n_trials=5)
    best = optimization._get_best_trial(study)
    pareto_first_values = [t.values[0] for t in study.best_trials]  # noqa: PD011
    assert best.values[0] == min(pareto_first_values)  # noqa: PD011


################
################
# _report_regional_scores
################
################


def make_trial() -> optuna.trial.Trial:
    return optuna.create_study().ask()


def test_report_regional_scores_separate_metrics():
    trial = make_trial()
    result = optimization._report_regional_scores(
        trial,
        residual_constraint_score=2.0,
        residual_amplitude_score=4.0,
        true_reg_score=1.5,
        optimize_on_true_regional_misfit=False,
        separate_metrics=True,
    )
    assert result == (2.0, 4.0)
    assert trial.user_attrs["true_reg_score"] == 1.5


def test_report_regional_scores_combined_metric():
    trial = make_trial()
    result = optimization._report_regional_scores(
        trial,
        residual_constraint_score=2.0,
        residual_amplitude_score=4.0,
        true_reg_score=None,
        optimize_on_true_regional_misfit=False,
        separate_metrics=False,
    )
    assert result == pytest.approx(0.5)


def test_report_regional_scores_true_regional():
    trial = make_trial()
    result = optimization._report_regional_scores(
        trial,
        residual_constraint_score=2.0,
        residual_amplitude_score=4.0,
        true_reg_score=1.5,
        optimize_on_true_regional_misfit=True,
        separate_metrics=True,
    )
    assert result == 1.5
    assert trial.user_attrs["residual constraint score"] == 2.0
    assert trial.user_attrs["residual amplitude score"] == 4.0


################
################
# _create_regional_separation_study
################
################


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_create_regional_separation_study_serial_uses_no_storage(tmp_path):
    """
    regression test: file storage used to be created even for serial runs,
    slowing the optimization and leaving journal files behind
    """
    study, storage = optimization._create_regional_separation_study(
        optimize_on_true_regional_misfit=False,
        separate_metrics=True,
        sampler=optuna.samplers.RandomSampler(seed=0),
        parallel=False,
        fname=str(tmp_path / "study"),
    )
    assert storage is None
    assert not (tmp_path / "study.log").exists()
    assert list(study.directions) == [
        optuna.study.StudyDirection.MINIMIZE,
        optuna.study.StudyDirection.MAXIMIZE,
    ]


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_create_regional_separation_study_parallel_uses_storage(tmp_path):
    _, storage = optimization._create_regional_separation_study(
        optimize_on_true_regional_misfit=False,
        separate_metrics=True,
        sampler=optuna.samplers.RandomSampler(seed=0),
        parallel=True,
        fname=str(tmp_path / "study"),
    )
    assert storage is not None
    assert (tmp_path / "study.log").exists()


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_create_regional_separation_study_true_regional_needs_grid(tmp_path):
    with pytest.raises(ValueError, match="must provide true_regional"):
        optimization._create_regional_separation_study(
            optimize_on_true_regional_misfit=True,
            separate_metrics=True,
            sampler=optuna.samplers.RandomSampler(seed=0),
            parallel=False,
            fname=str(tmp_path / "study"),
        )


################
################
# DuplicateIterationPruner
################
################


def test_duplicate_iteration_pruner_prunes_repeated_params():
    study = optuna.create_study(pruner=optimization.DuplicateIterationPruner())

    def objective(trial):
        trial.suggest_int("x", 0, 0)  # always the same value
        if trial.should_prune():
            raise optuna.TrialPruned
        return 0.0

    study.optimize(objective, n_trials=3)

    states = [trial.state for trial in study.trials]
    assert states[0] == optuna.trial.TrialState.COMPLETE
    assert all(state == optuna.trial.TrialState.PRUNED for state in states[1:])


################
################
# optimize_eq_source_params
################
################


def eq_sources_inputs() -> tuple[tuple[typing.Any, ...], typing.Any]:
    rng = np.random.default_rng(seed=0)
    easting = rng.uniform(0, 10000, 30)
    northing = rng.uniform(0, 10000, 30)
    upward = np.full_like(easting, 1000)
    data = 1e-7 * (easting**2 + northing**2)
    return (easting, northing, upward), data


def test_optimize_eq_source_params_no_limits_raises():
    coordinates, data = eq_sources_inputs()
    with pytest.raises(ValueError, match="No parameters to optimize"):
        optimization.optimize_eq_source_params(coordinates, data, n_trials=2)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_optimize_eq_source_params_returns_fitted_sources(tmp_path):
    coordinates, data = eq_sources_inputs()
    fname = str(tmp_path / "eqs_test")
    study, eqs = optimization.optimize_eq_source_params(
        coordinates,
        data,
        n_trials=5,
        damping_limits=(1e-3, 1),
        depth=5000,
        fname=fname,
        progressbar=False,
    )
    assert len(study.trials) == 5
    # the fitted model should use the best damping and the fixed depth kwarg
    assert eqs.damping == study.best_trial.params["damping"]
    assert eqs.depth == 5000
    assert hasattr(eqs, "coefs_")
    # the study should be saved to a pickle file
    assert pathlib.Path(f"{fname}.pickle").exists()


################
################
# optimize_regional_trend (end-to-end)
################
################


def observed_gravity() -> xr.Dataset:
    easting = [0.0, 10000.0, 20000.0, 30000.0, 40000.0]
    northing = [0.0, 10000.0, 20000.0, 30000.0]
    x, y = np.meshgrid(easting, northing)
    grav = (y**2 + x**2) / 1e7
    ds = vd.make_xarray_grid(
        (easting, northing),
        data=(grav, np.full_like(grav, 1000), np.full_like(grav, 100)),
        data_names=("gravity_anomaly", "upward", "forward_gravity"),
    )
    return invert4geom.inversion.create_data(ds)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_optimize_regional_trend_serial_leaves_no_journal_files(tmp_path):
    """
    regression test: serial regional optimizations used to always create
    journal storage files
    """
    grav_data = observed_gravity()
    constraints = pd.DataFrame(
        {
            "easting": [10000.0, 20000.0, 30000.0],
            "northing": [10000.0, 30000.0, 20000.0],
            "upward": [500.0, 500.0, 500.0],
        }
    )
    fname = str(tmp_path / "trend_test")
    study, result_ds, best_trial = optimization.optimize_regional_trend(
        testing_df=constraints,
        grav_ds=grav_data,
        trend_limits=(0, 1),
        plot=False,
        progressbar=False,
        fname=fname,
    )
    assert not (tmp_path / "trend_test.log").exists()
    assert best_trial.params["trend"] in (0, 1)
    assert "reg" in result_ds
    assert "res" in result_ds
    assert len(study.trials) == 2
