from __future__ import annotations

import tempfile
import warnings

import optuna

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
            lock_obj = optuna.storages.JournalFileOpenLock(file.name)
            file_storage = optuna.storages.JournalFileStorage(
                file.name, lock_obj=lock_obj
            )
            storage = optuna.storages.JournalStorage(file_storage)
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

        assert study.best_value < 5


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
