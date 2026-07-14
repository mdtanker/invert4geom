import typing

import numpy as np
import pandas as pd
import pytest
import verde as vd
import xarray as xr

import invert4geom


def test_remove_test_points():
    """
    test the remove_test_points function
    """
    # create 6x6 topography grid
    easting = [0, 10000, 20000, 30000, 40000]
    northing = [0, 10000, 20000, 30000]
    grav_vals = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    grav = vd.make_xarray_grid(
        (easting, northing),
        data=(grav_vals, np.full_like(grav_vals, 1000)),
        data_names=("gravity_anomaly", "upward"),
    )

    data = invert4geom.inversion.create_data(grav)

    resampled = invert4geom.cross_validation.add_test_points(data)

    removed = invert4geom.cross_validation.remove_test_points(resampled)

    pd.testing.assert_frame_equal(
        removed.inv.df,
        data.inv.df,
    )


def test_add_test_points():
    """
    test the add_test_points function
    """
    # create 6x6 topography grid
    easting = [0, 10000, 20000, 30000, 40000]
    northing = [0, 10000, 20000, 30000]
    grav_vals = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    grav = vd.make_xarray_grid(
        (easting, northing),
        data=(grav_vals, np.full_like(grav_vals, 1000)),
        data_names=("gravity_anomaly", "upward"),
    )

    data = invert4geom.inversion.create_data(grav)

    resampled = invert4geom.cross_validation.add_test_points(data)

    pd.testing.assert_frame_equal(
        resampled.inv.df[resampled.inv.df.test == False]  # noqa: E712
        .drop(columns="test")
        .reset_index(drop=True),
        data.inv.df.reset_index(drop=True),
    )


################
################
# split_test_train
################
################


@pytest.fixture(name="scattered_points")
def fixture_scattered_points() -> pd.DataFrame:
    """30 randomly scattered points with an extra data column"""
    rng = np.random.default_rng(seed=0)
    return pd.DataFrame(
        {
            "easting": rng.uniform(0, 10000, 30),
            "northing": rng.uniform(0, 10000, 30),
            "upward": rng.uniform(0, 100, 30),
        }
    )


def test_split_test_train_kfold_makes_fold_columns(scattered_points):
    df = invert4geom.cross_validation.split_test_train(
        scattered_points,
        method="KFold",
        n_splits=3,
    )
    folds = [c for c in df.columns if c.startswith("fold_")]
    assert len(folds) == 3
    for fold in folds:
        assert set(df[fold].unique()) == {"train", "test"}


def test_split_test_train_kfold_each_point_tested_once(scattered_points):
    df = invert4geom.cross_validation.split_test_train(
        scattered_points,
        method="KFold",
        n_splits=3,
    )
    folds = df[[c for c in df.columns if c.startswith("fold_")]]
    times_tested = (folds == "test").sum(axis=1)
    assert (times_tested == 1).all()


def test_split_test_train_spacing_uses_blocked_kfold(scattered_points, monkeypatch):
    """
    regression test: providing `spacing` used to silently fall back to a
    non-blocked KFold due to an operator precedence bug
    """
    used = {}
    original = vd.BlockKFold

    def spy(*args, **kwargs):
        used["blocked"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(vd, "BlockKFold", spy)
    invert4geom.cross_validation.split_test_train(
        scattered_points,
        method="KFold",
        spacing=5000,
        n_splits=3,
    )
    assert used.get("blocked") is True


def test_split_test_train_leave_one_out(scattered_points):
    points = scattered_points.head(5)
    df = invert4geom.cross_validation.split_test_train(
        points,
        method="LeaveOneOut",
    )
    folds = [c for c in df.columns if c.startswith("fold_")]
    assert len(folds) == len(points)
    for fold in folds:
        assert (df[fold] == "test").sum() == 1


def test_split_test_train_invalid_method_raises(scattered_points):
    with pytest.raises(ValueError, match="invalid string for `method`"):
        invert4geom.cross_validation.split_test_train(
            scattered_points,
            method="not_a_method",
        )


def test_split_test_train_does_not_alter_user_columns(scattered_points):
    """
    regression test: a dataframe-wide string replacement used to rewrite user
    data containing the placeholder string '_test'
    """
    points = scattered_points.copy()
    points["label"] = "_test"
    df = invert4geom.cross_validation.split_test_train(
        points,
        method="KFold",
        n_splits=2,
    )
    assert (df.label == "_test").all()


def test_split_test_train_n_splits_reduced_to_data_size():
    """n_splits larger than the number of points should be reduced"""
    points = pd.DataFrame({"easting": [0.0, 1.0], "northing": [0.0, 1.0]})
    df = invert4geom.cross_validation.split_test_train(
        points,
        method="KFold",
        n_splits=5,
    )
    folds = [c for c in df.columns if c.startswith("fold_")]
    assert len(folds) == 2


def test_split_test_train_single_point_raises():
    points = pd.DataFrame({"easting": [0.0], "northing": [0.0]})
    with pytest.raises(ValueError, match="n_splits must be greater than 1"):
        invert4geom.cross_validation.split_test_train(
            points,
            method="KFold",
        )


################
################
# kfold_df_to_lists
################
################


def test_kfold_df_to_lists_partitions_each_fold(scattered_points):
    df = invert4geom.cross_validation.split_test_train(
        scattered_points,
        method="KFold",
        n_splits=3,
    )
    test_dfs, train_dfs = invert4geom.cross_validation.kfold_df_to_lists(df)
    assert len(test_dfs) == len(train_dfs) == 3
    for test_df, train_df in zip(test_dfs, train_dfs, strict=True):
        assert len(test_df) + len(train_df) == len(scattered_points)


################
################
# random_split_test_train
################
################


def test_random_split_test_train_proportions(scattered_points):
    df = invert4geom.cross_validation.random_split_test_train(
        scattered_points,
        test_size=0.3,
    )
    assert len(df) == len(scattered_points)
    assert df.test.sum() == 9  # 30% of 30 points


################
################
# eq_sources_score
################
################


def eq_sources_inputs() -> tuple[tuple[typing.Any, ...], typing.Any]:
    rng = np.random.default_rng(seed=0)
    easting = rng.uniform(0, 10000, 30)
    northing = rng.uniform(0, 10000, 30)
    upward = np.full_like(easting, 1000)
    data = 1e-7 * (easting**2 + northing**2)
    return (easting, northing, upward), data


def test_eq_sources_score_returns_finite_score():
    coordinates, data = eq_sources_inputs()
    score = invert4geom.cross_validation.eq_sources_score(
        coordinates,
        data,
        depth=5000,
    )
    assert np.isfinite(score)
    assert score <= 1  # R^2 score


def test_eq_sources_score_nan_data_raises():
    coordinates, data = eq_sources_inputs()
    data[0] = np.nan
    with pytest.raises(ValueError, match="data contains NaN"):
        invert4geom.cross_validation.eq_sources_score(coordinates, data)


def test_eq_sources_score_nan_coordinates_raises():
    coordinates, data = eq_sources_inputs()
    coordinates[0][0] = np.nan
    with pytest.raises(ValueError, match="coordinates contain NaN"):
        invert4geom.cross_validation.eq_sources_score(coordinates, data)


################
################
# regional_separation_score
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


def test_regional_separation_score_trend():
    grav_data = observed_gravity()
    constraints = pd.DataFrame(
        {
            "easting": [10000.0, 20000.0],
            "northing": [10000.0, 30000.0],
            "upward": [500.0, 500.0],
        }
    )
    (
        residual_constraint_score,
        residual_amplitude_score,
        true_reg_score,
        ds,
    ) = invert4geom.cross_validation.regional_separation_score(
        grav_data,
        constraints,
        method="trend",
        trend=1,
    )
    assert np.isfinite(residual_constraint_score)
    assert np.isfinite(residual_amplitude_score)
    assert true_reg_score is None
    assert "reg" in ds
    assert "res" in ds


def test_regional_separation_score_constraints_cv_raises():
    grav_data = observed_gravity()
    constraints = pd.DataFrame(
        {
            "easting": [10000.0],
            "northing": [10000.0],
            "upward": [500.0],
        }
    )
    with pytest.raises(ValueError, match="should not be used here"):
        invert4geom.cross_validation.regional_separation_score(
            grav_data,
            constraints,
            method="constraints_cv",
        )
