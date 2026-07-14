import numpy as np
import pandas as pd
import pytest
import verde as vd
import xarray as xr

import invert4geom
from invert4geom import uncertainty


def test_create_lhc():
    """
    test the create_lhc function
    """

    # run the function
    lhc = uncertainty.create_lhc(
        n_samples=3,
        parameter_dict={
            "param1": {"distribution": "uniform", "loc": 0, "scale": 1},
        },
    )

    expected_values = np.array([0.16666667, 0.5, 0.83333333])

    assert lhc["param1"]["distribution"] == "uniform"
    assert lhc["param1"]["loc"] == 0
    assert lhc["param1"]["scale"] == 1
    np.testing.assert_allclose(
        np.sort(lhc["param1"]["sampled_values"]), expected_values
    )


################
################
# create_lhc
################
################


def test_create_lhc_uniform_within_bounds():
    lhc = uncertainty.create_lhc(
        n_samples=10,
        parameter_dict={
            "param1": {"distribution": "uniform", "loc": 5, "scale": 10},
        },
    )
    values = lhc["param1"]["sampled_values"]
    assert len(values) == 10
    assert values.min() >= 5
    assert values.max() <= 15


def test_create_lhc_log_scaling():
    """with log=True, loc and scale are base-10 exponents"""
    lhc = uncertainty.create_lhc(
        n_samples=10,
        parameter_dict={
            "param1": {"distribution": "uniform", "loc": -4, "scale": 6, "log": True},
        },
    )
    values = lhc["param1"]["sampled_values"]
    assert values.min() >= 1e-4
    assert values.max() <= 1e2


def test_create_lhc_int_dtype():
    lhc = uncertainty.create_lhc(
        n_samples=5,
        parameter_dict={
            "param1": {
                "distribution": "uniform",
                "loc": 0,
                "scale": 100,
                "dtype": int,
            },
        },
    )
    assert lhc["param1"]["sampled_values"].dtype.kind == "i"


def test_create_lhc_invalid_distribution_raises():
    with pytest.raises(ValueError, match="Unknown distribution type: not_a_dist"):
        uncertainty.create_lhc(
            n_samples=3,
            parameter_dict={
                "param1": {"distribution": "not_a_dist", "loc": 0, "scale": 1},
            },
        )


def test_create_lhc_invalid_criterion_raises():
    with pytest.raises(ValueError, match="Unknown criterion type: not_a_criterion"):
        uncertainty.create_lhc(
            n_samples=3,
            parameter_dict={
                "param1": {"distribution": "uniform", "loc": 0, "scale": 1},
            },
            criterion="not_a_criterion",
        )


def test_create_lhc_does_not_alter_input():
    parameter_dict = {"param1": {"distribution": "uniform", "loc": 0, "scale": 1}}
    uncertainty.create_lhc(n_samples=3, parameter_dict=parameter_dict)
    assert "sampled_values" not in parameter_dict["param1"]


################
################
# randomly_sample_data
################
################


def test_randomly_sample_data_reproducible():
    df = pd.DataFrame({"upward": [0.0, 100.0, 200.0], "uncert": [1.0, 2.0, 3.0]})
    first = uncertainty.randomly_sample_data(5, df, "upward", "uncert")
    second = uncertainty.randomly_sample_data(5, df, "upward", "uncert")
    pd.testing.assert_frame_equal(first, second)


def test_randomly_sample_data_does_not_alter_input():
    df = pd.DataFrame({"upward": [0.0, 100.0], "uncert": [1.0, 2.0]})
    uncertainty.randomly_sample_data(0, df, "upward", "uncert")
    assert df.upward.tolist() == [0.0, 100.0]


def test_randomly_sample_data_statistics():
    """with many samples of one point, mean and std should match the inputs"""
    df = pd.DataFrame({"upward": [100.0] * 5000, "uncert": [2.0] * 5000})
    sampled = uncertainty.randomly_sample_data(0, df, "upward", "uncert")
    assert sampled.upward.mean() == pytest.approx(100.0, abs=0.2)
    assert sampled.upward.std() == pytest.approx(2.0, abs=0.2)


################
################
# merge_simulation_results / model_ensemble_stats
################
################


def ensemble_grids() -> list[xr.DataArray]:
    easting = np.arange(0, 4000, 1000.0)
    northing = np.arange(0, 3000, 1000.0)
    return [
        xr.DataArray(
            np.full((len(northing), len(easting)), value),
            coords={"northing": northing, "easting": easting},
            dims=("northing", "easting"),
            name="topo",
        )
        for value in (10.0, 20.0)
    ]


def test_merge_simulation_results_names_runs():
    merged = uncertainty.merge_simulation_results(ensemble_grids())
    assert list(merged) == ["run_0", "run_1"]


def test_model_ensemble_stats_unweighted():
    merged = uncertainty.merge_simulation_results(ensemble_grids())
    stats = uncertainty.model_ensemble_stats(merged)
    assert float(stats.z_mean.mean()) == pytest.approx(15.0)
    assert float(stats.z_min.mean()) == pytest.approx(10.0)
    assert float(stats.z_max.mean()) == pytest.approx(20.0)
    # xarray's std uses ddof=0 (population standard deviation)
    assert float(stats.z_stdev.mean()) == pytest.approx(np.std([10.0, 20.0]))
    assert "weighted_mean" not in stats


def test_model_ensemble_stats_weighted():
    merged = uncertainty.merge_simulation_results(ensemble_grids())
    stats = uncertainty.model_ensemble_stats(merged, weights=[3.0, 1.0])
    # weighted mean of 10 (weight 3) and 20 (weight 1)
    assert float(stats.weighted_mean.mean()) == pytest.approx(12.5)
    assert "weighted_stdev" in stats


def test_model_ensemble_stats_region_subset():
    merged = uncertainty.merge_simulation_results(ensemble_grids())
    stats = uncertainty.model_ensemble_stats(merged, region=(0, 1000, 0, 1000))
    assert stats.easting.max() <= 1000
    assert stats.northing.max() <= 1000


def test_mean_and_stdev_prefers_weighted():
    merged = uncertainty.merge_simulation_results(ensemble_grids())
    weighted_stats = uncertainty.model_ensemble_stats(merged, weights=[1.0, 1.0])
    mean, stdev = uncertainty._mean_and_stdev(weighted_stats)
    assert mean.name == "weighted_mean"
    assert stdev.name == "weighted_stdev"

    unweighted_stats = uncertainty.model_ensemble_stats(merged)
    mean, stdev = uncertainty._mean_and_stdev(unweighted_stats)
    assert mean.name == "z_mean"
    assert stdev.name == "z_stdev"


################
################
# regional_misfit_uncertainty
################
################


def observed_gravity() -> xr.Dataset:
    easting = [0.0, 10000.0, 20000.0, 30000.0, 40000.0]
    northing = [0.0, 10000.0, 20000.0, 30000.0]
    x, y = np.meshgrid(easting, northing)
    grav = (y**2 + x**2) / 1e7
    ds = vd.make_xarray_grid(
        (easting, northing),
        data=(
            grav,
            np.full_like(grav, 1000),
            np.full_like(grav, 100),
            np.full_like(grav, 0.5),
        ),
        data_names=("gravity_anomaly", "upward", "forward_gravity", "uncert"),
    )
    return invert4geom.inversion.create_data(ds)


def test_regional_misfit_uncertainty_sample_gravity():
    """
    regression test: sampling the gravity used to crash with a
    MissingDimensionsError, and each run re-sampled from the previous run's noisy
    values instead of the originals
    """
    grav_data = observed_gravity()
    original_gravity = grav_data.gravity_anomaly.copy()

    stats_ds, _ = uncertainty.regional_misfit_uncertainty(
        runs=3,
        sample_gravity=True,
        grav_ds=grav_data,
        method="trend",
        trend=1,
        plot=False,
    )

    assert "z_mean" in stats_ds
    assert "z_stdev" in stats_ds
    # the ensemble must have spread, otherwise sampling did nothing
    assert float(stats_ds.z_stdev.mean()) > 0
    # the final sampled gravity must stay within a few uncertainties of the
    # original values (a random walk would drift further with more runs)
    max_deviation = np.abs(grav_data.gravity_anomaly - original_gravity).max()
    assert float(max_deviation) < 5 * 0.5


def test_regional_misfit_uncertainty_needs_grav_ds():
    with pytest.raises(ValueError, match="grav_ds must be provided"):
        uncertainty.regional_misfit_uncertainty(
            runs=2,
            method="trend",
            trend=1,
            plot=False,
        )
