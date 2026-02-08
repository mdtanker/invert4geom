# %%

import numpy as np
import pandas as pd
import pytest
import verde as vd
import xarray as xr

import invert4geom


def observed_gravity() -> xr.Dataset:
    easting = [0.0, 10000.0, 20000.0, 30000.0, 40000.0]
    northing = [0.0, 10000.0, 20000.0, 30000.0]

    # create synthetic data
    # x, y = np.meshgrid(easting, northing)
    # grav = (y**2 + x**2)/1e7
    grav = [
        [0.0, 10.0, 40.0, 90.0, 160.0],
        [10.0, 20.0, 50.0, 100.0, 170.0],
        [40.0, 50.0, 80.0, 130.0, 200.0],
        [90.0, 100.0, 130.0, 180.0, 250.0],
    ]

    ds = vd.make_xarray_grid(
        (easting, northing),
        data=(grav, np.full_like(grav, 1000), np.full_like(grav, 100)),
        data_names=("gravity_anomaly", "upward", "forward_gravity"),
    )

    return invert4geom.inversion.create_data(ds)


def test_regional_constant_constraints():
    """
    test the regional_constant function with a supplied constraints
    """
    grav_data = observed_gravity()

    constraints = pd.DataFrame(
        data={
            "easting": [10000, 20000],
            "northing": [10000, 30000],
            "upward": [500, 500],
        }
    )

    invert4geom.regional.regional_constant(
        grav_ds=grav_data,
        constraints_df=constraints,
    )

    # the constant regional value should be the mean of `misfit` at the constraints
    gravity_anomaly_at_constraints = [
        (y**2 + x**2) / 1e7
        for x, y in zip(constraints.easting, constraints.northing, strict=False)
    ]
    misfit_at_constraints = [x - 100 for x in gravity_anomaly_at_constraints]
    expected_regional_value = np.mean(misfit_at_constraints)

    assert np.mean(grav_data.reg) == expected_regional_value


def test_regional_constant():
    """
    test the regional_constant function with a supplied constant value
    """
    grav_data = observed_gravity()

    invert4geom.regional.regional_constant(
        grav_ds=grav_data,
        constant=-200,
    )

    assert grav_data.reg.mean() == -200


@pytest.mark.filterwarnings("ignore:dropping variables using `drop` is deprecated")
@pytest.mark.filterwarnings("ignore:Default ifft's behaviour")
def test_regional_filter():
    """
    test the regional_filter function
    """
    grav_data = observed_gravity()

    invert4geom.regional.regional_filter(
        filter_width=300e3,
        grav_ds=grav_data,
    )

    assert len(grav_data.misfit) == len(grav_data.reg)

    reg_range = np.max(grav_data.reg) - np.min(grav_data.reg)
    misfit_range = np.max(grav_data.misfit) - np.min(grav_data.misfit)

    # test  whether regional field has been remove correctly
    # by whether the limits of the regional are smaller than the limits of the gravity
    assert reg_range < misfit_range
    # test that the mean regional value is in the range of the misfit values
    assert np.mean(grav_data.reg) < np.max(grav_data.misfit)
    assert np.mean(grav_data.reg) > np.min(grav_data.misfit)


@pytest.mark.parametrize("trend", [0, 2])
def test_regional_trend(trend):
    """
    test the regional_trend function
    """
    grav_data = observed_gravity()

    invert4geom.regional.regional_trend(
        trend=trend,
        grav_ds=grav_data,
    )

    df = grav_data.inv.df

    assert len(df.misfit) == len(df.reg)

    # test whether regional field has been removed correctly
    # by whether the means of the reg and misfit are similar
    assert np.mean(df.reg) == pytest.approx(np.mean(df.misfit), rel=1e-10)

    # test whether regional field has been remove correctly
    # by ensuring the limits of the regional are not much larger than the range of the
    # misfit
    reg_range = np.max(df.reg) - np.min(df.reg)
    misfit_range = np.max(df.misfit) - np.min(df.misfit)

    # assert reg_range < misfit_range or at least close
    assert reg_range < misfit_range or reg_range == pytest.approx(
        misfit_range, rel=1e-10
    )
    # test that the regional values are between the misfit values, or similar
    assert np.max(df.reg) < np.max(df.misfit) or np.max(df.reg) == pytest.approx(
        np.max(df.misfit), rel=1e-10
    )
    assert np.min(df.reg) > np.min(df.misfit) or np.min(df.reg) == pytest.approx(
        np.min(df.misfit), rel=1e-10
    )


def test_regional_eq_sources():
    """
    test the regional_eq_sources function
    """
    grav_data = observed_gravity()

    # add noise
    grav_data["forward_gravity"], _ = invert4geom.synthetic.contaminate(
        grav_data.forward_gravity,
        stddev=0.01,
        percent=True,
        seed=0,
    )

    invert4geom.regional.regional_eq_sources(
        depth=100e3,
        damping=1,
        grav_ds=grav_data,
    )

    reg_range = np.max(grav_data.reg) - np.min(grav_data.reg)
    misfit_range = np.max(grav_data.misfit) - np.min(grav_data.misfit)

    # test whether regional field has been remove correctly
    # by whether the range of regional values are lower than the range of misfit values
    assert reg_range < misfit_range


@pytest.mark.filterwarnings(
    "ignore:The mindist parameter of verde.Spline is no longer required and will be removed in Verde 2.0.0. Use the default value to obtain the future behavior."
)
@pytest.mark.filterwarnings("ignore:: FutureWarning")
@pytest.mark.filterwarnings("ignore:The following error was raised:")
@pytest.mark.filterwarnings("ignore:Cannot have number of splits")
@pytest.mark.filterwarnings("ignore:decreasing number of splits by 1")
@pytest.mark.filterwarnings("ignore:: sklearn.exceptions.UndefinedMetricWarning")
@pytest.mark.parametrize(
    "test_input",
    [
        "verde",
        "pygmt",  # issue with pygmt RuntimeWarning
        "eq_sources",
    ],
)
def test_regional_constraints(test_input):
    """
    test the regional_constraints function
    """
    grav_data = observed_gravity()

    # 1 point near each corner of the grid
    constraints = pd.DataFrame(
        data={
            "easting": [5000, 35000, 5000, 35000],
            "northing": [5000, 25000, 25000, 5000],
            "upward": [500, 500, 500, 500],
        }
    )

    invert4geom.regional.regional_constraints(
        constraints_df=constraints,
        grav_ds=grav_data,
        grid_method=test_input,
        grav_obs_height=1e3,
        depth=100e3,
        spline_dampings=1e-3,
    )

    # ptk.grid_compare(
    #     ds.reg,
    #     ds.misfit,
    #     points=constraints,
    #     grid1_name="reg",
    #     grid2_name="misfit",
    # )

    if test_input == "verde":
        expected = [
            [-129.87915086, -86.90383075, -46.64122827, -6.83913916, 33.70990694],
            [-97.36785292, -55.07908713, -15.51259498, 23.75995967, 64.35825028],
            [-67.2664448, -25.44787136, 14.23583785, 53.39117545, 94.4596584],
            [-37.42348908, 4.35522248, 44.57059026, 84.41991407, 126.16556872],
        ]
    elif test_input == "pygmt":
        expected = [
            [-126.25, -86.25, -46.25, -6.25, 33.75],
            [-96.25, -56.25, -16.25, 23.75, 63.75],
            [-66.25, -26.25, 13.75, 53.75, 93.75],
            [-36.25, 3.75, 43.75, 83.75, 123.75],
        ]
    elif test_input == "eq_sources":
        expected = [
            [-121.35803396, -86.18668308, -46.75683717, -6.1701828, 32.34455063],
            [-94.98293087, -57.77247606, -16.86896395, 24.44519832, 62.84220412],
            [-65.33154372, -26.96784838, 14.3349387, 55.249826, 92.49359128],
            [-34.81156908, 3.67070039, 44.24627242, 83.68720067, 118.89101551],
        ]

    assert not np.testing.assert_allclose(grav_data.reg.values, expected)

    # delete the temp files created by optuna
    # pathlib.Path("tmp.log").unlink(missing_ok=True)
    # pathlib.Path("tmp.log.lock").unlink(missing_ok=True)

    # test whether regional field has been removed correctly
    # by whether the means of the reg and misfit are similar
    # print(np.mean(ds.reg), np.mean(ds.misfit))
    # assert np.mean(ds.reg) == pytest.approx(np.mean(ds.misfit)-100)

    # test whether regional field has been remove correctly by ensuring the limits of
    # the regional are not much larger than the range of the misfit
    # reg_range = np.max(df.reg) - np.min(df.reg)
    # misfit_range = np.max(df.misfit) - np.min(df.misfit)

    # # assert reg_range < misfit_range or at least close
    # print(reg_range, misfit_range)
    # assert reg_range < misfit_range or
    #   (reg_range == pytest.approx(misfit_range, rel=1e-10))

    # # test that the regional values are between the misfit values, or similar
    # assert np.max(df.reg) < np.max(df.misfit) or
    #   (np.max(df.reg) == pytest.approx(np.max(df.misfit), rel=1e-10))
    # assert np.min(df.reg) > np.min(df.misfit) or
    #   (np.min(df.reg) == pytest.approx(np.min(df.misfit), rel=1e-10))

    # reg_range = np.max(ds.reg) - np.min(ds.reg)
    # misfit_range = np.max(ds.misfit) - np.min(ds.misfit)

    # # test whether regional field has been remove correctly
    # # by whether the range of regional values are lower than the range of misfit values
    # assert reg_range < misfit_range


# %%
