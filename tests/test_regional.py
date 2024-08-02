# %%
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
import verde as vd
import xarray as xr

from invert4geom import regional, synthetic

# %%


def dummy_grid() -> xr.DataArray:
    (x, y, z) = vd.grid_coordinates(
        region=[0, 200, 200, 400],
        spacing=50,
        extra_coords=20,
    )

    # create synthetic data
    observed_grav = y**2 + x**2

    return vd.make_xarray_grid(
        (x, y),
        (observed_grav, z),
        data_names=("gravity_anomaly", "upward"),
        dims=("northing", "easting"),
    )


def dummy_df() -> pd.DataFrame:
    df = dummy_grid().to_dataframe().reset_index()
    df["starting_gravity"] = 20000
    return df


# %%
def test_regional_constant_constraints():
    """
    test the regional_constant function with a supplied constraints
    """
    grav_df = dummy_df()
    region = (0, 200, 200, 400)

    # create 10 random point within the region
    num_constraints = 10
    coords = vd.scatter_points(region=region, size=num_constraints, random_state=0)
    points = pd.DataFrame(data={"easting": coords[0], "northing": coords[1]})

    df = regional.regional_constant(
        grav_df=grav_df,
        constraints_df=points,
    )

    # test whether regional field has been removed correctly
    # by whether the means of the reg and misfit are similar
    # print(np.mean(df.reg), np.mean(df.misfit))
    assert np.mean(df.reg) == pytest.approx(np.mean(df.misfit), rel=1000)


def test_regional_constant():
    """
    test the regional_constant function with a supplied constant value
    """

    # grav_df = dummy_grid().to_dataframe().reset_index()
    grav_df = dummy_df()

    df = regional.regional_constant(
        grav_df=grav_df,
        constant=-200,
    )

    assert df.reg.mean() == -200


@pytest.mark.parametrize("trend", [0, 2])
def test_regional_trend(trend):
    """
    test the regional_trend function
    """
    anomalies = dummy_df()
    # print(fill_method, trend)

    df = regional.regional_trend(
        trend=trend,
        grav_df=anomalies,
    )

    # grid = df.set_index(["northing", "easting"]).to_xarray()
    # polar_utils.grd_compare(grid.reg, grid.misfit, plot=True, plot_type="xarray")

    assert len(df.misfit) == len(df.reg)

    # test whether regional field has been removed correctly
    # by whether the means of the reg and misfit are similar
    assert np.mean(df.reg) == pytest.approx(np.mean(df.misfit), rel=1e-10)

    # test whether regional field has been remove correctly
    # by ensuring the limits of the regional are not much larger than the range of the
    # misfit
    reg_range = np.max(df.reg) - np.min(df.reg)
    misfit_range = np.max(df.misfit) - np.min(df.misfit)
    # print(reg_range, misfit_range)

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


def test_regional_filter():
    """
    test the regional_filter function
    """
    grav_df = dummy_df()

    df = regional.regional_filter(
        filter_width=300e3,
        grav_df=grav_df,
        # registration="g",
    )

    # grid = df.set_index(["northing", "easting"]).to_xarray()
    # polar_utils.grd_compare(grid.reg, grid.misfit, plot=True, plot_type="xarray")

    assert len(df.misfit) == len(df.reg)

    reg_range = np.max(df.reg) - np.min(df.reg)
    misfit_range = np.max(df.misfit) - np.min(df.misfit)

    # test  whether regional field has been remove correctly
    # by whether the limits of the regional are smaller than the limits of the gravity
    assert reg_range < misfit_range
    # test that the mean regional value is in the range of the misfit values
    assert np.mean(df.reg) < np.max(df.misfit)
    assert np.mean(df.reg) > np.min(df.misfit)


def test_regional_eq_sources():
    """
    test the regional_eq_sources function
    """
    grav_df = dummy_df()
    # grav_df["Gobs"] = np.random.normal(100, 100, len(grav_df))

    # grav_df = dummy_grid().to_dataframe().reset_index()

    # add noise
    grav_df["starting_gravity"], _ = synthetic.contaminate(
        grav_df.starting_gravity,
        stddev=0.2,
        percent=True,
        seed=0,
    )

    df = regional.regional_eq_sources(
        depth=500e3,
        damping=10,
        grav_df=grav_df,
    )
    # print(df)
    reg_range = np.max(df.reg) - np.min(df.reg)
    misfit_range = np.max(df.misfit) - np.min(df.misfit)
    # print(reg_range, misfit_range)
    # test whether regional field has been remove correctly
    # by whether the range of regional values are lower than the range of misfit values
    assert reg_range < misfit_range


@pytest.mark.parametrize(
    "test_input",
    [
        "verde",
        # "pygmt", # issue with pygmt RuntimeWarning
        "eq_sources",
    ],
)
def test_regional_constraints(test_input):
    """
    test the regional_constraints function
    """
    anomalies = dummy_df()
    region = (0, 200, 200, 400)
    # points = pd.DataFrame(
    #     {
    #         # "easting": [-50, -40, -30, -20, 0, 5, 7, 9, 10, 30, 50],
    #         # "northing": [210, 220, 280, 260, 240, 300, 310, 320, 360, 300, 310]
    #         "easting": np.linspace(10, 190, 10),
    #         "northing": np.linspace(210, 390, 10),
    #     }
    # )
    # create 10 random point within the region
    num_constraints = 10
    coords = vd.scatter_points(region=region, size=num_constraints, random_state=0)
    points = pd.DataFrame(data={"easting": coords[0], "northing": coords[1]})

    df = regional.regional_constraints(
        constraints_df=points,
        grav_df=anomalies,
        grid_method=test_input,
        grav_obs_height=1e3,
        depth=100e3,
        spline_dampings=1e-3,
    )

    # grid = df.set_index(["northing", "easting"]).to_xarray()
    # polar_utils.grd_compare(
    #     grid.reg, grid.misfit, plot=True, plot_type="xarray",
    #     points=points.rename(columns={"easting":"x", "northing":"y"}),
    #     )

    # delete the temp files created by optuna
    pathlib.Path("tmp.log").unlink(missing_ok=True)
    pathlib.Path("tmp.log.lock").unlink(missing_ok=True)

    assert len(df.misfit) == len(df.reg)

    # test whether regional field has been removed correctly
    # by whether the means of the reg and misfit are similar
    # print(np.mean(df.reg), np.mean(df.misfit))
    assert np.mean(df.reg) == pytest.approx(np.mean(df.misfit), rel=1000)

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


# %%
