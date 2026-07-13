import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import xarray as xr

from invert4geom import synthetic

################
################
# _gaussian2d
################
################


def test_gaussian2d_peak_at_center():
    """the un-normalized Gaussian should be 1 at its center"""
    value = synthetic._gaussian2d(
        x=np.array([500.0]),
        y=np.array([-200.0]),
        sigma_x=100,
        sigma_y=50,
        x0=500,
        y0=-200,
    )
    assert value[0] == pytest.approx(1.0)


def test_gaussian2d_decays_away_from_center():
    x = np.array([0.0, 50.0, 100.0])
    y = np.zeros(3)
    values = synthetic._gaussian2d(x, y, sigma_x=100, sigma_y=100)
    assert values[0] == pytest.approx(1.0)
    assert values[0] > values[1] > values[2]


def test_gaussian2d_rotation_by_90_swaps_sigmas():
    """
    regression test: the first quadratic-form coefficient was missing a square on
    the cosine term, so rotated Gaussians were misshapen. Rotating an anisotropic
    Gaussian by 90 degrees must be identical to swapping its sigmas.
    """
    x, y = np.meshgrid(np.linspace(-100, 100, 21), np.linspace(-100, 100, 21))
    rotated = synthetic._gaussian2d(x, y, sigma_x=50, sigma_y=20, angle=90)
    swapped = synthetic._gaussian2d(x, y, sigma_x=20, sigma_y=50, angle=0)
    npt.assert_allclose(rotated, swapped, atol=1e-12)


def test_gaussian2d_rotation_by_180_is_identity():
    """rotating any Gaussian by 180 degrees must not change it"""
    x, y = np.meshgrid(np.linspace(-100, 100, 21), np.linspace(-100, 100, 21))
    unrotated = synthetic._gaussian2d(x, y, sigma_x=50, sigma_y=20, angle=0)
    rotated = synthetic._gaussian2d(x, y, sigma_x=50, sigma_y=20, angle=180)
    npt.assert_allclose(rotated, unrotated, atol=1e-12)


def test_gaussian2d_values_bounded():
    """a Gaussian must never exceed its peak value of 1, at any rotation angle"""
    x, y = np.meshgrid(np.linspace(-500, 500, 41), np.linspace(-500, 500, 41))
    for angle in (0, 30, 45, 90, 135, 200):
        values = synthetic._gaussian2d(x, y, sigma_x=100, sigma_y=20, angle=angle)
        assert values.max() <= 1.0 + 1e-12


################
################
# synthetic topography
################
################


def test_synthetic_topography_simple_grid_properties():
    region = (0.0, 40000.0, 0.0, 30000.0)
    grid = synthetic.synthetic_topography_simple(spacing=1000, region=region)
    assert isinstance(grid, xr.DataArray)
    assert grid.name == "upward"
    assert grid.dims == ("northing", "easting")
    assert grid.easting.min() == region[0]
    assert grid.easting.max() == region[1]
    assert grid.northing.min() == region[2]
    assert grid.northing.max() == region[3]


def test_synthetic_topography_simple_scale_and_offset():
    region = (0.0, 40000.0, 0.0, 30000.0)
    grid = synthetic.synthetic_topography_simple(spacing=1000, region=region)
    shifted = synthetic.synthetic_topography_simple(
        spacing=1000,
        region=region,
        scale=2,
        yoffset=100,
    )
    npt.assert_allclose(shifted.to_numpy(), grid.to_numpy() * 2 + 100, rtol=1e-10)


@pytest.mark.parametrize(
    "function",
    [
        synthetic.synthetic_topography_simple,
        synthetic.synthetic_topography_regional,
    ],
)
def test_synthetic_topography_invalid_registration_raises(function):
    with pytest.raises(ValueError, match="registration must be"):
        function(spacing=1000, region=(0, 10000, 0, 10000), registration="x")


def test_synthetic_topography_regional_zero_mean():
    """the regional topography is centered before the offset is added"""
    grid = synthetic.synthetic_topography_regional(
        spacing=1000,
        region=(0, 40000, 0, 30000),
        yoffset=500,
    )
    assert grid.to_numpy().mean() == pytest.approx(500)


################
################
# contaminate
################
################


def simple_grid() -> xr.DataArray:
    easting = np.arange(0, 5000, 1000.0)
    northing = np.arange(0, 4000, 1000.0)
    values = np.outer(northing, easting) / 1e6 + 10
    return xr.DataArray(
        values,
        coords={"northing": northing, "easting": easting},
        dims=("northing", "easting"),
        name="gravity",
    )


def test_contaminate_zero_stddev_leaves_data_unchanged():
    grid = simple_grid()
    contaminated, stddev = synthetic.contaminate(grid, stddev=0.0)
    npt.assert_allclose(contaminated.to_numpy(), grid.to_numpy())
    assert stddev == 0.0


def test_contaminate_reproducible_with_seed():
    grid = simple_grid()
    first, _ = synthetic.contaminate(grid, stddev=1.0, seed=42)
    second, _ = synthetic.contaminate(grid, stddev=1.0, seed=42)
    npt.assert_allclose(first.to_numpy(), second.to_numpy())


def test_contaminate_noise_has_zero_mean():
    grid = simple_grid()
    contaminated, _ = synthetic.contaminate(grid, stddev=1.0, seed=0)
    noise = contaminated - grid
    assert noise.to_numpy().mean() == pytest.approx(0.0, abs=1e-12)


def test_contaminate_percent_uses_max_abs():
    grid = simple_grid()
    _, stddev = synthetic.contaminate(grid, stddev=0.1, percent=True)
    assert stddev == pytest.approx(0.1 * np.abs(grid.to_numpy()).max())


def test_contaminate_percent_pointwise_returns_series():
    grid = simple_grid()
    _, stddev = synthetic.contaminate(
        grid,
        stddev=0.1,
        percent=True,
        percent_as_max_abs=False,
    )
    assert isinstance(stddev, pd.Series)
    assert len(stddev) == grid.size


def test_contaminate_rejects_non_dataarray():
    with pytest.raises(DeprecationWarning, match="must be a xarray dataarray"):
        synthetic.contaminate(pd.DataFrame({"a": [1.0]}), stddev=1.0)
