import typing

import harmonica as hm
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import verde as vd
import xarray as xr
import xarray.testing as xrt

from invert4geom import utils

pd.set_option("display.max_columns", None)
################
################
# DUMMY FUNCTIONS
################
################


def dummy_grid() -> xr.Dataset:
    (x, y, z) = vd.grid_coordinates(
        region=[000, 200, 200, 400],
        spacing=100,
        extra_coords=20,
    )
    # create topographic features
    data = y**2 + x**2
    return vd.make_xarray_grid(
        (x, y),
        (data, z),
        data_names=("scalars", "upward"),
        dims=("northing", "easting"),
    )


def dummy_prism_layer() -> xr.Dataset:
    """
    Create a dummy prism layer
    """
    (easting, northing) = vd.grid_coordinates(region=[-200, 200, 100, 500], spacing=200)
    data = [[0, 0, 0], [-30, -30, -30], [30, 30, 30]]
    surface = vd.make_xarray_grid(
        (easting, northing), data, data_names="surface"
    ).surface
    dens = 2670
    reference = 0
    density = xr.where(surface >= reference, dens, -dens)
    return hm.prism_layer(
        coordinates=(easting[0, :], northing[:, 0]),
        surface=surface,
        reference=reference,
        properties={"density": density},
    )


def dummy_prism_layer_flat_bottom() -> xr.Dataset:
    """
    Create a dummy prism layer
    """
    (easting, northing) = vd.grid_coordinates(region=[-200, 200, 100, 500], spacing=200)
    surface = [[0, 0, 0], [-30, -30, -30], [30, 30, 30]]
    density = 2670.0 * np.ones_like(surface)
    return hm.prism_layer(
        coordinates=(easting[0, :], northing[:, 0]),
        surface=surface,
        reference=-100,
        properties={"density": density},
    )


def dummy_prism_layer_flat() -> xr.Dataset:
    """
    Create a dummy prism layer
    """
    (easting, northing) = vd.grid_coordinates(region=[-200, 200, 100, 500], spacing=200)
    surface = np.zeros_like(easting)
    density = 2670.0 * np.ones_like(surface)
    return hm.prism_layer(
        coordinates=(easting[0, :], northing[:, 0]),
        surface=surface,
        reference=-100,
        properties={"density": density},
    )


################
################
# TESTS
################
################


def test_rmeanse():
    """
    test the Root Mean Square Error function
    """
    # create some dummy data
    data = np.array([1, 2, 3])
    # calculate the RMSE
    value = utils.rmse(data)
    # test that the RMSE is correct
    assert value == 2.1602468994692867437


def test_rmedianse():
    """
    test the Root Median Square Error function
    """
    # create some dummy data
    data = np.array([1, 2, 3])
    # calculate the RMSE
    value = utils.rmse(data, as_median=True)
    # test that the RMSE is correct
    assert value == 2


@pytest.mark.parametrize(
    "test_input",
    [
        # "pygmt",
        "rioxarray",
        "verde",
    ],
)
def test_nearest_grid_fill(test_input):
    # make a grid with a hole in it
    grid = dummy_grid().scalars
    grid.loc[{"easting": 100, "northing": 200}] = np.nan
    # check the grid has a hole
    assert grid.isnull().any()  # noqa: PD003
    # fill the hole
    filled = utils._nearest_grid_fill(grid, method=test_input, crs="epsg:3031")
    # check that the hole has been filled
    assert not filled.isnull().any()  # noqa: PD003
    # check fill value is equal to one of the adjacent cells
    expected = [
        filled.loc[{"easting": 0, "northing": 200}],
        filled.loc[{"easting": 200, "northing": 200}],
        filled.loc[{"easting": 100, "northing": 300}],
    ]
    assert filled.loc[{"easting": 100, "northing": 200}] in expected


@pytest.mark.filterwarnings(
    "ignore:dropping variables using `drop` is deprecated; use drop_vars."
)
@pytest.mark.filterwarnings("ignore:: FutureWarning")
@pytest.mark.parametrize(
    "test_input",
    [
        "lowpass",
        "highpass",
        "up_deriv",
        "easting_deriv",
        "northing_deriv",
    ],
)
def test_filter_grid(test_input):
    """
    test the filter_grid function returns a valid grid, testing of actual filter is
    covered in harmonica
    """
    # create some dummy data
    grid = dummy_grid().scalars
    # filter the grid
    filtered = utils.filter_grid(grid, 10000, filter_type=test_input)
    # check the filtered grid is not identical to the original grid
    with pytest.raises(AssertionError):
        xrt.assert_identical(grid, filtered)
    # set all grid values to 0, and check grids are identical to check metadata is
    # correct
    grid = grid.where(grid == 0, other=0)
    filtered = filtered.where(filtered == 0, other=0)
    xrt.assert_identical(grid, filtered)


def test_filter_grid_wrong_filter_type():
    """
    ensure ValueError is raised with wrong filter_type
    """
    # create some dummy data
    grid = dummy_grid().scalars
    # assert error is raised
    with pytest.raises(ValueError, match="filter_type must"):
        utils.filter_grid(grid, 10000, filter_type="wrong_filter_type")


@pytest.mark.filterwarnings(
    "ignore:dropping variables using `drop` is deprecated; use drop_vars."
)
@pytest.mark.filterwarnings("ignore:: FutureWarning")
def test_filter_grid_nans():
    """
    test the filter_grid function with nans in input
    """
    # create some dummy data
    grid = dummy_grid().scalars
    # add a nan to the grid
    grid.loc[{"easting": 100, "northing": 200}] = np.nan
    # check the grid has a hole
    assert grid.isnull().any()  # noqa: PD003
    # filter the grid
    filtered = utils.filter_grid(grid, 10000, filter_type="lowpass")
    # check that the grid has been low-pass filtered
    assert np.max(filtered) < np.mean(grid)


# def test_filter_grid_change_spacing():
#     """
#     test the filter_grid function with changing the grid spacing option
#     """
#     # create some dummy data
#     grid = dummy_grid().scalars
#     # filter the grid
#     filtered = utils.filter_grid(grid, filter_width=500, change_spacing=True)
#     # get grid spacings
#     original_spacing = float(ptk.get_grid_info(grid)[0])
#     new_spacing = float(ptk.get_grid_info(filtered)[0])
#     print(original_spacing)
#     print(new_spacing)
#     # check that the gridspacing has been changed
#     assert original_spacing != new_spacing


def test_dist_nearest_points():
    """
    test the dist_nearest_points function
    """
    # create 3 targets, one on grid node, one nearby, and one outside grid
    targets = pd.DataFrame({"easting": [0, 37, -1050], "northing": [200, -244, -700]})
    # create dataarray, dataset and dataframe
    ds = dummy_grid()
    da = ds.upward
    df = vd.grid_to_table(da)
    # print(df)
    # calculate the distance with a df
    dist_df = utils.dist_nearest_points(
        targets,
        df,
        coord_names=("easting", "northing"),
    )
    # calculate the distance with a da
    dist_da = utils.dist_nearest_points(
        targets,
        da,
        coord_names=("easting", "northing"),
    )
    # calculate the distance with a ds
    dist_ds = utils.dist_nearest_points(
        targets,
        ds,
        coord_names=("easting", "northing"),
    )
    # print(dist_df)
    da_results = np.array(vd.grid_to_table(dist_da).min_dist)
    ds_results = np.array(vd.grid_to_table(dist_ds).min_dist)
    df_results = np.array(dist_df.min_dist)
    # test that the results all match
    npt.assert_array_equal(ds_results, da_results)
    npt.assert_array_equal(da_results, df_results)
    # test that smallest min_dist and largest min_dist are correct
    assert np.min(df_results) == pytest.approx(0)
    assert np.max(df_results) == pytest.approx(282.842712)


def test_normalize_xarray_range():
    """
    Ensure the output is within the [low, high] range.
    """
    data = np.array([1, 2, 3, 4, 5])
    da = xr.DataArray(data, dims="easting")
    da_normalized = utils.normalize_xarray(da, low=2, high=5)
    assert da_normalized.min() >= 2
    assert da_normalized.max() <= 5


def test_normalize_xarray_values():
    """
    Ensure the normalized data matches the expected values.
    """
    data = np.array([1, 2, 3, 4, 5])
    da = xr.DataArray(data, dims="easting")
    da_normalized = utils.normalize_xarray(da, low=0, high=1)
    expected_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    npt.assert_array_almost_equal(da_normalized.values, expected_values, decimal=6)


def test_normalize_xarray_negative_values():
    """
    Ensure the function handles negative values correctly.
    """
    data = np.array([-5, 0, 5])
    da = xr.DataArray(data, dims="easting")
    da_normalized = utils.normalize_xarray(da, low=0, high=1)
    expected_values = np.array([0.0, 0.5, 1.0])
    npt.assert_array_almost_equal(da_normalized.values, expected_values, decimal=6)


def test_normalize_xarray_custom_range():
    """
    Ensure the function handles a custom low and high range correctly.
    """
    data = np.array([10, 20, 30])
    da = xr.DataArray(data, dims="easting")
    da_normalized = utils.normalize_xarray(da, low=-1, high=1)
    expected_values = np.array([-1.0, 0.0, 1.0])
    npt.assert_array_almost_equal(da_normalized.values, expected_values, decimal=6)


def test_normalized_mindist_defaults():
    """
    test the normalized_mindist function
    """
    # create 2 constraint points
    points = pd.DataFrame({"easting": [0, 2], "northing": [0, -1]})
    # create dataarray
    df = pd.DataFrame(
        {
            "easting": [-4, 4, 0, 4, -4],
            "northing": [-4, 4, 0, -4, 4],
            "z": [0, 1, 2, 3, 4],
        }
    )
    da = df.set_index(["northing", "easting"]).to_xarray().z
    # calculate the min distance with defaults
    min_dist = utils.normalized_mindist(
        points=points,
        grid=da,
    )
    # test the grid is correct
    expected = np.array(
        [
            [5.65685425, np.nan, 3.60555128],
            [np.nan, 0.0, np.nan],
            [5.65685425, np.nan, 5.38516481],
        ]
    )
    npt.assert_almost_equal(expected, min_dist.values)


def test_normalized_mindist_mindist():
    """
    test the normalized_mindist function with any values lower than 4 set to 0 with
    parameter `mindist`.
    """
    # create 2 constraint points
    points = pd.DataFrame({"easting": [0, 2], "northing": [0, -1]})
    # create dataarray
    df = pd.DataFrame(
        {
            "easting": [-4, 4, 0, 4, -4],
            "northing": [-4, 4, 0, -4, 4],
            "z": [0, 1, 2, 3, 4],
        }
    )
    da = df.set_index(["northing", "easting"]).to_xarray().z

    # calculate the min distance with mindist value of 4
    min_dist = utils.normalized_mindist(
        points=points,
        grid=da,
        mindist=4,
    )
    expected = np.array(
        [
            [5.65685425, np.nan, 0],
            [np.nan, 0.0, np.nan],
            [5.65685425, np.nan, 5.38516481],
        ]
    )
    npt.assert_almost_equal(expected, min_dist.values)


def test_normalized_mindist_region():
    """
    test the normalized_mindist function with region parameter
    """
    # create 2 constraint points
    points = pd.DataFrame({"easting": [0, 2], "northing": [0, -1]})
    # create dataarray
    df = pd.DataFrame(
        {
            "easting": [-4, 4, 0, 4, -4],
            "northing": [-4, 4, 0, -4, 4],
            "z": [0, 1, 2, 3, 4],
        }
    )
    da = df.set_index(["northing", "easting"]).to_xarray().z
    # calculate the min distance and points outside region set to 0
    min_dist = utils.normalized_mindist(
        points=points,
        grid=da,
        region=[-4.0, -3.0, -4.0, 0],
    )
    # test the grid is correct
    expected = np.array(
        [[5.65685425, np.nan, 0.0], [np.nan, 0.0, np.nan], [0.0, np.nan, 0.0]]
    )
    npt.assert_almost_equal(expected, min_dist.values)


def test_normalized_mindist_high_low():
    """
    test the normalized_mindist function with set high and low values
    """
    # create 2 constraint points
    points = pd.DataFrame({"easting": [0, 2], "northing": [0, -1]})
    # create dataarray
    df = pd.DataFrame(
        {
            "easting": [-4, 4, 0, 4, -4],
            "northing": [-4, 4, 0, -4, 4],
            "z": [0, 1, 2, 3, 4],
        }
    )
    da = df.set_index(["northing", "easting"]).to_xarray().z
    # calculate the min distance with normalizing limits
    min_dist = utils.normalized_mindist(
        points=points,
        grid=da,
        low=0.2,
        high=5,
    )
    # test that smallest min_dist and largest min_dist are correct
    assert np.min(min_dist) == pytest.approx(0.2)
    assert np.max(min_dist) == pytest.approx(5)


################
################
# region_mask
################
################


def asymmetric_grid() -> xr.DataArray:
    """create a grid with different easting and northing extents"""
    (x, y) = vd.grid_coordinates(region=(0, 100, 0, 50), spacing=10)
    return vd.make_xarray_grid(
        (x, y),
        np.ones_like(x),
        data_names="dummy",
        dims=("northing", "easting"),
    ).dummy


def test_region_mask_inside_cells_are_one():
    """cells inside the region should have a mask value of 1"""
    grid = asymmetric_grid()
    masked = utils.region_mask(grid, region=(0, 30, 0, 50))
    assert masked.sel(easting=slice(0, 30)).to_numpy().all()


def test_region_mask_cells_east_of_region_are_zero():
    """
    regression test for easting and northing being swapped, which left cells
    outside the region's easting bounds unmasked
    """
    grid = asymmetric_grid()
    masked = utils.region_mask(grid, region=(0, 30, 0, 50))
    assert masked.sel(easting=slice(40, 100)).sum() == 0


def test_region_mask_cells_north_of_region_are_zero():
    """cells outside the region's northing bounds should have a mask value of 0"""
    grid = asymmetric_grid()
    masked = utils.region_mask(grid, region=(0, 100, 0, 20))
    assert masked.sel(northing=slice(30, 50)).sum() == 0


def test_region_mask_full_region_all_ones():
    """a region covering the whole grid should mask nothing"""
    grid = asymmetric_grid()
    masked = utils.region_mask(grid, region=(0, 100, 0, 50))
    assert masked.to_numpy().all()


################
################
# get_epsg
################
################


def test_get_epsg_no_env_var_coast_false():
    """without the env vars, should fall back to 3857 without warning"""
    with utils._environ(POLARTOOLKIT_EPSG=None, POLARTOOLKIT_HEMISPHERE=None):
        epsg, coast = utils.get_epsg(coast=False)
    assert epsg == "3857"
    assert coast is False


def test_get_epsg_no_env_var_coast_true_warns():
    """without the env vars, requesting coastlines should warn and disable them"""
    with (
        utils._environ(POLARTOOLKIT_EPSG=None, POLARTOOLKIT_HEMISPHERE=None),
        pytest.warns(UserWarning, match="POLARTOOLKIT_EPSG"),
    ):
        epsg, coast = utils.get_epsg(coast=True)
    assert epsg == "3857"
    assert coast is False


def test_get_epsg_with_env_var():
    """with the env var set, should return its EPSG code and keep coast enabled"""
    with utils._environ(POLARTOOLKIT_EPSG="3031"):
        epsg, coast = utils.get_epsg(coast=True)
    assert str(epsg) == "3031"
    assert coast is True


################
################
# normalized_mindist input validation
################
################


def normalized_mindist_inputs() -> tuple[pd.DataFrame, xr.DataArray]:
    points = pd.DataFrame({"easting": [0, 2], "northing": [0, -1]})
    df = pd.DataFrame(
        {
            "easting": [-4, 4, 0, 4, -4],
            "northing": [-4, 4, 0, -4, 4],
            "z": [0, 1, 2, 3, 4],
        }
    )
    da = df.set_index(["northing", "easting"]).to_xarray().z
    return points, da


def test_normalized_mindist_only_low_raises():
    """providing `low` without `high` should raise a clear error"""
    points, da = normalized_mindist_inputs()
    with pytest.raises(ValueError, match="both `low` and `high`"):
        utils.normalized_mindist(points=points, grid=da, low=0.2)


def test_normalized_mindist_only_high_raises():
    """providing `high` without `low` should raise a clear error"""
    points, da = normalized_mindist_inputs()
    with pytest.raises(ValueError, match="both `low` and `high`"):
        utils.normalized_mindist(points=points, grid=da, high=5)


################
################
# _block_reduce_points
################
################


def block_reduce_inputs() -> tuple[typing.Any, typing.Any, typing.Any]:
    rng = np.random.default_rng(seed=0)
    coords = (rng.uniform(0, 1000, 40), rng.uniform(0, 1000, 40))
    data = coords[0] + coords[1]
    weights = np.ones_like(data)
    return coords, data, weights


def test_block_reduce_points_weighted_mean_reduces_weights():
    """a weighted mean reduction should return weights matching the reduced data"""
    coords, data, weights = block_reduce_inputs()
    (
        reduced_coords,
        reduced_data,
        reduced_weights,
    ) = utils._block_reduce_points(
        coords,
        data,
        weights,
        block_size=500,
        block_reduction="mean",
        region=(0, 1000, 0, 1000),
    )
    assert len(reduced_data) < len(data)
    assert len(reduced_weights) == len(reduced_data) == len(reduced_coords[0])


def test_block_reduce_points_median_drops_weights():
    """a median reduction can't use weights, so they should be dropped"""
    coords, data, weights = block_reduce_inputs()
    _, reduced_data, reduced_weights = utils._block_reduce_points(
        coords,
        data,
        weights,
        block_size=500,
        block_reduction="median",
        region=(0, 1000, 0, 1000),
    )
    assert len(reduced_data) < len(data)
    assert reduced_weights is None


def test_block_reduce_points_invalid_reduction_raises():
    coords, data, weights = block_reduce_inputs()
    with pytest.raises(ValueError, match="block_reduction must be"):
        utils._block_reduce_points(
            coords,
            data,
            weights,
            block_size=500,
            block_reduction="mode",
            region=(0, 1000, 0, 1000),
        )


################
################
# best_equivalent_source_damping
################
################


def test_best_equivalent_source_damping_keeps_kwargs():
    """
    regression test: the returned equivalent sources must be fitted with the same
    kwargs (e.g. depth) that were used during cross-validation scoring
    """
    rng = np.random.default_rng(seed=0)
    easting = rng.uniform(0, 10000, 30)
    northing = rng.uniform(0, 10000, 30)
    upward = np.full_like(easting, 1000)
    data = 1e-7 * (easting**2 + northing**2)

    eqs = utils.best_equivalent_source_damping(
        coordinates=(easting, northing, upward),
        data=data,
        dampings=[None],
        depth=5000,
    )
    assert eqs.depth == 5000


################
################
# gravity_decay_buffer
################
################


def test_gravity_decay_buffer_returns_valid_outputs():
    max_decay, buffer_width, buffer_cells, grav_ds = utils.gravity_decay_buffer(
        buffer_perc=10,
        spacing=1000,
        inner_region=(0, 10000, 0, 10000),
        top=0,
        zref=-1000,
        obs_height=1000,
        density=2670,
        plot=False,
    )
    assert buffer_width % 1000 == 0
    assert buffer_cells == buffer_width / 1000
    assert max_decay > 0
    assert "forward" in grav_ds
    assert "forward_no_edge_effects" in grav_ds


def test_gravity_decay_buffer_as_density_contrast_no_decay():
    """
    discretizing the topography as a density contrast should result in no
    edge effects, and therefore no decay
    """
    max_decay, _, _, _ = utils.gravity_decay_buffer(
        buffer_perc=10,
        spacing=1000,
        inner_region=(0, 10000, 0, 10000),
        top=0,
        zref=-1000,
        obs_height=1000,
        density=2670,
        as_density_contrast=True,
        checkerboard=True,
        amplitude=100,
        wavelength=5000,
        plot=False,
    )
    assert max_decay == pytest.approx(0, abs=1e-10)


def test_gravity_decay_buffer_flat_topo_equal_top_and_zref_raises():
    with pytest.raises(ValueError, match="top and zref must be different"):
        utils.gravity_decay_buffer(
            buffer_perc=10,
            spacing=1000,
            inner_region=(0, 10000, 0, 10000),
            top=0,
            zref=0,
            obs_height=1000,
            density=2670,
            plot=False,
        )
