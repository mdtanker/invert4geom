from __future__ import annotations

import harmonica as hm
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
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
    assert grid.isnull().any()
    # fill the hole
    filled = utils.nearest_grid_fill(grid, method=test_input)
    # check that the hole has been filled
    assert not filled.isnull().any()
    # check fill value is equal to one of the adjacent cells
    expected = [
        filled.loc[{"easting": 0, "northing": 200}],
        filled.loc[{"easting": 200, "northing": 200}],
        filled.loc[{"easting": 100, "northing": 300}],
    ]
    assert filled.loc[{"easting": 100, "northing": 200}] in expected


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
    filtered = utils.filter_grid(grid, 10000, filt_type=test_input)
    # check the filtered grid is not identical to the original grid
    with pytest.raises(AssertionError):
        xrt.assert_identical(grid, filtered)
    # set all grid values to 0, and check grids are identical to check metadata is
    # correct
    grid = grid.where(grid == 0, other=0)
    filtered = filtered.where(filtered == 0, other=0)
    xrt.assert_identical(grid, filtered)


def test_filter_grid_wrong_filt_type():
    """
    ensure ValueError is raised with wrong filt_type
    """
    # create some dummy data
    grid = dummy_grid().scalars
    # assert error is raised
    with pytest.raises(ValueError, match="filt_type must"):
        utils.filter_grid(grid, 10000, filt_type="wrong_filt_type")


def test_filter_grid_nans():
    """
    test the filter_grid function with nans in input
    """
    # create some dummy data
    grid = dummy_grid().scalars
    # add a nan to the grid
    grid.loc[{"easting": 100, "northing": 200}] = np.nan
    # check the grid has a hole
    assert grid.isnull().any()
    # filter the grid
    filtered = utils.filter_grid(grid, 10000, filt_type="lowpass")
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
#     original_spacing = float(ap_utils.get_grid_info(grid)[0])
#     new_spacing = float(ap_utils.get_grid_info(filtered)[0])
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
    da = xr.DataArray(data, dims="x")
    da_normalized = utils.normalize_xarray(da, low=2, high=5)
    assert da_normalized.min() >= 2
    assert da_normalized.max() <= 5


def test_normalize_xarray_values():
    """
    Ensure the normalized data matches the expected values.
    """
    data = np.array([1, 2, 3, 4, 5])
    da = xr.DataArray(data, dims="x")
    da_normalized = utils.normalize_xarray(da, low=0, high=1)
    expected_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    npt.assert_array_almost_equal(da_normalized.values, expected_values, decimal=6)


def test_normalize_xarray_negative_values():
    """
    Ensure the function handles negative values correctly.
    """
    data = np.array([-5, 0, 5])
    da = xr.DataArray(data, dims="x")
    da_normalized = utils.normalize_xarray(da, low=0, high=1)
    expected_values = np.array([0.0, 0.5, 1.0])
    npt.assert_array_almost_equal(da_normalized.values, expected_values, decimal=6)


def test_normalize_xarray_custom_range():
    """
    Ensure the function handles a custom low and high range correctly.
    """
    data = np.array([10, 20, 30])
    da = xr.DataArray(data, dims="x")
    da_normalized = utils.normalize_xarray(da, low=-1, high=1)
    expected_values = np.array([-1.0, 0.0, 1.0])
    npt.assert_array_almost_equal(da_normalized.values, expected_values, decimal=6)


def test_normalized_mindist_defaults():
    """
    test the normalized_mindist function
    """
    # create 2 constraint points
    points = pd.DataFrame({"x": [0, 2], "y": [0, -1]})
    # create dataarray
    df = pd.DataFrame(
        {"x": [-4, 4, 0, 4, -4], "y": [-4, 4, 0, -4, 4], "z": [0, 1, 2, 3, 4]}
    )
    da = df.set_index(["y", "x"]).to_xarray().z
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
    points = pd.DataFrame({"x": [0, 2], "y": [0, -1]})
    # create dataarray
    df = pd.DataFrame(
        {"x": [-4, 4, 0, 4, -4], "y": [-4, 4, 0, -4, 4], "z": [0, 1, 2, 3, 4]}
    )
    da = df.set_index(["y", "x"]).to_xarray().z

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
    points = pd.DataFrame({"x": [0, 2], "y": [0, -1]})
    # create dataarray
    df = pd.DataFrame(
        {"x": [-4, 4, 0, 4, -4], "y": [-4, 4, 0, -4, 4], "z": [0, 1, 2, 3, 4]}
    )
    da = df.set_index(["y", "x"]).to_xarray().z
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
    points = pd.DataFrame({"x": [0, 2], "y": [0, -1]})
    # create dataarray
    df = pd.DataFrame(
        {"x": [-4, 4, 0, 4, -4], "y": [-4, 4, 0, -4, 4], "z": [0, 1, 2, 3, 4]}
    )
    da = df.set_index(["y", "x"]).to_xarray().z
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


def test_sample_grids_on_nodes():
    """
    Test if the sampled column contains valid values at grid nodes.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    df = pd.DataFrame({"x": [0, 100, 200], "y": [200, 300, 400]})
    result_df = utils.sample_grids(df, grid, sampled_name=name)
    expected = pd.DataFrame(
        {"x": [0, 100, 200], "y": [200, 300, 400], name: [40000, 100000, 200000]}
    )
    pdt.assert_frame_equal(result_df, expected)


def test_sample_grids_off_nodes():
    """
    Test if the sampled column contains valid values not on grid nodes.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    df = pd.DataFrame({"x": [50, 101], "y": [280, 355]})
    result_df = utils.sample_grids(df, grid, sampled_name=name)
    expected = pd.DataFrame(
        {"x": [50, 101], "y": [280, 355], name: [83790.0, 138949.640109]}
    )
    pdt.assert_frame_equal(result_df, expected)


def test_sample_grids_custom_coordinate_names():
    """
    Test if the function handles custom coordinate names correctly.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    df = pd.DataFrame({"lon": [0, 100, 200], "lat": [200, 300, 400]})
    # check function raises KeyError if coordinate names are not found in the grid
    with pytest.raises(KeyError):
        utils.sample_grids(df, grid, sampled_name=name)
    # check function works if coordinate names are provided
    result_df = utils.sample_grids(
        df, grid, sampled_name=name, coord_names=("lon", "lat")
    )
    expected = pd.DataFrame(
        {"lon": [0, 100, 200], "lat": [200, 300, 400], name: [40000, 100000, 200000]}
    )
    pdt.assert_frame_equal(result_df, expected)


def test_sample_grids_one_out_of_grid_coordinates():
    """
    Test if the function handles out-of-grid coordinates gracefully by setting NaN
    values.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    df = pd.DataFrame({"x": [0, -1000, 200], "y": [200, -1000, 400]})
    result_df = utils.sample_grids(df, grid, sampled_name=name)
    assert np.isnan(result_df[name].iloc[1])


def test_sample_grids_first_out_of_grid_coordinates():
    """
    Test if the function handles out-of-grid coordinates gracefully by setting NaN
    values.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    df = pd.DataFrame({"x": [-50, 150, 200], "y": [500, 350, 400]})
    result_df = utils.sample_grids(df, grid, sampled_name=name)
    assert np.isnan(result_df[name].iloc[0])


def test_sample_grids_last_out_of_grid_coordinates():
    """
    Test if the function handles out-of-grid coordinates gracefully by setting NaN
    values.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    df = pd.DataFrame({"x": [200, 150, 0], "y": [200, 350, 0]})
    result_df = utils.sample_grids(df, grid, sampled_name=name)
    assert np.isnan(result_df[name].iloc[2])


def test_sample_grids_all_out_of_grid_coordinates_all():
    """
    Test if the function handles out-of-grid coordinates gracefully by setting NaN
    values.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    points = pd.DataFrame({"x": [-100, -200, -300], "y": [500, 1000, 600]})
    result_df = utils.sample_grids(points, grid, sampled_name=name)
    assert result_df[name].isnull().all()  # All values should be NaN


def test_extract_prism_data():
    """
    Test if function for checking prisms and extract data works properly
    """
    prism_layer = dummy_prism_layer()
    results = utils.extract_prism_data(prism_layer)
    prisms_df, prisms_ds, density_contrast, zref, spacing, topo_grid = results
    assert density_contrast == 2670
    assert zref == 0
    assert spacing == 200
    expected = np.array([[0.0, 0.0, 0.0], [-30.0, -30.0, -30.0], [30.0, 30.0, 30.0]])
    npt.assert_array_equal(expected, topo_grid)
    npt.assert_array_equal(expected, prisms_ds.starting_topo)


def test_get_spacing():
    """
    Test that the correct spacing is extracts from a data frame which represents a
    harmonica prism layer.
    """
    prisms_df = dummy_prism_layer().to_dataframe().reset_index().dropna().astype(float)
    assert utils.get_spacing(prisms_df) == 200


def test_sample_bounding_surfaces_valid_values():
    """
    Ensure that correct values are sampled, including a NaN
    """
    lower_confining_layer = dummy_grid().scalars
    points = pd.DataFrame({"x": [0, -100, 200], "y": [200, -300, 400]})
    result_df = utils.sample_grids(
        points,
        lower_confining_layer,
        sampled_name="sampled",
    )
    expected = pd.DataFrame(
        {"x": [0, -100, 200], "y": [200, -300, 400], "sampled": [40000, np.nan, 200000]}
    )
    pdt.assert_frame_equal(result_df, expected)


def test_sample_bounding_surfaces_upper_and_lower():
    """
    Ensure that when both upper and lower layers are provided, both are sampled.
    """
    points = pd.DataFrame({"easting": [0, -100, 200], "northing": [200, -300, 400]})
    upper_confining_layer = dummy_grid().upward
    lower_confining_layer = dummy_grid().scalars
    result_df = utils.sample_bounding_surfaces(
        points, upper_confining_layer, lower_confining_layer
    )
    assert "upper_bounds" in result_df.columns
    assert "lower_bounds" in result_df.columns


def test_sample_bounding_surfaces_upper_only():
    """
    Ensure that when only the upper layer is provided, it is sampled.
    """
    points = pd.DataFrame({"easting": [0, -100, 200], "northing": [200, -300, 400]})
    upper_confining_layer = dummy_grid().upward
    result_df = utils.sample_bounding_surfaces(
        points, upper_confining_layer=upper_confining_layer
    )
    assert "upper_bounds" in result_df.columns
    assert "lower_bounds" not in result_df.columns


def test_sample_bounding_surfaces_lower_only():
    """
    Ensure that when only the lower layer is provided, it is sampled.
    """
    points = pd.DataFrame({"easting": [0, -100, 200], "northing": [200, -300, 400]})
    lower_confining_layer = dummy_grid().scalars
    result_df = utils.sample_bounding_surfaces(
        points, lower_confining_layer=lower_confining_layer
    )
    assert "upper_bounds" not in result_df.columns
    assert "lower_bounds" in result_df.columns


def test_sample_bounding_surfaces_no_layers():
    """
    Ensure that when neither upper nor lower layers are provided, the dataframe remains
    unchanged.
    """
    points = pd.DataFrame({"easting": [0, -100, 200], "northing": [200, -300, 400]})
    result_df = utils.sample_bounding_surfaces(points)
    assert "upper_bounds" not in result_df.columns
    assert "lower_bounds" not in result_df.columns


def test_enforce_confining_surface_none():
    """
    Test that supplying no confining surfaces is correctly handled
    """
    prisms_ds = dummy_prism_layer()
    topo_grid = xr.where(prisms_ds.density > 0, prisms_ds.top, prisms_ds.bottom)
    prisms_ds["topo"] = topo_grid
    prisms_df = prisms_ds.to_dataframe().reset_index().dropna().astype(float)
    prisms_df["iter_3_correction"] = [20.0] * 9
    enforced = utils.enforce_confining_surface(prisms_df, iteration_number=3)
    pdt.assert_frame_equal(prisms_df, enforced)


def test_enforce_confining_surface_upper():
    """
    Test that supplying an upper surface works correctly
    """
    prisms_ds = dummy_prism_layer()
    topo_grid = xr.where(prisms_ds.density > 0, prisms_ds.top, prisms_ds.bottom)
    prisms_ds["topo"] = topo_grid
    prisms_df = prisms_ds.to_dataframe().reset_index().dropna().astype(float)
    prisms_df["upper_bounds"] = [40.0] * 9
    prisms_df["iter_3_correction"] = [20.0] * 9
    prisms_df["initial_correction"] = prisms_df.iter_3_correction
    enforced = utils.enforce_confining_surface(prisms_df, iteration_number=3)
    # print(enforced)
    # check max change above is correct
    expected = pd.Series(
        [40.0, 40.0, 40.0, 70.0, 70.0, 70.0, 10.0, 10.0, 10.0], name="max_change_above"
    )
    pdt.assert_series_equal(expected, enforced.max_change_above)
    # check that constrained correction values are correct
    expected = pd.Series(
        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0], name="iter_3_correction"
    )
    pdt.assert_series_equal(expected, enforced.iter_3_correction)


def test_enforce_confining_surface_upper_crosses():
    """
    Test that supplying an upper surface works correctly if it crosses the topography
    """
    prisms_ds = dummy_prism_layer()
    topo_grid = xr.where(prisms_ds.density > 0, prisms_ds.top, prisms_ds.bottom)
    prisms_ds["topo"] = topo_grid
    prisms_df = prisms_ds.to_dataframe().reset_index().dropna().astype(float)
    prisms_df["upper_bounds"] = [0.0] * 9
    prisms_df["iter_3_correction"] = [40.0, 0.0, -40.0] * 3
    prisms_df["initial_correction"] = prisms_df.iter_3_correction
    enforced = utils.enforce_confining_surface(prisms_df, iteration_number=3)
    # print(enforced)
    # check max change above is correct
    expected = pd.Series(
        [0.0, 0.0, 0.0, 30.0, 30.0, 30.0, -30.0, -30.0, -30.0], name="max_change_above"
    )
    pdt.assert_series_equal(expected, enforced.max_change_above)
    # check that constrained correction values are correct
    expected = pd.Series(
        [0.0, 0.0, -40.0, 30.0, 0.0, -40.0, -30.0, -30.0, -40.0],
        name="iter_3_correction",
    )
    pdt.assert_series_equal(expected, enforced.iter_3_correction)


def test_enforce_confining_surface_lower():
    """
    Test that supplying a lower surface works correctly
    """
    prisms_ds = dummy_prism_layer()
    topo_grid = xr.where(prisms_ds.density > 0, prisms_ds.top, prisms_ds.bottom)
    prisms_ds["topo"] = topo_grid
    prisms_df = prisms_ds.to_dataframe().reset_index().dropna().astype(float)
    prisms_df["lower_bounds"] = [-40.0] * 9
    prisms_df["iter_3_correction"] = [-20.0] * 9
    prisms_df["initial_correction"] = prisms_df.iter_3_correction
    enforced = utils.enforce_confining_surface(prisms_df, iteration_number=3)
    # print(enforced)
    # check max change below is correct
    expected = pd.Series(
        [-40.0, -40.0, -40.0, -10.0, -10.0, -10.0, -70.0, -70.0, -70.0],
        name="max_change_below",
    )
    pdt.assert_series_equal(expected, enforced.max_change_below)
    # check that constrained correction values are correct
    expected = pd.Series(
        [-20.0, -20.0, -20.0, -10.0, -10.0, -10.0, -20.0, -20.0, -20.0],
        name="iter_3_correction",
    )
    pdt.assert_series_equal(expected, enforced.iter_3_correction)


def test_enforce_confining_surface_lower_crosses():
    """
    Test that supplying a lower surface works correctly if it crosses the topography
    """
    prisms_ds = dummy_prism_layer()
    topo_grid = xr.where(prisms_ds.density > 0, prisms_ds.top, prisms_ds.bottom)
    prisms_ds["topo"] = topo_grid
    prisms_df = prisms_ds.to_dataframe().reset_index().dropna().astype(float)
    prisms_df["lower_bounds"] = [0.0] * 9
    prisms_df["iter_3_correction"] = [40.0, 0.0, -40.0] * 3
    prisms_df["initial_correction"] = prisms_df.iter_3_correction
    enforced = utils.enforce_confining_surface(prisms_df, iteration_number=3)
    # print(enforced)
    # check max change below is correct
    expected = pd.Series(
        [0.0, 0.0, 0.0, 30.0, 30.0, 30.0, -30.0, -30.0, -30.0], name="max_change_below"
    )
    pdt.assert_series_equal(expected, enforced.max_change_below)
    # check that constrained correction values are correct
    expected = pd.Series(
        [40.0, 0.0, 0.0, 40.0, 30.0, 30.0, 40.0, 0.0, -30.0], name="iter_3_correction"
    )
    pdt.assert_series_equal(expected, enforced.iter_3_correction)


def test_enforce_confining_surface_both():
    """
    Test that supplying both an upper and lower surface works correctly
    """
    prisms_ds = dummy_prism_layer()
    topo_grid = xr.where(prisms_ds.density > 0, prisms_ds.top, prisms_ds.bottom)
    prisms_ds["topo"] = topo_grid
    prisms_df = prisms_ds.to_dataframe().reset_index().dropna().astype(float)
    prisms_df["lower_bounds"] = [-20.0] * 9
    prisms_df["upper_bounds"] = [20.0] * 9
    prisms_df["iter_3_correction"] = [40.0, 0.0, -40.0] * 3
    prisms_df["initial_correction"] = prisms_df.iter_3_correction
    enforced = utils.enforce_confining_surface(prisms_df, iteration_number=3)
    print(enforced)
    # check max change below is correct
    expected = pd.Series(
        [-20.0, -20.0, -20.0, 10.0, 10.0, 10.0, -50.0, -50.0, -50.0],
        name="max_change_below",
    )
    pdt.assert_series_equal(expected, enforced.max_change_below)
    # check max change above is correct
    expected = pd.Series(
        [20.0, 20.0, 20.0, 50.0, 50.0, 50.0, -10.0, -10.0, -10.0],
        name="max_change_above",
    )
    pdt.assert_series_equal(expected, enforced.max_change_above)
    # check that constrained correction values are correct
    expected = pd.Series(
        [20.0, 0.0, -20.0, 40.0, 10.0, 10.0, -10.0, -10.0, -40.0],
        name="iter_3_correction",
    )
    pdt.assert_series_equal(expected, enforced.iter_3_correction)


def test_apply_surface_correction():
    pass


def test_update_prisms_df():
    pass


def test_add_updated_prism_properties():
    pass


################
################
# Not implemented yet
################
################


def test_constraints_grid():
    pass


def test_prep_grav_data():
    pass


def test_block_reduce_gravity():
    pass


def test_grids_to_prisms():
    pass


def test_forward_grav_of_prismlayer():
    pass


def test_constrain_surface_correction():
    pass
