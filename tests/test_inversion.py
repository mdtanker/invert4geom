import pathlib
import pickle

import harmonica as hm
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import verde as vd
import xarray as xr

import invert4geom

pd.set_option("display.max_columns", None)

################
################
# functions used for tests
################
################


def true_topography() -> xr.Dataset:
    easting = [0, 10000, 20000, 30000, 40000]
    northing = [0, 10000, 20000, 30000]
    surface = [
        [637, 545, 474, 434, 430],
        [494, 522, 448, 407, 435],
        [646, 302, 486, 483, 443],
        [718, 639, 439, 545, 541],
    ]
    return vd.make_xarray_grid(
        (easting, northing),
        data=surface,
        data_names="upward",
    )


def flat_topography_500m() -> xr.Dataset:
    topo = true_topography()
    topo["upward"] = xr.full_like(topo.upward, 500)
    return topo


def observed_gravity() -> xr.Dataset:
    easting = [0, 10000, 20000, 30000, 40000]
    northing = [0, 10000, 20000, 30000]
    grav = [
        [14.26481884, 4.71750172, -2.58265577, -6.88771746, -7.3176891],
        [-0.22250109, 2.30457796, -5.20320139, -9.73491892, -7.06766808],
        [15.05282739, -19.32882461, -2.17347104, -1.94021662, -6.00456422],
        [22.84735344, 14.25307194, -6.03071986, 4.63738305, 4.47834945],
    ]

    return vd.make_xarray_grid(
        (easting, northing),
        data=(grav, np.full_like(grav, 1000)),
        data_names=("gravity_anomaly", "upward"),
    )


################
################
# TESTS
################
################


def test_data_attributes():
    """
    test the data attributes are properly set
    """
    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )

    attrs = {
        "region": (0.0, 40000.0, 0.0, 30000.0),
        "spacing": 10000.0,
        "buffer_width": 10000.0,
        "inner_region": (10000.0, 30000.0, 10000.0, 20000.0),
        "dataset_type": "data",
        "model_type": "prisms",
    }
    assert grav_data.attrs == attrs


@pytest.mark.filterwarnings("ignore:Grid may have irregular spacing in the")
@pytest.mark.filterwarnings("ignore:grid zmax can't be extracted")
def test_model_attributes():
    """
    test the model attributes are properly set
    """
    model = true_topography()

    # add mask to dataset
    model["mask"] = xr.where(model.upward > 600, 1, np.nan)

    model = invert4geom.inversion.create_model(
        starting_topography=model, zref=100, density_contrast=200
    )

    attrs = {
        "inner_region": (0.0, 10000.0, 0.0, 30000.0),
        "zref": 100,
        "density_contrast": 200,
        "region": (0.0, 40000.0, 0.0, 30000.0),
        "spacing": 10000.0,
        "dataset_type": "model",
        "model_type": "prisms",
    }

    assert model.attrs == attrs


def test_inv_accessor_df():
    """
    test the inv accessor .df property
    """
    data = [
        [0.00000000e00, 0.00000000e00, 1.42648188e01, 1.00000000e03],
        [0.00000000e00, 1.00000000e04, 4.71750172e00, 1.00000000e03],
        [0.00000000e00, 2.00000000e04, -2.58265577e00, 1.00000000e03],
        [0.00000000e00, 3.00000000e04, -6.88771746e00, 1.00000000e03],
        [0.00000000e00, 4.00000000e04, -7.31768910e00, 1.00000000e03],
        [1.00000000e04, 0.00000000e00, -2.22501090e-01, 1.00000000e03],
        [1.00000000e04, 1.00000000e04, 2.30457796e00, 1.00000000e03],
        [1.00000000e04, 2.00000000e04, -5.20320139e00, 1.00000000e03],
        [1.00000000e04, 3.00000000e04, -9.73491892e00, 1.00000000e03],
        [1.00000000e04, 4.00000000e04, -7.06766808e00, 1.00000000e03],
        [2.00000000e04, 0.00000000e00, 1.50528274e01, 1.00000000e03],
        [2.00000000e04, 1.00000000e04, -1.93288246e01, 1.00000000e03],
        [2.00000000e04, 2.00000000e04, -2.17347104e00, 1.00000000e03],
        [2.00000000e04, 3.00000000e04, -1.94021662e00, 1.00000000e03],
        [2.00000000e04, 4.00000000e04, -6.00456422e00, 1.00000000e03],
        [3.00000000e04, 0.00000000e00, 2.28473534e01, 1.00000000e03],
        [3.00000000e04, 1.00000000e04, 1.42530719e01, 1.00000000e03],
        [3.00000000e04, 2.00000000e04, -6.03071986e00, 1.00000000e03],
        [3.00000000e04, 3.00000000e04, 4.63738305e00, 1.00000000e03],
        [3.00000000e04, 4.00000000e04, 4.47834945e00, 1.00000000e03],
    ]
    df = pd.DataFrame(
        data, columns=["northing", "easting", "gravity_anomaly", "upward"]
    )

    grav_data = invert4geom.inversion.create_data(observed_gravity())

    pd.testing.assert_frame_equal(grav_data.inv.df, df, check_dtype=False)


def test_inv_accessor_inner_df():
    """
    test the inv accessor .df property
    """
    data = [
        [1.00000000e04, 1.00000000e04, 2.30457796e00, 1.00000000e03],
        [1.00000000e04, 2.00000000e04, -5.20320139e00, 1.00000000e03],
        [1.00000000e04, 3.00000000e04, -9.73491892e00, 1.00000000e03],
        [2.00000000e04, 1.00000000e04, -1.93288246e01, 1.00000000e03],
        [2.00000000e04, 2.00000000e04, -2.17347104e00, 1.00000000e03],
        [2.00000000e04, 3.00000000e04, -1.94021662e00, 1.00000000e03],
    ]
    df = pd.DataFrame(
        data, columns=["northing", "easting", "gravity_anomaly", "upward"]
    )

    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )

    pd.testing.assert_frame_equal(grav_data.inv.inner_df, df, check_dtype=False)


def test_inv_accessor_inner():
    """
    test the inv accessor .inner property
    """
    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )

    easting = [10000, 20000, 30000]
    northing = [10000, 20000]
    grav = [
        [2.30457796, -5.20320139, -9.73491892],
        [-19.32882461, -2.17347104, -1.94021662],
    ]

    true_ds = vd.make_xarray_grid(
        (easting, northing),
        data=(grav, np.full_like(grav, 1000)),
        data_names=("gravity_anomaly", "upward"),
    )
    xr.testing.assert_equal(grav_data.inv.inner, true_ds)


def test_inv_accessor_masked_df():
    """
    test the inv accessor .masked property
    """
    model = true_topography()
    # add mask to dataset
    model["mask"] = xr.where(model.upward > 600, 1, np.nan)

    model = invert4geom.inversion.create_model(
        starting_topography=model, zref=100, density_contrast=200
    )

    topo = [637.0, 646.0, 718.0, 639.0]

    npt.assert_array_equal(topo, model.inv.masked_df.topography.to_numpy())

    # check error is raised if called for data object
    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )
    with pytest.raises(
        ValueError, match="Method is only available for the model dataset"
    ):
        _ = grav_data.inv.masked_df


def test_inv_accessor_masked():
    """
    test the inv accessor .masked property
    """
    model = true_topography()
    # add mask to dataset
    model["mask"] = xr.where(model.upward > 600, 1, np.nan)

    model = invert4geom.inversion.create_model(
        starting_topography=model, zref=100, density_contrast=200
    )

    topo = [[637.0, np.nan], [646.0, np.nan], [718.0, 639.0]]

    npt.assert_array_equal(topo, model.inv.masked.topography.values)

    # check error is raised if called for data object
    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )
    with pytest.raises(
        ValueError, match="Method is only available for the model dataset"
    ):
        _ = grav_data.inv.masked


def test_forward_gravity():
    """
    test the forward gravity method
    """
    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )
    model = invert4geom.inversion.create_model(
        starting_topography=flat_topography_500m(), zref=100, density_contrast=200
    )

    grav_data.inv.forward_gravity(model, name="test_forward_gravity")

    expected = np.array(
        [
            3.08686619,
            3.15942294,
            3.16671307,
            3.15942294,
            3.08686619,
            3.15754725,
            3.24517211,
            3.25452486,
            3.24517211,
            3.15754725,
            3.15754725,
            3.24517211,
            3.25452486,
            3.24517211,
            3.15754725,
            3.08686619,
            3.15942294,
            3.16671307,
            3.15942294,
            3.08686619,
        ]
    )

    npt.assert_allclose(
        grav_data.inv.df.test_forward_gravity.to_numpy(), expected, rtol=1e-5
    )

    # check error is raised if called for model object
    with pytest.raises(
        ValueError, match="Method is only available for the data dataset"
    ):
        model.inv.forward_gravity(model)

    model = model.assign_attrs(model_type="no_prism")
    with pytest.raises(ValueError, match="layer must have attribute 'model_type'"):
        grav_data.inv.forward_gravity(model)


def test_forward_gravity_rename():
    """
    test the forward gravity method is able to update an existing column
    """
    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )
    model = invert4geom.inversion.create_model(
        starting_topography=flat_topography_500m(), zref=100, density_contrast=200
    )

    grav_data.inv.forward_gravity(model, name="gravity_anomaly")

    expected = np.array(
        [
            3.08686619,
            3.15942294,
            3.16671307,
            3.15942294,
            3.08686619,
            3.15754725,
            3.24517211,
            3.25452486,
            3.24517211,
            3.15754725,
            3.15754725,
            3.24517211,
            3.25452486,
            3.24517211,
            3.15754725,
            3.08686619,
            3.15942294,
            3.16671307,
            3.15942294,
            3.08686619,
        ]
    )

    npt.assert_allclose(
        grav_data.inv.df.gravity_anomaly.to_numpy(), expected, rtol=1e-5
    )


def test_regional_separation():
    """
    test the regional separation method
    """
    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )
    model = invert4geom.inversion.create_model(
        starting_topography=flat_topography_500m(), zref=100, density_contrast=200
    )
    grav_data.inv.forward_gravity(model)
    grav_data.inv.regional_separation(
        method="constant",
        constant=10,
    )

    assert "reg" in grav_data.inv.df.columns
    assert "res" in grav_data.inv.df.columns
    assert "misfit" in grav_data.inv.df.columns
    assert grav_data.inv.df.reg.eq(10).all()

    # check error is raised if called for model object
    with pytest.raises(
        ValueError, match="Method is only available for the data dataset"
    ):
        model.inv.regional_separation(
            method="constant",
            constant=10,
        )


def test_check_grav_vars_for_regional():
    """
    test an error is raised if gravity dataset missing variables needed for regional
    """
    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )

    with pytest.raises(
        AssertionError, match="`gravity dataset` needs all the following variables"
    ):
        grav_data.inv._check_grav_vars_for_regional()

    # check error is raised if called for model object
    model = invert4geom.inversion.create_model(
        starting_topography=flat_topography_500m(), zref=100, density_contrast=200
    )
    with pytest.raises(
        ValueError, match="Method is only available for the data dataset"
    ):
        model.inv._check_grav_vars_for_regional()


def test_check_grav_vars():
    """
    test an error is raised if gravity dataset missing variables needed for inversion
    """
    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )

    with pytest.raises(
        AssertionError, match="`gravity dataset` needs all the following variables"
    ):
        grav_data.inv._check_grav_vars()

    # check error is raised if called for model object
    model = invert4geom.inversion.create_model(
        starting_topography=flat_topography_500m(), zref=100, density_contrast=200
    )
    with pytest.raises(
        ValueError, match="Method is only available for the data dataset"
    ):
        model.inv._check_grav_vars()


def test_check_gravity_inside_topography_region():
    """
    test an error is raised if gravity is outside topography region method
    """
    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )

    topo = flat_topography_500m()

    # shift topo to east by 100km
    topo = topo.assign_coords(easting=topo.easting + 100000)

    with pytest.raises(
        ValueError, match="Some gravity data are outside the region of the topography"
    ):
        grav_data.inv._check_gravity_inside_topography_region(topo)

    # check error is raised if called for model object
    model = invert4geom.inversion.create_model(
        starting_topography=topo, zref=100, density_contrast=200
    )
    with pytest.raises(
        ValueError, match="Method is only available for the data dataset"
    ):
        model.inv._check_gravity_inside_topography_region(topo)


def test_update_gravity_and_residual():
    """
    test gravity variables 'res' and 'forward_gravity' are correctly updated with a new prism layer
    """
    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )
    model = invert4geom.inversion.create_model(
        starting_topography=flat_topography_500m(), zref=100, density_contrast=200
    )
    grav_data.inv.forward_gravity(model)
    grav_data.inv.regional_separation(
        method="constant",
        constant=10,
    )

    # save current values for testing
    res = grav_data.inv.df.res.to_numpy()
    forward_gravity = grav_data.inv.df.forward_gravity.to_numpy()

    # replace values to see if they get updated
    grav_data["res"] = xr.full_like(grav_data.res, 100)
    grav_data["forward_gravity"] = xr.full_like(grav_data.forward_gravity, 100)

    # update the dataset with the model
    grav_data.inv._update_gravity_and_residual(model)

    npt.assert_equal(grav_data.inv.df.res.to_numpy(), res)
    npt.assert_equal(grav_data.inv.df.forward_gravity.to_numpy(), forward_gravity)

    # check error is raised if called for model object
    with pytest.raises(
        ValueError, match="Method is only available for the data dataset"
    ):
        model.inv._update_gravity_and_residual(model)


def test_add_topography_correction():
    """
    test the surface correction values are correctly added to the model dataset
    """
    model = invert4geom.inversion.create_model(
        starting_topography=flat_topography_500m(),
        zref=100,
        density_contrast=200,
    )
    # set surface correction to be 10 m
    step = np.full_like(model.inv.df.topography.to_numpy(), 10)

    # update the dataset with the surface corrections
    model = model.inv.add_topography_correction(step)

    assert "topography_correction" in model.inv.df.columns
    npt.assert_array_equal(model.inv.df.topography_correction.to_numpy(), step)

    # check error is raised if called for data object
    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )
    with pytest.raises(
        ValueError, match="Method is only available for the model dataset"
    ):
        grav_data.inv.add_topography_correction(step)


def test_add_topography_correction_negated_density():
    """
    test the surface correction values are correctly added to the model dataset with
    positive and negative densities
    """
    model = invert4geom.inversion.create_model(
        starting_topography=true_topography(),
        zref=500,  # this is in middle of topography
        density_contrast=200,
    )
    # set surface correction to be 10 m
    step = np.full_like(model.inv.df.topography.to_numpy(), 10)

    # update the dataset with the surface corrections
    model = model.inv.add_topography_correction(step)

    assert "topography_correction" in model.inv.df.columns
    npt.assert_array_equal(np.abs(model.inv.df.topography_correction.to_numpy()), step)


def test_add_topography_correction_confining_layers():
    """
    test the surface correction values are correctly added to the model dataset with
    confining layers
    """
    # make model confined below at 500 and above at 600 m
    model = invert4geom.inversion.create_model(
        starting_topography=true_topography(),
        zref=500,
        density_contrast=200,
        upper_confining_layer=xr.full_like(flat_topography_500m().upward, 600),
        lower_confining_layer=xr.full_like(flat_topography_500m().upward, 500),
    )
    # set surface correction to be 200 m, which would move the surface outside of both
    # confining layers
    step = np.full_like(model.inv.df.topography.to_numpy(), 200)

    # update the dataset with the surface corrections
    model = model.inv.add_topography_correction(step)

    true_step = [
        [-37, 55, 26, 66, 70],
        [6, 78, 52, 93, 65],
        [-46, 198, 14, 17, 57],
        [-118, -39, 61, 55, 59],
    ]

    assert "topography_correction" in model.inv.df.columns
    npt.assert_array_equal(model.topography_correction.to_numpy(), true_step)


def test_update_model_ds():
    """
    test the model is update with the surface correction values
    """
    model = invert4geom.inversion.create_model(
        starting_topography=flat_topography_500m(),
        zref=100,
        density_contrast=200,
    )
    # set surface correction to be 10 m
    step = np.full_like(model.inv.df.topography.to_numpy(), 10)

    # update the dataset with the surface corrections and update the model
    model = model.inv.add_topography_correction(step)
    model = model.inv.update_model_ds()

    # test the topography is raised by 10m
    npt.assert_array_equal(
        model.topography.to_numpy(), (flat_topography_500m().upward + 10).to_numpy()
    )

    # test the top values are raised by 10m
    npt.assert_array_equal(model.top, flat_topography_500m().upward + 10)

    # check error is raised if called for data object
    grav_data = invert4geom.inversion.create_data(
        observed_gravity(), buffer_width=10000
    )
    with pytest.raises(
        ValueError, match="Method is only available for the model dataset"
    ):
        grav_data.inv.update_model_ds()


def test_create_data_variable_name_error_raised():
    """
    test the create_data function correctly raises an error
    """
    with pytest.raises(AssertionError, match="gravity dataset needs variables"):
        invert4geom.inversion.create_data(
            observed_gravity().rename({"upward": "not_upward"}),
        )


def test_create_data_coord_name_error_raised():
    """
    test the create_data function with prisms correctly raises an error
    """
    with pytest.raises(AssertionError, match="gravity dataset must have dims"):
        invert4geom.inversion.create_data(
            observed_gravity().rename({"easting": "not_easting"}),
        )


def test_create_data_buffer_spacing_error():
    """
    test the create_data function raises an error with buffer zone not being a multiple of spacing
    """
    with pytest.raises(
        AssertionError,
        match=r"buffer_width \(1111\) must be a multiple of the grid spacing",
    ):
        invert4geom.inversion.create_data(
            observed_gravity(),
            buffer_width=1111,
        )


def test_create_data_large_buffer_error():
    """
    test the create_data function raises an error with buffer zone being too big
    """
    with pytest.raises(
        AssertionError,
        match="buffer_width must be smaller than half the smallest dimension of the region",
    ):
        invert4geom.inversion.create_data(
            observed_gravity(),
            buffer_width=1000e3,
        )


def test_create_model():
    """
    test the create_model function works correctly
    """
    with pytest.raises(
        ValueError, match="model_type must be either 'prisms' or 'tesseroids'"
    ):
        invert4geom.inversion.create_model(
            starting_topography=flat_topography_500m(),
            zref=100,
            density_contrast=200,
            model_type="not_prisms_or_tesseroids",
        )

    with pytest.raises(
        ValueError, match=r"`density\_contrast` must be a float or xarray.DataArray"
    ):
        invert4geom.inversion.create_model(
            starting_topography=flat_topography_500m(),
            zref=100,
            density_contrast=flat_topography_500m(),
        )

    with pytest.raises(AssertionError, match="density DataArray must have dims"):
        invert4geom.inversion.create_model(
            starting_topography=flat_topography_500m(),
            zref=100,
            density_contrast=flat_topography_500m().upward.rename(
                {"easting": "not_easting"}
            ),
        )


def test_create_model_variable_name_error_raised():
    """
    test the create_model function correctly raises an error
    """
    with pytest.raises(
        AssertionError,
        match="starting_topography Dataset must contain an 'upward' variable",
    ):
        invert4geom.inversion.create_model(
            0,
            2700,
            true_topography().rename({"upward": "not_upward"}),
        )


def test_create_model_coord_name_error_raised():
    """
    test the create_model function with prisms correctly raises an error
    """
    with pytest.raises(
        AssertionError,
        match=r"topography DataArray must have dims \('easting', 'northing'\)",
    ):
        invert4geom.inversion.create_model(
            0,
            2700,
            true_topography().rename({"easting": "not_easting"}),
        )


def test_inversion_properties():
    """
    test the inversion properties work correctly
    """
    data = invert4geom.inversion.create_data(observed_gravity())
    model = invert4geom.inversion.create_model(500, 2669, flat_topography_500m())

    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=10,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        max_iterations=1,
    )
    rmse = inv.rmse
    l2_norm = inv.l2_norm

    inv.invert(progressbar=False)

    assert inv.already_inverted is True
    assert rmse != inv.rmse
    assert l2_norm != inv.l2_norm

    assert inv.rmse == np.sqrt(np.mean(inv.data.inv.inner.res**2))
    assert inv.l2_norm == np.sqrt(np.sqrt(np.mean(inv.data.inv.inner.res**2)))
    assert inv.delta_l2_norm == l2_norm / np.sqrt(
        np.sqrt(np.mean(inv.data.inv.inner.res**2))
    )


def test_end_inversion_l2_norm_increasing():
    """
    test the end_inversion method works correctly
    """
    data = invert4geom.inversion.create_data(observed_gravity())
    model = invert4geom.inversion.create_model(500, 2669, flat_topography_500m())

    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=10,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        perc_increase_limit=0.1,  # 10%
    )
    inv.iteration = 2
    stats_df = pd.DataFrame(
        data={
            "iteration": [0, 1, 2],
            "rmse": [10, 10, 10],
            "l2_norm": [1, 1.08, 1.11],
            "delta_l2_norm": [np.inf, np.inf, np.inf],
            "iter_time_sec": [np.nan, np.nan, np.nan],
        }
    )
    inv.stats_df = stats_df
    inv.end_inversion()
    assert inv.end is True
    assert inv.termination_reason == ["l2-norm increasing"]

    inv.perc_increase_limit = 0.2
    inv.end_inversion()
    assert inv.end is False


def test_end_inversion_delta_l2_norm_tolerance_both_below():
    """
    test the end_inversion method works correctly
    """
    data = invert4geom.inversion.create_data(observed_gravity())
    model = invert4geom.inversion.create_model(500, 2669, flat_topography_500m())

    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=10,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        delta_l2_norm_tolerance=1.001,
    )
    inv.iteration = 2
    stats_df = pd.DataFrame(
        data={
            "iteration": [0, 1, 2],
            "rmse": [10, 10, 10],
            "l2_norm": [1, 1, 1],
            "delta_l2_norm": [np.inf, 1.0001, 1],
        }
    )
    inv.stats_df = stats_df

    inv.end_inversion()

    assert inv.end is True
    assert inv.termination_reason == ["delta l2-norm tolerance"]


def test_end_inversion_delta_l2_norm_tolerance_one_below():
    """
    test the end_inversion method works correctly
    """
    data = invert4geom.inversion.create_data(observed_gravity())
    model = invert4geom.inversion.create_model(500, 2669, flat_topography_500m())

    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=10,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        delta_l2_norm_tolerance=1.001,
    )
    inv.iteration = 2
    stats_df = pd.DataFrame(
        data={
            "iteration": [0, 1, 2],
            "rmse": [10, 10, 10],
            "l2_norm": [1, 1, 1],
            "delta_l2_norm": [np.inf, 1.1, 1],
        }
    )
    inv.stats_df = stats_df

    inv.end_inversion()

    assert inv.end is False


def test_end_inversion_l2_norm_tolerance():
    """
    test the end_inversion method works correctly
    """
    data = invert4geom.inversion.create_data(observed_gravity())
    model = invert4geom.inversion.create_model(500, 2669, flat_topography_500m())

    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=10,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        l2_norm_tolerance=1,
    )
    inv.iteration = 2
    stats_df = pd.DataFrame(
        data={
            "iteration": [0, 1, 2],
            "rmse": [10, 10, 10],
            "l2_norm": [1, 1, 0.9],
            "delta_l2_norm": [np.inf, np.inf, np.inf],
        }
    )
    inv.stats_df = stats_df
    inv.end_inversion()
    assert inv.end is True
    assert inv.termination_reason == ["l2-norm tolerance"]

    inv.l2_norm_tolerance = 0.8
    inv.end_inversion()
    assert inv.end is False


def test_end_inversion_max_iterations():
    """
    test the end_inversion method works correctly
    """
    data = invert4geom.inversion.create_data(observed_gravity())
    model = invert4geom.inversion.create_model(500, 2669, flat_topography_500m())

    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=10,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        max_iterations=2,
    )
    inv.iteration = 2
    stats_df = pd.DataFrame(
        data={
            "iteration": [0, 1, 2],
            "rmse": [10, 10, 10],
            "l2_norm": [1, 1, 1],
            "delta_l2_norm": [np.inf, np.inf, np.inf],
        }
    )
    inv.stats_df = stats_df

    inv.end_inversion()
    assert inv.end is True
    assert inv.termination_reason == ["max iterations"]

    inv.max_iterations = 3
    inv.end_inversion()
    assert inv.end is False


def test_end_inversion_multiple_reasons():
    """
    test the end_inversion method works correctly
    """
    data = invert4geom.inversion.create_data(observed_gravity())
    model = invert4geom.inversion.create_model(500, 2669, flat_topography_500m())

    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=10,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        max_iterations=2,
        l2_norm_tolerance=1,
        delta_l2_norm_tolerance=1.001,
    )
    inv.iteration = 2
    stats_df = pd.DataFrame(
        data={
            "iteration": [0, 1, 2],
            "rmse": [10, 10, 10],
            "l2_norm": [1, 1, 0.9],
            "delta_l2_norm": [np.inf, 1.0001, 1],
        }
    )
    inv.stats_df = stats_df

    inv.end_inversion()
    assert inv.end is True
    assert inv.termination_reason == [
        "delta l2-norm tolerance",
        "l2-norm tolerance",
        "max iterations",
    ]


def test_model_properties():
    """
    test the _model_properties function
    """
    model = invert4geom.inversion.create_model(
        starting_topography=flat_topography_500m(), zref=100, density_contrast=200
    )
    itertools_result = invert4geom.inversion._model_properties(
        model, method="itertools"
    )
    forloops_result = invert4geom.inversion._model_properties(model, method="forloops")
    generator_result = invert4geom.inversion._model_properties(
        model, method="generator"
    )
    # test that the prism properties are the same with 3 methods
    np.array_equal(itertools_result, forloops_result)
    np.array_equal(itertools_result, generator_result)
    # test that the first prism's properties are correct
    np.array_equal(itertools_result[0], np.array([-300, -100, 0, 200, -100, 2670]))

    with pytest.raises(ValueError, match="method must be"):
        invert4geom.inversion._model_properties(model, method="wrong_input")


@pytest.mark.use_numba
def test_jacobian_error_raised():
    """
    test the jacobian method raises the correct errors
    """
    # create 2x2 topography grid
    easting = [0, 5]
    northing = [0, 5]
    surface = [
        [100, 100],
        [100, 100],
    ]
    topo = vd.make_xarray_grid(
        (easting, northing),
        data=surface,
        data_names="upward",
    )

    # create 2x2 gravity dataset
    grav = vd.make_xarray_grid(
        (easting, northing),
        data=([[10, 10], [10, 10]], [[100, 100], [100, 100]]),
        data_names=("gravity_anomaly", "upward"),
    )

    data = invert4geom.inversion.create_data(grav)
    model = invert4geom.inversion.create_model(-1e3, 10000, topo)
    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=0,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        deriv_type="annulus",
    )

    with pytest.raises(
        ValueError,
        match="All prism tops coincides exactly with the elevation of the gravity",
    ):
        inv.jacobian()

    topo = xr.full_like(topo.upward, 80).to_dataset(name="upward")
    model = invert4geom.inversion.create_model(-1e3, 10000, topo)
    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        deriv_type="not_annulus_or_finite_difference",
    )

    with pytest.raises(ValueError, match="invalid string for deriv_type"):
        inv.jacobian()


@pytest.mark.use_numba
def test_jacobian_annulus():
    """
    test the jacobian method works correctly
    """
    # create 2x2 topography grid
    easting = [0, 5]
    northing = [0, 5]
    surface = [
        [80, 80],
        [100, 100],
    ]
    topo = vd.make_xarray_grid(
        (easting, northing),
        data=surface,
        data_names="upward",
    )

    # create 2x2 gravity dataset
    grav = vd.make_xarray_grid(
        (easting, northing),
        data=([[10, 10], [10, 10]], [[120, 140], [120, 140]]),
        data_names=("gravity_anomaly", "upward"),
    )
    # import matplotlib.pyplot as plt
    # _, axs = plt.subplots(1, 2, figsize=(8, 4))
    # topo.upward.plot(ax=axs[0])
    # grav.upward.plot(ax=axs[1])

    data = invert4geom.inversion.create_data(grav)
    model = invert4geom.inversion.create_model(-1e3, 10000, topo)
    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=0,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        deriv_type="annulus",
    )
    # create jacobian, will be 4x4 matrix, with a row for each grav point and a column
    # for each prism
    # so index (0,0) is the sensitivity of the first gravity point to the first prism
    # and index (3,3) is the sensitivity of the last gravity point to the last prism
    inv.jacobian()

    # check shape is correct
    assert np.shape(inv.jac) == (4, 4)

    # pairs which are same distance apart should have same sensitivity
    assert inv.jac[0, 0] == pytest.approx(inv.jac[3, 3])
    assert inv.jac[0, 1] == pytest.approx(inv.jac[2, 0])
    assert inv.jac[0, 1] == pytest.approx(inv.jac[3, 2])
    assert inv.jac[0, 1] == pytest.approx(inv.jac[1, 3])
    assert inv.jac[0, 2] == pytest.approx(inv.jac[2, 3])
    assert inv.jac[1, 0] == pytest.approx(inv.jac[3, 1])
    assert inv.jac[1, 2] == pytest.approx(inv.jac[2, 1])

    # highest sensitivity pair should be prism 2, grav 2
    assert np.argmax(inv.jac) == 10

    # lowest sensitivity pair should be prism 0, grav 3
    assert np.argmin(inv.jac) == 12


@pytest.mark.use_numba
def test_jacobian_finite_difference():
    """
    test the jacobian method works correctly
    """
    # create 2x2 topography grid
    easting = [0, 5]
    northing = [0, 5]
    surface = [
        [80, 80],
        [100, 100],
    ]
    topo = vd.make_xarray_grid(
        (easting, northing),
        data=surface,
        data_names="upward",
    )

    # create 2x2 gravity dataset
    grav = vd.make_xarray_grid(
        (easting, northing),
        data=([[10, 10], [10, 10]], [[120, 140], [120, 140]]),
        data_names=("gravity_anomaly", "upward"),
    )

    data = invert4geom.inversion.create_data(grav)
    model = invert4geom.inversion.create_model(-1e3, 10000, topo)
    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=0,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        deriv_type="finite_difference",
    )
    # create jacobian, will be 4x4 matrix, with a row for each grav point and a column
    # for each prism
    # so index (0,0) is the sensitivity of the first gravity point to the first prism
    # and index (3,3) is the sensitivity of the last gravity point to the last prism
    inv.jacobian()

    # check shape is correct
    assert np.shape(inv.jac) == (4, 4)

    # pairs which are same distance apart should have same sensitivity
    assert inv.jac[0, 0] == pytest.approx(inv.jac[3, 3])
    assert inv.jac[0, 1] == pytest.approx(inv.jac[2, 0])
    assert inv.jac[0, 1] == pytest.approx(inv.jac[3, 2])
    assert inv.jac[0, 1] == pytest.approx(inv.jac[1, 3])
    assert inv.jac[0, 2] == pytest.approx(inv.jac[2, 3])
    assert inv.jac[1, 0] == pytest.approx(inv.jac[3, 1])
    assert inv.jac[1, 2] == pytest.approx(inv.jac[2, 1])

    # highest sensitivity pair should be prism 2, grav 2
    assert np.argmax(inv.jac) == 10

    # lowest sensitivity pair should be prism 0, grav 3
    assert np.argmin(inv.jac) == 12


@pytest.mark.use_numba
def test_solver():
    """
    test the solver
    """
    # create 2x2 topography grid
    easting = [0, 5]
    northing = [0, 5]
    surface = [
        [0, 0],
        [0, 0],
    ]
    topo = vd.make_xarray_grid(
        (easting, northing),
        data=surface,
        data_names="upward",
    )

    # create 2x2 gravity dataset
    grav = vd.make_xarray_grid(
        (easting, northing),
        data=([[1, 1], [-1, -1]], [[100, 100], [100, 100]]),
        data_names=("gravity_anomaly", "upward"),
    )

    data = invert4geom.inversion.create_data(grav)
    model = invert4geom.inversion.create_model(0, 2670, topo)
    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=0,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        solver_damping=0.01,
    )

    inv.jacobian()

    inv.solver()

    # check step shape is correct
    assert np.shape(inv.step) == (4,)

    # check step sign matches sign of residual
    assert np.sign(inv.step[0]) == np.sign(data.inv.df.res.to_numpy()[0])

    # check approx correct values
    # doesn't work cause of damping
    # bouguer slab formula
    # height = grav_anom (in mGal) / (0.42 * density_contrast (in g/cm3))
    # 1 / (0.042 * 2.670) = 9 m
    # print(inv.step)

    inv.solver_type = "not_least_squares"
    with pytest.raises(ValueError, match="invalid string for solver_type"):
        inv.solver()


def test_reinitialize_inversion():
    """
    test the reinitialize inversion function
    """
    # create 6x6 topography grid
    easting = [0, 50, 100, 150, 200, 250]
    northing = [0, 50, 100, 150, 200, 250]
    surface = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    topo = vd.make_xarray_grid(
        (easting, northing),
        data=surface,
        data_names="upward",
    )

    # create 6x6 gravity dataset
    grav_vals = [
        [10, 10, 10, 10, 10, 10],
        [10, 10, 9.5, 9.5, 10, 10],
        [10, 9.5, 9, 9, 9.5, 10],
        [10, 9.5, 9, 9, 9.5, 10],
        [10, 10, 9.5, 9.5, 10, 10],
        [10, 10, 10, 10, 10, 10],
    ]

    grav = vd.make_xarray_grid(
        (easting, northing),
        data=(grav_vals, np.full_like(surface, 100)),
        data_names=("gravity_anomaly", "upward"),
    )

    data = invert4geom.inversion.create_data(grav)
    model = invert4geom.inversion.create_model(0, 2670, topo)
    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=10,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        max_iterations=2,
        solver_damping=0.01,
        l2_norm_tolerance=0.12,
        delta_l2_norm_tolerance=1.001,
    )

    inv.invert(
        progressbar=False,
    )

    with pytest.raises(
        ValueError,
        match="this inversion object has already been used to run an inversion",
    ):
        inv.invert(
            progressbar=False,
        )

    inv.reinitialize_inversion()

    assert inv.iteration is None
    # pd.testing.assert_series_equal(inv.data.inv.df.misfit, data.inv.df.misfit)
    # pd.testing.assert_series_equal(inv.data.inv.df.res, data.inv.df.res)
    # pd.testing.assert_series_equal(inv.data.inv.df.reg, data.inv.df.reg)
    # assert not any("iter_" in var for var in inv.data.variables)
    # npt.assert_equal(inv.model.topography.to_numpy(), model.topography.to_numpy())
    xr.testing.assert_equal(inv.model, model)
    xr.testing.assert_equal(inv.data, data)

    # try to rerun inversion
    inv.invert(
        progressbar=False,
    )

    # check topography decreased in centre where residual gravity was negative
    inner_topo = inv.model.topography.sel(
        northing=slice(100, 150), easting=slice(100, 150)
    )
    assert np.mean(inner_topo) < 0


def test_invert_annulus():
    """
    test the inversion
    """
    # create 6x6 topography grid
    easting = [0, 50, 100, 150, 200, 250]
    northing = [0, 50, 100, 150, 200, 250]
    surface = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    topo = vd.make_xarray_grid(
        (easting, northing),
        data=surface,
        data_names="upward",
    )

    # create 6x6 gravity dataset
    grav_vals = [
        [10, 10, 10, 10, 10, 10],
        [10, 10, 9.5, 9.5, 10, 10],
        [10, 9.5, 9, 9, 9.5, 10],
        [10, 9.5, 9, 9, 9.5, 10],
        [10, 10, 9.5, 9.5, 10, 10],
        [10, 10, 10, 10, 10, 10],
    ]
    # grav_vals = np.full_like(surface, -1)

    grav = vd.make_xarray_grid(
        (easting, northing),
        data=(grav_vals, np.full_like(surface, 100)),
        data_names=("gravity_anomaly", "upward"),
    )

    data = invert4geom.inversion.create_data(grav)
    model = invert4geom.inversion.create_model(0, 2670, topo)
    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=10,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        solver_damping=0.01,
        l2_norm_tolerance=0.12,
        delta_l2_norm_tolerance=1.001,
        max_iterations=10,
        deriv_type="annulus",
    )

    inv.invert(
        progressbar=False,
    )

    # inv.plot_inversion_results()
    # print(inv.stats_df)

    # check residual decreased
    # assert inv.stats_df.rmse.iloc[-1] < inv.stats_df.rmse.iloc[0]

    # check topography decreased in centre where residual gravity was negative
    inner_topo = inv.model.topography.sel(
        northing=slice(100, 150), easting=slice(100, 150)
    )
    assert np.mean(inner_topo) < 0

    assert inv.deriv_type == "annulus"
    # gravity anomaly of 1 mGal and density contrast of 2670 kg/m3 should result in a
    # topography change of approx 9 m from bouguer slab formula
    # 1 / (0.042 * 2.670) = 9 m
    # print(np.mean(inner_topo))
    # assert np.mean(inner_topo) == pytest.approx(-0.9, abs=0.5)


def test_invert_finite_difference():
    """
    test the inversion
    """
    # create 6x6 topography grid
    easting = [0, 50, 100, 150, 200, 250]
    northing = [0, 50, 100, 150, 200, 250]
    surface = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    topo = vd.make_xarray_grid(
        (easting, northing),
        data=surface,
        data_names="upward",
    )

    # create 6x6 gravity dataset
    grav_vals = [
        [10, 10, 10, 10, 10, 10],
        [10, 10, 9.5, 9.5, 10, 10],
        [10, 9.5, 9, 9, 9.5, 10],
        [10, 9.5, 9, 9, 9.5, 10],
        [10, 10, 9.5, 9.5, 10, 10],
        [10, 10, 10, 10, 10, 10],
    ]
    # grav_vals = np.full_like(surface, -1)

    grav = vd.make_xarray_grid(
        (easting, northing),
        data=(grav_vals, np.full_like(surface, 100)),
        data_names=("gravity_anomaly", "upward"),
    )

    data = invert4geom.inversion.create_data(grav)
    model = invert4geom.inversion.create_model(0, 2670, topo)
    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=10,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        solver_damping=0.01,
        l2_norm_tolerance=0.12,
        delta_l2_norm_tolerance=1.001,
        max_iterations=10,
        deriv_type="finite_difference",
    )

    inv.invert(
        progressbar=False,
    )

    # inv.plot_inversion_results()
    # print(inv.stats_df)

    # check residual decreased
    # assert inv.stats_df.rmse.iloc[-1] < inv.stats_df.rmse.iloc[0]

    # check topography decreased in centre where residual gravity was negative
    inner_topo = inv.model.topography.sel(
        northing=slice(100, 150), easting=slice(100, 150)
    )
    assert np.mean(inner_topo) < 0

    assert inv.deriv_type == "finite_difference"
    # gravity anomaly of 1 mGal and density contrast of 2670 kg/m3 should result in a
    # topography change of approx 9 m from bouguer slab formula
    # 1 / (0.042 * 2.670) = 9 m
    # print(np.mean(inner_topo))
    # assert np.mean(inner_topo) == pytest.approx(-0.9, abs=0.5)


def test_invert_weighting():
    """
    test the inversion with a weighting grid
    """
    # create 6x6 topography grid
    easting = [0, 50, 100, 150, 200, 250]
    northing = [0, 50, 100, 150, 200, 250]
    surface = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    topo = vd.make_xarray_grid(
        (easting, northing),
        data=surface,
        data_names="upward",
    )

    # create 6x6 gravity dataset
    grav_vals = [
        [10, 10, 10, 10, 10, 10],
        [10, 10, 9.5, 9.5, 10, 10],
        [10, 9.5, 9, 9, 9.5, 10],
        [10, 9.5, 9, 9, 9.5, 10],
        [10, 10, 9.5, 9.5, 10, 10],
        [10, 10, 10, 10, 10, 10],
    ]

    grav = vd.make_xarray_grid(
        (easting, northing),
        data=(grav_vals, np.full_like(surface, 100)),
        data_names=("gravity_anomaly", "upward"),
    )

    data = invert4geom.inversion.create_data(grav)
    model = invert4geom.inversion.create_model(0, 2670, topo)
    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=10,
    )
    weighting_grid = vd.make_xarray_grid(
        (easting, northing),
        data=[
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [
                1,
                1,
                0,
                1,
                1,
                1,
            ],  # weight of zero at cell (100,100) so topography here shouldn't change
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        data_names="weight",
    ).weight
    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        solver_damping=0.01,
        l2_norm_tolerance=0.12,
        delta_l2_norm_tolerance=1.001,
        max_iterations=10,
        apply_weighting_grid=False,
        weighting_grid=weighting_grid,
    )
    with pytest.raises(
        ValueError,
        match="weighting grid supplied but not used because apply_weighting_grid is False",
    ):
        inv.invert(
            progressbar=False,
        )

    inv.weighting_grid = None
    inv.apply_weighting_grid = True
    with pytest.raises(
        ValueError, match="must supply weighting grid if apply_weighting_grid is True"
    ):
        inv.invert(progressbar=False)

    inv.weighting_grid = weighting_grid
    inv.invert(
        progressbar=False,
    )
    # check topography decreased in centre where residual gravity was negative
    inner_topo = inv.model.topography.sel(
        northing=slice(100, 150), easting=slice(100, 150)
    )
    assert np.mean(inner_topo) < 0

    # check the topography didn't change at the cell that is constrained with the weighting grid
    assert inv.model.topography.sel(northing=100, easting=100).item() == 0


def test_invert_pickle(tmp_path):
    """
    test the invert function saving results to a pickle file
    """
    # Use tmp_path to create a temporary directory
    temp_dir = tmp_path / "test_dir"
    temp_dir.mkdir()

    temp_file = temp_dir / "test_invert"
    # temp_file = "test_invert"

    data = invert4geom.inversion.create_data(observed_gravity(), buffer_width=10000)
    model = invert4geom.inversion.create_model(
        starting_topography=flat_topography_500m(), zref=100, density_contrast=200
    )
    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=10,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        max_iterations=2,
        solver_damping=0.01,
        l2_norm_tolerance=0.12,
        delta_l2_norm_tolerance=1.001,
    )

    inv.invert(progressbar=False, results_fname=temp_file)

    assert pathlib.Path(f"{temp_file}.pickle").exists()

    # load the pickle file
    with pathlib.Path(f"{temp_file}.pickle").open("rb") as f:
        inversion_results = pickle.load(f)

    xr.testing.assert_equal(inv.model, inversion_results.model)
    xr.testing.assert_equal(inv.data, inversion_results.data)
    npt.assert_equal(inversion_results.params, inv.params)

    # delete the file
    pathlib.Path(f"{temp_file}.pickle").unlink()


def test_invert_with_confining_layers():
    """
    test the inversion with confining layers
    """
    data = invert4geom.inversion.create_data(observed_gravity(), buffer_width=10000)
    model = invert4geom.inversion.create_model(
        starting_topography=flat_topography_500m(),
        zref=100,
        density_contrast=200,
        upper_confining_layer=xr.full_like(flat_topography_500m().upward, 510),
        lower_confining_layer=xr.full_like(flat_topography_500m().upward, 500),
    )
    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=0,
    )

    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        max_iterations=2,
        solver_damping=0.01,
        l2_norm_tolerance=0.12,
        delta_l2_norm_tolerance=1.001,
    )

    inv.invert(progressbar=False)

    assert inv.params["Upper confining layer"] == "Enabled"  # type: ignore[index]
    assert inv.params["Lower confining layer"] == "Enabled"  # type: ignore[index]

    assert np.min(inv.model.topography) >= 500
    assert np.max(inv.model.topography) <= 510


def test_gravity_score():
    """
    test the gravity_score function
    """
    data = invert4geom.inversion.create_data(observed_gravity())
    model = invert4geom.inversion.create_model(500, 2669, flat_topography_500m())
    resampled = invert4geom.cross_validation.add_test_points(data)
    resampled.inv.forward_gravity(model)
    resampled.inv.regional_separation(
        method="constant",
        constant=10,
    )
    inv = invert4geom.inversion.Inversion(
        data=resampled,
        model=model,
        solver_damping=0.01,
        max_iterations=2,
    )
    _ = inv.gravity_score()

    assert inv.gravity_best_score == pytest.approx(32.04, 0.01)

    xr.testing.assert_equal(inv.model, model)
    xr.testing.assert_equal(inv.data, resampled)


@pytest.mark.filterwarnings("ignore:QMCSampler is experimental")
@pytest.mark.filterwarnings("ignore:GPSampler is experimental")
def test_optimize_inversion_damping():
    """
    test the optimize_inversion_damping function
    """
    data = invert4geom.inversion.create_data(observed_gravity())
    model = invert4geom.inversion.create_model(500, 2669, flat_topography_500m())
    resampled = invert4geom.cross_validation.add_test_points(data)
    resampled.inv.forward_gravity(model)
    resampled.inv.regional_separation(
        method="constant",
        constant=0,
    )
    inv = invert4geom.inversion.Inversion(
        data=resampled,
        model=model,
        max_iterations=10,
    )

    damping_opti_obj = inv.optimize_inversion_damping(
        damping_limits=(0.001, 10),
        n_trials=5,
        grid_search=False,
        plot_scores=False,
        progressbar=False,
        fname="test_damping",
    )

    assert damping_opti_obj.best_trial.params["damping"] == inv.solver_damping  # type: ignore[attr-defined]
    assert inv.solver_damping == inv.params["Solver damping"]  # type: ignore[index]
    assert inv.solver_damping == pytest.approx(0.25, rel=0.01)

    assert pathlib.Path("test_damping.pickle").exists()
    assert pathlib.Path("test_damping_study.pickle").exists()

    # delete files
    pathlib.Path("test_damping.pickle").unlink()
    pathlib.Path("test_damping_study.pickle").unlink()

    inv = invert4geom.inversion.Inversion(
        data=resampled,
        model=model,
        max_iterations=10,
    )

    damping_opti_obj = inv.optimize_inversion_damping(
        damping_limits=(0.001, 10),
        n_trials=5,
        grid_search=True,
        plot_scores=False,
        progressbar=False,
        fname="test_damping2",
    )
    assert damping_opti_obj.best_trial.params["damping"] == inv.solver_damping  # type: ignore[attr-defined]
    assert inv.solver_damping == inv.params["Solver damping"]  # type: ignore[index]
    assert inv.solver_damping == 0.1
    dampings = np.logspace(np.log10(0.001), np.log10(10), 5)
    assert inv.solver_damping in dampings

    # delete files
    pathlib.Path("test_damping2.pickle").unlink()
    pathlib.Path("test_damping2_study.pickle").unlink()


def test_constraints_score():
    """
    test the constraints_score function
    """
    data = invert4geom.inversion.create_data(observed_gravity())
    model = invert4geom.inversion.create_model(500, 2669, flat_topography_500m())
    constraints_df = pd.DataFrame(
        data={
            "easting": [0, 1000],
            "northing": [0, 1000],
            "upward": [500, 500],
        }
    )
    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=10,
    )
    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        solver_damping=0.01,
        max_iterations=2,
    )
    _ = inv.constraints_score(constraints_df)

    assert inv.constraints_best_score == pytest.approx(102.84, 0.01)

    xr.testing.assert_equal(inv.model, model)
    xr.testing.assert_equal(inv.data, data)


@pytest.mark.filterwarnings("ignore:QMCSampler is experimental")
@pytest.mark.filterwarnings("ignore:GPSampler is experimental")
def test_optimize_inversion_zref_density_contrast():
    """
    test the optimize_inversion_zref_density_contrast function
    """
    data = invert4geom.inversion.create_data(observed_gravity())
    model = invert4geom.inversion.create_model(500, 2669, flat_topography_500m())
    constraints_df = pd.DataFrame(
        data={
            "easting": [0, 1000],
            "northing": [0, 1000],
            "upward": [500, 500],
        }
    )
    data.inv.forward_gravity(model)
    data.inv.regional_separation(
        method="constant",
        constant=0,
    )
    inv = invert4geom.inversion.Inversion(
        data=data,
        model=model,
        max_iterations=10,
    )

    opti_obj = inv.optimize_inversion_zref_density_contrast(
        zref_limits=(0, 2e3),
        density_contrast_limits=(1000, 3000),
        n_trials=5,
        constraints_df=constraints_df,
        grid_search=False,
        plot_scores=False,
        progressbar=False,
        regional_grav_kwargs={"method": "constant", "constant": 0},
        starting_topography_kwargs={
            "method": "flat",
            "region": inv.model.region,
            "spacing": inv.model.spacing,
        },
        fname="test_zref_density_contrast",
    )

    assert opti_obj.best_trial.params["zref"] == inv.model.zref  # type: ignore[attr-defined]
    assert inv.model.zref == float(inv.params["Reference level"][:-2])  # type: ignore[index]
    assert inv.model.zref == pytest.approx(698.24, rel=0.01)

    assert opti_obj.best_trial.params["density_contrast"] == inv.model.density_contrast  # type: ignore[attr-defined]
    assert inv.model.density_contrast == float(inv.params["Density contrast(s)"][1:-7])  # type: ignore[index]
    assert inv.model.density_contrast == pytest.approx(1774, rel=0.01)

    assert pathlib.Path("test_zref_density_contrast.pickle").exists()
    assert pathlib.Path("test_zref_density_contrast_study.pickle").exists()

    # delete files
    pathlib.Path("test_zref_density_contrast.pickle").unlink()
    pathlib.Path("test_zref_density_contrast_study.pickle").unlink()


@pytest.mark.use_numba
def test_grav_column_der_relative_values():
    """
    test the grav_column_der function
    Below is a map view of a prism, with the locations of the various observation
    points a-h. The prism is 5x5, with a density of 2670kg/m^3.

    5  d---g---c
       |       |
       |   e   f      h
       |       |
    0  a-------b
       0       5
    """
    a = invert4geom.inversion.grav_column_der(
        grav_easting=0,
        grav_northing=0,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    b = invert4geom.inversion.grav_column_der(
        grav_easting=5,
        grav_northing=0,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    c = invert4geom.inversion.grav_column_der(
        grav_easting=5,
        grav_northing=5,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    d = invert4geom.inversion.grav_column_der(
        grav_easting=0,
        grav_northing=5,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    e = invert4geom.inversion.grav_column_der(
        grav_easting=2.5,
        grav_northing=2.5,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    f = invert4geom.inversion.grav_column_der(
        grav_easting=5,
        grav_northing=2.5,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    g = invert4geom.inversion.grav_column_der(
        grav_easting=2.5,
        grav_northing=5,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    h = invert4geom.inversion.grav_column_der(
        grav_easting=10,
        grav_northing=2.5,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    # test that derivative at all 4 corners of prism is same
    assert a == b == c == d
    # test that derivate on 2 prism edges are the same
    assert f == g
    # test that derivate within prism is same as on the edge
    assert e == f
    # test that derivative further away from prism is smaller
    assert h < a


@pytest.mark.use_numba
def test_grav_column_der():
    """
    test the grav_column_der function against a small prism approximation
    """
    # expected result
    dg_z = invert4geom.inversion.grav_column_der(
        grav_easting=20,
        grav_northing=20,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )[0]
    # harmonica prism approximation
    step = 1e-3
    hm_dg_z = (
        hm.prism_gravity(
            coordinates=(20, 20, 100),
            prisms=(0, 5, 0, 5, -10, -10 + step),
            density=2670,
            field="g_z",
        )
        / step
    )
    # test that the derivative matches a small prism approximation from
    assert dg_z == pytest.approx(hm_dg_z, rel=1e-2)


# @pytest.mark.use_numba
# def test_jacobian_annular():
#     """
#     test the jacobian_annular function
#     """
#     grav_df = dummy_grav_ds().to_dataframe().reset_index()

#     model = dummy_prism_layer()

#     prisms_df = model.inv.df

#     # prisms_df = prisms_layer.to_dataframe().reset_index().dropna().astype(float)
#     jac = np.empty(
#         (len(grav_df), model.top.size),
#         dtype=np.float64,
#     )
#     jac = inversion.jacobian_annular(
#         np.array(grav_df.easting),
#         np.array(grav_df.northing),
#         np.array(grav_df.upward),
#         np.array(prisms_df.easting),
#         np.array(prisms_df.northing),
#         np.array(prisms_df.top),
#         np.array(prisms_df.density),
#         model.spacing,
#         jac,
#     )
#     # test that prisms above observation point have negative vertical derivatives
#     assert jac[:, -3:].max() < 0
#     # test that prisms below observation point have positive vertical derivatives
#     assert jac[:, 0:-3].min() > 0


# solver_types = [
#     "scipy least squares",
#     # "verde least squares",
#     # "scipy constrained",
#     # "scipy nonlinear lsqr",
#     # "CLR",
#     # "scipy conjugate",
#     # "numpy least squares",
#     # "steepest descent", # off by 2 orders of magnitude
#     # "gauss newton",
# ]


# @pytest.mark.use_numba
# @pytest.mark.parametrize("solver_type", solver_types)
# def test_solver_square(solver_type):
#     """
#     test the solver function with equal number of prisms and misfit values
#     """
#     misfit = dummy_grav_ds_big().misfit.to_numpy()
#     jac = dummy_jacobian_square()
#     correction = inversion.solver(jac, misfit, solver_type=solver_type)
#     # test that correction is negative for negative misfits
#     assert correction[0:3].max() < -9
#     # test that correction is near 0 for misfits with values of 0
#     npt.assert_allclose(correction[3:6], np.array([0, 0, 0]), atol=1e-8)
#     # test that correction is positive for positive misfits
#     assert correction[6:9].min() > 9


# solver_types = [
#     # "verde least squares", # step is not negative where it needs to be
#     "scipy least squares",
#     "scipy constrained",
#     # "scipy nonlinear lsqr",
#     # "CLR",
#     # "scipy conjugate",
#     "numpy least squares",
#     "steepest descent",
#     # "gauss newton",
# ]


# @pytest.mark.parametrize("solver_type", solver_types)
# def test_solver_underdetermined(solver_type):
#     """
#     test the solver function
#     flat prisms surface and base, all obs points above surface, consistent misfit
#     values, should result if relatively uniform corrections
#     """
#     jac = dummy_jacobian()
#     # test that correction is 0 if misfits are 0
#     misfit = np.array([0, 0, 0, 0])
#     correction = inversion.solver(jac.copy(), misfit, solver_type=solver_type)
#     assert correction.all() == 0
#     # test that all corrections are negative
#     misfit = np.array([-10, -10, -10, -10])
#     correction = inversion.solver(jac.copy(), misfit, solver_type=solver_type)
#     assert correction.max() < 0
#     # test that all corrections are positive
#     misfit = np.array([100, 100, 100, 100])
#     correction = inversion.solver(jac.copy(), misfit, solver_type=solver_type)
#     assert correction.min() > 0
#     # test that mean correction is close to 0
#     misfit = np.array([-100, -100, 100, 100])
#     correction = inversion.solver(jac.copy(), misfit, solver_type=solver_type)
#     assert correction.mean() == pytest.approx(0, abs=1e-5)
