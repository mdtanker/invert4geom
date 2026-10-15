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
        "splines",
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
        dampings=1e-3,
    )

    # ptk.grid_compare(
    #     ds.reg,
    #     ds.misfit,
    #     points=constraints,
    #     grid1_name="reg",
    #     grid2_name="misfit",
    # )

    if test_input == "splines":
        expected = [
            [-135.44952408, -90.62970577, -49.74718528, -10.56501417, 28.13953372],
            [-101.7301447, -57.61095466, -17.42773067, 21.22809215, 59.99595849],
            [-71.62873659, -27.97973888, 12.32070216, 50.85930792, 90.09736661],
            [-42.99386229, 0.62934746, 41.46463324, 80.69403906, 120.59519551],
        ]
    elif test_input == "pygmt":
        # the misfit is an exact quadratic, so sampling it at the 4 constraint points
        # gives 4 exact values which define an exact plane
        expected = [
            [-130.0, -90.0, -50.0, -10.0, 30.0],
            [-100.0, -60.0, -20.0, 20.0, 60.0],
            [-70.0, -30.0, 10.0, 50.0, 90.0],
            [-40.0, 0.0, 40.0, 80.0, 120.0],
        ]
    elif test_input == "eq_sources":
        expected = [
            [-125.05856164, -89.9359067, -50.52268429, -9.91940642, 28.64402296],
            [-98.71694027, -61.55645115, -20.67000183, 20.66122323, 59.10819473],
            [-69.06555311, -30.75182347, 10.53390083, 51.46585091, 88.75958188],
            [-38.51209675, -0.07852322, 40.48042529, 79.93797706, 115.19048784],
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


def grid_point_constraints() -> pd.DataFrame:
    """3 constraints on grid nodes and 1 between them"""
    return pd.DataFrame(
        data={
            "easting": [0.0, 40000.0, 0.0, 15000.0],
            "northing": [0.0, 0.0, 30000.0, 15000.0],
            "upward": [500.0, 500.0, 500.0, 500.0],
            "grid_points": [True, True, True, False],
        }
    )


@pytest.mark.filterwarnings("ignore:The mindist parameter of verde.Spline")
@pytest.mark.filterwarnings("ignore:The default scoring will change")
@pytest.mark.filterwarnings("ignore:Cannot have number of splits")
@pytest.mark.filterwarnings("ignore:decreasing number of splits by 1")
@pytest.mark.filterwarnings("ignore:: sklearn.exceptions.UndefinedMetricWarning")
@pytest.mark.parametrize(
    "grid_method", ["pygmt", "splines", "nearest", "linear", "cubic"]
)
def test_regional_constraints_grid_points(grid_method):
    """
    the misfit at the constraints which fall on grid nodes should be used as the
    regional field directly, giving a residual of exactly zero there
    """
    grav_data = observed_gravity()
    constraints = grid_point_constraints()

    invert4geom.regional.regional_constraints(
        constraints_df=constraints,
        grav_ds=grav_data,
        grid_method=grid_method,
        grid_points_col="grid_points",
        dampings=1e-3 if grid_method == "splines" else None,
    )

    on_grid = constraints[constraints.grid_points]
    residuals = [
        float(grav_data.res.sel(easting=e, northing=n))
        for e, n in zip(on_grid.easting, on_grid.northing, strict=True)
    ]
    np.testing.assert_allclose(residuals, 0, atol=1e-8)
    if grid_method not in ("linear", "cubic"):
        # `linear` and `cubic` don't extrapolate outside of the supplied points
        assert not np.any(np.isnan(grav_data.reg.to_numpy()))


def test_regional_constraints_grid_points_with_masked_data():
    """
    a `mask` which excludes the cells of the on-grid constraints, as when the model
    holds those cells fixed, shouldn't stop them being used as the regional field
    """
    grav_data = observed_gravity()
    constraints = grid_point_constraints()

    # mask out the cells of the on-grid constraints, using the `create_model`
    # convention of nans marking the excluded cells
    on_grid = constraints[constraints.grid_points]
    mask = xr.ones_like(grav_data.upward)
    for e, n in zip(on_grid.easting, on_grid.northing, strict=True):
        mask.loc[{"easting": e, "northing": n}] = np.nan
    grav_data["mask"] = mask

    # the forward gravity is needed at every cell, including the masked-out ones
    grav_data.inv.forward_gravity(
        invert4geom.inversion.create_model(
            zref=0,
            density_contrast=2670,
            topography=grav_data[["upward"]] * 0 + 500,
        )
    )
    assert not np.isnan(grav_data.forward_gravity).any()

    invert4geom.regional.regional_constraints(
        constraints_df=constraints,
        grav_ds=grav_data,
        grid_method="pygmt",
        grid_points_col="grid_points",
    )

    # the masked-out cells are zeroed by the mask, which is what the on-grid
    # constraints give anyway
    residuals = [
        float(grav_data.res.sel(easting=e, northing=n))
        for e, n in zip(on_grid.easting, on_grid.northing, strict=True)
    ]
    np.testing.assert_allclose(residuals, 0, atol=1e-8)
    assert not np.isnan(grav_data.res).any()


def test_regional_constraints_grid_points_eq_sources_raises():
    with pytest.raises(ValueError, match="can't be used if `grid_method`"):
        invert4geom.regional.regional_constraints(
            constraints_df=grid_point_constraints(),
            grav_ds=observed_gravity(),
            grid_method="eq_sources",
            grid_points_col="grid_points",
        )


def test_regional_constraints_grid_points_missing_column_raises():
    with pytest.raises(ValueError, match="not in `constraints_df`"):
        invert4geom.regional.regional_constraints(
            constraints_df=grid_point_constraints(),
            grav_ds=observed_gravity(),
            grid_method="pygmt",
            grid_points_col="not_a_column",
        )


def test_regional_constraints_grid_points_none_true_raises():
    constraints = grid_point_constraints()
    constraints["grid_points"] = False
    with pytest.raises(ValueError, match="no constraint points have"):
        invert4geom.regional.regional_constraints(
            constraints_df=constraints,
            grav_ds=observed_gravity(),
            grid_method="pygmt",
            grid_points_col="grid_points",
        )


# %%

################
################
# shared behavior of the regional separation functions
################
################

# (method suffix, minimal kwargs) pairs for functions sharing common behavior
REGIONAL_METHODS = [
    ("constant", {"constant": -100.0}),
    ("trend", {"trend": 1}),
    ("filter", {"filter_width": 300e3}),
]


@pytest.mark.filterwarnings("ignore:dropping variables using `drop` is deprecated")
@pytest.mark.filterwarnings("ignore:Default ifft's behaviour")
@pytest.mark.parametrize(("method", "kwargs"), REGIONAL_METHODS)
def test_regional_and_residual_sum_to_misfit(method, kwargs):
    """the regional and residual components should always sum to the misfit"""
    grav_data = observed_gravity()
    func = getattr(invert4geom.regional, f"regional_{method}")
    func(grav_ds=grav_data, **kwargs)
    xr.testing.assert_allclose(grav_data.reg + grav_data.res, grav_data.misfit)


@pytest.mark.filterwarnings("ignore:dropping variables using `drop` is deprecated")
@pytest.mark.filterwarnings("ignore:Default ifft's behaviour")
@pytest.mark.parametrize(("method", "kwargs"), REGIONAL_METHODS)
def test_regional_shift_added_to_regional(method, kwargs):
    """regional_shift should shift the regional field by a constant"""
    func = getattr(invert4geom.regional, f"regional_{method}")

    grav_data = observed_gravity()
    func(grav_ds=grav_data, **kwargs)

    grav_data_shifted = observed_gravity()
    func(grav_ds=grav_data_shifted, regional_shift=10, **kwargs)

    xr.testing.assert_allclose(grav_data_shifted.reg, grav_data.reg + 10)


@pytest.mark.filterwarnings("ignore:dropping variables using `drop` is deprecated")
@pytest.mark.filterwarnings("ignore:Default ifft's behaviour")
@pytest.mark.parametrize(("method", "kwargs"), REGIONAL_METHODS)
def test_reverse_regional_residual_swaps_fields(method, kwargs):
    """reverse_regional_residual should swap the two calculated fields"""
    func = getattr(invert4geom.regional, f"regional_{method}")

    grav_data = observed_gravity()
    func(grav_ds=grav_data, **kwargs)

    grav_data_reversed = observed_gravity()
    func(grav_ds=grav_data_reversed, reverse_regional_residual=True, **kwargs)

    xr.testing.assert_allclose(grav_data_reversed.reg, grav_data.res)
    xr.testing.assert_allclose(grav_data_reversed.res, grav_data.reg)


def test_regional_mask_zeroes_residual():
    """a mask of zeros should zero the residual and put the misfit in the regional"""
    grav_data = observed_gravity()
    grav_data["mask"] = xr.zeros_like(grav_data.upward)

    invert4geom.regional.regional_constant(grav_ds=grav_data, constant=-100)

    assert (grav_data.res == 0).all()
    xr.testing.assert_allclose(grav_data.reg, grav_data.misfit)


def test_regional_mask_saves_res_mask():
    """the mask should be saved as `res_mask` for use in the convergence statistics"""
    grav_data = observed_gravity()
    grav_data["mask"] = xr.ones_like(grav_data.upward).where(
        grav_data.easting > grav_data.easting.mean(), 0.0
    )

    invert4geom.regional.regional_constant(grav_ds=grav_data, constant=-100)
    xr.testing.assert_allclose(grav_data.res_mask, grav_data.mask)

    # re-running without a mask should remove the now-stale mask
    no_mask = grav_data.drop_vars("mask")
    invert4geom.regional.regional_constant(grav_ds=no_mask, constant=-100)
    assert "res_mask" not in no_mask


def test_regional_mask_column_deprecated():
    """`mask_column` should warn but give the same result as a `mask` variable"""
    old_ds = observed_gravity()
    mask = xr.ones_like(old_ds.upward).where(old_ds.easting > 20000.0, 0.0)
    old_ds["my_mask"] = mask
    with pytest.warns(UserWarning, match="`mask_column` is deprecated"):
        invert4geom.regional.regional_constant(
            grav_ds=old_ds,
            constant=-100,
            mask_column="my_mask",
        )

    new_ds = observed_gravity()
    new_ds["mask"] = mask
    invert4geom.regional.regional_constant(grav_ds=new_ds, constant=-100)

    xr.testing.assert_allclose(old_ds.res, new_ds.res)
    xr.testing.assert_allclose(old_ds.res_mask, new_ds.res_mask)


def test_regional_mask_column_with_nans():
    """
    a mask using the `create_model` convention of nans marking the excluded cells
    should zero the residual there, not make it nan
    """
    grav_data = observed_gravity()
    grav_data["mask"] = xr.ones_like(grav_data.upward).where(
        grav_data.easting > grav_data.easting.mean()
    )

    invert4geom.regional.regional_constant(grav_ds=grav_data, constant=-100)

    excluded = grav_data.easting <= grav_data.easting.mean()
    assert not np.isnan(grav_data.res).any()
    assert (grav_data.res.where(excluded, 0) == 0).all()
    assert (grav_data.res_mask.where(excluded, 0) == 0).all()
    assert (grav_data.res.where(~excluded, 1) != 0).all()


def test_regional_mask_reversed_saves_no_res_mask():
    """after reversing, the mask applies to the regional field, not the residual"""
    grav_data = observed_gravity()
    grav_data["mask"] = xr.zeros_like(grav_data.upward)

    invert4geom.regional.regional_constant(
        grav_ds=grav_data,
        constant=-100,
        reverse_regional_residual=True,
    )
    assert (grav_data.reg == 0).all()
    assert "res_mask" not in grav_data


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("constant", {"constant": -100.0}),
        ("trend", {"trend": 1}),
        ("filter", {"filter_width": 300e3}),
        ("eq_sources", {}),
        ("constraints", {"constraints_df": pd.DataFrame()}),
    ],
)
def test_regional_functions_reject_non_dataset(method, kwargs):
    """passing a dataframe (the old API) should raise a deprecation error"""
    func = getattr(invert4geom.regional, f"regional_{method}")
    with pytest.raises(DeprecationWarning, match=f"regional_{method}"):
        func(grav_ds=pd.DataFrame(), **kwargs)


################
################
# input validation
################
################


def test_regional_constant_no_args_raises():
    grav_data = observed_gravity()
    with pytest.raises(ValueError, match="need to provide either"):
        invert4geom.regional.regional_constant(grav_ds=grav_data)


def test_regional_constant_both_args_raises():
    grav_data = observed_gravity()
    constraints = pd.DataFrame(
        {
            "easting": [10000.0],
            "northing": [10000.0],
            "upward": [500.0],
        }
    )
    with pytest.raises(ValueError, match="not used"):
        invert4geom.regional.regional_constant(
            grav_ds=grav_data,
            constant=-100,
            constraints_df=constraints,
        )


def scattered_constraints() -> pd.DataFrame:
    """several constraints per gravity cell, with weights"""
    rng = np.random.default_rng(seed=0)
    return pd.DataFrame(
        data={
            "easting": rng.uniform(0, 40000, 30),
            "northing": rng.uniform(0, 30000, 30),
            "upward": np.full(30, 500.0),
            "weights": np.full(30, 2.0),
            "reduce": np.tile([True, False], 15),
        }
    )


@pytest.mark.parametrize(
    ("deprecated", "new"),
    [
        ({"tension_factor": 0.5}, {"tension": 0.5}),
        ({"constraints_block_size": 15000}, {"block_size": 15000}),
        (
            {"constraints_block_size": 15000, "constraints_weights_column": "weights"},
            {"block_size": 15000, "weights_col": "weights"},
        ),
    ],
)
def test_regional_constraints_deprecated_parameters(deprecated, new):
    """the deprecated parameter names should warn but give the same result"""
    constraints = scattered_constraints()

    old_ds = observed_gravity()
    with pytest.warns(UserWarning, match="is deprecated"):
        invert4geom.regional.regional_constraints(
            constraints_df=constraints,
            grav_ds=old_ds,
            grid_method="pygmt",
            **deprecated,
        )

    new_ds = observed_gravity()
    invert4geom.regional.regional_constraints(
        constraints_df=constraints,
        grav_ds=new_ds,
        grid_method="pygmt",
        **new,
    )

    np.testing.assert_allclose(old_ds.reg.to_numpy(), new_ds.reg.to_numpy())


@pytest.mark.filterwarnings("ignore:The mindist parameter of verde.Spline")
@pytest.mark.filterwarnings("ignore:The default scoring will change")
@pytest.mark.filterwarnings("ignore:Cannot have number of splits")
@pytest.mark.filterwarnings("ignore:decreasing number of splits by 1")
def test_regional_constraints_deprecated_verde_grid_method():
    """`verde` should warn and give the same result as `splines`"""
    constraints = scattered_constraints()

    old_ds = observed_gravity()
    with pytest.warns(UserWarning, match="'verde' is deprecated"):
        invert4geom.regional.regional_constraints(
            constraints_df=constraints,
            grav_ds=old_ds,
            grid_method="verde",
            dampings=1e-3,
        )

    new_ds = observed_gravity()
    invert4geom.regional.regional_constraints(
        constraints_df=constraints,
        grav_ds=new_ds,
        grid_method="splines",
        dampings=1e-3,
    )

    np.testing.assert_allclose(old_ds.reg.to_numpy(), new_ds.reg.to_numpy())


@pytest.mark.parametrize("block_reduction", ["mean", "median"])
def test_regional_constraints_block_reduction(block_reduction):
    """both block reductions should give a regional field without missing values"""
    grav_data = observed_gravity()

    invert4geom.regional.regional_constraints(
        constraints_df=scattered_constraints(),
        grav_ds=grav_data,
        grid_method="pygmt",
        block_size=15000,
        block_reduction=block_reduction,
        weights_col="weights",
    )

    assert not np.any(np.isnan(grav_data.reg.to_numpy()))


def test_regional_constraints_block_reduction_col():
    """only the points where `block_reduction_col` is True should be reduced"""
    constraints = scattered_constraints()

    subset_ds = observed_gravity()
    invert4geom.regional.regional_constraints(
        constraints_df=constraints,
        grav_ds=subset_ds,
        grid_method="pygmt",
        block_size=15000,
        block_reduction="median",
        block_reduction_col="reduce",
    )

    all_ds = observed_gravity()
    invert4geom.regional.regional_constraints(
        constraints_df=constraints,
        grav_ds=all_ds,
        grid_method="pygmt",
        block_size=15000,
        block_reduction="median",
    )

    assert not np.allclose(subset_ds.reg.to_numpy(), all_ds.reg.to_numpy())


def test_regional_constraints_block_reduction_col_missing_raises():
    with pytest.raises(ValueError, match="`block_reduction_col`"):
        invert4geom.regional.regional_constraints(
            constraints_df=scattered_constraints(),
            grav_ds=observed_gravity(),
            grid_method="pygmt",
            block_size=15000,
            block_reduction_col="not_a_column",
        )


def test_regional_constraints_maxdist():
    """`maxdist` should limit which points fill the cells without a constraint"""
    grav_data = observed_gravity()

    invert4geom.regional.regional_constraints(
        constraints_df=grid_point_constraints(),
        grav_ds=grav_data,
        grid_method="pygmt",
        grid_points_col="grid_points",
        maxdist=50000,
    )

    assert not np.any(np.isnan(grav_data.reg.to_numpy()))


@pytest.mark.parametrize("grid_method", ["nearest", "linear", "cubic"])
def test_regional_constraints_fill_only_methods_require_grid_points(grid_method):
    with pytest.raises(ValueError, match="can only be used if `grid_points_col`"):
        invert4geom.regional.regional_constraints(
            constraints_df=grid_point_constraints(),
            grav_ds=observed_gravity(),
            grid_method=grid_method,
        )


def test_regional_constraints_invalid_grid_method_raises():
    grav_data = observed_gravity()
    constraints = pd.DataFrame(
        {
            "easting": [5000.0, 35000.0, 5000.0, 35000.0],
            "northing": [5000.0, 25000.0, 25000.0, 5000.0],
            "upward": [500.0, 500.0, 500.0, 500.0],
        }
    )
    with pytest.raises(ValueError, match="invalid string for grid_method"):
        invert4geom.regional.regional_constraints(
            constraints_df=constraints,
            grav_ds=grav_data,
            grid_method="not_a_method",
        )
