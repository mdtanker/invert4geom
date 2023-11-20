from __future__ import annotations

import harmonica as hm
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import verde as vd
import xarray as xr
from nptyping import NDArray

from invert4geom import inversion

pd.set_option("display.max_columns", None)

################
################
# DUMMY FUNCTIONS
################
################


def dummy_df() -> pd.DataFrame:
    data = {
        "northing": [
            200,
            200,
            400,
            400,
        ],
        "easting": [
            -100,
            100,
            -100,
            100,
        ],
        "upward": [20, 20, 20, 20],
        "observed_grav": [113, 111, 115, 114],
        "forward_grav": [12, 13, 14, 15],
    }
    return pd.DataFrame(data)


def dummy_misfit_df(regional: bool = True) -> pd.DataFrame:
    data = {
        "northing": [
            200,
            200,
            400,
            400,
        ],
        "easting": [
            -100,
            100,
            -100,
            100,
        ],
        "upward": [20, 20, 20, 20],
        "observed_grav": [6.5, 6.8, 7.2, 8.0],
        "forward_grav": [7.0, 7.0, 7.0, 7.0],
    }
    df = pd.DataFrame(data)
    # calculate misfit -> [0.5, -0.2, 0.2, 2.0]
    df["misfit"] = df.observed_grav - df.forward_grav
    # set regional component of misfit
    if regional is True:
        df["reg"] = [3, 2, 1, 0]
    elif regional is False:
        df["reg"] = [0, 0, 0, 0]
    # calculate residual component
    df["res"] = df.misfit - df.reg
    # without regional
    #    northing  easting  upward  observed_grav  forward_grav  misfit  reg  res
    # 0       200     -100      20            6.5           7.0    -0.5    0 -0.5
    # 1       200      100      20            6.8           7.0    -0.2    0 -0.2
    # 2       400     -100      20            7.2           7.0     0.2    0  0.2
    # 3       400      100      20            8.0           7.0     1.0    0  1.0
    # with regional
    #    northing  easting  upward  observed_grav  forward_grav  misfit  reg  res
    # 0       200     -100      20            6.5           7.0    -0.5    3 -3.5
    # 1       200      100      20            6.8           7.0    -0.2    2 -2.2
    # 2       400     -100      20            7.2           7.0     0.2    1 -0.8
    # 3       400      100      20            8.0           7.0     1.0    0  1.0
    return df


def dummy_df_big() -> pd.DataFrame:
    df = dummy_prism_layer().to_dataframe().reset_index().dropna().astype(float)
    df = df.drop(columns=["top", "bottom", "density"])
    df["upward"] = 20
    df["misfit"] = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    return df


def dummy_prism_layer() -> xr.Dataset:
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


def dummy_jacobian() -> NDArray:
    """
    Create a under-determined jacobian with vertical derivative values
    """
    grav = dummy_df()
    prisms_layer = dummy_prism_layer_flat()
    prisms_properties = inversion.prism_properties(prisms_layer, method="itertools")
    jac = np.empty(
        (len(grav), prisms_layer.top.size),
        dtype=np.float64,
    )
    return inversion.jacobian_prism(
        prisms_properties,
        np.array(grav.easting),
        np.array(grav.northing),
        np.array(grav.upward),
        0.001,
        jac,
    )


def dummy_jacobian_square() -> NDArray:
    """
    Create a square jacobian with vertical derivative values
    """
    grav = dummy_df_big()
    prisms_layer = dummy_prism_layer_flat()
    prisms_properties = inversion.prism_properties(prisms_layer, method="itertools")
    jac = np.empty(
        (len(grav), prisms_layer.top.size),
        dtype=np.float64,
    )
    return inversion.jacobian_prism(
        prisms_properties,
        np.array(grav.easting),
        np.array(grav.northing),
        np.array(grav.upward),
        0.001,
        jac,
    )


################
################
# TESTS
################
################


@pytest.mark.use_numba()
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
    a = inversion.grav_column_der(
        grav_easting=0,
        grav_northing=0,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    b = inversion.grav_column_der(
        grav_easting=5,
        grav_northing=0,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    c = inversion.grav_column_der(
        grav_easting=5,
        grav_northing=5,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    d = inversion.grav_column_der(
        grav_easting=0,
        grav_northing=5,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    e = inversion.grav_column_der(
        grav_easting=2.5,
        grav_northing=2.5,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    f = inversion.grav_column_der(
        grav_easting=5,
        grav_northing=2.5,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    g = inversion.grav_column_der(
        grav_easting=2.5,
        grav_northing=5,
        grav_upward=100,
        prism_easting=np.array([2.5]),
        prism_northing=np.array([2.5]),
        prism_top=np.array([-10]),
        prism_spacing=5,
        prism_density=np.array([2670]),
    )
    h = inversion.grav_column_der(
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


@pytest.mark.use_numba()
def test_grav_column_der():
    """
    test the grav_column_der function against a small prism approximation
    """
    # expected result
    dg_z = inversion.grav_column_der(
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
    step = 0.000001
    hm_dg_z = (
        hm.prism_gravity(
            (20, 20, 100), (0, 5, 0, 5, -10, -10 + step), 2670, field="g_z"
        )
        / step
    )
    # test that the derivative matches a small prism approximation from
    assert dg_z == pytest.approx(hm_dg_z, rel=1e-3)


@pytest.mark.use_numba()
def test_jacobian_annular():
    """
    test the jacobian_annular function
    """
    grav = dummy_df()
    prisms_layer = dummy_prism_layer()
    prisms_df = prisms_layer.to_dataframe().reset_index().dropna().astype(float)
    jac = np.empty(
        (len(grav), prisms_layer.top.size),
        dtype=np.float64,
    )
    jac = inversion.jacobian_annular(
        np.array(grav.easting),
        np.array(grav.northing),
        np.array(grav.upward),
        np.array(prisms_df.easting),
        np.array(prisms_df.northing),
        np.array(prisms_df.top),
        np.array(prisms_df.density),
        200,
        jac,
    )
    # test that prisms above observation point have negative vertical derivatives
    assert jac[:, -3:].max() < 0
    # test that prisms below observation point have positive vertical derivatives
    assert jac[:, 0:-3].min() > 0


def test_prism_properties():
    """
    test the prism_properties function
    """
    prisms_layer = dummy_prism_layer()
    itertools_result = inversion.prism_properties(prisms_layer, method="itertools")
    forloops_result = inversion.prism_properties(prisms_layer, method="forloops")
    generator_result = inversion.prism_properties(prisms_layer, method="generator")
    # test that the prism properties are the same with 3 methods
    np.array_equal(itertools_result, forloops_result)
    np.array_equal(itertools_result, generator_result)
    # test that the first prism's properties are correct
    np.array_equal(itertools_result[0], np.array([-300, -100, 0, 200, -100, 2670]))


def test_prism_properties_error():
    """
    test the prism_properties function raises the correct error
    """
    prisms_layer = dummy_prism_layer()
    with pytest.raises(ValueError, match="method must be"):
        inversion.prism_properties(prisms_layer, method="wrong_input")


@pytest.mark.use_numba()
def test_jacobian_prism():
    """
    test the jacobian_prism function
    """
    grav = dummy_df()
    prisms_layer = dummy_prism_layer()
    prisms_properties = inversion.prism_properties(prisms_layer, method="itertools")
    jac = np.empty(
        (len(grav), prisms_layer.top.size),
        dtype=np.float64,
    )
    jac = inversion.jacobian_prism(
        prisms_properties,
        np.array(grav.easting),
        np.array(grav.northing),
        np.array(grav.upward),
        0.001,
        jac,
    )
    # test that prisms above observation point have negative vertical derivatives
    assert jac[:, -3:].max() < 0
    # test that prisms below observation point have positive vertical derivatives
    assert jac[:, 0:-3].min() > 0


@pytest.mark.use_numba()
def test_jacobian():
    """
    test the jacobian dispatcher function
    """
    grav = dummy_df()
    prisms_layer = dummy_prism_layer()
    annulus_jac = inversion.jacobian(
        deriv_type="annulus",
        coordinates=grav,
        empty_jac=None,
        prisms_layer=prisms_layer,
        prism_spacing=200,
        prism_size=None,
        prisms_properties_method="itertools",
    )
    prisms_jac = inversion.jacobian(
        deriv_type="prisms",
        coordinates=grav,
        empty_jac=None,
        prisms_layer=prisms_layer,
        prism_spacing=200,
        prism_size=0.01,
        prisms_properties_method="itertools",
    )
    np.array_equal(annulus_jac, prisms_jac)


def test_jacobian_error():
    """
    test the jacobian dispatcher function raises the correct error
    """
    prisms_layer = dummy_prism_layer()
    with pytest.raises(ValueError, match="invalid string"):
        inversion.jacobian(
            deriv_type="wrong_input",
            coordinates=dummy_df(),
            empty_jac=None,
            prisms_layer=prisms_layer,
            prism_spacing=200,
            prism_size=None,
            prisms_properties_method="itertools",
        )


def test_jacobian_prism_height():
    """
    test the jacobian dispatcher function raises error for no prism height
    """
    prisms_layer = dummy_prism_layer()
    with pytest.raises(ValueError, match="need to set"):
        inversion.jacobian(
            deriv_type="prisms",
            coordinates=dummy_df(),
            empty_jac=None,
            prisms_layer=prisms_layer,
            prism_spacing=200,
            prism_size=None,
            prisms_properties_method="itertools",
        )


def test_solver_square_error():
    """
    test the solver function raises the correct error
    """
    misfit = dummy_df_big().misfit.to_numpy()
    jac = dummy_jacobian_square()
    with pytest.raises(ValueError, match="invalid string"):
        inversion.solver(jac, misfit, solver_type="wrong_input")


solver_types = [
    "scipy least squares",
    # "verde least squares",
    # "scipy constrained",
    # "scipy nonlinear lsqr",
    # "CLR",
    # "scipy conjugate",
    # "numpy least squares",
    # "steepest descent", # off by 2 orders of magnitude
    # "gauss newton",
]


@pytest.mark.use_numba()
@pytest.mark.parametrize("solver_type", solver_types)
def test_solver_square(solver_type):
    """
    test the solver function with equal number of prisms and misfit values
    """
    misfit = dummy_df_big().misfit.to_numpy()
    jac = dummy_jacobian_square()
    correction = inversion.solver(jac, misfit, solver_type=solver_type)
    # test that correction is negative for negative misfits
    assert correction[0:3].max() < -9
    # test that correction is near 0 for misfits with values of 0
    npt.assert_allclose(correction[3:6], np.array([0, 0, 0]), atol=1e-8)
    # test that correction is positive for positive misfits
    assert correction[6:9].min() > 9


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


def test_update_l2_norms_updated_l2_norm():
    """
    Test if the updated L2 norm is correctly computed.
    """
    rmse = 4.0
    l2_norm = 2.0
    updated_l2_norm, _ = inversion.update_l2_norms(rmse, l2_norm)
    expected_updated_l2_norm = np.sqrt(rmse)
    assert updated_l2_norm == pytest.approx(expected_updated_l2_norm, rel=1e-6)


def test_update_l2_norms_updated_delta_l2_norm():
    """
    Test if the updated delta L2 norm is correctly computed.
    """
    rmse = 4.0
    l2_norm = 2.0
    _, updated_delta_l2_norm = inversion.update_l2_norms(rmse, l2_norm)
    expected_updated_delta_l2_norm = l2_norm / np.sqrt(rmse)
    assert updated_delta_l2_norm == pytest.approx(
        expected_updated_delta_l2_norm, rel=1e-6
    )


def test_end_inversion_first_iteration():
    """
    Test that the inversion is not terminated in the first iteration.
    """
    iteration_number = 1
    max_iterations = 100
    l2_norm = 1.0
    starting_l2_norm = 1.0
    l2_norm_tolerance = 0.1
    delta_l2_norm = 0.01
    previous_delta_l2_norm = 0.01
    delta_l2_norm_tolerance = 0.01
    end, termination_reason = inversion.end_inversion(
        iteration_number,
        max_iterations,
        l2_norm,
        starting_l2_norm,
        l2_norm_tolerance,
        delta_l2_norm,
        previous_delta_l2_norm,
        delta_l2_norm_tolerance,
    )
    assert not end
    assert termination_reason == []


def test_end_inversion_l2_norm_increasing():
    """
    Test that the inversion is terminated when L2 norm increases beyond a limit.
    """
    iteration_number = 2
    max_iterations = 100
    l2_norm = 1.3
    starting_l2_norm = 1.0
    l2_norm_tolerance = 0.1
    delta_l2_norm = 0.01
    previous_delta_l2_norm = 0.01
    delta_l2_norm_tolerance = 0.01
    perc_increase_limit = 0.20
    end, termination_reason = inversion.end_inversion(
        iteration_number,
        max_iterations,
        l2_norm,
        starting_l2_norm,
        l2_norm_tolerance,
        delta_l2_norm,
        previous_delta_l2_norm,
        delta_l2_norm_tolerance,
        perc_increase_limit,
    )
    assert end
    assert "l2-norm increasing" in termination_reason


def test_end_inversion_delta_l2_norm_tolerance():
    """
    Test that the inversion is terminated when delta L2 norm is below a tolerance.
    """
    iteration_number = 2
    max_iterations = 100
    l2_norm = 1.0
    starting_l2_norm = 1.0
    l2_norm_tolerance = 0.1
    delta_l2_norm = 0.01
    previous_delta_l2_norm = 0.01
    delta_l2_norm_tolerance = 0.01
    end, termination_reason = inversion.end_inversion(
        iteration_number,
        max_iterations,
        l2_norm,
        starting_l2_norm,
        l2_norm_tolerance,
        delta_l2_norm,
        previous_delta_l2_norm,
        delta_l2_norm_tolerance,
    )
    assert end
    assert "delta l2-norm tolerance" in termination_reason


def test_end_inversion_l2_norm_tolerance():
    """
    Test that the inversion is terminated when L2 norm is below a tolerance.
    """
    iteration_number = 2
    max_iterations = 100
    l2_norm = 0.05
    starting_l2_norm = 1.0
    l2_norm_tolerance = 0.1
    delta_l2_norm = 0.01
    previous_delta_l2_norm = 0.01
    delta_l2_norm_tolerance = 0.01
    end, termination_reason = inversion.end_inversion(
        iteration_number,
        max_iterations,
        l2_norm,
        starting_l2_norm,
        l2_norm_tolerance,
        delta_l2_norm,
        previous_delta_l2_norm,
        delta_l2_norm_tolerance,
    )
    assert end
    assert "l2-norm tolerance" in termination_reason


def test_end_inversion_max_iterations():
    """
    Test that the inversion is terminated when the maximum number of iterations is
    reached.
    """
    iteration_number = 101
    max_iterations = 100
    l2_norm = 0.5
    starting_l2_norm = 1.0
    l2_norm_tolerance = 0.1
    delta_l2_norm = 0.01
    previous_delta_l2_norm = 0.01
    delta_l2_norm_tolerance = 0.01
    end, termination_reason = inversion.end_inversion(
        iteration_number,
        max_iterations,
        l2_norm,
        starting_l2_norm,
        l2_norm_tolerance,
        delta_l2_norm,
        previous_delta_l2_norm,
        delta_l2_norm_tolerance,
    )
    assert end
    assert "max iterations" in termination_reason


def test_update_gravity_and_misfit_forward_gravity():
    """
    Test if the forward gravity is correctly updated.
    """
    gravity_df_copy = dummy_misfit_df(regional=False)
    # without regional
    #    northing  easting  upward  observed_grav  forward_grav  misfit  reg  res
    # 0       200     -100      20            6.5           7.0    -0.5    0 -0.5
    # 1       200      100      20            6.8           7.0    -0.2    0 -0.2
    # 2       400     -100      20            7.2           7.0     0.2    0  0.2
    # 3       400      100      20            8.0           7.0     1.0    0  1.0

    updated_gravity_df = inversion.update_gravity_and_misfit(
        gravity_df=gravity_df_copy,
        prisms_ds=dummy_prism_layer(),
        input_grav_column="observed_grav",
        iteration_number=1,
    )
    # Check that 'iter_1_forward_grav' column is created
    assert "iter_1_forward_grav" in updated_gravity_df.columns
    # Ensure that the 'iter_1_forward_grav' values are as expected
    expected_forward_grav = [7.18, 7.18, 7.70, 7.70]
    assert updated_gravity_df.iter_1_forward_grav.tolist() == pytest.approx(
        expected_forward_grav, 0.01
    )
    # Check that 'iter_1_final_misfit' column is created
    assert "iter_1_final_misfit" in updated_gravity_df.columns
    # Ensure that the 'iter_1_final_misfit' values are as expected
    # since regional is 0, the new misfit should be observed grav - iter_1_forward_grav
    expected_misfit = [-0.68, -0.38, -0.5, 0.30]
    assert updated_gravity_df.iter_1_final_misfit.tolist() == pytest.approx(
        expected_misfit, 0.01
    )


def test_update_gravity_and_misfit_forward_gravity_regional():
    """
    Test if the forward gravity is correctly updated with regional.
    """
    gravity_df_copy = dummy_misfit_df(regional=True)
    # with regional
    #    northing  easting  upward  observed_grav  forward_grav  misfit  reg  res
    # 0       200     -100      20            6.5           7.0    -0.5    3 -3.5
    # 1       200      100      20            6.8           7.0    -0.2    2 -2.2
    # 2       400     -100      20            7.2           7.0     0.2    1 -0.8
    # 3       400      100      20            8.0           7.0     1.0    0  1.0
    updated_gravity_df = inversion.update_gravity_and_misfit(
        gravity_df=gravity_df_copy,
        prisms_ds=dummy_prism_layer(),
        input_grav_column="observed_grav",
        iteration_number=1,
    )
    # expected_forward_grav = [7.18, 7.18, 7.70, 7.70]
    # Ensure that the 'iter_1_final_misfit' values are as expected
    # new misfit should be observed grav - iter_5_forward_grav - regional
    expected_misfit = [-3.68, -2.38, -1.5, 0.30]
    assert updated_gravity_df.iter_1_final_misfit.tolist() == pytest.approx(
        expected_misfit, 0.01
    )


# @pytest.mark.use_numba()
# def test_run_inversion_returns():
#     """
#     Test the inversions returned values.
#     """
#     gravity_df = dummy_misfit_df(regional=False)
#     prisms_ds = dummy_prism_layer()
#     print(gravity_df)
#     print(prisms_ds)
#     results = inversion.run_inversion(
#         input_grav=gravity_df,
#         input_grav_column="observed_grav",
#         prism_layer=prisms_ds,
#         max_iterations=3,
#     )
#     prisms_df, gravity, params, elapsed_time = results
#     # print(prisms_df)
#     # print(gravity)
#     # print(params)

#     # check elapsed time is reasonable
#     assert elapsed_time < 30


# def test_update_gravity_and_misfit_forward_gravity():
#     """
#     Test if the forward gravity is correctly updated.
#     """
#     gravity_df_copy = dummy_misfit_df(regional=False)
#     # print(dummy_misfit_df(regional=False))
#     # print(dummy_misfit_df(regional=True))
#     # without regional
#     #    northing  easting  upward  observed_grav  forward_grav  misfit  reg  res
#     # 0       200     -100      20            6.5           7.0    -0.5    0 -0.5
#     # 1       200      100      20            6.8           7.0    -0.2    0 -0.2
#     # 2       400     -100      20            7.2           7.0     0.2    0  0.2
#     # 3       400      100      20            8.0           7.0     1.0    0  1.0
#     gravity_df_copy["iter_4_forward_grav"] = [6.9, 6.9, 7.1, 7.4]
#     # calculate misfit -> [-0.4, -0.1, 0.1, 0.6]
#     gravity_df_copy["iter_4_final_misfit"] = gravity_df_copy.observed_grav -
# gravity_df_copy.iter_4_forward_grav
#     # print(gravity_df_copy)
#     updated_gravity_df = inversion.update_gravity_and_misfit(
#         gravity_df=gravity_df_copy,
#         prisms_ds=dummy_prism_layer(),
#         input_grav_column="observed_grav",
#         iteration_number=5,
#     )
#     # Check that 'iter_5_forward_grav' column is created
#     assert 'iter_5_forward_grav' in updated_gravity_df.columns
#     # Ensure that the 'iter_5_forward_grav' values are as expected
#     expected_forward_grav = [7.18, 7.18, 7.70, 7.70]
#     assert updated_gravity_df.iter_5_forward_grav.tolist() == pytest.approx(
#      expected_forward_grav, 0.01)
#     # Check that 'iter_5_final_misfit' column is created
#     assert 'iter_5_final_misfit' in updated_gravity_df.columns
#     # Ensure that the 'iter_5_final_misfit' values are as expected
#     # since regional is 0, the new misfit should be observed grav -
#       iter_5_forward_grav
#     expected_misfit = [-0.68, -0.38, -0.5, 0.30]
#     assert updated_gravity_df.iter_5_final_misfit.tolist() == pytest.approx(
#           expected_misfit, 0.01)
#     print(updated_gravity_df)
