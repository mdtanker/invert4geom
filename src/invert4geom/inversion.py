from __future__ import annotations  # pylint: disable=too-many-lines

import copy
import itertools
import logging
import time
import typing

import harmonica as hm
import numba
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr
from nptyping import NDArray

from invert4geom import plotting, utils


@numba.jit(cache=True, nopython=True)  # type: ignore[misc]
def grav_column_der(
    grav_easting: NDArray,
    grav_northing: NDArray,
    grav_upward: NDArray,
    prism_easting: NDArray,
    prism_northing: NDArray,
    prism_top: NDArray,
    prism_spacing: float,
    prism_density: NDArray,
) -> NDArray:
    """
        Function to calculate the vertical derivate of the gravitational acceleration at
        an observation point caused by a right, rectangular prism. Approximated with
        Hammer's annulus approximation.

    Parameters
    ----------
    grav_easting, grav_northing, grav_upward : NDArray
        coordinates of gravity observation points.
    prism_easting, prism_northing, prism_top : NDArray
        coordinates of prism's center in northing, easting, and upward directions,
        respectively
    prism_spacing : float
        resolution of prism layer in meters
    prism_density : NDArray
        density of prisms, in kg/m^3

    Returns
    -------
    NDArray
        array of vertical derivative of gravity at observation point for series of
        prisms
    """

    r = np.sqrt(
        np.square(grav_northing - prism_northing)
        + np.square(grav_easting - prism_easting)
    )
    r1 = r - 0.5 * prism_spacing
    r2 = r + 0.5 * prism_spacing

    # gravity observation point can't be within prism
    # if it is, instead calculate gravity on prism edge
    r1[r1 < 0] = 0
    r2[r2 < prism_spacing] = prism_spacing

    f = np.square(prism_spacing) / (
        np.pi * (np.square(r2) - np.square(r1))
    )  # eq 2.19 in McCubbine 2016 Thesis
    # 2*pi*G = 0.0000419
    return (
        0.0000419
        * f
        * prism_density
        * (prism_top - grav_upward)
        * (
            1 / np.sqrt(np.square(r2) + np.square(prism_top - grav_upward))
            - 1 / np.sqrt(np.square(r1) + np.square(prism_top - grav_upward))
        )
    )


@numba.njit(parallel=True)  # type: ignore[misc]
def jacobian_annular(
    grav_easting: NDArray,
    grav_northing: NDArray,
    grav_upward: NDArray,
    prism_easting: NDArray,
    prism_northing: NDArray,
    prism_top: NDArray,
    prism_density: NDArray,
    prism_spacing: float,
    jac: NDArray,
) -> NDArray:
    """
        Function to calculate the Jacobian matrix using the annular cylinder
        approximation. The resulting Jacobian is a matrix (numpy array) with a row per
        gravity observation and a column per prism. This approximates the prisms as an
        annulus, and calculates it's vertical gravity derivative. Takes arrays from
        `jacobian`, feeds them into `grav_column_der`, and returns the jacobian.

    Parameters
    ----------
    grav_easting, grav_northing, grav_upward : NDArray
        coordinates of gravity observation points.
    prism_easting, prism_northing, prism_top, : NDArray
        coordinates of prism's center in northing, easting, and upward directions,
        respectively
    prism_density : NDArray
        density of prisms, in kg/m^3
    prism_spacing : float
        resolution of prism layer in meters
    jac : NDArray
        empty jacobian matrix with a row per gravity observation and a column per prism

    Returns
    -------
    NDArray
        returns a jacobian matrix of shape (number of gravity points, number of prisms)
    """

    for i in numba.prange(len(grav_easting)):  # pylint: disable=not-an-iterable
        jac[i, :] = grav_column_der(
            grav_easting[i],
            grav_northing[i],
            grav_upward[i],
            prism_easting,
            prism_northing,
            prism_top,
            prism_spacing,
            prism_density,
        )

    return jac


def prism_properties(
    prisms_layer: xr.Dataset,
    method: str = "itertools",
) -> NDArray:
    """
    extract prism properties from prism layer

    Parameters
    ----------
    prisms_layer : xr.Dataset
       harmonica prism layer
    method : str, optional
        choice of method to extract properties, by default "itertools"

    Returns
    -------
    NDArray
        array of prism properties
    """

    if method == "itertools":
        prisms_properties: NDArray = []
        for (
            y,
            x,
        ) in itertools.product(
            range(prisms_layer.northing.size), range(prisms_layer.easting.size)
        ):
            prisms_properties.append(
                [
                    *list(prisms_layer.prism_layer.get_prism((y, x))),
                    prisms_layer.density.values[y, x],
                ]
            )
        prisms_properties = np.array(prisms_properties)
    elif method == "forloops":
        prisms_properties = []
        for y in range(prisms_layer.northing.size):
            for x in range(prisms_layer.easting.size):
                prisms_properties.append(
                    [
                        *list(prisms_layer.prism_layer.get_prism((y, x))),
                        prisms_layer.density.values[y, x],
                    ]
                )
        np.asarray(prisms_properties)
    elif method == "generator":
        # slower, but doesn't allocate memory
        prisms_properties = [
            list(prisms_layer.prism_layer.get_prism((y, x)))  # noqa: RUF005
            + [prisms_layer.density.values[y, x]]
            for y in range(prisms_layer.northing.size)
            for x in range(prisms_layer.easting.size)
        ]
    else:
        msg = "method must be one of 'itertools', 'forloops', or 'generator'"
        raise ValueError(msg)

    return prisms_properties


@numba.jit(forceobj=True, parallel=True)  # type: ignore[misc]
def jacobian_prism(
    prisms_properties: NDArray,
    grav_easting: NDArray,
    grav_northing: NDArray,
    grav_upward: NDArray,
    delta: float,
    jac: NDArray,
) -> NDArray:
    """
    Function to calculate the Jacobian matrix with the vertical gravity derivative
    as a numerical approximation with small prisms

    Takes arrays from `jacobian` and calculates the jacobian.

    Parameters
    ----------
    prisms_properties : NDArray
        array of prism properties of shape (number of prisms, 7) with the 7 entries for
        each prism being: west, east, south, north, bottom, top, density
    grav_easting, grav_northing,grav_upward : NDArray
        coordinates of gravity observation points.
    delta : float
        thickness in meters of small prisms used to calculate vertical derivative
    jac : NDArray
        empty jacobian matrix with a row per gravity observation and a column per prism

    Returns
    -------
    NDArray
        returns a NDArray of shape (number of gravity points, number of prisms)
    """

    # Build a small prism on top of existing prism (thickness equal to delta)
    for i in numba.prange(len(prisms_properties)):  # pylint: disable=not-an-iterable
        prism = prisms_properties[i]
        density = prism[6]
        bottom = prism[5]
        top = prism[5] + delta
        delta_prism = (prism[0], prism[1], prism[2], prism[3], bottom, top)

        jac[:, i] = (
            hm.prism_gravity(
                coordinates=(grav_easting, grav_northing, grav_upward),
                prisms=delta_prism,
                density=density,
                field="g_z",
                parallel=True,
            )
            / delta
        )

    return jac


def jacobian(
    deriv_type: str,
    coordinates: pd.DataFrame,
    empty_jac: NDArray | None = None,
    prisms_layer: xr.Dataset | None = None,
    prism_spacing: float | None = None,
    prism_size: float | None = None,
    prisms_properties_method: str = "itertools",
) -> NDArray:
    """
    dispatcher for creating the jacobian matrix with 2 method options

    Parameters
    ----------
    deriv_type : str
        choose between "annulus" and "prisms" methods of calculating the vertical
        derivative of gravity of a prism
    coordinates : pd.DataFrame
        coordinate dataframe of gravity observation points with columns "easting",
        "northing", "upward"
    empty_jac : NDArray, optional
        optionally provide an empty jacobian matrix of shape (number of gravity
        observations x number of prisms), by default None
    prisms_layer : xr.Dataset, optional
        harmonica prism layer, by default None
    prism_spacing : float, optional
        resolution of prism layer, by default None
    prism_size : float, optional
        height of prisms for small prism vertical derivative method, by default None
    prisms_properties_method : str, optional
        method for extracting prism properties, by default "itertools"

    Returns
    -------
    NDArray
        a filled out jacobian matrix
    """

    # convert dataframes to numpy arrays
    coordinates_array = coordinates.to_numpy()

    # get various arrays based on gravity column names
    grav_easting = coordinates_array[:, coordinates.columns.get_loc("easting")]
    grav_northing = coordinates_array[:, coordinates.columns.get_loc("northing")]
    grav_upward = coordinates_array[:, coordinates.columns.get_loc("upward")]

    assert len(grav_easting) == len(grav_northing) == len(grav_upward)

    if empty_jac is None:
        empty_jac = np.empty(
            (len(grav_easting), prisms_layer.top.size),  # type: ignore[union-attr]
            dtype=np.float64,
        )
        logging.warning("no empty jacobian supplied")

    jac = empty_jac.copy()

    if deriv_type == "annulus":
        # convert dataframe to arrays
        # arrays = {
        #   k:prisms_layer[k].to_numpy().ravel() for k in list(prisms_layer.variables)}
        df = prisms_layer.to_dataframe().reset_index().dropna().astype(float)  # type: ignore[union-attr]
        prism_easting = df.easting.to_numpy()
        prism_northing = df.northing.to_numpy()
        prism_top = df.top.to_numpy()
        prism_density = df.density.to_numpy()

        jac = jacobian_annular(
            grav_easting,
            grav_northing,
            grav_upward,
            prism_easting,
            prism_northing,
            prism_top,
            prism_density,
            prism_spacing,
            jac,
        )

    elif deriv_type == "prisms":
        # get prisms info in following format, 3 methods:
        # ((west, east, south, north, bottom, top), density)
        assert prisms_layer is not None
        prisms_properties = prism_properties(
            prisms_layer,
            method=prisms_properties_method,
        )
        if prism_size is None:
            msg = "need to set small prism height"
            raise ValueError(msg)

        jac = jacobian_prism(
            prisms_properties,
            grav_easting,
            grav_northing,
            grav_upward,
            prism_size,
            jac,
        )

    else:
        msg = "invalid string for deriv_type"
        raise ValueError(msg)

    return jac


def solver(
    jac: NDArray,
    residuals: NDArray,
    damping: float | None = None,
    solver_type: str = "scipy least squares",
    # bounds =None,
    # surface=None,
) -> NDArray:
    """
    Calculate shift to add to prism's for each iteration of the inversion. Finds
    the least-squares solution to the Jacobian and the gravity residual

    Parameters
    ----------
    jac : NDArray
        input jacobian matrix with a row per gravity observation, and a column per
        prisms.
    residuals : NDArray
        array of gravity residuals
    damping : float | None, optional
        positive damping (Tikhonov 0th order) regularization
    solver_type : {
        'verde least squares',
        'scipy least squares',
        'scipy conjugate',
        'numpy least squares',
        'steepest descent',
        'gauss newton',
        } optional
        choose which solving method to use, by default "scipy least squares"

    Returns
    -------
    NDArray
        array of correction values to apply to each prism.
    """

    if solver_type == "scipy least squares":
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html

        if damping is None:
            damping = 0
        results = sp.sparse.linalg.lsqr(
            A=jac,
            b=residuals,
            show=False,
            damp=damping,  # float, typically 0-1
            # atol= ,
            # btol=1e-4, # if 1e-6, residuals should be accurate to ~6 digits
            iter_lim=5000,  # limit of iterations, just in case of issues
        )
        # print(f"number of solver iters:{results[2]}")
        step = results[0]

    # elif solver_type == "verde least squares":
    #     """
    #     if damping not None, uses sklearn.linear_model.Ridge(alpha=damping)
    #     alpha: 0 to +inf. multiplies the L2 term, can also pass an array
    #     https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html # noqa: E501
    #     """
    #     step = vd.base.least_squares(
    #         jacobian=jac,
    #         data=residuals,
    #         weights=weights,
    #         damping=damping,  # float, typically 100-10,000
    #         copy_jacobian=False,
    #     )

    # elif solver_type == "scipy constrained":
    #     """
    #     https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html#scipy.optimize.lsq_linear # noqa: E501
    #     """
    #     if bounds is None:
    #         step = sp.optimize.lsq_linear(
    #             A=jac,
    #             b=residuals,
    #             method="trf",
    #             max_iter=5,
    #         )["x"]
    #     else:
    #         step = sp.optimize.lsq_linear(
    #             A=jac,
    #             b=residuals,
    #             bounds=bounds,
    #             method="trf",
    #             max_iter=5,
    #         )["x"]
    # # elif solver_type == "scipy nonlinear lsqr":
    # #     """
    # #     https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares # noqa: E501
    # #     """
    # #     if bounds is None:
    # #         bounds = [-np.inf, np.inf]

    # elif solver_type == "CLR":
    #     """
    #     https://github.com/avidale/constrained-linear-regression
    #     """
    #     model = ConstrainedLinearRegression(
    #         # max_iter=2,
    #         ridge=damping,
    #         # fit_intercept=False,
    #     )
    #     if bounds is None:
    #         step = model.fit(
    #             X=jac,
    #             y=residuals,
    #         ).coef_
    #     else:
    #         step = model.fit(
    #             X=jac,
    #             y=residuals,
    #             min_coef=bounds[0],
    #             max_coef=bounds[1],
    #         ).coef_

    # elif solver_type == "scipy conjugate":
    #     step = sp.sparse.linalg.cg(
    #         jac,
    #         residuals,
    #     )[0]

    # elif solver_type == "numpy least squares":
    #     step = np.linalg.lstsq(
    #         jac,
    #         residuals,
    #     )[0]

    # elif solver_type == "steepest descent":
    #     """Jacobian transppose algorithm"""
    #     residuals = residuals
    #     step = jac.T @ residuals

    # elif solver_type == "gauss newton":
    #     """
    #     Gauss Newton w/ 1st order Tikhonov regularization
    #     from https://nbviewer.org/github/compgeolab/2020-aachen-inverse-problems/blob/main/gravity-inversion.ipynb # noqa: E501
    #     """
    #     if damping in [None, 0]:
    #         hessian = jac.T @ jac
    #         gradient = jac.T @ residuals
    #     else:
    #         fdmatrix = finite_difference_matrix(jac[0].size)
    #         hessian = jac.T @ jac + damping * fdmatrix.T @ fdmatrix
    #         gradient = (
    #             jac.T @ residuals - damping * fdmatrix.T @ fdmatrix @ surface
    #         )

    #     # scipy solver appears to be slightly faster
    #     # step = np.linalg.solve(hessian, gradient)
    #     step = sp.linalg.solve(hessian, gradient)

    else:
        msg = "invalid string for solver_type"
        raise ValueError(msg)

    return step


# def finite_difference_matrix(nparams):
#     """
#     Create the finite difference matrix for regularization.
#     """
#     fdmatrix = np.zeros((nparams - 1, nparams))
#     for i in range(fdmatrix.shape[0]):
#         fdmatrix[i, i] = -1
#         fdmatrix[i, i + 1] = 1
#     return fdmatrix


def update_l2_norms(
    rmse: float,
    l2_norm: float,
) -> tuple[float, float]:
    """
    update the l2 norm and delta l2 norm of the misfit

    Parameters
    ----------
    rmse : float
        root mean square error of the residual gravity misfit
    l2_norm : float
        l2 norm of the residual gravity misfit
    Returns
    -------
    tuple[float, float]
        updated l2 norm and delta l2 norm
    """

    # square-root of RMSE is the l-2 norm
    updated_l2_norm = np.sqrt(rmse)

    updated_delta_l2_norm = l2_norm / updated_l2_norm

    # we want the misfit (L2-norm) to be steadily decreasing with each iteration.
    # If it increases, something is wrong, stop inversion
    # If it doesn't decrease enough, inversion has finished and can be stopped
    # delta L2 norm starts at +inf, and should decreases with each iteration.
    # if it gets close to 1, the iterations aren't making progress and can be stopped.
    # a value of 1.001 means the L2 norm has only decrease by 0.1% between iterations.
    # and RMSE has only decreased by 0.05%.

    # update the l2_norm
    l2_norm = updated_l2_norm

    # updated the delta l2_norm
    delta_l2_norm = updated_delta_l2_norm

    return (
        l2_norm,
        delta_l2_norm,
    )


def end_inversion(
    iteration_number: int,
    max_iterations: int,
    l2_norm: float,
    starting_l2_norm: float,
    l2_norm_tolerance: float,
    delta_l2_norm: float,
    previous_delta_l2_norm: float,
    delta_l2_norm_tolerance: float,
    perc_increase_limit: float = 0.20,
) -> tuple[bool, list[str]]:
    """
    check if the inversion should be terminated

    Parameters
    ----------
    iteration_number : int
        the iteration number, starting at 1 not 0
    max_iterations : int
        the maximum allowed iterations, inclusive and starting at 1
    l2_norm : float
        the current iteration's l2 norm
    starting_l2_norm : float
        the l2 norm of iteration 1
    l2_norm_tolerance : float
        the l2 norm value to end the inversion at
    delta_l2_norm : float
        the current iteration's delta l2 norm
    previous_delta_l2_norm : float
        the delta l2 norm of the previous iteration
    delta_l2_norm_tolerance : float
        the delta l2 norm value to end the inversion at
    perc_increase_limit : float, optional
        the set tolerance for decimal percentage increase relative to the starting l2
        norm, by default 0.20

    Returns
    -------
    tuple[bool, list[str]]
        first term is a boolean of whether or not to end the inversion, second term is a
        list of termination reasons.
    """
    end = False
    termination_reason = []

    # ignore for first iteration
    if iteration_number == 1:
        pass
    else:
        if l2_norm > starting_l2_norm * (1 + perc_increase_limit):
            logging.info(
                "\nInversion terminated after %s iterations because L2 norm (%s) \n"
                "was over %s%% greater than starting L2 norm (%s) \n"
                "Change parameter 'perc_increase_limit' if desired.",
                iteration_number,
                l2_norm,
                perc_increase_limit * 100,
                starting_l2_norm,
            )
            end = True
            termination_reason.append("l2-norm increasing")

        if (delta_l2_norm <= delta_l2_norm_tolerance) & (
            previous_delta_l2_norm <= delta_l2_norm_tolerance
        ):
            logging.info(
                "\nInversion terminated after %s iterations because there was no "
                "significant variation in the L2-norm over 2 iterations \n"
                "Change parameter 'delta_l2_norm_tolerance' if desired.",
                iteration_number,
            )

            end = True
            termination_reason.append("delta l2-norm tolerance")

        if l2_norm < l2_norm_tolerance:
            logging.info(
                "\nInversion terminated after %s iterations because L2-norm (%s) was "
                "less then set tolerance: %s \nChange parameter "
                "'l2_norm_tolerance' if desired.",
                iteration_number,
                l2_norm,
                l2_norm_tolerance,
            )

            end = True
            termination_reason.append("l2-norm tolerance")

    if iteration_number >= max_iterations:
        logging.info(
            "\nInversion terminated after %s iterations with L2-norm=%s because "
            "maximum number of iterations (%s) reached.",
            iteration_number,
            round(l2_norm, 2),
            max_iterations,
        )

        end = True
        termination_reason.append("max iterations")

    return end, termination_reason


def update_gravity_and_misfit(
    gravity_df: pd.DataFrame,
    prisms_ds: xr.Dataset,
    input_grav_column: str,
    iteration_number: int,
) -> pd.DataFrame:
    """
    calculate the forward gravity of the supplied prism layer, add the results to a
    new dataframe column, and update the residual misfit. The supplied gravity dataframe
    needs a 'reg' column, which describes the regional component and can be 0.

    Parameters
    ----------
    gravity_df : pd.DataFrame
        gravity dataframe with gravity observation coordinate columns ('easting',
        'northing', 'upwards'), a gravity data column, set by `input_grav_column`,
        and a regional gravity column ('reg').

    prisms_ds : xr.Dataset
        harmonica prism layer
    input_grav_column : str
        name of gravity data column
    iteration_number : int
        iteration number to use in updated column names

    Returns
    -------
    pd.DataFrame
        a gravity dataframe with 2 new columns, one for the iterations forward gravity
        and one for the iterations residual misfit.
    """
    gravity = gravity_df.copy()

    # update the forward gravity
    gravity[f"iter_{iteration_number}_forward_grav"] = prisms_ds.prism_layer.gravity(
        coordinates=(gravity.easting, gravity.northing, gravity.upward),
        field="g_z",
    )

    # each iteration updates the topography of the layer to minizime the residual
    # portion of the misfit. We then want to recalculate the forward gravity of the
    # new layer, use the same original regional misfit, and re-calculate the residual
    # Gmisfit  = Gobs_corr - Gforward
    # Gres = Gmisfit - Greg
    # Gres = Gobs_corr_shift - Gforward - Greg
    # update the residual misfit with the new forward gravity and the same regional
    gravity[f"iter_{iteration_number}_final_misfit"] = (
        gravity[input_grav_column]
        - gravity[f"iter_{iteration_number}_forward_grav"]
        - gravity.reg
    )

    return gravity


def run_inversion(
    input_grav: pd.DataFrame,
    input_grav_column: str,
    prism_layer: xr.Dataset,
    max_iterations: int,
    l2_norm_tolerance: float = 0.2,
    delta_l2_norm_tolerance: float = 1.001,
    perc_increase_limit: float = 0.10,
    deriv_type: str = "annulus",
    jacobian_prism_size: float = 1,
    solver_type: str = "scipy least squares",
    solver_damping: float | None = None,
    upper_confining_layer: xr.DataArray | None = None,
    lower_confining_layer: xr.DataArray | None = None,
    weights_after_solving: bool = False,
    plot_convergence: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float]:
    """
    perform a geometric inversion, where the topography is updated to minimize the
    residual misfit between the forward gravity of the layer, and the observed gravity.
    To aid in regularizing an ill-posed problem choose any of the following options:
    * add damping to the solver, with `solver_damping`
    * weight the surface correction values with a weighting grid with
    `weights_after_solving` and the `weights` variable of the prisms dataset
    * bound the topography of the layer, with `upper_confining_layer` and
    `lower_confining_layer`

    Parameters
    ----------
    input_grav : pd.DataFrame
        dataframe with gravity data and coordinates, must have columns "res" and "reg"
        for residual and regional gravity.
    input_grav_column : str
        column name containing the gravity data *before* regional separation
    prism_layer : xr.Dataset
        starting prism layer
    max_iterations : int
        the maximum allowed iterations, inclusive and starting at 1
    l2_norm_tolerance : float, optional
        _the l2 norm value to end the inversion at, by default 0.2
    delta_l2_norm_tolerance : float, optional
        the delta l2 norm value to end the inversion at, by default 1.001
    perc_increase_limit : float, optional
        the set tolerance for decimal percentage increase relative to the starting l2
        norm, by default 0.10
    deriv_type : str, optional
        either "annulus" or "prism" to determine method of calculating the vertical
        derivate of gravity of a prism, by default "annulus"
    jacobian_prism_size : float, optional
        height of prisms in meters for vertical derivative, by default 1
    solver_type : str, optional
        solver type to use, by default "scipy least squares"
    solver_damping : float | None, optional
        damping parameter for regularization of the solver, by default None
    upper_confining_layer : xr.DataArray | None, optional
        topographic layer to use as upper limit for inverted topography, by default None
    lower_confining_layer : xr.DataArray | None, optional
        topographic layer to use as lower limit for inverted topography, by default None
    weights_after_solving : bool, optional
        use "weights" variable of prisms dataset to scale surface corrections grid, by
        default False, by default False
    plot_convergence : bool, optional
        plot the misfit convergence, by default False

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float]
        prisms_df: pd.DataFrame, prism properties for each iteration,
        gravity: pd.DataFrame, gravity anomalies for each iteration,
        params: dict, Properties of the inversion such as kwarg values,
        elapsed_time: float, time in seconds for the inversion to run
    """

    logging.info("starting inversion")

    time_start = time.perf_counter()

    gravity = copy.deepcopy(input_grav)

    # extract variables from starting prism layer
    (
        prisms_df,
        prisms_ds,
        density_contrast,
        zref,
        prism_spacing,
        _,
    ) = utils.extract_prism_data(prism_layer)

    logging.info("extracted zref is %s", zref)
    logging.info("extracted prism spacing is %s", prism_spacing)
    logging.info("extracted density contrast is %s", density_contrast)

    # create empty jacobian matrix
    empty_jac: NDArray = np.empty(
        (len(gravity[input_grav_column]), prisms_ds.top.size),
        dtype=np.float64,
    )

    # if there is a confining surface (above or below), which the inverted layer
    # shouldn't intersect, then sample those layers into the df
    prisms_df = utils.sample_bounding_surfaces(
        prisms_df,
        upper_confining_layer,
        lower_confining_layer,
    )

    # set starting delta L2 norm to positive infinity
    delta_l2_norm = np.Inf

    # iteration times
    iter_times = []

    for iteration, _ in enumerate(range(max_iterations), start=1):
        logging.info(
            "\n #################################### \n iteration %s", iteration
        )
        # start iteration timer
        iter_time_start = time.perf_counter()

        # after first iteration reset residual with previous iteration's results
        if iteration == 1:
            pass
        else:
            gravity["res"] = gravity[f"iter_{iteration-1}_final_misfit"]
            prisms_df["density"] = prisms_df[f"iter_{iteration-1}_density"]

        # add starting residual to df
        gravity[f"iter_{iteration}_initial_misfit"] = gravity.res

        # set iteration stats
        initial_rmse = utils.rmse(gravity[f"iter_{iteration}_initial_misfit"])
        l2_norm = np.sqrt(initial_rmse)

        if iteration == 1:
            starting_l2_norm = l2_norm

        # calculate jacobian sensitivity matrix
        jac = jacobian(
            deriv_type,
            gravity.select_dtypes(include=["number"]),
            empty_jac,
            prisms_layer=prisms_ds,
            prism_spacing=prism_spacing,
            prism_size=jacobian_prism_size,
        )

        # calculate correction for each prism
        surface_correction = solver(
            jac=jac,
            residuals=gravity.res.values,
            damping=solver_damping,
            solver_type=solver_type,
        )

        # print correction values
        logging.info(
            "Layer correction median: %s m, RMSE:%s m",
            round(np.median(surface_correction), 4),
            round(utils.rmse(surface_correction), 4),
        )

        # add corrections to prisms_df
        prisms_df = pd.concat(
            [
                prisms_df,
                pd.DataFrame({f"iter_{iteration}_correction": surface_correction}),
            ],
            axis=1,
        )
        # apply the surface correction to the prisms dataframe and enforce confining
        # layer if supplied
        prisms_df, correction_grid = utils.apply_surface_correction(
            prisms_df, iteration
        )

        # instead of applying weights to the Jacobian, apply them to the topo
        # correction grid
        if weights_after_solving is True:
            correction_grid = correction_grid * prisms_ds.weights

        # add the corrections to the topo and update the prisms dataset
        prisms_ds = utils.update_prisms_ds(prisms_ds, correction_grid, zref)

        # add updated properties to prisms dataframe
        prisms_df = utils.add_updated_prism_properties(
            prisms_df,
            prisms_ds,
            iteration,
        )

        if upper_confining_layer is not None:
            assert np.all(prisms_df.upper_bounds - prisms_df.topo) >= 0

        # update the forward gravity and the misfit
        gravity = update_gravity_and_misfit(
            gravity,
            prisms_ds,
            input_grav_column,
            iteration,
        )

        # update the misfit RMSE
        updated_rmse = utils.rmse(gravity[f"iter_{iteration}_final_misfit"])
        logging.info("updated misfit RMSE: %s", round(updated_rmse, 4))
        final_rmse = updated_rmse

        # update the l2 and delta l2 norms
        previous_delta_l2_norm = copy.copy(delta_l2_norm)
        l2_norm, delta_l2_norm = update_l2_norms(updated_rmse, l2_norm)
        final_l2_norm = l2_norm
        logging.info(
            "updated L2-norm: %s, tolerance: %s", round(l2_norm, 4), l2_norm_tolerance
        )
        logging.info(
            "updated delta L2-norm : %s, tolerance: %s",
            round(delta_l2_norm, 4),
            delta_l2_norm_tolerance,
        )

        # end iteration timer
        iter_time_end = time.perf_counter()
        iter_times.append(iter_time_end - iter_time_start)

        # decide if to end the inversion
        end, termination_reason = end_inversion(
            iteration,
            max_iterations,
            l2_norm,
            starting_l2_norm,
            l2_norm_tolerance,
            delta_l2_norm,
            previous_delta_l2_norm,
            delta_l2_norm_tolerance,
            perc_increase_limit=perc_increase_limit,
        )
        if end is True:
            break
        # end of inversion loop

    time_end = time.perf_counter()

    elapsed_time = time_end - time_start

    # collect input parameters into a dictionary
    params = {
        # first column
        "density_contrast": f"{density_contrast} kg/m3",
        "max_iterations": max_iterations,
        "l2_norm_tolerance": f"{l2_norm_tolerance}",
        "delta_l2_norm_tolerance": f"{delta_l2_norm_tolerance}",
        "deriv_type": deriv_type,
        # second column
        "solver_type": solver_type,
        "solver_damping": solver_damping,
        "upper_confining_layer": "Not enabled"
        if upper_confining_layer is None
        else "Enabled",
        "lower_confining_layer": "Not enabled"
        if lower_confining_layer is None
        else "Enabled",
        # third column
        "time_elapsed": f"{int(elapsed_time)} seconds",
        "average_iteration_time": f"{round(np.mean(iter_times), 2)} seconds",
        "Final misfit RMSE / L2-norm": (
            f"{round(final_rmse,4)} /{round(final_l2_norm,4)} mGal"
        ),
        "Termination reason": termination_reason,
        "iter_times": iter_times,
    }

    if plot_convergence is True:
        plotting.plot_convergence(gravity, iter_times=iter_times)

    return prisms_df, gravity, params, elapsed_time
