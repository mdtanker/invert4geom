from __future__ import annotations  # pylint: disable=too-many-lines

import copy
import itertools
import logging
import math
import pathlib
import pickle
import random
import time
import typing

import harmonica as hm
import numba
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr
from numpy.typing import NDArray
from tqdm.autonotebook import tqdm

from invert4geom import cross_validation, log, optimization, plotting, regional, utils


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
    Hammer's annulus approximation :footcite:p:`mccubbineairborne2016`.

    Parameters
    ----------
    grav_easting, grav_northing, grav_upward : numpy.ndarray
        coordinates of gravity observation points.
    prism_easting, prism_northing, prism_top : numpy.ndarray
        coordinates of prism's center in northing, easting, and upward directions,
        respectively
    prism_spacing : float
        resolution of prism layer in meters
    prism_density : numpy.ndarray
        density of prisms, in kg/m^3

    Returns
    -------
    numpy.ndarray
        array of vertical derivative of gravity at observation point for series of
        prisms

    References
    ----------
    .. footbibliography::
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
    annulus :footcite:p:`mccubbineairborne2016`, and calculates it's vertical gravity
    derivative. Takes arrays from `jacobian`, feeds them into `grav_column_der`, and
    returns the jacobian.

    Parameters
    ----------
    grav_easting, grav_northing, grav_upward : numpy.ndarray
        coordinates of gravity observation points
    prism_easting, prism_northing, prism_top : numpy.ndarray
        coordinates of prism's center in northing, easting, and upward directions,
        respectively
    prism_density : numpy.ndarray
        density of prisms, in kg/m^3
    prism_spacing : float
        resolution of prism layer in meters
    jac : numpy.ndarray
        empty jacobian matrix with a row per gravity observation and a column per prism

    Returns
    -------
    numpy.ndarray
        returns a jacobian matrix of shape (number of gravity points, number of prisms)

    References
    ----------
    .. footbibliography::
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


def _prism_properties(
    prisms_layer: xr.Dataset,
    method: str = "itertools",
) -> NDArray:
    """
    extract prism properties from prism layer

    Parameters
    ----------
    prisms_layer : xarray.Dataset
       harmonica prism layer
    method : str, optional
        choice of method to extract properties, by default "itertools"

    Returns
    -------
    numpy.ndarray
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


# @numba.jit(forceobj=True, parallel=True)
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
    prisms_properties : numpy.ndarray
        array of prism properties of shape (number of prisms, 7) with the 7 entries for
        each prism being: west, east, south, north, bottom, top, density
    grav_easting, grav_northing,grav_upward : numpy.ndarray
        coordinates of gravity observation points.
    delta : float
        thickness in meters of small prisms used to calculate vertical derivative
    jac : numpy.ndarray
        empty jacobian matrix with a row per gravity observation and a column per prism

    Returns
    -------
    numpy.ndarray
        returns a numpy.ndarray of shape (number of gravity points, number of prisms)
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
    coordinates : pandas.DataFrame
        coordinate dataframe of gravity observation points with columns "easting",
        "northing", "upward"
    empty_jac : numpy.ndarray, optional
        optionally provide an empty jacobian matrix of shape (number of gravity
        observations x number of prisms), by default None
    prisms_layer : xarray.Dataset, optional
        harmonica prism layer, by default None
    prism_spacing : float, optional
        resolution of prism layer, by default None
    prism_size : float, optional
        height of prisms for small prism vertical derivative method, by default None
    prisms_properties_method : str, optional
        method for extracting prism properties, by default "itertools"

    Returns
    -------
    numpy.ndarray
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
        log.warning("no empty jacobian supplied")

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

        if np.all((prism_top - grav_upward.mean()) == 0):
            msg = (
                "All prism tops coincides exactly with the elevation of the gravity "
                "observation points, leading to issues with calculating the vertical "
                "derivative of gravity with the annulus technique. Either slightly "
                "change the prism tops or gravity elevations, or use the small-prisms "
                "vertical derivative technique."
            )
            raise ValueError(msg)

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
        prisms_properties = _prism_properties(
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
    jac : numpy.ndarray
        input jacobian matrix with a row per gravity observation, and a column per
        prisms.
    residuals : numpy.ndarray
        array of gravity residuals
    damping : float | None, optional
        positive damping (Tikhonov 0th order) regularization
    solver_type : str, optional
        choose which solving method to use, by default "scipy least squares"

    Returns
    -------
    numpy.ndarray
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
    #     """Jacobian transpose algorithm"""
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
    current_rmse: float,
    last_l2_norm: float,
) -> tuple[float, float]:
    """
    update the l2 norm and delta l2 norm of the misfit.

    We want the misfit (L2-norm) to be steadily decreasing with each iteration.
    If it increases, something is wrong, stop inversion.
    If it doesn't decrease enough, inversion has finished and can be stopped.
    Delta L2 norm starts at +inf, and should decreases with each iteration.
    If it gets close to 1, the iterations aren't making progress and can be stopped.
    A value of 1.001 means the L2 norm has only decrease by 0.1% between iterations and
    RMSE has only decreased by 0.05%.

    Parameters
    ----------
    current_rmse : float
        root mean square error of the residual gravity misfit
    last_l2_norm : float
        l2 norm of the residual gravity misfit
    Returns
    -------
    updated_l2_norm : float
        the updated l2 norm
    updated_delta_l2_norm : float
        the updated delta l2 norm
    """

    # square-root of current RMSE is the updated L-2 norm
    updated_l2_norm = np.sqrt(current_rmse)

    # updated delta l2 norm is the ratio of the last l2 norm to the current l2 norm
    updated_delta_l2_norm = last_l2_norm / updated_l2_norm

    return updated_l2_norm, updated_delta_l2_norm


def end_inversion(
    iteration_number: int,
    max_iterations: int,
    l2_norms: list[float],
    l2_norm_tolerance: float,
    delta_l2_norm: float,
    previous_delta_l2_norm: float,
    delta_l2_norm_tolerance: float,
    perc_increase_limit: float,
) -> tuple[bool, list[str]]:
    """
    check if the inversion should be terminated

    Parameters
    ----------
    iteration_number : int
        the iteration number, starting at 1 not 0
    max_iterations : int
        the maximum allowed iterations, inclusive and starting at 1
    l2_norms : float
        a list of each iteration's l2 norm
    l2_norm_tolerance : float
        the l2 norm value to end the inversion at
    delta_l2_norm : float
        the current iteration's delta l2 norm
    previous_delta_l2_norm : float
        the delta l2 norm of the previous iteration
    delta_l2_norm_tolerance : float
        the delta l2 norm value to end the inversion at
    perc_increase_limit : float
        the set tolerance for decimal percentage increase relative to the starting l2
        norm

    Returns
    -------
    end : bool
        whether or not to end the inversion
    termination_reason : list[str]
        a list of termination reasons
    """
    end = False
    termination_reason = []

    l2_norm = l2_norms[-1]

    # ignore for first iteration
    if iteration_number == 1:
        pass
    else:
        if l2_norm > np.min(l2_norms) * (1 + perc_increase_limit):
            log.info(
                "\nInversion terminated after %s iterations because L2 norm (%s) \n"
                "was over %s times greater than minimum L2 norm (%s) \n"
                "Change parameter 'perc_increase_limit' if desired.",
                iteration_number,
                l2_norm,
                1 + perc_increase_limit,
                np.min(l2_norms),
            )
            end = True
            termination_reason.append("l2-norm increasing")

        if (delta_l2_norm <= delta_l2_norm_tolerance) & (
            previous_delta_l2_norm <= delta_l2_norm_tolerance
        ):
            log.info(
                "\nInversion terminated after %s iterations because there was no "
                "significant variation in the L2-norm over 2 iterations \n"
                "Change parameter 'delta_l2_norm_tolerance' if desired.",
                iteration_number,
            )

            end = True
            termination_reason.append("delta l2-norm tolerance")

        if l2_norm < l2_norm_tolerance:
            log.info(
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
        log.info(
            "\nInversion terminated after %s iterations with L2-norm=%s because "
            "maximum number of iterations (%s) reached.",
            iteration_number,
            round(l2_norm, 2),
            max_iterations,
        )

        end = True
        termination_reason.append("max iterations")

    if "max iterations" in termination_reason:
        msg = (
            "Inversion terminated due to max_iterations limit. Consider increasing "
            "this limit."
        )
        log.warning(msg)

    return end, termination_reason


def update_gravity_and_misfit(
    gravity_df: pd.DataFrame,
    prisms_ds: xr.Dataset,
    iteration_number: int,
) -> pd.DataFrame:
    """
    calculate the forward gravity of the supplied prism layer, add the results to a
    new dataframe column, and update the residual misfit. The supplied gravity dataframe
    needs a 'reg' column, which describes the regional component and can be 0.

    Parameters
    ----------
    gravity_df : pandas.DataFrame
        gravity dataframe with gravity observation coordinate columns ('easting',
        'northing'), a gravity data column 'gravity_anomaly', and a regional gravity
        column ('reg').
    prisms_ds : xarray.Dataset
        harmonica prism layer
    iteration_number : int
        iteration number to use in updated column names

    Returns
    -------
    pandas.DataFrame
        a gravity dataframe with 2 new columns, one for the iterations forward gravity
        and one for the iterations residual misfit.
    """
    gravity = gravity_df.copy()

    # update the forward gravity
    gravity[f"iter_{iteration_number}_forward_grav"] = prisms_ds.prism_layer.gravity(
        coordinates=(gravity.easting, gravity.northing, gravity.upward),
        field="g_z",
    )

    # each iteration updates the topography of the layer to minimize the residual
    # portion of the misfit. We then want to recalculate the forward gravity of the
    # new layer, use the same original regional misfit, and re-calculate the residual
    # Gmisfit  = Gobs - Gforward
    # Gres = Gmisfit - Greg
    # Gres = Gobs - Gforward - Greg

    # update the residual misfit with the new forward gravity and the same regional
    gravity[f"iter_{iteration_number}_final_misfit"] = (
        gravity.gravity_anomaly
        - gravity[f"iter_{iteration_number}_forward_grav"]
        - gravity.reg
    )

    return gravity


def run_inversion(
    grav_df: pd.DataFrame,
    prism_layer: xr.Dataset,
    max_iterations: int,
    l2_norm_tolerance: float = 0.2,
    delta_l2_norm_tolerance: float = 1.001,
    perc_increase_limit: float = 0.20,
    deriv_type: str = "annulus",
    jacobian_prism_size: float = 1,
    solver_type: str = "scipy least squares",
    solver_damping: float | None = None,
    upper_confining_layer: xr.DataArray | None = None,
    lower_confining_layer: xr.DataArray | None = None,
    apply_weighting_grid: bool = False,
    weighting_grid: xr.DataArray | None = None,
    plot_convergence: bool = False,
    plot_dynamic_convergence: bool = False,
    results_fname: str | None = None,
    progressbar: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float]:
    """
    perform a geometric inversion, where the topography is updated to minimize the
    residual misfit between the forward gravity of the layer, and the observed gravity.
    To aid in regularizing an ill-posed problem choose any of the following options:
    * add damping to the solver, with `solver_damping`
    * weight the surface correction values with a weighting grid with
    `apply_weighting_grid` and the `weighting_grid` argument
    * bound the topography of the layer, with `upper_confining_layer` and
    `lower_confining_layer`

    Parameters
    ----------
    grav_df : pandas.DataFrame
        dataframe with gravity data and coordinates, must have columns
        "gravity_anomaly", "misfit", "res" and "reg", and coordinate columns "easting"
        and "northing".
    prism_layer : xarray.Dataset
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
    upper_confining_layer : xarray.DataArray | None, optional
        topographic layer to use as upper limit for inverted topography, by default None
    lower_confining_layer : xarray.DataArray | None, optional
        topographic layer to use as lower limit for inverted topography, by default None
    apply_weighting_grid : bool, optional
        use "weighting_grid" to scale surface corrections grid, by
        default False, by default False
    plot_convergence : bool, optional
        plot the misfit convergence, by default False
    plot_dynamic_convergence : bool, optional
        plot the misfit convergence dynamically, by default False
    results_fname : str, optional
        filename to save results to, by default None
    progressbar : bool, optional
        set to False to turn off the inversion progressbar

    Returns
    -------
    prisms_df : pandas.DataFrame
        prism properties for each iteration, including columns `starting_topo` for the
        unchanged starting topography model and `topo` for the updated topography model
    gravity : pandas.DataFrame
        gravity anomalies for each iteration
    params : dict
        Properties of the inversion such as kwarg values
    elapsed_time : float
        time in seconds for the inversion to run
    """
    cols = [
        "easting",
        "northing",
        "gravity_anomaly",
        "misfit",
        "reg",
        "res",
    ]
    if all(i in grav_df.columns for i in cols) is False:
        msg = f"`grav_df` needs all the following columns: {cols}"
        raise ValueError(msg)

    utils._check_gravity_inside_topography_region(grav_df, prism_layer)  # pylint: disable=protected-access

    # check no nans in gravity df
    if grav_df.res.isnull().values.any():
        msg = "gravity dataframe contains NaN values in the 'res' column"
        raise ValueError(msg)

    log.info("starting inversion")

    time_start = time.perf_counter()

    gravity = copy.deepcopy(grav_df)

    # use prism layer to re-create starting topography model and save it as variables
    # 'starting_topo', which remains unchanged, and 'topo', which is updated at the end
    # of each iteration. Also extract the prism spacing.
    (
        prisms_df,
        prisms_ds,
        prism_spacing,
        _,
    ) = utils.extract_prism_data(prism_layer)

    # extract density contrast value(s) (only used in parameter list for plots)
    density_contrast = np.unique(np.abs(prism_layer.density))

    # create empty jacobian matrix
    empty_jac: NDArray = np.empty(
        (len(gravity.gravity_anomaly), prisms_ds.top.size),
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
    delta_l2_norm = np.inf

    # iteration times
    iter_times = []

    l2_norms = []
    delta_l2_norms = []

    if progressbar is True:
        pbar = tqdm(range(max_iterations), initial=1, desc="Iteration")
    elif progressbar is False:
        pbar = range(max_iterations)
    else:
        msg = "progressbar must be a boolean"  # type: ignore[unreachable]
        raise ValueError(msg)

    for iteration, _ in enumerate(pbar, start=1):
        log.info("\n #################################### \n iteration %s", iteration)
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
            starting_misfit = initial_rmse

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

        # log correction values
        log.info(
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

        # warn if weighting grid supplied by unused
        if (weighting_grid is not None) & (apply_weighting_grid is False):
            msg = (
                "weighting grid supplied but not used because apply_weighting_grid is "
                "False"
            )
            raise ValueError(msg)

        # apply weights to the topo correction grid
        if apply_weighting_grid is True:
            if weighting_grid is None:
                msg = "must supply weighting grid if apply_weighting_grid is True"
                raise ValueError(msg)
            correction_grid = correction_grid * weighting_grid

        # add the corrections to the topo and update the prisms dataset
        prisms_ds = utils.update_prisms_ds(prisms_ds, correction_grid)

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
            iteration,
        )

        # update the misfit RMSE
        updated_rmse = utils.rmse(gravity[f"iter_{iteration}_final_misfit"])
        log.info("updated misfit RMSE: %s", round(updated_rmse, 4))
        final_rmse = updated_rmse

        # update the l2 and delta l2 norms
        previous_delta_l2_norm = copy.copy(delta_l2_norm)
        l2_norm, delta_l2_norm = update_l2_norms(
            current_rmse=updated_rmse,
            last_l2_norm=l2_norm,
        )
        final_l2_norm = l2_norm

        l2_norms.append(l2_norm)
        delta_l2_norms.append(delta_l2_norm)

        log.info(
            "updated L2-norm: %s, tolerance: %s", round(l2_norm, 4), l2_norm_tolerance
        )
        log.info(
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
            l2_norms,
            l2_norm_tolerance,
            delta_l2_norm,
            previous_delta_l2_norm,
            delta_l2_norm_tolerance,
            perc_increase_limit,
        )

        if plot_dynamic_convergence is True:
            plotting.plot_dynamic_convergence(
                l2_norms,
                l2_norm_tolerance,
                delta_l2_norms,
                delta_l2_norm_tolerance,
                starting_misfit,  # pylint: disable=possibly-used-before-assignment
            )

        if end is True:
            if progressbar is True:
                pbar.set_description(f"Inversion ended due to {termination_reason}")
            break
        # end of inversion loop

    time_end = time.perf_counter()

    elapsed_time = time_end - time_start

    # collect input parameters into a dictionary
    params = {
        # first column
        "Density contrast(s)": f"{density_contrast} kg/m3",
        "Reference level": f"{prisms_ds.attrs.get('zref')} m",
        "Max iterations": max_iterations,
        "L2 norm tolerance": f"{l2_norm_tolerance}",
        "Delta L2 norm tolerance": f"{delta_l2_norm_tolerance}",
        # second column
        "Deriv type": deriv_type,
        "Solver type": solver_type,
        "Solver damping": solver_damping,
        "Upper confining layer": "Not enabled"
        if upper_confining_layer is None
        else "Enabled",
        "Lower confining layer": "Not enabled"
        if lower_confining_layer is None
        else "Enabled",
        "Regularization weighting grid": "Not enabled"
        if apply_weighting_grid is False
        else "Enabled",
        # third column
        "Time elapsed": f"{int(elapsed_time)} seconds",
        "Avg. iteration time": f"{round(np.mean(iter_times), 2)} seconds",
        "Final misfit RMSE / L2-norm": (
            f"{round(final_rmse,4)} /{round(final_l2_norm,4)} mGal"
        ),
        "Termination reason": termination_reason,
        "Iteration times": iter_times,
    }

    if plot_convergence is True:
        plotting.plot_convergence(
            gravity,
            params,
        )

    results = prisms_df, gravity, params, elapsed_time
    if results_fname is not None:
        # remove if exists
        pathlib.Path(f"{results_fname}.pickle").unlink(missing_ok=True)
        with pathlib.Path(f"{results_fname}.pickle").open("wb") as f:
            pickle.dump(results, f)
        log.info("results saved to %s.pickle", results_fname)

    return results


def run_inversion_workflow(
    grav_df: pd.DataFrame,
    create_starting_topography: bool = False,
    create_starting_prisms: bool = False,
    calculate_starting_gravity: bool = False,
    calculate_regional_misfit: bool = False,
    run_damping_cv: bool = False,
    run_zref_or_density_cv: bool = False,
    run_kfolds_zref_or_density_cv: bool = False,
    plot_cv: bool = False,
    fname: str | None = None,
    **kwargs: typing.Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float]:
    """
    This function runs the full inversion workflow. Depending on the input parameters,
    it will:
    1) create a starting topography model
    2) create a starting prism model
    3) calculate the starting gravity of the prism model
    4) calculate the gravity misfit
    5) calculate the regional and residual components of the misfit
    6) run the inversion
    you can choose to run cross validations for damping, density, and zref

    Parameters
    ----------
    grav_df : pandas.DataFrame
        gravity dataframe with gravity data, must have coordinate columns 'easting', and
        'northing'. It must also have a gravity data column 'gravity_anomaly'.
        Optionally should have columns 'starting_gravity', 'misfit', 'reg', 'res'.
    create_starting_topography : bool, optional
        Choose whether to create starting topography model. If True, must provide
        `starting_topography_kwargs`, if False must provide `starting_topography` by
        default False
    create_starting_prisms : bool, optional
        Choose whether to create starting prisms model. If False, must provide prisms
        model, by default False
    calculate_starting_gravity : bool, optional
        Choose whether to calculate starting gravity from prisms model. If False, must
        provide column "starting_gravity" in grav_df , by default False
    calculate_regional_misfit : bool, optional
        Choose whether to calculate regional misfit. If False, must provide column "reg"
        in grav_df, if True, must provide`regional_grav_kwargs`, by default False
    run_damping_cv : bool, optional
        Choose whether to run cross validation for damping, if True, must supplied
        damping values with kwarg `damping_limits`, by default False
    run_zref_or_density_cv : bool, optional
        Choose whether to run cross validation for zref or density, if True, must
        provide zref values, density values, or both  with kwargs `zref_values` or `
        density_values`, by default False
    run_kfolds_zref_or_density_cv : bool, optional
        Choose whether to run internal kfolds cross validation for zref or density, if
        True, must provide kwargs for splitting constraints into test/train points with
        `split_kwargs`, by default False
    plot_cv : bool, optional
        Choose whether to plot the cross validation results, by default False
    fname : str, optional
        filename and path to use for saving results. If running a damping
        CV, will save the study to <fname>_damping_cv_study.pickle and the tuple of the
        best inversion results to <fname>_damping_cv_results.pickle. If running a
        density/zref CV, will save the study to <fname>_density_zref_cv_study.pickle and
        the tuple of the best inversion results to
        <fname>_density_zref_cv_results.pickle. The final inversion result for all
        methods will be saved to <fname>_results.pickle, by default will be "tmp_<x>"
        where x is a random integer between 0 and 999.
    kwargs : typing.Any
        keyword arguments for the workflow and inversion, such as
        `starting_topography_kwargs`, `regional_grav_kwargs`,
        `split_kwargs` and all the other kwargs
        supplied to `run_inversion`.

    Returns
    -------
    prisms_df : pandas.DataFrame
        prism properties for each iteration, including columns `starting_topo` for the
        unchanged starting topography model and `topo` for the updated topography model
    gravity : pandas.DataFrame
        gravity anomalies for each iteration
    params : dict
        Properties of the inversion such as kwarg values
    elapsed_time : float
        time in seconds for the inversion to run
    """

    kwargs = copy.deepcopy(kwargs)
    grav_df = grav_df.copy()

    # get kwargs
    starting_topography = kwargs.pop("starting_topography", None)
    starting_prisms = kwargs.pop("starting_prisms", None)
    starting_topography_kwargs = kwargs.pop("starting_topography_kwargs", None)
    score_as_median = kwargs.pop("score_as_median", False)
    grid_search = kwargs.pop("grid_search", False)
    regional_grav_kwargs = kwargs.pop("regional_grav_kwargs", None)
    zref = kwargs.pop("zref", None)
    density_contrast = kwargs.pop("density_contrast", None)
    constraints_df = kwargs.pop("constraints_df", None)
    zref_density_cv_trials = kwargs.pop("zref_density_cv_trials", None)
    density_contrast_limits = kwargs.pop("density_contrast_limits", None)
    zref_limits = kwargs.pop("zref_limits", None)
    split_kwargs = kwargs.pop("split_kwargs", None)
    data_spacing = kwargs.pop("grav_spacing", None)
    inversion_region = kwargs.pop("inversion_region", None)
    damping_limits = kwargs.pop("damping_limits", (0.001, 1))
    damping_cv_trials = kwargs.pop("damping_cv_trials", None)
    starting_grav_kwargs = kwargs.pop("starting_grav_kwargs", None)

    # set file name for saving results with random number between 0 and 999
    if fname is None:
        fname = f"tmp_{random.randint(0,999)}"

    log.info("saving all results with root name '%s'", fname)
    ###
    ###
    # figure out what needs to be done
    ###
    ###
    # pylint: disable=duplicate-code
    # if creating starting topo, must also create starting prisms
    if create_starting_topography is True:
        create_starting_prisms = True
    # if creating starting prisms, must also calculate starting gravity
    if create_starting_prisms is True:
        calculate_starting_gravity = True
    # if calculating starting gravity, must also calculate gravity misfit
    if calculate_starting_gravity is True:
        calculate_regional_misfit = True
    log.debug("creating starting topo: %s", create_starting_topography)
    log.debug("creating starting prisms: %s", create_starting_prisms)
    log.debug("calculating starting gravity: %s", calculate_starting_gravity)
    log.debug("calculating regional misfit: %s", calculate_regional_misfit)

    # pylint: enable=duplicate-code
    ###
    ###
    # check needed inputs are provided at the beginning
    ###
    ###
    if (calculate_regional_misfit is True) or (run_zref_or_density_cv is True):  # noqa: SIM102
        if regional_grav_kwargs is None:
            msg = (
                "regional_grav_kwargs must be provided if recalculating regional "
                "gravity or performing zref or density CV"
            )
            raise ValueError(msg)
    if (create_starting_prisms is True) or (run_zref_or_density_cv is True):  # noqa: SIM102
        if create_starting_prisms is True:
            if density_contrast is None:
                msg = "density must be provided if create_starting_prisms is True"
                raise ValueError(msg)
            if zref is None:
                msg = "zref must be provided if create_starting_prisms is True"
                raise ValueError(msg)
    if run_zref_or_density_cv is True:
        if constraints_df is None:
            msg = "must provide constraints_df if run_zref_or_density_cv is True"
            raise ValueError(msg)
        if zref_density_cv_trials is None:
            msg = (
                "must provide zref_density_cv_trials if run_zref_or_density_cv is True"
            )
            raise ValueError(msg)
        if run_kfolds_zref_or_density_cv is True:
            if split_kwargs is None:
                msg = "split_kwargs must be provided if performing internal kfolds CV"
                raise ValueError(msg)
        else:
            if "constraints_df" in regional_grav_kwargs:
                msg = (
                    "if performing density/zref CV, it's best to not use constraints "
                    "in the regional separation"
                )
                log.warning(msg)
        if density_contrast_limits is None:
            assert density_contrast is not None
        if zref_limits is None:
            assert zref is not None
        if (density_contrast_limits is None) & (zref_limits is None):
            msg = (
                "must provide density_contrast_limits or zref_limits if run_zref_or_"
                "density_cv is True"
            )
            raise ValueError(msg)
    if run_damping_cv is True:
        if (
            ("test" in grav_df.columns)
            and (False in grav_df.test.unique())
            and (True in grav_df.test.unique())
        ):
            pass
        else:
            # resample to half spacing
            if data_spacing is None:
                msg = "need to supply `grav_spacing` if `run_damping_cv` is True"
                raise ValueError(msg)
            if inversion_region is None:
                msg = "need to supply `inversion_region` if `run_damping_cv` is True"
                raise ValueError(msg)
            grav_df = cross_validation.resample_with_test_points(
                data_spacing=data_spacing,
                data=grav_df,
                region=inversion_region,
            )
        if damping_limits is None:
            msg = "must provide damping_limits if run_damping_cv is True"
            raise ValueError(msg)
        if damping_cv_trials is None:
            msg = "must provide damping_cv_trials if run_zref_or_density_cv is True"
            raise ValueError(msg)

    # Starting Topography
    if create_starting_topography is False:
        if (
            (starting_topography is None)
            & (starting_prisms is None)
            & (create_starting_prisms is False)
        ):
            msg = (
                "starting_topography must be provided since create_starting_topography "
                "is False, create_starting_prisms is False, and starting_prisms is not "
                "provided"
            )
            raise ValueError(msg)
        log.debug("not creating starting topo because it is provided")
    elif create_starting_topography is True:
        if starting_topography is not None:
            msg = (
                "starting_topography provided but unused since "
                "create_starting_topography is True"
            )
            log.warning(msg)
        if starting_topography_kwargs is None:
            msg = (
                "starting_topography_kwargs must be provided if "
                "create_starting_topography is True"
            )
            raise ValueError(msg)

        with utils._log_level(logging.WARN):  # pylint: disable=protected-access
            # create the starting topography
            starting_topography = utils.create_topography(
                **starting_topography_kwargs,
            )
        log.debug("starting topo created")

    # Starting Prism Model
    if create_starting_prisms is False:
        if (starting_prisms is None) & (run_zref_or_density_cv is False):
            msg = "starting_prisms must be provided if create_starting_prisms is False"
            raise ValueError(msg)
        log.debug("not creating starting prisms because it is provided")
    elif create_starting_prisms is True:
        if starting_prisms is not None:
            msg = (
                "starting_prisms provided but unused since create_starting_prisms is "
                "True"
            )
            log.warning(msg)
        if starting_topography is None:
            msg = (
                "starting_topography must be provided if create_starting_prisms is True"
                " and create_starting_topography is False"
            )
            raise ValueError(msg)

        density_grid = xr.where(
            starting_topography >= zref,
            density_contrast,
            -density_contrast,
        )
        starting_prisms = utils.grids_to_prisms(
            starting_topography,
            reference=zref,
            density=density_grid,
        )
        log.debug("starting prisms created")
    # Starting Gravity of Prism Model
    if calculate_starting_gravity is False:
        if "starting_gravity" not in grav_df:
            msg = (
                "'starting_gravity' must be a column of `grav_df` if "
                "calculate_starting_gravity is False"
            )
            raise ValueError(msg)
        log.debug("not calculating starting gravity because it is provided")
    elif calculate_starting_gravity is True:
        if "starting_gravity" in grav_df:
            msg = (
                "'starting_gravity' already a column of `grav_df`, but is being "
                "overwritten since calculate_starting_gravity is True"
            )
            log.warning(msg)
        if starting_grav_kwargs is None:
            starting_grav_kwargs = {
                "field": "g_z",
                "coordinates": (
                    grav_df.easting,
                    grav_df.northing,
                    grav_df.upward,
                ),
                "progressbar": False,
            }
        log.debug("calculating starting gravity")
        grav_df["starting_gravity"] = starting_prisms.prism_layer.gravity(
            **starting_grav_kwargs,
        )
        log.debug("starting gravity calculated")

    # Regional Component of Misfit
    if calculate_regional_misfit is False:
        if ("misfit" not in grav_df) & (run_zref_or_density_cv is False):
            msg = (
                "'misfit' must be a column of `grav_df` if calculate_regional_misfit is"
                " False"
            )
            raise ValueError(msg)
        if ("reg" not in grav_df) & (run_zref_or_density_cv is False):
            msg = (
                "'reg' must be a column of `grav_df` if calculate_regional_misfit is"
                " False"
            )
            raise ValueError(msg)
        log.debug("not calculating regional misfit because it is provided")
    elif calculate_regional_misfit is True:
        if "reg" in grav_df:
            msg = (
                "'reg' already a column of `grav_df`, but is being overwritten since"
                " calculate_regional_misfit is True"
            )
            log.warning(msg)
        log.debug("calculating regional misfit")
        log.debug("regional_grav_kwargs: %s", regional_grav_kwargs)

        # pop out grav_df if in regional_grav_kwargs
        regional_grav_kwargs.pop("grav_df", None)

        grav_df = regional.regional_separation(
            grav_df=grav_df,
            **regional_grav_kwargs,
        )
        log.debug("regional misfit calculated")

    inversion_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key
        not in [
            "starting_prisms",
            "starting_grav_kwargs",
            "grav_spacing",
            "constraints_df",
            "inversion_region",
            "grav_df",
            "progressbar",
            "prism_layer",
        ]
    }

    log.debug("inversion kwargs: %s", inversion_kwargs)

    ###
    ###
    # SINGLE INVERSION
    ###
    ###
    # run only the inversion with specified damping, density, and zref values
    if (run_damping_cv is False) & (run_zref_or_density_cv is False):
        log.info("running individual inversion")
        utils._check_gravity_inside_topography_region(grav_df, starting_prisms)  # pylint: disable=protected-access
        if inversion_kwargs.get("plot_dynamic_convergence", False) is True:
            with utils._log_level(logging.WARN):  # pylint: disable=protected-access
                inversion_results = run_inversion(
                    grav_df=grav_df,
                    prism_layer=starting_prisms,
                    progressbar=False,
                    results_fname=f"{fname}_results",
                    **inversion_kwargs,
                )
        else:
            inversion_results = run_inversion(
                grav_df=grav_df,
                prism_layer=starting_prisms,
                progressbar=False,
                results_fname=f"{fname}_results",
                **inversion_kwargs,
            )
        return inversion_results

    ###
    ###
    # DAMPING CV
    ###
    ###
    if run_damping_cv is True:
        log.info("running damping cross validation")
        utils._check_gravity_inside_topography_region(grav_df, starting_prisms)  # pylint: disable=protected-access
        study, inversion_results = optimization.optimize_inversion_damping(
            training_df=grav_df[grav_df.test == False],  # noqa: E712 pylint: disable=singleton-comparison
            testing_df=grav_df[grav_df.test == True],  # noqa: E712 pylint: disable=singleton-comparison
            damping_limits=damping_limits,
            n_trials=damping_cv_trials,
            grid_search=grid_search,
            plot_grids=False,
            fname=f"{fname}_damping_cv",
            prism_layer=starting_prisms,
            score_as_median=score_as_median,
            plot_cv=plot_cv,
            **inversion_kwargs,
        )
        # use the best damping parameter if performing zref/density CV
        best_damping = study.best_params.get("damping")
        inversion_kwargs["solver_damping"] = best_damping

        # save inversion results to pickle
        pathlib.Path(f"{fname}_results.pickle").unlink(missing_ok=True)
        with pathlib.Path(f"{fname}_results.pickle").open("wb") as f:
            pickle.dump(inversion_results, f)
        log.info("results saved to %s.pickle", f"{fname}_results.pickle")

        assert best_damping == inversion_results[2]["Solver damping"]

        if run_zref_or_density_cv is False:
            return inversion_results

    ###
    ###
    # DENSITY / ZREF CV
    ###
    ###
    log.info("running zref and/or density contrast cross validation")
    # drop the testing data
    if "test" in grav_df.columns:
        grav_df = grav_df[grav_df.test == False].copy()  # noqa: E712 pylint: disable=singleton-comparison
        grav_df = grav_df.drop(columns=["test"])
        log.debug("dropped testing data")

    zref_density_parameters = dict(  # pylint: disable=use-dict-literal
        grav_df=grav_df,
        density_contrast_limits=density_contrast_limits,
        zref_limits=zref_limits,
        density_contrast=density_contrast,
        zref=zref,
        n_trials=zref_density_cv_trials,
        starting_topography=starting_topography,
        regional_grav_kwargs=regional_grav_kwargs,
        grid_search=grid_search,
        fname=f"{fname}_zref_density_cv",
        score_as_median=score_as_median,
        plot_cv=plot_cv,
        **inversion_kwargs,
    )

    utils._check_constraints_inside_gravity_region(constraints_df, grav_df)  # pylint: disable=protected-access
    # if chosen, run an internal K-folds CV for regional separation within the
    # density/Zref CV
    if run_kfolds_zref_or_density_cv is True:
        log.info("running internal K-folds CV for regional separation")
        study, inversion_results = (
            optimization.optimize_inversion_zref_density_contrast_kfolds(
                constraints_df=constraints_df,
                fold_progressbar=True,
                split_kwargs=split_kwargs,
                **zref_density_parameters,
            )
        )
    else:
        # run the normal non-kfolds CV
        study, inversion_results = (
            optimization.optimize_inversion_zref_density_contrast(
                constraints_df=constraints_df,
                fold_progressbar=False,
                **zref_density_parameters,
            )
        )

    # save inversion results to pickle
    pathlib.Path(f"{fname}_results.pickle").unlink(missing_ok=True)
    with pathlib.Path(f"{fname}_results.pickle").open("wb") as f:
        pickle.dump(inversion_results, f)
    log.info("results saved to %s", f"{fname}_results.pickle")

    # ensure final inversion used the best parameters
    best_trial = study.best_trial
    best_zref = best_trial.params.get("zref", zref)
    best_density_contrast = best_trial.params.get("density_contrast", density_contrast)

    used_zref = float(inversion_results[2]["Reference level"][:-2])
    used_density_contrast = float(inversion_results[2]["Density contrast(s)"][1:-7])

    assert math.isclose(used_density_contrast, best_density_contrast, rel_tol=0.02)
    assert math.isclose(used_zref, best_zref, rel_tol=0.02)

    if run_damping_cv is True:
        assert inversion_results[2]["Solver damping"] == best_damping

    return inversion_results
