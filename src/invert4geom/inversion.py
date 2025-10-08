import copy  # pylint: disable=too-many-lines
import itertools
import logging
import math
import pathlib
import pickle
import random
import time
import typing

import harmonica as hm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numba
import numpy as np
import optuna
import pandas as pd
import scipy as sp
import seaborn as sns
import verde as vd
import xarray as xr
from IPython.display import clear_output
from numpy.typing import NDArray
from polartoolkit import maps
from polartoolkit import utils as polar_utils
from tqdm.autonotebook import tqdm

from invert4geom import (
    cross_validation,
    logger,
    optimization,
    plotting,
    regional,
    utils,
)


@numba.jit(cache=True, nopython=True)
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


@numba.njit(parallel=True)
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
    annulus :footcite:p:`mccubbineairborne2016`, and calculates it's derivative of
    vertical gravity with respect to thickness of the prisms/tesseroids. Takes arrays
    from `jacobian`, feeds them into `grav_column_der`, and returns the jacobian.

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
    prisms_layer: xr.Dataset,  # noqa: ARG001
    method: str = "itertools",  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `_model_properties` function instead
    """
    # pylint: disable=W0613
    msg = (
        "Function `_prism_properties` deprecated, use the `_model_properties` function "
        "instead"
    )
    raise DeprecationWarning(msg)


def _model_properties(
    layer: xr.Dataset,
    method: str = "itertools",
) -> NDArray:
    """
    extract properties from prism or tesseroid layer

    Parameters
    ----------
    layer : xarray.Dataset
       harmonica prism or tesseroid layer
    method : str, optional
        choice of method to extract properties, by default "itertools"

    Returns
    -------
    numpy.ndarray
        array of layer properties
    """

    if method == "itertools":
        layer_properties = []
        for (
            y,
            x,
        ) in itertools.product(range(layer.northing.size), range(layer.easting.size)):
            if layer.model_type == "prisms":
                layer_properties.append(
                    [
                        *list(layer.prism_layer.get_prism((y, x))),
                        layer.density.to_numpy()[y, x],
                    ]
                )
            elif layer.model_type == "tesseroids":
                layer_properties.append(
                    [
                        *list(layer.tesseroid_layer.get_tesseroid((y, x))),
                        layer.density.to_numpy()[y, x],
                    ]
                )
        layer_properties = np.array(layer_properties)
    elif method == "forloops":
        layer_properties = []
        for y in range(layer.northing.size):
            for x in range(layer.easting.size):
                if layer.model_type == "prisms":
                    layer_properties.append(
                        [
                            *list(layer.prism_layer.get_prism((y, x))),
                            layer.density.to_numpy()[y, x],
                        ]
                    )
                elif layer.model_type == "tesseroids":
                    layer_properties.append(
                        [
                            *list(layer.tesseroid_layer.get_tesseroid((y, x))),
                            layer.density.to_numpy()[y, x],
                        ]
                    )
        np.asarray(layer_properties)
    elif method == "generator":
        # slower, but doesn't allocate memory
        if layer.model_type == "prisms":
            layer_properties = [
                list(layer.prism_layer.get_prism((y, x)))  # noqa: RUF005
                + [layer.density.to_numpy()[y, x]]
                for y in range(layer.northing.size)
                for x in range(layer.easting.size)
            ]
        elif layer.model_type == "tesseroids":
            layer_properties = [
                list(layer.tesseroid_layer.get_tesseroid((y, x)))  # noqa: RUF005
                + [layer.density.to_numpy()[y, x]]
                for y in range(layer.northing.size)
                for x in range(layer.easting.size)
            ]
    else:
        msg = "method must be one of 'itertools', 'forloops', or 'generator'"
        raise ValueError(msg)

    return layer_properties


def jacobian_prism(
    prisms_properties: NDArray,  # noqa: ARG001
    grav_easting: NDArray,  # noqa: ARG001
    grav_northing: NDArray,  # noqa: ARG001
    grav_upward: NDArray,  # noqa: ARG001
    delta: float,  # noqa: ARG001
    jac: NDArray,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `jacobian_finite_difference_prisms` function instead
    """
    # pylint: disable=W0613
    msg = (
        "Function `jacobian_prism` deprecated, use the `jacobian_finite_difference_prisms` function "
        "instead"
    )
    raise DeprecationWarning(msg)


def jacobian_finite_difference_prisms(
    model_properties: NDArray,
    grav_easting: NDArray,
    grav_northing: NDArray,
    grav_upward: NDArray,
    delta: float,
    jac: NDArray,
) -> NDArray:
    """
    Function to calculate the Jacobian matrix where each entry is the derivative of
    vertical gravity with respect to thickness of the prisms/tesseroids, calculated
    using a numerical approximation with small prisms/tesseroids added on top of the
    existing model.

    Takes arrays from `jacobian` and calculates the jacobian.

    Parameters
    ----------
    model_properties : numpy.ndarray
        array of prism/tesseroid properties of shape (number of prisms, 7) with the 7 entries for
        each prism being: west, east, south, north, bottom, top, density
    grav_easting, grav_northing,grav_upward : numpy.ndarray
        coordinates of gravity observation points.
    delta : float
        thickness in meters of small prisms/tesseroids used to calculate vertical derivative
    jac : numpy.ndarray
        empty jacobian matrix with a row per gravity observation and a column per prism

    Returns
    -------
    numpy.ndarray
        returns a numpy.ndarray of shape (number of gravity points, number of prisms)
    """

    # Add a small model element on top of existing model element with thickness equal
    # to delta
    for i in numba.prange(len(model_properties)):  # pylint: disable=not-an-iterable
        element = model_properties[i]
        density = element[6]
        bottom = element[5]  # new prism bottom is top of old prism
        top = element[5] + delta  # new prism top is old prism top + delta
        # bottom = element[4] - delta / 2 # new prism bottom is old prism bottom - delta/2
        # top = element[4] + delta / 2 # new prism top is old prism bottom + delta/2
        delta_element = (element[0], element[1], element[2], element[3], bottom, top)
        jac[:, i] = (
            hm.prism_gravity(
                coordinates=(grav_easting, grav_northing, grav_upward),
                prisms=delta_element,
                density=density,
                field="g_z",
                parallel=True,
            )
            / delta
        )

    return jac


def jacobian_finite_difference_tesseroids(
    model_properties: NDArray,
    grav_easting: NDArray,
    grav_northing: NDArray,
    grav_upward: NDArray,
    delta: float,
    jac: NDArray,
) -> NDArray:
    """
    Function to calculate the Jacobian matrix where each entry is the derivative of
    vertical gravity with respect to thickness of the prisms/tesseroids, calculated
    using a numerical approximation with small prisms/tesseroids added on top of the
    existing model.

    Takes arrays from `jacobian` and calculates the jacobian.

    Parameters
    ----------
    model_properties : numpy.ndarray
        array of prism/tesseroid properties of shape (number of prisms, 7) with the 7 entries for
        each prism being: west, east, south, north, bottom, top, density
    grav_easting, grav_northing,grav_upward : numpy.ndarray
        coordinates of gravity observation points.
    delta : float
        thickness in meters of small prisms/tesseroids used to calculate vertical derivative
    jac : numpy.ndarray
        empty jacobian matrix with a row per gravity observation and a column per prism

    Returns
    -------
    numpy.ndarray
        returns a numpy.ndarray of shape (number of gravity points, number of prisms)
    """

    # Add a small model element on top of existing model element with thickness equal
    # to delta
    for i in numba.prange(len(model_properties)):  # pylint: disable=not-an-iterable
        element = model_properties[i]
        density = element[6]
        bottom = element[5]
        top = element[5] + delta
        delta_element = (element[0], element[1], element[2], element[3], bottom, top)

        jac[:, i] = (
            hm.tesseroid_gravity(
                coordinates=(grav_easting, grav_northing, grav_upward),
                tesseroids=delta_element,
                density=density,
                field="g_z",
                parallel=True,
            )
            / delta
        )

    return jac


def jacobian(
    deriv_type: str,  # noqa: ARG001
    coordinates: pd.DataFrame,  # noqa: ARG001
    empty_jac: NDArray | None = None,  # noqa: ARG001
    prisms_layer: xr.Dataset | None = None,  # noqa: ARG001
    prism_spacing: float | None = None,  # noqa: ARG001
    prism_size: float | None = None,  # noqa: ARG001
    prisms_properties_method: str = "itertools",  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `Inversion` class method `jacobian`  instead
    """
    # pylint: disable=W0613
    msg = (
        "Function `jacobian` deprecated, use the `Inversion` class method `jacobian` "
        "instead"
    )
    raise DeprecationWarning(msg)


def solver(
    jac: NDArray,  # noqa: ARG001
    residuals: NDArray,  # noqa: ARG001
    damping: float | None = None,  # noqa: ARG001
    solver_type: str = "scipy least squares",  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `Inversion` class method `solver`  instead
    """
    # pylint: disable=W0613
    msg = (
        "Function `solver` deprecated, use the `Inversion` class method `solver` "
        "instead"
    )
    raise DeprecationWarning(msg)


def update_l2_norms(
    current_rmse: float,  # noqa: ARG001
    last_l2_norm: float,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: function deprecated
    """
    # pylint: disable=W0613
    msg = "Function `update_l2_norms` has been deprecated"
    raise DeprecationWarning(msg)


def end_inversion(
    iteration_number: int,  # noqa: ARG001
    max_iterations: int,  # noqa: ARG001
    l2_norms: list[float],  # noqa: ARG001
    l2_norm_tolerance: float,  # noqa: ARG001
    delta_l2_norm: float,  # noqa: ARG001
    previous_delta_l2_norm: float,  # noqa: ARG001
    delta_l2_norm_tolerance: float,  # noqa: ARG001
    perc_increase_limit: float,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `Inversion` class method `end_inversion`  instead
    """
    # pylint: disable=W0613
    msg = (
        "Function `end_inversion` deprecated, use the `Inversion` class method "
        "`end_inversion` instead"
    )
    raise DeprecationWarning(msg)


def update_gravity_and_misfit(
    gravity_df: pd.DataFrame,  # noqa: ARG001
    prisms_ds: xr.Dataset,  # noqa: ARG001
    iteration_number: int,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: function deprecated
    """
    # pylint: disable=W0613
    msg = "Function `update_gravity_and_misfit` has been deprecated"
    raise DeprecationWarning(msg)


def run_inversion(
    grav_df: pd.DataFrame,  # noqa: ARG001
    prism_layer: xr.Dataset,  # noqa: ARG001
    max_iterations: int,  # noqa: ARG001
    l2_norm_tolerance: float = 0.2,  # noqa: ARG001
    delta_l2_norm_tolerance: float = 1.001,  # noqa: ARG001
    perc_increase_limit: float = 0.20,  # noqa: ARG001
    deriv_type: str = "annulus",  # noqa: ARG001
    jacobian_prism_size: float = 1,  # noqa: ARG001
    solver_type: str = "scipy least squares",  # noqa: ARG001
    solver_damping: float | None = None,  # noqa: ARG001
    upper_confining_layer: xr.DataArray | None = None,  # noqa: ARG001
    lower_confining_layer: xr.DataArray | None = None,  # noqa: ARG001
    apply_weighting_grid: bool = False,  # noqa: ARG001
    weighting_grid: xr.DataArray | None = None,  # noqa: ARG001
    plot_convergence: bool = False,  # noqa: ARG001
    plot_dynamic_convergence: bool = False,  # noqa: ARG001
    results_fname: str | None = None,  # noqa: ARG001
    progressbar: bool = True,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `Inversion` class method `run_inversion`  instead
    """
    # pylint: disable=W0613
    msg = (
        "Function `run_inversion` deprecated, use the `Inversion` class method "
        "`invert` instead"
    )
    raise DeprecationWarning(msg)


@xr.register_dataset_accessor("inv")
class Invert4GeomAccessor:
    """
    A class which allows adding properties and methods as xarray dataset accessors.
    """

    def __init__(
        self,
        ds: xr.Dataset,
    ) -> None:
        self._ds = ds

    @property
    def df(self) -> pd.DataFrame:
        """return the dataframe representation of the xarray dataset"""
        return self._ds.to_dataframe().reset_index()

    @property
    def inner_df(self) -> pd.DataFrame:
        """return the dataframe representation of the inner region of the xarray dataset"""
        return self.inner.to_dataframe().reset_index()

    @property
    def inner(self) -> xr.Dataset:
        """return only the inside region of the xarray dataset"""
        return self._ds.sel(
            easting=slice(self._ds.inner_region[0], self._ds.inner_region[1]),
            northing=slice(self._ds.inner_region[2], self._ds.inner_region[3]),
        )

    @property
    def masked_df(self) -> xr.Dataset:
        """update the dataframe from the masked xarray dataset"""
        if self._ds.dataset_type == "data":
            msg = (
                "The `masked_df` property is only available for the model dataset. "
                "Use the `inner_df` property to get the inner region of the data "
                "dataset."
            )
            raise ValueError(msg)
        return self.masked.to_dataframe().reset_index()

    @property
    def masked(self) -> xr.Dataset:
        """return only the model elements with a mask value of 1"""
        if self._ds.dataset_type == "data":
            msg = (
                "The `masked` property is only available for the model dataset. "
                "Use the `inner_df` property to get the inner region of the data "
                "dataset."
            )
            raise ValueError(msg)
        return self._ds.where(self._ds.mask == 1, drop=True)

    ###
    ###
    # Methods for the gravity data dataset
    ###
    ###

    def forward_gravity(
        self,
        layer: xr.Dataset,
        name: str = "forward_gravity",
        field: str = "g_z",
        progressbar: bool = False,
        **kwargs: typing.Any,
    ) -> None:
        """
        calculate the forward gravity of a prism model

        Parameters
        ----------
        layer : xr.Dataset
            a prism or tesseroid layer
        name : str, optional
            Name to assigned the variable for the calculated gravity, by default
            "forward_gravity"
        field : str, optional
            Choose which gravitational field to be calculated, by default "g_z" which is
            the downward acceleration.
        progressbar : bool, optional
            Display a progress bar of the calculation, by default False
        """

        if self._ds.dataset_type == "model":
            msg = "The `forward_gravity` method is only available for the data dataset."
            raise ValueError(msg)

        df = self.df

        if layer.model_type == "prisms":
            coord_names = ["northing", "easting"]
            df[name] = layer.prism_layer.gravity(
                coordinates=(df.easting, df.northing, df.upward),
                field=field,
                progressbar=progressbar,
                **kwargs,
            )
        elif layer.model_type == "tesseroids":
            coord_names = ["latitude", "longitude"]
            df[name] = layer.tesseroid_layer.gravity(
                coordinates=(df.longitude, df.latitude, df.upward),
                field=field,
                progressbar=progressbar,
                **kwargs,
            )

        else:
            msg = "layer must have attribute 'model_type' which is either 'prisms' or 'tesseroids'"
            raise ValueError(msg)

        ds = df.set_index(coord_names).to_xarray()
        self._ds[name] = ds[name]

    def starting_gravity(
        self,
        layer: xr.Dataset,
        field: str = "g_z",
        progressbar: bool = False,
        **kwargs: typing.Any,
    ) -> None:
        """calculate the forward gravity of the starting prism model"""
        if self._ds.dataset_type == "model":
            msg = (
                "The `starting_gravity` method is only available for the data dataset. "
            )
            raise ValueError(msg)
        df = self.df

        if layer.model_type == "prisms":
            df["starting_gravity"] = layer.prism_layer.gravity(
                coordinates=(df.easting, df.northing, df.upward),
                field=field,
                progressbar=progressbar,
                **kwargs,
            )
        elif layer.model_type == "tesseroids":
            df["starting_gravity"] = layer.tesseroid_layer.gravity(
                coordinates=(df.easting, df.northing, df.upward),
                field=field,
                progressbar=progressbar,
                **kwargs,
            )
        else:
            msg = "layer must have attribute 'model_type' which is either 'prisms' or 'tesseroids'"
            raise ValueError(msg)

        ds = df.set_index(["northing", "easting"]).to_xarray()
        self._ds["starting_gravity"] = ds.starting_gravity
        self._ds["forward_gravity"] = ds.starting_gravity

    def regional_separation(self, **kwargs: typing.Any) -> xr.Dataset:
        """perform a regional-residual separation"""
        if self._ds.dataset_type == "model":
            msg = "The `regional_separation` method is only available for the data dataset. "
            raise ValueError(msg)

        self.check_grav_vars_for_regional()

        ds = regional.regional_separation(
            grav_ds=self._ds,
            **kwargs,
        )

        # save columns to reset inversion later
        ds["starting_forward_gravity"] = ds.forward_gravity.copy()
        ds["starting_misfit"] = ds.misfit.copy()
        ds["starting_reg"] = ds.reg.copy()
        ds["starting_res"] = ds.res.copy()
        ds.attrs.update(self._ds.attrs)
        return ds

    def check_grav_vars_for_regional(self) -> None:
        """
        ensure the gravity dataset has the required variables for performing regional
        estimation.
        """
        if self._ds.dataset_type == "model":
            msg = (
                "The `check_grav_vars` method is only available for the data dataset. "
            )
            raise ValueError(msg)
        variables = [
            "easting",
            "northing",
            "upward",
            "gravity_anomaly",
            "forward_gravity",
        ]
        assert all(i in self._ds for i in variables), (
            f"`gravity dataset` needs all the following variables: {variables}"
        )

    def check_grav_vars(self) -> None:
        """
        ensure the gravity dataset has the required variables for performing the
        inversion.
        """
        if self._ds.dataset_type == "model":
            msg = (
                "The `check_grav_vars` method is only available for the data dataset. "
            )
            raise ValueError(msg)
        variables = [
            "easting",
            "northing",
            "upward",
            "gravity_anomaly",
            "forward_gravity",
            "misfit",
            "reg",
            "res",
        ]
        assert all(i in self._ds for i in variables), (
            f"`gravity dataset` needs all the following variables: {variables}"
        )

    def check_gravity_inside_topography_region(
        self,
        topography: xr.DataArray,
    ) -> None:
        """check that all gravity data is inside the region of the topography grid"""
        if self._ds.dataset_type == "model":
            msg = "The `check_gravity_inside_topography_region` method is only available for the data dataset. "
            raise ValueError(msg)
        topo_region = vd.get_region(
            (topography.easting.to_numpy(), topography.northing.to_numpy())
        )
        df = self.df
        inside = vd.inside((df.easting, df.northing), region=topo_region)
        if not inside.all():
            msg = (
                "Some gravity data are outside the region of the topography grid. "
                "This may result in unexpected behavior."
            )
            raise ValueError(msg)

    def check_for_nans(self) -> None:
        """
        ensure there are no NaN values in the gravity residual.
        """
        if self._ds.dataset_type == "model":
            msg = "The `check_for_nans` method is only available for the data dataset. "
            raise ValueError(msg)
        if self.df.res.isna().to_numpy().any():
            msg = "gravity dataframe contains NaN values in the 'res' column"
            raise ValueError(msg)

    def plot_observed(self) -> None:
        """plot observed gravity"""
        if self._ds.dataset_type == "model":
            msg = "The `plot_observed` method is only available for the data dataset. "
            raise ValueError(msg)
        fig = maps.plot_grd(
            self._ds.gravity_anomaly,
            title="Observed gravity",
            cbar_label="mGal",
            cmap="viridis",
            hemisphere="south",
            robust=True,
            hist=True,
            scalebar=True,
        )
        if self._ds.buffer_width is not None:
            fig.plot(
                x=[
                    self._ds.inner_region[0],
                    self._ds.inner_region[1],
                    self._ds.inner_region[1],
                    self._ds.inner_region[0],
                    self._ds.inner_region[0],
                ],
                y=[
                    self._ds.inner_region[2],
                    self._ds.inner_region[2],
                    self._ds.inner_region[3],
                    self._ds.inner_region[3],
                    self._ds.inner_region[2],
                ],
                pen="2p,black",
                label="Inner region",
            )
            fig.legend()
        fig.show()

    def plot_anomalies(self) -> None:
        """plot gravity anomalies"""
        if self._ds.dataset_type == "model":
            msg = "The `plot_anomalies` method is only available for the data dataset. "
            raise ValueError(msg)

        ds = self.inner

        grids = [
            ds.gravity_anomaly,
            ds.forward_gravity,
            ds.misfit,
            ds.reg,
            ds.res,
        ]
        titles = [
            "Input anomaly",
            "Forward gravity",
            "Misfit",
            "Regional misfit",
            "Residual misfit",
        ]
        cmaps = [
            "viridis",
            "viridis",
            "balance+h0",
            "balance+h0",
            "balance+h0",
        ]
        fig = maps.subplots(
            grids,
            dims=(1, 5),
            region=self._ds.inner_region,
            fig_title="Gravity anomalies",
            titles=titles,
            cbar_label="mGal",
            cmaps=cmaps,
            hemisphere="south",
            robust=True,
            hist=True,
        )
        fig.show()

    def update_gravity_and_residual(
        self,
        model: xr.Dataset,
    ) -> None:
        """
        calculate the forward gravity of the supplied prism layer, add the results to a
        new dataframe column, and update the residual misfit. The supplied gravity
        dataframe needs a 'reg' column, which describes the regional component and can
        be 0.
        """
        if self._ds.dataset_type == "model":
            msg = "The `update_gravity_and_residual` method is only available for the data dataset. "
            raise ValueError(msg)

        # update the forward gravity
        self.forward_gravity(model)

        # each iteration updates the topography of the layer to minimize the residual
        # portion of the misfit. We then want to recalculate the forward gravity of the
        # new layer, use the same original regional misfit, and re-calculate the
        # residual.
        # Gmisfit  = Gobs - Gforward
        # Gres = Gmisfit - Greg
        # Gres = Gobs - Gforward - Greg

        # update the residual misfit with the new forward gravity and the same regional
        self._ds["res"] = (
            self._ds.gravity_anomaly - self._ds.forward_gravity - self._ds.reg
        )

    ###
    ###
    # Methods for the prism model dataset
    ###
    ###

    def add_topography_correction(self, step: NDArray) -> xr.Dataset:
        """
        update the model dataset with the surface corrections. Ensure
        that the updated surface doesn't intersect the optional confining surfaces.
        """
        if self._ds.dataset_type == "data":
            msg = "The `add_topography_correction` method is only available for the model dataset. "
            raise ValueError(msg)
        # get dataframe of prism layer
        df = self.masked_df

        # add column of topography correction values
        df = pd.concat(
            [
                df.drop(columns=["topography_correction"], errors="ignore"),
                pd.DataFrame({"topography_correction": step}),
            ],
            axis=1,
        )

        # for negative densities, negate the correction
        df.loc[df.density < 0, "topography_correction"] *= -1

        # optionally constrain the surface correction with bounding surfaces
        # alter the surface correction values to ensure when added to the current iteration's
        # topography it doesn't intersect optional confining layers.
        if df.upper_confining_layer.notna().any():
            # get max upward change allowed for each prism
            # positive values indicate max allowed upward change
            # negative values indicate topography is already too far above upper bound
            df["max_change_above"] = df.upper_confining_layer - df.topography
            number_enforced = 0
            for i, j in enumerate(df.topography_correction):
                if j > df.max_change_above[i]:
                    number_enforced += 1
                    df.loc[i, "topography_correction"] = df.max_change_above[i]
            logger.info(
                "enforced upper confining surface at %s prisms", number_enforced
            )
        if df.lower_confining_layer.notna().any():
            # get max downward change allowed for each prism
            # negative values indicate max allowed downward change
            # positive values indicate topography is already too far below lower bound
            df["max_change_below"] = df.lower_confining_layer - df.topography
            number_enforced = 0
            for i, j in enumerate(df.topography_correction):
                if j < df.max_change_below[i]:
                    number_enforced += 1
                    df.loc[i, "topography_correction"] = df.max_change_below[i]

            logger.info(
                "enforced lower confining surface at %s prisms", number_enforced
            )

        # check that when constrained correction is added to topography it doesn't intersect
        # either bounding layer
        updated_topo = df.topography_correction + df.topography
        if np.any((df.upper_confining_layer - updated_topo) < -0.001):
            msg = (
                "Constraining didn't work and updated topography intersects upper "
                "constraining surface"
            )
            raise ValueError(msg)
        if np.any((updated_topo - df.lower_confining_layer) < -0.001):
            msg = (
                "Constraining didn't work and updated topography intersects lower "
                "constraining surface"
            )
            raise ValueError(msg)

        # df is only for masked prisms, so fill in 0 for unmasked prisms
        df_full = self.df.drop(columns=["topography_correction"], errors="ignore")
        df_full = df_full.merge(
            df[["northing", "easting", "topography_correction"]],
            how="left",
            on=["northing", "easting"],
        )
        df_full["topography_correction"] = df_full["topography_correction"].fillna(0)

        # add the correction values to the prism layer dataset
        ds = df_full.set_index(["northing", "easting"]).to_xarray()
        ds.attrs.update(self._ds.attrs)
        return ds

    def update_model_ds(self) -> xr.Dataset:
        """
        apply the corrections grid and update the model tops, bottoms, topo, and
        densities.
        """
        if self._ds.dataset_type == "data":
            msg = (
                "The `update_model_ds` method is only available for the model dataset. "
            )
            raise ValueError(msg)

        ds = self._ds
        # apply correction to topo
        ds["topography"] = ds.topography + ds.topography_correction

        # update the layer
        if self._ds.model_type == "prisms":
            ds.prism_layer.update_top_bottom(surface=ds.topography, reference=ds.zref)
        elif self._ds.model_type == "tesseroids":
            ds.tesseroid_layer.update_top_bottom(
                surface=ds.topography, reference=ds.zref
            )

        # update the density
        ds["density"] = xr.where(
            ds.top > ds.zref,
            ds.density_contrast,
            -ds.density_contrast,
        )

        ds.attrs.update(self._ds.attrs)
        return ds


def create_data(
    gravity: xr.Dataset,
    buffer_width: float | None = None,
    model_type: str = "prisms",
) -> xr.Dataset:
    """
    Convert a dataset of gravity data into the format needed for the inversion. This
    includes adding various attributes to the dataset, and defining an inner region
    which will be used for plotting and statistics to avoid edge effects.

    Parameters
    ----------
    gravity : xr.Dataset
        A dataset with coordinates "easting" and "northing" (if using prisms) or
        "longitude" and "latitude" (if using tesseroids), as well and variables "upward"
        defining the gravity observation height and "gravity_anomaly" with the observed
        gravity values.
    buffer_width : float | None, optional
        The width in meters of a buffer zone which will be used to zoom-in on the
        provided data creating an inner region. This inner region will be used for
        plotting and calculating statistics for processes such as cross validation, and
        l2-norms, this avoids skewing plots and values by edge effects, by default will
        use a width of the 10% of the shortest dimension length, to the nearest multiple
        of grid spacing.
    model_type : str, optional
        Choose between "prisms" and "tesseroids", which affects whether geographic or
        projected coordinates are expect, by default "prisms"

    Returns
    -------
    xr.Dataset
        A dataset with new attributes.
    """
    gravity = gravity.copy()

    if model_type == "prisms":
        coord_names = ("easting", "northing")
        keys = ["upward"]
    elif model_type == "tesseroids":
        coord_names = ("longitude", "latitude")
        keys = ["upward", "geocentric_radius"]
    assert all(s in gravity.dims for s in coord_names), (
        f"gravity dataset must have dims {coord_names}, you can rename your dimensions with `.rename({{'old_name':'new_name'}})`"
    )

    assert all(i in list(gravity.keys()) for i in keys), (
        f"gravity dataset needs variables {keys}."
    )

    if model_type == "tesseroids":
        gravity["geoidal_upward"] = gravity.upward
        gravity["upward"] = gravity.geoidal_upward + gravity.geocentric_radius

    # set region and spacing from provided grid
    spacing, region = polar_utils.get_grid_info(gravity.upward)[0:2]

    # default buffer with is 10% of the shortest dimension of the region, rounded to
    # nearest multiple of spacing
    if buffer_width is None:
        # check that buffer width is a multiple of the grid spacing
        min_dimension = min(region[1] - region[0], region[3] - region[2])
        buffer_width = min_dimension * 0.1
        buffer_width = round(buffer_width / spacing) * spacing

    assert buffer_width % spacing == 0, (
        f"buffer_width ({buffer_width}) must be a multiple of the grid spacing ({spacing}) of the provided gravity dataframe"
    )
    min_region_width = min(region[1] - region[0], region[3] - region[2])
    assert buffer_width * 2 < min_region_width, (
        "buffer_width must be smaller than half the smallest dimension of the region"
    )

    inner_region = vd.pad_region(region, -buffer_width)

    # Append some attributes to the xr.Dataset
    attrs = {
        "region": region,
        "spacing": spacing,
        "buffer_width": buffer_width,
        "inner_region": inner_region,
        "dataset_type": "data",
        "model_type": model_type,
    }
    gravity.attrs = attrs

    return gravity


def create_model(
    zref: float,
    density_contrast: xr.DataArray | float,
    starting_topography: xr.Dataset,
    model_type: str = "prisms",
    upper_confining_layer: xr.DataArray | None = None,
    lower_confining_layer: xr.DataArray | None = None,
) -> xr.Dataset:
    """
    Convert a topography grid into a model, which can be used as the starting model for
    an inversion. Choose between using prisms or tesseroids, and constraining the
    topography during the inversion with upper and lower confining layers.


    Parameters
    ----------
    zref : float
        The reference elevation (for prisms) or geocentric radius (for tesseroids)
        which separates positive from negative density values.
    density_contrast : xr.DataArray | float
        The density contrast to use for the prisms or tesseroids. This can be
        a constant value, or a DataArray with the same dimensions as the
        `starting_topography` DataArray.
    starting_topography : xr.Dataset
        The starting topography dataset, which must contain an "upward" variable
        defining the topography, and an optional "mask" variable. Mask values of 0 won't
        be altered during the inversion, while mask values of 1 are free to change.
        If you don't have a starting topography grid, you can create a flat grid with
        `verde.grid_coordinates` and `verde.make_xarray_grid`.
    model_type : str, optional
        The type of model to create, either "prisms" or "tesseroids", by default "prisms"
    upper_confining_layer : xr.DataArray | None, optional
        The upper confining layer for the model, by default None
    lower_confining_layer : xr.DataArray | None, optional
        The lower confining layer for the model, by default None

    Returns
    -------
    xarray.Dataset
        A dataset containing the prism / tesseroids, variables for topography and
        density, and various attributes.

    """

    starting_topography = starting_topography.copy()

    assert "upward" in starting_topography, (
        "starting_topography Dataset must contain an 'upward' variable"
    )
    if model_type == "prisms":
        coord_names = ("easting", "northing")
    elif model_type == "tesseroids":
        coord_names = ("longitude", "latitude")
        assert "geocentric_radius" in starting_topography, (
            "for tesseroid models, starting_topography Dataset must contain a 'geocentric_radius' variable, which can be created using the Python package Boule."
        )
        starting_topography["geoidal_upward"] = starting_topography.upward
        starting_topography["upward"] = (
            starting_topography.geoidal_upward + starting_topography.geocentric_radius
        )
    else:
        msg = "model_type must be either 'prisms' or 'tesseroids'"
        raise ValueError(msg)

    assert all(s in starting_topography.upward.dims for s in coord_names), (
        f"topography DataArray must have dims {coord_names}, you can rename your dimensions with `.rename({{'old_name':'new_name'}})`"
    )
    # set region and spacing from provided grid
    spacing, region = polar_utils.get_grid_info(starting_topography.upward)[0:2]

    if isinstance(density_contrast, xr.DataArray):
        assert all(s in density_contrast.dims for s in coord_names), (
            f"density DataArray must have dims {coord_names}, you can rename your dimensions with `.rename({{'old_name':'new_name'}})`"
        )
    elif isinstance(density_contrast, float | int):
        pass
    else:
        msg = "`density_contrast` must be a float or xarray.DataArray"
        raise ValueError(msg)

    if model_type == "tesseroids":
        zref_used = starting_topography.geocentric_radius + zref
    else:
        zref_used = zref

    # create grid of density values, positive above zref, negative below
    density_grid = xr.where(
        starting_topography.upward >= zref_used,
        density_contrast,
        -density_contrast,
    )

    # create prism layer from topography and density grid
    model = utils.grid_to_model(
        starting_topography.upward,
        reference=zref_used,
        density=density_grid,
        model_type=model_type,
    )

    # add starting topography and current topography to prism layer dataset
    model["starting_topography"] = starting_topography.upward
    model["topography"] = starting_topography.upward
    model["mask"] = (
        starting_topography.mask
        if "mask" in starting_topography
        else xr.ones_like(starting_topography.upward)
    )

    # add optional confining layers as variables
    if upper_confining_layer is not None:
        model["upper_confining_layer"] = upper_confining_layer
    else:
        model["upper_confining_layer"] = xr.full_like(
            model.topography,
            np.nan,
            dtype=np.double,
        )
    if lower_confining_layer is not None:
        model["lower_confining_layer"] = lower_confining_layer
    else:
        model["lower_confining_layer"] = xr.full_like(
            model.topography,
            np.nan,
            dtype=np.double,
        )

    # use extent of un-masked cells to define a region to use for plotting
    masked_extent = model.mask.where(model.mask == 1, drop=True)

    # Append some attributes to the xr.Dataset
    attrs = {
        "inner_region": polar_utils.get_grid_info(masked_extent)[1],
        "zref": zref,
        "density_contrast": density_contrast,
        "region": region,
        "spacing": spacing,
        "dataset_type": "model",
        "model_type": model_type,
    }
    model.attrs = attrs

    return model


class Inversion:
    """
    A class which holds the gravity inversion parameters and methods.
    """

    def __init__(
        self,
        data: xr.Dataset,
        model: xr.Dataset,
        max_iterations: int = 100,
        l2_norm_tolerance: float = 0.2,
        delta_l2_norm_tolerance: float = 1.001,
        perc_increase_limit: float = 0.20,
        deriv_type: str = "annulus",
        jacobian_finite_step_size: float = 1,
        model_properties_method: str = "itertools",
        solver_type: str = "scipy least squares",
        solver_damping: float | None = None,
        apply_weighting_grid: bool = False,
        weighting_grid: xr.DataArray | None = None,
    ) -> None:
        """
        Initialize the inversion object, setting the stopping criteria, parameters
        controlling the inversion, and add the data and model as attributes. Data and
        model are both deep copied, so changes to the original datasets will not be
        reflected by the datasets assigned as attributes to this instance.

        Parameters
        ----------
        data : xr.Dataset
            _description_
        model : xr.Dataset
            _description_
        max_iterations : int, optional
            _description_, by default 100
        l2_norm_tolerance : float, optional
            _description_, by default 0.2
        delta_l2_norm_tolerance : float, optional
            _description_, by default 1.001
        perc_increase_limit : float, optional
            _description_, by default 0.20
        deriv_type : str, optional
            _description_, by default "annulus"
        jacobian_finite_step_size : float, optional
            _description_, by default 1
        model_properties_method : str, optional
            _description_, by default "itertools"
        solver_type : str, optional
            _description_, by default "scipy least squares"
        solver_damping : float | None, optional
            _description_, by default None
        apply_weighting_grid : bool, optional
            _description_, by default False
        weighting_grid : xr.DataArray | None, optional
            _description_, by default None
        """
        self.data = copy.deepcopy(data)
        self.model = copy.deepcopy(model)
        self.max_iterations = max_iterations
        self.l2_norm_tolerance = l2_norm_tolerance
        self.delta_l2_norm_tolerance = delta_l2_norm_tolerance
        self.perc_increase_limit = perc_increase_limit
        self.deriv_type = deriv_type
        self.jacobian_finite_step_size = jacobian_finite_step_size
        self.model_properties_method = model_properties_method
        self.solver_type = solver_type
        self.solver_damping = solver_damping
        self.apply_weighting_grid = apply_weighting_grid
        self.weighting_grid = weighting_grid
        self.end = None
        self.jac = None
        self.step = None
        self.iteration = None
        self.stats_df = None
        self.time_start = None
        self.iter_time_start = None
        self.iter_time_end = None
        self.past_l2_norm = None
        self.elapsed_time = None
        self.termination_reason = None
        self.params = None
        self.results_fname = None
        self.gravity_cv_best_score = None
        self.constraints_cv_best_score = None
        self.best_trial = None
        self.study = None
        self.damping_cv_results_fname = None
        self.damping_cv_study_fname = None
        self.zref_density_cv_study_fname = None
        self.zref_density_cv_results_fname = None

        # check that gravity dataset has necessary dimensions
        self.data.inv.check_grav_vars()

        # check that gravity data is inside topography region
        self.data.inv.check_gravity_inside_topography_region(
            self.model.starting_topography
        )

        # check that gravity dataset has no NaNs in residual column
        self.data.inv.check_for_nans()

        # if there is a confining surface (above or below), which the inverted layer
        # shouldn't intersect, then sample those layers into the df
        # self.Model.inv.sample_bounding_surfaces()

    @property
    def already_inverted(self) -> bool:
        """
        check if an inversion has already be performed on this instance.
        """
        return self.iteration is not None

    @property
    def rmse(self) -> float:
        """
        return the root mean square error of the current residual misfit
        """
        return utils.rmse(self.data.inv.inner.res)

    @property
    def l2_norm(self) -> float:
        """
        return the l2 norm of the current residual misfit
        """
        x: float = np.sqrt(self.rmse)
        return x

    @property
    def delta_l2_norm(self) -> float:
        """
        return the current iterations delta l2 norm
        """
        x: float = self.past_l2_norm / self.l2_norm  # type: ignore[operator]
        return x

    def end_inversion(self) -> None:
        """
        check if the inversion should be terminated
        """
        end = False
        termination_reason = []

        l2_norms = self.stats_df.l2_norm.to_numpy()  # type: ignore[attr-defined]
        l2_norm = self.stats_df.l2_norm.iloc[self.iteration]  # type: ignore[attr-defined]
        delta_l2_norm = self.stats_df.delta_l2_norm.iloc[self.iteration]  # type: ignore[attr-defined]

        # ignore for first iteration
        if self.iteration == 1:
            pass
        else:
            if l2_norm > np.min(l2_norms) * (1 + self.perc_increase_limit):
                logger.info(
                    "\nInversion terminated after %s iterations because L2 norm (%s) \n"
                    "was over %s times greater than minimum L2 norm (%s) \n"
                    "Change parameter 'perc_increase_limit' if desired.",
                    self.iteration,
                    l2_norm,
                    1 + self.perc_increase_limit,
                    np.min(l2_norms),
                )
                end = True
                termination_reason.append("l2-norm increasing")

            if (delta_l2_norm <= self.delta_l2_norm_tolerance) & (
                self.stats_df.delta_l2_norm.iloc[self.iteration - 1]  # type: ignore[operator, attr-defined]
                <= self.delta_l2_norm_tolerance
            ):
                logger.info(
                    "\nInversion terminated after %s iterations because there was no "
                    "significant variation in the L2-norm over 2 iterations \n"
                    "Change parameter 'delta_l2_norm_tolerance' if desired.",
                    self.iteration,
                )

                end = True
                termination_reason.append("delta l2-norm tolerance")

            if l2_norm < self.l2_norm_tolerance:
                logger.info(
                    "\nInversion terminated after %s iterations because L2-norm (%s) was "
                    "less then set tolerance: %s \nChange parameter "
                    "'l2_norm_tolerance' if desired.",
                    self.iteration,
                    l2_norm,
                    self.l2_norm_tolerance,
                )

                end = True
                termination_reason.append("l2-norm tolerance")

        if self.iteration >= self.max_iterations:  # type: ignore[operator]
            logger.info(
                "\nInversion terminated after %s iterations with L2-norm=%s because "
                "maximum number of iterations (%s) reached.",
                self.iteration,
                round(l2_norm, 2),
                self.max_iterations,
            )

            end = True
            termination_reason.append("max iterations")

        if "max iterations" in termination_reason:
            msg = (
                "Inversion terminated due to max_iterations limit. Consider increasing "
                "this limit."
            )
            logger.warning(msg)

        self.end = end  # type: ignore[assignment]
        self.termination_reason = termination_reason  # type: ignore[assignment]

    def jacobian(self) -> None:
        """
        dispatcher for creating the jacobian matrix with 2 method options
        """

        # convert gravity dataframe to numpy arrays
        coordinates = self.data.inv.df.select_dtypes(include=["number"])
        coordinates_array = coordinates.to_numpy()

        # get various arrays based on gravity column names
        grav_easting = coordinates_array[:, coordinates.columns.get_loc("easting")]
        grav_northing = coordinates_array[:, coordinates.columns.get_loc("northing")]
        grav_upward = coordinates_array[:, coordinates.columns.get_loc("upward")]

        assert len(grav_easting) == len(grav_northing) == len(grav_upward)

        # create empty jacobian to fill in
        # first discard prisms based on mask
        self.jac = np.empty(
            (len(grav_easting), self.model.inv.masked.top.size),
            dtype=np.float64,
        )
        if self.deriv_type == "annulus":
            # convert dataframe to arrays
            df = self.model.inv.masked_df
            model_element_easting = df.easting.to_numpy()
            model_element_northing = df.northing.to_numpy()
            model_element_top = df.top.to_numpy()
            model_element_density = df.density.to_numpy()

            if np.all((model_element_top - grav_upward.mean()) == 0):
                msg = (
                    "All prism tops coincides exactly with the elevation of the gravity "
                    "observation points, leading to issues with calculating the vertical "
                    "derivative of gravity with the annulus technique. Either slightly "
                    "change the prism tops or gravity elevations, or use the small-prisms "
                    "vertical derivative technique."
                )
                raise ValueError(msg)

            self.jac = jacobian_annular(
                grav_easting,
                grav_northing,
                grav_upward,
                model_element_easting,
                model_element_northing,
                model_element_top,
                model_element_density,
                self.model.spacing,
                self.jac,
            )

        elif self.deriv_type == "finite_difference":
            # get prisms info in following format, 3 methods:
            # ((west, east, south, north, bottom, top), density)
            model_properties = _model_properties(
                self.model.inv.masked,
                method=self.model_properties_method,
            )

            if self.model.model_type == "prisms":
                self.jac = jacobian_finite_difference_prisms(
                    model_properties,
                    grav_easting,
                    grav_northing,
                    grav_upward,
                    self.jacobian_finite_step_size,
                    self.jac,
                )
            elif self.model.model_type == "tesseroids":
                self.jac = jacobian_finite_difference_tesseroids(
                    model_properties,
                    grav_easting,
                    grav_northing,
                    grav_upward,
                    self.jacobian_finite_step_size,
                    self.jac,
                )

        else:
            msg = "invalid string for deriv_type"
            raise ValueError(msg)

    def solver(self) -> None:
        """
        Calculate shift to add to prism's for each iteration of the inversion. Finds
        the least-squares solution to the Jacobian and the gravity residual
        """
        if self.solver_type == "scipy least squares":
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
            damping = 0 if self.solver_damping is None else self.solver_damping
            step = sp.sparse.linalg.lsqr(
                A=self.jac,
                b=self.data.inv.df.res.to_numpy(),
                show=False,
                damp=damping,  # float, typically 0-1
                # atol= ,
                # btol=1e-4, # if 1e-6, residuals should be accurate to ~6 digits
                iter_lim=5000,  # limit of iterations, just in case of issues
            )[0]
        else:
            msg = "invalid string for solver_type"
            raise ValueError(msg)

        self.step = step

    def reinitialize_inversion(self) -> None:
        """
        reset inversion object attributes, gravity data misfit, and model topography to
        what it was before the inversion.
        """
        self.iteration = None

        self.data["misfit"] = self.data["starting_misfit"]
        self.data["res"] = self.data["starting_res"]
        self.data["reg"] = self.data["starting_reg"]
        self.data["forward_gravity"] = self.data["starting_forward_gravity"]

        self.data = self.data.drop_vars(
            [v for v in list(self.data.keys()) if v.startswith("iter_")],
            errors="ignore",
        )

        model = create_model(
            zref=self.model.zref,
            density_contrast=self.model.density_contrast,
            model_type=self.model.model_type,
            starting_topography=self.model.starting_topography.to_dataset(
                name="upward"
            ),
        )
        self.model = model
        # self.model = self.model.drop_vars(
        #     ["topography_correction"]
        #     + [v for v in list(self.model.keys()) if v.startswith("iter_")],
        #     errors="ignore",
        # )
        # self.model["topography"] = self.model.starting_topography

    def invert(
        self,
        results_fname: str | None = None,
        progressbar: bool = True,
        plot_convergence: bool = False,
        plot_dynamic_convergence: bool = False,
    ) -> None:
        """
        Start the inversion, saving the results and parameters as attributes and new
        variables in the data and model datasets attributes.

        Parameters
        ----------
        results_fname : str | None, optional
            file name to save results to as pickle file, by default None
        progressbar : bool, optional
            whether to show an iteration progress bar, by default True
        plot_convergence : bool, optional
            whether to plot convergence, by default False
        plot_dynamic_convergence : bool, optional
            whether to plot convergence iteration-by-iteration, by default False
        """
        if self.already_inverted is True:
            msg = "this inversion object has already been used to run an inversion, re-initialize the object or run `reinitialize_inversion` to run another inversion"
            raise ValueError(msg)

        # warn if weighting grid supplied by unused
        if (self.weighting_grid is not None) & (self.apply_weighting_grid is False):
            msg = (
                "weighting grid supplied but not used because apply_weighting_grid is "
                "False"
            )
            raise ValueError(msg)
        if (self.apply_weighting_grid is True) & (self.weighting_grid is None):
            msg = "must supply weighting grid if apply_weighting_grid is True"
            raise ValueError(msg)

        self.iteration = 0  # type: ignore[assignment]

        # create empty dataframe to hold iteration stats
        self.stats_df = pd.DataFrame(
            columns=[
                "iteration",
                "rmse",
                "l2_norm",
                "delta_l2_norm",
                "iter_time_sec",
            ]
        )
        # add row for iteration 0 (initial model)
        self.stats_df.loc[self.iteration] = [  # type: ignore[attr-defined]
            self.iteration,
            self.rmse,
            self.l2_norm,
            np.inf,
            np.nan,
        ]

        # start timing of overall inversion
        self.time_start = time.perf_counter()  # type: ignore[assignment]

        if progressbar is True:
            pbar = tqdm(range(self.max_iterations), initial=1, desc="Iteration")
        elif progressbar is False:
            pbar = range(self.max_iterations)
        else:
            msg = "progressbar must be a boolean"  # type: ignore[unreachable]
            raise ValueError(msg)

        for iteration, _ in enumerate(pbar, start=1):
            logger.info(
                "\n #################################### \n iteration %s", iteration
            )
            # start iteration timer
            self.iter_time_start = time.perf_counter()  # type: ignore[assignment]
            self.iteration = iteration  # type: ignore[assignment]

            # calculate jacobian sensitivity matrix
            self.jacobian()

            # calculate array of topographic correction for each prism
            self.solver()

            # log correction values
            logger.info(
                "Topography correction (before weighting) median: %s m, RMS:%s m",
                round(np.median(self.step), 6),
                round(utils.rmse(self.step), 6),
            )

            # add correction array to prism layer dataset, optionally enforcing
            # confining layers
            self.model = self.model.inv.add_topography_correction(self.step)

            # optionally apply weights to the topo correction grid
            if self.apply_weighting_grid is True:
                self.model["topography_correction"] = (
                    self.model.topography_correction * self.weighting_grid
                )

            # add the corrections to the topo and update the prisms dataset
            self.model = self.model.inv.update_model_ds()

            # save results with iteration number
            self.model[f"iter_{self.iteration}_top"] = self.model.top
            self.model[f"iter_{self.iteration}_bottom"] = self.model.bottom
            self.model[f"iter_{self.iteration}_density"] = self.model.density
            self.model[f"iter_{self.iteration}_layer"] = self.model.topography
            self.model[f"iter_{self.iteration}_correction"] = (
                self.model.topography_correction
            )
            # save current residual with iteration number
            self.data[f"iter_{self.iteration}_initial_residual"] = self.data.res

            # update the forward gravity, residual, and the l2 / delta l2 norms
            self.data.inv.update_gravity_and_residual(self.model)

            # end iteration timer
            self.iter_time_end = time.perf_counter()  # type: ignore[assignment]

            self.past_l2_norm = self.stats_df.loc[self.iteration - 1, "l2_norm"]  # type: ignore[attr-defined, operator]

            # add row for current iteration
            self.stats_df.loc[self.iteration] = [  # type: ignore[attr-defined]
                self.iteration,
                self.rmse,
                self.l2_norm,
                self.delta_l2_norm,
                self.iter_time_end - self.iter_time_start,  # type: ignore[operator]
            ]

            # decide if to end the inversion
            self.end_inversion()

            if plot_dynamic_convergence is True:
                self.plot_dynamic_convergence()

            if self.end is True:
                # self.already_inverted = True
                if progressbar is True:  # type: ignore[unreachable]
                    pbar.set_description(
                        f"Inversion ended due to {self.termination_reason}"
                    )
                break
            # end of inversion loop

        self.elapsed_time = time.perf_counter() - self.time_start  # type: ignore[assignment, operator]

        if plot_convergence is True:
            try:
                self.plot_convergence()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("plotting failed with error: %s", e)

        self.params = {  # type: ignore[assignment]
            # first column
            "Density contrast(s)": f"{np.unique(np.abs(self.model.density_contrast))} kg/m3",
            "Reference level": f"{self.model.zref} m",
            "Max iterations": self.max_iterations,
            "L2 norm tolerance": f"{self.l2_norm_tolerance}",
            "Delta L2 norm tolerance": f"{self.delta_l2_norm_tolerance}",
            # second column
            "Deriv type": self.deriv_type,
            "Solver type": self.solver_type,
            "Solver damping": self.solver_damping,
            "Upper confining layer": "Not enabled"
            if self.model.upper_confining_layer is None
            else "Enabled",
            "Lower confining layer": "Not enabled"
            if self.model.lower_confining_layer is None
            else "Enabled",
            "Regularization weighting grid": "Not enabled"
            if self.apply_weighting_grid is False
            else "Enabled",
            # third column
            "Time elapsed": f"{int(self.elapsed_time)} seconds",  # type: ignore[call-overload]
            "Avg. iteration time": f"{round(np.mean(self.stats_df.iter_time_sec[1:].to_numpy()), 2)} seconds",  # type: ignore[attr-defined]
            "Final misfit RMSE / L2-norm": (
                f"{round(self.rmse, 4)} /{round(self.l2_norm, 4)} mGal"
            ),
            "Termination reason": self.termination_reason,
            "Iteration times": self.stats_df.iter_time_sec[1:].to_numpy(),  # type: ignore[attr-defined]
        }

        if results_fname is not None:
            self.results_fname = results_fname  # type: ignore[assignment]
            # remove if exists
            pathlib.Path(f"{results_fname}.pickle").unlink(missing_ok=True)
            with pathlib.Path(f"{results_fname}.pickle").open("wb") as f:
                pickle.dump(self, f)
            logger.info("results saved to %s.pickle", results_fname)

    def grav_cv_score(
        self,
        results_fname: str | None = None,
        rmse_as_median: bool = False,
        plot: bool = False,
    ) -> "Inversion":
        """
        Find the score, represented by the root mean (or median) squared error (RMSE),
        between the testing gravity data, and the predict gravity data after an
        inversion. Follows methods of :footcite:t:`uiedafast2017`. Used in
        `optimization.optimize_inversion_damping()`.

        Parameters
        ----------
        rmse_as_median : bool, optional
            calculate the RMSE as the median as opposed to the mean, by default False
        plot : bool, optional
            choose to plot the observed and predicted data grids, and their difference,
            located at the testing points, by
            default False

        Returns
        -------
        inv_copy : Inversion
            a copy of the Inversion object after running the inversion on the training

        References
        ----------
        :footcite:t:`uiedafast2017`
        """
        # make copies of Inversion and underlying data and model so as not
        # to alter the original
        inv_copy = copy.deepcopy(self)
        df = inv_copy.data.inv.df

        inv_copy.results_fname = results_fname  # type: ignore[assignment]

        test = df[df.test == True].copy()  # noqa: E712 # pylint: disable=singleton-comparison
        # test = inv_copy.data.where(inv_copy.data.test == True).copy()

        # temporarily set the gravity dataframe to only the training data
        train = df[df.test == False].copy()  # noqa: E712 # pylint: disable=singleton-comparison
        inv_copy.data = train.set_index(["northing", "easting"]).to_xarray()

        # inv_copy.data = inv_copy.data.where(inv_copy.data.test == False).copy()

        # retrain attributes
        inv_copy.data.attrs.update(self.data.attrs)  # pylint: disable=protected-access

        with utils._log_level(logging.WARN):  # pylint: disable=protected-access
            # run inversion
            inv_copy.invert(
                results_fname=inv_copy.results_fname,
                progressbar=False,
                plot_convergence=False,
                plot_dynamic_convergence=False,
            )

        density_grid = xr.where(
            inv_copy.model.topography >= inv_copy.model.zref,
            inv_copy.model.density_contrast,
            -inv_copy.model.density_contrast,
        )

        # create new layer or prisms / tesseriods
        layer = utils.grid_to_model(
            inv_copy.model.topography,
            reference=inv_copy.model.zref,
            density=density_grid,
            model_type=inv_copy.model.model_type,
        )

        # calculate forward gravity of inverted prism layer
        if layer.model_type == "prisms":
            test["test_point_grav"] = layer.prism_layer.gravity(
                coordinates=(
                    test.easting,
                    test.northing,
                    test.upward,
                ),
                field="g_z",
                progressbar=False,
            )
        elif layer.model_type == "tesseroids":
            test["test_point_grav"] = layer.tesseroid_layer.gravity(
                coordinates=(
                    test.easting,
                    test.northing,
                    test.upward,
                ),
                field="g_z",
                progressbar=False,
            )
        else:
            msg = "layer must have attribute 'model_type' which is either 'prisms' or 'tesseroids'"
            raise ValueError(msg)

        # subset to only points inside the inner region so edge effects have less effect
        # on scores
        test = polar_utils.points_inside_region(
            test,
            self.data.inner_region,
            names=("easting", "northing"),
        )

        # compare forward of inverted layer with observed
        observed = test.gravity_anomaly - test.reg
        predicted = test.test_point_grav

        self.gravity_cv_best_score = utils.rmse(  # type: ignore[assignment]
            predicted - observed, as_median=rmse_as_median
        )

        if plot:
            try:
                test_grid = test.set_index(["northing", "easting"]).to_xarray()
                obs = test_grid.gravity_anomaly - test_grid.reg
                pred = test_grid.test_point_grav.rename("")

                _ = polar_utils.grd_compare(
                    pred,
                    obs,
                    grid1_name="Predicted gravity",
                    grid2_name="Observed gravity",
                    plot_type="xarray",
                    robust=True,
                    title=f"Score={self.gravity_cv_best_score}",
                    rmse_in_title=False,
                    hist=True,
                    inset=False,
                    hemisphere="south",
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("plotting failed with error: %s", e)

        return inv_copy

    def constraints_cv_score(
        self,
        constraints_df: pd.DataFrame,
        results_fname: str | None = None,
        rmse_as_median: bool = False,
    ) -> "Inversion":
        """
        Find the score, represented by the root mean squared error (RMSE), between the
        constraint point elevation, and the inverted topography at the constraint points.
        Follows methods of :footcite:t:`uiedafast2017`. Used in
        `optimization.optimize_inversion_zref_density_contrast()`.

        Parameters
        ----------

        constraints_df : pd.DataFrame
            a dataframe with columns "easting", "northing", and "upward" for
            coordinates and elevation of the constraint points.
        results_fname : str | None, optional
            file name to save results to as pickle file, by default fname is None
        rmse_as_median : bool, optional
            calculate the RMSE as the median of the , as opposed to the mean, by default
            False

        Returns
        -------
        inv_copy : Inversion
            a copy of the Inversion object after running the inversion on the training
            data and sampling the inverted topography at the constraint points
        References
        ----------
        .. footbibliography::
        """
        # make copies of Inversion and underlying data and model dataset so as not
        # to alter the original
        inv_copy = copy.deepcopy(self)

        inv_copy.results_fname = results_fname  # type: ignore[assignment]

        with utils._log_level(logging.WARN):  # pylint: disable=protected-access
            # run inversion
            inv_copy.invert(
                results_fname=inv_copy.results_fname,
                progressbar=False,
                plot_convergence=False,
                plot_dynamic_convergence=False,
            )

        # sample the inverted topography at the constraint points
        constraints_df = utils.sample_grids(
            constraints_df,
            inv_copy.model.topography,
            "inverted_topo",
        )

        # calculate the difference between the inverted topography and the constraint
        # elevations
        dif = constraints_df.upward - constraints_df.inverted_topo

        self.constraints_cv_best_score: float = utils.rmse(  # type: ignore[no-redef, assignment]
            dif, as_median=rmse_as_median
        )

        return inv_copy

    def optimize_inversion_damping(
        self,
        n_trials: int,
        damping_limits: tuple[float, float],
        n_startup_trials: int | None = None,
        score_as_median: bool = False,
        sampler: optuna.samplers.BaseSampler | None = None,
        grid_search: bool = False,
        fname: str | None = None,
        plot_cv: bool = True,
        plot_grids: bool = False,
        logx: bool = True,
        logy: bool = True,
        progressbar: bool = True,
        parallel: bool = False,
        seed: int = 0,
    ) -> "Inversion":
        """
        Use Optuna to find the optimal damping regularization parameter for a gravity
        inversion. The optimization aims to minimize the cross-validation score,
        represented by the root mean (or median) squared error (RMSE), between the testing
        gravity data, and the predict gravity data after and inversion. Follows methods of
        :footcite:t:`uiedafast2017`.

        Provide upper and low damping values, number of trials to run, and specify to let
        Optuna choose the best damping value for each trial or to use a grid search. The
        results are saved to a pickle file with the best inversion results and the study.

        Parameters
        ----------
        n_trials : int
            number of damping values to try
        n_startup_trials : int | None, optional
            number of startup trials, by default is automatically determined
        damping_limits : tuple[float, float]
            upper and lower limits
        score_as_median : bool, optional
            if True, changes the scoring from the root mean square to the root median
            square, by default False
        sampler : optuna.samplers.BaseSampler | None, optional
            customize the optuna sampler, by default either GPsampler or GridSampler
            depending on if grid_search is True or False
        grid_search : bool, optional
            search the entire parameter space between damping_limits in n_trial steps, by
            default False
        fname : str, optional
            file name to save both study and inversion results to as pickle files, by
            default fname is `tmp_x_damping_cv` where x is a random integer between 0 and
            999 and will save study to <fname>_study.pickle and tuple of inversion results
            to <fname>.pickle.
        plot_cv : bool, optional
            plot the cross-validation results, by default True
        plot_grids : bool, optional
            for each damping value, plot comparison of predicted and testing gravity data,
            by default False
        logx : bool, optional
            make x axis of CV result plot on log scale, by default True
        logy : bool, optional
            make y axis of CV result plot on log scale, by default True
        progressbar : bool, optional
            add a progressbar, by default True
        parallel : bool, optional
            run the optimization in parallel, by default False
        seed : int, optional
            random seed for the samplers, by default 0

        Returns
        -------
        inv_copy : Inversion
            a copy of the Inversion object after running the inversion with the best
            damping value
        """

        # make copies of Inversion and underlying data and model dataset so as not
        # to alter the original
        inv_copy = copy.deepcopy(self)

        optuna.logging.set_verbosity(optuna.logging.WARN)

        # set file name for saving results with random number between 0 and 999
        if fname is None:
            inv_copy.results_fname = f"tmp_{random.randint(0, 999)}_damping_cv"  # type: ignore[assignment]
        else:
            inv_copy.results_fname = fname  # type: ignore[assignment]

        if parallel:
            pathlib.Path(f"{inv_copy.results_fname}.log").unlink(missing_ok=True)
            pathlib.Path(f"{inv_copy.results_fname}.lock").unlink(missing_ok=True)
            pathlib.Path(f"{inv_copy.results_fname}.log.lock").unlink(missing_ok=True)
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(
                    f"{inv_copy.results_fname}.log"
                ),
            )
        else:
            storage = None

        # define the objective function
        objective = optimization.OptimalInversionDamping(
            inversion_obj=inv_copy,
            damping_limits=damping_limits,
            rmse_as_median=score_as_median,
            fname=inv_copy.results_fname,  # type: ignore[arg-type]
            plot_grids=plot_grids,
        )
        if grid_search:
            if n_trials < 4:
                msg = (
                    "if grid_search is True, n_trials must be at least 4, "
                    "resetting n_trials to 4 now."
                )
                logger.warning(msg)
                n_trials = 4
            space = np.logspace(
                np.log10(damping_limits[0]), np.log10(damping_limits[1]), n_trials
            )
            sampler = optuna.samplers.GridSampler(
                search_space={"damping": space},
                seed=seed,
            )

            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                load_if_exists=False,
                study_name=inv_copy.results_fname,
                storage=storage,
                pruner=optimization.DuplicateIterationPruner,
            )

            # run optimization
            with utils.DuplicateFilter(logger):  # type: ignore[no-untyped-call]
                study = optimization.run_optuna(
                    study=study,
                    storage=storage,
                    objective=objective,
                    n_trials=n_trials,
                    # callbacks=[_warn_limits_better_than_trial_1_param],
                    maximize_cpus=True,
                    parallel=parallel,
                    progressbar=progressbar,
                )
        else:
            # define number of startup trials, whichever is bigger; 1/4 of trials or 4
            if n_startup_trials is None:
                n_startup_trials = max(4, int(n_trials / 4))
                n_startup_trials = min(n_startup_trials, n_trials)
            logger.info("using %s startup trials", n_startup_trials)
            if n_startup_trials >= n_trials:
                logger.warning(
                    "n_startup_trials is >= n_trials resulting in all trials sampled from "
                    "a QMC sampler instead of the GP sampler",
                )
            # create study
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.QMCSampler(
                    seed=seed,
                    qmc_type="halton",
                    scramble=True,
                ),
                load_if_exists=False,
                study_name=inv_copy.results_fname,
                storage=storage,
                pruner=optimization.DuplicateIterationPruner,
            )

            # explicitly add the limits as trials
            study.enqueue_trial({"damping": damping_limits[0]}, skip_if_exists=True)
            study.enqueue_trial({"damping": damping_limits[1]}, skip_if_exists=True)

            with utils.DuplicateFilter(logger):  # type: ignore[no-untyped-call]
                study = optimization.run_optuna(
                    study=study,
                    storage=storage,
                    objective=objective,
                    n_trials=n_startup_trials,
                    # callbacks=[_warn_limits_better_than_trial_1_param],
                    maximize_cpus=True,
                    parallel=parallel,
                    progressbar=progressbar,
                )
            # continue with remaining trials with user-defined sampler
            # if sampler not provided, used GPsampler as default
            if sampler is None:
                sampler = optuna.samplers.GPSampler(
                    n_startup_trials=0,
                    seed=seed,
                    deterministic_objective=True,
                )
            study.sampler = sampler
            with utils.DuplicateFilter(logger):  # type: ignore[no-untyped-call]
                study = optimization.run_optuna(
                    study=study,
                    storage=storage,
                    objective=objective,
                    n_trials=n_trials - n_startup_trials,
                    # callbacks=[_warn_limits_better_than_trial_1_param],
                    maximize_cpus=True,
                    parallel=parallel,
                    progressbar=progressbar,
                )
        inv_copy.best_trial = study.best_trial

        # warn if any best parameter values are at their limits
        optimization.warn_parameter_at_limits(inv_copy.best_trial)

        # log the results of the best trial
        optimization.log_optuna_results(inv_copy.best_trial)

        # get best inversion result of each set
        with pathlib.Path(
            f"{inv_copy.results_fname}_trial_{inv_copy.best_trial.number}.pickle"  # type: ignore[attr-defined]
        ).open("rb") as f:
            inv_results = pickle.load(f)

        # remove if exists
        pathlib.Path(f"{inv_copy.results_fname}_study.pickle").unlink(missing_ok=True)
        pathlib.Path(f"{inv_copy.results_fname}.pickle").unlink(missing_ok=True)

        # save study to pickle
        with pathlib.Path(f"{inv_copy.results_fname}_study.pickle").open("wb") as f:
            pickle.dump(study, f)

        # save inversion results tuple to pickle
        with pathlib.Path(f"{inv_copy.results_fname}.pickle").open("wb") as f:
            pickle.dump(inv_results, f)

        # delete all inversion results
        for i in range(n_trials):
            pathlib.Path(f"{inv_copy.results_fname}_trial_{i}.pickle").unlink(
                missing_ok=True
            )

        inv_copy.damping_cv_study_fname = f"{inv_copy.results_fname}_study.pickle"  # type: ignore[assignment]
        inv_copy.damping_cv_results_fname = f"{inv_copy.results_fname}.pickle"  # type: ignore[assignment]
        inv_copy.solver_damping = inv_copy.best_trial.params["damping"]  # type: ignore[attr-defined]
        inv_copy.study = study

        # update the inversion object with the best inversion results
        self.__dict__.update(inv_results.__dict__)

        if plot_cv is True:
            try:
                plotting.plot_cv_scores(
                    study.trials_dataframe().value.to_numpy(),
                    study.trials_dataframe().params_damping.to_numpy(),
                    param_name="Damping",
                    logx=logx,
                    logy=logy,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("plotting failed with error: %s", e)

        return inv_copy

    def optimize_inversion_zref_density_contrast(
        self,
        n_trials: int,
        constraints_df: pd.DataFrame,
        n_startup_trials: int | None = None,
        zref_limits: tuple[float, float] | None = None,
        density_contrast_limits: tuple[float, float] | None = None,
        starting_topography: xr.Dataset | None = None,
        starting_topography_kwargs: dict[str, typing.Any] | None = None,
        regional_grav_kwargs: dict[str, typing.Any] | None = None,
        score_as_median: bool = False,
        sampler: optuna.samplers.BaseSampler | None = None,
        grid_search: bool = False,
        fname: str | None = None,
        plot_cv: bool = True,
        logx: bool = False,
        logy: bool = False,
        progressbar: bool = True,
        parallel: bool = False,
        fold_progressbar: bool = True,
        seed: int = 0,
    ) -> "Inversion":
        """
        Run an Optuna optimization to find the optimal zref and or density contrast values
        for a gravity inversion. The optimization aims to minimize the cross-validation
        score, represented by the root mean (or median) squared error (RMSE), between
        points of known topography and the inverted topography. Follows methods of
        :footcite:t:`uiedafast2017`. This can optimize for either zref, density contrast,
        or both at the same time. Provide upper and low limits for each parameter, number of
        trials and let Optuna choose the best parameter values for each trial or use a grid
        search to test all values between the limits in intervals of n_trials. The results
        are saved to a pickle file with the best inversion results and the study. Since each
        new set of zref and density values changes the starting model, for each set of
        parameters this function re-calculates the starting gravity, the gravity misfit
        and its regional and residual components. `regional_grav_kwargs` are passed to
        `regional.regional_separation`. Once the optimal parameters are found, the regional
        separation and inversion are performed again and saved to <fname>.pickle and
        the study is saved to <fname>_study.pickle.
        The constraint point minimization regional separation technique uses constraints
        points to estimate the regional field, and since constraints are used to calculating
        the scoring metric of this function, the constraints need to be separated into
        training (regional estimation) and testing (scoring) sets. To do this, supply the
        training constraints to`regional_grav_kwargs` via `method="constraint"` or
        `method="constraint_cv"` and `constraints_df`, and the testing constraints to this
        function as `constraints_df`.
        Typically there are not many constraints and omitting some of them from the training
        set will significantly impact the regional estimation. To help with this, we can use
        a K-Folds approach, where for each set of parameter values, we perform this entire
        procedure K times, each time with a different separation of training and testing
        points, called a fold. The score associated with that parameter set is the mean of
        the K scores. Once the optimal parameter values are found, we then repeat the
        inversion using all of the constraints in the regional estimation. For a K-folds
        approach, supply lists of dataframes containing only each fold's testing or training
        points to the two `constraints_df` arguments. To automatically perform the
        test/train split and K-folds optimization, you can also use the convenience function
        `optimize_inversion_zref_density_contrast_kfolds`.

        Parameters
        ----------
        n_trials : int
            number of trials, if grid_search is True, needs to be a perfect square and >=16.
        n_startup_trials : int | None, optional
            number of startup trials, by default is automatically determined
        zref_limits : tuple[float, float] | None, optional
            upper and lower limits for the reference level, in meters, by default None
        density_contrast_limits : tuple[float, float] | None, optional
            upper and lower limits for the density contrast, in kg/m^-3, by default None
        starting_topography_kwargs : dict[str, typing.Any] | None, optional
            dictionary with key: value pairs of "region":tuple[float, float, float, float].
            "spacing":float, and "dampings":float | list[float] | None, used to create
            a flat starting topography at each zref value if starting_topography not
            provided, by default None
        regional_grav_kwargs : dict[str, typing.Any] | None, optional
            dictionary with kwargs to supply to `regional.regional_separation()`, by default
            None
        score_as_median : bool, optional
            change scoring metric from root mean square to root median square, by default
            False
        sampler : optuna.samplers.BaseSampler | None, optional
            customize the optuna sampler, by default uses GPsampler unless grid_search
            is True, then uses GridSampler.
        grid_search : bool, optional
            Switch the sampler to GridSampler and search entire parameter space between
            provided limits in intervals set by n_trials (for 1 parameter optimizations), or
            by the square root of n_trials (for 2 parameter optimizations), by default False
        fname : str | None, optional
            file name to save both study and inversion results to as pickle files, by
            default fname is `tmp_x_zref_density_cv` where x is a random integer between 0
            and 999 and will save study to <fname>_study.pickle and the inversion
            object to <fname>.pickle.
        plot_cv : bool, optional
            plot the cross-validation results, by default True
        logx : bool, optional
            use a log scale for the cross-validation plot x-axis, by default False
        logy : bool, optional
            use a log scale for the cross-validation plot y-axis, by default False
        progressbar : bool, optional
            add a progressbar, by default True
        parallel : bool, optional
            run the optimization in parallel, by default False
        fold_progressbar : bool, optional
            show a progress bar for each fold of the constraint-point minimization
            cross-validation, by default True
        seed : int, optional
            random seed for the samplers, by default 0

        Returns
        -------
        Inversion
            Inversion object with best inversion results and the optimally determined
            zref and or density contrast values as attributes of the `model` attribute.
        """

        # make copies of Inversion and underlying data and model dataset so as not
        # to alter the original
        inv_copy = copy.deepcopy(self)

        if regional_grav_kwargs is not None:
            regional_grav_kwargs = copy.deepcopy(regional_grav_kwargs)
        if starting_topography_kwargs is not None:
            starting_topography_kwargs = copy.deepcopy(starting_topography_kwargs)

        optuna.logging.set_verbosity(optuna.logging.WARN)

        # set file name for saving results with random number between 0 and 999
        if fname is None:
            inv_copy.results_fname = f"tmp_{random.randint(0, 999)}_zref_density_cv"  # type: ignore[assignment]
        else:
            inv_copy.results_fname = fname  # type: ignore[assignment]

        if "test" in inv_copy.data.inv.df.columns:
            assert inv_copy.data.inv.df.test.any(), (
                "test column contains True value, not needed except for during damping CV"
            )

        if parallel:
            pathlib.Path(f"{inv_copy.results_fname}.log").unlink(missing_ok=True)
            pathlib.Path(f"{inv_copy.results_fname}.lock").unlink(missing_ok=True)
            pathlib.Path(f"{inv_copy.results_fname}.log.lock").unlink(missing_ok=True)
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(
                    f"{inv_copy.results_fname}.log"
                ),
            )
        else:
            storage = None
        # get number of parameters included in optimization
        num_params = sum(x is not None for x in [zref_limits, density_contrast_limits])

        # define the objective function
        objective = optimization.OptimalInversionZrefDensity(
            inversion_obj=inv_copy,
            constraints_df=constraints_df,
            fname=inv_copy.results_fname,  # type: ignore[arg-type]
            regional_grav_kwargs=regional_grav_kwargs,  # type: ignore[arg-type]
            starting_topography=starting_topography,
            starting_topography_kwargs=starting_topography_kwargs,
            zref_limits=zref_limits,
            density_contrast_limits=density_contrast_limits,
            rmse_as_median=score_as_median,
            progressbar=fold_progressbar,
        )
        if grid_search:
            if num_params == 1:
                if n_trials < 4:
                    msg = (
                        "if grid_search is True, n_trials must be at least 4, "
                        "resetting n_trials to 4 now."
                    )
                    logger.warning(msg)
                    n_trials = 4

                if zref_limits is None:
                    space = np.linspace(
                        int(density_contrast_limits[0]),  # type: ignore[index]
                        int(density_contrast_limits[1]),  # type: ignore[index]
                        n_trials,
                        dtype=int,
                    )
                    sampler = optuna.samplers.GridSampler(
                        search_space={"density_contrast": space},
                        seed=seed,
                    )
                if density_contrast_limits is None:
                    space = np.linspace(zref_limits[0], zref_limits[1], n_trials)  # type: ignore[index]
                    sampler = optuna.samplers.GridSampler(
                        search_space={"zref": space},
                        seed=seed,
                    )
            else:
                if n_trials < 16:
                    msg = (
                        "if grid_search is True, n_trials must be at least 16, "
                        "resetting n_trials to 16 now."
                    )
                    logger.warning(msg)
                    n_trials = 16

                # n_trials needs to be square for 2 param grid search so each param has
                # sqrt(n_trials).
                if np.sqrt(n_trials).is_integer() is False:
                    # get next largest square number
                    old_n_trials = n_trials
                    n_trials = (math.floor(math.sqrt(n_trials)) + 1) ** 2
                    msg = (
                        "if grid_search is True with provided limits for both zref and "
                        "density contrast, n_trials (%s) must have an integer square "
                        "root. Resetting n_trials to to next largest compatible value "
                        "now (%s)"
                    )
                    logger.warning(msg, old_n_trials, n_trials)

                zref_space = np.linspace(
                    zref_limits[0],  # type: ignore[index]
                    zref_limits[1],  # type: ignore[index]
                    int(np.sqrt(n_trials)),
                )

                density_contrast_space = np.linspace(
                    int(density_contrast_limits[0]),  # type: ignore[index]
                    int(density_contrast_limits[1]),  # type: ignore[index]
                    int(np.sqrt(n_trials)),
                    dtype=int,
                )

                sampler = optuna.samplers.GridSampler(
                    search_space={
                        "zref": zref_space,
                        "density_contrast": density_contrast_space,
                    },
                    seed=seed,
                )

            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                load_if_exists=False,
                study_name=inv_copy.results_fname,
                storage=storage,
                pruner=optimization.DuplicateIterationPruner,
            )

            # run optimization
            with utils.DuplicateFilter(logger):  # type: ignore[no-untyped-call]
                study = optimization.run_optuna(
                    study=study,
                    storage=storage,
                    objective=objective,
                    n_trials=n_trials,
                    # callbacks=[_warn_limits_better_than_trial_multi_params],
                    maximize_cpus=True,
                    parallel=parallel,
                    progressbar=progressbar,
                )

        else:
            # define number of startup trials, whichever is bigger between 1/4 of trials, or
            # 4 x the number of parameters
            if n_startup_trials is None:
                n_startup_trials = max(num_params * 4, int(n_trials / 4))
                n_startup_trials = min(n_startup_trials, n_trials)
            logger.info("using %s startup trials", n_startup_trials)
            if n_startup_trials >= n_trials:
                logger.warning(
                    "n_startup_trials is >= n_trials resulting in all trials sampled from "
                    "a QMC sampler instead of the GP sampler",
                )

            # if sampler not provided, use GPsampler as default unless grid_search is True
            if sampler is None:
                sampler = optuna.samplers.GPSampler(
                    n_startup_trials=0,
                    seed=seed,
                    deterministic_objective=True,
                )

            # create study
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.QMCSampler(
                    seed=seed,
                    qmc_type="halton",
                    scramble=True,
                ),
                load_if_exists=False,
                study_name=inv_copy.results_fname,
                storage=storage,
                pruner=optimization.DuplicateIterationPruner,
            )

            # explicitly add the limits as trials
            to_enqueue = []

            if zref_limits is not None:
                zref_trials = [
                    {"zref": zref_limits[0]},
                    {"zref": zref_limits[1]},
                ]
                to_enqueue.append(zref_trials)
            if density_contrast_limits is not None:
                density_contrast_trials = [
                    {"density_contrast": density_contrast_limits[0]},
                    {"density_contrast": density_contrast_limits[1]},
                ]
                to_enqueue.append(density_contrast_trials)

            # get 2 lists of lists of dicts to enqueue (2 trials)
            to_enqueue = np.array(to_enqueue).transpose()

            for i in to_enqueue:
                # turn list of dicts into single dict
                x = {k: v for d in i for k, v in d.items()}
                study.enqueue_trial(x, skip_if_exists=True)

            # run optimization
            with utils.DuplicateFilter(logger):  # type: ignore[no-untyped-call]
                study = optimization.run_optuna(
                    study=study,
                    storage=storage,
                    objective=objective,
                    n_trials=n_startup_trials,
                    # callbacks=[_warn_limits_better_than_trial_multi_params],
                    maximize_cpus=True,
                    parallel=parallel,
                    progressbar=progressbar,
                )

            # continue with remaining trials with user-defined sampler
            # if sampler not provided, used GPsampler as default
            if sampler is None:
                sampler = optuna.samplers.GPSampler(
                    n_startup_trials=0,
                    seed=seed,
                    deterministic_objective=True,
                )
            study.sampler = sampler
            with utils.DuplicateFilter(logger):  # type: ignore[no-untyped-call]
                study = optimization.run_optuna(
                    study=study,
                    storage=storage,
                    objective=objective,
                    n_trials=n_trials - n_startup_trials,
                    # callbacks=[_warn_limits_better_than_trial_multi_params],
                    maximize_cpus=True,
                    parallel=parallel,
                    progressbar=progressbar,
                )

        # warn if any best parameter values are at their limits
        optimization.warn_parameter_at_limits(study.best_trial)

        # log the results of the best trial
        optimization.log_optuna_results(study.best_trial)

        # combine testing and training to get a full constraints dataframe
        reg_constraints = regional_grav_kwargs.pop("constraints_df", None)  # type: ignore[union-attr]
        if starting_topography_kwargs is not None:
            starting_topography_kwargs.pop("constraints_df", None)

        if isinstance(constraints_df, pd.DataFrame):
            constraints_df = (
                pd.concat([constraints_df, reg_constraints])
                .drop_duplicates(subset=["easting", "northing", "upward"])
                .sort_index()
            )
        else:
            constraints_df = (
                pd.concat(constraints_df + reg_constraints)
                .drop_duplicates(subset=["easting", "northing", "upward"])
                .sort_index()
            )
        # add to regional grav kwargs
        if reg_constraints is not None:
            regional_grav_kwargs["constraints_df"] = constraints_df  # type: ignore[index]
            if starting_topography_kwargs is not None:
                starting_topography_kwargs["constraints_df"] = constraints_df
                if "weights" in starting_topography_kwargs:
                    starting_topography_kwargs["weights_col"] = (
                        starting_topography_kwargs["weights"].name
                    )

        # redo inversion with best parameters
        inv_copy.model = inv_copy.model.assign_attrs(
            {
                "zref": study.best_trial.params.get("zref", inv_copy.model.zref),
            }
        )
        inv_copy.model = inv_copy.model.assign_attrs(
            {
                "density_contrast": study.best_trial.params.get(
                    "density_contrast", inv_copy.model.density_contrast
                )
            }
        )

        if starting_topography_kwargs is not None:
            starting_topography_kwargs["upwards"] = inv_copy.model.zref

        # run the inversion workflow with the new best parameters
        with utils._log_level(logging.WARN):  # pylint: disable=protected-access
            create_starting_topography = True if starting_topography is None else False  # noqa: SIM210  # pylint: disable=simplifiable-if-expression

            inv_copy = run_inversion_workflow(
                grav_ds=inv_copy.data,
                create_starting_topography=create_starting_topography,
                starting_topography=starting_topography,
                starting_topography_kwargs=starting_topography_kwargs,
                regional_grav_kwargs=regional_grav_kwargs,
                calculate_regional_misfit=True,
                zref=inv_copy.model.zref,
                density_contrast=inv_copy.model.density_contrast,
                fname=inv_copy.results_fname,
                inversion_kwargs={
                    "max_iterations": inv_copy.max_iterations,
                    "l2_norm_tolerance": inv_copy.l2_norm_tolerance,
                    "delta_l2_norm_tolerance": inv_copy.delta_l2_norm_tolerance,
                    "perc_increase_limit": inv_copy.perc_increase_limit,
                    "deriv_type": inv_copy.deriv_type,
                    "jacobian_finite_step_size": inv_copy.jacobian_finite_step_size,
                    "solver_type": inv_copy.solver_type,
                    "solver_damping": inv_copy.solver_damping,
                    "apply_weighting_grid": inv_copy.apply_weighting_grid,
                    "weighting_grid": inv_copy.weighting_grid,
                },
                progressbar=False,
            )

        used_zref = float(inv_copy.params["Reference level"][:-2])  # type: ignore[index]
        used_density_contrast = float(inv_copy.params["Density contrast(s)"][1:-7])  # type: ignore[index]

        assert math.isclose(
            used_density_contrast, inv_copy.model.density_contrast, rel_tol=0.02
        )
        assert math.isclose(used_zref, inv_copy.model.zref, rel_tol=0.02)

        # remove if exists
        pathlib.Path(f"{inv_copy.results_fname}_study.pickle").unlink(missing_ok=True)
        pathlib.Path(f"{inv_copy.results_fname}.pickle").unlink(missing_ok=True)

        # save study to pickle
        with pathlib.Path(f"{inv_copy.results_fname}_study.pickle").open("wb") as f:
            pickle.dump(study, f)

        # save inversion results tuple to pickle
        with pathlib.Path(f"{inv_copy.results_fname}.pickle").open("wb") as f:
            pickle.dump(inv_copy, f)

        # delete all inversion results
        for i in range(n_trials):
            pathlib.Path(f"{inv_copy.results_fname}_trial_{i}.pickle").unlink(
                missing_ok=True
            )

        inv_copy.zref_density_cv_study_fname = f"{inv_copy.results_fname}_study.pickle"  # type: ignore[assignment]
        inv_copy.zref_density_cv_results_fname = f"{inv_copy.results_fname}.pickle"  # type: ignore[assignment]
        inv_copy.study = study
        inv_copy.best_trial = study.best_trial

        # update the inversion object with the best inversion results
        self.__dict__.update(inv_copy.__dict__)

        if plot_cv is True:
            try:
                if zref_limits is None:
                    plotting.plot_cv_scores(
                        study.trials_dataframe().value.to_numpy(),
                        study.trials_dataframe().params_density_contrast.to_numpy(),
                        param_name="Density contrast (kg/m$^3$)",
                        plot_title="Density contrast Cross-validation",
                        logx=logx,
                        logy=logy,
                    )
                elif density_contrast_limits is None:
                    plotting.plot_cv_scores(
                        study.trials_dataframe().value.to_numpy(),
                        study.trials_dataframe().params_zref.to_numpy(),
                        param_name="Reference level (m)",
                        plot_title="Reference level Cross-validation",
                        logx=logx,
                        logy=logy,
                    )
                elif grid_search is True:
                    parameter_pairs = list(
                        zip(
                            study.trials_dataframe().params_zref,
                            study.trials_dataframe().params_density_contrast,
                            strict=False,
                        )
                    )
                    plotting.plot_2_parameter_cv_scores(
                        study.trials_dataframe().value.to_numpy(),
                        parameter_pairs,
                        param_names=(
                            "Reference level (m)",
                            "Density contrast (kg/m$^3$)",
                        ),
                    )
                else:
                    plotting.plot_2_parameter_cv_scores_uneven(
                        study,
                        param_names=(
                            "params_zref",
                            "params_density_contrast",
                        ),
                        plot_param_names=(
                            "Reference level (m)",
                            "Density contrast (kg/m$^3$)",
                        ),
                    )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("plotting failed with error: %s", e)

        return inv_copy

    def optimize_inversion_zref_density_contrast_kfolds(
        self,
        constraints_df: pd.DataFrame,
        split_kwargs: dict[str, typing.Any] | None = None,
        **kwargs: typing.Any,
    ) -> "Inversion":
        """
        Perform an optimization for zref and density contrast values same as
        function `optimize_inversion_zref_density_contrast`, but pass a dataframe of
        constraint points and `split_kwargs` which are both passed `split_test_train` create
        K-folds of testing and training constraints. For each set of zref/density values,
        regional separation and inversion are performed for each of the K-folds in the
        constraints dataframe. The score for each parameter set will be the mean of the
        K-folds scores.
        This then repeats for all parameters. Within each parameter set and fold, the
        training constraints are used for the regional separation and the testing
        constraints are used for scoring. This optimization performs a total number of
        inversions equal to  K-folds * number of parameter sets. For 20 parameter sets and 5
        K-folds, this is 100 inversions. This extra computational expense is only useful if
        the regional separation technique you supply via `regional_grav_kwargs` uses
        constraints points for the estimations, such as constraint point minimization
        (method='constraints_cv' or method='constraints'). It is more
        efficient, but less accurate, to simple use a different regional estimation
        technique, which doesn't require constraint points, to find the optimal zref and
        density values. Then use these again in another inversion with the desired regional
        separation technique. Using the regional method of "constraints" will simply use the
        training points and supplied `grid_method` parameter values to calculate a regional
        field. Using the regional method of "constraints_cv" will take the training points
        and split these into a secondary set of training and testing points. These will be
        used internally in the regional separation to find the optimal `grid_method`
        parameters.

        Parameters
        ----------
        constraints_df
            constraints dataframe with columns "easting", "northing", and "upward".
        split_kwargs : dict[str, typing.Any] | None, optional
            kwargs to be passed to `split_test_train` for splitting constraints_df into
            test and train sets, by default None
        **kwargs : typing.Any
            kwargs to be passed to `optimize_inversion_zref_density_contrast`

        Returns
        -------
        Inversion
            Inversion object with best inversion results and the optimally determined
            zref and or density contrast values as attributes of the `model` attribute.
        """
        # make copies of Inversion and underlying data and model dataset so as not
        # to alter the original
        inv_copy = copy.deepcopy(self)

        # drop any existing fold columns
        df = constraints_df.copy()
        df = df[df.columns.drop(list(df.filter(regex="fold_")))]

        kwargs = copy.deepcopy(kwargs)

        # split into test and training sets
        testing_training_df = cross_validation.split_test_train(
            df,
            **split_kwargs,  # type: ignore[arg-type]
        )

        # get list of training and testing dataframes
        test_dfs, train_dfs = cross_validation.kfold_df_to_lists(testing_training_df)
        logger.info("Constraints split into %s folds", len(test_dfs))

        regional_grav_kwargs = kwargs.pop("regional_grav_kwargs", None)

        starting_topography_kwargs = kwargs.pop("starting_topography_kwargs", None)

        if regional_grav_kwargs is None:
            msg = "must provide regional_grav_kwargs"
            raise ValueError(msg)

        if starting_topography_kwargs is None:
            msg = "must provide starting_topography_kwargs"
            raise ValueError(msg)

        regional_grav_kwargs["constraints_df"] = train_dfs

        starting_topography_kwargs["constraints_df"] = train_dfs

        if "weights" in starting_topography_kwargs:
            starting_topography_kwargs["weights_col"] = starting_topography_kwargs[
                "weights"
            ].name

        inv_copy = inv_copy.optimize_inversion_zref_density_contrast(
            constraints_df=test_dfs,
            regional_grav_kwargs=regional_grav_kwargs,
            starting_topography_kwargs=starting_topography_kwargs,
            **kwargs,
        )

        # update the inversion object with the best inversion results
        self.__dict__.update(inv_copy.__dict__)

        return inv_copy

    ###
    ###
    # PLOTTING METHODS
    ###
    ###

    def plot_dynamic_convergence(self) -> None:
        """
        plot a dynamic graph of L2-norm and delta L2-norm vs iteration number.
        """
        sns.set_theme()

        clear_output(wait=True)

        # create figure instance
        _fig, ax1 = plt.subplots(figsize=(5, 3.5))

        # make second y axis for delta l2 norm
        ax2 = ax1.twinx()

        # plot L2-norm convergence
        ax1.plot(
            list(range(self.iteration + 1)),  # type: ignore[operator]
            self.stats_df.l2_norm.to_numpy(),  # type: ignore[attr-defined]
            "b-",
        )

        # plot delta L2-norm convergence
        if self.iteration > 1:  # type: ignore[operator]
            ax2.plot(
                list(range(self.iteration + 1)),  # type: ignore[operator]
                self.stats_df.delta_l2_norm.to_numpy(),  # type: ignore[attr-defined]
                "g-",
            )

        # set axis labels, ticks and gridlines
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("L2-norm", color="b")
        ax1.tick_params(axis="y", colors="b", which="both")
        ax2.set_ylabel("Δ L2-norm", color="g")
        ax2.tick_params(axis="y", colors="g", which="both")
        ax2.grid(False)

        # add buffer to y axis limits
        ax1.set_ylim(0.9 * self.l2_norm_tolerance, self.stats_df.l2_norm.iloc[0])  # type: ignore[attr-defined]
        if self.iteration > 1:  # type: ignore[operator]
            ax2.set_ylim(
                self.delta_l2_norm_tolerance,
                np.nanmax(self.stats_df.delta_l2_norm.to_numpy()[1:]),  # type: ignore[attr-defined]
            )

        # set x axis to integer values
        ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        # plot current L2-norm and Δ L2-norm
        ax1.plot(
            self.iteration,
            self.l2_norm,
            "^",
            markersize=6,
            color=sns.color_palette()[3],
            # label="current L2-norm",
        )
        if self.iteration > 1:  # type: ignore[operator]
            ax2.plot(
                self.iteration,
                self.delta_l2_norm,
                "^",
                markersize=6,
                color=sns.color_palette()[3],
                # label="current Δ L2-norm",
            )

        # make both y axes align at tolerance levels
        plotting.align_yaxis(
            ax1, self.l2_norm_tolerance, ax2, self.delta_l2_norm_tolerance
        )

        # plot horizontal line of tolerances
        ax2.axhline(
            y=self.delta_l2_norm_tolerance,
            linewidth=1,
            color="r",
            linestyle="dashed",
            label="tolerances",
        )

        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")

        plt.title("Inversion convergence")
        plt.tight_layout()
        plt.show()

    def plot_convergence(self) -> None:
        """
        plot a graph of L2-norm and delta L2-norm vs iteration number.
        """
        sns.set_theme()

        # create figure instance
        _fig, ax1 = plt.subplots(figsize=(5, 3.5))

        # make second y axis for delta l2 norm
        ax2 = ax1.twinx()

        # plot L2-norm convergence
        ax1.plot(range(self.iteration + 1), self.stats_df.l2_norm.to_numpy(), "b-")  # type: ignore[attr-defined, operator]

        # plot delta L2-norm convergence
        ax2.plot(
            range(self.iteration + 1),  # type: ignore[operator]
            self.stats_df.delta_l2_norm.to_numpy(),  # type: ignore[attr-defined]
            "g-",
        )

        # set axis labels, ticks and gridlines
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("L2-norm", color="b")
        ax1.tick_params(axis="y", colors="b", which="both")
        ax2.set_ylabel("Δ L2-norm", color="g")
        ax2.tick_params(axis="y", colors="g", which="both")
        ax2.grid(False)

        # add buffer to y axis limits
        ax1.set_ylim(0.9 * self.l2_norm_tolerance, self.stats_df.l2_norm.iloc[0])  # type: ignore[attr-defined]
        ax2.set_ylim(
            self.delta_l2_norm_tolerance,
            np.nanmax(self.stats_df.delta_l2_norm.to_numpy()[1:]),  # type: ignore[attr-defined]
        )

        # set x axis to integer values
        ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        # make both y axes align at tolerance levels
        plotting.align_yaxis(
            ax1, self.l2_norm_tolerance, ax2, self.delta_l2_norm_tolerance
        )

        # plot horizontal line of tolerances
        ax2.axhline(
            y=self.delta_l2_norm_tolerance,
            linewidth=1,
            color="r",
            linestyle="dashed",
            label="tolerances",
        )

        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")

        plt.title("Inversion convergence")
        plt.tight_layout()
        plt.show()

    def plot_inversion_results(
        self,
        iters_to_plot: int | None = None,
        plot_iter_results: bool = True,
        plot_topo_results: bool = True,
        plot_grav_results: bool = True,
        constraints_df: pd.DataFrame | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """
        plot various results from the inversion

        Parameters
        ----------
        iters_to_plot : int | None, optional
            number of iterations to plot, including the first and last, by default None
        plot_iter_results : bool, optional
            plot the iteration results, by default True
        plot_topo_results : bool, optional
            plot the topography results, by default True
        plot_grav_results : bool, optional
            plot the gravity results, by default True
        constraints_df : pandas.DataFrame, optional
            constraint points to include in the plots
        """
        # get lists of columns to grid
        misfits = [
            s for s in self.data.inv.df.columns.to_list() if "initial_residual" in s
        ]
        topos = [
            s
            for s in self.model.inv.df.columns.to_list()
            if "_layer" in s and "confining" not in s
        ]
        corrections = [
            s for s in self.model.inv.df.columns.to_list() if "_correction" in s
        ]
        corrections = corrections[1:]

        # list of iterations, e.g. [1,2,3,4]
        its = list(range(1, self.iteration + 1))  # type: ignore[operator]

        # get on x amount of iterations to plot
        if iters_to_plot is not None:
            if iters_to_plot > max(its):
                iterations = its
            else:
                iterations = list(np.linspace(1, max(its), iters_to_plot, dtype=int))
        else:
            iterations = its

        # subset columns based on iterations to plot
        misfits = [misfits[i] for i in [x - 1 for x in iterations]]
        topos = [topos[i] for i in [x - 1 for x in iterations]]
        corrections = [corrections[i] for i in [x - 1 for x in iterations]]

        # grid all results
        ds = self.data.inv.inner
        misfit_grids = [ds[g] for g in misfits]
        ds = self.model.inv.inner
        topo_grids = [ds[g] for g in topos]
        correction_grids = [ds[g] for g in corrections]

        grids = (misfit_grids, topo_grids, correction_grids)

        if plot_iter_results is True:
            plotting.plot_inversion_iteration_results(
                grids,
                self.data.inv.inner_df,
                self.model.inv.masked_df,
                self.params,  # type: ignore[arg-type]
                iterations,
                topo_cmap_perc=kwargs.get("topo_cmap_perc", 1),
                misfit_cmap_perc=kwargs.get("misfit_cmap_perc", 1),
                corrections_cmap_perc=kwargs.get("corrections_cmap_perc", 1),
                constraints_df=constraints_df,
                constraint_size=kwargs.get("constraint_size", 1),
            )

        if plot_topo_results is True:
            plotting.plot_inversion_topo_results(
                self.model.inv.masked,
                constraints_df=constraints_df,
                constraint_style=kwargs.get("constraint_style", "x.3c"),
                fig_height=kwargs.get("fig_height", 12),
            )

        if plot_grav_results is True:
            plotting.plot_inversion_grav_results(
                self.data.inv.inner_df,
                self.data.inner_region,
                constraints_df=constraints_df,
                fig_height=kwargs.get("fig_height", 12),
                constraint_style=kwargs.get("constraint_style", "x.3c"),
            )


def run_inversion_workflow(
    grav_ds: xr.Dataset,
    create_starting_topography: bool = False,
    calculate_starting_gravity: bool = False,
    calculate_regional_misfit: bool = False,
    run_damping_cv: bool = False,
    run_zref_or_density_cv: bool = False,
    run_kfolds_zref_or_density_cv: bool = False,
    fname: str | None = None,
    starting_topography: xr.Dataset | None = None,
    starting_topography_kwargs: dict[str, typing.Any] | None = None,
    density_contrast: float | None = None,
    zref: float | None = None,
    model_type: str = "prisms",
    regional_grav_kwargs: dict[str, typing.Any] | None = None,
    constraints_df: pd.DataFrame | None = None,
    inversion_kwargs: dict[str, typing.Any] | None = None,
    damping_cv_kwargs: dict[str, typing.Any] | None = None,
    zref_density_cv_kwargs: dict[str, typing.Any] | None = None,
    progressbar: bool = True,
) -> "Inversion":
    """
    This function runs the full inversion workflow. Depending on the input parameters,
    it will:
    1) create a starting topography model
    2) create a starting prism model
    3) calculate the starting gravity of the prism model
    4) calculate the gravity misfit
    5) calculate the regional and residual components of the misfit
    6) run the inversion to update the prism model
    7) run a cross-validation for determining optimal values for damping, density, and
    zref

    Parameters
    ----------
    grav_ds : xr.Dataset
        gravity data with variables 'upward' and 'gravity_anomaly'.
    create_starting_topography : bool, optional
        Choose whether to create starting topography model. If True, must provide
        `starting_topography_kwargs`, if False must provide `starting_topography` by
        default False
    calculate_starting_gravity : bool, optional
        Choose whether to calculate starting gravity from prisms model. If False, must
        provide column "starting_gravity" in grav_df , by default False
    calculate_regional_misfit : bool, optional
        Choose whether to calculate regional misfit. If False, must provide column "reg"
        in grav_df, if True, must provide`regional_grav_kwargs`, by default False
    run_damping_cv : bool, optional
        Choose whether to run cross validation for damping, if True, must provide
        dictionary `damping_cv_kwargs` with parameters `n_trials` and `damping_limits`,
        by default False
    run_zref_or_density_cv : bool, optional
        Choose whether to run cross validation for zref or density, if True, must
        provide dictionary `zref_density_cv_kwargs` with parameters `n_trials`, and
        either `zref_values` or `density_values`, by default False
    run_kfolds_zref_or_density_cv : bool, optional
        Choose whether to run internal kfolds cross validation for zref or density, if
        True, must provide `split_kwargs` as argument to `zref_density_cv_kwargs`, by
        default False
    fname : str | None, optional
        filename and path to use for saving results. If running a damping
        CV, will save the study to <fname>_damping_cv_study.pickle and the tuple of the
        best inversion results to <fname>_damping_cv.pickle. If running a
        density/zref CV, will save the study to <fname>_zref_density_cv_study.pickle and
        the tuple of the best inversion results to
        <fname>_zref_density_cv.pickle. The final inversion result for all
        methods will be saved to <fname>.pickle, by default will be "tmp_<x>"
        where x is a random integer between 0 and 999.
    starting_topography : xr.Dataset | None, optional
        a starting topography model with variable `upward`, by default None
    starting_topography_kwargs : dict[str, typing.Any] | None, optional
        kwargs needed for create a starting topography grid, passed to
        `create_topography()`, by default None
    density_contrast : float | None, optional
        density contrast for the starting prisms, by default None
    zref : float | None, optional
        reference depth for the starting prisms, by default None
    model_type : str, optional
        type of model to create, either "prisms" or "tesseroids", by default "prisms"
    regional_grav_kwargs : dict[str, typing.Any] | None, optional
        kwargs needed for estimating regional gravity, passed to
        `regional_separation()`, by default None
    constraints_df : pd.DataFrame | None, optional
        Dataframe of constraint points, by default None
    inversion_kwargs : dict[str, typing.Any] | None, optional
        kwargs to be passed to `Inversion()`, by default None
    damping_cv_kwargs : dict[str, typing.Any] | None, optional
        kwargs to be passed to `optimize_inversion_damping()`, by default None
    zref_density_cv_kwargs : dict[str, typing.Any] | None, optional
        kwargs to be passed to `optimize_inversion_zref_density_contrast()` or
        `optimize_inversion_zref_density_contrast_kfolds()`, by default None
    progressbar : bool, optional
        whether to show progress bar during inversion, by default True

    Returns
    -------
    Inversion
        a inversion object with all results and optimal values as attributes.
    """

    if isinstance(grav_ds, pd.DataFrame):
        msg = "`run_inversion_workflow` function has been updated, gravity data must be provided to parameter `grav_ds` created through the `create_data` function"
        raise DeprecationWarning(msg)

    # set file name for saving results with random number between 0 and 999
    if fname is None:
        fname = f"tmp_{random.randint(0, 999)}"

    logger.info("saving all results with root name '%s'", fname)
    ###
    ###
    # figure out what needs to be done
    ###
    ###
    if starting_topography is None:
        create_starting_topography = True
    # if creating starting topo, must also create starting prisms
    if create_starting_topography is True:
        calculate_starting_gravity = True
    # if calculating starting gravity, must also calculate gravity misfit
    if calculate_starting_gravity is True:
        calculate_regional_misfit = True
    logger.debug("creating starting topo: %s", create_starting_topography)
    logger.debug("calculating starting gravity: %s", calculate_starting_gravity)
    logger.debug("calculating regional misfit: %s", calculate_regional_misfit)

    grav_ds = grav_ds.copy()

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
    if run_kfolds_zref_or_density_cv is True and run_zref_or_density_cv is False:
        msg = "run_zref_or_density_cv must be True if run_kfolds_zref_or_density_cv is True"
        raise ValueError(msg)
    if run_zref_or_density_cv is True:
        if constraints_df is None:
            msg = "must provide constraints_df if run_zref_or_density_cv is True"
            raise ValueError(msg)
        if zref_density_cv_kwargs is None:
            msg = "must provide zref_density_cv_kwargs with parameters `n_trials`, and 1 or both of `zref_limits` and `density_contrast_limits` if run_zref_or_density_cv is True"
            raise ValueError(msg)
        if run_kfolds_zref_or_density_cv is True:
            if zref_density_cv_kwargs.get("split_kwargs") is None:
                msg = "split_kwargs must be provided if performing internal kfolds CV"
                raise ValueError(msg)
        elif "constraints_df" in regional_grav_kwargs:  # type: ignore[operator]
            msg = (
                "if performing density/zref CV, it's best to not use constraints "
                "in the regional separation"
            )
            logger.warning(msg)
        if zref_density_cv_kwargs.get("density_contrast_limits") is None:
            assert density_contrast is not None
        if zref_density_cv_kwargs.get("zref_limits") is None:
            assert zref is not None
        if (zref_density_cv_kwargs.get("density_contrast_limits") is None) & (
            zref_density_cv_kwargs.get("zref_limits") is None
        ):
            msg = (
                "must provide density_contrast_limits or zref_limits if run_zref_or_"
                "density_cv is True"
            )
            raise ValueError(msg)
    if run_damping_cv is True:
        if (
            ("test" in grav_ds)
            and (False in grav_ds.inv.df.test.unique())
            and (True in grav_ds.inv.df.test.unique())
        ):
            pass
        else:
            # resample data at 1/2 spacing to include test points for cross-validation
            grav_ds = cross_validation.add_test_points(grav_ds)
        if damping_cv_kwargs is None:
            msg = "must provide damping_cv_kwargs with parameters `damping_limits` and `n_trials` if run_damping_cv is True"
            raise ValueError(msg)

    # Starting Topography
    if create_starting_topography is False:
        if (starting_topography is None) & (run_zref_or_density_cv is False):
            msg = (
                "starting_topography must be provided since create_starting_topography "
                "is False."
            )
            raise ValueError(msg)
        logger.debug("not creating starting topo because it is provided")
    elif create_starting_topography is True:
        if starting_topography is not None:
            msg = (
                "starting_topography provided but unused since "
                "create_starting_topography is True"
            )
            logger.warning(msg)
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
            ).to_dataset(name="upward")
        logger.debug("starting topo created")

    # starting prism model
    model = create_model(
        zref=zref,  # type: ignore[arg-type]
        density_contrast=density_contrast,
        model_type=model_type,
        starting_topography=starting_topography,
    )
    logger.debug("starting prisms created")

    # starting gravity of prism model
    if calculate_starting_gravity is False:
        if "starting_gravity" not in grav_ds:
            msg = (
                "'starting_gravity' must be a variable of `grav_ds` if "
                "calculate_starting_gravity is False"
            )
            raise ValueError(msg)
        logger.debug("not calculating starting gravity because it is provided")
    elif calculate_starting_gravity is True:
        if "starting_gravity" in grav_ds:
            msg = (
                "'starting_gravity' already a variable of `grav_ds`, but is being "
                "overwritten since calculate_starting_gravity is True"
            )
            logger.warning(msg)
        logger.debug("calculating starting gravity")
        grav_ds.inv.forward_gravity(
            model,
            progressbar=False,
        )
        logger.debug("starting gravity calculated")

    # Regional Component of Misfit
    if calculate_regional_misfit is False:
        if ("misfit" not in grav_ds) & (run_zref_or_density_cv is False):
            msg = (
                "'misfit' must be a column of `grav_df` if calculate_regional_misfit is"
                " False"
            )
            raise ValueError(msg)
        if ("reg" not in grav_ds) & (run_zref_or_density_cv is False):
            msg = (
                "'reg' must be a column of `grav_df` if calculate_regional_misfit is"
                " False"
            )
            raise ValueError(msg)
        logger.debug("not calculating regional misfit because it is provided")
    elif calculate_regional_misfit is True:
        if "reg" in grav_ds:
            msg = (
                "'reg' already a column of `grav_df`, but is being overwritten since"
                " calculate_regional_misfit is True"
            )
            logger.warning(msg)
        logger.debug("calculating regional misfit")
        logger.debug("regional_grav_kwargs: %s", regional_grav_kwargs)
        grav_ds.inv.regional_separation(**regional_grav_kwargs)
        logger.debug("regional misfit calculated")

    # initialize inversion
    inv = Inversion(
        grav_ds,
        model,
        **inversion_kwargs,  # type: ignore[arg-type]
    )
    inv.results_fname = fname  # type: ignore[assignment]

    ###
    ###
    # SINGLE INVERSION
    ###
    ###
    # run only the inversion with specified damping, density, and zref values
    if (run_damping_cv is False) & (run_zref_or_density_cv is False):
        logger.info("running individual inversion")
        with utils._log_level(logging.WARN):  # pylint: disable=protected-access
            inv.invert(results_fname=fname, progressbar=progressbar)
        logger.info("inversion complete, results saved to '%s'", fname)
        return inv

    ###
    ###
    # DAMPING CV
    ###
    ###
    if run_damping_cv is True:
        logger.info("running damping cross validation")
        _damping_cv_obj = inv.optimize_inversion_damping(
            fname=f"{fname}_damping_cv",
            **damping_cv_kwargs,  # type: ignore[arg-type]
        )
        assert inv.solver_damping is not None
        logger.info(
            "damping cross validation complete with optimal damping: %f",
            inv.solver_damping,
        )

    ###
    ###
    # DENSITY / ZREF CV
    ###
    ###
    if run_zref_or_density_cv is True:
        logger.info("running zref and/or density contrast cross validation")
        # drop the testing data
        inv.data = cross_validation.remove_test_points(inv.data)

        # if chosen, run an internal K-folds CV for regional separation within the
        # density/Zref CV
        if run_kfolds_zref_or_density_cv is True:
            logger.info("running internal K-folds CV for regional separation")
            _zref_density_cv_obj = inv.optimize_inversion_zref_density_contrast_kfolds(  # type: ignore[arg-type]
                fname=f"{fname}_zref_density_cv",
                constraints_df=constraints_df,
                fold_progressbar=True,
                **zref_density_cv_kwargs,  # typing: ignore[arg-type]
            )
        # run the normal non-kfolds CV
        else:
            _zref_density_cv_obj = inv.optimize_inversion_zref_density_contrast(
                fname=f"{fname}_zref_density_cv",
                constraints_df=constraints_df,
                **zref_density_cv_kwargs,  # type: ignore[arg-type]
            )
        logger.info(
            "zref and/or density contrast cross validation complete with density contrast %s kg/m3 and zref %s m",
            inv.model.density_contrast,
            inv.model.zref,
        )

    # save inversion results to pickle
    pathlib.Path(f"{fname}.pickle").unlink(missing_ok=True)
    with pathlib.Path(f"{fname}.pickle").open("wb") as f:
        pickle.dump(inv, f)
    logger.info("results saved to %s", f"{fname}.pickle")

    # ensure final inversion used the best parameters
    if run_damping_cv is True:
        assert inv.solver_damping == _damping_cv_obj.study.best_trial.params["damping"]  # type: ignore[attr-defined]
        assert inv.params["Solver damping"] == inv.solver_damping  # type: ignore[index]
    if run_zref_or_density_cv is True:
        assert (
            inv.model.density_contrast
            == _zref_density_cv_obj.study.best_trial.params.get(  # type: ignore[attr-defined]
                "density_contrast", density_contrast
            )
        )
        assert inv.model.zref == _zref_density_cv_obj.study.best_trial.params.get(  # type: ignore[attr-defined]
            "zref", zref
        )

        used_zref = float(inv.params["Reference level"][:-2])  # type: ignore[index]
        used_density_contrast = float(inv.params["Density contrast(s)"][1:-7])  # type: ignore[index]
        assert math.isclose(
            used_density_contrast, inv.model.density_contrast, rel_tol=0.02
        )
        assert math.isclose(used_zref, inv.model.zref, rel_tol=0.02)

    return inv
