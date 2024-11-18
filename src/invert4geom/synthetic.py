from __future__ import annotations

import copy
import logging

import harmonica as hm
import numpy as np
import pandas as pd
import pooch
import verde as vd
import xarray as xr
from numpy.typing import NDArray
from polartoolkit import fetch, maps
from polartoolkit import utils as polar_utils

from invert4geom import cross_validation, log, utils

try:
    import xesmf
except ImportError:
    xesmf = None


def load_synthetic_model(
    spacing: float = 1e3,
    region: tuple[float, float, float, float] = (0, 40000, 0, 30000),
    buffer: float = 0,
    topography_coarsen_factor: float = 2,
    topography_percent_noise: float | None = None,
    number_of_constraints: int | None = None,
    density_contrast: float | None = None,
    zref: float | None = None,
    gravity_obs_height: float = 1000,
    gravity_noise: float | None = 0.2,
    resample_for_cv: bool = False,
    plot_topography: bool = False,
    plot_topography_diff: bool = True,
    plot_gravity: bool = True,
) -> tuple[xr.DataArray, xr.DataArray, pd.DataFrame, pd.DataFrame]:
    """
    Function to perform all necessary steps to create a synthetic model for the examples
    in the documentation.

    Parameters
    ----------
    spacing : float, optional
        spacing of the grid and gravity, by default 1e3
    region : tuple[float, float, float, float], optional
        bounding region for the grid, by default (0, 40000, 0, 30000)
    buffer : float, optional
        buffer to add around the region, by default 0. Buffer region used for creating
        topography and prisms, while inner region used for extent of gravity and
        constraints.
    topography_coarsen_factor : float, optional
        factor to coarsen the topography data by for adding noise, by default 2
    topography_percent_noise : float | None, optional
        noise decimal percent to add to topography data, by default None
    number_of_constraints : int | None, optional
        number of random constraints to use, by default None
    density_contrast : float | None, optional
        density contrast to use, by default None
    zref : float | None, optional
        reference level to use, by default None
    gravity_obs_height : float, optional
        gravity observation height to use, by default 1000
    gravity_noise : float | None, optional
        decimal percentage noise level to add to gravity data, by default 0.2
    resample_for_cv : bool, optional
        resample gravity data at half spacing to create train and test sets, by default
        False
    plot_topography : bool, optional
        plot the topography, by default False
    plot_topography_diff : bool, optional
        plot the difference between the true and starting topography, by default True
    plot_gravity : bool, optional
        plot the gravity data, by default True

    Returns
    -------
    true_topography : xarray.DataArray
        the true topography
    starting_topography : xarray.DataArray
        the starting topography
    constraint_points : pandas.DataFrame
        the constraint points
    grav_df : pandas.DataFrame
        the gravity data
    """

    buffer_region = vd.pad_region(region, buffer) if buffer != 0 else region

    true_topography = synthetic_topography_simple(spacing, buffer_region)

    if topography_percent_noise is not None:
        true_topography = contaminate_with_long_wavelength_noise(
            true_topography,
            coarsen_factor=topography_coarsen_factor,
            noise=topography_percent_noise,
            noise_as_percent=True,
        )
    # create random points within the region
    if number_of_constraints is not None:
        coords = vd.scatter_points(
            region=region,
            size=number_of_constraints,
            random_state=7,
        )
        constraint_points = pd.DataFrame(
            data={"easting": coords[0], "northing": coords[1]},
        )

        # sample simple topography at these points
        constraint_points = utils.sample_grids(
            constraint_points,
            true_topography,
            "upward",
            coord_names=("easting", "northing"),
        )

        with utils._log_level(logging.WARN):  # pylint: disable=protected-access
            # grid the sampled values using verde
            starting_topography = utils.create_topography(
                method="splines",
                region=buffer_region,
                spacing=spacing,
                constraints_df=constraint_points,
                dampings=np.logspace(-20, 0, 100),
            )

        # re-sample the starting topography at the constraint points to see how the
        # gridded did
        constraint_points = utils.sample_grids(
            constraint_points,
            starting_topography,
            "starting_topography",
            coord_names=("easting", "northing"),
        )
        rmse = utils.rmse(
            constraint_points.upward - constraint_points.starting_topography
        )
        msg = "RMSE at the constraints between the starting and true topography: %s m"
        log.info(msg, rmse)

        if plot_topography_diff is True:
            _ = polar_utils.grd_compare(
                true_topography,
                starting_topography,
                plot=True,
                grid1_name="True topography",
                grid2_name="Starting topography",
                robust=True,
                hist=True,
                inset=False,
                verbose="q",
                title="difference",
                grounding_line=False,
                reverse_cpt=True,
                cmap="rain",
                points=constraint_points.rename(
                    columns={"easting": "x", "northing": "y"}
                ),
                points_style="x.3c",
            )
    else:
        starting_topography = None
        constraint_points = None

    if plot_topography is True:
        # plot the topography
        fig = maps.plot_grd(
            true_topography,
            fig_height=10,
            title="True topography",
            reverse_cpt=True,
            cmap="rain",
            cbar_label="elevation (m)",
            frame=["nSWe", "xaf10000", "yaf10000"],
        )
        fig.show()

    if density_contrast is not None:
        if zref is None:
            zref = true_topography.values.mean()
        # prisms above zref have positive density contrast and prisms below zref have
        # negative density contrast
        density_grid = xr.where(
            true_topography >= zref,
            density_contrast,
            -density_contrast,
        )

        # create layer of prisms
        prisms = utils.grids_to_prisms(
            true_topography,
            zref,
            density=density_grid,
        )

        # make pandas dataframe of locations to calculate gravity
        # this represents the station locations of a gravity survey
        # create lists of coordinates
        coords = vd.grid_coordinates(
            region=region,
            spacing=spacing,
            pixel_register=False,
            extra_coords=gravity_obs_height,  # survey elevation
        )

        # grid the coordinates
        observations = vd.make_xarray_grid(
            (coords[0], coords[1]),
            data=coords[2],
            data_names="upward",
            dims=("northing", "easting"),
        ).upward

        grav_df = vd.grid_to_table(observations)

        # resample to half spacing
        if resample_for_cv is True:
            grav_df = cross_validation.resample_with_test_points(
                spacing, grav_df, region
            )
        # pylint: disable=duplicate-code
        grav_df["gravity_anomaly"] = prisms.prism_layer.gravity(
            coordinates=(
                grav_df.easting,
                grav_df.northing,
                grav_df.upward,
            ),
            field="g_z",
            progressbar=False,
        )
        # pylint: enable=duplicate-code
        # contaminate gravity with random noise
        if gravity_noise is not None:
            grav_df["gravity_anomaly"], _ = contaminate(
                grav_df.gravity_anomaly,
                stddev=gravity_noise,
                percent=False,
                seed=0,
            )
        if plot_gravity is True:
            # plot the observed gravity
            fig = maps.plot_grd(
                grav_df.set_index(["northing", "easting"]).to_xarray().gravity_anomaly,
                fig_height=10,
                title="Forward gravity of true topography",
                cmap="balance+h0",
                cbar_label="mGal",
                frame=["nSWe", "xaf10000", "yaf10000"],
            )
            fig.show()
    else:
        grav_df = None

    return true_topography, starting_topography, constraint_points, grav_df


def contaminate_with_long_wavelength_noise(
    grid: xr.DataArray,
    noise: float,
    coarsen_factor: float | None = None,
    spacing: float | None = None,
    noise_as_percent: bool = True,
) -> xr.DataArray:
    """
    Contaminate a grid with long wavelength noise.

    Parameters
    ----------
    grid : xarray.DataArray
        Grid to contaminate
    noise : float
        noise to add to the data, can be either absolute or percent of max value of
        data
    coarsen_factor : float | None, optional
        Factor to coarsen the data by, by default None
    spacing : float | None, optional
        Spacing for the long wavelength noise, by default None
    noise_as_percent : bool, optional
        if True, the value given to `noise` is treated as a percentage of the max value
        of the data.

    Returns
    -------
    xarray.DataArray
        Contaminated grid
    """
    if xesmf is None:
        msg = (
            "To use the `contaminate_with_long_wavelength_noise` function, you must "
            "have the `xesmf` package installed."
        )
        raise ImportError(msg)

    grid = copy.deepcopy(grid)

    # get original coordinate names
    original_dims = list(grid.sizes.keys())

    # get original grid name
    original_name = grid.name

    # get original spacing and region
    info = polar_utils.get_grid_info(grid)
    original_spacing = info[0]
    original_region = info[1]

    # resample at lower resolution
    if coarsen_factor is not None:
        new_spacing = original_spacing * coarsen_factor
    elif spacing is not None:
        new_spacing = spacing
    else:
        new_spacing = original_spacing

    low_res_grid = vd.make_xarray_grid(
        vd.grid_coordinates(original_region, spacing=new_spacing),
        data=None,
        data_names=None,
    ).rename({"northing": "lat", "easting": "lon"})

    low_res_grid = xesmf.Regridder(
        grid.rename({"northing": "lat", "easting": "lon"}),
        low_res_grid,
        method="bilinear",
    )(grid)

    low_res_grid = low_res_grid.rename(
        {
            list(low_res_grid.sizes.keys())[0]: original_dims[0],  # noqa: RUF015
            list(low_res_grid.sizes.keys())[1]: original_dims[1],
        }
    ).rename(original_name)

    # turn to dataframe and contaminate with noise
    df = low_res_grid.to_dataframe()

    df["noisy"], _ = contaminate(
        df[original_name],
        stddev=noise,
        percent=noise_as_percent,
        seed=1,
    )
    df["noise"] = df[original_name] - df.noisy

    new_grid = df.to_xarray().noise

    # resample back to original spacing
    new_grid = fetch.resample_grid(
        new_grid,
        spacing=original_spacing,
        region=original_region,
    )

    new_grid = new_grid.rename(
        {
            list(new_grid.sizes.keys())[0]: original_dims[0],  # noqa: RUF015
            list(new_grid.sizes.keys())[1]: original_dims[1],
        }
    ).rename(original_name)

    final_grid = new_grid + grid

    return final_grid.rename(
        {
            list(final_grid.sizes.keys())[0]: original_dims[0],  # noqa: RUF015
            list(final_grid.sizes.keys())[1]: original_dims[1],
        }
    ).rename(original_name)


def load_bishop_model(
    coarsen_factor: float | None = None,
) -> xr.Dataset:
    """
    Download and return a dataset of the Bishop model which contains basement
    topography, moho topography, and synthetically generated forward gravity of
    both topographies. See https://wiki.seg.org/wiki/Bishop_Model for more info on the
    derivation of this dataset.

    Parameters
    ----------
    coarsen_factor : float, optional
        Factor to coarsen the data by. Data originally at 200m resolution, by default
        None

    Returns
    -------
    xarray.Dataset
        Dataset with variables "basement_topo", "moho_topo", and "gravity".
    """

    url = "https://drive.usercontent.google.com/download?id=0B_notXWcvuh8dGZWbHlGODRMWEE&export=download&authuser=0&confirm=t&uuid=5d6bd6a1-a14b-48b6-ac76-6eda9f1baf1d&at=APZUnTVqVzwqlXHZAiYK-RoxHsH6%3A1713697379614"
    fname = "bishop_Geosoft_grids.tar.gz"
    known_hash = None
    path = pooch.retrieve(
        url=url,
        fname=fname,
        path=pooch.os_cache("bishop"),
        known_hash=known_hash,
        progressbar=True,
        processor=pooch.Untar(extract_dir=pooch.os_cache("bishop")),
    )
    geosoft_grids = [p for p in path if p.endswith((".grd", ".GRD"))]

    # get the necessary file paths
    basement_path = [p for p in geosoft_grids if p.endswith("bishop5x_basement.grd")][0]  # noqa: RUF015
    moho_path = [p for p in geosoft_grids if p.endswith("bishop5x_moho.GRD")][0]  # noqa: RUF015
    gravity_path = [p for p in geosoft_grids if p.endswith("bishop5x_gravity.grd")][0]  # noqa: RUF015

    # convert the .grd into xarray data arrays
    basement = hm.load_oasis_montaj_grid(basement_path)
    moho = hm.load_oasis_montaj_grid(moho_path)
    gravity = hm.load_oasis_montaj_grid(gravity_path)

    # merge into a dataset
    data = xr.Dataset(
        {
            "basement_topo": basement,
            "moho_topo": moho,
            "gravity": gravity,
        }
    )

    if coarsen_factor is not None:
        data = data.coarsen(  # pylint: disable=no-member
            easting=coarsen_factor, northing=coarsen_factor, boundary="trim"
        ).mean()

    return data


def gaussian2d(
    x: NDArray,
    y: NDArray,
    sigma_x: float,
    sigma_y: float,
    x0: float = 0,
    y0: float = 0,
    angle: float = 0.0,
) -> NDArray:
    """
    Non-normalized 2D Gaussian function for creating synthetic topography.

    Parameters
    ----------
    x, y : numpy.ndarray
        Coordinates at which to calculate the Gaussian function
    sigma_x, sigma_y : float
        Standard deviation in the x and y directions
    x0, y0 : float, optional
        Coordinates of the center of the distribution, by default 0
    angle : float, optional
        Rotation angle of the gaussian measure from the x axis (north) growing positive
        to the east (positive y axis), by default 0.0

    Returns
    -------
    numpy.ndarray
        Gaussian function evaluated at *x*, *y*

    Notes
    -----
    This function was adapted from the Fatiando-Legacy function
    gaussian2d: https://legacy.fatiando.org/api/utils.html?highlight=gaussian#fatiando.utils.gaussian2d
    """

    theta = -1 * angle * np.pi / 180.0
    tmpx = 1.0 / sigma_x**2
    tmpy = 1.0 / sigma_y**2
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    a = tmpx * costheta + tmpy * sintheta**2
    b = (tmpy - tmpx) * costheta * sintheta
    c = tmpx * sintheta**2 + tmpy * costheta**2
    xhat = x - x0
    yhat = y - y0
    return np.exp(-(a * xhat**2 + 2.0 * b * xhat * yhat + c * yhat**2))


def synthetic_topography_simple(
    spacing: float,
    region: tuple[float, float, float, float],
    registration: str = "g",
    scale: float = 1,
    yoffset: float = 0,
) -> xr.Dataset:
    """
    Create a synthetic topography dataset with a few features.

    Parameters
    ----------
    spacing : float
        grid spacing in meters
    region : tuple[float, float, float, float]
        bounding edges of the grid in meters in format (xmin, xmax, ymin, ymax)
    registration : str, optional
        grid registration type, either "g" for gridline or "p" for pixel, by default "g"
    scale : float, optional
        value to scale the topography by, by default 1
    yoffset : float, optional
        value to offset the topography by, by default 0

    Returns
    -------
    xarray.Dataset
        synthetic topography dataset
    """
    if registration == "g":
        pixel_register = False
    elif registration == "p":
        pixel_register = True
    else:
        msg = "registration must be either 'g' or 'p'"
        raise ValueError(msg)

    # create grid of coordinates
    (x, y) = vd.grid_coordinates(  # pylint: disable=unbalanced-tuple-unpacking
        region=region,
        spacing=spacing,
        pixel_register=pixel_register,
    )

    # get x and y range
    x_range = abs(region[1] - region[0])
    y_range = abs(region[3] - region[2])

    # create topographic features
    # regional
    f1 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 1.6,
            sigma_y=y_range * 1.6,
            x0=region[0] + x_range * 0.9,
            y0=region[2] + y_range * 0.3,
        )
        * -800
    )

    # high-frequency
    # circular
    f2 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.03,
            sigma_y=y_range * 0.03,
            x0=region[0] + x_range * 0.35,
            y0=region[2] + y_range * 0.5,
        )
        * -100
    )
    f3 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.08,
            sigma_y=y_range * 0.08,
            x0=region[0] + x_range * 0.65,
            y0=region[2] + y_range * 0.5,
        )
        * 200
    )

    # elongate
    f4 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.5,
            sigma_y=y_range * 0.06,
            x0=region[0] + x_range * 0.3,
            y0=region[2] + y_range * 0.7,
            angle=45,
        )
        * -300
    )
    f5 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 1.4,
            sigma_y=y_range * 0.04,
            x0=region[0] + x_range * 0.7,
            y0=region[2] + y_range * 0.7,
            angle=-45,
        )
        * 50
    )

    features = [
        f1,
        f2,
        f3,
        f4,
        f5,
    ]

    topo = sum(features)

    topo += 1200

    topo = topo * scale

    topo += yoffset

    return vd.make_xarray_grid(
        (x, y),
        topo,
        data_names="upward",
        dims=("northing", "easting"),
    ).upward


def synthetic_topography_regional(
    spacing: float,
    region: tuple[float, float, float, float],
    registration: str = "g",
    scale: float = 1,
    yoffset: float = 0,
) -> xr.Dataset:
    """
    Create a synthetic topography dataset with a few features which represent the
    surface responsible for the regional component of gravity.

    Parameters
    ----------
    spacing : float
        grid spacing in meters
    region : tuple[float, float, float, float]
        bounding edges of the grid in meters in format (xmin, xmax, ymin, ymax)
    registration : str, optional
        grid registration type, either "g" for gridline or "p" for pixel, by default "g"
    scale : float, optional
        value to scale the topography by, by default 1
    yoffset : float, optional
        value to offset the topography by, by default 0

    Returns
    -------
    xarray.Dataset
        synthetic topography dataset
    """

    if registration == "g":
        pixel_register = False
    elif registration == "p":
        pixel_register = True
    else:
        msg = "registration must be either 'g' or 'p'"
        raise ValueError(msg)

    # create grid of coordinates
    (x, y) = vd.grid_coordinates(  # pylint: disable=unbalanced-tuple-unpacking
        region=region,
        spacing=spacing,
        pixel_register=pixel_register,
    )

    # get x and y range
    x_range = abs(region[1] - region[0])
    y_range = abs(region[3] - region[2])

    # create topographic features
    feature1 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 2,
            sigma_y=y_range * 2,
            x0=region[0] + x_range,
            y0=region[2] + y_range * 0.5,
            angle=10,
        )
        * -150
        * scale
    ) - 3500
    feature2 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 3,
            sigma_y=y_range * 0.4,
            x0=region[0] + x_range * 0.2,
            y0=region[2] + y_range * 0.4,
            angle=-10,
        )
        * -100
        * scale
    )
    feature3 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.2,
            sigma_y=y_range * 7,
            x0=region[0] + x_range * 0.8,
            y0=region[2] + y_range * 0.7,
            angle=-80,
        )
        * 150
        * scale
    )

    features = [feature1, feature2, feature3]

    topo = sum(features)

    topo -= topo.mean()

    topo += yoffset
    return vd.make_xarray_grid(
        (x, y),
        topo,
        data_names="upward",
        dims=("northing", "easting"),
    ).upward


def contaminate(
    data: NDArray | list[NDArray],
    stddev: float | list[float],
    percent: bool = False,
    percent_as_max_abs: bool = True,
    seed: float = 0,
) -> tuple[NDArray | list[NDArray], float | list[float]]:
    """
    Add pseudorandom gaussian noise to an array.
    Noise added is normally distributed with zero mean and a standard deviation from
    *stddev*.

    Parameters
    ----------
    data : numpy.ndarray | list[numpy.ndarray]
        data to contaminate, can be a single array, or a list of arrays.
    stddev : float | list[float]
        standard deviation of the Gaussian noise that will be added to
        *data*. Length must be the same as *data* if *data* is a list.
    percent : bool, optional
        If ``True``, will consider *stddev* as a decimal percentage of the **data** and
        the standard deviation of the Gaussian noise will be calculated with this, by
        default False
    percent_as_max_abs : bool, optional
        If ``True``, and **percent** is ``True``, the *stddev* used as the standard
        deviation of the Gaussian noise will be the max absolute value of the **data**.
        If ``False``, and **percent** is ``True``, the *stddev* will be calculated on a
        point-by-point basis, so each **data** points' noise will be the same
        percentage, by default True
    seed : float, optional
        seed to use for the random number generator, by default 0

    Returns
    -------
    contam : numpy.ndarray | list[numpy.ndarray]
        contaminated data. If *data* is a list, will return a list of arrays.
    stddev : float | list[float]
        standard deviation of the Gaussian noise added to the data. If *stddev* is a
        list, will return a list of floats.

    Notes
    -----
    This function was adapted from the Fatiando-Legacy function
    gaussian2d: https://legacy.fatiando.org/api/utils.html?highlight=gaussian#fatiando.utils.contaminate

    Examples
    --------

    >>> import numpy as np
    >>> data = np.ones(5)
    >>> noisy, std = contaminate(data, 0.05, seed=0, percent=True)
    >>> print(std)
    0.05
    >>> print(noisy)
    array([1.00425372, 0.99136197, 1.02998834, 1.00321222, 0.97118374])
    >>> data = [np.zeros(5), np.ones(3)]
    >>> noisy = contaminate(data, [0.1, 0.2], seed=0)
    >>> print(noisy[0])
    array([ 0.00850745, -0.01727606,  0.05997669,  0.00642444, -0.05763251])
    >>> print(noisy[1])
    array([0.89814061, 1.0866216 , 1.01523779])

    """
    # initiate a random number generator
    rng = np.random.default_rng(seed)

    # Check if dealing with an array or list of arrays
    if not isinstance(stddev, list):
        stddev = [stddev]
    if not isinstance(data, list):
        data = [data]

    # Check that length of stdevs and data are the same
    assert len(stddev) == len(data), "Length of stddev and data must be the same"

    # ensure all stddevs are floats
    stddev = [float(i) for i in stddev]

    # Contaminate each array
    contam = []
    for i, _ in enumerate(stddev):
        # get list of standard deviations to use in Normal distribution
        # if stdev is zero, just add the uncontaminated data
        if stddev[i] == 0.0:
            contam.append(np.array(data[i]))
            continue
        if percent:
            if percent_as_max_abs:
                stddev[i] = stddev[i] * max(abs(data[i]))
            else:
                stddev[i] = stddev[i] * abs(data[i])
        if percent_as_max_abs is True:
            log.info("Standard deviation used for noise: %s", stddev)
        # use stdevs to generate random noise
        noise = rng.normal(scale=stddev[i], size=len(data[i]))
        # Subtract the mean so that the noise doesn't introduce a systematic shift in
        # the data
        noise -= noise.mean()
        # add the noise to the data
        contam.append(np.array(data[i]) + noise)

    if len(contam) == 1:
        contam = contam[0]
        stddev = stddev[0]

    return contam, stddev
