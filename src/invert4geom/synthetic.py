from __future__ import annotations

import logging

import numpy as np
import verde as vd
import xarray as xr
from nptyping import NDArray


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
    x, y : NDArray
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
    NDArray
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
    xr.Dataset
        synthetic topography dataset
    """
    if registration == "g":
        pixel_register = False
    elif registration == "p":
        pixel_register = True

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
    xr.Dataset
        synthetic topography dataset
    """

    if registration == "g":
        pixel_register = False
    elif registration == "p":
        pixel_register = True

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
    data : NDArray | list[NDArray]
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
    tuple[NDArray | list[NDArray], float | list[float]]
        a tuple of (contaminated data, standard deviations).

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
            logging.info("Standard deviation used for noise: %s", stddev)
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
