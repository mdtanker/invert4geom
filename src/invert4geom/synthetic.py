from __future__ import annotations

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
    From Fatiando-Legacy
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

    topo = topo + 1200

    return vd.make_xarray_grid(
        (x, y),
        topo,
        data_names="upward",
        dims=("northing", "easting"),
    ).upward
