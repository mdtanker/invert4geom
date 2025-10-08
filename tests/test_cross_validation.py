import numpy as np
import pandas as pd
import verde as vd

import invert4geom


def test_remove_test_points():
    """
    test the remove_test_points function
    """
    # create 6x6 topography grid
    easting = [0, 10000, 20000, 30000, 40000]
    northing = [0, 10000, 20000, 30000]
    grav_vals = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    grav = vd.make_xarray_grid(
        (easting, northing),
        data=(grav_vals, np.full_like(grav_vals, 1000)),
        data_names=("gravity_anomaly", "upward"),
    )

    data = invert4geom.inversion.create_data(grav)

    resampled = invert4geom.cross_validation.add_test_points(data)

    removed = invert4geom.cross_validation.remove_test_points(resampled)

    pd.testing.assert_frame_equal(
        removed.inv.df,
        data.inv.df,
    )


def test_add_test_points():
    """
    test the add_test_points function
    """
    # create 6x6 topography grid
    easting = [0, 10000, 20000, 30000, 40000]
    northing = [0, 10000, 20000, 30000]
    grav_vals = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    grav = vd.make_xarray_grid(
        (easting, northing),
        data=(grav_vals, np.full_like(grav_vals, 1000)),
        data_names=("gravity_anomaly", "upward"),
    )

    data = invert4geom.inversion.create_data(grav)

    resampled = invert4geom.cross_validation.add_test_points(data)

    pd.testing.assert_frame_equal(
        resampled.inv.df[resampled.inv.df.test == False]  # noqa: E712
        .drop(columns="test")
        .reset_index(drop=True),
        data.inv.df.reset_index(drop=True),
    )
