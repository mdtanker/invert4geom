from __future__ import annotations

import logging
import pathlib
import pickle
import typing

import numpy as np
import pandas as pd
import verde as vd
import xarray as xr
from antarctic_plots import utils as ap_utils
from nptyping import NDArray
from tqdm.autonotebook import tqdm

from invert4geom import inversion, plotting, utils


def resample_with_test_points(
    data_spacing: float,
    data: pd.DataFrame,
    region: tuple[float, float, float, float],
) -> pd.DataFrame:
    """
    take a dataframe of coordinates and make all rows that fall on the data_spacing
    grid training points. Add rows at each point which falls on the grid points of
    half the data_spacing, assign these with label "test". If other data is present
    in dataframe, will sample at each new location.

    Parameters
    ----------
    data_spacing : float
        full spacing size which will be halved
    data : pd.DataFrame
        dataframe with coordinate columns "easting" and "northing", all other columns
        will be sampled at new grid spacing
    region : tuple[float, float, float, float]
        region to create grid over, in the form (min_easting, max_easting, min_northing,
        max_northing)

    Returns
    -------
    pd.DataFrame
        a new dataframe with new column "test" of booleans which shows whether each row
        is a testing or training point.
    """

    # create coords for full data
    coords = vd.grid_coordinates(
        region=region,
        spacing=data_spacing / 2,
        pixel_register=False,
    )

    # turn coordinates into dataarray
    full_points = vd.make_xarray_grid(
        (coords[0], coords[1]),
        data=np.ones_like(coords[0]),
        data_names="tmp",
        dims=("northing", "easting"),
    )
    # turn dataarray in dataframe
    full_df = vd.grid_to_table(full_points).drop(columns="tmp")
    # set all points to test
    full_df["test"] = True  # pylint: disable=unsupported-assignment-operation

    # subset training points, every other value
    train_df = full_df[  # pylint: disable=unsubscriptable-object
        (full_df.easting.isin(full_points.easting.values[::2]))
        & (full_df.northing.isin(full_points.northing.values[::2]))
    ].copy()
    # set training points to not be test points
    train_df["test"] = False

    # merge training and testing dfs
    df = full_df.set_index(["northing", "easting"])
    df.update(train_df.set_index(["northing", "easting"]))
    df2 = df.reset_index()

    df2["test"] = df2.test.astype(bool)

    grid = data.set_index(["northing", "easting"]).to_xarray()
    for i in list(grid):
        if i == "test":
            pass
        else:
            df2[i] = utils.sample_grids(
                df2,
                grid[i],
                i,
                coord_names=("easting", "northing"),
            )[i].astype(data[i].dtype)

    # test with this, using same input spacing as original
    # pd.testing.assert_frame_equal(df2, full_res_grav, check_like=True,)

    return df2


def inversion_damping_score(
    damping: float,
    training_data: pd.DataFrame,
    testing_data: pd.DataFrame,
    results_fname: str | None = None,
    progressbar: bool = False,
    plot: bool = False,
    **kwargs: typing.Any,
) -> float:
    """
    Find the score, represente by the root mean squared error (RMSE), between the
    testing gravity data, and the predict gravity data after and
    inversion. Follows methods of Uieda and Barbosa 2017.

    Parameters
    ----------
    damping : float
        damping parameter to use in inversion
    training_data : pd.DataFrame
       rows of the data frame which are just the training data
    testing_data : pd.DataFrame
        rows of the data frame which are just the testing data
    results_fname : str | None, optional
        file path and name if storing resulting pickle file, by default None
    progressbar : bool, optional
        choose to show the progress bar for the forward gravity calculation, by default
        False
    plot : bool, optional
        choose to plot the observed and predicted data grids, and their difference,
        located at the testing points, by default False

    Returns
    -------
    float
        a score, represented by the root mean squared error, between the testing gravity
        data and the predicted gravity data.
    """

    train = training_data.copy()
    test = testing_data.copy()

    zref = kwargs.get("zref")
    density_contrast = kwargs.get("density_contrast")

    new_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key
        not in [
            "zref",
            "density_contrast",
        ]
    }

    # run inversion
    results = inversion.run_inversion(
        input_grav=train,
        solver_damping=damping,
        zref=zref,
        density_contrast=density_contrast,
        **new_kwargs,
    )
    prism_results, _, _, _ = results

    # save results to pickle
    if results_fname is not None:
        # remove if exists
        pathlib.Path(f"{results_fname}.pickle").unlink(missing_ok=True)
        with pathlib.Path.open(f"{results_fname}.pickle", "wb", encoding="utf-8") as f:
            pickle.dump(results, f)

    # grid resulting prisms dataframe
    prism_ds = prism_results.set_index(["northing", "easting"]).to_xarray()

    # get last iteration's layer result
    cols = [s for s in prism_results.columns.to_list() if "_layer" in s]
    final_surface = prism_ds[cols[-1]]

    # create new prism layer
    prism_layer = utils.grids_to_prisms(
        surface=final_surface,
        reference=zref,
        density=xr.where(final_surface >= zref, density_contrast, -density_contrast),
    )

    # calculate forward gravity of starting prism layer
    test["test_point_grav"] = prism_layer.prism_layer.gravity(
        coordinates=(
            test.easting,
            test.northing,
            test.upward,
        ),
        field="g_z",
        progressbar=progressbar,
    )

    # compare new layer1 forward with observed
    observed = test[kwargs.get("input_grav_column")] - test.reg
    predicted = test.test_point_grav

    dif = predicted - observed

    score = utils.rmse(dif)

    if plot:
        test_grid = test.set_index(["northing", "easting"]).to_xarray()
        obs = test_grid[kwargs.get("input_grav_column")] - test_grid.reg
        pred = test_grid.test_point_grav.rename("")

        ap_utils.grd_compare(
            pred,
            obs,
            grid1_name="Predicted gravity",
            grid2_name="Observed gravity",
            plot=True,
            plot_type="xarray",
            robust=True,
            title=f"Damping={damping}, Score={score}",
            rmse_in_title=False,
        )

    return score


def inversion_optimal_damping(
    training_data: pd.DataFrame,
    testing_data: pd.DataFrame,
    damping_values: list | NDArray,
    progressbar=False,
    plot_grids=False,
    plot_cv=False,
    verbose=False,
    **kwargs,
) -> tuple[float, float]:
    """
    Calculate the optimal damping parameter, defined as the value which gives the lowest
    RMSE between the testing gravity data, and the predict gravity data after and
    inversion. Follows method of Uieda and Barbosa 2017.

    Parameters
    ----------
    training_data : pd.DataFrame
        just the training data rows
    testing_data : pd.DataFrame
        just the testing data rows
    damping_values : list | NDArray
        damping values to run inversions and collect scores for
    progressbar : bool, optional
        display a progress bar for the number of damping values, by default False
    plot_grids : bool, optional
        plot all the grids of observed and predicted data for each damping value, by
        default False
    plot_cv : bool, optional
       plot a graph of scores vs damping values, by default False
    verbose : bool, optional
       log the results, by default False

    Returns
    -------
    tuple[float, float]
        the optimal damping value and the score associated with it
    """

    train = training_data.copy()
    test = testing_data.copy()

    # run inversions and collect scores
    scores = []
    for value in tqdm(damping_values, desc="Damping Values", disable=not progressbar):
        score = inversion_damping_score(
            value,
            train,
            test,
            plot=plot_grids,
            **kwargs,
        )
        scores.append(score)

    if verbose:
        # set Python's logging level to get information about the inversion\s progress
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    # print damping and score pairs
    for damp, score in zip(damping_values, scores):
        logging.info("Damping: %s -> Score: %s", damp, score)

    best_score = np.argmin(scores)
    best_damping = damping_values[best_score]
    logging.info("Best score of %s with damping value=%s", best_score, best_damping)

    if plot_cv:
        # plot scores
        plotting.plot_cv_scores(
            scores,
            damping_values,
            param_name="Damping",
            logx=True,
            logy=True,
        )

    return best_damping, best_score
