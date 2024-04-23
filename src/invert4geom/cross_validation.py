from __future__ import annotations

import logging
import typing

import numpy as np
import pandas as pd
import verde as vd
import xarray as xr
from polartoolkit import utils as polar_utils
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


def grav_cv_score(
    training_data: pd.DataFrame,
    testing_data: pd.DataFrame,
    progressbar: bool = False,
    plot: bool = False,
    **kwargs: typing.Any,
) -> float:
    """
    Find the score, represented by the root mean squared error (RMSE), between the
    testing gravity data, and the predict gravity data after and
    inversion. Follows methods of :footcite:t:`uiedafast2017`.

    Parameters
    ----------
    training_data : pd.DataFrame
       rows of the data frame which are just the training data
    testing_data : pd.DataFrame
        rows of the data frame which are just the testing data
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

    References
    ----------
    :footcite:t:`uiedafast2017`
    """

    train = training_data.copy()
    test = testing_data.copy()

    zref: float = kwargs.get("zref")  # type: ignore[assignment]
    density_contrast: float = kwargs.get("density_contrast")  # type: ignore[assignment]

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
        zref=zref,
        density_contrast=density_contrast,
        **new_kwargs,
    )
    prism_results, _, _, _ = results

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

    # compare forward of inverted layer with observed
    observed = test[kwargs.get("input_grav_column")] - test.reg
    predicted = test.test_point_grav

    dif = predicted - observed

    score = utils.rmse(dif)

    if plot:
        test_grid = test.set_index(["northing", "easting"]).to_xarray()
        obs = test_grid[kwargs.get("input_grav_column")] - test_grid.reg
        pred = test_grid.test_point_grav.rename("")

        polar_utils.grd_compare(
            pred,
            obs,
            grid1_name="Predicted gravity",
            grid2_name="Observed gravity",
            plot=True,
            plot_type="xarray",
            robust=True,
            title=f"Score={score}",
            rmse_in_title=False,
        )

    return score


def grav_optimal_parameter(
    training_data: pd.DataFrame,
    testing_data: pd.DataFrame,
    param_to_test: tuple[str, list[float]],
    progressbar: bool = False,
    plot_grids: bool = False,
    plot_cv: bool = False,
    verbose: bool = False,
    **kwargs: typing.Any,
) -> tuple[float, float, list[float], list[float]]:
    """
    Calculate the cross validation scores for a set of parameter values and return the
    best score and value.

    Parameters
    ----------
    training_data : pd.DataFrame
        just the training data rows
    testing_data : pd.DataFrame
        just the testing data rows
    param_to_test : tuple[str, list[float]]
        first value is a string of the parameter that is being tested, and the second
        value is a list of the values to test
    progressbar : bool, optional
        display a progress bar for the number of tested values, by default False
    plot_grids : bool, optional
        plot all the grids of observed and predicted data for each parameter value, by
        default False
    plot_cv : bool, optional
       plot a graph of scores vs parameter values, by default False
    verbose : bool, optional
       log the results, by default False

    Returns
    -------
    tuple[float, float, list[float], list[float]]
        the optimal parameter value, the score associated with it, the parameter values
        and the scores for each parameter value
    """

    train = training_data.copy()
    test = testing_data.copy()

    # pull parameter out of kwargs
    param_name = param_to_test[0]
    param_values = param_to_test[1]

    # run inversions and collect scores
    scores = []
    for value in tqdm(param_values, desc="Parameter values", disable=not progressbar):
        # update parameter value in kwargs
        kwargs[param_name] = value
        # run cross validation
        score = grav_cv_score(
            training_data=train,
            testing_data=test,
            plot=plot_grids,
            **kwargs,
        )
        scores.append(score)

    if verbose:
        # set Python's logging level to get information about the inversion\s progress
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    # print value and score pairs
    for value, score in zip(param_values, scores):
        logging.info("Parameter value: %s -> Score: %s", value, score)

    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_param_value = param_values[best_idx]
    logging.info(
        "Best score of %s with parameter value=%s", best_score, best_param_value
    )

    if plot_cv:
        # plot scores
        plotting.plot_cv_scores(
            scores,
            param_values,
            param_name=param_name,
            # logx=True,
            # logy=True,
        )

    return best_param_value, best_score, param_values, scores


def constraints_cv_score(
    grav: pd.DataFrame,
    constraints: pd.DataFrame,
    **kwargs: typing.Any,
) -> float:
    """
    Find the score, represented by the root mean squared error (RMSE), between the
    constraint point elevation, and the inverted topography at the constraint points.
    Follows methods of :footcite:t:`uiedafast2017`.

    Parameters
    ----------
    grav : pd.DataFrame
       gravity dataframe with columns "res", "reg", and column set by kwarg
       input_grav_column
    constraints : pd.DataFrame
        constraints dataframe with columns "easting", "northing", and "upward"

    Returns
    -------
    float
        a score, represented by the root mean squared error, between the testing gravity
        data and the predicted gravity data.

    References
    ----------
    .. footbibliography::
    """

    zref: float = kwargs.get("zref")  # type: ignore[assignment]
    density_contrast: float = kwargs.get("density_contrast")  # type: ignore[assignment]

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
        input_grav=grav,
        zref=zref,
        density_contrast=density_contrast,
        **new_kwargs,
    )
    prism_results, _, _, _ = results

    # grid resulting prisms dataframe
    prism_ds = prism_results.set_index(["northing", "easting"]).to_xarray()

    # get last iteration's layer result
    cols = [s for s in prism_results.columns.to_list() if "_layer" in s]
    final_surface = prism_ds[cols[-1]]

    # sample the inverted topography at the constraint points
    constraints = utils.sample_grids(
        constraints,
        final_surface,
        "inverted_topo",
        coord_names=("easting", "northing"),
    )

    dif = constraints.upward - constraints.inverted_topo

    return utils.rmse(dif)
