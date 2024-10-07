from __future__ import annotations  # pylint: disable=too-many-lines

import copy
import itertools
import logging
import pathlib
import pickle
import random
import typing
import warnings

import deprecation
import harmonica as hm
import numpy as np
import pandas as pd
import sklearn
import verde as vd
import xarray as xr
from numpy.typing import NDArray
from polartoolkit import maps
from polartoolkit import utils as polar_utils
from tqdm.autonotebook import tqdm

import invert4geom
from invert4geom import inversion, log, plotting, regional, utils


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
    data : pandas.DataFrame
        dataframe with coordinate columns "easting" and "northing", all other columns
        will be sampled at new grid spacing
    region : tuple[float, float, float, float]
        region to create grid over, in the form (min_easting, max_easting, min_northing,
        max_northing)

    Returns
    -------
    pandas.DataFrame
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

    return df2.dropna()


def grav_cv_score(
    training_data: pd.DataFrame,
    testing_data: pd.DataFrame,
    progressbar: bool = True,
    rmse_as_median: bool = False,
    plot: bool = False,
    **kwargs: typing.Any,
) -> tuple[float, tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float]]:
    """
    Find the score, represented by the root mean (or median) squared error (RMSE),
    between the testing gravity data, and the predict gravity data after an
    inversion. Follows methods of :footcite:t:`uiedafast2017`. Used in
    `optimization.optimize_inversion_damping()`.

    Parameters
    ----------
    training_data : pandas.DataFrame
       rows of the gravity data frame which are just the training data
    testing_data : pandas.DataFrame
        rows of the gravity data frame which are just the testing data
    rmse_as_median : bool, optional
        calculate the RMSE as the median as opposed to the mean, by default False
    progressbar : bool, optional
        choose to show the progress bar for the forward gravity calculation, by default
        True
    plot : bool, optional
        choose to plot the observed and predicted data grids, and their difference,
        located at the testing points, by
        default False

    Returns
    -------
    score : float
        the root mean squared error, between the testing gravity data and the predicted
        gravity data
    results : tuple[pandas.DataFrame, pandas.DataFrame, dict[str, typing.Any], float]
        a tuple of the inversion results.

    References
    ----------
    :footcite:t:`uiedafast2017`
    """

    train = training_data.copy()
    test = testing_data.copy()

    # extract density contrast and zref from prism layer
    prism_layer: xr.Dataset = kwargs.get("prism_layer")
    density_contrast = np.fabs(prism_layer.density)
    zref = prism_layer.attrs.get("zref")

    # make sure dynamic plotting of inversion iterations is off
    kwargs["plot_dynamic_convergence"] = False
    with utils._log_level(logging.WARN):  # pylint: disable=protected-access
        # run inversion
        results = inversion.run_inversion(
            grav_df=train,
            progressbar=False,
            **kwargs,
        )

    prism_results, _, _, _ = results

    # get last iteration's layer result
    final_topography = prism_results.set_index(["northing", "easting"]).to_xarray().topo

    density_grid = xr.where(
        final_topography >= zref,
        density_contrast,
        -density_contrast,
    )

    # create new prism layer
    prism_layer = utils.grids_to_prisms(
        final_topography,
        reference=zref,
        density=density_grid,
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
    observed = test.gravity_anomaly - test.reg
    predicted = test.test_point_grav

    dif = predicted - observed

    score = utils.rmse(dif, as_median=rmse_as_median)

    if plot:
        test_grid = test.set_index(["northing", "easting"]).to_xarray()
        obs = test_grid.gravity_anomaly - test_grid.reg
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

    return score, results


@deprecation.deprecated(  # type: ignore[misc]
    deprecated_in="0.8.0",
    removed_in="0.14.0",
    current_version=invert4geom.__version__,
    details=(
        "Use the new function `optimization.optimize_inversion_damping()`"
        "instead, which uses Optuna for optimization. If you would still like to "
        "conduct a grid search, set `grid_search=True` in the new function.",
    ),
)
def grav_optimal_parameter(
    training_data: pd.DataFrame,
    testing_data: pd.DataFrame,
    param_to_test: tuple[str, list[float]],
    rmse_as_median: bool = False,
    progressbar: bool = True,
    plot_grids: bool = False,
    plot_cv: bool = False,
    results_fname: str | None = None,
    **kwargs: typing.Any,
) -> tuple[
    tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float],
    float,
    float,
    list[float],
    list[float],
]:
    """
    Calculate the cross validation scores for a set of parameter values and return the
    best score and value.

    Parameters
    ----------
    training_data : pandas.DataFrame
        just the training data rows
    testing_data : pandas.DataFrame
        just the testing data rows
    param_to_test : tuple[str, list[float]]
        first value is a string of the parameter that is being tested, and the second
        value is a list of the values to test
    rmse_as_median : bool, optional
        calculate the RMSE as the median as opposed to the mean, by default False
    progressbar : bool, optional
        display a progress bar for the number of tested values, by default True
    plot_grids : bool, optional
        plot all the grids of observed and predicted data for each parameter value, by
        default False
    plot_cv : bool, optional
       plot a graph of scores vs parameter values, by default False
    results_fname : str, optional
        file name to save results to, by default "tmp" with an attached random number

    Returns
    -------
    tuple[ tuple[pandas.DataFrame, pandas.DataFrame, dict[str, typing.Any], float],
        float, float, list[float], list[float], ]
        the inversion results, the optimal parameter value, the score associated with
        it, the parameter values and the scores for each parameter value
    """

    train = training_data.copy()
    test = testing_data.copy()

    # pull parameter out of kwargs
    param_name = param_to_test[0]
    param_values = param_to_test[1]

    # set file name for saving results with random number between 0 and 999
    if results_fname is None:
        results_fname = f"tmp_{random.randint(0,999)}"

    # run inversions and collect scores
    scores = []
    if progressbar is True:
        pbar = tqdm(
            param_values,
            desc=f"{param_name} values",
        )
    elif progressbar is False:
        pbar = param_values
    else:
        msg = "progressbar must be a boolean"  # type: ignore[unreachable]
        raise ValueError(msg)

    for i, value in enumerate(pbar):
        # update parameter value in kwargs
        kwargs[param_name] = value
        # run cross validation
        score, _ = grav_cv_score(
            training_data=train,
            testing_data=test,
            rmse_as_median=rmse_as_median,
            plot=plot_grids,
            results_fname=f"{results_fname}_trial_{i}",
            progressbar=False,
            **kwargs,
        )
        scores.append(score)
        if (i == 1) and (score > scores[0]):
            msg = (
                "first score was lower than second, consider changing the lower"
                " parameter value range"
            )
            log.warning(msg)

    # print value and score pairs
    for value, score in zip(param_values, scores):
        log.info("%s value: %s -> Score: %s", param_name, value, score)

    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_param_value = param_values[best_idx]
    log.info("Best score of %s with %s=%s", best_score, param_name, best_param_value)

    # get best inversion result of each set
    with pathlib.Path(f"{results_fname}_trial_{best_idx}.pickle").open("rb") as f:
        inv_results = pickle.load(f)

    # delete other inversion results
    for i in range(len(scores)):
        if i == best_idx:
            pass
        else:
            pathlib.Path(f"{results_fname}_trial_{i}.pickle").unlink(missing_ok=True)

    # put scores and parameter values into dict
    results = {
        "scores": scores,
        "param_values": param_values,
    }

    if best_param_value in [np.min(param_values), np.max(param_values)]:
        log.warning(
            "Best parameter value (%s) for %s CV is at the limit of provided "
            "values (%s, %s) and thus is likely not a global minimum, expand the range "
            "of values tested to ensure the best parameter value is found.",
            best_param_value,
            param_name,
            np.nanmin(param_values),
            np.nanmax(param_values),
        )

    # remove if exists
    pathlib.Path(results_fname).unlink(missing_ok=True)

    # save scores and dampings to pickle
    with pathlib.Path(f"{results_fname}.pickle").open("wb") as f:
        pickle.dump(results, f)

    if plot_cv:
        # plot scores
        plotting.plot_cv_scores(
            scores,
            param_values,
            param_name=param_name,
            # logx=True,
            # logy=True,
        )

    return inv_results, best_param_value, best_score, param_values, scores


def constraints_cv_score(
    grav_df: pd.DataFrame,
    constraints_df: pd.DataFrame,
    rmse_as_median: bool = False,
    **kwargs: typing.Any,
) -> tuple[float, tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float]]:
    """
    Find the score, represented by the root mean squared error (RMSE), between the
    constraint point elevation, and the inverted topography at the constraint points.
    Follows methods of :footcite:t:`uiedafast2017`. Used in
    `optimization.optimize_inversion_zref_density_contrast()`.

    Parameters
    ----------
    grav_df : pandas.DataFrame
       gravity dataframe with columns "res", "reg", and "gravity_anomaly"
    constraints_df : pandas.DataFrame
        constraints dataframe with columns "easting", "northing", and "upward"
    rmse_as_median : bool, optional
        calculate the RMSE as the median of the , as opposed to the mean, by default
        False

    Returns
    -------
    score : float
        the root mean squared error, between the constraint point elevation and the
        inverted topography at the constraint points
    results : tuple[pandas.DataFrame, pandas.DataFrame, dict[str, typing.Any], float]
        a tuple of the inversion results.

    References
    ----------
    .. footbibliography::
    """

    constraints_df = constraints_df.copy()

    with utils._log_level(logging.WARN):  # pylint: disable=protected-access
        # run inversion
        results = inversion.run_inversion(
            grav_df=grav_df,
            progressbar=False,
            **kwargs,
        )

    prism_results, _, _, _ = results

    # get last iteration's layer result
    final_topography = prism_results.set_index(["northing", "easting"]).to_xarray().topo

    # sample the inverted topography at the constraint points
    constraints_df = utils.sample_grids(
        constraints_df,
        final_topography,
        "inverted_topo",
        coord_names=("easting", "northing"),
    )

    dif = constraints_df.upward - constraints_df.inverted_topo

    return utils.rmse(dif, as_median=rmse_as_median), results


# pylint: disable=duplicate-code
@deprecation.deprecated(  # type: ignore[misc]
    deprecated_in="0.8.0",
    removed_in="0.14.0",
    current_version=invert4geom.__version__,
    details=(
        "Use the new function `optimization.optimize_inversion_zref_density_contrast()`"
        "instead, which uses Optuna for optimization. If you would still like to "
        "conduct a grid search, set `grid_search=True` in the new function.",
    ),
)
def zref_density_optimal_parameter(
    grav_df: pd.DataFrame,
    constraints_df: pd.DataFrame,
    starting_topography: xr.DataArray | None = None,
    zref_values: list[float] | None = None,
    density_contrast_values: list[float] | None = None,
    starting_topography_kwargs: dict[str, typing.Any] | None = None,
    regional_grav_kwargs: dict[str, typing.Any] | None = None,
    rmse_as_median: bool = False,
    progressbar: bool = True,
    plot_cv: bool = False,
    results_fname: str | None = None,
    **kwargs: typing.Any,
) -> tuple[
    tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float],
    float,
    float,
    float,
    list[typing.Any],
    list[float],
]:
    """
    Calculate the cross validation scores for a set of zref and density values and
    return the best score and values. If only 1 parameter is needed to be test, can pass
    a single value of the other parameter. This uses constraint points, where the target
    topography is known. The inverted topography at each of these points is compared to
    the known value and used to calculate the score.

    Parameters
    ----------
    grav_df : pandas.DataFrame
        dataframe with gravity data and coordinates, must have coordinate columns
        "easting", "northing", and "upward", and gravity data column "gravity_anomaly"
    constraints_df : pandas.DataFrame
        dataframe with points where the topography of interest has been previously
        measured, to be used for score, must have coordinate columns "easting",
        "northing", and "upward".
    starting_topography : xarray.DataArray | None, optional
        starting topography to use to create the starting prism model. If not provided,
        will make a flat surface at each provided zref value using the region and
        spacing values provided in starting_topography_kwargs.
    zref_values : list[float] | None, optional
        Reference level values to test, by default None
    density_contrast_values : list[float] | None, optional
        Density contrast values to test, by default None
    starting_topography_kwargs : dict[str, typing.Any] | None, optional
        region, spacing and dampings used to create a flat starting topography for each
        zref value, by default None.
    regional_grav_kwargs : dict[str, typing.Any] | None, optional
        Keywords used to calculate the regional field, by default None. If method is
        `constraints` for constraint point minimization, must separate the constraints
        into testing and training sets and provide the training set to this argument and
        the testing set to `constraints_df` to avoid biasing the scores.
    rmse_as_median : bool, optional
        Use the median instead of the root mean square as the scoring metric, by default
        False
    progressbar : bool, optional
        display a progress bar for the number of tested values, by default True
    plot_cv : bool, optional
        plot a graph of scores vs parameter values, by default False
    results_fname : str, optional
        file name to save results to, by default "tmp" with an attached random number

    Returns
    -------
    tuple[ tuple[pandas.DataFrame, pandas.DataFrame, dict[str, typing.Any], float],
        float, float, float, list[typing.Any], list[float], ]
        the inversion results, the optimal parameter value, the score associated with
        it, the parameter values and the scores for each parameter value
    """

    # set file name for saving results with random number between 0 and 999
    if results_fname is None:
        results_fname = f"tmp_{random.randint(0,999)}"

    if (zref_values is None) & (density_contrast_values is None):
        msg = "must provide either or both zref_values and density_contrast_values"
        raise ValueError(msg)

    if zref_values is None:
        zref = kwargs.get("zref")
        if zref is None:
            msg = "must provide zref_values or zref in kwargs"
            raise ValueError(msg)
        zref_values = [zref]
    elif density_contrast_values is None:
        density_contrast = kwargs.get("density_contrast")
        if density_contrast is None:
            msg = "must provide density_contrast_values or density_contrast in kwargs"
            raise ValueError(msg)
        density_contrast_values = [density_contrast]

    if starting_topography is None:
        msg = (
            "starting_topography not provided, will create a flat surface at each zref "
            "value to be the starting topography."
        )
        log.warning(msg)
        if starting_topography_kwargs is None:
            msg = (
                "must provide `starting_topography_kwargs` with items `region` "
                "`spacing`, and `dampings` to create the starting topography for each "
                "zref level."
            )
            raise ValueError(msg)

    # raise warning about using constraint point minimization for regional estimation
    if (
        (regional_grav_kwargs is not None)
        and (regional_grav_kwargs.get("method") == "constraints")
        and (len(regional_grav_kwargs.get("constraints_df")) == len(constraints_df))  # type: ignore[arg-type]
    ):
        msg = (
            "Using constraint point minimization technique for regional field "
            "estimation. This is not recommended as the constraint points are used for "
            "the density / reference level cross-validation scoring, which biases the "
            "scoring. Consider using a different method for regional field estimation, "
            "or set separate constraints in training and testing sets and provide the "
            "training set to `regional_grav_kwargs` and the testing set to "
            "constraints_df to use for scoring."
        )
        log.warning(msg)

    # create all possible combinations of zref and density contrast
    parameter_pairs = list(itertools.product(zref_values, density_contrast_values))  # type: ignore[arg-type]

    if "test" in grav_df.columns:
        assert (
            grav_df.test.any()
        ), "test column contains True value, not needed except for during damping CV"

    # run inversions and collect scores
    scores = []
    if progressbar is True:
        pbar = tqdm(
            parameter_pairs,
            desc="Zref/Density pairs",
        )
    elif progressbar is False:
        pbar = parameter_pairs
    else:
        msg = "progressbar must be a boolean"  # type: ignore[unreachable]
        raise ValueError(msg)

    for i, (zref, density_contrast) in enumerate(pbar):
        if starting_topography is None:
            starting_topo = utils.create_topography(
                method="flat",
                dampings=starting_topography_kwargs.get("dampings"),  # type: ignore[union-attr]
                region=starting_topography_kwargs.get("region"),  # type: ignore[union-attr, arg-type]
                spacing=starting_topography_kwargs.get("spacing"),  # type: ignore[union-attr, arg-type]
                upwards=zref,
            )
        else:
            starting_topo = starting_topography.copy()

        # re-calculate density grid with new density contrast
        density_grid = xr.where(
            starting_topo >= zref, density_contrast, -density_contrast
        )

        # create layer of prisms
        starting_prisms = utils.grids_to_prisms(
            starting_topo,
            reference=zref,
            density=density_grid,
        )
        # pylint: disable=duplicate-code
        # calculate forward gravity of starting prism layer
        grav_df["starting_gravity"] = starting_prisms.prism_layer.gravity(
            coordinates=(
                grav_df.easting,
                grav_df.northing,
                grav_df.upward,
            ),
            field="g_z",
            progressbar=False,
        )
        # pylint: enable=duplicate-code
        # calculate regional field
        reg_kwargs = copy.deepcopy(regional_grav_kwargs)

        grav_df = regional.regional_separation(
            grav_df=grav_df,
            **reg_kwargs,  # type: ignore[arg-type]
        )

        # update starting model in kwargs
        kwargs["prism_layer"] = starting_prisms
        # pylint: disable=duplicate-code
        new_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            not in [
                "zref",
                "density_contrast",
            ]
        }
        # pylint: enable=duplicate-code
        # run cross validation
        score, _ = constraints_cv_score(
            grav_df=grav_df,
            constraints_df=constraints_df,
            results_fname=f"{results_fname}_trial_{i}",
            rmse_as_median=rmse_as_median,
            **new_kwargs,
        )
        scores.append(score)

    # print parameter and score pairs
    for (zref, density_contrast), score in zip(parameter_pairs, scores):
        log.info(
            "Reference level: %s, Density contrast: %s -> Score: %s",
            zref,
            density_contrast,
            score,
        )

    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_zref = parameter_pairs[best_idx][0]
    best_density = parameter_pairs[best_idx][1]
    log.info(
        "Best score of %s with reference level=%s and density contrast=%s",
        best_score,
        best_zref,
        best_density,
    )

    # get best inversion result of each set
    with pathlib.Path(f"{results_fname}_trial_{best_idx}.pickle").open("rb") as f:
        inv_results = pickle.load(f)

    # delete other inversion results
    for i in range(len(scores)):
        if i == best_idx:
            pass
        else:
            pathlib.Path(f"{results_fname}_trial_{i}.pickle").unlink(missing_ok=True)

    # put scores and parameter pairs into dict
    results = {
        "scores": scores,
        "zref_values": parameter_pairs[0],
        "density_contrast_values": parameter_pairs[1],
    }

    # remove if exists
    pathlib.Path(results_fname).unlink(missing_ok=True)

    # save scores and parameter pairs to pickle
    with pathlib.Path(f"{results_fname}.pickle").open("wb") as f:
        pickle.dump(results, f)

    if plot_cv:
        if len(zref_values) == 1:
            plotting.plot_cv_scores(
                scores,
                density_contrast_values,  # type: ignore[arg-type]
                param_name="Density contrast (kg/m$^3$)",
                plot_title="Density contrast Cross-validation",
                # logx=True,
                # logy=True,
            )
        elif len(density_contrast_values) == 1:  # type: ignore[arg-type]
            plotting.plot_cv_scores(
                scores,
                zref_values,
                param_name="Reference level (m)",
                plot_title="Reference level Cross-validation",
                # logx=True,
                # logy=True,
            )
        else:
            plotting.plot_2_parameter_cv_scores(
                scores,
                parameter_pairs,
                param_names=("Reference level (m)", "Density contrast (kg/m$^3$)"),
                # logx=True,
                # logy=True,
            )

    return inv_results, best_zref, best_density, best_score, parameter_pairs, scores


# pylint: enable=duplicate-code


def random_split_test_train(
    data_df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 10,
    plot: bool = False,
) -> pd.DataFrame:
    """
    split data into training and testing sets randomly with a specified percentage of
    points to be in the test set set by test_size.

    Parameters
    ----------
    data_df : pandas.DataFrame
        data to be split, must have columns "easting" and "northing".
    test_size : float, optional
        decimal percentage of points to put in the testing set, by default 0.3
    random_state : int, optional
        number to set th random splitting, by default 10
    plot : bool, optional
        choose to plot the results, by default False

    Returns
    -------
    pandas.DataFrame
        dataframe with a new column "test" which is a boolean value of whether the row
        is in the training or testing set.
    """
    data_df = data_df.copy()

    train, test = vd.train_test_split(
        coordinates=(data_df.easting, data_df.northing),
        data=data_df.index,
        test_size=test_size,
        random_state=random_state,
    )
    train_df = pd.DataFrame(
        data={
            "easting": train[0][0],
            "northing": train[0][1],
            "index": train[1][0],
            "test": False,
        }
    )
    test_df = pd.DataFrame(
        data={
            "easting": test[0][0],
            "northing": test[0][1],
            "index": test[1][0],
            "test": True,
        }
    )
    random_split_df = pd.concat([train_df, test_df])

    random_split_df = pd.merge(data_df, random_split_df, on=["easting", "northing"])  # noqa: PD015

    random_split_df = random_split_df.drop(columns=["index"])

    if plot is True:
        df_train = random_split_df[random_split_df.test == False]  # noqa: E712 # pylint: disable=singleton-comparison
        df_test = random_split_df[random_split_df.test == True]  # noqa: E712 # pylint: disable=singleton-comparison

        region = vd.get_region((random_split_df.easting, random_split_df.northing))
        plot_region = vd.pad_region(region, (region[1] - region[0]) / 10)

        fig = maps.basemap(
            region=plot_region,
            title="Random split",
        )
        fig.plot(
            x=df_train.easting,
            y=df_train.northing,
            style="c.3c",
            fill="blue",
            label="Train",
        )
        fig.plot(
            x=df_test.easting,
            y=df_test.northing,
            style="t.5c",
            fill="red",
            label="Test",
        )
        maps.add_box(fig, box=region)
        fig.legend()
        fig.show()

    return random_split_df


def split_test_train(
    data_df: pd.DataFrame,
    method: str,
    spacing: float | tuple[float, float] | None = None,
    shape: tuple[float, float] | None = None,
    n_splits: int = 5,
    random_state: int = 10,
    plot: bool = False,
) -> pd.DataFrame:
    """
    Split data into training or testing sets either using KFold, BlockedKFold or
    LeaveOneOut methods.

    Parameters
    ----------
    data_df : pandas.DataFrame
        dataframe with coordinate columns "easting" and "northing"
    method : str
        choose between "LeaveOneOut" or "KFold" methods.
    spacing : float | tuple[float, float] | None, optional
        grid spacing to use for Block K-Folds, by default None
    shape : tuple[float, float] | None, optional
        number of blocks to use for Block K-Folds, by default None
    n_splits : int, optional
        number for folds to make for K-Folds method, by default 5
    random_state : int, optional
        random state used for both methods, by default 10
    plot : bool, optional
        plot the separated training and testing dataset, by default False

    Returns
    -------
    pandas.DataFrame
        a dataset with a new column for each fold in the form fold_0, fold_1 etc., with
        the value "train" or "test"
    """
    df = data_df.copy()

    if method == "LeaveOneOut":
        kfold = sklearn.model_selection.LeaveOneOut()
    elif method == "KFold":
        if spacing or shape is None:
            kfold = sklearn.model_selection.KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
        else:
            kfold = vd.BlockKFold(
                spacing=spacing,
                shape=shape,
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
            # kfold = vd.BlockShuffleSplit(
            #     spacing=spacing,
            #     shape=shape,
            #     n_splits=n_splits,
            #     test_size=test_size,
            #     random_state = random_state,
            # )
    else:
        msg = "invalid string for `method`"
        raise ValueError(msg)

    coords = (df.easting, df.northing)
    feature_matrix = np.transpose(coords)
    coord_shape = coords[0].shape
    mask = np.full(shape=coord_shape, fill_value="     ")

    for iteration, (train, test) in enumerate(kfold.split(feature_matrix)):
        mask[np.unravel_index(train, coord_shape)] = "train"
        mask[np.unravel_index(test, coord_shape)] = "_test"
        df = pd.concat(
            [df, pd.DataFrame({f"fold_{iteration}": mask}, index=df.index)], axis=1
        )

    df = df.replace("_test", "test")
    if plot is True:
        folds = list(df.columns[df.columns.str.startswith("fold_")])
        _, ncols = polar_utils.square_subplots(len(folds))
        df = df.copy()
        for i in range(len(folds)):
            if i == 0:
                fig = (None,)
                origin_shift = "initialize"
                xshift_amount = None
                yshift_amount = None
            elif i % ncols == 0:
                # fig = fig
                origin_shift = "both_shift"
                xshift_amount = -ncols + 1
                yshift_amount = -1
            else:
                # fig= fig
                origin_shift = "xshift"
                xshift_amount = 1
                yshift_amount = 1

            df_test = df[df[f"fold_{i}"] == "test"]
            df_train = df[df[f"fold_{i}"] == "train"]

            region = vd.get_region((df.easting, df.northing))
            plot_region = vd.pad_region(region, (region[1] - region[0]) / 10)
            fig = maps.basemap(
                region=plot_region,
                title=f"Fold {i} ({len(df_test)} testing points)",
                origin_shift=origin_shift,
                xshift_amount=xshift_amount,
                yshift_amount=yshift_amount,
                fig=fig,
            )
            maps.add_box(fig, box=region)
            fig.plot(  # type: ignore[attr-defined]
                x=df_train.easting,
                y=df_train.northing,
                style="c.4c",
                fill="blue",
                label="Train",
            )
            fig.plot(  # type: ignore[attr-defined]
                x=df_test.easting,
                y=df_test.northing,
                style="t.7c",
                fill="red",
                label="Test",
            )
            fig.legend()  # type: ignore[attr-defined]
        fig.show()  # type: ignore[attr-defined]

    return df


def kfold_df_to_lists(
    df: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    convert a single dataframe with fold columns in the form fold_0, fold_1 etc. into
    a list of testing dataframes for each fold and a list of training dataframes for
    each fold.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe with fold columns in the form fold_0, fold_1 etc., as output by
        function `split_test_train()`.

    Returns
    -------
    test_dfs : list[pandas.DataFrame]
        a list of testing dataframes for each fold
    train_dfs : list[pandas.DataFrame]
        a list of training dataframes for each fold
    """
    # get list of fold column names
    folds = list(df.columns[df.columns.str.startswith("fold_")])

    test_dfs = []
    train_dfs = []
    # add train and test df for each fold to each own list
    for f in folds:
        # remove other fold column for clarity
        cols_to_remove = [i for i in folds if i != f]
        df1 = df.drop(cols_to_remove, axis=1)
        # append new dfs to lists
        test_dfs.append(df1[df1[f] == "test"])
        train_dfs.append(df1[df1[f] == "train"])
    return test_dfs, train_dfs


def eq_sources_score(
    coordinates: tuple[NDArray, NDArray, NDArray],
    data: pd.Series | NDArray,
    delayed: bool = False,
    weights: NDArray | None = None,
    **kwargs: typing.Any,
) -> float:
    """
    Calculate the cross-validation score for fitting gravity data to equivalent sources.
    Uses Verde's cross_val_score function to calculate the score.
    All kwargs are passed to the harmonica.EquivalentSources class.

    Parameters
    ----------
    coordinates : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        tuple of easting, northing, and upward coordinates of the gravity data
    data : pandas.Series | numpy.ndarray
        the gravity data
    delayed : bool, optional
        compute the scores in parallel if True, by default False
    weights : numpy.ndarray | None, optional
        optional weight values for each gravity data point, by default None

    Keyword Arguments
    -----------------
    damping : float | None
        The positive damping regularization parameter. Controls how much
        smoothness is imposed on the estimated coefficients.
        If None, no regularization is used.
    points : list[numpy.ndarray] | None
        List containing the coordinates of the equivalent point sources.
        Coordinates are assumed to be in the following order:
        (``easting``, ``northing``, ``upward``).
        If None, will place one point source below each observation point at
        a fixed relative depth below the observation point.
        Defaults to None.
    depth : float | str
        Parameter used to control the depth at which the point sources will be
        located.
        If a value is provided, each source is located beneath each data point
        (or block-averaged location) at a depth equal to its elevation minus
        the ``depth`` value.
        If set to ``"default"``, the depth of the sources will be estimated as
        4.5 times the mean distance between first neighboring sources.
        This parameter is ignored if *points* is specified.
        Defaults to ``"default"``.
    block_size: float | tuple[float, float] | None
        Size of the blocks used on block-averaged equivalent sources.
        If a single value is passed, the blocks will have a square shape.
        Alternatively, the dimensions of the blocks in the South-North and
        West-East directions can be specified by passing a tuple.
        If None, no block-averaging is applied.
        This parameter is ignored if *points* are specified.
        Default to None.
    parallel : bool
        If True any predictions and Jacobian building is carried out in
        parallel through Numba's ``jit.prange``, reducing the computation time.
        If False, these tasks will be run on a single CPU. Default to True.
    dtype : str
        The desired data-type for the predictions and the Jacobian matrix.
        Default to ``"float64"``.

    Returns
    -------
    float
        a float of the score, the higher the value to better the fit.
    """

    if np.isnan(coordinates).any():
        msg = "coordinates contain NaN"
        raise ValueError(msg)
    if np.isnan(data).any():
        msg = "data contains is NaN"
        raise ValueError(msg)

    eqs = hm.EquivalentSources(
        **kwargs,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=sklearn.exceptions.UndefinedMetricWarning
        )
        score = np.nan
        n_splits = 5
        while np.isnan(score):
            score = np.mean(
                vd.cross_val_score(
                    eqs,
                    coordinates,
                    data,
                    delayed=delayed,
                    weights=weights,
                    cv=sklearn.model_selection.KFold(
                        n_splits=n_splits,
                        shuffle=True,
                        random_state=0,
                    ),
                )
            )
            if (n_splits == 5) and (np.isnan(score)):
                msg = (
                    "eq sources score is NaN, reduce n_splits (5) by 1 until "
                    "scoring metric is defined"
                )
                log.warning(msg)

            n_splits -= 1

    if np.isnan(score):
        msg = (
            "score is still NaN after reduce n_splits, makes sure you're supplying "
            "enough points for the equivalent sources"
        )
        raise ValueError(msg)

    return score  # type: ignore[no-any-return]


def regional_separation_score(
    testing_df: pd.DataFrame,
    score_as_median: bool = False,
    **kwargs: typing.Any,
) -> tuple[float, float, float | None, pd.DataFrame]:
    """
    Evaluate the effectiveness of the gravity regional-residual separation.
    The optimal regional component is that which results in a residual component which
    is lowest at constraint points, while still contains a high amplitude elsewhere.

    Parameters
    ----------
    testing_df : pandas.DataFrame
        dataframe containing a priori measurements of the topography of interest with
        columns "upward", "easting", and "northing"
    score_as_median : bool, optional
        switch from using the root mean square to the root median square for the score,
        by default is False., by default False
    **kwargs: typing.Any,
        additional keyword arguments for the specified method.

    Returns
    -------
    residual_constraint_score : float
        the RMS of the residual at constraint points
    residual_amplitude_score : float
        the RMS of the residuals amplitude at all grid points
    true_reg_score : float | None
        the RMSE between the true regional field and the estimated field, if provided,
        otherwise None
    df_anomalies : pandas.DataFrame
        the dataframe of the regional and residual gravity anomalies
    """

    # pull out kwargs
    kwargs = copy.deepcopy(kwargs)
    method = kwargs.pop("method")
    grav_df = kwargs.pop("grav_df")
    true_regional = kwargs.pop("true_regional", None)
    remove_starting_grav_mean = kwargs.pop("remove_starting_grav_mean", False)

    if method == "constraints_cv":
        msg = (
            "method `constraints_cv` internally calculated regional separation scores "
            "so it should not be used here."
        )
        raise ValueError(msg)
    df_anomalies = regional.regional_separation(
        method=method,
        grav_df=grav_df,
        remove_starting_grav_mean=remove_starting_grav_mean,
        **kwargs,
    )

    # grid the anomalies
    grid = df_anomalies.set_index(["northing", "easting"]).to_xarray()

    # sample the residual and regional at the constraint points
    df = utils.sample_grids(
        df=testing_df,
        grid=grid.res,
        sampled_name="res",
    )

    residual_constraint_score = utils.rmse(df.res, as_median=score_as_median)
    if np.isnan(residual_constraint_score):
        msg = "residual_constraint_score is NaN"
        raise ValueError(msg)
    residual_amplitude_score = utils.rmse(grid.res, as_median=score_as_median)
    if np.isnan(residual_amplitude_score):
        msg = "residual_amplitude_score is NaN"
        raise ValueError(msg)

    if true_regional is not None:
        true_reg_score = utils.rmse(
            np.abs(true_regional - grid.reg), as_median=score_as_median
        )
    else:
        true_reg_score = None

    # misfit = reg + res
    # want residual at constraint to be low
    # don't want this accomplished by having reg = misfit everywhere

    # optimize on
    # 1) low residual at constraints
    # 2) high residual everywhere else

    return (
        residual_constraint_score,
        residual_amplitude_score,
        true_reg_score,
        df_anomalies,
    )
