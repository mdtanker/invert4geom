from __future__ import annotations

import itertools
import logging
import pathlib
import pickle
import random
import typing

import numpy as np
import pandas as pd
import verde as vd
import xarray as xr
from polartoolkit import utils as polar_utils
from tqdm.autonotebook import tqdm

from invert4geom import inversion, plotting, regional, utils


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
    progressbar: bool = True,
    rmse_as_median: bool = False,
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
    rmse_as_median : bool, optional
        calculate the RMSE as the median as opposed to the mean, by default False
    progressbar : bool, optional
        choose to show the progress bar for the forward gravity calculation, by default
        True
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

    # extract density contrast and zref from prism layer
    prism_layer: xr.Dataset = kwargs.get("prism_layer")
    density_contrast = np.fabs(prism_layer.density)
    zref = prism_layer.attrs.get("zref")

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
    observed = test[kwargs.get("grav_data_column")] - test.reg
    predicted = test.test_point_grav

    dif = predicted - observed

    score = utils.rmse(dif, as_median=rmse_as_median)

    if plot:
        test_grid = test.set_index(["northing", "easting"]).to_xarray()
        obs = test_grid[kwargs.get("grav_data_column")] - test_grid.reg
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
    rmse_as_median: bool = False,
    progressbar: bool = True,
    plot_grids: bool = False,
    plot_cv: bool = False,
    verbose: bool = False,
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
    training_data : pd.DataFrame
        just the training data rows
    testing_data : pd.DataFrame
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
    verbose : bool, optional
       log the results, by default False
    results_fname : str, optional
        file name to save results to, by default "tmp" with an attached random number

    Returns
    -------
    tuple[ tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float],
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
    pbar = tqdm(
        param_values,
        desc=f"{param_name} values",
        disable=not progressbar,
    )
    for i, value in enumerate(pbar):
        # update parameter value in kwargs
        kwargs[param_name] = value
        # run cross validation
        score = grav_cv_score(
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
            logging.warning(msg)
    if verbose:
        # set Python's logging level to get information about the inversion\s progress
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    # print value and score pairs
    for value, score in zip(param_values, scores):
        logging.info("%s value: %s -> Score: %s", param_name, value, score)

    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_param_value = param_values[best_idx]
    logging.info(
        "Best score of %s with %s=%s", best_score, param_name, best_param_value
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

    # put scores and parameter values into dict
    results = {
        "scores": scores,
        "param_values": param_values,
    }

    if best_param_value in [np.min(param_values), np.max(param_values)]:
        logging.warning(
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
) -> float:
    """
    Find the score, represented by the root mean squared error (RMSE), between the
    constraint point elevation, and the inverted topography at the constraint points.
    Follows methods of :footcite:t:`uiedafast2017`.

    Parameters
    ----------
    grav_df : pd.DataFrame
       gravity dataframe with columns "res", "reg", and column set by kwarg
       grav_data_column
    constraints_df : pd.DataFrame
        constraints dataframe with columns "easting", "northing", and "upward"
    rmse_as_median : bool, optional
        calculate the RMSE as the median of the , as opposed to the mean, by default
        False
    Returns
    -------
    float
        a score, represented by the root mean squared error, between the testing gravity
        data and the predicted gravity data.

    References
    ----------
    .. footbibliography::
    """

    constraints_df = constraints_df.copy()

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

    return utils.rmse(dif, as_median=rmse_as_median)


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
    verbose: bool = False,
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
    grav_df : pd.DataFrame
        dataframe with gravity data and coordinates, must have coordinate columns
        "easting", "northing", and "upward", and gravity data column defined by kwarg
        "grav_data_column".
    constraints_df : pd.DataFrame
        dataframe with points where the topography of interest has been previously
        measured, must have coordinate columns "easting", "northing", and "upward".
    starting_topography : xr.DataArray,optional
        starting topography to use to create the starting prism model. If not supplied,
        starting_topography_kwargs must be provided to create the starting topography.
        By default None.
    zref_values : list[float] | None, optional
        Reference level values to test, by default None
    density_contrast_values : list[float] | None, optional
        Density contrast values to test, by default None
    starting_topography_kwargs : dict[str, typing.Any] | None, optional
        Keywords used to create the starting topography, by default None
    regional_grav_kwargs : dict[str, typing.Any] | None, optional
        Keywords used to calculate the regional field, by default None
    rmse_as_median : bool, optional
        Use the median instead of the root mean square as the scoring metric, by default
        False
    progressbar : bool, optional
        display a progress bar for the number of tested values, by default True
    plot_cv : bool, optional
        plot a graph of scores vs parameter values, by default False
    verbose : bool, optional
        log the results, by default False
    results_fname : str, optional
        file name to save results to, by default "tmp" with an attached random number

    Returns
    -------
    tuple[ tuple[pd.DataFrame, pd.DataFrame, dict[str, typing.Any], float],
        float, float, float, list[typing.Any], list[float], ]
        the inversion results, the optimal parameter value, the score associated with
        it, the parameter values and the scores for each parameter value
    """

    if verbose:
        # set Python's logging level to get information about the inversion\s progress
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)

    # set file name for saving results with random number between 0 and 999
    if results_fname is None:
        results_fname = f"tmp_{random.randint(0,999)}"

    if constraints_df is None:
        msg = "must provide constraints_df"
        raise ValueError(msg)

    if (zref_values is None) & (density_contrast_values is None):
        msg = "must provide either or both zref_values and density_contrast_values"
        raise ValueError(msg)

    if zref_values is None:
        zref = kwargs.get("zref", None)
        if zref is None:
            msg = "must provide zref_values or zref in kwargs"
            raise ValueError(msg)
        zref_values = [zref]
    elif density_contrast_values is None:
        if starting_topography is None:
            if starting_topography_kwargs is None:
                msg = (
                    "starting_topography_kwargs must be provided if "
                    "starting_topography is not provided"
                )
                raise ValueError(msg)
            density_contrast = starting_topography_kwargs.get("density_contrast", None)
            if density_contrast is None:
                msg = (
                    "density_contrast must be provided to starting_topography_kwargs if"
                    " starting_topography is not provided"
                )
                raise ValueError(msg)
        else:
            density_contrast = starting_topography.density
        if density_contrast is None:
            msg = "must provide density_contrast_values or density_contrast in kwargs"
            raise ValueError(msg)
        density_contrast_values = [density_contrast]

    # raise warning about using constraint point minimization for regional estimation
    if (regional_grav_kwargs is not None) and (
        regional_grav_kwargs.get("regional_method") == "constraints"
    ):
        msg = (
            "Using constraint points for regional field estimation. This is not "
            "recommended as the constraint points are used for the cross-validation, "
            "making the scoring metric meaningless. Consider using a different method "
            "for regional field estimation."
        )
        logging.warning(msg)

    # create all possible combinations of zref and density contrast
    parameter_pairs = list(itertools.product(zref_values, density_contrast_values))  # type: ignore[arg-type]

    if "test" in grav_df.columns:
        assert (
            grav_df.test.any()
        ), "test column contains True value, not needed except for during damping CV"

    # run inversions and collect scores
    scores = []
    pbar = tqdm(
        parameter_pairs,
        desc="Zref/Density pairs",
        disable=not progressbar,
    )

    for i, (zref, density_contrast) in enumerate(pbar):
        # create starting topography
        if starting_topography is None:
            if starting_topography_kwargs is None:
                msg = (
                    "starting_topography_kwargs must be provided if "
                    "starting_topography is not provided"
                )
                raise ValueError(msg)
            method = starting_topography_kwargs.get("method")
            upwards = starting_topography_kwargs.get("upwards", None)
            if (method == "flat") & (upwards is None):
                upwards = zref

            created_starting_topography = utils.create_topography(
                method=method,  # type: ignore [arg-type]
                region=starting_topography_kwargs.get("region", None),
                spacing=starting_topography_kwargs.get("spacing", None),
                upwards=upwards,
                constraints_df=constraints_df,
                dampings=starting_topography_kwargs.get(
                    "dampings", np.logspace(-10, 0, 100)
                ),
            )
        else:
            created_starting_topography = starting_topography.copy()

        # re-calculate density grid with new density contrast
        density_grid = xr.where(
            created_starting_topography >= zref, density_contrast, -density_contrast
        )

        # create layer of prisms
        starting_prisms = utils.grids_to_prisms(
            created_starting_topography,
            reference=zref,
            density=density_grid,
        )

        # calculate forward gravity of starting prism layer
        grav_df["starting_grav"] = starting_prisms.prism_layer.gravity(
            coordinates=(
                grav_df.easting,
                grav_df.northing,
                grav_df.upward,
            ),
            field="g_z",
            progressbar=False,
        )

        # calculate misfit as observed - starting
        grav_data_column = kwargs.get("grav_data_column")
        grav_df["misfit"] = grav_df[grav_data_column] - grav_df.starting_grav

        # calculate regional field
        reg_kwargs = regional_grav_kwargs.copy()  # type: ignore[union-attr]

        grav_df = regional.regional_separation(
            method=reg_kwargs.pop("regional_method", None),
            grav_df=grav_df,
            regional_column="reg",
            grav_data_column="misfit",
            **reg_kwargs,
        )

        # remove the regional from the misfit to get the residual
        grav_df["res"] = grav_df.misfit - grav_df.reg

        # update starting model in kwargs
        kwargs["prism_layer"] = starting_prisms

        # run cross validation
        score = constraints_cv_score(
            grav_df=grav_df,
            constraints_df=constraints_df,
            results_fname=f"{results_fname}_trial_{i}",
            rmse_as_median=rmse_as_median,
            **kwargs,
        )
        scores.append(score)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # print parameter and score pairs
    for (zref, density_contrast), score in zip(parameter_pairs, scores):
        logging.info(
            "Reference level: %s, Density contrast: %s -> Score: %s",
            zref,
            density_contrast,
            score,
        )

    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_zref = parameter_pairs[best_idx][0]
    best_density = parameter_pairs[best_idx][1]
    logging.info(
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


def eq_sources_score(
    coordinates: tuple[pd.Series | NDArray, pd.Series | NDArray, pd.Series | NDArray],
    data: pd.Series | NDArray,
    damping: float | None = None,
    depth: str | float | None = "default",
    block_size: float | None = None,
    points: NDArray | None = None,
    delayed: bool = False,
    weights: NDArray | None = None,
) -> float:
    """
    Calculate the cross-validation score for fitting gravity data to equivalent sources.
    Uses Verde's cross_val_score function to calculate the score.

    Parameters
    ----------
    coordinates : tuple[pd.Series | NDArray, pd.Series | NDArray, pd.Series | NDArray]
        tuple of easting, northing, and upward coordinates of the gravity data
    data : pd.Series | NDArray
        the gravity data
    damping : float | None, optional
        damping parameter to use in the fitting, by default None
    depth : str | float | None, optional
        depth of the sources, positive downward in meters, by default "default"
    block_size : float | None, optional
        block size in meters to reduce the gravity data by, by default None
    points : NDArray | None, optional
        use to specify point locations, by default None
    delayed : bool, optional
        compute the scores in parallel if True, by default False
    weights : NDArray | None, optional
        optional weight values for each gravity data point, by default None

    Returns
    -------
    float
        a float of the score, the higher the value to better the fit.
    """
    eqs = hm.EquivalentSources(
        damping=damping,
        depth=depth,
        block_size=block_size,
        points=points,
    )

    return float(
        np.mean(
            vd.cross_val_score(
                eqs,
                coordinates,
                data,
                delayed=delayed,
                weights=weights,
            )
        )
    )
