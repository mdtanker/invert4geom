import copy  # pylint: disable=too-many-lines
import typing
import warnings

import harmonica as hm
import numpy as np
import pandas as pd
import sklearn
import verde as vd
import xarray as xr
from numpy.typing import NDArray
from polartoolkit import maps
from polartoolkit import utils as polar_utils

from invert4geom import logger, utils

# pylint: enable=duplicate-code


def remove_test_points(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove the test rows and the column denoting test points

    Parameters
    ----------
    ds : xarray.Dataset
        gravity dataset with a boolean variable "test".

    Returns
    -------
    xarray.Dataset
        gravity dataset with test points removed and no "test" column.
    """
    df = ds.inv.df

    df = df[df.test == False].copy()  # noqa: E712 # pylint: disable=singleton-comparison
    df = df.drop(columns=["test"])

    ds_new = df.set_index(list(ds.dims)).to_xarray()

    # retrain attributes
    ds_new.attrs.update(ds.attrs)  # pylint: disable=protected-access

    return ds_new


def add_test_points(
    ds: xr.Dataset,
) -> xr.Dataset:
    """
    take a dataframe of coordinates and make all rows that fall on the data_spacing
    grid training points. Add rows at each point which falls on the grid points of
    half the data_spacing, assign these with label "test". If other data is present
    in dataframe, will sample at each new location.

    Parameters
    ----------
    ds : xarray.Dataset
        gravity dataset to be resampled.

    Returns
    -------
    xarray.Dataset
        gravity dataset with a boolean variable "test" denoting whether the row is
        a training or testing point.
    """
    df = ds.inv.df

    if "test" in df.columns:
        msg = "gravity dataframe already contains a 'test' column, not resampling. If you'd like to remove the test points, use `remove_test_points()."
        raise ValueError(msg)

    coord_names = list(ds.dims)

    # create coords for full data at half spacing
    coords = vd.grid_coordinates(
        region=ds.region,
        spacing=ds.spacing / 2,
        pixel_register=False,
    )

    # turn coordinates into dataarray
    full_points = vd.make_xarray_grid(
        (coords[0], coords[1]),
        data=np.ones_like(coords[0]),
        data_names="tmp",
        dims=coord_names,
    )
    # turn dataarray in dataframe
    full_df: pd.DataFrame = vd.grid_to_table(full_points).drop(columns="tmp")
    # set all points to test
    full_df["test"] = True  # pylint: disable=unsupported-assignment-operation

    # subset training points, every other value
    train_df = full_df[  # pylint: disable=unsubscriptable-object
        (full_df[coord_names[0]].isin(full_points[coord_names[0]].to_numpy()[::2]))  # pylint: disable=unsubscriptable-object
        & (full_df[coord_names[1]].isin(full_points[coord_names[1]].to_numpy()[::2]))  # pylint: disable=unsubscriptable-object
    ].copy()
    # set training points to not be test points
    train_df["test"] = False

    # merge training and testing dfs
    df = full_df.set_index(coord_names)
    df.update(train_df.set_index(coord_names))
    df2 = df.reset_index()

    df2["test"] = df2.test.astype(bool)

    # sample any other columns in original df at new locations
    for i in list(ds):
        if i == "test" or ds[i].dtype == object:
            pass
        else:
            if not bool(ds[i].coords):
                msg = ("Issue with dataset variable '%s'.", i)  # type: ignore[assignment]
                raise ValueError(msg)
            try:
                df2[i] = utils.sample_grids(
                    df2,
                    ds[i],
                    i,
                )[i].astype(ds[i].dtype)
            except pd.errors.IntCastingNaNError as e:
                logger.error(e)
                df2[i] = utils.sample_grids(
                    df2,
                    ds[i],
                    i,
                )[i]

    # retain original data types
    dtypes = {k: v for k, v in ds.inv.df.dtypes.items() if k in ds.inv.df}
    df2 = df2.astype(dtypes)

    # test with this, using same input spacing as original
    # pd.testing.assert_frame_equal(df2, full_res_grav, check_like=True,)
    ds_new = df2.set_index(coord_names).to_xarray()

    # retain attributes
    ds_new.attrs.update(ds.attrs)  # pylint: disable=protected-access

    return ds_new


def resample_with_test_points(
    data_spacing: typing.Any,  # noqa: ARG001
    data: typing.Any,  # noqa: ARG001
    region: typing.Any,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use function `add_test_points` instead
    """
    # pylint: disable=W0613
    msg = "Function `resample_with_test_points` deprecated, use function `add_test_points` instead"
    raise DeprecationWarning(msg)


def grav_cv_score(
    training_data: pd.DataFrame,  # noqa: ARG001
    testing_data: pd.DataFrame,  # noqa: ARG001
    progressbar: bool = True,  # noqa: ARG001
    rmse_as_median: bool = False,  # noqa: ARG001
    plot: bool = False,  # noqa: ARG001
    **kwargs: typing.Any,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `Inversion` class method `grav_cv_score` instead
    """
    # pylint: disable=W0613
    msg = (
        "Function `grav_cv_score` deprecated, use the `Inversion` class method "
        "`grav_cv_score` instead"
    )
    raise DeprecationWarning(msg)


def constraints_cv_score(
    grav_df: pd.DataFrame,  # noqa: ARG001
    constraints_df: pd.DataFrame,  # noqa: ARG001
    rmse_as_median: bool = False,  # noqa: ARG001
    **kwargs: typing.Any,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `Inversion` class method `constraints_cv_score` instead
    """
    # pylint: disable=W0613
    msg = (
        "Function `constraints_cv_score` deprecated, use the `Inversion` class method "
        "`constraints_cv_score` instead"
    )
    raise DeprecationWarning(msg)


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
        try:
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
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("plotting failed with error: %s", e)

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
    Split data into training or testing sets either using KFold (optional blocked) or
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
        if n_splits > len(df):
            msg = (
                "n_splits must be less than or equal to the number of data points, "
                "decreasing n_splits"
            )
            logger.warning(msg)
            n_splits = len(df)

        if n_splits == 1:
            msg = "n_splits must be greater than 1"
            raise ValueError(msg)

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
        try:
            folds = list(df.columns[df.columns.str.startswith("fold_")])
            _, ncols = polar_utils.square_subplots(len(folds))
            df = df.copy()
            for i in range(len(folds)):
                if i == 0:
                    fig = None
                    origin_shift = "initialize"
                    xshift_amount = None
                    yshift_amount = None
                elif i % ncols == 0:
                    origin_shift = "both"
                    xshift_amount = -ncols + 1
                    yshift_amount = -1
                else:
                    origin_shift = "x"
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
                fig.plot(
                    x=df_train.easting,
                    y=df_train.northing,
                    style="c.4c",
                    fill="blue",
                    label="Train",
                )
                fig.plot(
                    x=df_test.easting,
                    y=df_test.northing,
                    style="t.7c",
                    fill="red",
                    label="Test",
                )
                fig.legend()
            fig.show()  # type: ignore[union-attr]
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("plotting failed with error: %s", e)

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
            try:
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
                        scoring="r2",
                    )
                )
            except ValueError:
                score = np.nan
            if (n_splits == 5) and (np.isnan(score)):
                msg = (
                    "eq sources score is NaN, reducing n_splits (5) by 1 until "
                    "scoring metric is defined"
                )
                logger.warning(msg)

            n_splits -= 1
            if n_splits == 0:
                break

    if np.isnan(score):
        msg = (
            "score is still NaN after reducing n_splits, makes sure you're supplying "
            "enough points for the equivalent sources"
        )
        raise ValueError(msg)

    return score  # type: ignore[no-any-return]


def regional_separation_score(
    grav_ds: xr.Dataset,
    testing_df: pd.DataFrame,
    score_as_median: bool = False,
    **kwargs: typing.Any,
) -> tuple[float, float, float | None, xr.Dataset]:
    """
    Evaluate the effectiveness of the gravity regional-residual separation.
    The optimal regional component is that which results in a residual component which
    is lowest at constraint points, while still contains a high amplitude elsewhere.

    Parameters
    ----------
    grav_ds : xarray.Dataset
        gravity dataset with variables "gravity_anomaly" and "forward_gravity".
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
    ds_anomalies : xarray.Dataset
        the dataframe of the regional and residual gravity anomalies
    """

    # pull out kwargs
    kwargs = copy.deepcopy(kwargs)
    method = kwargs.pop("method")
    true_regional = kwargs.pop("true_regional", None)

    if method == "constraints_cv":
        msg = (
            "method `constraints_cv` internally calculated regional separation scores "
            "so it should not be used here."
        )
        raise ValueError(msg)

    # estimate the regional field
    grav_ds.inv.regional_separation(
        method=method,
        **kwargs,
    )

    # sample the residual at the constraint points
    df = utils.sample_grids(
        df=testing_df,
        grid=grav_ds.res,
        sampled_name="res",
    )

    # calculate scores
    residual_constraint_score = utils.rmse(df.res, as_median=score_as_median)
    if np.isnan(residual_constraint_score):
        msg = "residual_constraint_score is NaN"
        raise ValueError(msg)
    residual_amplitude_score = utils.rmse(grav_ds.res, as_median=score_as_median)
    if np.isnan(residual_amplitude_score):
        msg = "residual_amplitude_score is NaN"
        raise ValueError(msg)

    if true_regional is not None:
        true_reg_score = utils.rmse(
            np.abs(true_regional - grav_ds.reg), as_median=score_as_median
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
        grav_ds,
    )
