import copy  # pylint: disable=too-many-lines
import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly
import plotly.io as pio
import pyvista
import scipy as sp
import seaborn as sns
import verde as vd
import xarray as xr
from numpy.typing import NDArray
from polartoolkit import maps, profiles
from polartoolkit import utils as polar_utils

from invert4geom import logger, utils

# This ensures Plotly output works in multiple places:
# plotly_mimetype: VS Code notebook UI
# notebook: "Jupyter: Export to HTML" command in VS Code
# See https://plotly.com/python/renderers/#multiple-renderers
pio.renderers.default = "plotly_mimetype+notebook"


def plot_2_parameter_cv_scores(
    scores: list[float],  # noqa: ARG001
    parameter_pairs: list[tuple[float, float]],  # noqa: ARG001
    param_names: tuple[str, str] = ("Hyperparameter 1", "Hyperparameter 2"),  # noqa: ARG001
    figsize: tuple[float, float] = (5, 3.5),  # noqa: ARG001
    cmap: str | None = None,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `plot_2_parameter_scores` function instead
    """
    # pylint: disable=W0613
    msg = "Function `plot_2_parameter_cv_scores` renamed to `plot_2_parameter_scores`"
    raise DeprecationWarning(msg)


def plot_2_parameter_scores(
    scores: list[float],
    parameter_pairs: list[tuple[float, float]],
    param_names: tuple[str, str] = ("Hyperparameter 1", "Hyperparameter 2"),
    figsize: tuple[float, float] = (5, 3.5),
    cmap: str | None = None,
) -> None:
    """
    plot a scatter plot graph with x axis equal to parameter 1, y axis equal to
    parameter 2, and points colored by cross-validation scores.

    Parameters
    ----------
    scores : list[float]
        score values
    parameter_pairs : list[float]
        parameter values
    param_names : tuple[str, str], optional
        name to give for the parameters, by default "Hyperparameter"
    figsize : tuple[float, float], optional
        size of the figure, by default (5, 3.5)
    cmap : str, optional
        matplotlib colormap for scores, by default "viridis"
    """
    sns.set_theme()

    if cmap is None:
        cmap = sns.color_palette("mako", as_cmap=True)

    df = pd.DataFrame(
        {
            "scores": scores,
            param_names[0]: [
                parameter_pairs[i][0] for i in list(range(len(parameter_pairs)))
            ],
            param_names[1]: [
                parameter_pairs[i][1] for i in list(range(len(parameter_pairs)))
            ],
        }
    )
    df = df.sort_values(by="scores")

    best = df.iloc[0]

    plt.figure(figsize=figsize)
    plt.title("Two parameter cross-validation")

    grid = df.set_index([param_names[1], param_names[0]]).to_xarray().scores
    grid.plot(
        cmap=cmap,
        # norm=plt.Normalize(df.scores.min(), df.scores.max()),
    )
    # plt.contourf(
    #     df[param_names[0]],
    #     df[param_names[1]],
    #     Z = grid,
    #     cmap = cmap,
    # )
    plt.scatter(
        df[param_names[0]],  # pylint: disable=unsubscriptable-object
        df[param_names[1]],  # pylint: disable=unsubscriptable-object
        # c = df.scores,
        # cmap = cmap,
        # marker="o",
        marker=".",
        color="gray",
    )
    plt.plot(
        best[param_names[0]],
        best[param_names[1]],
        "s",
        markersize=10,
        color=sns.color_palette()[3],
        label="Minimum",
    )
    plt.legend(
        loc="upper right",
    )

    plt.xlabel(param_names[0])
    plt.ylabel(param_names[1])
    # plt.colorbar()
    plt.tight_layout()


def plot_2_parameter_cv_scores_uneven(
    study: optuna.study.Study,  # noqa: ARG001
    param_names: tuple[str, str],  # noqa: ARG001
    plot_param_names: tuple[str, str] = ("Hyperparameter 1", "Hyperparameter 2"),  # noqa: ARG001
    figsize: tuple[float, float] = (5, 3.5),  # noqa: ARG001
    cmap: str | None = None,  # noqa: ARG001
    best: str = "min",  # noqa: ARG001
    logx: bool = False,  # noqa: ARG001
    logy: bool = False,  # noqa: ARG001
    robust: bool = False,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `plot_2_parameter_scores_uneven` function instead
    """
    # pylint: disable=W0613
    msg = "Function `plot_2_parameter_cv_scores_uneven` renamed to `plot_2_parameter_scores_uneven`"
    raise DeprecationWarning(msg)


def plot_2_parameter_scores_uneven(
    study: optuna.study.Study,
    param_names: tuple[str, str],
    plot_param_names: tuple[str, str] = ("Hyperparameter 1", "Hyperparameter 2"),
    figsize: tuple[float, float] = (5, 3.5),
    cmap: str | None = None,
    best: str = "min",
    logx: bool = False,
    logy: bool = False,
    robust: bool = False,
) -> None:
    """
    plot a scatter plot graph with x axis equal to parameter 1, y axis equal to
    parameter 2, and points colored by cross-validation scores.

    Parameters
    ----------
    study : optuna.study.Study
    param_names : tuple[str, str], optional
        name to give for the parameters, by default "Hyperparameter"
    figsize : tuple[float, float], optional
        size of the figure, by default (5, 3.5)
    cmap : str, optional
        matplotlib colormap for scores, by default "viridis"
    best : str, optional
        whether the 'min' or 'max' score is considered best, by default 'min'
    logx, logy : bool, optional
        make the x or y axes log scale, by default False
    robust: bool, optional
        use robust color limits
    """

    sns.set_theme()

    if cmap is None:
        cmap = sns.color_palette("mako", as_cmap=True)

    df = study.trials_dataframe()
    df = df[[param_names[0], param_names[1], "value"]]

    if best == "min":
        best_ind = df.value.idxmin()
        label = "Minimum"
    elif best == "max":
        best_ind = df.value.idxmax()
        label = "Maximum"
    else:
        msg = "best must be either 'min' or 'max'"
        raise ValueError(msg)

    plt.figure(figsize=figsize)
    plt.title("Two parameter cross-validation")

    x = df[param_names[0]].values  # noqa: PD011
    y = df[param_names[1]].values  # noqa: PD011
    z = df.value.to_numpy()

    x_buffer = (max(x) - min(x)) / 50
    y_buffer = (max(y) - min(y)) / 50

    # 2D grid for interpolation
    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))
    xi, yi = np.meshgrid(xi, yi)

    try:
        interp = sp.interpolate.CloughTocher2DInterpolator(
            list(zip(x, y, strict=False)), z
        )
    except ValueError as e:
        logger.error(
            "Error interpolating value in plot_2_parameter_scores_uneven: %s", e
        )
        return
    zi = interp(xi, yi)

    vmin, vmax = polar_utils.get_min_max(
        df.value,
        robust=robust,
        # robust_percentiles=(.89,.9)
    )
    plt.pcolormesh(
        xi,
        yi,
        zi,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        # shading='auto',
    )
    # plt.contourf(
    #     xi,
    #     yi,
    #     zi,
    #     30,
    #     cmap=cmap,
    #     vmin=vmin,
    #     vmax=vmax,
    # )
    plt.colorbar().set_label("Scores")

    plt.plot(
        df.iloc[best_ind][param_names[0]],
        df.iloc[best_ind][param_names[1]],
        "s",
        markersize=10,
        color=sns.color_palette()[3],
        label=label,
    )
    plt.scatter(
        x,
        y,
        marker=".",
        color="lightgray",
        edgecolor="black",
    )
    plt.legend(
        loc="best",
    )
    plt.xlim([min(x) - x_buffer, max(x) + x_buffer])
    plt.ylim([min(y) - y_buffer, max(y) + y_buffer])
    plt.xlabel(plot_param_names[0])
    plt.ylabel(plot_param_names[1])
    plt.xticks(rotation=20)

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    plt.tight_layout()


def plot_cv_scores(
    scores: list[float],  # noqa: ARG001
    parameters: list[float],  # noqa: ARG001
    logx: bool = False,  # noqa: ARG001
    logy: bool = False,  # noqa: ARG001
    param_name: str = "Hyperparameter",  # noqa: ARG001
    figsize: tuple[float, float] = (5, 3.5),  # noqa: ARG001
    plot_title: str | None = None,  # noqa: ARG001
    fname: str | None = None,  # noqa: ARG001
    best: str = "min",  # noqa: ARG001
) -> typing.Any:
    """
    DEPRECATED: use the `plot_scores` function instead
    """
    # pylint: disable=W0613
    msg = "Function `plot_cv_scores` renamed to `plot_scores`"
    raise DeprecationWarning(msg)


def plot_scores(
    scores: list[float],
    parameters: list[float],
    logx: bool = False,
    logy: bool = False,
    param_name: str = "Hyperparameter",
    figsize: tuple[float, float] = (5, 3.5),
    plot_title: str | None = None,
    fname: str | None = None,
    best: str = "min",
) -> typing.Any:
    """
    plot a graph of cross-validation scores vs hyperparameter values

    Parameters
    ----------
    scores : list[float]
        score values
    parameters : list[float]
        parameter values
    logx, logy : bool, optional
        make the x or y axes log scale, by default False
    param_name : str, optional
        name to give for the parameters, by default "Hyperparameter"
    figsize : tuple[float, float], optional
        size of the figure, by default (5, 3.5)
    plot_title : str | None, optional
        title of figure, by default None
    fname : str | None, optional
        filename to save figure, by default None
    best : str, optional
        which value to plot as the best, 'min' or 'max', by default "min"
    Returns
    -------
    a matplotlib figure instance
    """

    sns.set_theme()

    df0 = pd.DataFrame({"scores": scores, "parameters": parameters})
    df = df0.sort_values(by="parameters")

    if best == "min":
        best_score = df.scores.argmin()
        label = "Minimum"
    elif best == "max":
        best_score = df.scores.argmax()
        label = "Maximum"
    else:
        msg = f"best must be 'min' or 'max', not {best}"
        raise ValueError(msg)

    fig, ax = plt.subplots(figsize=figsize)
    if plot_title is not None:
        ax.set_title(plot_title)
    else:
        ax.set_title(f"{param_name} Cross-validation")
    ax.plot(
        df.parameters.iloc[best_score],
        df.scores.iloc[best_score],
        "s",
        markersize=10,
        color=sns.color_palette()[3],
        label=label,
    )
    ax.plot(df.parameters, df.scores, marker="o")
    ax.scatter(
        df.parameters,
        df.scores,
        s=1,
        marker=".",
        color="black",
        edgecolors="black",
        zorder=10,
    )
    ax.legend(loc="best")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(f"{param_name} value")
    ax.set_ylabel("Root Mean Square Error")

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname)

    return fig


def plot_convergence(
    results: pd.DataFrame,  # noqa: ARG001
    params: dict[str, typing.Any],  # noqa: ARG001
    inversion_region: tuple[float, float, float, float] | None = None,  # noqa: ARG001
    figsize: tuple[float, float] = (5, 3.5),  # noqa: ARG001
    fname: str | None = None,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `Inversion` class method `plot_convergence` instead
    """
    # pylint: disable=W0613
    msg = (
        "Function `plot_convergence` deprecated, use the `Inversion` class method "
        "`plot_convergence` instead"
    )
    raise DeprecationWarning(msg)


def plot_dynamic_convergence(
    l2_norms: list[float],  # noqa: ARG001
    l2_norm_tolerance: float,  # noqa: ARG001
    delta_l2_norms: list[float],  # noqa: ARG001
    delta_l2_norm_tolerance: float,  # noqa: ARG001
    starting_misfit: float,  # noqa: ARG001
    figsize: tuple[float, float] = (5, 3.5),  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `Inversion` class method `plot_dynamic_convergence` instead
    """
    # pylint: disable=W0613
    msg = (
        "Function `plot_dynamic_convergence` deprecated, use the `Inversion` class method "
        "`plot_dynamic_convergence` instead"
    )
    raise DeprecationWarning(msg)


def align_yaxis(
    ax1: mpl.axes.Axes,
    v1: float,
    ax2: mpl.axes.Axes,
    v2: float,
) -> None:
    """
    adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1.
    From https://stackoverflow.com/a/10482477/18686384
    """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + dy, maxy + dy)


def grid_inversion_results(
    misfits: list[str],
    topos: list[str],
    corrections: list[str],
    prisms_ds: xr.Dataset,
    grav_results: pd.DataFrame,
    region: tuple[float, float, float, float],
) -> tuple[list[xr.DataArray], list[xr.DataArray], list[xr.DataArray]]:
    """
    create grids from the various data variables of the supplied gravity dataframe and
    prism dataset

    Parameters
    ----------
    misfits : list[str]
        list of misfit column names in the gravity results dataframe
    topos : list[str]
        list of topography variable names in the prism dataset
    corrections : list[str]
        list of correction variable names in the prism dataset
    prisms_ds : xarray.Dataset
        resulting dataset of prism layer from the inversion
    grav_results : pandas.DataFrame
        resulting dataframe of gravity data from the inversion
    region : tuple[float, float, float, float]
        region to use for gridding in format (xmin, xmax, ymin, ymax)

    Returns
    -------
    misfit_grids : list[xarray.DataArray]
        list of misfit grids
    topo_grids : list[xarray.DataArray]
        list of topography grids
    corrections_grids : list[xarray.DataArray]
        list of correction grids
    """
    misfit_grids = []
    for m in misfits:
        grid = grav_results.set_index(["northing", "easting"]).to_xarray()[m]
        misfit_grids.append(grid)

    topo_grids = []
    for t in topos:
        topo_grids.append(
            prisms_ds[t].sel(
                easting=slice(region[0], region[1]),
                northing=slice(region[2], region[3]),
            )
        )

    corrections_grids = []
    for m in corrections:
        corrections_grids.append(
            prisms_ds[m].sel(
                easting=slice(region[0], region[1]),
                northing=slice(region[2], region[3]),
            )
        )

    return (misfit_grids, topo_grids, corrections_grids)


def plot_inversion_topo_results(
    prisms_ds: xr.Dataset,
    constraints_df: pd.DataFrame | None = None,
    constraint_style: str = "x.3c",
    fig_height: float = 12,
) -> None:
    """
    plot the initial and final topography grids from the inversion and their difference

    Parameters
    ----------
    prisms_ds : xarray.Dataset
        dataset resulting from inversion
    topo_cmap_perc : float, optional
        value to multiple min and max values by for colorscale, by default 1
    region : tuple[float, float, float, float], optional
        clip grids to this region before plotting
    constraints_df : pandas.DataFrame, optional
        constraint points to include in the plots
    constraint_style : str, optional
        pygmt style string for for constraint points, by default 'x.3c'
    fig_height : float, optional
        height of the figure, by default 12
    """

    initial_topo = prisms_ds.starting_topography

    final_topo = prisms_ds.topography

    points = constraints_df if constraints_df is not None else None

    # pylint: disable=duplicate-code
    _ = polar_utils.grd_compare(
        initial_topo,
        final_topo,
        fig_height=fig_height,
        grid1_name="Initial topography",
        grid2_name="Inverted topography",
        robust=True,
        hist=True,
        inset=False,
        verbose="q",
        title="difference",
        coast=False,
        reverse_cpt=True,
        cmap="rain",
        points=points,
        points_style=constraint_style,
        hemisphere="south",
    )
    # pylint: enable=duplicate-code


def plot_inversion_grav_results(
    grav_results: pd.DataFrame,
    region: tuple[float, float, float, float],
    constraints_df: pd.DataFrame | None = None,
    fig_height: float = 12,
    constraint_style: str = "x.3c",
) -> None:
    """
    plot the initial and final misfit grids from the inversion and their difference

    Parameters
    ----------
    grav_results : pandas.DataFrame
        resulting dataframe of gravity data from the inversion
    region : tuple[float, float, float, float]
        region to use for gridding in format (xmin, xmax, ymin, ymax)
    iterations : list[int]
        list of all the iteration numbers
    constraints_df : pandas.DataFrame, optional
        constraint points to include in the plots
    fig_height : float, optional
        height of the figure, by default 12
    constraint_style : str, optional
        pygmt style string for for constraint points, by default 'x.3c'
    """

    grid = grav_results.set_index(["northing", "easting"]).to_xarray()

    initial_residual = grid["iter_1_initial_residual"]
    final_residual = grid.res

    initial_rmse = utils.rmse(grav_results["iter_1_initial_residual"])
    final_rmse = utils.rmse(grav_results.res)

    points = constraints_df if constraints_df is not None else None

    dif, initial, final = polar_utils.grd_compare(
        initial_residual,
        final_residual,
        plot=False,
    )
    robust = True
    diff_maxabs = vd.maxabs(polar_utils.get_min_max(dif, robust=robust))
    initial_maxabs = vd.maxabs(polar_utils.get_min_max(initial, robust=robust))
    final_maxabs = vd.maxabs(polar_utils.get_min_max(final, robust=robust))
    fig = maps.plot_grd(
        initial,
        fig_height=fig_height,
        region=region,
        cmap="balance+h0",
        # robust=True,
        cpt_lims=(-initial_maxabs, initial_maxabs),
        hist=True,
        cbar_label="mGal",
        title=f"Initial residual: RMSE:{round(initial_rmse, 2)} mGal",
        points=points,
        points_style=constraint_style,
        hemisphere="south",
    )
    fig = maps.plot_grd(
        dif,
        fig=fig,
        origin_shift="x",
        fig_height=fig_height,
        region=region,
        cmap="balance+h0",
        cpt_lims=(-diff_maxabs, diff_maxabs),
        hist=True,
        cbar_label="mGal",
        title=f"difference: RMSE:{round(utils.rmse(dif), 2)} mGal",
        points=points,
        points_style=constraint_style,
        hemisphere="south",
    )
    fig = maps.plot_grd(
        final,
        fig=fig,
        origin_shift="x",
        fig_height=fig_height,
        region=region,
        cmap="balance+h0",
        # robust=True,
        cpt_lims=(-final_maxabs, final_maxabs),
        hist=True,
        cbar_label="mGal",
        title=f"Final residual: RMSE:{round(final_rmse, 2)} mGal",
        points=points,
        points_style=constraint_style,
        hemisphere="south",
    )
    fig.show()


def plot_inversion_iteration_results(
    grids: tuple[list[xr.DataArray], list[xr.DataArray], list[xr.DataArray]],
    grav_results: pd.DataFrame,
    updated_results: pd.DataFrame,
    parameters: dict[str, typing.Any],
    iterations: list[int],
    style: str,
    topo_cmap_perc: float = 1,
    misfit_cmap_perc: float = 1,
    corrections_cmap_perc: float = 1,
    constraints_df: pd.DataFrame | None = None,
    constraint_size: float = 1,
) -> None:
    """
    plot the starting misfit, updated topography, and correction grids for a specified
    number of the iterations of an inversion

    Parameters
    ----------
    grids : tuple[list[xarray.DataArray], list[xarray.DataArray],
        list[xarray.DataArray]]
        lists of misfit, topography, and correction grids
    grav_results : pandas.DataFrame
        gravity dataframe resulting from the inversion
    updated_results : pandas.DataFrame
        updated topography or density values resulting from the inversion
    parameters : dict[str, typing.Any]
        inversion parameters resulting from the inversion
    iterations : list[int]
        list of all the iteration numbers which occurred in the inversion
    style : str
        inversion style, either 'geometry' or 'density'
    topo_cmap_perc : float, optional
        value to multiply the max and min colorscale values by, by default 1
    misfit_cmap_perc : float, optional
        value to multiply the max and min colorscale values by, by default 1
    corrections_cmap_perc : float, optional
        value to multiply the max and min colorscale values by, by default 1
    constraints_df : pandas.DataFrame, optional
        constraint points to include in the plots
    constraint_size : float, optional
        size for constraint points, by default 1
    """

    misfit_grids, updated_grids, corrections_grids = grids

    params = copy.deepcopy(parameters)

    # set figure parameters
    sub_width = 5
    nrows, ncols = len(iterations), 3

    # setup subplot figure
    _fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(sub_width * ncols, sub_width * nrows),
    )

    # set color limits for each column
    misfit_lims = []
    updated_lims = []
    corrections_lims = []

    for g in misfit_grids:
        misfit_lims.append(polar_utils.get_min_max(g))
    for g in updated_grids:
        updated_lims.append(polar_utils.get_min_max(g))
    for g in corrections_grids:
        corrections_lims.append(polar_utils.get_min_max(g))

    misfit_min = min([i[0] for i in misfit_lims])  # pylint: disable=consider-using-generator
    misfit_max = max([i[1] for i in misfit_lims])  # pylint: disable=consider-using-generator
    misfit_lim = vd.maxabs(misfit_min, misfit_max) * misfit_cmap_perc

    updated_min = min([i[0] for i in updated_lims]) * topo_cmap_perc  # pylint: disable=consider-using-generator
    updated_max = max([i[1] for i in updated_lims]) * topo_cmap_perc  # pylint: disable=consider-using-generator

    corrections_min = min([i[0] for i in corrections_lims])  # pylint: disable=consider-using-generator
    corrections_max = max([i[1] for i in corrections_lims])  # pylint: disable=consider-using-generator
    corrections_lim = (
        vd.maxabs(corrections_min, corrections_max) * corrections_cmap_perc
    )

    for column, j in enumerate(grids):
        for row, _y in enumerate(j):
            # if only 1 iteration
            axes = ax[column] if max(iterations) == 1 else ax[row, column]
            # add iteration number as text
            plt.text(
                -0.1,
                0.5,
                f"Iteration #{iterations[row]}",
                transform=axes.transAxes,
                rotation="vertical",
                ha="center",
                va="center",
                fontsize=20,
            )
            # set colormaps and limits
            if column == 0:  # misfit grids
                cmap = "RdBu_r"
                lims = (-misfit_lim, misfit_lim)
                robust = True
                norm = None
            elif column == 1:  # updated grids
                if style == "density":
                    if (updated_min < 0) & (updated_max > 0):
                        cmap = "RdBu_r"
                        maxabs = vd.maxabs(updated_min, updated_max)
                        lims = (-maxabs, maxabs)
                    else:
                        cmap = "viridis"
                        lims = (updated_min, updated_max)
                elif style == "geometry":
                    cmap = "gist_earth"
                    lims = (updated_min, updated_max)

                robust = True
                norm = None
            elif column == 2:  # correction grids
                cmap = "RdBu_r"
                lims = (-corrections_lim, corrections_lim)
                robust = True
                norm = None
            # plot grids
            _y.plot(
                ax=axes,
                cmap=cmap,  # pylint: disable=possibly-used-before-assignment
                norm=norm,  # pylint: disable=possibly-used-before-assignment
                robust=robust,  # pylint: disable=possibly-used-before-assignment
                vmin=lims[0],  # pylint: disable=possibly-used-before-assignment
                vmax=lims[1],  # pylint: disable=possibly-used-before-assignment
                cbar_kwargs={
                    "orientation": "horizontal",
                    "anchor": (1, 1),
                    "fraction": 0.05,
                    "pad": 0.04,
                },
            )

            # add subplot titles
            if column == 0:  # misfit grids
                rmse = utils.rmse(
                    grav_results[f"iter_{iterations[row]}_initial_residual"]
                )
                axes.set_title(f"initial residual RMSE = {round(rmse, 2)} mGal")
            elif column == 1:  # updated grids
                axes.set_title(f"updated {style}")
            elif column == 2:  # correction grids
                rmse = utils.rmse(updated_results[f"iter_{iterations[row]}_correction"])
                axes.set_title(f"iteration correction RMSE = {round(rmse, 2)} m")

            if (constraints_df is not None) & (column in (0, 1, 2)):  # misfit grids
                axes.plot(
                    constraints_df.easting,  # type: ignore[union-attr]
                    constraints_df.northing,  # type: ignore[union-attr]
                    "k.",
                    markersize=constraint_size,
                    markeredgewidth=1,
                )

            # set axes labels and make proportional
            axes.set_xticklabels([])
            axes.set_yticklabels([])
            axes.set_xlabel("")
            axes.set_ylabel("")
            axes.set_aspect("equal")

    # add text with inversion parameter info
    text1, text2, text3 = [], [], []
    params.pop("Iteration times")
    for i, (k, v) in enumerate(params.items(), start=1):
        if i <= 5:
            text1.append(f"{k}: {v}\n")
        elif i <= 11:
            text2.append(f"{k}: {v}\n")
        else:
            text3.append(f"{k}: {v}\n")

    text1 = "".join(text1)  # type: ignore[assignment]
    text2 = "".join(text2)  # type: ignore[assignment]
    text3 = "".join(text3)  # type: ignore[assignment]

    # if only 1 iteration
    if max(iterations) == 1:
        plt.text(
            x=0.0,
            y=1.1,
            s=text1,
            transform=ax[0].transAxes,
        )

        plt.text(
            x=0.0,
            y=1.1,
            s=text2,
            transform=ax[1].transAxes,
        )

        plt.text(
            x=0.0,
            y=1.1,
            s=text3,
            transform=ax[2].transAxes,
        )
    else:
        plt.text(
            x=0.0,
            y=1.1,
            s=text1,
            transform=ax[0, 0].transAxes,
        )

        plt.text(
            x=0.0,
            y=1.1,
            s=text2,
            transform=ax[0, 1].transAxes,
        )

        plt.text(
            x=0.0,
            y=1.1,
            s=text3,
            transform=ax[0, 2].transAxes,
        )


def plot_inversion_results(
    grav_results: pd.DataFrame | str,  # noqa: ARG001
    topo_results: pd.DataFrame | str,  # noqa: ARG001
    parameters: dict[str, typing.Any] | str,  # noqa: ARG001
    grav_region: tuple[float, float, float, float] | None,  # noqa: ARG001
    iters_to_plot: int | None = None,  # noqa: ARG001
    plot_iter_results: bool = True,  # noqa: ARG001
    plot_topo_results: bool = True,  # noqa: ARG001
    plot_grav_results: bool = True,  # noqa: ARG001
    constraints_df: pd.DataFrame | None = None,  # noqa: ARG001
    **kwargs: typing.Any,  # noqa: ARG001
) -> None:
    """
    DEPRECATED: use the `Inversion` class method `plot_inversion_results` instead
    """
    # pylint: disable=W0613
    msg = (
        "Function `plot_inversion_results` deprecated, use the `Inversion` class method "
        "`plot_inversion_results` instead"
    )
    raise DeprecationWarning(msg)


def add_light(
    plotter: pyvista.Plotter,
    prisms: xr.Dataset,
) -> None:
    """
    add a light to a pyvista plotter object

    Parameters
    ----------
    plotter : pyvista.Plotter
        pyvista plotter object
    prisms : xarray.Dataset
        harmonica prisms layer
    """

    # Add a ceiling light
    west, east, south, north = vd.get_region((prisms.easting, prisms.northing))
    easting_center, northing_center = (east + west) / 2, (north + south) / 2
    light = pyvista.Light(
        position=(easting_center, northing_center, 100e3),
        focal_point=(easting_center, northing_center, 0),
        intensity=1,  # 0 to 1
        light_type="scene light",  # the light doesn't move with the camera
        positional=False,  # the light comes from infinity
        shadow_attenuation=0,  # 0 to 1,
    )
    plotter.add_light(light)


def show_prism_layers(
    **kwargs: typing.Any,  # noqa: ARG001 # pylint: disable=unused-argument
) -> None:
    """
    DEPRECATED: use `plot_prism_layers` instead
    """
    msg = "Function `show_prism_layers` deprecated, use `plot_prism_layers` instead"
    raise DeprecationWarning(msg)


def plot_prism_layers(
    prisms: list[xr.Dataset] | xr.Dataset,
    cmap: str = "viridis",
    color_by: str = "density",
    region: tuple[float, float, float, float] | None = None,
    opacity: float = 1,
    zscale: float = 75,
    log_scale: bool = False,
    clip_box: bool = False,
    box_buffer: float = 5e3,
    show_axes: bool = True,
    camera_position: str = "xz",
    elevation: float = 20,
    azimuth: float = -25,
    zoom: float = 1.2,
    backend: str = "static",
    cbar_args: dict[str, typing.Any] | None = None,
    constant_colors: list[str] | None = None,
    show_edges: bool = False,
) -> None:
    """
    show prism layers using PyVista

    Parameters
    ----------
    prisms : list | xarray.Dataset
        either a single harmonica prism layer of list of layers,
    cmap : str, optional
        matplotlib colorscale to use, by default "viridis"
    color_by : str, optional
        either use a variable of the prism_layer dataset, typically 'density' or
        'thickness', or choose 'constant' to have each layer colored by a unique color
        use kwarg `colors` to alter these colors, by default is "density"
    region : tuple[float, float, float, float], optional
        region to clip the model to, by default None
    clip_box : bool, optional
        clip a corner out of the model to help visualize, by default False

    """
    pyvista.global_theme.allow_empty_mesh = True

    plotter = pyvista.Plotter(
        lighting="three_lights",
        notebook=True,
    )

    if isinstance(prisms, xr.Dataset):
        prisms = [prisms]

    if constant_colors is None:
        constant_colors = [
            "goldenrod",
            "saddlebrown",
            "black",
            "lavender",
            "aqua",
        ]

    if cbar_args is None:
        if color_by == "density":
            title = "Density contrast (kg/m³)"
        elif color_by == "thickness":
            title = "Prism thickness (m)"
        elif color_by == "mask":
            title = "Model mask"
        elif color_by == "topography":
            title = "Topography (m)"
        else:
            title = ""
        cbar_args = {
            "title": title,
            "title_font_size": 35,
            "fmt": "%.0f",
            "width": 0.6,
            "position_x": 0.2,
        }

    # clip corner out of model to help visualize
    if clip_box is True:
        # extract region from first prism layer
        reg = vd.get_region(
            (prisms[0].easting.to_numpy(), prisms[0].northing.to_numpy())
        )

    for i, j in enumerate(prisms):
        # if region is given, clip model
        if region is not None:
            j = j.sel(  # noqa: PLW2901
                easting=slice(region[0], region[1]),
                northing=slice(region[2], region[3]),
            )

        # turn prisms into pyvista object
        pv_grid = j.prism_layer.to_pyvista(drop_null_prisms=False)

        # clip corner out of model to help visualize
        if clip_box is True:
            # set 6 edges of cube to clip out
            bounds = [
                reg[0] - box_buffer,
                reg[0] + box_buffer + ((reg[1] - reg[0]) / 2),
                reg[2] - box_buffer,
                reg[2] + box_buffer + ((reg[3] - reg[2]) / 2),
                np.nanmin(j.bottom),
                np.nanmax(j.top),
            ]
            pv_grid = pv_grid.clip_box(
                bounds,
                invert=True,
            )

        if color_by == "constant":
            plotter.add_mesh(
                pv_grid,
                color=constant_colors[i],
                # smooth_shading=kwargs.get("smooth_shading", False),
                # style=kwargs.get("style", "surface"),
                show_edges=show_edges,
                log_scale=log_scale,
                opacity=opacity,
                scalar_bar_args=cbar_args,
            )
        else:
            plotter.add_mesh(
                pv_grid,
                scalars=color_by,
                cmap=cmap,
                # flip_scalars=kwargs.get("flip_scalars", False),
                # smooth_shading=kwargs.get("smooth_shading", False),
                # style=kwargs.get("style", "surface"),
                show_edges=show_edges,
                log_scale=log_scale,
                opacity=opacity,
                scalar_bar_args=cbar_args,
            )

    plotter.set_scale(zscale=zscale)  # exaggerate the vertical coordinate
    plotter.camera_position = camera_position
    plotter.camera.elevation = elevation
    plotter.camera.azimuth = azimuth
    plotter.camera.zoom = zoom

    # Add a ceiling light
    add_light(plotter, prisms[i])  # pylint: disable=undefined-loop-variable

    if show_axes:
        plotter.show_axes()

    plotter.show(jupyter_backend=backend)


def combined_slice(
    study: optuna.study.Study,  # noqa: ARG001 # pylint: disable=unused-argument
    attribute_names: list[str],  # noqa: ARG001 # pylint: disable=unused-argument
    parameter_name: str | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
) -> plotly.graph_objects.Figure:
    """
    DEPRECATED: use :func:`plot_optimization_combined_slice` instead
    """

    msg = (
        "Function `combined_slice` deprecated, use the `plot_optimization_combined_slice` function "
        "instead"
    )
    raise DeprecationWarning(msg)


def plot_optimization_combined_slice(
    study: optuna.study.Study,
    attribute_names: list[str],
    parameter_name: str | None = None,
) -> plotly.graph_objects.Figure:
    """
    plot combined slice plots for optimizations.

    Parameters
    ----------
    study : optuna.study.Study
        the optuna study object
    target_names : list[str]
        list of names for parameters in the study

    Returns
    -------
    plotly.graph_objects.Figure
        a plotly figure
    """

    figs = []
    names = []
    for i, j in enumerate(study.metric_names):
        f = optuna.visualization.plot_slice(
            study,
            params=parameter_name,
            target=lambda t: t.values[i],  # noqa: B023 PD011 # pylint: disable=cell-var-from-loop
            target_name=j,
        )
        if i == 0:
            figs.append(f)
            names.append(j)

    for i in attribute_names:  # type: ignore[assignment]
        f = optuna.visualization.plot_slice(
            study,
            params=parameter_name,
            target=lambda t: t.user_attrs[i],  # noqa: B023 # pylint: disable=cell-var-from-loop
            target_name=i,
        )
        figs.append(f)
        names.append(i)

    yaxes = {}
    for i, j in enumerate(names, start=1):
        if i == 1:
            pass
        else:
            yax = plotly.graph_objs.layout.YAxis(
                title=j,
                overlaying="y",
                side="left",
                anchor="free",
                autoshift=True,
            )
            yaxes[f"yaxis{i}"] = yax
    layout = plotly.graph_objects.Layout(
        yaxis1=plotly.graph_objs.layout.YAxis(
            title=names[0],
            side="right",
        ),
        **yaxes,
    )

    # Create figure with secondary x-axis
    fig = plotly.graph_objects.Figure(layout=layout)  # pylint: disable=possibly-used-before-assignment

    # Add traces
    for i, j in enumerate(names):
        fig.add_trace(
            plotly.graph_objects.Scatter(
                x=figs[i].data[0]["x"],
                y=figs[i].data[0]["y"],
                name=j,
                mode="markers",
                yaxis=f"y{i + 1}",
            )
        )

    fig.update_layout(
        xaxis=f.layout.xaxis,
        title=f.layout.title.text,
    )

    return fig


def plot_optuna_figures(
    study: optuna.study.Study,
    target_names: list[str],
    include_duration: bool = False,
    # params=None,
    # separate_param_importances=False,
    plot_history: bool = True,
    plot_slice: bool = True,
    plot_importance: bool = True,
    # plot_edf=True,
    # plot_pareto=True,
) -> None:
    """
    plot the results of an optuna optimization

    Parameters
    ----------
    study : optuna.study.Study
        the optuna study object
    target_names : list[str]
        list of names for parameters in the study
    include_duration : bool, optional
        whether to add the duration to the plot, by default False
    plot_history : bool, optional
        choose to plot the optimization history, by default True
    plot_slice : bool, optional
        choose to plot the parameter values vs. score for each parameter, by default
        True
    """

    if plot_history:
        optuna.visualization.plot_optimization_history(study).show()

    if plot_slice:
        for i, j in enumerate(target_names):
            optuna.visualization.plot_slice(
                study,
                target=lambda t: t.values[i],  # noqa: B023 PD011 # pylint: disable=cell-var-from-loop
                target_name=j,
            ).show()
        if include_duration is True and "duration" not in target_names:
            optuna.visualization.plot_slice(
                study,
                target=lambda t: t.duration.total_seconds(),
                target_name="Execution time",
            ).show()

    if plot_importance and (
        len([k for k, v in study.get_trials()[0].params.items()]) > 1
    ):
        optuna.visualization.plot_param_importances(study).show()


def plot_stochastic_results(
    stats_ds: xr.Dataset,
    points: pd.DataFrame | None = None,
    region: tuple[float, float, float, float] | None = None,
    **kwargs: typing.Any,
) -> None:
    """
    Plot the (weighted) standard deviation (uncertainty) and mean of the stochastic
    ensemble. Optionally, plot points as well.

    Parameters
    ----------
    stats_ds : xarray.Dataset
        dataset with the merged inversion results, generate from function
        `uncertainty.model_ensemble_stats`.
    points : pandas.DataFrame | None, optional
        dataframe with points to plot, by default None

    Keyword Arguments
    -----------------
    cmap : str, optional
        colormap to use for the ensemble mean, by default "rain"
    unit : str, optional
        unit of the data, by default "m"
    reverse_cpt : bool, optional
        reverse the ensemble mean colormap, by default True
    label : str, optional
        label for the colorbar, by default "ensemble mean"
    points_label : str, optional
        label for the points, by default None
    fig_height : float, optional
        height of the figure, by default 12
    """
    cmap = kwargs.get("cmap", "viridis")
    unit = kwargs.get("unit", "m")
    reverse_cpt = kwargs.get("reverse_cpt", True)
    label = kwargs.get("label", "ensemble mean")
    points_label = kwargs.get("points_label")
    fig_height = kwargs.get("fig_height", 12)

    try:
        stdev = stats_ds.weighted_stdev
        weighted = "weighted"
    except AttributeError:
        stdev = stats_ds.z_stdev
        weighted = ""

    if region is not None:
        stdev = stdev.sel(
            easting=slice(region[0], region[1]),
            northing=slice(region[2], region[3]),
        )

    fig = maps.plot_grd(
        stdev,
        fig_height=fig_height,
        cmap="thermal",
        robust=True,
        hist=True,
        cbar_label=f"{label}: {weighted} standard deviation, {unit}",
        title="Ensemble uncertainty",
    )
    if points is not None:
        fig.plot(
            x=points.easting,
            y=points.northing,
            fill="black",
            style="x.3c",
            pen="1p",
            label=points_label,
        )
        fig.legend()

    try:
        mean = stats_ds.weighted_mean
    except AttributeError:
        mean = stats_ds.z_mean

    if region is not None:
        mean = mean.sel(
            easting=slice(region[0], region[1]),
            northing=slice(region[2], region[3]),
        )

    fig = maps.plot_grd(
        mean,
        fig_height=fig_height,
        cmap=cmap,
        reverse_cpt=reverse_cpt,
        robust=True,
        hist=True,
        cbar_label=f"{label}: {weighted} mean ({unit})",
        title="Ensemble mean",
        fig=fig,
        origin_shift="x",
    )
    if points is not None:
        fig.plot(
            x=points.easting,
            y=points.northing,
            fill="black",
            style="x.3c",
            pen="1p",
            label=points_label,
        )
        fig.legend()

    fig.show()


def remove_df_from_hoverdata(
    plot: plotly.graph_objects.Figure,
) -> plotly.graph_objects.Figure:
    """
    Remove the dataframe from the hoverdata of a plotly plot

    Parameters
    ----------
    plot : plotly.graph_objects.Figure
        plotly figure

    Returns
    -------
    plotly.graph_objects.Figure
        plotly figure with the dataframe removed from the hoverdata
    """

    text = []
    for s in plot.data[1].text:
        sub1 = '<br>    "results"'
        sub2 = "<br>    "
        start = s.split(sub1)[0]
        end = s.split(sub1)[1]
        new_string = start + end.split(sub2)[1]
        text.append(new_string)
    text = tuple(text)  # type: ignore[assignment]

    return plot.update_traces(text=text)


def plot_latin_hypercube(
    params_dict: dict[str, dict[str, typing.Any]],
    plot_individual_dists: bool = True,
    plot_2d_projections: bool = True,
) -> None:
    """
    With a dictionary of parameters and their sampled values, plot the individual
    distributions and or the 2D projections of the parameter pairs.

    Parameters
    ----------
    params_dict : dict[str, dict[str, typing.Any]]
        dictionary of sampled parameter values, can be created manually or from the
        output of func:`.uncertainty.create_lhc`
    plot_individual_dists : bool, optional
        choose to plot distribution of each parameter, by default True
    plot_2d_projections : bool, optional
        choose to plot the 2D projection of each parameter pair, by default True
    """
    df = pd.DataFrame(
        [params_dict[x]["sampled_values"] for x in params_dict],
    ).transpose()

    df.columns = params_dict.keys()

    # plot individual variables
    if plot_individual_dists is True:
        _, axes = plt.subplots(
            1,
            len(df.columns),
            figsize=(3 * len(df.columns), 1.8),
        )
        if len(df.columns) == 1:
            axes = [axes]

        for i, j in enumerate(df.columns):
            sns.kdeplot(
                ax=axes[i],
                data=df,
                x=j,
            )
            sns.rugplot(ax=axes[i], data=df, x=j, linewidth=2.5, height=0.07)
            axes[i].set_xlabel(j.replace("_", " ").capitalize())
            axes[i].ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(0, 0),
            )
            axes[i].set_ylabel(None)

        plt.show()

    dim = np.shape(df)[1]

    param_values = df.to_numpy()

    problem = {
        "num_vars": dim,
        "names": [i.replace("_", " ") for i in df.columns],
        "bounds": [[-1, 1]] * dim,
    }

    # Rescale to the unit hypercube for the analysis
    sample = utils.scale_normalized(param_values, problem["bounds"])

    # 2D projection
    if plot_2d_projections:
        if len(df.columns) == 1:
            pass
        else:
            plot_sampled_projection_2d(sample, problem["names"])


def projection_2d(
    sample: NDArray,  # noqa: ARG001 # pylint: disable=unused-argument
    var_names: list[str],  # noqa: ARG001 # pylint: disable=unused-argument
) -> None:
    """
    DEPRECATED: use :func:`plot_sampled_projection_2d` instead
    """
    msg = (
        "Function `projection_2d` deprecated, use the `plot_sampled_projection_2d` function "
        "instead"
    )
    raise DeprecationWarning(msg)


def plot_sampled_projection_2d(
    sample: NDArray,
    var_names: list[str],
) -> None:
    """
    Plots the samples projected on each 2D plane

    Parameters
    ----------
    sample : numpy.ndarray
        The sampled values
    var_names : list[str]
        The names of the variables
    """
    dim = sample.shape[1]

    for i in range(dim):
        for j in range(dim):
            plt.subplot(dim, dim, i * dim + j + 1)
            plt.scatter(
                sample[:, j],
                sample[:, i],
                s=2,
            )
            if j == 0:
                plt.ylabel(var_names[i], rotation=0, ha="right")
            if i == dim - 1:
                plt.xlabel(var_names[j], rotation=20, ha="right")

            plt.xticks([])
            plt.yticks([])
    plt.show()


def edge_effects(
    grav_ds: xr.Dataset,  # noqa: ARG001 # pylint: disable=unused-argument
    layer: xr.DataArray,  # noqa: ARG001 # pylint: disable=unused-argument
    inner_region: tuple[float, float, float, float],  # noqa: ARG001 # pylint: disable=unused-argument
    plot_profile: bool = True,  # noqa: ARG001 # pylint: disable=unused-argument
) -> None:
    """
    DEPRECATED: use :func:`plot_edge_effects` instead
    """
    msg = (
        "Function `edge_effects` deprecated, use the `plot_edge_effects` function "
        "instead"
    )
    raise DeprecationWarning(msg)


def plot_edge_effects(
    grav_ds: xr.Dataset,
    layer: xr.DataArray,
    inner_region: tuple[float, float, float, float],
    plot_profile: bool = True,
) -> None:
    """
    Show the gravity edge effects and the percentage decay within the inner region and
    optionally a profile across the region.

    Parameters
    ----------
    grav_ds : xarray.Dataset
        the gravity dataset
    layer : xarray.DataArray
        the prism/tesseroid layer
    inner_region : tuple[float, float, float, float]
        the inside region, where forward gravity is calculated
    plot_profile : bool, optional
        plot a profile across the region, by default True
    """
    # plot profiles
    if plot_profile:
        data_dict = profiles.make_data_dict(
            ["calculated forward gravity", "true gravity (without edge effects)"],
            [grav_ds.forward, grav_ds.forward_no_edge_effects],
            ["black", "red"],
        )

        layers_dict = profiles.make_data_dict(
            ["surface", "reference"],
            [layer.top, layer.bottom],
            ["blue", "darkorange"],
        )

        fig, _, _ = profiles.plot_profile(
            "points",
            start=(inner_region[0], (inner_region[3] - inner_region[2]) / 2),
            stop=(inner_region[1], (inner_region[3] - inner_region[2]) / 2),
            layers_dict=layers_dict,
            data_dict=data_dict,
            fill_layers=False,
            fig_width=10,
            fig_height=8,
            data_height=6,
            hemisphere="south",
        )
        fig.show()

    dif = grav_ds.forward - grav_ds.forward_no_edge_effects
    max_grav = grav_ds.forward.to_numpy().max()
    percent_decay = 100 * (max_grav - (max_grav + dif)) / max_grav

    hist_vals = vd.grid_to_table(percent_decay).reset_index().scalars

    # plot histogram of gravity decay values
    sns.displot(hist_vals, kde=True, stat="percent")

    plt.xlabel("Percent of max forward gravity")
    plt.ylabel("Percent")
    plt.title("Percent gravity decay within inner region")
    plt.show()

    # plot gravity and percentage contours
    fig = maps.plot_grd(
        grav_ds.forward,
        cmap="viridis",
        region=inner_region,
        title="Forward gravity",
        cbar_label="mGal",
        scalebar=False,
        hist=True,
        hemisphere="south",
    )

    fig = maps.plot_grd(
        percent_decay,
        fig=fig,
        origin_shift="x",
        cmap="thermal",
        region=inner_region,
        title="Gravity edge effect",
        cbar_label="Percentage decay",
        scalebar=False,
        hist=True,
        hemisphere="south",
    )

    fig.grdcontour(grid=percent_decay)

    fig.show()
