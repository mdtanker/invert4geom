from __future__ import annotations  # pylint: disable=too-many-lines

import copy
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
from IPython.display import clear_output
from polartoolkit import maps
from polartoolkit import utils as polar_utils

from invert4geom import log, utils

# This ensures Plotly output works in multiple places:
# plotly_mimetype: VS Code notebook UI
# notebook: "Jupyter: Export to HTML" command in VS Code
# See https://plotly.com/python/renderers/#multiple-renderers
pio.renderers.default = "plotly_mimetype+notebook"


def plot_2_parameter_cv_scores(
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
    study: optuna.study.Study,
    param_names: tuple[str, str],
    plot_param_names: tuple[str, str] = ("Hyperparameter 1", "Hyperparameter 2"),
    figsize: tuple[float, float] = (5, 3.5),
    cmap: str | None = None,
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
    """

    sns.set_theme()

    if cmap is None:
        cmap = sns.color_palette("mako", as_cmap=True)

    df = study.trials_dataframe().sort_values(by="value")
    df = df[[param_names[0], param_names[1], "value"]]
    best = df.iloc[0]

    plt.figure(figsize=figsize)
    plt.title("Two parameter cross-validation")

    x = df[param_names[0]].values
    y = df[param_names[1]].values
    z = df.value.values

    x_buffer = (max(x) - min(x)) / 50
    y_buffer = (max(y) - min(y)) / 50

    # 2D grid for interpolation
    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))
    xi, yi = np.meshgrid(xi, yi)

    try:
        interp = sp.interpolate.CloughTocher2DInterpolator(list(zip(x, y)), z)
    except ValueError as e:
        log.error(
            "Error interpolating value in plot_2_parameter_cv_scores_uneven: %s", e
        )
        return
    zi = interp(xi, yi)

    # plt.pcolormesh(xi, yi, zi, cmap=cmap, shading='auto')
    plt.contourf(xi, yi, zi, 30, cmap=cmap)
    plt.colorbar().set_label("Scores")

    plt.plot(
        best[param_names[0]],
        best[param_names[1]],
        "s",
        markersize=10,
        color=sns.color_palette()[3],
        label="Minimum",
    )
    plt.scatter(
        x,
        y,
        marker=".",
        color="lightgray",
        edgecolor="black",
    )
    plt.legend(
        loc="upper right",
    )
    plt.xlim([min(x) - x_buffer, max(x) + x_buffer])
    plt.ylim([min(y) - y_buffer, max(y) + y_buffer])
    plt.xlabel(plot_param_names[0])
    plt.ylabel(plot_param_names[1])
    plt.xticks(rotation=20)

    plt.tight_layout()


def plot_cv_scores(
    scores: list[float],
    parameters: list[float],
    logx: bool = False,
    logy: bool = False,
    param_name: str = "Hyperparameter",
    figsize: tuple[float, float] = (5, 3.5),
    plot_title: str | None = None,
) -> None:
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
    """

    sns.set_theme()

    df0 = pd.DataFrame({"scores": scores, "parameters": parameters})
    df = df0.sort_values(by="parameters")

    best = df.scores.argmin()

    plt.figure(figsize=figsize)
    if plot_title is not None:
        plt.title(plot_title)
    else:
        plt.title(f"{param_name} Cross-validation")
    plt.plot(df.parameters, df.scores, marker="o")
    plt.plot(
        df.parameters.iloc[best],
        df.scores.iloc[best],
        "s",
        markersize=10,
        color=sns.color_palette()[3],
        label="Minimum",
    )
    plt.legend(loc="best")
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.xlabel(f"{param_name} value")
    plt.ylabel("Root Mean Square Error")

    plt.tight_layout()


def plot_convergence(
    results: pd.DataFrame,
    params: dict[str, typing.Any],
    inversion_region: tuple[float, float, float, float] | None = None,
    figsize: tuple[float, float] = (5, 3.5),
) -> None:
    """
    plot a graph of L2-norm and delta L2-norm vs iteration number.

    Parameters
    ----------
    results : pandas.DataFrame
        gravity result dataframe
    params : dict[str, typing.Any]
        inversion parameters output from function `run_inversion()`
    inversion_region : tuple[float, float, float, float] | None, optional
        inside region of inversion, by default None
    figsize : tuple[float, float], optional
        width and height of figure, by default (5, 3.5)
    """

    sns.set_theme()

    # get misfit data at end of each iteration
    cols = [s for s in results.columns.to_list() if "_final_misfit" in s]
    iters = len(cols)

    if inversion_region is not None:
        l2_norms = [np.sqrt(utils.rmse(results[results.inside][i])) for i in cols]
        starting_misfit = utils.rmse(results[results.inside]["iter_1_initial_misfit"])
        starting_l2_norm = np.sqrt(starting_misfit)
    else:
        l2_norms = [np.sqrt(utils.rmse(results[i])) for i in cols]
        starting_misfit = utils.rmse(results["iter_1_initial_misfit"])
        starting_l2_norm = np.sqrt(starting_misfit)

    # add starting l2 norm to the beginning of the list
    l2_norms.insert(0, starting_l2_norm)

    # calculate delta L2-norms
    delta_l2_norms = []
    for i, m in enumerate(l2_norms):
        if i == 0:
            delta_l2_norms.append(np.nan)
        else:
            delta_l2_norms.append(l2_norms[i - 1] / m)

    # get tolerance values
    l2_norm_tolerance = float(params["L2 norm tolerance"])
    delta_l2_norm_tolerance = float(params["Delta L2 norm tolerance"])

    # create figure instance
    _fig, ax1 = plt.subplots(figsize=figsize)

    # make second y axis for delta l2 norm
    ax2 = ax1.twinx()

    # plot L2-norm convergence
    ax1.plot(range(iters + 1), l2_norms, "b-")

    # plot delta L2-norm convergence
    ax2.plot(range(iters + 1), delta_l2_norms, "g-")

    # set axis labels, ticks and gridlines
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("L2-norm", color="b")
    ax1.tick_params(axis="y", colors="b", which="both")
    ax2.set_ylabel("Δ L2-norm", color="g")
    ax2.tick_params(axis="y", colors="g", which="both")
    ax2.grid(False)

    # add buffer to y axis limits
    ax1.set_ylim(0.9 * l2_norm_tolerance, starting_l2_norm)
    ax2.set_ylim(delta_l2_norm_tolerance, np.nanmax(delta_l2_norms))

    # set x axis to integer values
    ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    # make both y axes align at tolerance levels
    align_yaxis(ax1, l2_norm_tolerance, ax2, delta_l2_norm_tolerance)

    # plot horizontal line of tolerances
    ax2.axhline(
        y=delta_l2_norm_tolerance,
        linewidth=1,
        color="r",
        linestyle="dashed",
        label="tolerances",
    )

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Inversion convergence")
    plt.tight_layout()
    plt.show()


def plot_dynamic_convergence(
    l2_norms: list[float],
    l2_norm_tolerance: float,
    delta_l2_norms: list[float],
    delta_l2_norm_tolerance: float,
    starting_misfit: float,
    figsize: tuple[float, float] = (5, 3.5),
) -> None:
    """
    plot a dynamic graph of L2-norm and delta L2-norm vs iteration number.

    Parameters
    ----------
    l2_norms : list[float]
        list of l2 norm values
    l2_norm_tolerance : float
        l2 norm tolerance
    delta_l2_norms : list[float]
        list of delta l2 norm values
    delta_l2_norm_tolerance : float
        delta l2 norm tolerance
    starting_misfit : float
        starting misfit rmse
    figsize : tuple[float, float], optional
        width and height of figure, by default (5, 3.5)
    """

    sns.set_theme()

    clear_output(wait=True)

    l2_norms = l2_norms.copy()
    delta_l2_norms = delta_l2_norms.copy()

    assert len(delta_l2_norms) == len(l2_norms)

    l2_norms.insert(0, np.sqrt(starting_misfit))
    delta_l2_norms.insert(0, np.nan)

    iters = len(l2_norms)

    # create figure instance
    _fig, ax1 = plt.subplots(figsize=figsize)

    # make second y axis for delta l2 norm
    ax2 = ax1.twinx()

    # plot L2-norm convergence
    ax1.plot(list(range(len(l2_norms))), l2_norms, "b-")

    # plot delta L2-norm convergence
    if iters > 1:
        ax2.plot(list(range(len(delta_l2_norms))), delta_l2_norms, "g-")

    # set axis labels, ticks and gridlines
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("L2-norm", color="b")
    ax1.tick_params(axis="y", colors="b", which="both")
    ax2.set_ylabel("Δ L2-norm", color="g")
    ax2.tick_params(axis="y", colors="g", which="both")
    ax2.grid(False)

    # add buffer to y axis limits
    ax1.set_ylim(0.9 * (l2_norm_tolerance), np.sqrt(starting_misfit))
    if iters > 1:
        ax2.set_ylim(delta_l2_norm_tolerance, np.nanmax(delta_l2_norms))

    # set x axis to integer values
    ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    # plot current L2-norm and Δ L2-norm
    ax1.plot(
        iters - 1,
        l2_norms[-1],
        "^",
        markersize=6,
        color=sns.color_palette()[3],
        # label="current L2-norm",
    )
    if iters > 1:
        ax2.plot(
            iters - 1,
            delta_l2_norms[-1],
            "^",
            markersize=6,
            color=sns.color_palette()[3],
            # label="current Δ L2-norm",
        )

    # make both y axes align at tolerance levels
    align_yaxis(ax1, l2_norm_tolerance, ax2, delta_l2_norm_tolerance)

    # plot horizontal line of tolerances
    ax2.axhline(
        y=delta_l2_norm_tolerance,
        linewidth=1,
        color="r",
        linestyle="dashed",
        label="tolerances",
    )

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Inversion convergence")
    plt.tight_layout()
    plt.show()


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
    region: tuple[float, float, float, float] | None = None,
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

    initial_topo = prisms_ds.starting_topo

    final_topo = prisms_ds.topo

    if constraints_df is not None:
        points = constraints_df.rename(columns={"easting": "x", "northing": "y"})
    else:
        points = None

    # pylint: disable=duplicate-code
    _ = polar_utils.grd_compare(
        initial_topo,
        final_topo,
        fig_height=fig_height,
        region=region,
        plot=True,
        grid1_name="Initial topography",
        grid2_name="Inverted topography",
        robust=True,
        hist=True,
        inset=False,
        verbose="q",
        title="difference",
        grounding_line=False,
        reverse_cpt=True,
        cmap="rain",
        points=points,
        points_style=constraint_style,
    )
    # pylint: enable=duplicate-code


def plot_inversion_grav_results(
    grav_results: pd.DataFrame,
    region: tuple[float, float, float, float],
    iterations: list[int],
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

    initial_misfit = grid["iter_1_initial_misfit"]
    final_misfit = grid[f"iter_{max(iterations)}_final_misfit"]

    initial_rmse = utils.rmse(grav_results["iter_1_initial_misfit"])
    final_rmse = utils.rmse(grav_results[f"iter_{max(iterations)}_final_misfit"])

    if constraints_df is not None:
        points = constraints_df.rename(columns={"easting": "x", "northing": "y"})
    else:
        points = None

    dif, initial, final = polar_utils.grd_compare(
        initial_misfit,
        final_misfit,
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
        title=f"Initial misfit: RMSE:{round(initial_rmse, 2)} mGal",
        points=points,
        points_style=constraint_style,
    )
    fig = maps.plot_grd(
        dif,
        fig=fig,
        origin_shift="xshift",
        fig_height=fig_height,
        region=region,
        cmap="balance+h0",
        cpt_lims=(-diff_maxabs, diff_maxabs),
        hist=True,
        cbar_label="mGal",
        title=f"difference: RMSE:{round(utils.rmse(dif), 2)} mGal",
        points=points,
        points_style=constraint_style,
    )
    fig = maps.plot_grd(
        final,
        fig=fig,
        origin_shift="xshift",
        fig_height=fig_height,
        region=region,
        cmap="balance+h0",
        # robust=True,
        cpt_lims=(-final_maxabs, final_maxabs),
        hist=True,
        cbar_label="mGal",
        title=f"Final misfit: RMSE:{round(final_rmse, 2)} mGal",
        points=points,
        points_style=constraint_style,
    )
    fig.show()


def plot_inversion_iteration_results(
    grids: tuple[list[xr.DataArray], list[xr.DataArray], list[xr.DataArray]],
    grav_results: pd.DataFrame,
    topo_results: pd.DataFrame,
    parameters: dict[str, typing.Any],
    iterations: list[int],
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
    topo_results : pandas.DataFrame
        topography dataframe resulting from the inversion
    parameters : dict[str, typing.Any]
        inversion parameters resulting from the inversion
    iterations : list[int]
        list of all the iteration numbers which occurred in the inversion
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

    misfit_grids, topo_grids, corrections_grids = grids

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
    topo_lims = []
    corrections_lims = []

    for g in misfit_grids:
        misfit_lims.append(polar_utils.get_min_max(g))
    for g in topo_grids:
        topo_lims.append(polar_utils.get_min_max(g))
    for g in corrections_grids:
        corrections_lims.append(polar_utils.get_min_max(g))

    misfit_min = min([i[0] for i in misfit_lims])  # pylint: disable=consider-using-generator
    misfit_max = max([i[1] for i in misfit_lims])  # pylint: disable=consider-using-generator
    misfit_lim = vd.maxabs(misfit_min, misfit_max) * misfit_cmap_perc

    topo_min = min([i[0] for i in topo_lims]) * topo_cmap_perc  # pylint: disable=consider-using-generator
    topo_max = max([i[1] for i in topo_lims]) * topo_cmap_perc  # pylint: disable=consider-using-generator

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
            elif column == 1:  # topography grids
                cmap = "gist_earth"
                lims = (topo_min, topo_max)

                robust = True
                norm = None
            elif column == 2:  # correction grids
                cmap = "RdBu_r"
                lims = (-corrections_lim, corrections_lim)
                robust = True
                norm = None
            # plot grids
            j[row].plot(
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
                    grav_results[f"iter_{iterations[row]}_initial_misfit"]
                )
                axes.set_title(f"initial misfit RMSE = {round(rmse, 2)} mGal")
            elif column == 1:  # topography grids
                axes.set_title("updated topography")
            elif column == 2:  # correction grids
                rmse = utils.rmse(topo_results[f"iter_{iterations[row]}_correction"])
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
    grav_results: pd.DataFrame | str,
    topo_results: pd.DataFrame | str,
    parameters: dict[str, typing.Any] | str,
    grav_region: tuple[float, float, float, float] | None,
    iters_to_plot: int | None = None,
    plot_iter_results: bool = True,
    plot_topo_results: bool = True,
    plot_grav_results: bool = True,
    constraints_df: pd.DataFrame | None = None,
    **kwargs: typing.Any,
) -> None:
    """
    plot various results from the inversion

    Parameters
    ----------
    grav_results : pandas.DataFrame | str
        gravity results dataframe or filename
    topo_results : pandas.DataFrame | str
        topography results dataframe or filename
    parameters : dict[str, typing.Any] | str
        inversion parameters dictionary or filename
    grav_region : tuple[float, float, float, float] | None
        region to use for gridding in format (xmin, xmax, ymin, ymax), by default None
    iters_to_plot : int | None, optional
        number of iterations to plot, including the first and last, by default None
    plot_iter_results : bool, optional
        plot the iteration results, by default True
    plot_topo_results : bool, optional
        plot the topography results, by default True
    plot_grav_results : bool, optional
        plot the gravity results, by default True
    constraints_df : pandas.DataFrame, optional
        constraint points to include in the plots
    """
    # if results are given as filenames (strings), load them
    if isinstance(grav_results, str):
        grav_results = pd.read_csv(
            grav_results,
            sep=",",
            header="infer",
            index_col=None,
            compression="gzip",
        )
    if isinstance(topo_results, str):
        topo_results = pd.read_csv(
            topo_results,
            sep=",",
            header="infer",
            index_col=None,
            compression="gzip",
        )
    if isinstance(parameters, str):
        params = np.load(parameters, allow_pickle="TRUE").item()
    else:
        params = parameters

    prisms_ds = topo_results.set_index(["northing", "easting"]).to_xarray()

    # either set input inversion region or get from input gravity data extent
    if grav_region is None:
        grav_region = vd.get_region((grav_results.easting, grav_results.northing))

    # get lists of columns to grid
    misfits = [s for s in grav_results.columns.to_list() if "initial_misfit" in s]
    topos = [s for s in topo_results.columns.to_list() if "_layer" in s]
    corrections = [s for s in topo_results.columns.to_list() if "_correction" in s]

    # list of iterations, e.g. [1,2,3,4]
    its = [int(s[5:][:-15]) for s in misfits]

    # get on x amount of iterations to plot
    if iters_to_plot is not None:
        if iters_to_plot > max(its):
            iterations = its
        else:
            iterations = list(np.linspace(1, max(its), iters_to_plot, dtype=int))
    else:
        iterations = its

    # subset columns based on iterations to plot
    misfits = [misfits[i] for i in [x - 1 for x in iterations]]
    topos = [topos[i] for i in [x - 1 for x in iterations]]
    corrections = [corrections[i] for i in [x - 1 for x in iterations]]

    # grid all results
    grids = grid_inversion_results(
        misfits,
        topos,
        corrections,
        prisms_ds,
        grav_results,
        grav_region,
    )

    if plot_iter_results is True:
        plot_inversion_iteration_results(
            grids,
            grav_results,
            topo_results,
            params,
            iterations,
            topo_cmap_perc=kwargs.get("topo_cmap_perc", 1),
            misfit_cmap_perc=kwargs.get("misfit_cmap_perc", 1),
            corrections_cmap_perc=kwargs.get("corrections_cmap_perc", 1),
            constraints_df=constraints_df,
            constraint_size=kwargs.get("constraint_size", 1),
        )

    if plot_topo_results is True:
        plot_inversion_topo_results(
            prisms_ds,
            region=grav_region,
            constraints_df=constraints_df,
            constraint_style=kwargs.get("constraint_style", "x.3c"),
            fig_height=kwargs.get("fig_height", 12),
        )

    if plot_grav_results is True:
        plot_inversion_grav_results(
            grav_results,
            grav_region,
            iterations,
            constraints_df=constraints_df,
            fig_height=kwargs.get("fig_height", 12),
            constraint_style=kwargs.get("constraint_style", "x.3c"),
        )


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
    prisms: list[xr.Dataset] | xr.Dataset,
    cmap: str = "viridis",
    color_by: str = "density",
    **kwargs: typing.Any,
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
    """

    # Plot with pyvista
    plotter = pyvista.Plotter(
        lighting="three_lights",
        notebook=True,
    )

    opacity = kwargs.get("opacity")

    if isinstance(prisms, xr.Dataset):
        prisms = [prisms]

    for i, j in enumerate(prisms):
        # turn prisms into pyvista object
        pv_grid = j.prism_layer.to_pyvista()

        trans = opacity[i] if opacity is not None else None

        if color_by == "constant":
            colors = kwargs.get(
                "colors", ["lavender", "aqua", "goldenrod", "saddlebrown", "black"]
            )
            plotter.add_mesh(
                pv_grid,
                color=colors[i],
                smooth_shading=kwargs.get("smooth_shading", False),
                style=kwargs.get("style", "surface"),
                show_edges=kwargs.get("show_edges", False),
                opacity=trans,
                scalar_bar_args=kwargs.get("scalar_bar_args"),
            )
        else:
            plotter.add_mesh(
                pv_grid,
                scalars=color_by,
                cmap=cmap,
                flip_scalars=kwargs.get("flip_scalars", False),
                smooth_shading=kwargs.get("smooth_shading", False),
                style=kwargs.get("style", "surface"),
                show_edges=kwargs.get("show_edges", False),
                log_scale=kwargs.get("log_scale", True),
                opacity=trans,
                scalar_bar_args=kwargs.get("scalar_bar_args"),
            )
        plotter.set_scale(
            zscale=kwargs.get("zscale", 75)
        )  # exaggerate the vertical coordinate
        plotter.camera_position = kwargs.get("camera_position", "xz")
        plotter.camera.elevation = kwargs.get("elevation", 20)
        plotter.camera.azimuth = kwargs.get("azimuth", -25)
        plotter.camera.zoom(kwargs.get("zoom", 1.2))

    # Add a ceiling light
    add_light(plotter, prisms[i])  # pylint: disable=undefined-loop-variable

    if kwargs.get("show_axes", True):
        plotter.show_axes()

    plotter.show(jupyter_backend=kwargs.get("backend", "client"))


def combined_slice(
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
            target=lambda t: t.values[i],  # noqa: B023 # pylint: disable=cell-var-from-loop
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
                yaxis=f"y{i+1}",
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
                target=lambda t: t.values[i],  # noqa: B023 # pylint: disable=cell-var-from-loop
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
        origin_shift="xshift",
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
