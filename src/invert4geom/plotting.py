from __future__ import annotations

import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt

try:
    import pyvista
except ImportError:
    pyvista = None
import verde as vd
import xarray as xr
from antarctic_plots import utils as ap_utils

from invert4geom import utils


def plot_convergence(
    results: pd.DataFrame,
    iter_times: list[float] | None = None,
    logy: bool = False,
    figsize: tuple[float, float] = (5, 3.5),
) -> None:
    """
    _summary_

    Parameters
    ----------
    results : pd.DataFrame
        gravity result dataframe
    iter_times : list[float] | None, optional
        list of iteration execution times, by default None
    logy : bool, optional
        choose whether to plot y axis in log scale, by default False
    figsize : tuple[float, float], optional
        width and height of figure, by default (5, 3.5)
    """
    # get misfit data at end of each iteration
    cols = [s for s in results.columns.to_list() if "_final_misfit" in s]
    iters = len(cols)
    final_misfits = [utils.rmse(results[i]) for i in cols]

    _fig, ax1 = plt.subplots(figsize=figsize)
    plt.title("Inversion convergence")
    ax1.plot(range(iters), final_misfits, "b-")
    ax1.set_xlabel("Iteration")
    if logy:
        ax1.set_yscale("log")
    ax1.set_ylabel("RMS (mGal)", color="b")
    ax1.tick_params(axis="y", colors="b", which="both")

    if iter_times is not None:
        ax2 = ax1.twinx()
        ax2.plot(range(iters), np.cumsum(iter_times), "g-")
        ax2.set_ylabel("Cumulative time (s)", color="g")
        ax2.tick_params(axis="y", colors="g")
        ax2.grid(False)
    plt.tight_layout()


def grid_inversion_results(
    misfits: list[str],
    topos: list[str],
    corrections: list[str],
    prisms_ds: xr.Dataset,
    grav_results: pd.DataFrame,
    region: tuple[float, float, float, float],
    spacing: float,
    registration: str,
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
    prisms_ds : xr.Dataset
        resulting dataset of prism layer from the inversion
    grav_results : pd.DataFrame
        resulting dataframe of gravity data from the inversion
    region : tuple[float, float, float, float]
        region to use for gridding in format (xmin, xmax, ymin, ymax)
    spacing : float
        grid spacing in meters
    registration : str
       grid registration type, either "g" for gridline or "p" for pixel

    Returns
    -------
    tuple[list[xr.DataArray], list[xr.DataArray], list[xr.DataArray]]
        lists of misfit, topography, and correction grids
    """
    misfit_grids = []
    for m in misfits:
        grid = pygmt.xyz2grd(
            data=grav_results[["easting", "northing", m]],
            region=region,
            spacing=spacing,
            registration=registration,
            verbose="q",
        )
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
    topo_cmap_perc: float = 1,
) -> None:
    """
    plot the initial and final topography grids from the inversion and their difference

    Parameters
    ----------
    prisms_ds : xr.Dataset
        dataset resulting from inversion
    topo_cmap_perc : float, optional
        value to multiple min and max values by for colorscale, by default 1
    """
    initial_topo = prisms_ds.starting_topo

    # list of variables ending in "_layer"
    topos = [s for s in list(prisms_ds.keys()) if "_layer" in s]
    # list of iterations, e.g. [1,2,3,4]
    its = [int(s[5:][:-6]) for s in topos]

    final_topo = prisms_ds[f"iter_{max(its)}_layer"]

    dif = initial_topo - final_topo

    robust = True

    topo_lims = []
    for g in [initial_topo, final_topo]:
        topo_lims.append(ap_utils.get_min_max(g))

    topo_min = min([i[0] for i in topo_lims]) * topo_cmap_perc  # pylint: disable=consider-using-generator
    topo_max = max([i[1] for i in topo_lims]) * topo_cmap_perc  # pylint: disable=consider-using-generator

    # set figure parameters
    sub_width = 5
    nrows, ncols = 1, 3

    # setup subplot figure
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(sub_width * ncols, sub_width * nrows),
    )

    initial_topo.plot(
        ax=ax[0],
        robust=robust,
        cmap="gist_earth",
        vmin=topo_min,
        vmax=topo_max,
        cbar_kwargs={
            "orientation": "horizontal",
            "anchor": (1, 1),
            "fraction": 0.05,
            "pad": 0.04,
        },
    )
    ax[0].set_title("initial topography")

    dif.plot(
        ax=ax[1],
        robust=True,
        cmap="RdBu_r",
        cbar_kwargs={
            "orientation": "horizontal",
            "anchor": (1, 1),
            "fraction": 0.05,
            "pad": 0.04,
        },
    )
    rmse = utils.rmse(dif)
    ax[1].set_title(f"difference, RMSE: {round(rmse,2)}m")
    final_topo.plot(
        ax=ax[2],
        robust=robust,
        cmap="gist_earth",
        vmin=topo_min,
        vmax=topo_max,
        cbar_kwargs={
            "orientation": "horizontal",
            "anchor": (1, 1),
            "fraction": 0.05,
            "pad": 0.04,
        },
    )
    ax[2].set_title("final topography")

    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_xlabel("")
        a.set_ylabel("")
        a.set_aspect("equal")

    fig.tight_layout()


def plot_inversion_grav_results(
    grav_results: pd.DataFrame,
    region: tuple[float, float, float, float],
    spacing: float,
    registration: str,
    iterations: list[int],
) -> None:
    """
    plot the initial and final misfit grids from the inversion and their difference

    Parameters
    ----------
    grav_results : pd.DataFrame
        resulting dataframe of gravity data from the inversion
    region : tuple[float, float, float, float]
        region to use for gridding in format (xmin, xmax, ymin, ymax)
    spacing : float
        grid spacing in meters
    registration : str
        grid registration type, either "g" for gridline or "p" for pixel
    iterations : list[int]
        list of all the iteration numbers
    """
    initial_misfit = pygmt.xyz2grd(
        data=grav_results[["easting", "northing", "iter_1_initial_misfit"]],
        region=region,
        spacing=spacing,
        registration=registration,
        verbose="q",
    )

    final_misfit = pygmt.xyz2grd(
        data=grav_results[
            ["easting", "northing", f"iter_{max(iterations)}_final_misfit"]
        ],
        region=region,
        spacing=spacing,
        registration=registration,
        verbose="q",
    )

    initial_rmse = utils.rmse(grav_results["iter_1_initial_misfit"])
    final_rmse = utils.rmse(grav_results[f"iter_{max(iterations)}_final_misfit"])

    _ = ap_utils.grd_compare(
        initial_misfit,
        final_misfit,
        plot=True,
        grid1_name=f"Initial misfit: RMSE={round(initial_rmse, 2)} mGal",
        grid2_name=f"Final misfit: RMSE={round(final_rmse, 2)} mGal",
        plot_type="xarray",
        cmap="RdBu_r",
        robust=False,
        hist=True,
        inset=False,
        verbose="q",
        title="difference",
    )


def plot_inversion_iteration_results(
    grids: tuple[list[xr.DataArray], list[xr.DataArray], list[xr.DataArray]],
    grav_results: pd.DataFrame,
    topo_results: pd.DataFrame,
    parameters: dict[str, typing.Any],
    iterations: list[int],
    topo_cmap_perc: float = 1,
    misfit_cmap_perc: float = 1,
    corrections_cmap_perc: float = 1,
) -> None:
    """
    plot the starting misfit, updated topography, and correction grids for a specified
    number of the iterations of an inversion

    Parameters
    ----------
    grids : tuple[list[xr.DataArray], list[xr.DataArray], list[xr.DataArray]]
        lists of misfit, topography, and correction grids
    grav_results : pd.DataFrame
        gravity dataframe resulting from the inversion
    topo_results : pd.DataFrame
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
    """
    misfit_grids, topo_grids, corrections_grids = grids

    params = parameters.copy()

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
        misfit_lims.append(ap_utils.get_min_max(g))
    for g in topo_grids:
        topo_lims.append(ap_utils.get_min_max(g))
    for g in corrections_grids:
        corrections_lims.append(ap_utils.get_min_max(g))

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
                cmap=cmap,
                norm=norm,
                robust=robust,
                vmin=lims[0],
                vmax=lims[1],
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

            # set axes labels and make proportional
            axes.set_xticklabels([])
            axes.set_yticklabels([])
            axes.set_xlabel("")
            axes.set_ylabel("")
            axes.set_aspect("equal")

    # add text with inversion parameter info
    text1, text2, text3 = [], [], []
    params.pop("iter_times")
    for i, (k, v) in enumerate(params.items(), start=1):
        if i <= 5:
            text1.append(f"{k}: {v}\n")
        elif i <= 9:
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
    grav_spacing: float,
    registration: str = "g",
    iters_to_plot: int | None = None,
    plot_iter_results: bool = True,
    plot_topo_results: bool = True,
    plot_grav_results: bool = True,
    **kwargs: typing.Any,
) -> None:
    """
    plot various results from the inversion

    Parameters
    ----------
    grav_results : pd.DataFrame | str
        gravity results dataframe or filename
    topo_results : pd.DataFrame | str
        topography results dataframe or filename
    parameters : dict[str, typing.Any] | str
        inversion parameters dictionary or filename
    grav_region : tuple[float, float, float, float] | None
        region to use for gridding in format (xmin, xmax, ymin, ymax), by default None
    grav_spacing : float
        grid spacing in meters
    registration : str, optional
        grid registration type, either "g" for gridline or "p" for pixel, by default "g"
    iters_to_plot : int | None, optional
        number of iterations to plot, including the first and last, by default None
    plot_iter_results : bool, optional
        plot the iteration results, by default True
    plot_topo_results : bool, optional
        plot the topography results, by default True
    plot_grav_results : bool, optional
        plot the gravity results, by default True
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
        grav_spacing,
        registration,
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
        )

    if plot_topo_results is True:
        plot_inversion_topo_results(
            prisms_ds,
            topo_cmap_perc=kwargs.get("topo_cmap_perc", 1),
        )

    if plot_grav_results is True:
        plot_inversion_grav_results(
            grav_results,
            grav_region,
            grav_spacing,
            registration,
            iterations,
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
    prisms : xr.Dataset
        harmonica prisms layer
    """
    # Check if pyvista are installed
    if pyvista is None:
        msg = (
            "Missing optional dependency 'pyvista' required for building pyvista grids."
        )
        raise ImportError(msg)
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
    prisms : list | xr.Dataset
        either a single harmonica prism layer of list of layers,
    cmap : str, optional
        matplotlib colorscale to use, by default "viridis"
    color_by : str, optional
        either use a variable of the prism_layer dataset, typically 'density' or
        'thickness', or choose 'constant' to have each layer colored by a unique color
        use kwarg `colors` to alter these colors, by default is "density"
    """
    # Check if pyvista are installed
    if pyvista is None:
        msg = (
            "Missing optional dependency 'pyvista' required for building pyvista grids."
        )
        raise ImportError(msg)

    # Plot with pyvista
    plotter = pyvista.Plotter(
        lighting="three_lights",
        notebook=True,
    )

    opacity = kwargs.get("opacity", None)

    if isinstance(prisms, xr.Dataset):
        prisms = [prisms]

    for i, j in enumerate(prisms):
        # turn prisms into pyvist object
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

    plotter.show_axes()

    plotter.show(jupyter_backend=kwargs.get("backend", "client"))
