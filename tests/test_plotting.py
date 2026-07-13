import matplotlib as mpl

mpl.use("Agg")  # must be set before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import verde as vd
import xarray as xr

import invert4geom
from invert4geom import plotting

################
################
# grid_inversion_results
################
################


def test_grid_inversion_results_with_tuple_coord_names():
    """
    regression test: the model dataset's `coord_names` attribute is a tuple, but it
    was compared against lists, so this function always raised a ValueError
    """
    easting = [0, 10000, 20000, 30000, 40000]
    northing = [0, 10000, 20000, 30000]
    surface = np.full((len(northing), len(easting)), 500.0)
    topography = vd.make_xarray_grid(
        (easting, northing),
        data=surface,
        data_names="upward",
        dims=("northing", "easting"),
    )
    model = invert4geom.inversion.create_model(
        topography=topography,
        zref=100,
        density_contrast=200,
    )

    grav_results = model.inv.df[["easting", "northing"]].copy()
    grav_results["iter_1_initial_residual"] = 1.0

    misfit_grids, topo_grids, corrections_grids = plotting.grid_inversion_results(
        misfits=["iter_1_initial_residual"],
        topos=["topography"],
        corrections=[],
        prisms_ds=model,
        grav_results=grav_results,
        region=(0, 20000, 0, 20000),
    )

    assert len(misfit_grids) == 1
    assert len(topo_grids) == 1
    assert corrections_grids == []
    # the topography grid should be clipped to the given region
    assert topo_grids[0].easting.max() <= 20000
    assert topo_grids[0].northing.max() <= 20000


def test_grid_inversion_results_invalid_coord_names_raises():
    ds = xr.Dataset(attrs={"coord_names": ("x", "y")})
    with pytest.raises(ValueError, match="must have either"):
        plotting.grid_inversion_results(
            misfits=[],
            topos=[],
            corrections=[],
            prisms_ds=ds,
            grav_results=pd.DataFrame(),
            region=(0, 1, 0, 1),
        )


################
################
# plot_scores
################
################


def test_plot_scores_returns_figure():
    fig = plotting.plot_scores(
        scores=[3.0, 1.0, 2.0],
        parameters=[0.1, 0.2, 0.3],
    )
    assert fig is not None
    plt.close("all")


def test_plot_scores_invalid_best_raises():
    with pytest.raises(ValueError, match="best must be"):
        plotting.plot_scores(
            scores=[3.0, 1.0],
            parameters=[0.1, 0.2],
            best="worst",
        )
    plt.close("all")


################
################
# align_yaxis
################
################


def test_align_yaxis_aligns_values():
    """after alignment, v1 on ax1 and v2 on ax2 share the same display position"""
    _fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_ylim(0, 10)
    ax2.set_ylim(0, 1)

    plotting.align_yaxis(ax1, 5.0, ax2, 0.2)

    display_y1 = ax1.transData.transform((0, 5.0))[1]
    display_y2 = ax2.transData.transform((0, 0.2))[1]
    assert display_y1 == pytest.approx(display_y2)
    plt.close("all")


################
################
# plot_latin_hypercube
################
################


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_plot_latin_hypercube_smoke():
    rng = np.random.default_rng(seed=0)
    params_dict = {
        "param1": {"sampled_values": rng.uniform(0, 1, 10)},
        "param2": {"sampled_values": rng.uniform(100, 200, 10)},
    }
    plotting.plot_latin_hypercube(params_dict)
    plt.close("all")


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_plot_latin_hypercube_single_parameter():
    rng = np.random.default_rng(seed=0)
    params_dict = {"param1": {"sampled_values": rng.uniform(0, 1, 10)}}
    plotting.plot_latin_hypercube(params_dict)
    plt.close("all")
