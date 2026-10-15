import matplotlib as mpl

mpl.use("Agg")  # must be set before pyplot is imported

import bordado as bd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import verde as vd
import xarray as xr

from invert4geom import maps

REGION = (0.0, 40000.0, 0.0, 30000.0)
SPACING = 2000.0


@pytest.fixture(name="grid")
def fixture_grid() -> xr.DataArray:
    """a smooth, gridline-registered test grid"""
    easting, northing = bd.grid_coordinates(region=REGION, spacing=SPACING)
    values = 100 * np.sin(easting / 8000) * np.cos(northing / 6000)
    return vd.make_xarray_grid(
        (easting, northing),
        values,
        data_names="topo",
        dims=("northing", "easting"),
    ).topo


@pytest.fixture(name="points")
def fixture_points() -> pd.DataFrame:
    return pd.DataFrame(
        {"easting": [5000.0, 15000.0, 25000.0], "northing": [5000.0, 25000.0, 15000.0]}
    )


################
################
# get_grid_info
################
################


def test_get_grid_info_gridline(grid):
    """a gridline registered grid's region is the extent of its coordinates"""
    spacing, region, zmin, zmax, registration = maps.get_grid_info(grid)

    assert spacing == pytest.approx(SPACING)
    assert region == pytest.approx(REGION)
    assert zmin == pytest.approx(float(grid.min()))
    assert zmax == pytest.approx(float(grid.max()))
    assert registration == "g"


def test_get_grid_info_pixel():
    """a pixel registered grid's region extends half a cell past its coordinates"""
    easting, northing = bd.grid_coordinates(
        region=REGION, spacing=SPACING, pixel_register=True
    )
    grid = vd.make_xarray_grid(
        (easting, northing),
        np.ones_like(easting),
        data_names="v",
        dims=("northing", "easting"),
    ).v
    grid.attrs["registration"] = "p"

    spacing, region, _, _, registration = maps.get_grid_info(grid)

    assert spacing == pytest.approx(SPACING)
    assert region == pytest.approx(REGION)
    assert registration == "p"


def test_get_grid_info_squeezes_extra_dims(grid):
    """extra length-1 dimensions are dropped before extracting the info"""
    expanded = grid.expand_dims({"time": 1})

    assert maps.get_grid_info(expanded)[1] == pytest.approx(REGION)


def test_get_grid_info_single_point_raises():
    """a grid with a single point has no spacing to extract"""
    grid = xr.DataArray(
        [[1.0]],
        coords={"northing": [0.0], "easting": [0.0]},
        dims=("northing", "easting"),
    )
    with pytest.raises(ValueError, match="at least 2 points"):
        maps.get_grid_info(grid)


def test_get_grid_info_all_nans(grid):
    """an all-NaN grid gives NaN limits rather than raising"""
    _, _, zmin, zmax, _ = maps.get_grid_info(xr.full_like(grid, np.nan))

    assert np.isnan(zmin)
    assert np.isnan(zmax)


################
################
# subset_grid / get_min_max
################
################


def test_subset_grid(grid):
    subset = maps.subset_grid(grid, (10000, 20000, 10000, 20000))

    assert float(subset.easting.min()) >= 10000
    assert float(subset.easting.max()) <= 20000
    assert float(subset.northing.min()) >= 10000
    assert float(subset.northing.max()) <= 20000


def test_get_min_max_robust(grid):
    """robust limits are inside the true limits"""
    v_min, v_max = maps.get_min_max(grid)
    r_min, r_max = maps.get_min_max(grid, robust=True)

    assert r_min > v_min
    assert r_max < v_max


def test_get_min_max_absolute(grid):
    v_min, v_max = maps.get_min_max(grid, absolute=True)

    assert v_min == pytest.approx(-v_max)


def test_get_min_max_ignores_nans(grid):
    with_nans = grid.where(grid > 0)

    assert maps.get_min_max(with_nans)[1] == pytest.approx(maps.get_min_max(grid)[1])


def test_get_min_max_region(grid):
    """subsetting to a region changes the limits"""
    full = maps.get_min_max(grid)
    subset = maps.get_min_max(grid, region=(0, 10000, 0, 10000))

    assert subset != full


################
################
# square_subplots / regions_overlap / points_inside_region
################
################


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, (1, 1)), (2, (1, 2)), (3, (2, 2)), (4, (2, 2)), (5, (2, 3)), (9, (3, 3))],
)
def test_square_subplots(n, expected):
    assert maps.square_subplots(n) == expected


def test_square_subplots_fits_all():
    """every layout has at least as many cells as items"""
    for n in range(1, 30):
        nrows, ncols = maps.square_subplots(n)
        assert nrows * ncols >= n


def test_square_subplots_invalid_raises():
    with pytest.raises(ValueError, match="at least 1"):
        maps.square_subplots(0)


def test_regions_overlap():
    assert maps.regions_overlap((0, 10, 0, 10), (5, 20, -5, 8)) == (5, 10, 0, 8)


def test_regions_overlap_disjoint_raises():
    with pytest.raises(ValueError, match="don't overlap"):
        maps.regions_overlap((0, 10, 0, 10), (20, 30, 20, 30))


def test_points_inside_region(points):
    inside = maps.points_inside_region(points, (0, 20000, 0, 20000))

    assert len(inside) == 1
    assert inside.easting.iloc[0] == 5000


def test_points_inside_region_reverse(points):
    outside = maps.points_inside_region(points, (0, 20000, 0, 20000), reverse=True)

    assert len(outside) == 2


################
################
# colormaps
################
################


def test_parse_cmap_alias():
    """GMT-style names are translated to their matplotlib equivalents"""
    cmap, hinge = maps._parse_cmap("thermal")  # pylint: disable=protected-access

    assert cmap.name == "inferno"
    assert hinge is False


def test_parse_cmap_hinge():
    """a `+h0` suffix requests color limits centered on zero"""
    cmap, hinge = maps._parse_cmap("balance+h0")  # pylint: disable=protected-access

    assert cmap.name == "RdBu_r"
    assert hinge is True


def test_parse_cmap_reverse():
    cmap, _ = maps._parse_cmap("viridis", reverse=True)  # pylint: disable=protected-access

    assert cmap.name == "viridis_r"


def test_parse_cmap_unknown_falls_back(caplog):
    """an unknown colormap name logs a warning and falls back to viridis"""
    cmap, _ = maps._parse_cmap("not_a_real_colormap")  # pylint: disable=protected-access

    assert cmap.name == "viridis"
    assert "unknown colormap" in caplog.text


################
################
# point styles
################
################


@pytest.mark.parametrize(
    ("style", "marker"),
    [("x.3c", "x"), ("c.2c", "o"), ("t.5c", "^"), ("s.1c", "s"), (None, "x")],
)
def test_parse_points_style_marker(style, marker):
    assert maps._parse_points_style(style)[0] == marker  # pylint: disable=protected-access


def test_parse_points_style_size_scales_with_gmt_size():
    """a larger GMT size gives a larger matplotlib marker area"""
    small = maps._parse_points_style("c.2c")[1]  # pylint: disable=protected-access
    large = maps._parse_points_style("c.4c")[1]  # pylint: disable=protected-access

    assert large > small


def test_parse_points_style_unparsable_size_uses_default():
    marker, size = maps._parse_points_style("cabc")  # pylint: disable=protected-access

    assert marker == "o"
    assert size == 40.0


################
################
# plot_grid
################
################


def test_plot_grid_smoke(grid):
    fig, ax = maps.plot_grid(grid, title="title", cbar_label="m")

    assert ax.get_title() == "title"
    assert ax.get_xlim() == pytest.approx(REGION[0:2])
    assert ax.get_ylim() == pytest.approx(REGION[2:4])
    # the map, the colorbar and the scalebar's own axes
    assert len(fig.axes) == 2
    plt.close("all")


def test_plot_grid_hist_adds_axes(grid):
    """the colorbar histogram gets its own axes"""
    without = maps.plot_grid(grid, hist=False)[0]
    n_without = len(without.axes)
    plt.close("all")

    with_hist = maps.plot_grid(grid, hist=True)[0]

    assert len(with_hist.axes) == n_without + 1
    plt.close("all")


def test_plot_grid_no_colorbar(grid):
    fig, _ = maps.plot_grid(grid, colorbar=False)

    assert len(fig.axes) == 1
    plt.close("all")


def test_plot_grid_cpt_lims(grid):
    fig, _ = maps.plot_grid(grid, cpt_lims=(-10, 10))

    assert fig.axes[0].collections[0].get_clim() == (-10, 10)
    plt.close("all")


def test_plot_grid_absolute_centers_on_zero(grid):
    fig, _ = maps.plot_grid(grid, absolute=True)
    vmin, vmax = fig.axes[0].collections[0].get_clim()

    assert vmin == pytest.approx(-vmax)
    plt.close("all")


def test_plot_grid_hinge_cmap_centers_on_zero(grid):
    """`+h0` implies symmetric color limits"""
    fig, _ = maps.plot_grid(grid, cmap="balance+h0")
    vmin, vmax = fig.axes[0].collections[0].get_clim()

    assert vmin == pytest.approx(-vmax)
    plt.close("all")


def test_plot_grid_region_clips(grid):
    inner = (10000, 20000, 10000, 20000)
    _, ax = maps.plot_grid(grid, region=inner)

    assert ax.get_xlim() == pytest.approx(inner[0:2])
    assert ax.get_ylim() == pytest.approx(inner[2:4])
    plt.close("all")


def test_plot_grid_points_and_legend(grid, points):
    _, ax = maps.plot_grid(
        grid, points=points, points_style="x.3c", points_label="constraints"
    )

    assert ax.get_legend() is not None
    plt.close("all")


def test_plot_grid_points_missing_coords_warns(grid, caplog):
    """a points dataframe without coordinate columns is skipped with a log message"""
    _, ax = maps.plot_grid(grid, points=pd.DataFrame({"a": [1], "b": [2]}))

    assert "no recognized coordinate columns" in caplog.text
    assert not ax.collections[1:]
    plt.close("all")


def test_plot_grid_into_existing_axes(grid):
    """passing `ax` plots into the supplied axes rather than making a new figure"""
    fig, axes = plt.subplots(1, 2)
    returned_fig, returned_ax = maps.plot_grid(grid, ax=axes[0])

    assert returned_fig is fig
    assert returned_ax is axes[0]
    plt.close("all")


def test_plot_grid_all_nans(grid, caplog):
    """an all-NaN grid plots without raising"""
    maps.plot_grid(xr.full_like(grid, np.nan), hist=False)

    assert "no finite values" in caplog.text
    plt.close("all")


def test_plot_grid_no_frame(grid):
    _, ax = maps.plot_grid(grid, frame=False)

    assert len(ax.get_xticks()) == 0
    plt.close("all")


def test_plot_grid_shading(grid):
    """hillshading adds a second mesh on top of the grid"""
    _, ax = maps.plot_grid(grid, shading=True)

    assert len(ax.collections) == 2
    plt.close("all")


################
################
# scalebar / box / inset
################
################


def test_add_scalebar_preserves_limits(grid):
    _, ax = maps.plot_grid(grid, scalebar=False)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    maps.add_scalebar(ax)

    assert ax.get_xlim() == xlim
    assert ax.get_ylim() == ylim
    assert len(ax.artists) == 1
    plt.close("all")


def test_add_scalebar_explicit_length(grid):
    _, ax = maps.plot_grid(grid, scalebar=False)
    maps.add_scalebar(ax, length=5)

    assert "5 km" in [t.get_text() for t in ax.artists[0].txt_label.get_children()]
    plt.close("all")


def test_add_box(grid):
    _, ax = maps.plot_grid(grid, scalebar=False)
    n_lines = len(ax.lines)

    maps.add_box(ax, (5000, 20000, 5000, 20000), label="box")

    assert len(ax.lines) == n_lines + 1
    plt.close("all")


def test_add_inset(grid):
    _, ax = maps.plot_grid(grid, scalebar=False)
    inset_ax = maps.add_inset(ax, REGION, inset_region_multiplier=4)
    assert inset_ax is not None

    # the inset covers a wider area than the map
    assert inset_ax.get_xlim()[1] - inset_ax.get_xlim()[0] > REGION[1] - REGION[0]
    plt.close("all")


def test_add_inset_invalid_position_raises(grid):
    _, ax = maps.plot_grid(grid, scalebar=False)

    with pytest.raises(ValueError, match="position must be one of"):
        maps.add_inset(ax, REGION, position="middle")
    plt.close("all")


################
################
# colorbar
################
################


def test_add_colorbar_requires_values_for_hist(grid):
    fig, ax = maps.plot_grid(grid, colorbar=False, scalebar=False)

    with pytest.raises(ValueError, match="values must be provided"):
        maps.add_colorbar(fig, ax, ax.collections[0], hist=True)
    plt.close("all")


@pytest.mark.parametrize(
    ("cpt_lims", "expected"),
    [
        ((-1e6, 1e6), "neither"),
        ((0, 1e6), "min"),
        ((-1e6, 0), "max"),
        ((-1, 1), "both"),
    ],
)
def test_colorbar_extend(grid, cpt_lims, expected):
    """end triangles show where the color limits clip the data"""
    assert maps._colorbar_extend(grid, *cpt_lims) == expected  # pylint: disable=protected-access


def test_colorbar_extend_no_values():
    assert maps._colorbar_extend(None, 0, 1) == "neither"  # pylint: disable=protected-access


################
################
# subplots
################
################


def test_subplots_smoke(grid):
    fig = maps.subplots([grid, grid * 2], dims=(1, 2), titles=["a", "b"])

    map_axes = [ax for ax in fig.axes if ax.get_title()]
    assert [ax.get_title() for ax in map_axes] == ["a", "b"]
    plt.close("all")


def test_subplots_single_grid(grid):
    """a lone DataArray is accepted"""
    fig = maps.subplots(grid)

    assert len(fig.axes) >= 1
    plt.close("all")


def test_subplots_removes_unused_axes(grid):
    """a 2x2 layout for 3 grids drops the empty 4th panel"""
    fig = maps.subplots([grid, grid, grid], dims=(2, 2), colorbar=False, scalebar=False)

    assert len(fig.axes) == 3
    plt.close("all")


def test_subplots_mismatched_list_raises(grid):
    with pytest.raises(ValueError, match="must match the number of grids"):
        maps.subplots([grid, grid], titles=["only one"])


def test_subplots_per_grid_cmaps(grid):
    fig = maps.subplots(
        [grid, grid],
        dims=(1, 2),
        cmaps=["viridis", "RdBu_r"],
        cpt_limits=[(-1, 1), (-2, 2)],
        scalebar=False,
    )
    meshes = [ax.collections[0] for ax in fig.axes if ax.collections]

    assert meshes[0].get_clim() == (-1, 1)
    assert meshes[1].get_clim() == (-2, 2)
    plt.close("all")


def test_subplots_fig_title(grid):
    fig = maps.subplots([grid], fig_title="the title")

    assert fig._suptitle.get_text() == "the title"  # pylint: disable=protected-access
    plt.close("all")


################
################
# grid_compare
################
################


def test_grid_compare_returns_difference(grid):
    dif, first, second = maps.grid_compare(grid, grid * 2, plot=False)

    xr.testing.assert_allclose(dif, grid - grid * 2)
    xr.testing.assert_allclose(first, grid)
    xr.testing.assert_allclose(second, grid * 2)


def test_grid_compare_clips_to_region(grid):
    inner = (10000, 20000, 10000, 20000)
    _, first, _ = maps.grid_compare(grid, grid, plot=False, region=inner)

    assert float(first.easting.max()) <= inner[1]


def test_grid_compare_plots_three_panels(grid):
    maps.grid_compare(grid, grid * 2, grid1_name="one", grid2_name="two")
    titles = [ax.get_title() for ax in plt.gcf().axes if ax.get_title()]

    assert titles[0] == "one"
    assert titles[2] == "two"
    assert "RMSE" in titles[1]
    plt.close("all")


def test_grid_compare_no_rmse_in_title(grid):
    maps.grid_compare(grid, grid * 2, title="diff", rmse_in_title=False)
    titles = [ax.get_title() for ax in plt.gcf().axes if ax.get_title()]

    assert "diff" in titles
    plt.close("all")


def test_grid_compare_shares_color_limits(grid):
    """the two grids use the same color limits so they can be compared directly"""
    maps.grid_compare(grid, grid * 2)
    meshes = [ax.collections[0] for ax in plt.gcf().axes if ax.collections]

    assert meshes[0].get_clim() == meshes[2].get_clim()
    plt.close("all")


################
################
# basemap
################
################


def test_basemap_smoke():
    fig, ax = maps.basemap(REGION, title="basemap")

    assert ax.get_title() == "basemap"
    assert ax.get_xlim() == pytest.approx(REGION[0:2])
    assert len(fig.axes) == 1
    plt.close("all")


def test_basemap_into_existing_axes():
    fig, axes = plt.subplots(1, 2)
    returned_fig, returned_ax = maps.basemap(REGION, ax=axes[1])

    assert returned_fig is fig
    assert returned_ax is axes[1]
    plt.close("all")


################
################
# map_axes
################
################


def test_map_axes_shape_and_aspect():
    fig, axes = maps.map_axes(nrows=2, ncols=3, region=REGION, fig_height=3)

    assert len(axes) == 6
    # each panel is region-width / region-height times as wide as it is tall
    width, height = fig.get_size_inches()
    assert width / height == pytest.approx((40000 / 30000) * 3 / 2)
    plt.close("all")


################
################
# plot_profile
################
################


def test_plot_profile_two_panels(grid):
    fig, axes = maps.plot_profile(
        start=(0, 15000),
        stop=(40000, 15000),
        data={"gravity": (grid, "black")},
        layers={"top": (grid, "blue")},
        data_label="mGal",
        layers_label="m",
    )

    assert len(axes) == 2
    assert axes[0].get_ylabel() == "mGal"
    assert axes[1].get_ylabel() == "m"
    assert fig is axes[0].get_figure()
    plt.close("all")


def test_plot_profile_single_panel(grid):
    _, axes = maps.plot_profile(
        start=(0, 15000), stop=(40000, 15000), layers={"top": (grid, "blue")}
    )

    assert len(axes) == 1
    plt.close("all")


def test_plot_profile_no_data_raises():
    with pytest.raises(ValueError, match="at least one of"):
        maps.plot_profile(start=(0, 0), stop=(1, 1))


################
################
# coastlines
################
################


def test_add_coast_no_data_warns(grid, monkeypatch, caplog):
    """without coastline data the map is left unchanged instead of raising"""
    monkeypatch.setattr(maps, "_coastline_segments", lambda epsg=None: None)  # noqa: ARG005
    _, ax = maps.plot_grid(grid, scalebar=False)
    n_collections = len(ax.collections)

    maps.add_coast(ax)

    assert "no coastline data available" in caplog.text
    assert len(ax.collections) == n_collections


def test_add_coast_plots_visible_segments(grid, monkeypatch):
    """only the segments which intersect the map region are drawn"""
    inside = np.array([[10000.0, 10000.0], [20000.0, 20000.0]])
    outside = np.array([[1e9, 1e9], [2e9, 2e9]])
    monkeypatch.setattr(
        maps,
        "_coastline_segments",
        lambda epsg=None: [inside, outside],  # noqa: ARG005
    )
    _, ax = maps.plot_grid(grid, scalebar=False)
    n_collections = len(ax.collections)

    maps.add_coast(ax)

    assert len(ax.collections) == n_collections + 1
    assert len(ax.collections[-1].get_paths()) == 1
    assert ax.get_xlim() == pytest.approx(REGION[0:2])
    plt.close("all")


################
################
# get_combined_min_max
################
################


def test_get_combined_min_max(grid):
    """the combined limits span both grids"""
    v_min, v_max = maps.get_combined_min_max([grid, grid * 2])

    assert v_min == pytest.approx(float((grid * 2).min()))
    assert v_max == pytest.approx(float((grid * 2).max()))


def test_get_combined_min_max_absolute(grid):
    v_min, v_max = maps.get_combined_min_max([grid, grid * 2], absolute=True)

    assert v_min == pytest.approx(-v_max)


def test_get_combined_min_max_robust(grid):
    """robust limits are inside the true combined limits"""
    v_min, v_max = maps.get_combined_min_max([grid, grid * 2])
    r_min, r_max = maps.get_combined_min_max([grid, grid * 2], robust=True)

    assert r_min > v_min
    assert r_max < v_max


def test_get_combined_min_max_empty_raises():
    with pytest.raises(ValueError, match="at least one set of values"):
        maps.get_combined_min_max([])
