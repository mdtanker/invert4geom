"""
Matplotlib-based map plotting.

This module provides the map-making functionality used throughout Invert4Geom
(gridded maps with colorbar histograms, scalebars, and optional coastlines and
inset locator maps) using only Matplotlib, replacing the PyGMT / PolarToolkit
based plotting used previously.
"""

from __future__ import annotations  # pylint: disable=too-many-lines

import json
import string
import typing
import warnings

import bordado as bd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pooch
import pyproj
import verde as vd
import xarray as xr
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from invert4geom import logger

if typing.TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

# height in inches of a single map panel, matching the default 12cm figure height
# used by the previous PyGMT-based plots
DEFAULT_FIG_HEIGHT = 4.75

# names of GMT / cmocean colormaps used by older versions of Invert4Geom mapped onto
# their closest Matplotlib equivalents, so that user code passing GMT-style names
# keeps working
CMAP_ALIASES: dict[str, str] = {
    "rain": "gist_earth_r",
    "thermal": "inferno",
    "balance": "RdBu_r",
    "polar": "RdBu_r",
    "vik": "RdBu_r",
    "roma": "Spectral",
    "batlow": "cividis",
    "haxby": "turbo",
    "globe": "terrain",
    "topo": "terrain",
    "grayc": "gray",
    "matter": "magma",
    "dense": "Blues",
    "deep": "YlGnBu",
    "ice": "Blues_r",
    "oleron": "terrain",
}

# GMT point style codes mapped onto matplotlib markers
_GMT_MARKERS: dict[str, str] = {
    "c": "o",
    "s": "s",
    "t": "^",
    "i": "v",
    "d": "D",
    "x": "x",
    "+": "+",
    "a": "*",
    "n": "p",
    "h": "h",
}

# pinned release of the Natural Earth vector data used for coastlines
_NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "v5.1.2/geojson/ne_50m_coastline.geojson"
)


###
###
# Grid information helpers
###
###


def get_grid_info(
    grid: xr.DataArray,
) -> tuple[
    float,
    tuple[float, float, float, float],
    float,
    float,
    str,
]:
    """
    Get the spacing, region, min, max and registration of a grid.

    Parameters
    ----------
    grid : xarray.DataArray
        a 2D grid with evenly spaced coordinates

    Returns
    -------
    tuple[float, tuple[float, float, float, float], float, float, str]
        the grid spacing, the region in format (xmin, xmax, ymin, ymax), the minimum
        and maximum data values, and the registration (``g`` for gridline or ``p``
        for pixel)
    """
    if len(grid.dims) > 2:
        grid = grid.squeeze()

    # dims are ordered (northing, easting)
    northing_name, easting_name = (str(d) for d in grid.dims)
    easting = np.asarray(grid[easting_name].values)
    northing = np.asarray(grid[northing_name].values)

    spacings = []
    for coord in (easting, northing):
        if coord.size < 2:
            msg = "grid must have at least 2 points along each dimension"
            raise ValueError(msg)
        spacings.append(float(np.abs(np.diff(coord)).mean()))
    spacing = float(np.mean(spacings))

    registration = get_grid_registration(grid)

    # pixel registered grids store the cell centers, so the region extends half a
    # cell beyond the coordinates
    pad = spacing / 2 if registration == "p" else 0.0
    region = (
        float(easting.min()) - pad,
        float(easting.max()) + pad,
        float(northing.min()) - pad,
        float(northing.max()) + pad,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        zmin = float(np.nanmin(grid))
        zmax = float(np.nanmax(grid))

    return spacing, region, zmin, zmax, registration


def get_grid_registration(grid: xr.DataArray) -> str:
    """
    Get the registration of a grid, ``g`` for gridline or ``p`` for pixel. Reads it
    from the GMT xarray accessor if PyGMT has been imported, otherwise from the grid's
    attributes, and defaults to ``g``.

    Parameters
    ----------
    grid : xarray.DataArray
        grid to get the registration of

    Returns
    -------
    str
        ``g`` for gridline registration or ``p`` for pixel registration
    """
    # the `gmt` accessor is only registered if pygmt has been imported
    try:
        return "g" if grid.gmt.registration == 0 else "p"
    except AttributeError:
        pass

    registration = grid.attrs.get("registration")
    if registration in ("g", "p"):
        return typing.cast("str", registration)
    return "g"


def subset_grid(
    grid: xr.DataArray,
    region: tuple[float, float, float, float],
) -> xr.DataArray:
    """
    Clip a grid to a region.

    Parameters
    ----------
    grid : xarray.DataArray
        grid to clip
    region : tuple[float, float, float, float]
        region to clip to, in format (xmin, xmax, ymin, ymax)

    Returns
    -------
    xarray.DataArray
        the clipped grid
    """
    northing_name, easting_name = (str(d) for d in grid.dims)
    return grid.sel(
        {
            easting_name: slice(min(region[0:2]), max(region[0:2])),
            northing_name: slice(min(region[2:4]), max(region[2:4])),
        }
    )


def get_min_max(
    values: xr.DataArray | pd.Series | NDArray,
    robust: bool = False,
    absolute: bool = False,
    robust_percentiles: tuple[float, float] = (0.02, 0.98),
    region: tuple[float, float, float, float] | None = None,
) -> tuple[float, float]:
    """
    Get the min and max values of a grid or array, optionally using percentiles instead
    of the true min/max, and optionally symmetric about zero.

    Parameters
    ----------
    values : xarray.DataArray | pandas.Series | numpy.ndarray
        values to get the min and max of
    robust : bool, optional
        use the 2nd and 98th percentiles (or those given by ``robust_percentiles``)
        instead of the min and max, by default False
    absolute : bool, optional
        make the limits symmetric about zero, by default False
    robust_percentiles : tuple[float, float], optional
        decimal percentiles to use if ``robust`` is True, by default (0.02, 0.98)
    region : tuple[float, float, float, float] | None, optional
        clip the values to this region before getting the limits, by default None

    Returns
    -------
    tuple[float, float]
        the min and max values
    """
    if region is not None and isinstance(values, xr.DataArray):
        values = subset_grid(values, region)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if robust:
            v_min, v_max = np.nanquantile(values, robust_percentiles)
        else:
            v_min, v_max = np.nanmin(values), np.nanmax(values)

    if absolute:
        maxabs = vd.maxabs([v_min, v_max])
        v_min, v_max = -maxabs, maxabs

    return (float(v_min), float(v_max))


def get_combined_min_max(
    values: typing.Sequence[xr.DataArray | pd.Series | NDArray],
    robust: bool = False,
    absolute: bool = False,
    robust_percentiles: tuple[float, float] = (0.02, 0.98),
    region: tuple[float, float, float, float] | None = None,
) -> tuple[float, float]:
    """
    Get the min and max values across several grids or arrays, for giving a set of
    maps shared color limits.

    Parameters
    ----------
    values : sequence of xarray.DataArray | pandas.Series | numpy.ndarray
        the grids or arrays to get the combined limits of
    robust : bool, optional
        use the 2nd and 98th percentiles (or those given by ``robust_percentiles``)
        of each input instead of its min and max, by default False
    absolute : bool, optional
        make the limits symmetric about zero, by default False
    robust_percentiles : tuple[float, float], optional
        decimal percentiles to use if ``robust`` is True, by default (0.02, 0.98)
    region : tuple[float, float, float, float] | None, optional
        clip the values to this region before getting the limits, by default None

    Returns
    -------
    tuple[float, float]
        the combined min and max values
    """
    if not values:
        msg = "at least one set of values must be provided"
        raise ValueError(msg)

    limits = [
        get_min_max(
            v,
            robust=robust,
            robust_percentiles=robust_percentiles,
            region=region,
        )
        for v in values
    ]
    v_min = min(lim[0] for lim in limits)
    v_max = max(lim[1] for lim in limits)

    if absolute:
        maxabs = vd.maxabs([v_min, v_max])
        v_min, v_max = -maxabs, maxabs

    return (float(v_min), float(v_max))


def square_subplots(n: int) -> tuple[int, int]:
    """
    Get the number of rows and columns to arrange ``n`` subplots as close to a square
    as possible.

    Parameters
    ----------
    n : int
        number of subplots

    Returns
    -------
    tuple[int, int]
        number of rows and number of columns
    """
    if n < 1:
        msg = "n must be at least 1"
        raise ValueError(msg)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def points_inside_region(
    df: pd.DataFrame,
    region: tuple[float, float, float, float],
    names: tuple[str, str] = ("easting", "northing"),
    reverse: bool = False,
) -> pd.DataFrame:
    """
    Subset a dataframe to the points inside (or outside) a region.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe with coordinate columns
    region : tuple[float, float, float, float]
        region in format (xmin, xmax, ymin, ymax)
    names : tuple[str, str], optional
        names of the x and y coordinate columns, by default ("easting", "northing")
    reverse : bool, optional
        return the points outside the region instead, by default False

    Returns
    -------
    pandas.DataFrame
        the subset dataframe
    """
    inside = bd.inside((df[names[0]], df[names[1]]), region=region)
    return df.loc[~inside].copy() if reverse else df.loc[inside].copy()


def regions_overlap(
    region1: tuple[float, float, float, float],
    region2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Get the overlapping area of two regions.

    Parameters
    ----------
    region1, region2 : tuple[float, float, float, float]
        regions in format (xmin, xmax, ymin, ymax)

    Returns
    -------
    tuple[float, float, float, float]
        the overlapping region in format (xmin, xmax, ymin, ymax)
    """
    overlap = (
        max(region1[0], region2[0]),
        min(region1[1], region2[1]),
        max(region1[2], region2[2]),
        min(region1[3], region2[3]),
    )
    if overlap[0] >= overlap[1] or overlap[2] >= overlap[3]:
        msg = "the supplied regions don't overlap"
        raise ValueError(msg)
    return overlap


###
###
# Colormaps
###
###


def _parse_cmap(
    cmap: str | mpl.colors.Colormap,
    reverse: bool = False,
) -> tuple[mpl.colors.Colormap, bool]:
    """
    Resolve a colormap name into a Matplotlib colormap. GMT-style names (see
    :data:`CMAP_ALIASES`) are translated, and a ``+h0`` suffix (GMT's "hinge at zero")
    is stripped and reported so that the color limits can be centered on zero.

    Returns the colormap and whether it requested a hinge at zero.
    """
    hinge = False
    if isinstance(cmap, str):
        name = cmap
        if "+h" in name:
            name = name.split("+h")[0]
            hinge = True
        name = CMAP_ALIASES.get(name.lower(), name)
        try:
            colormap = mpl.colormaps[name]
        except KeyError:
            logger.warning("unknown colormap '%s', using 'viridis' instead", cmap)
            colormap = mpl.colormaps["viridis"]
    else:
        colormap = cmap

    if reverse:
        colormap = colormap.reversed()

    return colormap, hinge


###
###
# Map decorations
###
###


def add_scalebar(
    ax: mpl.axes.Axes,
    length: float | None = None,
    length_perc: float = 0.25,
    position: str = "lower center",
    font_color: str = "black",
    box: bool = True,
) -> None:
    """
    Add a scalebar to a map, with a length in km.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes to add the scalebar to, with coordinates in meters
    length : float | None, optional
        length of the scalebar in km, by default None which uses ``length_perc`` of the
        shortest dimension of the map, rounded to 1 significant figure
    length_perc : float, optional
        fraction of the shortest map dimension to use for the scalebar length, by
        default 0.25
    position : str, optional
        Matplotlib legend-style location string, by default "lower center"
    font_color : str, optional
        color of the scalebar text, by default "black"
    box : bool, optional
        draw a white box behind the scalebar, by default True
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if length is None:
        shortest = min(abs(xmax - xmin), abs(ymax - ymin))
        # round to 1 significant figure, in km
        raw = shortest / 1e3 * length_perc
        if raw <= 0:
            return
        length = float(np.round(raw, -int(np.floor(np.log10(abs(raw))))))
        if length <= 0:
            return

    scalebar = AnchoredSizeBar(
        ax.transData,
        length * 1e3,
        f"{length:g} km",
        position,
        pad=0.3,
        borderpad=0.4,
        color=font_color,
        frameon=box,
        size_vertical=abs(ymax - ymin) * 0.01,
        fontproperties=mpl.font_manager.FontProperties(size=9),
    )
    if box:
        scalebar.patch.set(facecolor="white", edgecolor="none", alpha=0.7)
    ax.add_artist(scalebar)


def _coastline_segments(
    epsg: str | None = None,
) -> list[NDArray] | None:
    """
    Fetch the Natural Earth 1:50m coastlines and reproject them to the supplied EPSG
    code. Returns a list of (n, 2) arrays of projected coordinates, or None if the data
    couldn't be fetched or reprojected.
    """
    try:
        fname = pooch.retrieve(
            url=_NATURAL_EARTH_URL,
            known_hash=None,
            fname="ne_50m_coastline.geojson",
            path=pooch.os_cache("invert4geom"),
        )
    except Exception as e:  # pylint: disable=broad-exception-caught # noqa: BLE001
        logger.error("couldn't download the coastline data: %s", e)
        return None

    with open(fname, encoding="utf-8") as f:  # noqa: PTH123
        geojson = json.load(f)

    lines: list[NDArray] = []
    for feature in geojson["features"]:
        geometry = feature["geometry"]
        if geometry["type"] == "LineString":
            lines.append(np.asarray(geometry["coordinates"], dtype=float))
        elif geometry["type"] == "MultiLineString":
            lines.extend(np.asarray(c, dtype=float) for c in geometry["coordinates"])

    if epsg is None or str(epsg) == "4326":
        return lines

    try:
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", f"EPSG:{epsg}", always_xy=True
        )
    except pyproj.exceptions.CRSError as e:
        logger.error("couldn't create a transformer for EPSG:%s: %s", epsg, e)
        return None

    projected = []
    for line in lines:
        x, y = transformer.transform(line[:, 0], line[:, 1])
        x, y = np.asarray(x), np.asarray(y)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            projected.append(np.column_stack([x[finite], y[finite]]))
    return projected


def add_coast(
    ax: mpl.axes.Axes,
    epsg: str | None = None,
    pen: str = "black",
    linewidth: float = 0.7,
    zorder: float = 5,
) -> None:
    """
    Add coastlines to a map. Uses the Natural Earth 1:50m coastlines, downloaded and
    cached with Pooch, reprojected to the supplied EPSG code.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes to add the coastlines to
    epsg : str | None, optional
        EPSG code of the map's projection, by default None which assumes the map is in
        geographic (EPSG:4326) coordinates
    pen : str, optional
        color of the coastlines, by default "black"
    linewidth : float, optional
        width of the coastlines, by default 0.7
    zorder : float, optional
        drawing order of the coastlines, by default 5
    """
    segments = _coastline_segments(epsg)
    if not segments:
        logger.warning("no coastline data available, skipping coastlines")
        return

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # only keep the segments which intersect the map region
    visible = [
        s
        for s in segments
        if (s[:, 0].max() >= xmin)
        and (s[:, 0].min() <= xmax)
        and (s[:, 1].max() >= ymin)
        and (s[:, 1].min() <= ymax)
    ]
    if not visible:
        return

    ax.add_collection(
        LineCollection(visible, colors=pen, linewidths=linewidth, zorder=zorder)
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def add_box(
    ax: mpl.axes.Axes,
    region: tuple[float, float, float, float],
    pen: str = "black",
    linewidth: float = 1.5,
    linestyle: str = "-",
    label: str | None = None,
    zorder: float = 6,
) -> None:
    """
    Draw a rectangular region as a box on a map.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes to draw the box on
    region : tuple[float, float, float, float]
        region in format (xmin, xmax, ymin, ymax)
    pen : str, optional
        color of the box, by default "black"
    linewidth : float, optional
        width of the box's lines, by default 1.5
    linestyle : str, optional
        style of the box's lines, by default "-"
    label : str | None, optional
        label to add to the legend, by default None
    zorder : float, optional
        drawing order of the box, by default 6
    """
    ax.plot(
        [region[0], region[1], region[1], region[0], region[0]],
        [region[2], region[2], region[3], region[3], region[2]],
        color=pen,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        zorder=zorder,
    )


def add_inset(
    ax: mpl.axes.Axes,
    region: tuple[float, float, float, float],
    epsg: str | None = None,
    position: str = "upper left",
    size: float = 0.25,
    inset_region_multiplier: float = 8,
) -> mpl.axes.Axes | None:
    """
    Add an inset locator map to the corner of a map, showing where the map's region
    falls within a wider area. Coastlines are included if they're available.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes to add the inset to
    region : tuple[float, float, float, float]
        region of the main map, in format (xmin, xmax, ymin, ymax)
    epsg : str | None, optional
        EPSG code of the map's projection, by default None
    position : str, optional
        corner to place the inset in, one of "upper left", "upper right",
        "lower left" or "lower right", by default "upper left"
    size : float, optional
        size of the inset as a fraction of the main map, by default 0.25
    inset_region_multiplier : float, optional
        how much wider than the map's region the inset should cover, by default 8

    Returns
    -------
    matplotlib.axes.Axes | None
        the inset axes, or None if it couldn't be created
    """
    anchors = {
        "upper left": (0.02, 1 - size - 0.02),
        "upper right": (1 - size - 0.02, 1 - size - 0.02),
        "lower left": (0.02, 0.02),
        "lower right": (1 - size - 0.02, 0.02),
    }
    if position not in anchors:
        msg = f"position must be one of {list(anchors)}"
        raise ValueError(msg)

    x0, y0 = anchors[position]
    inset_ax = ax.inset_axes((x0, y0, size, size))

    width = (region[1] - region[0]) * inset_region_multiplier / 2
    height = (region[3] - region[2]) * inset_region_multiplier / 2
    center = ((region[0] + region[1]) / 2, (region[2] + region[3]) / 2)
    inset_region = (
        center[0] - width,
        center[0] + width,
        center[1] - height,
        center[1] + height,
    )

    inset_ax.set_xlim(inset_region[0], inset_region[1])
    inset_ax.set_ylim(inset_region[2], inset_region[3])
    inset_ax.set_aspect("equal")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_facecolor("white")
    inset_ax.patch.set_alpha(0.8)

    add_coast(inset_ax, epsg=epsg, linewidth=0.4)
    add_box(inset_ax, region, pen="red", linewidth=1)

    return inset_ax


def _parse_points_style(
    style: str | None,
) -> tuple[str, float]:
    """
    Translate a GMT point style string (e.g. ``x.3c``) into a Matplotlib marker and
    marker size in points. Returns the default marker and size if ``style`` is None or
    can't be parsed.
    """
    default_marker, default_size = "x", 40.0

    if not style:
        return default_marker, default_size

    marker = _GMT_MARKERS.get(style[0], default_marker)

    size_str = style[1:].rstrip("cip")
    try:
        # GMT sizes are in cm; convert to points and then to the area used by scatter
        size_pts = float(size_str) / 2.54 * 72
    except ValueError:
        return marker, default_size

    return marker, float(size_pts**2 / 4)


def add_points(
    ax: mpl.axes.Axes,
    points: pd.DataFrame,
    style: str | None = None,
    color: str = "black",
    label: str | None = None,
    zorder: float = 7,
) -> None:
    """
    Plot point data on a map.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes to plot the points on
    points : pandas.DataFrame
        dataframe with coordinate columns, either ``easting``/``northing``,
        ``x``/``y``, or ``longitude``/``latitude``
    style : str | None, optional
        GMT-style point style string, e.g. ``x.3c``, by default None which plots black
        crosses
    color : str, optional
        color of the points, by default "black"
    label : str | None, optional
        label to add to the legend, by default None
    zorder : float, optional
        drawing order of the points, by default 7
    """
    coord_names = _point_coord_names(points)
    if coord_names is None:
        logger.warning(
            "points dataframe has no recognized coordinate columns, skipping points"
        )
        return

    marker, size = _parse_points_style(style)

    # unfilled markers ignore (and warn about) an edgecolor
    edge_kwargs = (
        {} if marker in ("x", "+", "1", "2", "3", "4") else {"edgecolors": "black"}
    )

    ax.scatter(
        points[coord_names[0]],
        points[coord_names[1]],
        marker=marker,
        s=size,
        c=color,
        linewidths=1,
        label=label,
        zorder=zorder,
        **edge_kwargs,
    )


def _point_coord_names(points: pd.DataFrame) -> tuple[str, str] | None:
    """get the coordinate column names of a points dataframe"""
    for names in (
        ("easting", "northing"),
        ("x", "y"),
        ("longitude", "latitude"),
    ):
        if all(n in points.columns for n in names):
            return names
    return None


def add_colorbar(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    mappable: mpl.cm.ScalarMappable,
    values: xr.DataArray | NDArray | None = None,
    label: str | None = None,
    hist: bool = False,
    hist_bins: int = 100,
    width_perc: float = 0.8,
    height_perc: float = 0.04,
    hist_height_perc: float = 0.15,
    yoffset_perc: float = 0.04,
) -> mpl.colorbar.Colorbar:
    """
    Add a horizontal colorbar below a map, optionally with a histogram of the plotted
    values drawn above it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        figure containing the map
    ax : matplotlib.axes.Axes
        axes of the map
    mappable : matplotlib.cm.ScalarMappable
        the mappable (e.g. the return of ``pcolormesh``) to make the colorbar for
    values : xarray.DataArray | numpy.ndarray | None, optional
        values to use for the histogram, required if ``hist`` is True, by default None
    label : str | None, optional
        label for the colorbar, by default None
    hist : bool, optional
        add a histogram of the values above the colorbar, by default False
    hist_bins : int, optional
        number of histogram bins, by default 100
    width_perc : float, optional
        width of the colorbar as a fraction of the map's width, by default 0.8
    height_perc : float, optional
        height of the colorbar as a fraction of its width, by default 0.04
    hist_height_perc : float, optional
        height of the histogram as a fraction of the map's height, by default 0.15
    yoffset_perc : float, optional
        gap between the map and the colorbar as a fraction of the map's height, by
        default 0.04

    Returns
    -------
    matplotlib.colorbar.Colorbar
        the colorbar
    """
    if hist and values is None:
        msg = "values must be provided if hist is True"
        raise ValueError(msg)

    # position everything relative to the map, in figure coordinates
    pos = ax.get_position()
    cbar_width = pos.width * width_perc
    cbar_height = cbar_width * height_perc
    cbar_x = pos.x0 + (pos.width - cbar_width) / 2
    cbar_y = pos.y0 - pos.height * yoffset_perc - cbar_height

    vmin, vmax = mappable.get_clim()

    if hist:
        hist_height = pos.height * hist_height_perc
        cbar_y -= hist_height
        hist_ax = fig.add_axes((cbar_x, cbar_y + cbar_height, cbar_width, hist_height))

        vals = _flatten_values(values)
        vals = vals[(vals >= vmin) & (vals <= vmax)]
        if vals.size == 0:
            logger.warning("no values within the color limits, skipping the histogram")
            hist_ax.remove()
        else:
            counts, bin_edges = np.histogram(vals, bins=hist_bins, range=(vmin, vmax))
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            colors = mappable.to_rgba(centers)
            hist_ax.bar(
                centers,
                counts,
                width=np.diff(bin_edges),
                color=colors,
                align="center",
            )
            hist_ax.set_xlim(vmin, vmax)
            hist_ax.set_ylim(0, counts.max() * 1.05)
            hist_ax.set_axis_off()

    cbar_ax = fig.add_axes((cbar_x, cbar_y, cbar_width, cbar_height))
    cbar = fig.colorbar(
        mappable,
        cax=cbar_ax,
        orientation="horizontal",
        extend=_colorbar_extend(values, vmin, vmax),
    )
    if label:
        cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    return cbar


def _flatten_values(values: xr.DataArray | NDArray | pd.Series | None) -> NDArray:
    """flatten values into a 1D array of finite floats"""
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def _colorbar_extend(
    values: xr.DataArray | NDArray | None,
    vmin: float,
    vmax: float,
) -> str:
    """
    Determine whether the colorbar should have triangles at its ends to show that the
    color limits don't cover the full range of the data.
    """
    if values is None:
        return "neither"
    vals = _flatten_values(values)
    if vals.size == 0:
        return "neither"
    below = bool(vals.min() < vmin)
    above = bool(vals.max() > vmax)
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


###
###
# Maps
###
###


def _region_aspect(region: tuple[float, float, float, float] | None) -> float:
    """width / height of a region, defaulting to 1"""
    if region is None:
        return 1.0
    height = abs(region[3] - region[2])
    width = abs(region[1] - region[0])
    return width / height if height != 0 else 1.0


def map_axes(
    nrows: int = 1,
    ncols: int = 1,
    region: tuple[float, float, float, float] | None = None,
    fig_height: float = DEFAULT_FIG_HEIGHT,
    **kwargs: typing.Any,
) -> tuple[mpl.figure.Figure, NDArray]:
    """
    Create a grid of axes sized to the aspect ratio of a region, with room below each
    for a colorbar and histogram.

    Parameters
    ----------
    nrows, ncols : int, optional
        number of rows and columns of axes, by default 1
    region : tuple[float, float, float, float] | None, optional
        region the maps will show, used to set the aspect ratio, by default None
    fig_height : float, optional
        height of each map in inches, by default 4.75
    **kwargs : typing.Any
        additional keyword arguments passed to :func:`matplotlib.pyplot.subplots`

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        the figure and a flattened array of its axes
    """
    aspect = _region_aspect(region)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_height * aspect * ncols, fig_height * nrows),
        squeeze=False,
        **kwargs,
    )
    # leave room below each map for the colorbar and histogram
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    return fig, axes.ravel()


def _setup_axes(
    ax: mpl.axes.Axes | None,
    fig_height: float,
    region: tuple[float, float, float, float] | None,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """create a figure and axes sized to the aspect ratio of the region, if needed"""
    if ax is not None:
        return ax.get_figure(), ax

    fig, ax = plt.subplots(figsize=(fig_height * _region_aspect(region), fig_height))
    return fig, ax


def _format_map_axes(
    ax: mpl.axes.Axes,
    region: tuple[float, float, float, float],
    title: str | None = None,
    frame: bool = True,
) -> None:
    """apply the shared map axes styling"""
    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title, fontsize=12)
    if frame:
        ax.ticklabel_format(style="sci", scilimits=(-3, 4), useMathText=True)
        ax.tick_params(labelsize=8, direction="in", top=True, right=True)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def plot_grid(
    grid: xr.DataArray,
    ax: mpl.axes.Axes | None = None,
    region: tuple[float, float, float, float] | None = None,
    cmap: str | mpl.colors.Colormap = "viridis",
    reverse_cmap: bool = False,
    robust: bool = False,
    robust_percentiles: tuple[float, float] = (0.02, 0.98),
    cpt_lims: tuple[float, float] | None = None,
    absolute: bool = False,
    title: str | None = None,
    cbar_label: str | None = None,
    colorbar: bool = True,
    hist: bool = False,
    scalebar: bool = True,
    coast: bool = False,
    inset: bool = False,
    points: pd.DataFrame | None = None,
    points_style: str | None = None,
    points_label: str | None = None,
    epsg: str | None = None,
    fig_height: float = DEFAULT_FIG_HEIGHT,
    frame: bool = True,
    shading: bool = False,
    **kwargs: typing.Any,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot a grid as a map, with a colorbar, and optionally a colorbar histogram, a
    scalebar, coastlines, an inset locator map and point data.

    Parameters
    ----------
    grid : xarray.DataArray
        the 2D grid to plot
    ax : matplotlib.axes.Axes | None, optional
        axes to plot into, by default None which creates a new figure
    region : tuple[float, float, float, float] | None, optional
        region to clip the grid to before plotting, in format (xmin, xmax, ymin, ymax),
        by default None which uses the grid's extent
    cmap : str | matplotlib.colors.Colormap, optional
        colormap to use, by default "viridis". GMT-style names such as ``rain`` or
        ``balance+h0`` are translated to their Matplotlib equivalents
    reverse_cmap : bool, optional
        reverse the colormap, by default False
    robust : bool, optional
        use the 2nd and 98th percentiles of the data for the color limits instead of
        the min and max, by default False
    robust_percentiles : tuple[float, float], optional
        percentiles to use if ``robust`` is True, by default (0.02, 0.98)
    cpt_lims : tuple[float, float] | None, optional
        explicit color limits, by default None
    absolute : bool, optional
        make the color limits symmetric about zero, by default False
    title : str | None, optional
        title for the map, by default None
    cbar_label : str | None, optional
        label for the colorbar, by default None
    colorbar : bool, optional
        add a colorbar, by default True
    hist : bool, optional
        add a histogram of the grid's values above the colorbar, by default False
    scalebar : bool, optional
        add a scalebar, by default True
    coast : bool, optional
        add coastlines, by default False
    inset : bool, optional
        add an inset locator map, by default False
    points : pandas.DataFrame | None, optional
        point data to plot on the map, by default None
    points_style : str | None, optional
        GMT-style point style string, e.g. ``x.3c``, by default None
    points_label : str | None, optional
        label for the points, by default None
    epsg : str | None, optional
        EPSG code of the grid's projection, used for coastlines and the inset map, by
        default None
    fig_height : float, optional
        height of the figure in inches, by default 4.75
    frame : bool, optional
        show the axes frame with tick labels, by default True
    shading : bool, optional
        add hillshading to the grid, by default False
    **kwargs : typing.Any
        additional keyword arguments passed to
        :meth:`matplotlib.axes.Axes.pcolormesh`

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        the figure and axes of the map
    """
    if region is not None:
        grid = subset_grid(grid, region)
    else:
        region = get_grid_info(grid)[1]

    colormap, hinge = _parse_cmap(cmap, reverse=reverse_cmap)
    absolute = absolute or hinge

    if cpt_lims is None:
        vmin, vmax = get_min_max(
            grid,
            robust=robust,
            absolute=absolute,
            robust_percentiles=robust_percentiles,
        )
    else:
        vmin, vmax = cpt_lims

    if not np.isfinite([vmin, vmax]).all():
        logger.warning("grid has no finite values, skipping the color limits")
        vmin, vmax = None, None  # type: ignore[assignment]

    fig, ax = _setup_axes(ax, fig_height, region)

    northing_name, easting_name = (str(d) for d in grid.dims)
    mappable = ax.pcolormesh(
        grid[easting_name],
        grid[northing_name],
        grid,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
        rasterized=True,
        **kwargs,
    )

    if shading:
        _add_shading(ax, grid, easting_name, northing_name)

    _format_map_axes(ax, region, title=title, frame=frame)

    if coast:
        add_coast(ax, epsg=epsg)

    if points is not None:
        add_points(ax, points, style=points_style, label=points_label)
        if points_label is not None:
            ax.legend(loc="upper right", fontsize=8)

    if scalebar:
        add_scalebar(ax)

    if inset:
        add_inset(ax, region, epsg=epsg)

    if colorbar:
        add_colorbar(
            fig,
            ax,
            mappable,
            values=grid,
            label=cbar_label,
            hist=hist,
        )

    return fig, ax


def _add_shading(
    ax: mpl.axes.Axes,
    grid: xr.DataArray,
    easting_name: str,
    northing_name: str,
) -> None:
    """overlay a semi-transparent hillshade of the grid"""
    values = np.asarray(grid.values, dtype=float)
    if not np.isfinite(values).any():
        return
    light = mpl.colors.LightSource(azdeg=315, altdeg=45)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        shade = light.hillshade(np.nan_to_num(values, nan=float(np.nanmean(values))))
    ax.pcolormesh(
        grid[easting_name],
        grid[northing_name],
        shade,
        cmap="gray",
        alpha=0.3,
        shading="nearest",
        rasterized=True,
        zorder=2,
    )


def basemap(
    region: tuple[float, float, float, float],
    ax: mpl.axes.Axes | None = None,
    title: str | None = None,
    coast: bool = False,
    inset: bool = False,
    scalebar: bool = True,
    epsg: str | None = None,
    fig_height: float = DEFAULT_FIG_HEIGHT,
    frame: bool = True,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Create an empty map of a region, to plot points or other features onto.

    Parameters
    ----------
    region : tuple[float, float, float, float]
        region of the map, in format (xmin, xmax, ymin, ymax)
    ax : matplotlib.axes.Axes | None, optional
        axes to plot into, by default None which creates a new figure
    title : str | None, optional
        title for the map, by default None
    coast : bool, optional
        add coastlines, by default False
    inset : bool, optional
        add an inset locator map, by default False
    scalebar : bool, optional
        add a scalebar, by default True
    epsg : str | None, optional
        EPSG code of the map's projection, by default None
    fig_height : float, optional
        height of the figure in inches, by default 4.75
    frame : bool, optional
        show the axes frame with tick labels, by default True

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        the figure and axes of the map
    """
    fig, ax = _setup_axes(ax, fig_height, region)

    _format_map_axes(ax, region, title=title, frame=frame)

    if coast:
        add_coast(ax, epsg=epsg)
    if scalebar:
        add_scalebar(ax)
    if inset:
        add_inset(ax, region, epsg=epsg)

    return fig, ax


def subplots(
    grids: list[xr.DataArray] | xr.DataArray,
    dims: tuple[int, int] | None = None,
    region: tuple[float, float, float, float] | None = None,
    fig_title: str | None = None,
    titles: list[str] | None = None,
    cmap: str | mpl.colors.Colormap = "viridis",
    cmaps: list[str | mpl.colors.Colormap] | None = None,
    cpt_limits: list[tuple[float, float] | None] | None = None,
    cbar_label: str | None = None,
    cbar_labels: list[str] | None = None,
    subplot_labels: bool = True,
    fig_height: float = DEFAULT_FIG_HEIGHT,
    **kwargs: typing.Any,
) -> mpl.figure.Figure:
    """
    Plot a series of grids as subplots, each with its own colorbar.

    Parameters
    ----------
    grids : list[xarray.DataArray] | xarray.DataArray
        grids to plot
    dims : tuple[int, int] | None, optional
        (number of rows, number of columns), by default None which arranges the
        subplots as close to a square as possible
    region : tuple[float, float, float, float] | None, optional
        region to clip the grids to, by default None
    fig_title : str | None, optional
        title for the whole figure, by default None
    titles : list[str] | None, optional
        title for each subplot, by default None
    cmap : str | matplotlib.colors.Colormap, optional
        colormap to use for all grids, by default "viridis"
    cmaps : list | None, optional
        colormap to use for each grid, overrides ``cmap``, by default None
    cpt_limits : list | None, optional
        color limits for each grid, by default None
    cbar_label : str | None, optional
        colorbar label to use for all grids, by default None
    cbar_labels : list[str] | None, optional
        colorbar label for each grid, overrides ``cbar_label``, by default None
    subplot_labels : bool, optional
        add subplot labels (a, b, c ...), by default True
    fig_height : float, optional
        height of each subplot in inches, by default 4.75
    **kwargs : typing.Any
        additional keyword arguments passed to :func:`plot_grid`

    Returns
    -------
    matplotlib.figure.Figure
        the figure
    """
    if isinstance(grids, xr.DataArray):
        grids = [grids]

    for name, values in (
        ("titles", titles),
        ("cmaps", cmaps),
        ("cpt_limits", cpt_limits),
        ("cbar_labels", cbar_labels),
    ):
        if values is not None and len(values) != len(grids):
            msg = f"length of `{name}` must match the number of grids"
            raise ValueError(msg)

    nrows, ncols = square_subplots(len(grids)) if dims is None else dims

    plot_region = get_grid_info(grids[0])[1] if region is None else region

    fig, flat_axes = map_axes(
        nrows=nrows,
        ncols=ncols,
        region=plot_region,
        fig_height=fig_height,
    )

    for i, grid in enumerate(grids):
        plot_grid(
            grid,
            ax=flat_axes[i],
            region=region,
            cmap=cmap if cmaps is None else cmaps[i],
            cpt_lims=None if cpt_limits is None else cpt_limits[i],
            title=None if titles is None else titles[i],
            cbar_label=cbar_label if cbar_labels is None else cbar_labels[i],
            fig_height=fig_height,
            **kwargs,
        )
        if subplot_labels:
            flat_axes[i].annotate(
                f"{string.ascii_lowercase[i % 26]})",
                xy=(0.02, 0.98),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
                zorder=10,
            )

    # remove any unused axes
    for ax in flat_axes[len(grids) :]:
        ax.remove()

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16, fontweight="bold")

    return fig


def grid_compare(
    grid1: xr.DataArray,
    grid2: xr.DataArray,
    plot: bool = True,
    region: tuple[float, float, float, float] | None = None,
    grid1_name: str = "grid 1",
    grid2_name: str = "grid 2",
    title: str | None = "difference",
    rmse_in_title: bool = True,
    cmap: str | mpl.colors.Colormap = "viridis",
    reverse_cmap: bool = False,
    diff_cmap: str | mpl.colors.Colormap = "RdBu_r",
    robust: bool = True,
    hist: bool = True,
    cbar_label: str | None = None,
    fig_height: float = DEFAULT_FIG_HEIGHT,
    **kwargs: typing.Any,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Compare two grids by plotting them side by side with their difference, and return
    the difference along with the two (optionally clipped) grids.

    Parameters
    ----------
    grid1, grid2 : xarray.DataArray
        the grids to compare
    plot : bool, optional
        plot the grids, by default True
    region : tuple[float, float, float, float] | None, optional
        region to clip the grids to, by default None
    grid1_name, grid2_name : str, optional
        titles for the two grids
    title : str | None, optional
        title for the difference grid, by default "difference"
    rmse_in_title : bool, optional
        add the RMSE of the difference to its title, by default True
    cmap : str | matplotlib.colors.Colormap, optional
        colormap to use for the two grids, by default "viridis"
    reverse_cmap : bool, optional
        reverse the colormap of the two grids, by default False
    diff_cmap : str | matplotlib.colors.Colormap, optional
        colormap to use for the difference grid, by default "RdBu_r"
    robust : bool, optional
        use the 2nd and 98th percentiles for the color limits, by default True
    hist : bool, optional
        add histograms to the colorbars, by default True
    cbar_label : str | None, optional
        label for the colorbars, by default None
    fig_height : float, optional
        height of each subplot in inches, by default 4.75
    **kwargs : typing.Any
        additional keyword arguments passed to :func:`plot_grid`

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray, xarray.DataArray]
        the difference grid and the two grids
    """
    if region is not None:
        grid1 = subset_grid(grid1, region)
        grid2 = subset_grid(grid2, region)

    with xr.set_options(keep_attrs=True):
        dif = grid1 - grid2

    if not plot:
        return dif, grid1, grid2

    diff_title = title
    if diff_title is not None and rmse_in_title:
        rmse = float(np.sqrt(np.nanmean(np.asarray(dif, dtype=float) ** 2)))
        diff_title = f"{title}: RMSE={round(rmse, 2)}"

    plot_region = get_grid_info(grid1)[1] if region is None else region

    _, axes = map_axes(
        nrows=1,
        ncols=3,
        region=plot_region,
        fig_height=fig_height,
    )

    # use shared color limits for the two grids so they can be compared directly
    shared_lims = get_combined_min_max([grid1, grid2], robust=robust)

    plot_grid(
        grid1,
        ax=axes[0],
        title=grid1_name,
        cmap=cmap,
        reverse_cmap=reverse_cmap,
        cpt_lims=shared_lims,
        hist=hist,
        cbar_label=cbar_label,
        fig_height=fig_height,
        **kwargs,
    )
    plot_grid(
        dif,
        ax=axes[1],
        title=diff_title,
        cmap=diff_cmap,
        absolute=True,
        robust=robust,
        hist=hist,
        cbar_label=cbar_label,
        fig_height=fig_height,
        **kwargs,
    )
    plot_grid(
        grid2,
        ax=axes[2],
        title=grid2_name,
        cmap=cmap,
        reverse_cmap=reverse_cmap,
        cpt_lims=shared_lims,
        hist=hist,
        cbar_label=cbar_label,
        fig_height=fig_height,
        **kwargs,
    )

    return dif, grid1, grid2


def plot_profile(
    start: tuple[float, float],
    stop: tuple[float, float],
    data: dict[str, tuple[xr.DataArray, str]] | None = None,
    layers: dict[str, tuple[xr.DataArray, str]] | None = None,
    num_points: int = 200,
    data_label: str | None = None,
    layers_label: str | None = None,
    fig_width: float = 8,
    fig_height: float = 6,
) -> tuple[mpl.figure.Figure, NDArray]:
    """
    Plot a cross-section along a profile between two points, with an optional panel of
    data values above a panel of layer surfaces.

    Parameters
    ----------
    start, stop : tuple[float, float]
        the start and stop coordinates of the profile, in format (easting, northing)
    data : dict[str, tuple[xarray.DataArray, str]] | None, optional
        grids to sample and plot in the upper panel, as ``{name: (grid, color)}``, by
        default None
    layers : dict[str, tuple[xarray.DataArray, str]] | None, optional
        grids to sample and plot in the lower panel, as ``{name: (grid, color)}``, by
        default None
    num_points : int, optional
        number of points to sample along the profile, by default 200
    data_label : str | None, optional
        y axis label for the data panel, by default None
    layers_label : str | None, optional
        y axis label for the layers panel, by default None
    fig_width, fig_height : float, optional
        size of the figure in inches

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        the figure and its axes
    """
    easting = np.linspace(start[0], stop[0], num_points)
    northing = np.linspace(start[1], stop[1], num_points)
    distance = np.hypot(easting - start[0], northing - start[1])

    panels = [p for p in (data, layers) if p]
    if not panels:
        msg = "at least one of `data` or `layers` must be supplied"
        raise ValueError(msg)

    fig, axes = plt.subplots(
        nrows=len(panels),
        ncols=1,
        figsize=(fig_width, fig_height),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for ax, panel, label in zip(
        flat_axes,
        panels,
        [data_label, layers_label] if data else [layers_label],
        strict=False,
    ):
        for name, (grid, color) in panel.items():
            northing_name, easting_name = (str(d) for d in grid.dims)
            sampled = grid.interp(
                {
                    easting_name: xr.DataArray(easting, dims="profile"),
                    northing_name: xr.DataArray(northing, dims="profile"),
                },
                method="linear",
            )
            ax.plot(distance, np.asarray(sampled), color=color, label=name)
        ax.legend(loc="best", fontsize=8)
        if label:
            ax.set_ylabel(label)
        ax.grid(visible=True, alpha=0.3)

    flat_axes[-1].set_xlabel("Distance along profile (m)")
    fig.tight_layout()

    return fig, flat_axes
