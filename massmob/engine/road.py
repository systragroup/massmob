import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
from tqdm import tqdm


def split_line_into_segments(line: LineString, max_length: float) -> list[LineString]:
    """
    Split a LineString into multiple segments, each <= max_length.
    Vectorized with numpy for speed.
    """
    coords = np.array(line.coords)
    seg_lengths = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]

    # If the line is already short, return as is
    if total_length <= max_length:
        return [line]

    # Determine split points along the line
    n_splits = int(np.ceil(total_length / max_length))
    split_positions = np.linspace(0, total_length, n_splits + 1)

    segments = []
    start_idx = 0

    for i in range(1, len(split_positions)):
        target = split_positions[i]

        # Find index where cumulative length exceeds target
        idx = np.searchsorted(cum_lengths, target, side='right') - 1
        if idx >= len(coords) - 1:
            idx = len(coords) - 2  # last segment

        prev_len = cum_lengths[idx]
        next_len = cum_lengths[idx + 1]
        ratio = (target - prev_len) / (next_len - prev_len) if next_len > prev_len else 0

        x = coords[idx, 0] + ratio * (coords[idx + 1, 0] - coords[idx, 0])
        y = coords[idx, 1] + ratio * (coords[idx + 1, 1] - coords[idx, 1])
        split_point = (x, y)

        seg_coords = [tuple(c) for c in coords[start_idx:idx + 1]] + [split_point]
        segments.append(LineString(seg_coords))
        start_idx = idx
        coords[idx] = split_point  # update start for next segment

    # Add the last segment
    if start_idx < len(coords) - 1:
        segments.append(LineString(coords[start_idx:]))

    return segments


def multi_split_vectorized(gdf: gpd.GeoDataFrame, max_length: float) -> gpd.GeoDataFrame:
    """Split all long lines into segments <= max_length in a single pass."""
    temp = gdf.copy()
    temp["geometry"] = temp.geometry.apply(lambda geom: split_line_into_segments(geom, max_length))
    temp = gdf.explode("geometry", index_parts=False)
    gdf = gpd.GeoDataFrame(temp, geometry="geometry", crs=gdf.crs)
    gdf["length"] = gdf.geometry.length
    return gdf


def get_intersections_gdf(gdf: gpd.GeoDataFrame) -> set:
    """Return coordinates appearing in more than one feature."""
    coords = np.concatenate([np.array(g.coords) for g in gdf.geometry])
    coords_tuples = [tuple(pt) for pt in coords]
    counts = {}
    for pt in coords_tuples:
        counts[pt] = counts.get(pt, 0) + 1
    return {pt for pt, c in counts.items() if c > 1}


def get_nodes_gdf(gdf: gpd.GeoDataFrame) -> set:
    """Return start, end, and intersection node coordinates."""
    start_nodes = [tuple(g.coords[0]) for g in gdf.geometry]
    end_nodes = [tuple(g.coords[-1]) for g in gdf.geometry]
    intersections = get_intersections_gdf(gdf)
    return set(start_nodes) | set(end_nodes) | intersections


def split_links_vectorized(
    links: gpd.GeoDataFrame,
    max_length: float = 100,
    suffix: str = "network",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Split long road links into multiple segments <= max_length in a single pass
    and build node GeoDataFrame.
    """
    # --- Validate CRS ---
    assert links.crs is not None, "CRS must be set (in meters)."
    assert links.crs.to_epsg() not in [3857, 4326], "CRS must be projected in meters."

    crs = links.crs
    links = links.copy()
    links["length"] = links.geometry.length

    # --- Split long lines ---
    split_gdf = multi_split_vectorized(links, max_length)
    split_gdf["length"] = split_gdf.geometry.length

    # --- Build nodes ---
    node_coords = list(get_nodes_gdf(split_gdf))
    node_index = dict(zip(node_coords, [f"{suffix}_node_{i}" for i in range(len(node_coords))]))

    nodes_df = gpd.GeoDataFrame(
        [{"index": idx, "geometry": Point(coord)} for coord, idx in node_index.items()],
        geometry="geometry",
        crs=crs
    ).set_index("index")

    # --- Add node references to links ---
    split_gdf["a"] = split_gdf.geometry.apply(lambda geom: node_index[tuple(geom.coords[0])])
    split_gdf["b"] = split_gdf.geometry.apply(lambda geom: node_index[tuple(geom.coords[-1])])
    split_gdf.index = [f"{suffix}_link_{i}" for i in range(len(split_gdf))]

    return split_gdf, nodes_df
