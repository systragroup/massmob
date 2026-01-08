import pandas as pd
import numpy as np
from massmob.engine import road
from quetzal.engine import engine
from quetzal.io import road as road_io


def classify_nodes_by_network_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate link-level network types per node and produce a node-level label.

    This function expects a link DataFrame with columns 'a', 'b', and 'network_type'.
    It stacks endpoints 'a' and 'b', factors them, and computes, for each node,
    the set of distinct network types connected to that node, returning a label
    like "rail-road-walk" (sorted alphabetical order).

    Parameters
    ----------
    df : pd.DataFrame
        Link-level DataFrame with columns:
        - a, b : node identifiers for link endpoints
        - network_type : string identifying the network type for the link

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by node ID with one column:
        - network_type_label : concatenation of sorted unique network types for that node
    """
    nodes = np.concatenate([df["a"].to_numpy(), df["b"].to_numpy()])
    types = np.concatenate([df["network_type"].to_numpy(), df["network_type"].to_numpy()])

    node_codes, node_uniques = pd.factorize(nodes)
    type_codes, type_uniques = pd.factorize(types)

    order = np.argsort(node_codes)
    sorted_nodes = node_codes[order]
    sorted_types = type_codes[order]

    boundaries = np.flatnonzero(np.r_[True, sorted_nodes[1:] != sorted_nodes[:-1], True])

    node_type_lists = [
        np.unique(sorted_types[boundaries[i]:boundaries[i + 1]])
        for i in range(len(boundaries) - 1)
    ]

    labels = [
        "-".join(sorted(type_uniques[np.array(t)]))
        for t in node_type_lists
    ]

    result = pd.DataFrame({"node": node_uniques, "network_type_label": labels}).set_index("node")
    return result


def process_networks(
    osm_road_links: pd.DataFrame,
    osm_walk_links: pd.DataFrame | None = None,
    osm_rail_links: pd.DataFrame | None = None,
    osm_quais: pd.DataFrame | None = None,
    epsg: int = 2154,
    disagg_buffer_platform_m: float = 30.0,
    disagg_max_distance_walk_m: float = 20.0,
    disagg_max_distance_rail_m: float = 20.0,
    walk_to_platform_max_m: float = 5.0,
    rail_to_platform_max_m: float = 15.0,
    road_to_walk_max_m: float = 1.0,
):
    """
    Transform pre-downloaded OSM layers into processed networks with connectors and labels.

    This function assumes the OSM layers (road, walk, rail, platforms) were already
    downloaded and passed as GeoDataFrames. It standardizes columns/CRS, derives nodes,
    disaggregates around platforms, builds connectors (road-walk, rail/walk-to-platform),
    merges all layers, and classifies node types.

    Parameters
    ----------
    osm_road_links : pd.DataFrame
        Raw road links GeoDataFrame with typical OSM columns including geometry.
    osm_walk_links : pd.DataFrame, optional
        Raw walk links GeoDataFrame. If None, walk network is omitted.
    osm_rail_links : pd.DataFrame, optional
        Raw rail links GeoDataFrame. If None, rail network is omitted.
    osm_quais : pd.DataFrame, optional
        Raw platform geometries GeoDataFrame. Required if rail processing is desired.
    epsg : int, optional
        Target CRS EPSG code for processing. Default 2154.
    disagg_buffer_platform_m : float, optional
        Buffer distance around platforms used to disaggregate walk/rail networks.
    disagg_max_distance_walk_m : float, optional
        Max split distance for walk network disaggregation.
    disagg_max_distance_rail_m : float, optional
        Max split distance for rail network disaggregation.
    walk_to_platform_max_m : float, optional
        Maximum connector length for walk-to-platform legs.
    rail_to_platform_max_m : float, optional
        Maximum connector length for rail-to-platform legs.
    road_to_walk_max_m : float, optional
        Maximum connector length for road-to-walk legs.

    Returns
    -------
    tuple
        (network_links, network_nodes), where:
        - network_links: merged links GeoDataFrame with 'network_type', 'length', etc.
        - network_nodes: merged nodes GeoDataFrame with 'node_type' label added.

    Notes
    -----
    - If rail is provided, platforms should also be provided for disaggregation and connectors.
    - Index uniqueness is enforced for both links and nodes; assertions will raise if duplicates remain.
    """
    # Standardize road links
    road_cols = ['a', 'b', 'osmid', 'name', 'highway', 'lanes', 'oneway', 'maxspeed', 'geometry']
    for col in road_cols:
        if col not in osm_road_links.columns:
            osm_road_links[col] = None
    osm_road_links = osm_road_links[road_cols].set_index(['a', 'b'])
    osm_road_links = osm_road_links.set_crs(epsg=4326).to_crs(epsg=epsg)
    osm_road_links["network_type"] = "road"
    osm_road_links["oneway"] = osm_road_links["oneway"].replace({"yes": 1, "no": 0}).fillna(0)

    # Derive road nodes
    road_links, road_nodes = road_io.get_links_and_nodes_gdf(osm_road_links)
    road_links = road_links.set_crs(epsg=epsg, allow_override=True)
    road_links["length"] = road_links.length

    # Standardize walk links if provided
    walk_links = None
    walk_nodes = None
    if osm_walk_links is not None:
        walk_cols = ['a', 'b', 'osmid', 'name', 'highway', 'lanes', 'oneway', 'maxspeed', 'geometry']
        for col in walk_cols:
            if col not in osm_walk_links.columns:
                osm_walk_links[col] = None
        walk_links = osm_walk_links[walk_cols].set_index(['a', 'b'])
        walk_links = walk_links.set_crs(epsg=4326).to_crs(epsg=epsg)

        # Include walkable road types into walk network
        nowalk_types = ["trunk", "motorway", "motorway_link", "trunk_link"]
        walk_links = pd.concat([osm_road_links.loc[~osm_road_links.highway.isin(nowalk_types)], walk_links])
        walk_links["network_type"] = "walk"
        walk_links["oneway"] = 0

        walk_links, walk_nodes = road_io.get_links_and_nodes_gdf(walk_links, suffix="walk_")
        walk_links = walk_links.set_crs(epsg=epsg, allow_override=True)
        walk_links["length"] = walk_links.length

    # Standardize rail links and platforms if provided
    rail_links = None
    rail_nodes = None
    ql = None
    qn = None
    if osm_rail_links is not None:
        # Rail links
        osm_rail_links["lanes"] = 1  # Default to 1 lane if missing
        needed_rail_cols = ['a', 'b', 'osmid', 'name', 'railway', 'lanes', 'maxspeed', 'geometry']
        missing = [c for c in needed_rail_cols if c not in osm_rail_links.columns]
        if missing:
            # Fill missing typical columns with defaults
            for c in missing:
                osm_rail_links[c] = None
        rail_links = osm_rail_links[needed_rail_cols].set_index(['a', 'b'])
        rail_links = rail_links.rename(columns={'railway': 'highway'})
        rail_links["network_type"] = "rail"
        rail_links["oneway"] = 0
        rail_links = rail_links.set_crs(epsg=4326).to_crs(epsg=epsg)

        raill, railn = road_io.get_links_and_nodes_gdf(rail_links)
        raill = raill.set_crs(epsg=epsg, allow_override=True)
        raill["length"] = raill.length

        # Platforms (quais)
        if osm_quais is None:
            raise ValueError("Platforms (osm_quais) must be provided when rail links are processed.")
        gdf_quais = osm_quais.to_crs(epsg=epsg).copy()
        gdf_quais["oneway"] = 0
        ql, qn = road_io.get_links_and_nodes_gdf(gdf_quais)
        ql = ql.set_crs(epsg=epsg, allow_override=True)

        # Disaggregate networks around platforms
        platform_geoms = ql.buffer(disagg_buffer_platform_m).geometry.unary_union

        if walk_links is not None:
            walk_links, walk_nodes = road.disaggregate_network(
                walk_links,
                max_distance=disagg_max_distance_walk_m,
                geometry_filter=platform_geoms,
                suffix="walk"
            )

        rail_links, rail_nodes = road.disaggregate_network(
            raill,
            geometry_filter=platform_geoms,
            max_distance=disagg_max_distance_rail_m,
            suffix="rail"
        )

        # Disaggregate platform links too (drop index if present)
        ql_to_disagg = ql.drop(columns=[c for c in ["index"] if c in ql.columns])
        ql, qn = road.disaggregate_network(
            ql_to_disagg,
            max_distance=disagg_max_distance_rail_m,
            suffix='quais'
        )

        # Build connectors to platforms
        walk_to_quais = None
        rail_to_quais = None

        if walk_nodes is not None:
            walk_to_quais = engine.ntlegs_from_centroids_and_nodes(
                qn, walk_nodes, n_neighbors=2, coordinates_unit='meter'
            )
            walk_to_quais["length"] = walk_to_quais.geometry.apply(lambda x: x.length)
            walk_to_quais = walk_to_quais[walk_to_quais["length"] < walk_to_platform_max_m]
            walk_to_quais["highway"] = "walk_to_platform"

        rail_to_quais = engine.ntlegs_from_centroids_and_nodes(
            qn, rail_nodes, n_neighbors=2, coordinates_unit='meter'
        )
        rail_to_quais["length"] = rail_to_quais.geometry.apply(lambda x: x.length)
        rail_to_quais = rail_to_quais[rail_to_quais["length"] < rail_to_platform_max_m]
        rail_to_quais["highway"] = "rail_to_platform"

        # Label platforms and combine with connectors
        ql["highway"] = "platform"
        quais = pd.concat([x for x in [rail_to_quais, walk_to_quais, ql] if x is not None])
        quais.index = [f"quais_{x}" for x in range(len(quais))]
        quais["network_type"] = "quai"
    else:
        quais = None

    # Road-walk connectors (if walk is present)
    road_to_walk = None
    if (osm_walk_links is not None) and (walk_nodes is not None):
        road_to_walk = engine.ntlegs_from_centroids_and_nodes(
            road_nodes, walk_nodes, n_neighbors=1, coordinates_unit='meter'
        )
        road_to_walk["length"] = road_to_walk.geometry.apply(lambda x: x.length)
        road_to_walk = road_to_walk[road_to_walk["length"] < road_to_walk_max_m]
        road_to_walk["network_type"] = "rtw"
        road_to_walk.index = [f"rtw_link_{i}" for i in range(len(road_to_walk))]

    # Merge all links and nodes
    all_links = [road_links]
    all_nodes = [road_nodes]

    if walk_links is not None:
        all_links.append(walk_links)
        all_nodes.append(walk_nodes)
        if road_to_walk is not None:
            all_links.append(road_to_walk)

    if rail_links is not None and rail_nodes is not None:
        all_links.append(rail_links)
        all_nodes.append(rail_nodes)
        if quais is not None:
            all_links.append(quais)
        if qn is not None:
            all_nodes.append(qn)

    network_links = pd.concat(all_links)
    network_nodes = pd.concat(all_nodes)

    # Final attributes and cleanup
    network_links['length'] = network_links.geometry.length
    network_links = network_links.drop(columns='maxspeed', errors="ignore")

    # Ensure no duplicate indices
    assert len(network_nodes[network_nodes.index.duplicated()]) == 0, "Duplicate node indices detected."
    assert len(network_links[network_links.index.duplicated()]) == 0, "Duplicate link indices detected."

    # Node type labeling
    node_types = classify_nodes_by_network_type(network_links)
    network_nodes = network_nodes.join(
        node_types[["network_type_label"]].rename(columns={"network_type_label": "node_type"}),
        how='left'
    )

    return network_links, network_nodes