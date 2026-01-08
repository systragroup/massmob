import pandas as pd
import numpy as np
from massmob.io import osm as osm_io
from massmob.engine import road
from quetzal.engine import engine
from quetzal.io import road as road_io

# classify nodes
def classify_nodes_by_network_type(df):
    # On empile les colonnes a et b en vecteur numpy ultra-rapide
    nodes = np.concatenate([df["a"].to_numpy(), df["b"].to_numpy()])
    types = np.concatenate([df["network_type"].to_numpy(), df["network_type"].to_numpy()])

    # Factorisation très rapide (C-level)
    node_codes, node_uniques = pd.factorize(nodes)
    type_codes, type_uniques = pd.factorize(types)

    # On agrège les types par node_code via groupby numpy
    order = np.argsort(node_codes)
    sorted_nodes = node_codes[order]
    sorted_types = type_codes[order]

    # indices où un nouveau noeud commence
    boundaries = np.flatnonzero(np.r_[True, sorted_nodes[1:] != sorted_nodes[:-1], True])

    # pour chaque noeud : les types uniques (en numpy → ultra-rapide)
    node_type_lists = [
        np.unique(sorted_types[boundaries[i]:boundaries[i+1]])
        for i in range(len(boundaries) - 1)
    ]

    # conversion vers labels "rail/road/walk"
    labels = [
        "-".join(sorted(type_uniques[np.array(t)]))
        for t in node_type_lists
    ]

    # Construction du DataFrame résultat
    result = pd.DataFrame({
        "node": node_uniques,
        "network_type_label": labels
    }).set_index("node")

    return result

def download_networks(bbox, epsg=2154, download_rail=True, download_walk=True):
    
    if download_rail:
        assert download_walk==True, "To download rail network, walk network must be downloaded too."
    ### Prepare road / walk networks
    ## Download OSM data
    print("Downloading road network...")
    road_links = osm_io.download_from_overpass(network='road', bbox=bbox)

    # Reformat road_links
    columns = ['a', 'b', 'osmid', 'name', 'highway', 'lanes', 'oneway', 'maxspeed', 'geometry']
    for col in columns:
        if col not in road_links.columns:
            road_links[col] = None  # pour éviter une KeyError si la colonne n'existe pas
    road_links = road_links[columns].set_index(['a', 'b'])
    road_links = road_links.set_crs(epsg=4326).to_crs(epsg=epsg)
    road_links["network_type"] = "road"
    road_links["oneway"] = road_links["oneway"].replace({"yes": 1, "no": 0})
    road_links["oneway"] = road_links["oneway"].fillna(0)
    
    if download_walk:
        print("Downloading walk network...")
        walk_links = osm_io.download_from_overpass(network='walk', bbox=bbox)

        # Reformat walk_links
        columns = ['a', 'b', 'osmid', 'name', 'highway', 'lanes', 'oneway', 'maxspeed', 'geometry']
        for col in columns:
            if col not in walk_links.columns:
                walk_links[col] = None  # pour éviter une KeyError si la colonne n'existe pas
        walk_links = walk_links[columns].set_index(['a', 'b'])
        walk_links = walk_links.set_crs(epsg=4326).to_crs(epsg=epsg)

        nowalk_types = ["trunk", "motorway", "motorway_link", "trunk_link"]
        walk_links = pd.concat([road_links.loc[~road_links.highway.isin(nowalk_types)], walk_links])
        walk_links["network_type"] = "walk"
        walk_links["oneway"] = 0

        walk_links, walk_nodes = road_io.get_links_and_nodes_gdf(walk_links, suffix="walk_")
        walk_links = walk_links.set_crs(epsg=epsg, allow_override=True)
        walk_links["length"] = walk_links.length

    road_links, road_nodes = road_io.get_links_and_nodes_gdf(road_links)
    road_links = road_links.set_crs(epsg=epsg, allow_override=True)
    road_links["length"] = road_links.length

    ## Rail
    if download_rail:
        print("Downloading rail network...")
        rail_links = osm_io.download_from_overpass(network='rail', bbox=bbox)

        # Reformat rail_links
        rail_links['lanes'] = 1
        rail_links = rail_links[['a', 'b', 'osmid', 'name', 'railway', 'lanes', 'maxspeed', 'geometry']].set_index(['a', 'b'])
        rail_links = rail_links.rename(columns={'railway': 'highway'})
        rail_links["network_type"] = "rail"
        rail_links["oneway"] = 0 # à modifier ? 
        rail_links = rail_links.set_crs(epsg=4326).to_crs(epsg=epsg)

        raill, railn = road_io.get_links_and_nodes_gdf(rail_links)
        raill = raill.set_crs(epsg=epsg, allow_override=True)
        raill["length"] = raill.length

        # plateforms
        print("Downloading rail platforms...")
        gdf_quais = osm_io.get_quais_from_overpass(bbox)
        gdf_quais.to_crs(epsg=epsg, inplace=True)
        gdf_quais["oneway"] = 0  # pas de sens unique
        # to links and nodes
        ql, qn = road_io.get_links_and_nodes_gdf(gdf_quais)
        ql = ql.set_crs(epsg=epsg, allow_override=True)

        ### Disaggregate networks around rail platforms
        plateform_geoms = ql.buffer(30).geometry.unary_union
        walk_links, walk_nodes = road.disaggregate_network(walk_links, max_distance=20, geometry_filter=plateform_geoms, suffix="walk")  # walk
        rail_links, rail_nodes = road.disaggregate_network(raill, geometry_filter=plateform_geoms, max_distance=20, suffix="rail")  # rail
        ql, qn = road.disaggregate_network(ql.drop("index", axis=1), max_distance=20, suffix='quais')  # quais

        walk_to_quais = engine.ntlegs_from_centroids_and_nodes(
            qn,
            walk_nodes,
            n_neighbors=2,
            coordinates_unit='meter'
        )
        rail_to_quais = engine.ntlegs_from_centroids_and_nodes(
            qn,
            rail_nodes,
            n_neighbors=2,
            coordinates_unit='meter'
        )
        rail_to_quais["length"] = rail_to_quais.geometry.apply(lambda x: x.length)
        walk_to_quais["length"] = walk_to_quais.geometry.apply(lambda x: x.length)
        # TODO: améliorer avec Voronoï? pas d’urgence, ça n’est pas dimensionnant et pas forcément mieux
        # apply distance thresholds
        rail_to_quais = rail_to_quais[rail_to_quais["length"] < 15]
        walk_to_quais = walk_to_quais[walk_to_quais["length"] < 5]
    
    if download_walk:
        # road - walk connectors
        road_to_walk = engine.ntlegs_from_centroids_and_nodes(
            road_nodes,
            walk_nodes,
            n_neighbors=1,
            coordinates_unit='meter'
        )
        road_to_walk["length"] = road_to_walk.geometry.apply(lambda x: x.length)
        road_to_walk = road_to_walk[road_to_walk["length"] < 1]

        road_to_walk["network_type"] = "rtw"
        road_to_walk.index = [f"rtw_link_{i}" for i in range(len(road_to_walk))]

    if download_rail:
        # quais
        rail_to_quais["highway"] = "rail_to_platform"
        walk_to_quais["highway"] = "walk_to_platform"
        ql["highway"] = "platform"

        quais = pd.concat([rail_to_quais, walk_to_quais, ql])
        quais.index = [f"quais_{x}" for x in range(len(quais))]
        quais["network_type"] = "quai"

    # Merge
    all_links = [road_links]
    all_nodes =  [road_nodes]

    if download_walk:
        all_links += [walk_links, road_to_walk]
        all_nodes += [walk_nodes]

    if download_rail:
        all_links += [rail_links, quais]
        all_nodes += [rail_nodes, qn]
        
    network_links = pd.concat(all_links)
    network_nodes = pd.concat(all_nodes)

    network_links['length'] = network_links.geometry.length
    network_links = network_links.drop(columns='maxspeed', errors="ignore")

    assert len(network_nodes[network_nodes.index.duplicated()])==0
    assert len(network_links[network_links.index.duplicated()])==0

    node_types = classify_nodes_by_network_type(network_links)
    network_nodes = network_nodes.join(node_types[["network_type_label"]].rename(columns={"network_type_label": "node_type"}), how='left')

    return network_links, network_nodes