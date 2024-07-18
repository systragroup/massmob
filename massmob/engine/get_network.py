import osmnx as ox
ox.settings.requests_kwargs = {'verify': False}

def extract_road_network(studied_area):
    """
    Extract the road network from the studied area (road_links and road_nodes)
    :param studied_area: GeoDataFrame of the studied area
    :type studied_area: GeoDataFrame
    :return: road_links and road_nodes

    """
    zones = studied_area
    
    hull = zones.geometry.convex_hull[0]
    drive = ox.graph_from_polygon(hull, network_type='drive')
    road_nodes, road_links = ox.graph_to_gdfs(drive)

    return road_nodes, road_links


def network_processing(road_nodes, road_links):
    """
    Process the road network (road_links and road_nodes) ???
    :param road_nodes: GeoDataFrame of the road nodes
    :type road_nodes: GeoDataFrame
    :param road_links: GeoDataFrame of the road links
    :type road_links: GeoDataFrame
    :return: road_links and road_nodes

    
    """
    return