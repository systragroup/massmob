import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import transform
import requests
from typing import List
import time

ENDPOINTS: List[str] = [
    "https://overpass-api.de/api/interpreter",
    # "https://overpass.kumi.systems/api/interpreter",
    # "https://overpass.openstreetmap.ru/api/interpreter",
    # "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

def is_html_error(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "")
    text = resp.text[:300].lower()
    return ("text/html" in ctype) or ("<html" in text) or ("osm3s response" in text)

def fetch_with_retry(
    query: str,
    endpoints: List[str]=ENDPOINTS,
    max_retries_per_endpoint: int = 6,
    initial_backoff: float = 4.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 30.0,
    http_timeout: int = 60,
):
    for ep in endpoints:
        print(f"[INFO] Endpoint: {ep}")
        backoff = initial_backoff
        for attempt in range(1, max_retries_per_endpoint + 1):
            print(f"[TRY] Tentative {attempt}/{max_retries_per_endpoint}")
            try:
                resp = requests.post(ep, data=query, timeout=http_timeout, verify=False)
                if resp.status_code == 200 and not is_html_error(resp):
                    print("[OK] Succès.")
                    return resp.json()
                else:
                    print(f"[FAIL] Échec (HTTP {resp.status_code}).")
                    # print(resp.text)
            except requests.Timeout:
                print(f"[FAIL] Timeout après {http_timeout}s.")
            except requests.RequestException as e:
                print("[FAIL] Erreur réseau.")
                # print(e)
            if attempt < max_retries_per_endpoint:
                print(f"[WAIT] Attente {backoff}s avant retry…")
                time.sleep(min(backoff, max_backoff))
                backoff *= backoff_factor

        print("[INFO] Changement de miroir…")

    raise RuntimeError("Tous les endpoints ont échoué après plusieurs tentatives.")


def download_from_overpass(
    network: str,
    bbox: tuple,
    # overpass_url: str = "https://overpass-api.de/api/interpreter",
):
    """
    Returns links and nodes un gpd.GeoDataFrame epsg 4326 for the requested network
    network: str
        choose in 'road', 'rail', 'walk'
    bbox: tuple
        from previous selection leafmap (epsg 4326), format (minlon, minlat, maxlon, maxlat)
    """
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"       # minlat,minlon,maxlat,maxlon = (south,west,north,east) pour Overpass
    def query(network):
        if network == 'rail':
            return f'(\
                way["railway"~"^(rail|tram|subway|light_rail|monorail|funicular|narrow_gauge)$"]({bbox_str});\
                relation["railway"~"^(rail|tram|subway|light_rail|monorail|funicular|narrow_gauge)$"]({bbox_str});\
                    )'
        elif network == 'road':
            return f'(way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|service|road)$"]({bbox_str});)'
        elif network == 'walk':
            return f'(\
                way["highway"~"^(footway|path|pedestrian|track|steps)$"]["foot"!~"no"]({bbox_str});\
                way["highway"~"^(footway|pedestrian|steps|cycleway)$"]({bbox_str});\
                way["highway"="path"]["bicycle"~"designated|yes"]({bbox_str});\
                way["highway"="path"]["foot"~"designated|yes"]({bbox_str});\
                    )'
        else:
            return 'network type not available'
    overpass_query = f"""
        [out:json][timeout:60];
        {query(network)};
        out body;
        >;
        out skel qt;
        """

    # Request data
    data = fetch_with_retry(overpass_query)

    # Nodes to dict
    nodes_dict = {el['id']: (el['lon'], el['lat']) for el in data['elements'] if el['type'] == 'node'}

    # Links to geodataframe
    link_data = []
    for el in data['elements']:
        if el['type'] == 'way' and 'nodes' in el:
            node_ids = [nid for nid in el['nodes'] if nid in nodes_dict]
            coords = [nodes_dict[nid] for nid in node_ids]
            if len(coords) >= 2:
                feature = {
                    'geometry': LineString(coords),
                    'a': node_ids[0],              # noeud de début
                    'b': node_ids[-1],             # noeud de fin
                    'osmid': el['id'],
                    **el.get('tags', {})           # ajoute tous les tags OSM comme colonnes
                }
                link_data.append(feature)
    links = gpd.GeoDataFrame(link_data, geometry="geometry")

    # return links, nodes
    return links

def get_quais_from_overpass(bbox):
    overpass_query = f"""
    [out:json][timeout:60];
    (
    node["railway"="platform"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    way["railway"="platform"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    relation["railway"="platform"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    );
    out geom;
    """

    res = fetch_with_retry(overpass_query)

    features = []
    for element in res['elements']:
        if element['type'] == 'node':
            geom = Point(element['lon'], element['lat'])
            features.append({'osmid': element['id'], 'geometry': geom, **element.get('tags', {})})
        elif element['type'] == 'way' and 'geometry' in element:
            coords = [(pt['lon'], pt['lat']) for pt in element['geometry']]
            if coords[0] == coords[-1] and len(coords) > 3:
                geom = Polygon(coords)
            else:
                geom = LineString(coords)
            features.append({'osmid': element['id'], 'geometry': geom, **element.get('tags', {})})
        elif element['type'] == 'relation' and 'members' in element:
            # On reconstruit chaque "outer" de la relation
            outers = []
            for member in element['members']:
                if member.get('role', '') == 'outer' and 'geometry' in member:
                    coords = [(pt['lon'], pt['lat']) for pt in member['geometry']]
                    if coords[0] == coords[-1] and len(coords) > 3:
                        outers.append(Polygon(coords))
                    else:
                        outers.append(LineString(coords))
            # Si plusieurs outers, MultiPolygon
            if len(outers) == 1:
                geom = outers[0]
            elif len(outers) > 1:
                # MultiPolygon for outers that are Polygons, else MultiLineString
                if all(isinstance(outer, Polygon) for outer in outers):
                    geom = MultiPolygon(outers)
                else:
                    geom = outers  # List of LineString
            else:
                geom = None  # Pas de géométrie
            features.append({'osmid': element['id'], 'geometry': geom, **element.get('tags', {})})

    gdf_quais_polygon = gpd.GeoDataFrame(features, geometry='geometry', crs='EPSG:4326')
    gdf_quais_polygon = gdf_quais_polygon[gdf_quais_polygon.geometry != None][['osmid', 'geometry', 'railway', 'name']]
    gdf_quais_polygon = gdf_quais_polygon.loc[gdf_quais_polygon.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    gdf_quais_polygon = gdf_quais_polygon.reset_index()

    # Appliquer la fonction à toutes les géométries et "exploser" les listes
    gdf_quais_polygon['lines'] = gdf_quais_polygon.geometry.apply(_polygon_to_lines)
    gdf_quais = gdf_quais_polygon.explode('lines')

    # Remplacer la géométrie par les LineStrings
    gdf_quais = gdf_quais.set_geometry('lines').drop(columns='geometry')
    gdf_quais = gdf_quais.rename_geometry('geometry')

    # Le GeoDataFrame résultant contient toutes les lignes
    gdf_quais.crs = gdf_quais_polygon.crs

    return gdf_quais

def _polygon_to_lines(geom):
    """
    Renvoie une liste de LineString pour :
    - un Polygon (extérieur + intérieurs),
    - un MultiPolygon (tous les polygones qu'il contient).
    """
    lines = []

    if isinstance(geom, Polygon):
        lines.append(LineString(geom.exterior.coords))
        lines.extend(LineString(ring.coords) for ring in geom.interiors)

    elif isinstance(geom, MultiPolygon):
        # Appel récursif pour chaque polygone du multipolygone
        for poly in geom.geoms:
            lines.extend(_polygon_to_lines(poly))

    else:
        raise TypeError("L'objet fourni doit être un Polygon ou un MultiPolygon")

    return lines
