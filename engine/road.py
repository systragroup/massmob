import geopandas as gpd
import pandas as pd
from shapely.ops import transform
from shapely.geometry import LineString, Point
from tqdm import tqdm
from massmodel.engine import utils
import json


def split_line_exact_length_curvilinear(line):
    if not isinstance(line, LineString):
        raise ValueError("line input should be LineString")

    total_length = line.length
    target_length = total_length / 2.0

    current_length = 0.0
    for i in range(1, len(line.coords)):
        segment = LineString([line.coords[i-1], line.coords[i]])
        segment_length = segment.length

        if current_length + segment_length > target_length:
            # Trouver le point sur le segment où la division doit avoir lieu
            remaining_length = target_length - current_length
            ratio = remaining_length / segment_length
            intermediate_point = segment.interpolate(ratio, normalized=True)

            # Créer les deux segments résultants
            segment1 = LineString(line.coords[:i] + [(intermediate_point.x, intermediate_point.y)])
            segment2 = LineString([(intermediate_point.x, intermediate_point.y)] + line.coords[i:])

            return [segment1, segment2]

        current_length += segment_length

    # Si la boucle se termine sans trouver le point de division, retournez None
    print('failed')
    return None


def split_line(line, max_length):
    coords = line.coords[:]
    n = len(coords)
    if line.length > max_length:
            return split_line_exact_length_curvilinear(line)
    else:
        return [line]


def multi_split(df, max_length, n=10):

    for i in tqdm(range(n)):
        to_split = df.loc[df['length']>max_length].copy()
        ok = df.loc[df['length']<=max_length].copy()
        if len(to_split)>0:
            to_split['geometries'] = to_split['geometry'].apply(split_line, max_length=max_length)
            to_split = utils.df_explode(to_split, 'geometries')
            to_split['geometry'] = to_split['geometries']
            to_split['length'] = to_split['geometry'].apply(lambda x: x.length)
            df = pd.concat([to_split, ok])
        else: 
            return ok
    return df

def get_intersections(geojson_dict):
    count = {}
    for feature in geojson_dict['features']:
        for p in feature['geometry']['coordinates']:
            count[tuple(p)] = count.get(tuple(p), 0) + 1
    return {k for k, v in count.items() if v > 1}


def get_nodes(geojson_dict):
    nodes = set()
    for feature in geojson_dict['features']:
        nodes.add(tuple(feature['geometry']['coordinates'][0]))
        nodes.add(tuple(feature['geometry']['coordinates'][-1]))
    return nodes.union(get_intersections(geojson_dict))


def split_links(links, max_length=100, iterations=10):

    assert links.crs != None, 'road_links crs must be set (crs in meter, NOT 3857)'
    assert links.crs != 3857, 'CRS error. crs 3857 is not supported. use a local projection in meters.'
    assert links.crs != 4326, 'CRS error, crs 4326 is not supported, use a crs in meter (NOT 3857)'

    crs = links.crs
    links['length'] = links.length
    
    # split
    rl = multi_split(links, max_length, iterations)
    rl = gpd.GeoDataFrame(rl)
    rl = rl.set_crs(crs)
    
    # to geojson and reload (not necessary - historical reasons only (available in quetzal)
    roads =  json.loads(rl.drop('geometries', axis=1).to_json())
    node_coordinates = list(get_nodes(roads))
    node_index = dict(
        zip(
            node_coordinates, 
            ['road_node_%i' % i for i in range(len(node_coordinates))]
        )
    )
    df = pd.DataFrame(node_index.items(), columns=['coordinates', 'index'])
    df['geometry'] = df['coordinates'].apply(lambda t: Point(t))
    nodes = gpd.GeoDataFrame(df.set_index(['index'])[['geometry']])


    for f in roads['features']:
        first = tuple(f['geometry']['coordinates'][0])
        last = tuple(f['geometry']['coordinates'][-1])
        f['properties']['a'] = node_index[first]
        f['properties']['b'] = node_index[last]

    links = gpd.read_file(json.dumps(roads))
    links.index = ['road_link_%i' % i for i in range(len(links))]

    # we must force crs, as geojson is assumed to be 4326 by default
    links = links.set_crs(crs, allow_override=True)
    links['length'] = links.length

    return links, nodes




