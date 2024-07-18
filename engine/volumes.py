import geopandas as gpd
from shapely.geometry import Point


def get_first_point(row):
    return Point(row['geometry'].coords[0])

def get_last_point(row):
    return Point(row['geometry'].coords[-1])


def build_od_matrix(traces, zoning):
    traces_origin = traces[['track_id', 'departure_point']]
    traces_origin = gpd.GeoDataFrame(traces_origin, geometry='departure_point', crs='epsg:2154')
    merge = gpd.sjoin(traces_origin, zoning, how='left', predicate='within')
    merge = merge[~merge.index.duplicated(keep='first')]
    traces['origin'] = merge['zone_id']

    traces_destination = traces[['track_id', 'end_point']]
    traces_destination = gpd.GeoDataFrame(traces_destination, geometry='end_point', crs='epsg:2154')
    merge = gpd.sjoin(traces_destination, zoning, how='left', predicate='within')
    merge = merge[~merge.index.duplicated(keep='first')]
    traces['destination'] = merge['zone_id']

    OD = traces.groupby(['origin', 'destination']).size().reset_index(name='counts')
    OD.reset_index(inplace=True)

    return OD

def build_od_matrix_redressee(traces, zoning):
    traces_origin = traces[['track_id', 'departure_point']]
    traces_origin = gpd.GeoDataFrame(traces_origin, geometry='departure_point', crs='epsg:2154')
    merge = gpd.sjoin(traces_origin, zoning, how='left', predicate='within')
    merge = merge[~merge.index.duplicated(keep='first')]
    traces['origin'] = merge['zone_id']

    traces_destination = traces[['track_id', 'end_point']]
    traces_destination = gpd.GeoDataFrame(traces_destination, geometry='end_point', crs='epsg:2154')
    merge = gpd.sjoin(traces_destination, zoning, how='left', predicate='within')
    merge = merge[~merge.index.duplicated(keep='first')]
    traces['destination'] = merge['zone_id']

    OD = traces.groupby(['origin', 'destination'])['weight'].sum().reset_index(name='counts_reel')
    OD.reset_index(inplace=True)

    return OD