import geopandas as gpd
import pandas as pd
import ray
# import imp
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import shapely
from shapely.ops import linemerge, unary_union, transform
from shapely.geometry import LineString, Point
from tqdm import tqdm
import json
import copy
import sys
from time import sleep
import os
from glob import glob
import tempfile
import importlib
from massmob.engine import utils
from massmob.engine import road


class Network:
    '''
    Link Object for mapmatching 

    parameters
    ----------
    links (gpd.GeoDataFrame): links in a metre projection (not 4326 or 3857)
    gps_track (gpd.GeoDataFrame): [['index','geometry']] ordered list of geometry Points. only needed if links is None (to import the links from osmnx)
    n_neighbors_centroid (int) : number of neighbor using the links centroid. first quick pass to find all good candidat road.

    returns
    ----------
    Network links object for mapmatching
    '''
    def __init__(self, links, nodes=None, weight="length", penalty=None, n_neighbors_centroid=100, max_distance=None, iterations=20):

        self.links = links
        self.nodes = nodes
        self.weight = weight
        self.penalty = penalty
        assert self.links.crs != None, 'road_links crs must be set (crs in meter, NOT 3857)'
        assert self.links.crs != 3857, 'CRS error. crs 3857 is not supported. use a local projection in meters.'
        assert self.links.crs != 4326, 'CRS error, crs 4326 is not supported, use a crs in meter (NOT 3857)'

        self.crs = links.crs
        self.n_neighbors_centroid = n_neighbors_centroid
        

        try:
            self.links['length']
        except Exception:
            self.links['length'] = self.links.length

        self.links["weight+penalty"] = self.links[weight] + self.links.get(penalty, 0)

        if 'index' not in self.links.columns:
            self.links = self.links.reset_index()

        if max_distance is not None:
            self.disaggregated_links, self.disaggregated_nodes = road.split_links_vectorized(links, max_distance, suffix='network')
            self.disaggregated_links = self.disaggregated_links[self.disaggregated_links['length'] > 0]  # remove rtw, etc.
            self.disaggregated_links.reset_index(drop=True, inplace=True)
        else:
            self.disaggregated_links = self.links[self.links.length>0].copy()  # remove rtw, etc.
            self.disaggregated_nodes = self.nodes.copy()

        self.get_sparse_matrix()
        self.get_dict()
        self.fit_nearest_models()

        
    def get_sparse_matrix(self):
        self.mat, self.node_index = sparse_matrix(self.links[['a', 'b', "weight+penalty"]].values)
        self.index_node = {v: k for k, v in self.node_index.items()}

    def get_dict(self):
        # create dict of road network parameters
        self.dict_node_a = self.links['a'].to_dict()
        self.dict_node_b = self.links['b'].to_dict()
        self.links_index_dict = self.links['index'].to_dict()
        self.dict_link = self.links.sort_values("length", ascending=True).drop_duplicates(['a', 'b'], keep='first').set_index(['a', 'b'], drop=False)['index'].to_dict()
        self.length_dict = self.links['length'].to_dict()
        self.weight_dict = self.links[self.weight].to_dict()
        self.penalty_dict = self.links[self.penalty].to_dict() if self.penalty else {}
        self.geom_dict = dict(self.links['geometry'])
        self.disaggregated_geom_dict = dict(self.disaggregated_links['geometry'])
        self.maxspeed_series = self.links["maxspeed"]

    def fit_nearest_models(self, adhoc_nearest_models={}):
        # adhoc_nearest_models: {label: [links, neighbors]}

        # Fit Nearest neighbors model
        links = utils.add_geometry_coordinates(self.disaggregated_links, columns=['x_geometry', 'y_geometry'])

        # main model
        nneighbors = self.n_neighbors_centroid
        self.main_cluster_dict = links['index'].to_dict()
        x = links[['x_geometry', 'y_geometry']].values
        if len(links) < nneighbors: nneighbors = len(links)
        self.nbrs = {"main": NearestNeighbors(n_neighbors=nneighbors, algorithm='ball_tree').fit(x)}
        
        # adhoc models
        for k,v in adhoc_nearest_models.items():
            nneighbors = v[1]
            temp_links = v[0].reset_index(drop=True)
            temp_links = utils.add_geometry_coordinates(temp_links, columns=['x_geometry', 'y_geometry'])
            self.__setattr__(f"{k}_cluster_dict", temp_links['index'].to_dict())
            x = temp_links[['x_geometry', 'y_geometry']].values
            if len(temp_links) < nneighbors: nneighbors = len(temp_links)

            self.nbrs.update({k: NearestNeighbors(n_neighbors=nneighbors, algorithm='ball_tree').fit(x)})


def points_to_tracks(points, by='track_id', time_col='eventDate8601', seq_col='node_seq'):
    gps_tracks = points[['geometry', 'track_id', 'eventDate8601']].copy()
    gps_tracks.sort_values([by, time_col], inplace=True)
    # add sequence number
    counter = gps_tracks.groupby(by).agg(len)['geometry'].values
    order = [i for j in range(len(counter)) for i in range(counter[j])]
    gps_tracks.loc[:, seq_col] = order
    gps_tracks.index = gps_tracks[[by, seq_col]].apply(
        lambda x: '{}_{}'.format(x[0], x[1]), axis=1
    )
    return gps_tracks.rename(columns={by: 'trip_id'})


# def mapmatching_parallel(
#         points, road_links, road_link_weight='time', workers=2, verbose=True, paths=[],
#         ptt_kwargs={}, mapmatching_kwargs={}):

#     if verbose:
#         print('---- prepare data…')
#     gps_tracks = points_to_tracks(points, **ptt_kwargs)
#     road_links.links['length'] = road_links.links[road_link_weight]

#     if verbose:
#         print('---- init mapmatching…')
#     nc = NetworkCaster_MapMaptching(gps_tracks, road_links)
#     ray.shutdown()
#     import massmodel
#     ray.init(num_cpus=workers, runtime_env={"py_modules": [massmodel]})

#     if verbose:
#         print(' ---- mapmatching…')
#     vals, node_lists, unmatched_trips = nc.Multi_Mapmatching(
#         workers=workers, routing=True,
#         **mapmatching_kwargs
#     )

#     if verbose:
#         print(' ---- post processing…')
    
#     df = node_lists.copy()
#     # groupby trip_id: dissolve geometry, concat road_link_list, concat road_node_list
#     df['road_link_list'] = df['road_link_list'].apply(lambda x: list(x))
#     # deduplicate road_link_list
#     def dedup(x):
#         return [a for a, b in zip(x, x[1:] + [None]) if a != b]
#     df['road_link_list'] = df['road_link_list'].apply(dedup)

#     df['road_node_list'] = df['road_node_list'].apply(lambda x: list(x))

#     df['geometry'] = df['road_link_list'].apply(
#         lambda x: road_links.links.loc[x, 'geometry'].unary_union
#     )

#     df['geometry'] = df['geometry'].apply(
#         lambda x: linemerge(x) if isinstance(x, shapely.geometry.MultiLineString) or isinstance(x, shapely.geometry.GeometryCollection) else x)
#     df.dropna(subset=['geometry'], inplace=True)
#     df = df.groupby('trip_id').agg({'geometry': unary_union, 'road_link_list': 'sum', 'road_node_list': 'sum'})

#     df['geometry'] = df['geometry'].apply(lambda x: linemerge(x) if isinstance(x, shapely.geometry.MultiLineString) else x)

#     unmatched = []
#     for trips in unmatched_trips:
#         unmatched += trips

#     return df, unmatched


# buildindex
def build_index(edges):
    nodelist = {e[0] for e in edges}.union({e[1] for e in edges})
    nlen = len(nodelist)
    return dict(zip(nodelist, range(nlen)))


# build matrix
def sparse_matrix(edges, index=None):
    if index is None:
        index = build_index(edges)
    nlen = len(index)
    coefficients = zip(*((index[u], index[v], w) for u, v, w in edges))
    row, col, data = coefficients
    return csr_matrix((data, (row, col)), shape=(nlen, nlen)), index


# class NetworkCaster_MapMaptching:
#     def __init__(self, gps_tracks, road_links):
#         self.gps_tracks = gpd.GeoDataFrame(gps_tracks)
#         self.road_links = road_links

#     def Mapmatching(self, **kwargs):
#         return _mapmatching(self.road_links, **kwargs)

#     def Multi_Mapmatching(self, workers, **kwargs):
#         # Ray runs
#         val_ids = []
#         node_list_ids = []
#         unmatched_trip_ids = []
#         for track_ids in np.array_split(self.gps_tracks.trip_id.unique(), workers):
#             sub_gps_tracks = self.gps_tracks[self.gps_tracks.trip_id.isin(track_ids)]
#             val, node_list, unmatched_trip = _multi_mapmatching.remote(
#                 sub_gps_tracks, self.road_links, **kwargs
#             )
#             unmatched_trip_ids.append(unmatched_trip)
#             val_ids.append(val)
#             node_list_ids.append(node_list)

#         # get ray results
#         vals = pd.DataFrame()
#         node_lists = pd.DataFrame()
#         unmatched_trips = []
#         for val_id in val_ids:
#             vals = pd.concat([vals, ray.get(val_id)])
#         for node_list_id in node_list_ids:
#             node_lists = pd.concat([node_lists, ray.get(node_list_id)])
#         for unmatched_trip_id in unmatched_trip_ids:
#             unmatched_trips.append(ray.get(unmatched_trip_id))

#         return vals, node_lists, unmatched_trips


# @ray.remote(num_returns=3)
# def _multi_mapmatching(
#         self_gps_track,  road_links,
#         routing=False, n_neighbors=10,  distance_max=200, by='trip_id'):

#     vals = gpd.GeoDataFrame()
#     node_lists = gpd.GeoDataFrame()
#     unmatched_trip = []

#     for trip_id in ray.experimental.tqdm_ray.tqdm(self_gps_track[by].unique()):
#         gps_track = self_gps_track[self_gps_track[by] == trip_id].drop(columns=by)
#         # format index. keep dict to reindex after the mapmatching
#         gps_track = gps_track.reset_index()
#         gps_track.index = gps_track.index - 1
#         gps_index_dict = gps_track['index'].to_dict()
#         gps_track.index = gps_track.index + 1
#         gps_track = gps_track.drop(columns=['index'])
#         if len(gps_track) < 2:  # cannot mapmatch less than 2 points.
#             unmatched_trip.append(trip_id)
#         else:
#             try:
#                 val, node_list = _mapmatching(
#                     gps_track,  road_links,
#                     routing=routing,  n_neighbors=n_neighbors, distance_max=distance_max
#                 )

#                 # add the by column to every data
#                 val[by] = trip_id
#                 node_list[by] = trip_id
#                 # apply input index
#                 val.index = val.index.map(gps_index_dict)
#                 node_list.index = node_list.index.map(gps_index_dict)
#                 vals = pd.concat([vals, val])
#                 node_lists = pd.concat([node_lists, node_list])
#             except IndexError as e:
#                 # print(e)
#                 unmatched_trip.append(trip_id)
#     return vals, node_lists, unmatched_trip

def filter_noise(gps_data, min_time=1, max_speed=100):
    gps_data = gps_data[~((gps_data["t"] <= min_time) & (gps_data["s"] > max_speed))]
    return gps_data

def nearest(one, network):
    try:
        assert one.index.is_unique
    except AssertionError:
        msg = 'Index of one and many should not contain duplicates'
        print(msg)
        warnings.warn(msg)

    y = one[['x', 'y']].values
    all_indices = []
    # nearest among disagregated links for each network type
    for k, nbr in network.nbrs.items():
        distances, indices = nbr.kneighbors(y)

        indices = pd.DataFrame(indices)
        distances = pd.DataFrame(distances)
        indices = pd.DataFrame(indices.stack(), columns=['ix_many']).reset_index().rename(
            columns={'level_0': 'index_gps', 'level_1': 'rank'}
        )
        # convert to parent
        indices["index_link"] = indices["ix_many"].map(network.__getattribute__(f"{k}_cluster_dict").get)

        all_indices.append(indices)
 
    indices = pd.concat(all_indices).reset_index(drop=True)

    # drop duplicates parents
    indices = indices.drop_duplicates(subset=["index_gps", "index_link"], keep="first")

    return indices

def _compute_candidate_links(gps_data, road_links, distance_max):
    maxspeed_dict = road_links.links["maxspeed"].to_dict() # TODO: deplacer dans road_links

    candidat_links = nearest(gps_data, road_links)
    # print(candidat_links)
    # control speed adequation
    candidat_links["s"] = candidat_links["index_gps"].map(gps_data["s"].to_dict())
    candidat_links["max_speed"] = candidat_links['index_link'].apply(lambda x: maxspeed_dict.get(x))
    candidat_links = candidat_links[candidat_links["s"] < candidat_links["max_speed"]]

    ## compute gps / link distance
    gps_geometry_dict = gps_data['geometry'].to_dict()
    candidat_links['link_geom'] = candidat_links['index_link'].apply(lambda x: road_links.geom_dict.get(x))
    candidat_links['gps_geom'] = candidat_links['index_gps'].apply(lambda x: gps_geometry_dict.get(x))

    # Add gps distance to link, sort and filter on distance
    candidat_links['distance'] = shapely.distance(candidat_links['gps_geom'].values, candidat_links['link_geom'].values)
    candidat_links.sort_values(['index_gps', 'distance'], inplace=True)
    # We do not use GPS accuracy as some points are not on the network (buildings, etc.)
    # gps_accuracy_dict = gps_data["accuracy"].to_dict()
    # candidat_links['gps_accuracy'] = candidat_links['index_gps'].apply(lambda x: gps_accuracy_dict.get(x))
    # candidat_links = candidat_links[candidat_links['distance'] < candidat_links['gps_accuracy'].clip(upper=distance_max)].reset_index(drop=True)
    candidat_links = candidat_links[candidat_links['distance'] < distance_max].reset_index(drop=True)

    if len(candidat_links) == 0:
        raise IndexError('No candidat_links within distance_max')

    # Add weighted offset
    candidat_links['offset'] = shapely.line_locate_point(candidat_links['link_geom'].values, candidat_links['gps_geom'].values, normalized=False)
    if road_links.weight != "length":
        candidat_links["length"] = candidat_links["index_link"].apply(lambda x: road_links.length_dict.get(x))
        candidat_links["weight"] = candidat_links["index_link"].apply(lambda x: road_links.weight_dict.get(x))
        candidat_links["offset"] *= candidat_links["weight"] / candidat_links["length"]

    dict_distance = candidat_links.set_index(['index_gps', 'index_link'])['distance'].to_dict()

    # make tuple with road index and offset.
    candidat_links['index_link'] = list(zip(candidat_links['index_link'], candidat_links['offset']))
    candidat_links = candidat_links[['index_gps', 'index_link']]

    # add firt and last
    candidat_links.loc[len(candidat_links)] = [candidat_links['index_gps'].max() + 1, candidat_links['index_link'].iloc[-1]]
    candidat_links.loc[-1] = [-1, candidat_links['index_link'].iloc[0]]
    candidat_links.index = candidat_links.index + 1  # shifting index
    candidat_links = candidat_links.sort_index()  # sorting by index

    # dict of each linked point (index_gps). if pts 10 is NaN, point 9 will be linked to point 11
    dict_point_link = dict(
        zip(list(candidat_links['index_gps'].unique())[:-1], list(candidat_links['index_gps'].unique())[1:])
    )

    candidat_links = candidat_links.groupby('index_gps').agg(list)
    candidat_links = candidat_links.rename(columns={'index_link': 'road_a'})
    candidat_links = candidat_links.reset_index()

    candidat_links['road_b'] = candidat_links['road_a'].shift(-1)

    # remove last line (last node is virtual and linked to no one.)
    candidat_links = candidat_links.iloc[:-1]
    candidat_links = candidat_links.explode(column='road_a').explode(column='road_b').reset_index(drop=True)

    # unpack tuple road_ID, offset
    candidat_links[['road_a', 'road_a_offset']] = pd.DataFrame(
        candidat_links['road_a'].tolist(), index=candidat_links.index)
    candidat_links[['road_b', 'road_b_offset']] = pd.DataFrame(
        candidat_links['road_b'].tolist(), index=candidat_links.index)
    
    return candidat_links, dict_distance, dict_point_link
    
def emission_logprob(distance, SIGMA=2):
    return (distance / SIGMA) ** 2

def transition_logprob(dijkstra_time, gps_time, BETA=2, ALPHA=0.75, avg_rl_speed_kmh=30):
    time_delta = (abs(dijkstra_time - gps_time * ALPHA) * avg_rl_speed_kmh / 3.6 / BETA) ** 2 # tendance à favoriser les trajets plus longs si gps time est grand
    abs_time = (dijkstra_time * avg_rl_speed_kmh / 3.6 / BETA) ** 2 # pour éviter les trajets trop longs
    return (time_delta  + abs_time) 

def _compute_pathprob(
        candidat_links, road_links, gps_time_dict, dict_distance, 
        SIGMA=2, BETA=2, ALPHA=0.75, avg_rl_speed_kmh=30):
    # ======================================================
    # Calcul probabilité
    # ======================================================
    # applique la duree entre les point gps a vers b
    candidat_links['gps_duration'] = candidat_links['index_gps'].apply(lambda x: gps_time_dict.get(x))
    # applique la distance réelle entre la route et le point GPS.
    candidat_links['distance_to_road'] = candidat_links.set_index(['index_gps', 'road_a']).index.map(
        dict_distance.get)

    # path prob
    candidat_links['path_prob'] = emission_logprob(candidat_links['distance_to_road'], SIGMA) + transition_logprob(
        candidat_links['dijkstra'], candidat_links['gps_duration'], BETA, ALPHA, avg_rl_speed_kmh)

    # tous les liens avec les noeuds virtuels (start finish) ont une prob constante (1 par defaut).
    ind = candidat_links['index_gps'] == -1
    candidat_links.loc[ind, 'path_prob'] = 1

    ind = candidat_links['index_gps'] == candidat_links['index_gps'].max()
    candidat_links.loc[ind, 'path_prob'] = 1

    return candidat_links

def _mapmatching(
        gps_data, road_links, distance_max=50, plot=False, routing=True,
        SIGMA=2, BETA=2, ALPHA=0.75, avg_rl_speed_kmh=30, dijkstra_limit=1000):
    

    gps_data = filter_noise(gps_data)
    candidat_links, dict_distance, dict_point_link = _compute_candidate_links(gps_data, road_links, distance_max)
    
    # print(candidat_links)
    # ======================================================
    # DIJKSTRA sur road network
    # ======================================================

    # lien de la route a vers b dans le pseudo graph
    # mais le dijkstra est entre le noeud b du lien a vers le noeud a du lien b
    candidat_links['node_b'] = candidat_links['road_a'].apply(lambda x: road_links.dict_node_b.get(x))
    candidat_links['node_a'] = candidat_links['road_b'].apply(lambda x: road_links.dict_node_a.get(x))

    # Create sparse matrix of the road network
    # try:  # for multi-mapmatching, feeding it as an input save time (it's the same mat every time)
    #     road_links.mat
    # except Exception:
    #     mat, node_index = sparse_matrix(road_links.links[['a', 'b', 'length']].values.tolist())

    index_node = {v: k for k, v in road_links.node_index.items()}

    # liste des origines pour le dijkstra
    origin = list(candidat_links['node_b'].unique())
    origin_sparse = [road_links.node_index[x] for x in origin]

    # Dijktra on the road network from node = incices to every other nodes.
    # From b to a.
    dist_matrix, predecessors = dijkstra(
        csgraph=road_links.mat,
        directed=True,
        indices=origin_sparse,
        return_predecessors=True,
        limit=dijkstra_limit
    )

    dist_matrix = pd.DataFrame(dist_matrix)
    dist_matrix.index = origin

    # Dijkstra Destinations list
    destination = list(candidat_links['node_a'].unique())
    destination_sparse = [road_links.node_index[x] for x in destination]

    # Filter. on garde seulement les destination d'intéret (les nodes a)
    dist_matrix = dist_matrix[destination_sparse]
    # Then rename (less columns then less time)
    dist_matrix = dist_matrix.rename(columns=index_node)

    # identifie les routes pas trouvées (limit sur Dijkstra de 2000)
    dist_matrix = dist_matrix.replace(np.inf, np.nan)


    # Applique la distance routing a candidat_link
    temp_dist_matrix = dist_matrix.stack(dropna=True).reset_index().rename(
        columns={'level_0': 'b', 'level_1': 'a', 0: 'dijkstra'}
    )
    candidat_links = candidat_links.merge(
        temp_dist_matrix, left_on=['node_b', 'node_a'], right_on=['b', 'a'], how='left'
    ).drop(columns=['b', 'a'])

    # si des pair origine detination n'ont pas été trouvé dans le routing limité
    # on refait un Dijktra sans limite avec ces origin (noeud b).
    unfound_origin_nodes = (candidat_links[np.isnan(candidat_links['dijkstra'])]['node_b'].unique())
    if len(unfound_origin_nodes) > 0:
        origin_sparse2 = [road_links.node_index[x] for x in unfound_origin_nodes]
        # Dijktra on the road network from node = incices to every other nodes.
        # from b to a.
        dist_matrix2, predecessors2 = dijkstra(
            csgraph=road_links.mat,
            directed=True,
            indices=origin_sparse2,
            return_predecessors=True,
            limit=np.inf
        )

        dist_matrix2 = pd.DataFrame(dist_matrix2)
        # dist_matrix2 = dist_matrix2.rename(columns=index_node)
        dist_matrix2.index = unfound_origin_nodes

        # Filter. on garde seulement les destination d'intéret (les nodes a)
        dist_matrix2 = dist_matrix2[destination_sparse]
        dist_matrix2 = dist_matrix2.rename(columns=index_node)

        # Applique les nouvelles valeurs a la matrice originale
        dist_matrix.loc[dist_matrix2.index] = dist_matrix2

        candidat_links = candidat_links.drop(columns='dijkstra')
        temp_dist_matrix = dist_matrix.stack(dropna=True).reset_index().rename(
            columns={'level_0': 'b', 'level_1': 'a', 0: 'dijkstra'}
        )
        candidat_links = candidat_links.merge(
            temp_dist_matrix, left_on=['node_b', 'node_a'], right_on=['b', 'a'], how='left'
        ).drop(columns=['b', 'a'])

    candidat_links['weight'] = candidat_links['road_a'].apply(lambda x: road_links.weight_dict.get(x))
    candidat_links['dijkstra'] = candidat_links['dijkstra'] + candidat_links['weight'] - candidat_links[
        'road_a_offset'] + candidat_links['road_b_offset']

    cond = candidat_links['road_a'] == candidat_links['road_b']
    candidat_links.loc[cond, 'dijkstra'] = candidat_links.loc[cond, 'road_b_offset'] - candidat_links.loc[
        cond, 'road_a_offset']
    candidat_links = candidat_links[candidat_links["dijkstra"]>=0] # offset on reversed links can create negative dijkstra values
    candidat_links = candidat_links.drop(columns='weight')

    # =======================
    # PROBABILITIES
    # ===================
    # GPS duration to next point.
    gps_time_dict = gps_data['t'].shift(-1).to_dict()

    candidat_links = _compute_pathprob(
        candidat_links, road_links, gps_time_dict, dict_distance, SIGMA, BETA, ALPHA, avg_rl_speed_kmh)
    insight = candidat_links
    # ======================================================
    # Dijkstra sur pseudo graph
    # ======================================================
    candidat_links['a'] = list(zip(candidat_links['index_gps'], candidat_links['road_a'], candidat_links['road_a_offset']))
    candidat_links['b'] = list(
        zip(candidat_links['index_gps'].apply(lambda x: dict_point_link.get(x)), candidat_links['road_b'],
            candidat_links['road_b_offset'])
    )
    first_node = candidat_links.iloc[0]['a']
    last_node = candidat_links.iloc[-1]['b']
    pseudo_mat, pseudo_node_index = sparse_matrix(candidat_links[['a', 'b', 'path_prob']].values.tolist())
    pseudo_index_node = {v: k for k, v in pseudo_node_index.items()}
    insight = candidat_links
    # Dijkstra on the road network from node = indices to every other nodes.
    # From b to a.
    pseudo_dist_matrix, pseudo_predecessors = dijkstra(
        csgraph=pseudo_mat,
        directed=True,
        indices=pseudo_node_index[first_node],
        return_predecessors=True,
        limit=np.inf
    )

    pseudo_dist_matrix = pd.DataFrame(pseudo_dist_matrix)

    # pseudo_dist_matrix = pseudo_dist_matrix.rename(columns=pseudo_index_node)
    pseudo_dist_matrix.index = pseudo_dist_matrix.index.map(pseudo_index_node)

    pseudo_predecessors = pd.DataFrame(pseudo_predecessors)
    pseudo_predecessors.index = pseudo_predecessors.index.map(pseudo_index_node)
    pseudo_predecessors[0] = pseudo_predecessors[0].apply(lambda x: pseudo_index_node.get(x))

    path = []
    last_value = last_node
    for i in range(len(candidat_links['index_gps'].unique())):
        last_value = pseudo_predecessors.loc[last_value][0]
        path.append(last_value)
    temp_path = path.copy()
    temp_path.reverse()

    path = [x[1] for x in path]

    path.reverse()

    val = pd.DataFrame(temp_path, columns=['index', 'road_id', 'offset']).set_index('index')[1:]
    val['road_id_b'] = val['road_id'].shift(-1)
    val = val[:-1]
    val = val.rename(columns={'road_id': 'road_id_a'})
    dijkstra_dict = candidat_links.set_index(['index_gps', 'road_a', 'road_b'], drop=False)['dijkstra'].to_dict()
    val['length'] = val.set_index([val.index, 'road_id_a', 'road_id_b']).index.map(dijkstra_dict.get)

    if plot:
        f, ax = plt.subplots(figsize=(10, 10))
        gps_data.plot(ax=ax, marker='o', color='blue', markersize=20)
        road_links.links.loc[path].plot(ax=ax, color='red')
        plt.show()

    # ======================================================
    # Reconstruction du routing
    # ======================================================
    node_list = []
    if routing:
        predecessors = pd.DataFrame(predecessors)
        predecessors.index = origin_sparse

        # Si on a fait deux dijkstra
        if len(unfound_origin_nodes) > 0:
            predecessors2 = pd.DataFrame(predecessors2)
            predecessors2.index = origin_sparse2
            predecessors.loc[predecessors2.index] = predecessors2

        # predecessors = predecessors.apply(lambda x : index_node.get(x))
        df_path = pd.DataFrame(path[1:], columns=['road_id'])
        df_path['sparse_node_b'] = df_path['road_id'].apply(lambda x: road_links.node_index.get(road_links.dict_node_b.get(x)))

        node_mat = []

        for i in range(len(df_path) - 1):
            node_list = []
            node_list.append(int(df_path.iloc[-(1 + i)]['sparse_node_b']))  # premier noeud (noeud b)
            node = predecessors.loc[
                df_path.iloc[-(1 + i + 1)]['sparse_node_b'], df_path.iloc[-(1 + i)]['sparse_node_b']
            ]
            while node != -9999:  # Ajoute les noeds b jusqua ce qu'on arrive au prochain point gps
                node_list.append(node)
                node = predecessors.loc[df_path.iloc[-(1 + i + 1)]['sparse_node_b'], node]

            node_list.append(  # ajoute le noeud a
                int(road_links.node_index[road_links.links.loc[df_path.iloc[-(1 + i + 1)]['road_id']]['a']])
            )
            node_list = [index_node[x] for x in node_list[::-1]]  # reverse and swap index
            node_mat.append(node_list)

        # ajoute le noeud a du premier point. puisque le Dijkstra a été calculé à partir des noeds b.
        # le noed a du premier point gps doit être ajouté manuellement.
        node_mat = node_mat[::-1]
        # transforme la liste de noeud en liste de route
        link_mat = []
        all_links = []
        for node_list in node_mat:
            link_list = []
            for i in range(len(node_list) - 1):
                # probleme quand node list est egal a deux, liée au links_index_dict
                try:
                    link_list.append(road_links.links_index_dict[road_links.dict_link[node_list[i], node_list[i + 1]]])
                except Exception:
                    'links index issue'
                    pass
            link_mat.append(link_list)
            all_links+=link_list

        # format en liste dans un dataframe
        node_mat = pd.Series(node_mat, dtype='object').to_frame('road_node_list')
        node_mat['road_link_list'] = link_mat

        if plot:
            print(remove_consecutive_duplicates(list(road_links.links.loc[all_links, "network_type"].values)))
            f, ax = plt.subplots(figsize=(10, 10))
            # filter road_link in bbox of gps_track
            plot_links = road_links.links.cx[
                gps_data['geometry'].x.min() - 1000: gps_data['geometry'].x.max() + 1000,
                gps_data['geometry'].y.min() - 1000: gps_data['geometry'].y.max() + 1000
            ]
            

            plot_links.plot(ax=ax, linewidth=0.5, alpha=0.5)
            gps_data.plot(ax=ax, marker='o', color='red', markersize=20, zorder=5)
            # type plot
            # road_links.links["network_type"].unique()
            colors = ["blue", "green", "red", "purple", "brown"]
            zip_types_colors = zip(road_links.links["network_type"].unique(), colors)
            # plot by color
            type_color_dict = dict(zip_types_colors)
            for t, c in type_color_dict.items():
                type_links = road_links.links[road_links.links["network_type"] == t].index.intersection(all_links)
                road_links.links.loc[type_links].plot(ax=ax, color=c, linewidth=2, label=t)
            plt.legend()
            # road_links.links.loc[all_links].plot(ax=ax, color='orange', linewidth=2)
            plt.xlim([gps_data['geometry'].x.min() - 1000, gps_data['geometry'].x.max() + 1000])
            plt.ylim([gps_data['geometry'].y.min() - 1000, gps_data['geometry'].y.max() + 1000])
            plt.show()

    return val, node_mat, insight

def remove_consecutive_duplicates(words):
    """
    Supprime les doublons consécutifs d'une liste de mots.
    Ex: ["a","a","b","b","b","c","a","a"] → ["a","b","c","a"]
    """
    if not words:
        return []

    cleaned = [words[0]]

    for w in words[1:]:
        if w != cleaned[-1]:
            cleaned.append(w)

    return cleaned

def fast_assign(volume_array, paths):
    """
    :param volume_array: array of volume to assign
    :type volume_array: numpy array
    :type paths: list of list of link index
    :param paths: list of path to assign volume to
    :return: pandas series of volume assigned to each link
    """
    z = zip(volume_array, paths)
    d = {}
    for volume, path in list(z):
        for key in path:
            try:
                d[key] += volume
            except KeyError:
                d[key] = volume
    return pd.Series(d)