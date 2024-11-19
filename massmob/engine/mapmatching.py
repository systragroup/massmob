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


class RoadLinks:
    '''
    Link Object for mapmatching 

    parameters
    ----------
    links (gpd.GeoDataFrame): links in a metre projection (not 4326 or 3857)
    gps_track (gpd.GeoDataFrame): [['index','geometry']] ordered list of geometry Points. only needed if links is None (to import the links from osmnx)
    n_neighbors_centroid (int) : number of neighbor using the links centroid. first quick pass to find all good candidat road.

    returns
    ----------
    RoadLinks object for mapmatching
    '''
    def __init__(self, links, n_neighbors_centroid=100, max_distance=None, iterations=20):

        self.links = links
        assert self.links.crs != None, 'road_links crs must be set (crs in meter, NOT 3857)'
        assert self.links.crs != 3857, 'CRS error. crs 3857 is not supported. use a local projection in meters.'
        assert self.links.crs != 4326, 'CRS error, crs 4326 is not supported, use a crs in meter (NOT 3857)'

        self.crs = links.crs
        self.n_neighbors_centroid = n_neighbors_centroid
        
        try:
            self.links['length']
        except Exception:
            self.links['length'] = self.links.length

        if 'index' not in self.links.columns:
            self.links = self.links.reset_index()

        if max_distance is not None:
            self.disaggregated_links, self.disaggregated_nodes = road.split_links(links, max_distance, iterations)
            self.disaggregated_links.reset_index(drop=True, inplace=True)
        else:
            self.disaggregated_links = self.links.copy()
            self.disaggregated_nodes = self.nodes.copy()

        self.get_sparse_matrix()
        self.get_dict()
        self.fit_nearest_model()

        
    def get_sparse_matrix(self):
        self.mat, self.node_index = sparse_matrix(self.links[['a', 'b', 'length']].values)
        self.index_node = {v: k for k, v in self.node_index.items()}

    def get_dict(self):
        # create dict of road network parameters
        self.dict_node_a = self.links['a'].to_dict()
        self.dict_node_b = self.links['b'].to_dict()
        self.links_index_dict = self.links['index'].to_dict()
        self.dict_link = self.links.sort_values('length', ascending=True).drop_duplicates(['a', 'b'], keep='first').set_index(['a', 'b'], drop=False)['index'].to_dict()
        self.length_dict = self.links['length'].to_dict()
        self.geom_dict = dict(self.links['geometry'])
        self.disaggregated_geom_dict = dict(self.disaggregated_links['geometry'])
        self.cluster_dict = self.disaggregated_links['index'].to_dict()

    def fit_nearest_model(self):
        # Fit Nearest neighbors model
        links = utils.add_geometry_coordinates(self.disaggregated_links, columns=['x_geometry', 'y_geometry'])
        x = links[['x_geometry', 'y_geometry']].values

        if len(links) < self.n_neighbors_centroid: self.n_neighbors_centroid = len(links)

        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors_centroid, algorithm='ball_tree').fit(x)


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


def mapmatching_parallel(
        points, road_links, road_link_weight='time', workers=2, verbose=True, paths=[],
        ptt_kwargs={}, mapmatching_kwargs={}):

    if verbose:
        print('---- prepare data…')
    gps_tracks = points_to_tracks(points, **ptt_kwargs)
    road_links.links['length'] = road_links.links[road_link_weight]

    if verbose:
        print('---- init mapmatching…')
    nc = NetworkCaster_MapMaptching(gps_tracks, road_links)
    ray.shutdown()
    import massmodel
    ray.init(num_cpus=workers, runtime_env={"py_modules": [massmodel]})

    if verbose:
        print(' ---- mapmatching…')
    vals, node_lists, unmatched_trips = nc.Multi_Mapmatching(
        workers=workers, routing=True,
        **mapmatching_kwargs
    )

    if verbose:
        print(' ---- post processing…')
    
    df = node_lists.copy()
    # groupby trip_id: dissolve geometry, concat road_link_list, concat road_node_list
    df['road_link_list'] = df['road_link_list'].apply(lambda x: list(x))
    # deduplicate road_link_list
    def dedup(x):
        return [a for a, b in zip(x, x[1:] + [None]) if a != b]
    df['road_link_list'] = df['road_link_list'].apply(dedup)

    df['road_node_list'] = df['road_node_list'].apply(lambda x: list(x))

    df['geometry'] = df['road_link_list'].apply(
        lambda x: road_links.links.loc[x, 'geometry'].unary_union
    )

    df['geometry'] = df['geometry'].apply(
        lambda x: linemerge(x) if isinstance(x, shapely.geometry.MultiLineString) or isinstance(x, shapely.geometry.GeometryCollection) else x)
    df.dropna(subset=['geometry'], inplace=True)
    df = df.groupby('trip_id').agg({'geometry': unary_union, 'road_link_list': 'sum', 'road_node_list': 'sum'})

    df['geometry'] = df['geometry'].apply(lambda x: linemerge(x) if isinstance(x, shapely.geometry.MultiLineString) else x)

    unmatched = []
    for trips in unmatched_trips:
        unmatched += trips

    return df, unmatched


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


class NetworkCaster_MapMaptching:
    def __init__(self, gps_tracks, road_links):
        self.gps_tracks = gpd.GeoDataFrame(gps_tracks)
        self.road_links = road_links

    def Mapmatching(self, **kwargs):
        return _mapmatching(self.road_links, **kwargs)

    def Multi_Mapmatching(self, workers, **kwargs):
        # Ray runs
        val_ids = []
        node_list_ids = []
        unmatched_trip_ids = []
        for track_ids in np.array_split(self.gps_tracks.trip_id.unique(), workers):
            sub_gps_tracks = self.gps_tracks[self.gps_tracks.trip_id.isin(track_ids)]
            val, node_list, unmatched_trip = _multi_mapmatching.remote(
                sub_gps_tracks, self.road_links, **kwargs
            )
            unmatched_trip_ids.append(unmatched_trip)
            val_ids.append(val)
            node_list_ids.append(node_list)

        # get ray results
        vals = pd.DataFrame()
        node_lists = pd.DataFrame()
        unmatched_trips = []
        for val_id in val_ids:
            vals = pd.concat([vals, ray.get(val_id)])
        for node_list_id in node_list_ids:
            node_lists = pd.concat([node_lists, ray.get(node_list_id)])
        for unmatched_trip_id in unmatched_trip_ids:
            unmatched_trips.append(ray.get(unmatched_trip_id))

        return vals, node_lists, unmatched_trips


@ray.remote(num_returns=3)
def _multi_mapmatching(
        self_gps_track,  road_links,
        routing=False, n_neighbors=10,  distance_max=200, by='trip_id'):

    vals = gpd.GeoDataFrame()
    node_lists = gpd.GeoDataFrame()
    unmatched_trip = []

    for trip_id in ray.experimental.tqdm_ray.tqdm(self_gps_track[by].unique()):
        gps_track = self_gps_track[self_gps_track[by] == trip_id].drop(columns=by)
        # format index. keep dict to reindex after the mapmatching
        gps_track = gps_track.reset_index()
        gps_track.index = gps_track.index - 1
        gps_index_dict = gps_track['index'].to_dict()
        gps_track.index = gps_track.index + 1
        gps_track = gps_track.drop(columns=['index'])
        if len(gps_track) < 2:  # cannot mapmatch less than 2 points.
            unmatched_trip.append(trip_id)
        else:
            try:
                val, node_list = _mapmatching(
                    gps_track,  road_links,
                    routing=routing,  n_neighbors=n_neighbors, distance_max=distance_max
                )

                # add the by column to every data
                val[by] = trip_id
                node_list[by] = trip_id
                # apply input index
                val.index = val.index.map(gps_index_dict)
                node_list.index = node_list.index.map(gps_index_dict)
                vals = pd.concat([vals, val])
                node_lists = pd.concat([node_lists, node_list])
            except IndexError as e:
                # print(e)
                unmatched_trip.append(trip_id)
    return vals, node_lists, unmatched_trip


def _mapmatching(
        gps_track, road_links,
        n_neighbors=10, distance_max=1000,
        routing=False, plot=False):
    """
    gps_track: ordered list of geometry Point (in metre)
    nodes: data frame of street map nodes (crs in metre)
    distance_max: max radius to search candidat road for each gps point (default = 200m)
    routing: True return the complete routing from the first to the last point on the road network (default = False)

    Hidden Markov Map Matching Through Noise and Sparseness
        Paul Newson and John Krumm 2009
    """
    SIGMA = 4.07
    BETA = 3
    dijkstra_limit = 2000  # Limit on first dijkstra on road network.

    def nearest(one, nbrs, geometry=False):
        try:
            # Assert df_many.index.is_unique
            assert one.index.is_unique
        except AssertionError:
            msg = 'Index of one and many should not contain duplicates'
            print(msg)
            warnings.warn(msg)

        df_one = utils.add_geometry_coordinates(one.copy())

        y = df_one[['x_geometry', 'y_geometry']].values

        distances, indices = nbrs.kneighbors(y)

        indices = pd.DataFrame(indices)
        distances = pd.DataFrame(distances)
        indices = pd.DataFrame(indices.stack(), columns=['index_nn']).reset_index().rename(
            columns={'level_0': 'ix_one', 'level_1': 'rank'}
        )
        return indices

    def emission_logprob(distance, SIGMA=SIGMA):
        # c = 1 / (SIGMA * np.sqrt(2 * np.pi))
        # return c*np.exp(-0.5*(distance/SIGMA)**2)
        # return -np.log10(np.exp(-0.5*(distance/SIGMA)**2))
        return 0.5 * (distance / SIGMA) ** 2  # Drop constant with log. its the same for everyone.

    def transition_logprob(dijkstra_dist, gps_dist, BETA=BETA):
        c = 1 / BETA
        delta = abs(dijkstra_dist - gps_dist)
        # return c * np.exp(-c * delta)
        return c * delta

    # process map
    try:
        road_links.links['length']
    except Exception:
        road_links.links['length'] = road_links.links.length

    gps_dict = gps_track['geometry'].to_dict()
    # GPS point distance to next point.
    gps_dist_dict = gps_track['geometry'].distance(gps_track.shift(-1)).to_dict()

    # ======================================================
    # Nearest roads and data preparation
    # ======================================================
    candidat_links = nearest(gps_track, road_links.nbrs).drop(columns=['rank'])

    def project(A, B, normalized=False):
        return [a.project(b, normalized=normalized) for a, b in zip(A, B)]

    def distance(A, B):
        return [a.distance(b) for a, b in zip(A, B)]

    candidat_links['droad_geom'] = candidat_links['index_nn'].apply(lambda x: road_links.disaggregated_geom_dict.get(x)) 
    candidat_links['gps_geom'] = candidat_links['ix_one'].apply(lambda x: gps_dict.get(x))

    # Add gps distance to road.
    candidat_links['distance'] = distance(candidat_links['gps_geom'], candidat_links['droad_geom'])

    candidat_links.sort_values(['ix_one', 'distance'], inplace=True)
    ranks = list(range(road_links.n_neighbors_centroid)) * len(gps_track)
    # print(len(ranks), len(candidat_links), len(gps_track))
    candidat_links['actual_rank'] = ranks
    candidat_links = candidat_links.loc[candidat_links['actual_rank'] < n_neighbors]
    candidat_links = candidat_links[candidat_links['distance'] < distance_max]
    if len(candidat_links) == 0:
        raise IndexError('No candidat_links within distance_max')
    candidat_links['index_parent'] = candidat_links['index_nn'].map(road_links.cluster_dict)  # parent link instead of disaggregated
    candidat_links = candidat_links.drop_duplicates(subset=['ix_one', 'index_parent'], keep='first')
    candidat_links = candidat_links.reset_index().drop(columns=['index'])

    # Add offset
    candidat_links['road_geom'] = candidat_links['index_parent'].apply(lambda x: road_links.geom_dict.get(x))
    candidat_links['offset'] = project(candidat_links['road_geom'], candidat_links['gps_geom'], normalized=False)
    dict_distance = candidat_links.set_index(['ix_one', 'index_parent'])['distance'].to_dict()

    # make tuple with road index and offset.
    candidat_links['index_parent'] = list(zip(candidat_links['index_parent'], candidat_links['offset']))
    candidat_links = candidat_links[['ix_one', 'index_parent']]

    # add virtual nodes start and end.
    candidat_links.loc[len(candidat_links)] = [candidat_links['ix_one'].max() + 1, candidat_links['index_parent'].iloc[-1]]
    candidat_links.loc[-1] = [-1, candidat_links['index_parent'].iloc[0]]
    candidat_links.index = candidat_links.index + 1  # shifting index
    candidat_links = candidat_links.sort_index()  # sorting by index

    # dict of each linked point (ix_one). if pts 10 is NaN, point 9 will be linked to point 11
    dict_point_link = dict(
        zip(list(candidat_links['ix_one'].unique())[:-1], list(candidat_links['ix_one'].unique())[1:])
    )

    candidat_links = candidat_links.groupby('ix_one').agg(list)
    candidat_links = candidat_links.rename(columns={'index_parent': 'road_a'})
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

    # ======================================================
    # DIJKSTRA sur road network
    # ======================================================

    # lien de la route a vers b dans le pseudo graph
    # mais le dijkstra est entre le lien b(route a) vers le lien a(route b)
    candidat_links['node_b'] = candidat_links['road_a'].apply(lambda x: road_links.dict_node_b.get(x))
    candidat_links['node_a'] = candidat_links['road_b'].apply(lambda x: road_links.dict_node_a.get(x))

    # candidat_links = candidat_links.fillna(0)

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

    # ======================================================
    # Calcul probabilité
    # ======================================================
    candidat_links['length'] = candidat_links['road_a'].apply(lambda x: road_links.length_dict.get(x))

    candidat_links['dijkstra'] = candidat_links['dijkstra'] + candidat_links['length'] - candidat_links[
        'road_a_offset'] + candidat_links['road_b_offset']
    cond = candidat_links['road_a'] == candidat_links['road_b']
    candidat_links.loc[cond, 'dijkstra'] = candidat_links.loc[cond, 'road_b_offset'] - candidat_links.loc[
        cond, 'road_a_offset']
    candidat_links = candidat_links.drop(columns='length')

    # candidat_links['dijkstra'] = np.abs(candidat_links['dijkstra'])

    # applique la distance réelle entre la route et le point GPS.
    candidat_links['distance_to_road'] = candidat_links.set_index(['ix_one', 'road_a']).index.map(
        dict_distance.get)  # .fillna(5)

    # applique la distance entre les point gps a vers b
    candidat_links['gps_distance'] = candidat_links['ix_one'].apply(lambda x: gps_dist_dict.get(x))  # .fillna(3)

    # path prob
    candidat_links['path_prob'] = emission_logprob(candidat_links['distance_to_road']) + transition_logprob(
        candidat_links['dijkstra'], candidat_links['gps_distance'])

    # tous les liens avec les noeuds virtuels (start finish) ont une prob constante (1 par defaut).
    ind = candidat_links['ix_one'] == -1
    candidat_links.loc[ind, 'path_prob'] = 1

    ind = candidat_links['ix_one'] == candidat_links['ix_one'].max()
    candidat_links.loc[ind, 'path_prob'] = 1

    # ======================================================
    # Dijkstra sur pseudo graph
    # ======================================================
    candidat_links['a'] = list(zip(candidat_links['ix_one'], candidat_links['road_a'], candidat_links['road_a_offset']))
    candidat_links['b'] = list(
        zip(candidat_links['ix_one'].apply(lambda x: dict_point_link.get(x)), candidat_links['road_b'],
            candidat_links['road_b_offset'])
    )
    first_node = candidat_links.iloc[0]['a']
    last_node = candidat_links.iloc[-1]['b']
    pseudo_mat, pseudo_node_index = sparse_matrix(candidat_links[['a', 'b', 'path_prob']].values.tolist())
    pseudo_index_node = {v: k for k, v in pseudo_node_index.items()}

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
    pseudo_dist_matrix

    pseudo_predecessors = pd.DataFrame(pseudo_predecessors)
    pseudo_predecessors.index = pseudo_predecessors.index.map(pseudo_index_node)
    pseudo_predecessors[0] = pseudo_predecessors[0].apply(lambda x: pseudo_index_node.get(x))

    path = []
    last_value = last_node
    for i in range(len(candidat_links['ix_one'].unique())):
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
    dijkstra_dict = candidat_links.set_index(['ix_one', 'road_a', 'road_b'], drop=False)['dijkstra'].to_dict()
    val['length'] = val.set_index([val.index, 'road_id_a', 'road_id_b']).index.map(dijkstra_dict.get)

    if plot:
        f, ax = plt.subplots(figsize=(10, 10))
        gps_track.plot(ax=ax, marker='o', color='blue', markersize=20)
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
            node_list.append(int(df_path.iloc[-(1 + i)]['sparse_node_b']))  # premier noed (noed b)
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

        # ajoute le noed a du premier point. puisque le Dijkstra a été calculé à partir des noeds b.
        # le noed a du premier point gps doit être ajouté manuellement.
        node_mat = node_mat[::-1]
        # transforme la liste de noeud en liste de route
        link_mat = []
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

        # format en liste dans un dataframe
        node_mat = pd.Series(node_mat, dtype='object').to_frame('road_node_list')
        node_mat['road_link_list'] = link_mat

        if plot:
            f, ax = plt.subplots(figsize=(10, 10))
            road_links.links.plot(ax=ax, linewidth=1)
            gps_track.plot(ax=ax, marker='o', color='red', markersize=20)
            road_links.links.loc[node_list['road_id']].plot(ax=ax, color='orange', linewidth=2)
            plt.xlim([gps_track['geometry'].x.min() - 1000, gps_track['geometry'].x.max() + 1000])
            plt.ylim([gps_track['geometry'].y.min() - 100, gps_track['geometry'].y.max() + 1000])
            plt.show()
    return val, node_mat

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