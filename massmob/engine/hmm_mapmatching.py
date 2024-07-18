import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
# from quetzal.engine.pathfinder_utils import sparse_matrix
# from syspy.spatial.spatial import add_geometry_coordinates
import ray


def add_geometry_coordinates(df, columns=['x_geometry', 'y_geometry']):
    """
    Add x and y coordinates of the geometry to the dataframe
    :param df: GeoDataFrame
    :type df: GeoDataFrame
    :param columns: name of the columns to add
    :type columns: list
    :return: GeoDataFrame with x and y coordinates
    """
    df = df.copy()

    # if the geometry is not a point...
    centroids = df['geometry'].apply(lambda g: g.centroid)

    df[columns[0]] = centroids.apply(lambda g: g.coords[0][0])
    df[columns[1]] = centroids.apply(lambda g: g.coords[0][1])
    return df


# buildindex
def build_index(edges):
    """
    Build a dictionnary of node index
    :param edges: list of edges
    :type edges: list
    :return: dictionnary of node index"""
    nodelist = {e[0] for e in edges}.union({e[1] for e in edges})
    nlen = len(nodelist)
    return dict(zip(nodelist, range(nlen)))


# build matrix
def sparse_matrix(edges, index=None):
    """
    Build a sparse matrix of the road network
    :param edges: list of edges
    :type edges: list
    :param index: dictionnary of node index
    :type index: dict
    :return: sparse matrix of the road network and dictionnary of node index"""
    if index is None:
        index = build_index(edges)
    nlen = len(index)
    coefficients = zip(*((index[u], index[v], w) for u, v, w in edges))
    row, col, data = coefficients
    return csr_matrix((data, (row, col)), shape=(nlen, nlen)), index


class NetworkCaster_MapMaptching:
    def __init__(self, nodes, road_links, links, by='trip_id', sequence='link_sequence'):
        """
        :param nodes: GeoDataFrame of the road network nodes
        :type nodes: GeoDataFrame
        :param road_links: GeoDataFrame of the road network links
        :type road_links: GeoDataFrame
        :param links: GeoDataFrame of the gps tracks
        :type links: GeoDataFrame
        :param by: column name of the gps tracks to group by
        :type by: str
        :param sequence: column name of the gps tracks to order by
        :type sequence: str
        """
        self.links = gpd.GeoDataFrame(road_links)
        self.nodes = gpd.GeoDataFrame(nodes)

        assert self.links.crs is not None, 'nodes and road_links crs must be set (crs in meter, NOT 3857)'
        assert self.links.crs != 3857, 'CRS error. crs 3857 is not supported. use a local projection in meters.'
        assert self.links.crs == nodes.crs, 'nodes and road_links should avec the same crs, in meter'
        assert self.links.crs != 4326, 'CRS error, crs 4326 is not supported, use a crs in meter (NOT 3857)'

        # Reindex road links to integer
        if self.links.index.name in self.links.columns:
            # if index already in column, the reset index will bug. remove it first
            self.links = self.links.drop(columns=self.links.index.name)
        self.links = self.links.reset_index()
        self.links_index_dict = self.links['index'].to_dict()
        self.links_geom_dict = self.links.set_index('index')['geometry']  # this dict for later. link_x:linestring
        self.links = self.links.drop(columns=['index'])

        # Format links to a "gps track". keep node a,b of first links and node b of evey other ones.
        self.gps_tracks = links[['a', 'b', by, sequence, 'route_id']]
        node_dict = self.nodes['geometry'].to_dict()
        # gps_tracks['geometry'] = gps_tracks['b'].apply(lambda x: node_dict.get(x))
        self.gps_tracks.loc[:, 'node_seq'] = self.gps_tracks['b']
        # apply node a for the first link
        counter = self.gps_tracks.groupby(by).agg(len)['a'].values
        order = [i for j in range(len(counter)) for i in range(counter[j])]
        self.gps_tracks.loc[:, sequence] = order

        # for trip with single links, duplicate them to have a mapmatching between a and b.
        single_points = self.gps_tracks[self.gps_tracks[sequence] == 0]
        single_points.loc[:, 'node_seq'] = single_points.loc[:, 'a']
        single_points.loc[:, sequence] = -1
        single_points.index = 'node_a_' + single_points.index.map(str)

        self.gps_tracks = pd.concat([self.gps_tracks, single_points])

        # remove single links that a==b.
        self.gps_tracks = self.gps_tracks[self.gps_tracks['a'] != self.gps_tracks['b']]
        self.gps_tracks['geometry'] = self.gps_tracks['node_seq'].apply(lambda x: node_dict.get(x))

        self.gps_tracks = self.gps_tracks.sort_values([by, sequence])
        self.gps_tracks = gpd.GeoDataFrame(self.gps_tracks)
        self.gps_tracks = self.gps_tracks.drop(columns=['a', 'b', sequence])

    def Mapmatching(self, **kwargs):
        """
        Mapmatching
        :param kwargs: see _mapmatching
        :return: GeoDataFrame of the mapmatched gps tracks
        """
        return _mapmatching(self.links, **kwargs)

    def Multi_Mapmatching(self, workers, n_neighbors_centroid=100, **kwargs):
        """
        Multi Mapmatching
        :param workers: number of workers
        :type workers: int
        :param n_neighbors_centroid: number of neighbors to consider for the centroid
        :type n_neighbors_centroid: int
        :param kwargs: see _multi_mapmatching
        """
        # ====================
        # Map preparation
        # ====================
        self.links = add_geometry_coordinates(self.links, columns=['x_geometry', 'y_geometry'])
        # create sparse matrix of road network
        mat, node_index = sparse_matrix(self.links[['a', 'b', 'length']].values.tolist())
        # dict du node ID des road_id
        dict_node_a = self.links['a'].to_dict()
        dict_node_b = self.links['b'].to_dict()
        self.links['index'] = self.links.index
        dict_link = self.links.set_index(['a', 'b'], drop=False)['index'].to_dict()
        dict_link = self.links.set_index(['a', 'b'], drop=False)['index'].to_dict()
        length_dict = self.links['length'].to_dict()
        geom_dict = self.links['geometry'].to_dict()  # we need this dict with index 0,1,2,3,4,5...
        x = self.links[['x_geometry', 'y_geometry']].values
        # Fit Nearest neighbors model
        nbrs = NearestNeighbors(n_neighbors=n_neighbors_centroid, algorithm='ball_tree').fit(x)

        # Ray runs
        val_ids = []
        node_list_ids = []
        unmatched_trip_ids = []
        for track_ids in np.array_split(self.gps_tracks.trip_id.unique(), workers):
            sub_gps_tracks = self.gps_tracks[self.gps_tracks.trip_id.isin(track_ids)]
            # sub_gps_tracks = gpd.GeoDataFrame(sub_array, columns=self.gps_tracks.columns)
            val, node_list, unmatched_trip = _multi_mapmatching.remote(
                self.links, sub_gps_tracks, self.links_index_dict, mat, node_index,
                dict_node_a, dict_node_b, length_dict, geom_dict, dict_link,
                nbrs, **kwargs)
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
        self_links, self_gps_track, self_links_index_dict, mat,
        node_index, dict_node_a, dict_node_b, length_dict,
        geom_dict, dict_link, nbrs, routing=False,
        n_neighbors_centroid=100, n_neighbors=10,
        distance_max=200, by='trip_id'):
    """
    :param self_links: GeoDataFrame of the road network links
    :type self_links: GeoDataFrame
    :param self_gps_track: GeoDataFrame of the gps tracks
    :type self_gps_track: GeoDataFrame
    :param self_links_index_dict: dictionnary of road links index
    :type self_links_index_dict: dict
    :param mat: sparse matrix of the road network
    :type mat: sparse matrix
    :param node_index: dictionnary of node index
    :type node_index: dict
    :param dict_node_a: dictionnary of node a
    :type dict_node_a: dict
    :param dict_node_b: dictionnary of node b
    :type dict_node_b: dict
    :param length_dict: dictionnary of road links length
    :type length_dict: dict
    :param geom_dict: dictionnary of road links geometry
    :type geom_dict: dict
    :param dict_link: dictionnary of road links
    :type dict_link: dict
    :param nbrs: NearestNeighbors model
    :type nbrs: NearestNeighbors
    :param routing: True return the complete routing from the first to the last point on the road network (default = False)
    :type routing: bool
    :param n_neighbors_centroid: number of neighbors to consider for the centroid
    :type n_neighbors_centroid: int
    :param n_neighbors: number of neighbors to consider for the gps points
    :type n_neighbors: int
    :param distance_max: max radius to search candidat road for each gps point (default = 200m)
    :type distance_max: int
    :param by: column name of the gps tracks to group by

    :return: GeoDataFrame of the mapmatched gps tracks, GeoDataFrame of the mapmatched gps tracks nodes, list of unmatched trips

    """
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
                    self_links, gps_track, mat, node_index, dict_node_a,
                    dict_node_b, length_dict, geom_dict, dict_link, nbrs,
                    routing=routing, self_links_index_dict=self_links_index_dict,
                    n_neighbors_centroid=n_neighbors_centroid,
                    n_neighbors=n_neighbors, distance_max=distance_max)

                # add the by column to every data
                val[by] = trip_id
                node_list[by] = trip_id
                # apply input index
                val.index = val.index.map(gps_index_dict)
                node_list.index = node_list.index.map(gps_index_dict)
                vals = pd.concat([vals, val])
                node_lists = pd.concat([node_lists, node_list])
            except Exception as e:
                print(e)
                unmatched_trip.append(trip_id)
    return vals, node_lists, unmatched_trip


def _mapmatching(
        self_links, gps_track, mat, node_index, dict_node_a,
        dict_node_b, length_dict, geom_dict, dict_link, nbrs, self_links_index_dict,
        n_neighbors_centroid, n_neighbors=10, distance_max=1000,
        links_duplicated=False, normalized_offset=False, routing=False, plot=False):
    """
    gps_track: ordered list of geometry Point (in metre)
    nodes: data frame of street map nodes (crs in metre)
    distance_max: max radius to search candidat road for each gps point (default = 200m)
    links_duplicated:
    normalized_offset:
    routing: True return the complete routing from the first to the last point on the road network (default = False)

    Hidden Markov Map Matching Through Noise and Sparseness
        Paul Newson and John Krumm 2009
    :param self_links: GeoDataFrame of the road network links
    :type self_links: GeoDataFrame
    :param gps_track: GeoDataFrame of the gps tracks
    :type gps_track: GeoDataFrame
    :param mat: sparse matrix of the road network
    :type mat: sparse matrix
    :param node_index: dictionnary of node index
    :type node_index: dict
    :param dict_node_a: dictionnary of node a
    :type dict_node_a: dict
    :param dict_node_b: dictionnary of node b
    :type dict_node_b: dict
    :param length_dict: dictionnary of road links length
    :type length_dict: dict
    :param geom_dict: dictionnary of road links geometry
    :type geom_dict: dict
    :param dict_link: dictionnary of road links
    :type dict_link: dict
    :param nbrs: NearestNeighbors model
    :type nbrs: NearestNeighbors
    :param self_links_index_dict: dictionnary of road links index
    :type self_links_index_dict: dict
    :param n_neighbors_centroid: number of neighbors to consider for the centroid
    :type n_neighbors_centroid: int
    :param n_neighbors: number of neighbors to consider for the gps points
    :type n_neighbors: int
    :param distance_max: max radius to search candidat road for each gps point (default = 200m)
    :type distance_max: int
    :param links_duplicated:
    :type links_duplicated:
    :param normalized_offset:
    :type normalized_offset:
    :param routing: True return the complete routing from the first to the last point on the road network (default = False)
    :type routing: bool
    :param plot: True plot the mapmatching (default = False)
    :type plot: bool

    :return: GeoDataFrame of the mapmatched gps tracks, GeoDataFrame of the mapmatched gps tracks nodes
    """
    SIGMA = 4.07
    BETA = 3
    dijkstra_limit = 2000  # Limit on first dijkstra on road network.

    def nearest(one, nbrs, geometry=False):
        """
        Find the nearest road link for each gps point
        :param one: GeoDataFrame of the gps tracks
        :type one: GeoDataFrame
        :param nbrs: NearestNeighbors model
        :type nbrs: NearestNeighbors
        :param geometry: True return the geometry of the nearest road link (default = False)
        :type geometry: bool
        :return: GeoDataFrame of the nearest road link"""
        try:
            # Assert df_many.index.is_unique
            assert one.index.is_unique
        except AssertionError:
            msg = 'Index of one and many should not contain duplicates'
            print(msg)
            warnings.warn(msg)

        df_one = add_geometry_coordinates(one.copy())

        y = df_one[['x_geometry', 'y_geometry']].values

        distances, indices = nbrs.kneighbors(y)

        indices = pd.DataFrame(indices)
        distances = pd.DataFrame(distances)
        indices = pd.DataFrame(indices.stack(), columns=['index_nn']).reset_index().rename(
            columns={'level_0': 'ix_one', 'level_1': 'rank'}
        )
        return indices

    def emission_logprob(distance, SIGMA=SIGMA):
        """
        Compute the emission probability
        :param distance: distance between the gps point and the nearest road link
        :type distance: float
        :param SIGMA: constant
        :type SIGMA: float
        :return: emission probability"""
        # c = 1 / (SIGMA * np.sqrt(2 * np.pi))
        # return c*np.exp(-0.5*(distance/SIGMA)**2)
        # return -np.log10(np.exp(-0.5*(distance/SIGMA)**2))
        return 0.5 * (distance / SIGMA) ** 2  # Drop constant with log. its the same for everyone.

    def transition_logprob(dijkstra_dist, gps_dist, BETA=BETA):
        """
        Compute the transition probability
        :param dijkstra_dist: distance between the gps point and the nearest road link
        :type dijkstra_dist: float
        :param gps_dist: distance between the gps point and the nearest road link
        :type gps_dist: float
        :param BETA: constant
        :type BETA: float
        :return: transition probability
        """
        c = 1 / BETA
        delta = abs(dijkstra_dist - gps_dist)
        # return c * np.exp(-c * delta)
        return c * delta

    # process map
    try:
        self_links['length']
    except Exception:
        self_links['length'] = self_links.length

    gps_dict = gps_track['geometry'].to_dict()
    # GPS point distance to next point.
    gps_dist_dict = gps_track['geometry'].distance(gps_track.shift(-1)).to_dict()

    # ======================================================
    # Nearest roads and data preparation
    # ======================================================
    candidat_links = nearest(gps_track, nbrs).drop(columns=['rank'])

    def project(A, B, normalized=False):
        """
        Project a point on a line
        :param A: GeoDataFrame of the road links
        :type A: GeoDataFrame
        :param B: GeoDataFrame of the gps tracks
        :type B: GeoDataFrame
        :param normalized: True return the normalized offset (default = False)
        :type normalized: bool
        :return: offset
        """
        return [a.project(b, normalized=normalized) for a, b in zip(A, B)]

    def distance(A, B):
        """
        Compute the distance between two points
        :param A: GeoDataFrame of the road links
        :type A: GeoDataFrame
        :param B: GeoDataFrame of the gps tracks
        :type B: GeoDataFrame
        :return: distance"""
        return [a.distance(b) for a, b in zip(A, B)]

    candidat_links['road_geom'] = candidat_links['index_nn'].apply(lambda x: geom_dict.get(x))
    candidat_links['gps_geom'] = candidat_links['ix_one'].apply(lambda x: gps_dict.get(x))

    # Add gps distance to road.
    candidat_links['distance'] = distance(candidat_links['gps_geom'], candidat_links['road_geom'])

    candidat_links.sort_values(['ix_one', 'distance'], inplace=True)
    ranks = list(range(n_neighbors_centroid)) * len(gps_track)
    candidat_links['actual_rank'] = ranks
    candidat_links = candidat_links.loc[candidat_links['actual_rank'] < n_neighbors]
    candidat_links = candidat_links[candidat_links['distance'] < distance_max]
    candidat_links = candidat_links.reset_index().drop(columns=['index'])

    # Add offset
    candidat_links['offset'] = project(candidat_links['road_geom'], candidat_links['gps_geom'], normalized=False)
    dict_distance = candidat_links.set_index(['ix_one', 'index_nn'])['distance'].to_dict()

    # make tuple with road index and offset.
    candidat_links['index_nn'] = list(zip(candidat_links['index_nn'], candidat_links['offset']))
    candidat_links = candidat_links.drop(columns=['road_geom', 'gps_geom', 'offset', 'distance', 'actual_rank'])

    # add virtual nodes start and end.
    candidat_links.loc[len(candidat_links)] = [candidat_links['ix_one'].max() + 1, candidat_links['index_nn'].iloc[-1]]
    candidat_links.loc[-1] = [-1, candidat_links['index_nn'].iloc[0]]
    candidat_links.index = candidat_links.index + 1  # shifting index
    candidat_links = candidat_links.sort_index()  # sorting by index

    # dict of each linked point (ix_one). if pts 10 is NaN, point 9 will be linked to point 11
    dict_point_link = dict(
        zip(list(candidat_links['ix_one'].unique())[:-1], list(candidat_links['ix_one'].unique())[1:])
    )

    candidat_links = candidat_links.groupby('ix_one').agg(list)
    candidat_links = candidat_links.rename(columns={'index_nn': 'road_a'})
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
    candidat_links['node_b'] = candidat_links['road_a'].apply(lambda x: dict_node_b.get(x))
    candidat_links['node_a'] = candidat_links['road_b'].apply(lambda x: dict_node_a.get(x))

    # candidat_links = candidat_links.fillna(0)

    # Create sparse matrix of the road network
    try:  # for multi-mapmatching, feeding it as an input save time (it's the same mat every time)
        mat
    except Exception:
        mat, node_index = sparse_matrix(self_links[['a', 'b', 'length']].values.tolist())

    index_node = {v: k for k, v in node_index.items()}

    # liste des origines pour le dijkstra
    origin = list(candidat_links['node_b'].unique())
    origin_sparse = [node_index[x] for x in origin]

    # Dijktra on the road network from node = incices to every other nodes.
    # From b to a.
    dist_matrix, predecessors = dijkstra(
        csgraph=mat,
        directed=True,
        indices=origin_sparse,
        return_predecessors=True,
        limit=dijkstra_limit
    )

    dist_matrix = pd.DataFrame(dist_matrix)
    dist_matrix.index = origin

    # Dijkstra Destinations list
    destination = list(candidat_links['node_a'].unique())
    destination_sparse = [node_index[x] for x in destination]

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
        origin_sparse2 = [node_index[x] for x in unfound_origin_nodes]
        # Dijktra on the road network from node = incices to every other nodes.
        # from b to a.
        dist_matrix2, predecessors2 = dijkstra(
            csgraph=mat,
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
    candidat_links['length'] = candidat_links['road_a'].apply(lambda x: length_dict.get(x))

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
        self_links.loc[path].plot(ax=ax, color='red')
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
        df_path['sparse_node_b'] = df_path['road_id'].apply(lambda x: node_index.get(dict_node_b.get(x)))

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
                int(node_index[self_links.loc[df_path.iloc[-(1 + i + 1)]['road_id']]['a']])
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
                    link_list.append(self_links_index_dict[dict_link[node_list[i], node_list[i + 1]]])
                except Exception:
                    pass
            link_mat.append(link_list)

        # format en liste dans un dataframe
        node_mat = pd.Series(node_mat).to_frame('road_node_list')
        node_mat['road_link_list'] = link_mat

        if plot:
            f, ax = plt.subplots(figsize=(10, 10))
            self_links.plot(ax=ax, linewidth=1)
            gps_track.plot(ax=ax, marker='o', color='red', markersize=20)
            self_links.loc[node_list['road_id']].plot(ax=ax, color='orange', linewidth=2)
            plt.xlim([gps_track['geometry'].x.min() - 1000, gps_track['geometry'].x.max() + 1000])
            plt.ylim([gps_track['geometry'].y.min() - 100, gps_track['geometry'].y.max() + 1000])
            plt.show()
    return val, node_mat
