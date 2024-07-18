import pandas as pd
import geopandas as gpd
import numpy as np
import ciso8601
from shapely.geometry import LineString, Point
from fcdmodel.engine import stops


def tracks_from_points_with_stops(points):
    if 'geometry' not in points.columns:
        points['geometry'] = points[['x', 'y']].apply(Point, 1)

    # points.drop_duplicates(subset=['phone_id', 'geometry'], keep='first', inplace=True)

    points['duration'] = points['t']
    points['length'] = points['d']
    points = points.rename(
        columns={
            't': 'sampling_duration_median', 'd': 'sampling_distance_median',
            'accuracy': 'accuracy_median'
        }
    )
    points['point_ids'] = points.index

    points = points.groupby(['phone_id', 'track_id'], as_index=False).agg(
        {'accuracy_median':np.median, 'sampling_duration_median':np.median, 'sampling_distance_median': np.median,
        'duration': np.sum, 'length': np.sum, 
        'point_ids': list, 'geometry': lambda x: LineString(list(x))
        }
    )

    points['average_speed'] = points['length'] / points['duration']

    tracks = gpd.GeoDataFrame(points)

    return tracks


def analysis_tracks(tracks, points):
    accuracies = points['accuracy'].to_dict()
    durations = points['t'].to_dict()
    lengths = points['d'].to_dict()
    speeds = points['s'].to_dict()
    times = points['ts'].to_dict()
    tracks['accuracy_max'] = tracks['point_ids'].apply(lambda x: np.max([accuracies[i] for i in x]))
    tracks['accuracy_moy'] = tracks['point_ids'].apply(lambda x: np.mean([accuracies[i] for i in x]))
    tracks['sampling_duration_max'] = tracks['point_ids'].apply(lambda x: np.max([durations[i] for i in x]))
    tracks['sampling_duration_moy'] = tracks['point_ids'].apply(lambda x: np.mean([durations[i] for i in x]))
    tracks['sampling_distance_max'] = tracks['point_ids'].apply(lambda x: np.max([lengths[i] for i in x]))
    tracks['sampling_distance_moy'] = tracks['point_ids'].apply(lambda x: np.mean([lengths[i] for i in x]))
    tracks['speed_max'] = tracks['point_ids'].apply(lambda x: np.max([speeds[i] for i in x]))
    tracks['speed_median'] = tracks['point_ids'].apply(lambda x: np.median([speeds[i] for i in x]))
    tracks['speed_95th'] = tracks['point_ids'].apply(lambda x: np.percentile([speeds[i] for i in x], 95))
    tracks['first_ts'] = tracks['point_ids'].apply(lambda x: np.min([times[i] for i in x]))
    tracks['last_ts'] = tracks['point_ids'].apply(lambda x: np.max([times[i] for i in x]))
    tracks['departure_point'] = tracks['geometry'].apply(lambda line: Point(line.coords[0]))
    tracks['end_point'] = tracks['geometry'].apply(lambda line: Point(line.coords[-1]))

    return tracks


def build_tracked_points(points, MAX_SECONDS_DELAY_BETWEEN_POINTS =  60 * 60 ,    # délai maximum en minutes entre deux points consécutifs pouvant appartenir à une même trace
                    STOP_SPEED_THRESHOLD_KMH = 1,
                    IDLING_PHONE_METERS_DISTANCE = 200  ,
                    MAKING_A_STOP_SECONDS_DELAY = 10 * 60      ,                    
                    MIN_TRIP_DURATION_SECONDS = 60 * 2  ,                        # durée minimale en seconds d'une trace (non conservée en dessous)
                    MIN_TRIP_DISTANCE_METERS = 200):
    """
    Build the tracks and allocate the id of each tracks to the points.

    :param points: dataframe of points with columns ['phone_id', 'eventDate', 'x', 'y','ts'] (ts is the timestamp in seconds)
    :type points: pandas.DataFrame
    :param MAX_SECONDS_DELAY_BETWEEN_POINTS: maximum delay in seconds between two consecutive points that can belong to the same trace
    :type MAX_SECONDS_DELAY_BETWEEN_POINTS: int
    :param STOP_SPEED_THRESHOLD_KMH: speed threshold in km/h below which a point is considered as a stop
    :type STOP_SPEED_THRESHOLD_KMH: int
    :param IDLING_PHONE_METERS_DISTANCE: distance in meters below which a stop is considered as a phone idling
    :type IDLING_PHONE_METERS_DISTANCE: int
    :param MAKING_A_STOP_SECONDS_DELAY: delay in seconds below which a stop is considered as a making a stop
    :type MAKING_A_STOP_SECONDS_DELAY: int
    :param MIN_TRIP_DURATION_SECONDS: minimum duration in seconds of a trace (not kept below)
    :type MIN_TRIP_DURATION_SECONDS: int
    :param MIN_TRIP_DISTANCE_METERS: minimum distance in meters of a trace (not kept below)
    :type MIN_TRIP_DISTANCE_METERS: int
    :return: dataframe of tracks with the Linestring geometry of each trace

    """
    
    points = stops.points_clean(points)
    points = stops.stops_append_d_s_t(points)

    ## tag beginning of new trace if interval is above MAX_SECONDS_DELAY_BETWEEN_POINTS
    points['duration_threshold_exceeded'] = False
    points.loc[points['t'] > MAX_SECONDS_DELAY_BETWEEN_POINTS, 'duration_threshold_exceeded'] = True

    # set d, t, s value of each first point of a new sequence to 0
    points.loc[points['duration_threshold_exceeded'], ['d', 't', 's']] = 0

    points['cut'] = points['duration_threshold_exceeded'] | points['new_phone']
    points = points.drop(['duration_threshold_exceeded', 'new_phone'], axis=1)

    points['low_speed'] = False
    low_speed_loc = points['s'] < STOP_SPEED_THRESHOLD_KMH
    points.loc[low_speed_loc, 'low_speed'] = True
    points['trip_group'] = (points['cut'] | points['low_speed']).cumsum()

    points = stops.stops_identify_noise_trips(points, IDLING_PHONE_METERS_DISTANCE)

    points['stop'] = points['low_speed'] | points['noise_trip']
    points['short_stop'] = False
    points['stop_group'] = (~points['stop']).cumsum()
    stop_durations = points.groupby('stop_group')['ts'].max() - points.groupby('stop_group')['ts'].min()
    short_stop = stop_durations < MAKING_A_STOP_SECONDS_DELAY
    points.loc[points['stop_group'].isin(short_stop[short_stop].index), 'short_stop'] = True

    points.loc[points['short_stop'], 'stop'] = False
    points = points.drop(['low_speed', 'trip_group', 'noise_trip', 'short_stop', 'stop_group'], axis=1)

    def _set_trace_id(df, stop_column, cut_column, sort_by=['phone_id', 'ts']):
        """
        add column with trace identifier from stop and cut columns
        # stop_column - 1:1, 2:0, 3:0, 4:1, 5:1, 6:0, 7:1, 8:0                       # initial stop_column
        # stop_column - 1:0, 1:1, 2:0, 3:0, 4:0, 4:1, 5:0, 5:1, 6:0, 7:0, 7:1, 8:0   # after adding duplicates
        # trace -         0,   1    1    1    1    2    2    3    3    3    4    4   # trace identifier
        """
        # we duplicate stops that are potential start and end of tracks
        dup = df.loc[df[stop_column]].copy()
        dup[stop_column] = False
        df = pd.concat([df, dup]).sort_values(by=sort_by + [stop_column])
        # cumsum to tag each trace, taking into account both stop and cut columns
        df['track_id'] = (df[stop_column] | df[cut_column]).cumsum()

        return df
    points = _set_trace_id(points, 'stop', 'cut', sort_by=['phone_id', 'ts'])
    
    points = points.loc[points['track_id'].isin((points.groupby('track_id')['ts'].count() > 1).index)]
    keep_t = points.groupby('track_id')['t'].sum() > MIN_TRIP_DURATION_SECONDS

    keep_d_x = (points.groupby('track_id')['x'].max() - points.groupby('track_id')['x'].min()) > MIN_TRIP_DISTANCE_METERS
    keep_d_y = (points.groupby('track_id')['y'].max() - points.groupby('track_id')['y'].min()) > MIN_TRIP_DISTANCE_METERS
    keep_d = keep_d_x | keep_d_y

    keep = keep_d & keep_t
    points = points.loc[points['track_id'].isin(keep[keep].index)]
    points['eventDate8601'] = points['eventDate'].apply(lambda x:  ciso8601.parse_datetime(str(x).split(' UTC')[0].replace(' ', 'T')))
    points['day'] = points['eventDate8601'].apply(lambda x: f"{x.year}-{x.month}-{x.day}")

    tracks = points.copy()
    # remove type category
    tracks['phone_id'] = tracks['phone_id'].astype(str)
    # tracks = tracks.drop('phone_id', axis=1)
    
    return tracks

def filtering(pts, phone_id_column='phone_id', INACTIVE_PHONE_AREA_SIDE_METERS = 50, MAX_ACCURACY=50):
    
    """Filtre les points pour ne garder que ceux qui sont suffisamment éloignés les uns des autres
    
    :param pts: dataframe of points with columns [phone_id_column, 'eventDate', 'longitude', 'latitude']
    :type pts: pandas.DataFrame
    :param INACTIVE_PHONE_AREA_SIDE_METERS: distance in meters below which a phone is considered as inactive
    :type INACTIVE_PHONE_AREA_SIDE_METERS: int
    :return: dataframe of points with columns [phone_id_column, 'eventDate', 'longitude', 'latitude','ts']
    """
    df = pts.drop(['speed', 'eventId', 'Unnamed: 0'], axis=1, errors='ignore')
    df[phone_id_column] = df[phone_id_column].astype("category")
    del pts
    temp = gpd.GeoSeries([LineString(df[['longitude', 'latitude' ]].values)]).set_crs(epsg=4326) 
    temp = temp.to_crs(epsg=2154)
    df[['x', 'y']] = temp.geometry[0].coords[:]

    # optimize memory usage: convert to integer and drop unused columns
    del temp
    df[['x', 'y']] = df[['x', 'y']].astype(int)
    
    active_phones = df.groupby(phone_id_column).apply(_is_phone_moving_enough, INACTIVE_PHONE_AREA_SIDE_METERS)

    df_active = df[df[phone_id_column].isin(active_phones[active_phones].index)].copy()

    df_active['eventDate8601'] = df_active['eventDate'].apply(lambda x: str(x).split(' UTC')[0].replace(' ', 'T'))
    df_active['ts'] = df_active['eventDate8601'].apply(lambda x: ciso8601.parse_datetime(x).timestamp())
    df_active['ts'] = df_active['ts'].astype(int)
    df_active['accuracy'] = df_active['accuracy'].astype(int)
    df_active.drop(['eventDate8601'], axis=1, inplace=True)
    
    df_active = df_active[df_active.accuracy < MAX_ACCURACY]

    return df_active

def _is_phone_moving_enough(df_temp, INACTIVE_PHONE_AREA_SIDE_METERS):
        """
        Retourne "True" si tous les points (d'un téléphone donné = d'un phone_id donné)
        ne sont pas contenus dans un carré de INACTIVE_PHONE_AREA_SIDE_METERS mètres de côté.
        Autrement dit, détermine si un téléphone a donné des points qui valent le coup d'être calculés en tracks,
        et ne sont pas seulement tous des points quasi-immobiles dans une petite zone. 
        :param df_temp: dataframe of points with columns ['phone_id', 'eventDate', 'longitude', 'latitude']
        :type df_temp: pandas.DataFrame
        :param INACTIVE_PHONE_AREA_SIDE_METERS: distance in meters below which a phone is considered as inactive
        :type INACTIVE_PHONE_AREA_SIDE_METERS: int
        :return: True if the phone is moving enough, False otherwise
        """
        x_d = df_temp['x'].max() - df_temp['x'].min()
        y_d = df_temp['y'].max() - df_temp['y'].min()
        d = max(x_d, y_d)
        return(d > INACTIVE_PHONE_AREA_SIDE_METERS)


#### OLD VERSION BELOW ? ####

# def _v_instant(row):
#     """
#     Calcule la vitesse instantanée en km/h
#     :param row: ligne du dataframe
#     :type row: pandas.Series
#     :return: vitesse instantanée en km/h
#     """
#     if row['time_elapsed'] ==0:
#         return 0
#     else:
#         return(row['dist_from_prec'] / row['time_elapsed'] * 3.6)
    

def _acc_instant(row):
    if row['time_elapsed'] ==0:
        return 0
    else:
        return(row['delta_v_from_prec'] / row['time_elapsed']/3.6)


# def _calc_time_elapsed(row):
#     """
#     Calcule le temps écoulé entre le point courant et le point précédent
#     :param row: ligne du dataframe
#     :type row: pandas.Series
#     :return: temps écoulé entre le point courant et le point précédent

#     """
#     try:
#         if row['track_id'] != row['trace_id_prec']:
#             time_elapsed = 0
#             return time_elapsed
#         else:
#             time_elapsed = (row['eventDate8601'] - row['time_prec']).total_seconds()
#             if time_elapsed < 0:
#                 time_elapsed = 0
#             return time_elapsed
#     except:
#         time_elapsed = 0
#         return time_elapsed

# def _calc_dist_from_prec(row):
#     """
#     Calcul de la distance entre le point courant et le point précédent
#     :param row: ligne du dataframe
#     :type row: pandas.Series
#     :return: distance entre le point courant et le point précédent

#     """
#     try:
#         if row['track_id'] != row['trace_id_prec']:
#             dist = 0
#             return dist
#         else:
#             dist = row['geometry'].distance(row['loc_prec'])
#             if dist < 0:
#                 dist = 0
#             return dist
#     except:
#         dist = 0
#         return dist
    
# def _calc_delta_v_from_prec(row):
#     """
#     Calcul de la distance entre le point courant et le point précédent
#     :param row: ligne du dataframe
#     :type row: pandas.Series
#     :return: distance entre le point courant et le point précédent

#     """
#     try:
#         if row['track_id'] != row['trace_id_prec']:
#             delta_v = 0
#             return delta_v
#         else:
#             deta_v = row['v_instant'] - row['v_instant_prec']
#             if deta_v < 0:
#                 deta_v = 0
#             return deta_v
#     except:
#         dist = 0
#         return dist


def series_point_to_linestring(series):
    liste = list(series)
    if len(liste)>1:
        linestring = shapely.geometry.LineString(liste)
        return linestring


def _to_list(series):
    liste = list(series)
    return liste


def _to_list_cumul(series):
    #renvoie la liste en cumulant les valeur
    liste = list(series)
    liste_cumul = []
    for i in range(len(liste)):
        liste_cumul.append(sum(liste[:i+1]))
    return liste_cumul


def points_to_tracks(points):
    # """
    # Convertit un dataframe de points en un dataframe de tracks.
    # :param points: dataframe of points with columns ['phone_id', 'eventDate', 'x', 'y','ts']
    # :type points: pandas.DataFrame
    # :return: dataframe of tracks with the Linestring geometry of each trace

    # """
   
    points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.x, points.y), crs ='epsg:2154')
    points.drop_duplicates(subset=['phone_id', 'geometry'], keep='first', inplace=True)
    points['time_prec']= points['eventDate8601'].shift(1)
    points['trace_id_prec']= points['track_id'].shift(1)
    points['loc_prec']= points['geometry'].shift(1)
    points['time_elapsed'] = points.apply(_calc_time_elapsed, axis=1)
    points['dist_from_prec'] = points.apply(_calc_dist_from_prec, axis=1)
    points.dropna(inplace=True)
    points['v_instant'] = points.apply(_v_instant, axis=1)
    points['accuracy_max'] = points['accuracy']
    points['accuracy_moy'] = points['accuracy']
    points['time_prec_moy'] = points['time_elapsed']
    points['time_prec_max'] = points['time_elapsed']
    points['list_time_cumul'] = points['time_elapsed']
    points['list_v_instant'] = points['v_instant']
    points['list_time_elapsed'] = points['time_elapsed']
    points['first_ts'] = points['eventDate8601']
    points['last_ts'] = points['eventDate8601']
    points['dist_from_prec_max'] = points['dist_from_prec']
    points['dist_from_prec_moy'] = points['dist_from_prec']
    points['v_instant_max'] = points['v_instant']
    points['v_instant_moy'] = points['v_instant']
    points['v_median']= points['v_instant']
    points['v_95th_percentile']= points['v_instant']
    points['departure_point'] = points['geometry']
    points['arrival_point'] = points['geometry']
    points['linestring'] = points['geometry']
    points['duration'] = points['time_elapsed']
    points['track_length'] = points['dist_from_prec']
    points['day'] = points['eventDate8601'].apply(lambda x: f"{x.year}-{x.month}-{x.day}")
    #Calcul des champs utiles par id tracks

    points = points.groupby('track_id').agg({'accuracy_max':np.max,'accuracy_moy':np.mean,'time_prec_moy':np.mean,'time_prec_max':np.max,'list_v_instant':_to_list,'list_time_elapsed':_to_list,'v_median':np.median,'v_95th_percentile':lambda x: np.percentile(x,95),'dist_from_prec_max':np.max,'dist_from_prec_moy':np.mean,'v_instant_max':np.max,'v_instant_moy':np.mean,'departure_point':'first','arrival_point':'last','first_ts':'first','last_ts':'last','linestring':series_point_to_linestring,'duration':np.sum,'track_length':np.sum, 'day':'first','phone_id':'first'})
    points.reset_index(inplace=True)
    points['dist_departure_arrival'] = points.apply(lambda x: x['departure_point'].distance(x['arrival_point']), axis=1)
 
    tracks_linestring_processed = gpd.GeoDataFrame(points, geometry='linestring', crs='EPSG:2154') #Conversion GDF
    tracks_linestring_processed = pd.DataFrame(tracks_linestring_processed)
    tracks_linestring_processed.rename(columns={'linestring':'geometry'}, inplace=True)
    tracks_linestring_processed = gpd.GeoDataFrame(tracks_linestring_processed, geometry='geometry', crs='EPSG:2154') #Conversion GDF
    # tracks_linestring_processed['dist_departure_arrival'] = linestring_processed.apply(lambda x: x['departure_point'].distance(x['arrival_point']), axis=1)
    return tracks_linestring_processed
