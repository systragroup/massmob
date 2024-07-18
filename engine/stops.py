import pandas as pd


def points_clean(points, phone_column='phone_id', time_column='ts'):
    """
    Drop duplicates (same phone_id and timestamp) and sort points 
    by phone_id and timestamp.
    """
    # clean points
    # I found some duplicates: same phone_id, same eventDate, but different coordinates
    points = points.drop_duplicates([phone_column, time_column], keep='first')  

    # sort by phone_id and ts
    points = points.sort_values([phone_column, time_column]).reset_index(drop=True)

    return points


def stops_append_d_s_t(points, phone_column='phone_id'):
    """
    Compute and append distance, time and speed from previous point to points dataframe
    """
    # compute and append distance, time and speed from previous point
    g = points
    d = (((g[['x', 'y']].diff())**2).T.sum()**0.5).fillna(0)
    t = g['ts'].diff().fillna(0)
    s = (d / t * 3.6).fillna(0)  # km/h
    points[['d', 't', 's']] = pd.concat([d, t, s], axis=1)

    # set to 0 d, t and s when first point of a new phone
    new_phone_loc = (g[phone_column] != g[phone_column].shift())
    points['new_phone'] = False
    points.loc[new_phone_loc, 'new_phone'] = True
    points.loc[new_phone_loc, ['d', 't', 's']] = 0

    return points


def stops_identify_noise_trips(points, IDLING_PHONE_METERS_DISTANCE, trip_column='trip_group', noise_column='noise_trip'):
    """
    Group stops by trip_group and compute spanned distance.
    Set noise_trip to True for points of trips with spanned distance below IDLING_PHONE_METERS_DISTANCE.
    """
    # assert x and y are in meters
    assert points.x.max() > 100, "x and y should be in meters coordinates"
    
    points[noise_column] = False
    # group edges not splitted by low speed and compute spanned distance
    trip_distance_x = points.groupby(trip_column)['x'].max() - points.groupby(trip_column)['x'].min()
    trip_distance_y = points.groupby(trip_column)['y'].max() - points.groupby(trip_column)['y'].min()
    trip_bounding_box_distance = (trip_distance_x**2 + trip_distance_y**2)**0.5
    false_positive_distance = trip_bounding_box_distance < IDLING_PHONE_METERS_DISTANCE
    points.loc[points[trip_column].isin(false_positive_distance[false_positive_distance].index), noise_column] = True

    return points


def stops_set_trace_id(df, stop_column, cut_column, sort_by=['phone_id', 'ts']):
    """
    add column with trace identifier from stop and cut columns
    # stop_column - 1:1, 2:0, 3:0, 4:1, 5:1, 6:0, 7:1, 8:0                       # initial stop_column
    # stop_column - 1:0, 1:1, 2:0, 3:0, 4:0, 4:1, 5:0, 5:1, 6:0, 7:0, 7:1, 8:0   # after adding duplicates
    # trace -         0,   1    1    1    1    2    2    3    3    3    4    4   # trace identifier
    """
    df['fake_points'] = False
    # we duplicate stops that are potential start and end of tracks
    dup = df.loc[df[stop_column]].copy()
    dup['fake_points'] = True
    dup[stop_column] = False
    df = pd.concat([df, dup]).sort_values(by=sort_by + [stop_column])
    # cumsum to tag each trace, taking into account both stop and cut columns
    df['track_id'] = (df[stop_column] | df[cut_column]).cumsum()
    return df


def stops_drop_short_trips(points, MIN_TRIP_DURATION_SECONDS, MIN_TRIP_DISTANCE_METERS, method='remove'):
    """
    Drop short trips:
    - trip with less than 2 points
    - trip with duration below MIN_TRIP_DURATION_SECONDS
    - trip with distance below MIN_TRIP_DISTANCE_METERS
    - method: 'remove' or 'stick' (stick to previous trip)
    """
    # more than 1 point
    points = points.loc[points['track_id'].isin((points.groupby('track_id')['ts'].count() > 1).index)]

    # duration filter
    keep_t = points.groupby('track_id')['t'].sum() > MIN_TRIP_DURATION_SECONDS

    # distance filter
    keep_d_x = (points.groupby('track_id')['x'].max() - points.groupby('track_id')['x'].min()) > MIN_TRIP_DISTANCE_METERS
    keep_d_y = (points.groupby('track_id')['y'].max() - points.groupby('track_id')['y'].min()) > MIN_TRIP_DISTANCE_METERS
    keep_d = keep_d_x | keep_d_y
    
    # apply filters
    if method == 'stick': # TODO: implement stick method to keep all points !!
        raise NotImplementedError("method 'stick' not implemented yet")
    
    elif method == 'remove':
        keep = keep_d & keep_t
        points = points.loc[points['track_id'].isin(keep[keep].index)]
    else:
        raise ValueError("method should be 'remove' or 'stick'")

    return points

def stops_untag_short_stops(points, MAKING_A_STOP_SECONDS_DELAY):
    """
    Untag short stops (below MAKING_A_STOP_SECONDS_DELAY) as stops
    """
    points['short_stop'] = False
    points['stop_group'] = (~points['stop']).cumsum()
    stop_durations = points.groupby('stop_group')['ts'].max() - points.groupby('stop_group')['ts'].min()
    short_stop = stop_durations < MAKING_A_STOP_SECONDS_DELAY
    points.loc[points['stop_group'].isin(short_stop[short_stop].index), 'short_stop'] = True

    points.loc[points['short_stop'], 'stop'] = False
    
    return points
