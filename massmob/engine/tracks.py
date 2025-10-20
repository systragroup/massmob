import pandas as pd
import geopandas as gpd
import polars as pl
import numpy as np
from pyproj import Transformer
import ciso8601
import time
import numpy as np
import ciso8601
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from massmob.engine import stops, utils


def tracks_from_points_with_stops(points: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates point-level tracking data to track-level summaries using Polars.

    Each unique pair of 'phone_id' and 'track_id' defines a distinct track.
    For each track, aggregate statistics are computed:
    - Median accuracy
    - Median sampling duration
    - Median sampling distance
    - Total duration (sum)
    - Total length (sum)
    - List of point indices
    - Ordered list of (x, y) coordinate tuples for downstream use (e.g., as a LineString or similar)

    Parameters
    ----------
    points : pl.DataFrame
        A Polars DataFrame with at least the following columns:
        ['phone_id', 'track_id', 't', 'd', 'accuracy', 'duration', 'length', 'x', 'y']
        where:
        - 't' is sampling duration
        - 'd' is sampling distance
        - 'accuracy' is accuracy measurement
        - 'duration' and 'length' are per-point duration and distance
        - 'x', 'y' are coordinates

    Returns
    -------
    pl.DataFrame
        Aggregated DataFrame at track level with columns:
        ['phone_id', 'track_id', 'accuracy_median', 'sampling_duration_median',
         'sampling_distance_median', 'duration', 'length', 'point_ids', 'coordinates',
         'average_speed']
    """
    # Add a unique index to identify points within the track (analogous to point_ids)
    points = points.with_row_count("point_ids")

    # Rename columns for clearer semantics in downstream processing
    points = points.rename({
        "t": "duration",
        "d": "length",
        "accuracy": "accuracy_median"
    })

    # Group by phone_id and track_id, then aggregate relevant statistics and lists
    tracks = points.group_by(["phone_id", "track_id"]).agg([
        # Median accuracy per track
        pl.col("accuracy_median").median().alias("accuracy_median"),
        # Median sampling duration per track
        pl.col("duration").median().alias("sampling_duration_median"),
        # Median sampling distance per track
        pl.col("length").median().alias("sampling_distance_median"),
        # Total duration and length per track
        pl.col("duration").sum().alias("duration"),
        pl.col("length").sum().alias("length"),
        # List of structs representing (x, y) coordinates in order
        pl.struct(["x", "y"]).implode().alias("coordinates"),
        pl.col("point_id").implode().alias("point_ids")
    ])

    # Compute average speed (length divided by duration) for each track
    tracks = tracks.with_columns([
        (pl.col("length") / pl.col("duration")).alias("average_speed")
    ])

    return tracks


def analysis_tracks(tracks: pl.DataFrame, points: pl.DataFrame, point_id_col="point_id"):
    """
    Compute summary statistics for each track based on its related points.

    For every track, aggregate statistics (accuracy, duration, length, speed, timestamps)
    from the associated points listed in the 'point_ids' column.
    The departure and end points are returned as raw coordinate tuples, 
    taken from the 'coordinates' column, which should be a list of (x, y) tuples for each track.

    Parameters
    ----------
    tracks : pl.DataFrame
        Contains one row per track, with at least:
            - 'point_ids': list of ids referencing points for the track
            - 'coordinates': list of coordinate tuples for the track geometry
    points : pl.DataFrame
        Contains per-point data, must include:
            - 'accuracy', 't', 'd', 's', 'ts', and the column with id 'point_id_col'
    point_id_col : str, default "point_id"
        Name of the column in 'points' identifying each point

    Returns
    -------
    pl.DataFrame
        Input tracks DataFrame with additional columns containing aggregated statistics for each track.
        Departure and end points are given as coordinate tuples (x, y), not as geometric objects.
    """

    # Build fast lookup dicts: point_id -> value
    accuracy_map = dict(zip(points[point_id_col].to_list(), points['accuracy'].to_list()))
    t_map = dict(zip(points[point_id_col].to_list(), points['t'].to_list()))
    d_map = dict(zip(points[point_id_col].to_list(), points['d'].to_list()))
    s_map = dict(zip(points[point_id_col].to_list(), points['s'].to_list()))
    ts_map = dict(zip(points[point_id_col].to_list(), points['ts'].to_list()))
    day_map = dict(zip(points[point_id_col].to_list(), points['day'].to_list()))

    tracks = tracks.with_columns([
         # Accuracy statistics
        pl.col('point_ids').map_elements(lambda ids: max(accuracy_map[i] for i in ids), return_dtype=pl.Float64).alias('accuracy_max'),
        pl.col('point_ids').map_elements(lambda ids: sum(accuracy_map[i] for i in ids) / len(ids), return_dtype=pl.Float64).alias('accuracy_moy'),

        # Duration statistics
        pl.col('point_ids').map_elements(lambda ids: max(t_map[i] for i in ids), return_dtype=pl.Float64).alias('sampling_duration_max'),
        pl.col('point_ids').map_elements(lambda ids: sum(t_map[i] for i in ids) / len(ids), return_dtype=pl.Float64).alias('sampling_duration_moy'),

        # Distance statistics
        pl.col('point_ids').map_elements(lambda ids: max(d_map[i] for i in ids), return_dtype=pl.Float64).alias('sampling_distance_max'),
        pl.col('point_ids').map_elements(lambda ids: sum(d_map[i] for i in ids) / len(ids), return_dtype=pl.Float64).alias('sampling_distance_moy'),

        # Speed statistics
        pl.col('point_ids').map_elements(lambda ids: max(s_map[i] for i in ids), return_dtype=pl.Float64).alias('speed_max'),
        pl.col('point_ids').map_elements(lambda ids: sorted(s_map[i] for i in ids)[len(ids) // 2], return_dtype=pl.Float64).alias('speed_median'),
        pl.col('point_ids').map_elements(lambda ids: sorted(s_map[i] for i in ids)[int(0.95 * (len(ids)-1))], return_dtype=pl.Float64).alias('speed_95th'),

        # Timestamp statistics
        pl.col('point_ids').map_elements(lambda ids: min(ts_map[i] for i in ids), return_dtype=pl.Int64).alias('first_ts'),
        pl.col('point_ids').map_elements(lambda ids: max(ts_map[i] for i in ids), return_dtype=pl.Int64).alias('last_ts'),

        # day statistics
        pl.col('point_ids').map_elements(lambda ids: min(day_map[i] for i in ids), return_dtype=pl.String).alias('departure_day'),

        # Departure and end raw coordinate tuples (not geometric objects)
        pl.col('coordinates').list.first().alias('departure_point'),
        pl.col('coordinates').list.last().alias('end_point'),
    ])
    return tracks

def apply_time_and_phone_cut(points: pl.DataFrame, max_delay: int) -> pl.DataFrame:
    """
    Marks trajectory cuts based on excessive time delay between points or phone change.
    Sets values (d, t, s) to 0 if the threshold is exceeded.

    Args:
        points (pl.DataFrame): DataFrame of points with columns 't', 'd', 's', 'new_phone'.
        max_delay (int): Maximum allowed time between two points before forcing a cut.

    Returns:
        pl.DataFrame: Updated DataFrame with modified 'd', 't', 's', and a boolean 'cut' column.
    """
    # Flag when excessive time between points
    points = points.with_columns([
        (pl.col('t') > max_delay).alias('duration_threshold_exceeded')
    ])
    # Reset duration, distance, speed where needed
    points = points.with_columns([
        pl.when(pl.col('duration_threshold_exceeded')).then(0).otherwise(pl.col('d')).alias('d'),
        pl.when(pl.col('duration_threshold_exceeded')).then(0).otherwise(pl.col('t')).alias('t'),
        pl.when(pl.col('duration_threshold_exceeded')).then(0).otherwise(pl.col('s')).alias('s')
    ])
    # Mark cut column if excessive delay or new phone
    points = points.with_columns([
        (pl.col('duration_threshold_exceeded') | pl.col('new_phone')).alias('cut')
    ]).drop(['duration_threshold_exceeded', 'new_phone'])
    return points

def apply_user_cut(points: pl.DataFrame, cut_at_points: list = None, cut_id_col: str = None) -> pl.DataFrame:
    """
    Marks trajectory cuts at user-specified positions (row indices or column values).

    Args:
        points (pl.DataFrame): Input DataFrame with 'cut' column.
        cut_at_points (list): List of row indices or values in cut_id_col.
        cut_id_col (str): If specified, indicates the column for cut values.

    Returns:
        pl.DataFrame: Updated DataFrame with modified 'cut' column.
    """
    if cut_at_points is not None and len(cut_at_points) > 0:
        if cut_id_col is None:
            # By row indices (mask)
            mask = pl.Series([i in cut_at_points for i in range(points.height)])
            points = points.with_columns([
                (pl.col('cut') | mask).alias('cut')
            ])
        else:
            # By custom column values
            points = points.with_columns([
                (pl.col('cut') | pl.col(cut_id_col).is_in(cut_at_points)).alias('cut')
            ])
    return points

def mark_lowspeed(points: pl.DataFrame, stop_speed_threshold: float) -> pl.DataFrame:
    """
    Marks points as 'low_speed' if their speed is below threshold.

    Args:
        points (pl.DataFrame): DataFrame with 's' (speed) column.
        stop_speed_threshold (float): Speed threshold for stops (in km/h).

    Returns:
        pl.DataFrame: Updated with boolean 'low_speed' column.
    """
    points = points.with_columns([
        (pl.col('s') < stop_speed_threshold).alias('low_speed')
    ])
    return points

def assign_trip_group(points: pl.DataFrame) -> pl.DataFrame:
    """
    Assigns a temporary trip group based on the cumulated number of cuts or low speed.

    Args:
        points (pl.DataFrame): DataFrame with 'cut' and 'low_speed' columns.

    Returns:
        pl.DataFrame: Updated with integer 'trip_group' column.
    """
    points = points.with_columns([
        ((pl.col('cut') | pl.col('low_speed')).cast(pl.Int32).cum_sum()).alias('trip_group')
    ])
    return points

def mark_stop_and_noise(points: pl.DataFrame, idling_distance: float) -> pl.DataFrame:
    """
    Identifies noisy (idle) points and marks all stops.

    Args:
        points (pl.DataFrame): DataFrame with 'low_speed' column.
        idling_distance (float): Distance threshold for phone idling (in meters).

    Returns:
        pl.DataFrame: Updated with boolean 'noise_trip' and 'stop' columns.
    """
    # Use external logic for idling phone
    points = stops.stops_identify_noise_trips(points, idling_distance)
    points = points.with_columns([
        (pl.col('low_speed') | pl.col('noise_trip')).alias('stop')
    ])
    return points

def assign_stop_group(points: pl.DataFrame) -> pl.DataFrame:
    """
    Assigns a stop group ID. Each segment between stops receives a unique ID.

    Args:
        points (pl.DataFrame): DataFrame with 'stop' column.

    Returns:
        pl.DataFrame: Updated with integer 'stop_group' column.
    """
    points = points.with_columns([
        ((~pl.col('stop')).cast(pl.Int32).cum_sum()).alias('stop_group')
    ])
    return points

def remove_short_stops(points: pl.DataFrame, min_stop_duration: float) -> pl.DataFrame:
    """
    Identifies and removes (ignores) stops with a duration below the configured minimum.

    Args:
        points (pl.DataFrame): DataFrame with 'stop_group' and 'stop' columns.
        min_stop_duration (float): Minimal stop duration (seconds).

    Returns:
        pl.DataFrame: With filtered 'stop' column, and removes helper columns.
    """
    # Compute duration per stop group
    stop_durations = (
        points.group_by('stop_group')
        .agg((pl.col('ts').max() - pl.col('ts').min()).alias('duration'))
    )
    # Identify short stops
    short_stop_groups = stop_durations.filter(
        pl.col('duration') < min_stop_duration
    )['stop_group']
    # Flag stop points belonging to short stops
    points = points.with_columns([
        pl.col('stop_group').is_in(short_stop_groups).alias('short_stop'),
        pl.when(pl.col('stop_group').is_in(short_stop_groups))
          .then(False).otherwise(pl.col('stop')).alias('stop')
    ]).drop(['low_speed', 'trip_group', 'noise_trip', 'short_stop', 'stop_group'])
    return points

def assign_track_id(points: pl.DataFrame) -> pl.DataFrame:
    """
    Assigns a unique trajectory (track) id for each valid trip segment.

    Args:
        points (pl.DataFrame): DataFrame with 'stop' and 'cut' columns.

    Returns:
        pl.DataFrame: Updated with 'track_id' column.
    """
    # Duplicate stop points for tracking logic (fake_points)
    points = points.with_columns([pl.lit(False).alias('fake_points')])
    dup = points.filter(pl.col("stop")).with_columns([
        pl.lit(True).alias('fake_points'),
        pl.lit(False).alias('stop')
    ])
    # Concatenate original and duplicated points, sort for correct trace assignment
    points = pl.concat([points, dup])
    points = points.sort(['phone_id', 'ts', 'stop'])
    # Assigns track_id as cumulative sum of stop OR cut event
    points = points.with_columns([
        ((pl.col('stop').cast(pl.Int32) | pl.col('cut').cast(pl.Int32))
          .cum_sum()
          .alias('track_id')
        )
    ])
    return points

def filter_tracks(points: pl.DataFrame, min_trip_duration: float, min_trip_span: float) -> pl.DataFrame:
    """
    Removes insignificant tracks: too short in time, or too spatially limited, or with too few points.

    Args:
        points (pl.DataFrame): DataFrame including 'track_id', 't', 'x', 'y', 'ts'.
        min_trip_duration (float): Minimum trip duration (in seconds).
        min_trip_span (float): Minimal trip span (in meters).

    Returns:
        pl.DataFrame: Filtered DataFrame with valid tracks only.
    """
    # Gather statistics for each track
    trip_stats = (
        points.group_by("track_id")
        .agg([
            pl.count().alias("n_pts"),
            (pl.col("ts").max() - pl.col("ts").min()).alias("duration"),
            pl.col("t").sum().alias("trip_t"),
            (pl.col("x").max() - pl.col("x").min()).alias("span_x"),
            (pl.col("y").max() - pl.col("y").min()).alias("span_y"),
        ])
    )
    # Filter on minimal number of points, distance, and duration
    valid_tracks = trip_stats.filter(
        (pl.col("n_pts") > 1)
        & ((pl.col("span_x") > min_trip_span) | (pl.col("span_y") > min_trip_span))
        & (pl.col("trip_t") > min_trip_duration)
    )["track_id"]
    points = points.filter(pl.col("track_id").is_in(valid_tracks))
    return points

def add_local_day(points: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a readable 'day' column derived from 'eventDate' and ensures that phone_id is a string.

    Args:
        points (pl.DataFrame): DataFrame including 'eventDate' and 'phone_id'.

    Returns:
        pl.DataFrame: DataFrame with 'day' and string-typed 'phone_id'.
    """
    # TODO: je crois que c’est en UTC -> à corriger
    points = points.with_columns([
        (
            pl.col("eventDate").dt.year().cast(pl.Utf8) + "-" +
            pl.col("eventDate").dt.month().cast(pl.Utf8).str.zfill(2) + "-" +
            pl.col("eventDate").dt.day().cast(pl.Utf8).str.zfill(2)
        ).alias("day")
    ])
    points = points.with_columns([pl.col("phone_id").cast(pl.Utf8)])
    return points

def build_tracked_points(
    points: pl.DataFrame,
    max_seconds_delay_between_points: int = 60 * 60,
    stop_speed_threshold_kmh: float = 1,
    idling_phone_meters_distance: float = 200,
    making_a_stop_seconds_delay: float = 10 * 60,
    min_trip_duration_seconds: float = 60 * 2,
    min_trip_distance_meters: float = 200,
    cut_at_points: list = None,
    cut_id_col: str = None
) -> pl.DataFrame:
    """
    Identifies, segments, and filters valid trips in a Polars DataFrame of timestamped geolocated points.
    Cuts trajectories based on time/phone/user rules, applies stop detection and noise filtering,
    assigns unique trip identifiers, and removes trivial or noisy traces.

    Args:
        points (pl.DataFrame): Input Polars DataFrame of points.
        max_seconds_delay_between_points (int): Max time allowed between points before cut.
        stop_speed_threshold_kmh (float): Speed threshold (in km/h) to consider as stopped.
        idling_phone_meters_distance (float): Distance threshold for idling noise.
        making_a_stop_seconds_delay (float): Minimum stop duration to be considered valid.
        min_trip_duration_seconds (float): Minimum duration of trip to keep.
        min_trip_distance_meters (float): Minimum spatial span of trip to keep.
        cut_at_points (list): (Optional) List of row indices or IDs to force a cut.
        cut_id_col (str): (Optional) Column name if cut_at_points are IDs.

    Returns:
        pl.DataFrame: DataFrame segmented and filtered, ready for further trip analysis.
    """
    # Clean and enrich input
    points = stops.clean_points(points)
    points = stops.stops_append_d_s_t(points)
    # Apply cut logic based on time, phone, and user-requested cuts
    points = apply_time_and_phone_cut(points, max_seconds_delay_between_points)
    points = apply_user_cut(points, cut_at_points, cut_id_col)
    # Stop detection logic
    points = mark_lowspeed(points, stop_speed_threshold_kmh)
    points = assign_trip_group(points)
    points = mark_stop_and_noise(points, idling_phone_meters_distance)
    points = assign_stop_group(points)
    points = remove_short_stops(points, making_a_stop_seconds_delay)
    # Assign trip/segment IDs
    points = assign_track_id(points)
    # Filter irrelevant or noisy tracks
    points = filter_tracks(points, min_trip_duration_seconds, min_trip_distance_meters)
    # Final parsing and day annotation
    points = add_local_day(points)
    return points

def filtering(
    pts: pl.DataFrame,
    phone_id_column='phone_id',
    INACTIVE_PHONE_AREA_SIDE_METERS = 50,
    MAX_ACCURACY = 50,
    time_column="eventDate"
):
    """
    Filter points to keep only those that are sufficiently far from each other (phones not considered inactive).
    """
    pts = stops.clean_points(pts, phone_column=phone_id_column, time_column=time_column)
    # Drop useless columns if present
    cols_to_drop = [col for col in ['speed', 'eventId', 'Unnamed: 0'] if col in pts.columns]
    pts = pts.drop(cols_to_drop)
    # TODO: ajout gestion CRS lors de l’import
    # Prepare transformer for lon/lat --> x/y in Lambert-93 (EPSG:2154)
    transformer = Transformer.from_crs("epsg:4326", "epsg:2154", always_xy=True)
    # Vectorized conversion
    x, y = transformer.transform(pts["longitude"].to_numpy(), pts["latitude"].to_numpy())
    pts = pts.with_columns([
        pl.Series("x", np.array(x).astype(int)),
        pl.Series("y", np.array(y).astype(int))
    ])

    ## Identify active phones (moving enough)
    pts_bbox = (
        pts.group_by(phone_id_column)
        .agg([
            pl.col("x").min().alias("x_min"),
            pl.col("x").max().alias("x_max"),
            pl.col("y").min().alias("y_min"),
            pl.col("y").max().alias("y_max"),
        ])
        .with_columns(
            (
                ((pl.col("x_max") - pl.col("x_min")) ** 2 + (pl.col("y_max") - pl.col("y_min")) ** 2).sqrt()
            ).alias("max_dist")
        )
    )
    pts_active = (
        pts.join(pts_bbox.select([phone_id_column, 'max_dist']), on=phone_id_column)
        .filter(pl.col("max_dist") > INACTIVE_PHONE_AREA_SIDE_METERS)
        .select(pts.columns)  # revient à la structure initiale
    )

    # Format eventDate to ISO8601 and compute timestamps
    pts_active = pts_active.with_columns([
        pl.col("eventDate").cast(pl.Utf8).str.replace(" UTC", "").str.replace(" ", "T").alias("eventDate8601")
    ])
    timestamps = [
        int(ciso8601.parse_datetime(dt).timestamp()) if dt else None
        for dt in pts_active["eventDate8601"]
    ]
    pts_active = pts_active.with_columns([pl.Series("ts", timestamps)])

    # Filter on accuracy
    pts_active = pts_active.with_columns([
        pl.col("accuracy").cast(pl.Int64)
    ]).filter(pl.col("accuracy") < MAX_ACCURACY)

    # Drop the temp eventDate8601
    pts_active = pts_active.drop("eventDate8601")

    return pts_active


def points_to_tracks(points: pl.DataFrame) -> pl.DataFrame:
    # Remove duplicated points per phone_id and (x, y)
    df = points.unique(subset=["phone_id", "x", "y"])
    
    # Shift x, y, and timestamp within each track_id to get previous point/instant
    df = df.with_columns([
        pl.col("x").shift(1).over("track_id").alias("x_prec"),
        pl.col("y").shift(1).over("track_id").alias("y_prec"),
        pl.col("ts").shift(1).over("track_id").alias("ts_prec"),
    ])
   
    # Compute euclidean distance from the previous point (fast, no geometry object)
    df = df.with_columns([
        pl.when(pl.col("x_prec").is_not_null())
         .then(
             ((pl.col("x") - pl.col("x_prec"))**2 + (pl.col("y") - pl.col("y_prec"))**2).sqrt()
         )
         .otherwise(0)
         .alias("dist_from_prec")
    ])
   
    # Compute elapsed time between two consecutive GPS points (in seconds)
    df = df.with_columns([
        (pl.col("ts") - pl.col("ts_prec")).alias("time_elapsed"),
    ])
   
    # Compute instant speed (meters per second)
    df = df.with_columns([
        pl.when(pl.col("time_elapsed") > 0)
         .then(pl.col("dist_from_prec") / pl.col("time_elapsed"))
         .otherwise(0)
         .alias("v_instant")
    ])
   
    # Aggregate all desired statistics by trip (track_id)
    tracks = df.group_by("track_id").agg(
        pl.col("accuracy").max().alias("accuracy_max"),
        pl.col("accuracy").mean().alias("accuracy_moy"),
        pl.col("time_elapsed").mean().alias("time_prec_moy"),
        pl.col("time_elapsed").max().alias("time_prec_max"),
        pl.concat_list("v_instant").alias("list_v_instant"),
        pl.concat_list("time_elapsed").alias("list_time_elapsed"),
        pl.col("v_instant").median().alias("v_median"),
        pl.col("v_instant").quantile(0.95, interpolation='nearest').alias("v_95th_percentile"),
        pl.col("dist_from_prec").max().alias("dist_from_prec_max"),
        pl.col("dist_from_prec").mean().alias("dist_from_prec_moy"),
        pl.col("v_instant").max().alias("v_instant_max"),
        pl.col("v_instant").mean().alias("v_instant_moy"),
        pl.col("x").first().alias("departure_x"),
        pl.col("y").first().alias("departure_y"),
        pl.col("x").last().alias("arrival_x"),
        pl.col("y").last().alias("arrival_y"),
        pl.concat_list("x").alias("x_list"),
        pl.concat_list("y").alias("y_list"),
        pl.col("ts").first().alias("first_ts"),
        pl.col("ts").last().alias("last_ts"),
        pl.col("time_elapsed").sum().alias("duration"),
        pl.col("dist_from_prec").sum().alias("track_length"),
        pl.col("eventDate").dt.day().first().alias("day"),
        pl.col("phone_id").first().alias("phone_id"),
    )

    # Compute direct-euclidean distance between first and last point of each track
    tracks = tracks.with_columns([
        pl.struct(["departure_x", "departure_y", "arrival_x", "arrival_y"])
          .map_elements(
              lambda d: np.hypot(d["arrival_x"]-d["departure_x"], d["arrival_y"]-d["departure_y"]),
              return_dtype=pl.Float64
          )
          .alias("dist_departure_arrival")
    ])

    return tracks

def categorize(
    tracks: pl.DataFrame,
    perim: Polygon, 
    bbox_int: gpd.GeoDataFrame,
    bbox_ext: gpd.GeoDataFrame,
    points_crs: str = 'EPSG:2154',
) -> pl.DataFrame:
    """
    Returns tracks with type information : intern, extern, exchange - related to zoning perimeter
    Only converts points to GeoDataFrame if necessary (origin o destination points that are not in/out bbox_int/bbox_ext)
    """
    
    coords_dep = [(p['x'], p['y']) for p in tracks['departure_point'].to_list()]
    coords_end = [(p['x'], p['y']) for p in tracks['end_point'].to_list()]
    
    print('... check origin and destinations points ...')
    dep_in = utils.point_within_zoning(coords_dep, bbox_ext, bbox_int, perim, points_crs)
    end_in = utils.point_within_zoning(coords_end, bbox_ext, bbox_int, perim, points_crs)

    print('... conclusion on tracks ...')
    tracks = tracks.with_columns([
        pl.Series(name="dep_in", values=dep_in),
        pl.Series(name="end_in", values=end_in)
    ])
    tracks = tracks.with_columns([
        pl.when(pl.col('dep_in') & pl.col('end_in')).then(pl.lit('intern'))
        .when(pl.col('dep_in') | pl.col('end_in')).then(pl.lit('exchange'))
        .otherwise(pl.lit('extern'))
        .alias("typology")
    ])
    return tracks


def tracks_to_mapmatch(
    points: pl.DataFrame,
    tracks: pl.DataFrame, 
) -> pl.DataFrame :
    """
    points contains a columns 'point_in_bbox' processed formerly (function in utils, using the bbox used to download road network)
    Analysis on tracks : 
        - for point in track[point_ids], check if point is in bbox
        - if nb(point_in_bbox) > 2 : build track in bbox
    Returns an extract of tracks in bbox to be mapmatched on road / rail network
    """

    # 1. Explode tracks in point_ids to join with points and identify sequences
    tracks_exploded = (
        tracks.select(['phone_id', 'track_id', 'chunk', 'point_ids', 'weight'])
        .explode('point_ids')
        .with_columns([
            pl.col('point_ids').alias('point_id'),
            pl.int_range(0, pl.count()).over('track_id').alias('seq_idx'),
        ])
        .drop('point_ids')
    )

    # 2. Join with points on point_id
    tracks_points = (
        tracks_exploded
        .join(points.select(['point_id', 'x', 'y', 'duration', 'length', 'accuracy_median', 'point_in_bbox']), on='point_id', how='left')
        .sort(['track_id', 'seq_idx'])
        .with_columns([
            (pl.col("point_in_bbox") != pl.col("point_in_bbox").shift(1)).cast(pl.UInt8).over("track_id").alias("bbox_change"),
        ])
        .with_columns([
            pl.cum_sum('bbox_change').over("track_id").alias("bbox_seq"),
        ])
    )

    # 3. Filter on point_in_bbox and rebuild sequences - keep new sequences only if they contain more than 2 points
    tracks_to_mapmatch = (
        tracks_points
        .filter(pl.col("point_in_bbox") == True)
        .group_by(['chunk', 'phone_id', 'track_id', 'weight', 'bbox_seq'])
        .agg([
            pl.col('point_id').alias('point_ids'),
            pl.count('point_id').alias('n_points'),
            pl.col("accuracy_median").median().alias("accuracy_median"),
            pl.col("duration").median().alias("sampling_duration_median"),
            pl.col("length").median().alias("sampling_distance_median"),
            pl.col("duration").sum().alias("duration"),
            pl.col("length").sum().alias("length"),
            pl.struct(["x", "y"]).implode().alias("coordinates"),
        ])
        .with_columns([
            (pl.col("length") / pl.col("duration")).alias("average_speed")
        ])
        .filter(pl.col('n_points') >= 2)
        .drop('n_points')
    )
    return tracks_to_mapmatch


def old_categorize(tracks, perimeter):
    t0 = time.time()
    
    # 1. Extraction des coordonnées
    coords_dep = [(p['x'], p['y']) for p in tracks['departure_point'].to_list()]
    coords_end = [(p['x'], p['y']) for p in tracks['end_point'].to_list()]
    print(f"[TIMING] Extraction coords : {time.time()-t0:.3f} s")
    
    # 2. Création GeoDataFrames
    t1 = time.time()
    points_dep = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in coords_dep], crs='EPSG:2154'
    ).to_crs(perimeter.crs)
    points_end = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in coords_end], crs='EPSG:2154'
    ).to_crs(perimeter.crs)
    print(f"[TIMING] Création GeoDF : {time.time()-t1:.3f} s")
    
    # 3. Union du périmètre
    t2 = time.time()
    perim = perimeter.union_all()
    print(f"[TIMING] Union périmètre   : {time.time()-t2:.3f} s")
    
    # 4. Test de within
    t3 = time.time()
    dep_in = points_dep.within(perim).to_numpy()
    end_in = points_end.within(perim).to_numpy()
    print(f"[TIMING] Test within       : {time.time()-t3:.3f} s")
    print(f"[INFO] Nb évalués départ   : {dep_in.size}")
    print(f"[INFO] Nb évalués arrivée  : {end_in.size}")
    
    # 5. Assemblage Pl
    t4 = time.time()
    tracks = tracks.with_columns([
        pl.Series(dep_in).alias("dep_in"),
        pl.Series(end_in).alias("end_in"),
    ])
    tracks = tracks.with_columns([
        pl.when(pl.col('dep_in') & pl.col('end_in')).then(pl.lit('intern'))
        .when(pl.col('dep_in') | pl.col('end_in')).then(pl.lit('exchange'))
        .otherwise(pl.lit('extern'))
        .alias("typology")
    ])
    print(f"[TIMING] Col. Pl assemble  : {time.time()-t4:.3f} s")
    print(f"[TIMING] Temps total       : {time.time()-t0:.3f} s")
    
    return tracks


