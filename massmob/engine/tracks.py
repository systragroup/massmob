import pandas as pd
import geopandas as gpd
import polars as pl
import numpy as np
from pyproj import Transformer
import ciso8601
import numpy as np
import ciso8601
import shapely
from shapely.geometry import LineString, Point
from massmob.engine import stops


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
        pl.col('point_ids').map_elements(lambda ids: min(day_map[i] for i in ids), return_dtype=pl.Object).alias('departure_day'),

        # Departure and end raw coordinate tuples (not geometric objects)
        pl.col('coordinates').map_elements(lambda coords: tuple(coords[0]), return_dtype=pl.Object).alias('departure_point'),
        pl.col('coordinates').map_elements(lambda coords: tuple(coords[-1]), return_dtype=pl.Object).alias('end_point'),
    ])
    return tracks


def build_tracked_points(
    points: pl.DataFrame,
    MAX_SECONDS_DELAY_BETWEEN_POINTS: int = 60 * 60,
    STOP_SPEED_THRESHOLD_KMH: float = 1,
    IDLING_PHONE_METERS_DISTANCE: float = 200,
    MAKING_A_STOP_SECONDS_DELAY: float = 10 * 60,
    MIN_TRIP_DURATION_SECONDS: float = 60 * 2,
    MIN_TRIP_DISTANCE_METERS: float = 200
) -> pl.DataFrame:
    """
    Identifie, segmente et filtre les trajets dans un DataFrame Polars de points GPS/horodatés.

    Retourne le DataFrame filtré et segmenté, prêt à l'analyse.
    """

    # Nettoyage amont (fournir des alternatives si stops.* fonctionne sur Polars)
    points = stops.clean_points(points)
    points = stops.stops_append_d_s_t(points)

    # Marquages et coupures de points
    points = points.with_columns([
        (pl.col('t') > MAX_SECONDS_DELAY_BETWEEN_POINTS).alias('duration_threshold_exceeded')
    ])
    points = points.with_columns([
        pl.when(pl.col('duration_threshold_exceeded')).then(0).otherwise(pl.col('d')).alias('d'),
        pl.when(pl.col('duration_threshold_exceeded')).then(0).otherwise(pl.col('t')).alias('t'),
        pl.when(pl.col('duration_threshold_exceeded')).then(0).otherwise(pl.col('s')).alias('s')
    ])
    points = points.with_columns([
        (pl.col('duration_threshold_exceeded') | pl.col('new_phone')).alias('cut')
    ]).drop(['duration_threshold_exceeded', 'new_phone'])

    # Low speed
    points = points.with_columns([
        (pl.col('s') < STOP_SPEED_THRESHOLD_KMH).alias('low_speed')
    ])
    # Compteur cumulatif sur coupe/arrêt => trip_group
    points = points.with_columns([
        ((pl.col('cut') | pl.col('low_speed')).cast(pl.Int32).cum_sum()).alias('trip_group')
    ])

    # Mark "noise_trip"
    points = stops.stops_identify_noise_trips(points, IDLING_PHONE_METERS_DISTANCE)
    points = points.with_columns([
        (pl.col('low_speed') | pl.col('noise_trip')).alias('stop')
    ])

    # Stop group
    points = points.with_columns([
        # ((~pl.col('stop')).cast(pl.Int32).cum_sum().over("phone_id")).alias('stop_group')
        ((~pl.col('stop')).cast(pl.Int32).cum_sum()).alias('stop_group')
    ])

    stop_durations = (
        points.group_by('stop_group').agg(
            (pl.col('ts').max() - pl.col('ts').min()).alias('duration')
        )
    )
    short_stop_groups = stop_durations.filter(
        pl.col('duration') < MAKING_A_STOP_SECONDS_DELAY
    )['stop_group']

    points = points.with_columns([
        pl.col('stop_group').is_in(short_stop_groups).alias('short_stop'),
        pl.when(pl.col('stop_group').is_in(short_stop_groups))
          .then(False).otherwise(pl.col('stop')).alias('stop')
    ]).drop(['low_speed', 'trip_group', 'noise_trip', 'short_stop', 'stop_group'])

    # Segmentation finale
    def set_trace_id(df: pl.DataFrame, stop_column: str, cut_column: str, sort_by=['phone_id', 'ts']) -> pl.DataFrame:
        df = df.with_columns([pl.lit(False).alias('fake_points')])
        dup = df.filter(pl.col(stop_column)).with_columns([
            pl.lit(True).alias('fake_points'),
            pl.lit(False).alias(stop_column)
        ])
        df = pl.concat([df, dup])
        sort_cols = sort_by + [stop_column]
        df = df.sort(sort_cols)
        df = df.with_columns([
            ((pl.col(stop_column).cast(pl.Int32) | pl.col(cut_column).cast(pl.Int32))
              .cum_sum()
              .alias('track_id')
            )
        ])
        return df

    points = set_trace_id(points, 'stop', 'cut', sort_by=['phone_id', 'ts'])

    # On retire les trajets trop courts
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
    valid_tracks = trip_stats.filter(
        (pl.col("n_pts") > 1)
        & ((pl.col("span_x") > MIN_TRIP_DISTANCE_METERS) | (pl.col("span_y") > MIN_TRIP_DISTANCE_METERS))
        & (pl.col("trip_t") > MIN_TRIP_DURATION_SECONDS)
    )["track_id"]
    points = points.filter(pl.col("track_id").is_in(valid_tracks))

    # TODO: on est ici en UTC -> changer en local
    # Parsing du jour
    points = points.with_columns([
        (
            pl.col("eventDate").dt.year().cast(pl.Utf8) + "-" +
            pl.col("eventDate").dt.month().cast(pl.Utf8).str.zfill(2) + "-" +
            pl.col("eventDate").dt.day().cast(pl.Utf8).str.zfill(2)
        ).alias("day")
    ])
    points = points.with_columns([pl.col("phone_id").cast(pl.Utf8)])
    return points


def filtering(
    pts: pl.DataFrame,
    phone_id_column='phone_id',
    INACTIVE_PHONE_AREA_SIDE_METERS = 50,
    MAX_ACCURACY = 50
):
    """
    Filter points to keep only those that are sufficiently far from each other (phones not considered inactive).
    """
    # Drop useless columns if present
    cols_to_drop = [col for col in ['speed', 'eventId', 'Unnamed: 0'] if col in pts.columns]
    pts = pts.drop(cols_to_drop)

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