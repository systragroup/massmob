import pandas as pd
import polars as pl


def clean_points(
    points: pl.DataFrame, 
    phone_column: str = 'phone_id', 
    time_column: str = 'ts'
) -> pl.DataFrame:
    """
    Remove duplicate records (based on phone ID and timestamp) and sort by phone and time.

    This function removes duplicate points that share the same combination of phone identifier and timestamp,
    keeping only the first occurrence. The resulting DataFrame is then sorted by phone identifier and timestamp.

    Parameters
    ----------
    points : pl.DataFrame
        Polars DataFrame containing the raw location points.
    phone_column : str, optional
        Name of the column containing phone identifiers (default: 'phone_id').
    time_column : str, optional
        Name of the column containing timestamps (default: 'ts').

    Returns
    -------
    pl.DataFrame
        Cleaned Polars DataFrame, deduplicated and sorted by phone and time.
    """
    # Remove duplicates, keeping the first occurrence for each phone and timestamp
    points = points.unique(subset=[phone_column, time_column], keep='first')

    # Sort by phone and time
    points = points.sort([phone_column, time_column])

    return points


def stops_append_d_s_t(
    points: pl.DataFrame,
    phone_column: str = 'phone_id'
) -> pl.DataFrame:
    """
    Compute and append distance, time difference, and speed between consecutive points for each phone.

    For every record, this function calculates:
      - 'd': the Euclidean distance (in the unit of x/y) to the previous point of the same phone,
      - 't': the elapsed time to the previous point ('ts' column, assumed in seconds),
      - 's': the speed (in km/h) between points (distance divided by time difference, multiplied by 3.6).
    
    For the first point of each phone, 'd', 't', and 's' are set to zero.  
    An additional boolean column 'new_phone' is returned, indicating the first point of each phone sequence.

    Parameters
    ----------
    points : pl.DataFrame
        Polars DataFrame containing at least the columns 'x', 'y', 'ts', and the phone id column.
    phone_column : str, optional
        Name of the column representing the phone identifier (default: 'phone_id').

    Returns
    -------
    pl.DataFrame
        Input Polars DataFrame with the new columns: 'd' (distance), 't' (time difference), 's' (speed), and 'new_phone' (bool).
    """

    # Compute per-phone (partitioned) differences
    points = points.sort([phone_column, "ts"])

    points = points.with_columns([
        # Shift x, y, and ts by 1 within each phone group
        pl.col("x").shift(1).over(phone_column).alias("x_prev"),
        pl.col("y").shift(1).over(phone_column).alias("y_prev"),
        pl.col("ts").shift(1).over(phone_column).alias("ts_prev"),
    ])

    # Compute distance, time, speed
    points = points.with_columns([
        (
            ((pl.col("x") - pl.col("x_prev"))**2 + (pl.col("y") - pl.col("y_prev"))**2).sqrt()
        ).fill_null(0).alias("d"),
        (pl.col("ts") - pl.col("ts_prev")).fill_null(0).alias("t"),
    ])
    points = points.with_columns([
        ((pl.col("d") / pl.col("t")) * 3.6).fill_null(0).alias("s")
    ])

    # Detect first point of each phone
    points = points.with_columns([
        (pl.col("x_prev").is_null()).alias("new_phone")
    ])

    # Set d, t, s to zero for new phone starts
    points = points.with_columns([
        pl.when(pl.col("new_phone"))
          .then(0)
          .otherwise(pl.col("d"))
          .alias("d"),
        pl.when(pl.col("new_phone"))
          .then(0)
          .otherwise(pl.col("t"))
          .alias("t"),
        pl.when(pl.col("new_phone"))
          .then(0)
          .otherwise(pl.col("s"))
          .alias("s"),
    ])
    
    # Optionally drop the helper columns
    points = points.drop(["x_prev", "y_prev", "ts_prev"])

    return points


def stops_identify_noise_trips(
    points: pl.DataFrame, 
    idling_phone_meters_distance: float, 
    trip_column: str = "trip_group", 
    noise_column: str = "noise_trip"
) -> pl.DataFrame:
    """
    Identify and mark trips covering too short a distance as noise.

    For each trip group (as defined by `trip_column`), compute the spanned distance (the diagonal of the bounding box 
    covering the trip based on 'x' and 'y' in meters). Then flag all points belonging to trips where this spanned distance 
    is less than `idling_phone_meters_distance` by setting `noise_column` to True.

    Parameters
    ----------
    points : pl.DataFrame
        Polars DataFrame with at least the columns 'x', 'y', and the trip identifier column.
    idling_phone_meters_distance : float
        Maximum spanned distance under which a trip is considered as a noise trip.
    trip_column : str, optional
        Name of the column defining trip groups (default: 'trip_group').
    noise_column : str, optional
        Name of the output boolean column indicating noise trips (default: 'noise_trip').

    Returns
    -------
    pl.DataFrame
        The input DataFrame with an added boolean column `noise_column` where True indicates a noise trip.
    """

    # Safety check: verify coordinates are in meters
    assert points["x"].max() > 200, "x and y should be in meters coordinates"

    # Compute spanned distance for each trip group
    trip_dist = (
        points.group_by(trip_column)
        .agg([
            (pl.col("x").max() - pl.col("x").min()).alias("dx"),
            (pl.col("y").max() - pl.col("y").min()).alias("dy"),
        ])
        .with_columns([
            ((pl.col("dx")**2 + pl.col("dy")**2).sqrt().alias("bbox_dist"))
        ])
    )

    # Mark trips whose spanned distance is below threshold as noise
    noisy_trips = trip_dist.filter(
        pl.col("bbox_dist") < idling_phone_meters_distance
    ).select(trip_column)

    # Join to add the noise flag to main DataFrame
    points = points.with_columns([
        pl.col(trip_column).is_in(noisy_trips[trip_column]).alias(noise_column)
    ])

    return points


def stops_set_trace_id(
    df: pl.DataFrame,
    stop_column: str,
    cut_column: str,
    sort_by: list = ['phone_id', 'ts']
) -> pl.DataFrame:
    """
    Adds a track identifier to the DataFrame, incrementing at each stop or cut.

    For every row, a new track id (track_id) is assigned by cumulatively summing the appearance of stops or cuts.
    Original stop points are duplicated to separate overlapping trace boundaries: for every stop, a duplicate
    row is inserted immediately after with 'fake_points' = True and stop_column = False.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame, must contain the stop and cut columns.
    stop_column : str
        Name of the column indicating stop events (boolean or integer 0/1).
    cut_column : str
        Name of the column indicating cut events (boolean or integer 0/1).
    sort_by : list, optional
        List of columns to sort by chronologically within phone_id (default: ['phone_id', 'ts']).

    Returns
    -------
    pl.DataFrame
        DataFrame with added columns:
        - 'fake_points': bool, indicates which rows were added as duplicates,
        - 'track_id': int, the computed trace/track identifier.
    """

    # Mark all as real point initially
    df = df.with_columns([
        pl.lit(False).alias('fake_points')
    ])

    # Duplicate rows where stop_column is True, mark as fake, and set stop_column to False
    dup = df.filter(pl.col(stop_column)).with_columns([
        pl.lit(True).alias('fake_points'),
        pl.lit(False).alias(stop_column)
    ])

    # Concatenate original and duplicated fake stop points
    df = pl.concat([df, dup])

    # Sort by sort_by columns + stop_column (as in pandas)
    sort_cols = sort_by + [stop_column]
    df = df.sort(sort_cols)

    # Compute boolean for increment: (stop_column | cut_column)
    # Cumulative sum over all rows (to create the track id)
    df = df.with_columns([
        ((pl.col(stop_column).cast(bool)) | (pl.col(cut_column).cast(bool))).cumsum().alias('track_id')
    ])

    return df


def stops_drop_short_trips(
    points: pl.DataFrame,
    MIN_TRIP_DURATION_SECONDS: float,
    MIN_TRIP_DISTANCE_METERS: float,
    method: str = 'remove'
) -> pl.DataFrame:
    """
    Remove or flag trips that are too short (few points, duration, or distance).

    Filters out trips that do not meet the minimal criteria:
      - Have less than 2 points,
      - Duration below MIN_TRIP_DURATION_SECONDS,
      - Spanned distance below MIN_TRIP_DISTANCE_METERS (in either x or y direction).

    Parameters
    ----------
    points : pl.DataFrame
        Polars DataFrame containing at least columns: 'track_id', 'ts', 'x', 'y', and 't' (per-segment duration).
    MIN_TRIP_DURATION_SECONDS : float
        Minimal duration (in seconds) required for a trip to be kept.
    MIN_TRIP_DISTANCE_METERS : float
        Minimal spanned distance (in meters, in x or y) required for a trip to be kept.
    method : {'remove', 'stick'}, optional
        What to do with short trips: 'remove' (delete their points), 'stick' (not implemented yet).
    
    Returns
    -------
    pl.DataFrame
        Polars DataFrame filtered to keep only valid trips.
    """
    # Compute statistics per trip
    trip_stats = (
        points.group_by("track_id")
        .agg([
            pl.count().alias("n_pts"),
            pl.col("ts").max() - pl.col("ts").min(),                   # duration
            pl.col("t").sum().alias("trip_t"),
            (pl.col("x").max() - pl.col("x").min()).alias("span_x"),
            (pl.col("y").max() - pl.col("y").min()).alias("span_y"),
        ])
        .rename({
            "ts": "duration",
        })
    )

    # Criteria
    valid_track_mask = (
        (pl.col("n_pts") > 1) &
        ((pl.col("span_x") > MIN_TRIP_DISTANCE_METERS) | (pl.col("span_y") > MIN_TRIP_DISTANCE_METERS)) &
        (pl.col("trip_t") > MIN_TRIP_DURATION_SECONDS)
    )

    valid_tracks = trip_stats.filter(valid_track_mask).select("track_id")

    if method == 'remove':
        # Keep points whose track_id is in valid_tracks
        points = points.filter(pl.col("track_id").is_in(valid_tracks["track_id"]))
    elif method == 'stick':
        raise NotImplementedError("method 'stick' not implemented yet")
    else:
        raise ValueError("method should be 'remove' or 'stick'")

    return points


def stops_untag_short_stops(
    points: pl.DataFrame,
    MAKING_A_STOP_SECONDS_DELAY: float
) -> pl.DataFrame:
    """
    Untag and remove the 'stop' flag from stops whose duration is below a threshold.

    For each group of consecutive stop or non-stop points, compute the duration (max(ts) - min(ts)).
    If the duration is less than MAKING_A_STOP_SECONDS_DELAY, mark these points as 'short_stop'
    and unset the 'stop' flag for those groups.

    Parameters
    ----------
    points : pl.DataFrame
        Polars DataFrame with at least the columns 'stop' (bool) and 'ts' (timestamp in seconds).
    MAKING_A_STOP_SECONDS_DELAY : float
        Minimal duration (in seconds) to be considered a valid stop.

    Returns
    -------
    pl.DataFrame
        DataFrame with an added boolean column 'short_stop'; for the corresponding rows, 'stop' is set to False.
    """

    # Assign a unique stop_group id by cumulatively summing where stop is False (to mimic ~points['stop'].cumsum())
    points = points.with_columns([
        (~pl.col('stop')).cumsum().alias('stop_group')
    ])

    # Compute duration per stop_group
    stop_durations = (
        points.group_by('stop_group')
        .agg([
            (pl.col('ts').max() - pl.col('ts').min()).alias('duration')
        ])
    )

    # Identify short stops (duration < threshold)
    short_stop_groups = (
        stop_durations.filter(
            pl.col('duration') < MAKING_A_STOP_SECONDS_DELAY
        )['stop_group']
    )

    # Add 'short_stop' column and unset 'stop' for those groups
    points = points.with_columns([
        pl.col('stop_group').is_in(short_stop_groups).alias('short_stop'),
        pl.when(pl.col('stop_group').is_in(short_stop_groups))
        .then(False)
        .otherwise(pl.col('stop'))
        .alias('stop')
    ])

    return points
