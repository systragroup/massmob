import polars as pl
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
from typing import Optional, Set, List, Dict, Any

def filter_for_home(points: pl.DataFrame) -> pl.DataFrame:
    """
    Filter points falling in night hours (for home clustering).
    """
    night_hours = {20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7}
    return points.filter(
        pl.col("eventDate").dt.hour().is_in(night_hours)
    )

def filter_for_work(
    points: pl.DataFrame,
    exclude_weekends: bool = True, 
    excluded_jjmm: Optional[Set[int]] = None
) -> pl.DataFrame:
    """
    Filter points in work hours and (optionally) remove weekends and dates.
    """
    work_hours = list(range(9, 20))
    df = points.filter(
        pl.col("eventDate").dt.hour().is_in(work_hours)
    )
    if exclude_weekends:
        df = df.with_columns([
            pl.col("eventDate").dt.weekday().alias("weekday")
        ])
        df = df.filter(pl.col("weekday") < 5).drop("weekday")
    if excluded_jjmm:
        df = df.with_columns([
            pl.col("eventDate").dt.strftime("%d%m").cast(pl.Int32).alias("jjmm")
        ])
        excl = list(excluded_jjmm)
        df = df.filter(~pl.col("jjmm").is_in(excl)).drop("jjmm")
    return df

def filter_stationary(
    points: pl.DataFrame, 
    speed_threshold: float = 2.0
) -> pl.DataFrame:
    """
    Keep only points with instantaneous speed < threshold (km/h).
    """
    assert "s" in points.columns, "'s' column not found in points DataFrame, consider using stops_append_d_s_t function to add it."
    filtered = points.filter(pl.col("s") < speed_threshold)
    return filtered.drop("s")

def filter_min_points(points: pl.DataFrame, id_col: str = "phone_id", min_points: int = 10) -> pl.DataFrame:
    """
    Filtre les téléphones (id_col) ayant au moins min_points dans le DataFrame.
    """
    # Compte le nombre de points par identifiant
    counts = points[id_col].value_counts().rename({"count": "n_points"})
    valid_ids = counts.filter(pl.col("n_points") >= min_points)[id_col]
    # Ne conserve que les points des identifiants assez fréquents
    return points.filter(pl.col(id_col).is_in(valid_ids))

def cluster_location(
    points: pl.DataFrame,
    min_samples: int = 2,
    cluster_epsilon: float = 100.0,  # meters in projected CRS
    x_col: str = "x",
    y_col: str = "y",
    id_col: str = "phone_id",
) -> pl.DataFrame:
    """
    For each ID, compute centroid of main DBSCAN cluster using projected (x, y).
    Returns a Polars DataFrame with 'id_col', 'x', 'y'.
    """
    results: List[Dict[str, Any]] = []
    # Iterate all unique IDs
    for phone_id, subdf in points.group_by(id_col, maintain_order=True):
        coords = np.vstack([
            subdf[x_col].to_numpy(),
            subdf[y_col].to_numpy()
        ]).T
        if coords.shape[0] < min_samples:
            continue
        clustering = DBSCAN(
            eps=cluster_epsilon, min_samples=min_samples
        ).fit(coords)
        labels = clustering.labels_
        clusters = [l for l in labels if l >= 0]
        if not clusters:
            continue
        # Find largest non-noise cluster
        main_cluster = Counter(clusters).most_common(1)[0][0]
        indices_main = np.where(labels == main_cluster)[0]
        coords_main = coords[indices_main]
        centroid = coords_main.mean(axis=0)
        results.append({
            id_col: phone_id[0],
            "x": float(centroid[0]),
            "y": float(centroid[1]),
        })
    if results:
        return pl.DataFrame(results)
    # Return empty DataFrame with correct columns if no cluster found
    return pl.DataFrame(
        {id_col: [], "x": [], "y": []}
    )

def cluster_home(
    points: pl.DataFrame, 
    only_stationary: bool = True, 
    min_points: int = 5,
    **kwargs
) -> pl.DataFrame:
    """
    Filter for night hours (+ stationary if desired) and cluster for home location.
    """
    filtered = filter_for_home(points)
    if only_stationary:
        filtered = filter_stationary(filtered)
    filtered = filter_min_points(filtered, min_points=min_points)
    return cluster_location(filtered, **kwargs)

def cluster_work(
    points: pl.DataFrame,
    exclude_weekends: bool = True, 
    excluded_jjmm: Optional[Set[int]] = None, 
    only_stationary: bool = True, 
    min_points: int = 5,
    **kwargs
) -> pl.DataFrame:
    """
    Filter for work hours (+ stationary if desired), and cluster for work location.
    """
    filtered = filter_for_work(points, exclude_weekends, excluded_jjmm)
    if only_stationary:
        filtered = filter_stationary(filtered)
    filtered = filter_min_points(filtered, min_points=min_points)
    return cluster_location(filtered, **kwargs)

def add_missing_phone_ids(all_phone_ids, cluster_df, x_col="x", y_col="y"):
        """
        Retourne cluster_df complété des phone_id manquants, avec les coordonnées à None.
        """
        cluster_phone_ids = set(cluster_df["phone_id"].to_list())
        missing_ids = [pid for pid in all_phone_ids if pid not in cluster_phone_ids]

        if missing_ids:
            # Crée un DataFrame avec les missing_ids et les coordonnées à None
            missing_df = pl.DataFrame({
                "phone_id": missing_ids,
                x_col: [None] * len(missing_ids),
                y_col: [None] * len(missing_ids),
            })
            # Concatène et réordonne
            full_df = pl.concat([cluster_df, missing_df])
        else:
            full_df = cluster_df

        return full_df