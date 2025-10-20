import polars as pl
import geopandas as gpd
from shapely.geometry import Point

def point_in_zone(
    points: pl.DataFrame,
    zoning: gpd.GeoDataFrame,
    points_crs: float = 'EPSG:2154',
) -> pl.DataFrame:
    """
    Add a column 'zone_id' to the points DataFrame corresponding to the id of the zone in case point is in a zone of the zoning
    
    Parameters
    ----------
    points : pl.DataFrame
        DataFrame with 'x' and 'y' columns
    zoning : gpd.GeoDataFrame
        contains zones with 'zone_id' column (and geometry)
    
    Returns
    -------
    pl.DataFrame
        Original DataFrame, with an added 'zone_id' column (null if not in any zone)
    """

    valid_points = points.filter(
        pl.col("x").is_not_null() & pl.col("y").is_not_null()
    )

    gdf_points = gpd.GeoDataFrame(
        valid_points.to_pandas(),
        geometry=[
            Point(x, y) for x, y in zip(valid_points["x"], valid_points["y"])
        ],
        crs=points_crs
    )

    if gdf_points.crs != zoning.crs:
        gdf_points = gdf_points.to_crs(zoning.crs)

    joined = gpd.sjoin(gdf_points, zoning[["zone_id", "geometry"]], how="left", predicate="within")
    zone_id_df = pl.from_pandas(joined[["phone_id", "zone_id"]])

    return points.join(zone_id_df, on="phone_id", how="left")

def residents(
    home_locations: pl.DataFrame,
    zoning: gpd.GeoDataFrame,        
) -> pl.DataFrame:
    """
    Returns a table with phone_id, home location, and corresponding zone of the zoning for home locations within zoning
    """
    home_locations = point_in_zone(home_locations, zoning)
    home_locations = home_locations.with_columns(pl.col("zone_id").cast(pl.Int64))
    residents = home_locations.filter(pl.col("zone_id").is_not_null())
    return residents

def compute_ratio_pop_zone(
    residents: pl.DataFrame,
    zoning: gpd.GeoDataFrame,
    phone_penetration_rate: float = 0.7,
) -> pl.DataFrame:
    """
    Returns residents_by_zone = table with the number of residents captured in each zone and representativity ratio associated
    
    Parameters
    ----------
    residents : pl.DataFrame
        table with phone_id and zone_id
    zoning : gpd.GeoDataFrame
        contains zones with 'zone_id' and 'population' columns
    phone_penetration_rate : float
        part of the population that has a smartphone -> maximum captable population
    """
    residents_by_zone = (
        residents.group_by("zone_id")
        .len()
        .rename({"len": "n_residents_captured"})
    )
    pop_zone = pl.from_pandas(zoning[["zone_id", "population"]])
    residents_by_zone = residents_by_zone.with_columns(
        pl.col("zone_id").cast(pl.Int64)
    ).join(pop_zone, on="zone_id").with_columns([
        (pl.col("population") * phone_penetration_rate / pl.col("n_residents_captured")).alias("ratio_pop_zone")
    ])
    return residents_by_zone

def attribute_weight_zone(
    residents: pl.DataFrame,
    residents_by_zone: pl.DataFrame,
) -> pl.DataFrame:
    """
    Associates the ratio of population observed / real for each zone for each phone_id of the resident table
    """
    return residents.select(["phone_id", "zone_id"]).join(residents_by_zone, on="zone_id")

def extract_info_to_K(
    residents: pl.DataFrame,
    tracks: pl.DataFrame,
) -> pl.DataFrame:
    """
    Computes the weight associated to each track, if phone_id is resident or no.
    Returns the tracks pl.DataFrame with new column 'weight'

    Parameters
    ----------
    residents : pl.DataFrame
        table with phone_id and associated ratio
    tracks : pl.DataFrame
        tracks to find associated weight
    """

    tracks = tracks.join(residents.select(["phone_id", "ratio_pop_zone"]), on="phone_id", how="left")
    l_days = tracks.select(pl.col("departure_day")).unique()
    sum_weight_zone_dep = tracks.filter(
        pl.col("ratio_pop_zone").is_not_null()
        ).group_by(
            ["phone_id", "ratio_pop_zone"]
            ).len().with_columns([
            (pl.col("len") * pl.col("ratio_pop_zone")).alias("weighted_trips")
        ]).select(pl.col("weighted_trips")).sum().item()
    return tracks, l_days, sum_weight_zone_dep

def compute_K(
    residents: pl.DataFrame,
    sum_weight_residents: float,
    n_days_study_period: float,
    mobility_rate: float = 3.7,
) -> float:
    """
    Returns the corrective coefficient to get to a casual daily mobility rate for residents

    Parameters
    ----------
    residents : pl.DataFrame
        residents dataframe to get total population
    sum_weight_residents : float
        sum of the zone_weights associated to each track of resident = population ratio expansion
    n_days_study_period : float
        number of days in the study period during which residents travelled
    mobility_rate : float
        expected number of trips in a usual day for the local population
    """
    return mobility_rate / (sum_weight_residents / residents[['zone_id', 'population']].unique()['population'].sum() / n_days_study_period) if n_days_study_period != 0 else 1.

def compute_total_weight(
    tracks: pl.DataFrame,
    residents_by_zone: pl.DataFrame,
    K: float,
) -> pl.DataFrame:
    """
    Returns tracks dataframe with associated weight :
    - if phone_id is a resident, weight = ratio_pop_zone * K
    - else weight = mean of all zonal resident ratios
    """
    tracks_with_weights = tracks.with_columns(
        [(pl.col("ratio_pop_zone") * K).alias("weight")]
    )
    resident_zone_weights = residents_by_zone.select(
        (pl.col("ratio_pop_zone") * K).alias("zone_weight")
    )
    resident_weight_mean = resident_zone_weights.select(pl.col("zone_weight").mean()).item()
    tracks_with_final_weight = tracks_with_weights.with_columns([
        pl.when(pl.col("weight").is_null())
        .then(resident_weight_mean)
        .otherwise(pl.col("weight"))
        .alias("weight")
    ])

    return tracks_with_final_weight.drop(["ratio_pop_zone"])