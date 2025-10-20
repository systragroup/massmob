import os
import polars as pl
import geopandas as gpd
from math import ceil
from tqdm import tqdm
from typing import List, Any, Optional, Set
from massmob.engine import mode, tracks, stops, analysis, mapmatching, clustering, volumes, expansion, utils


class ChunkModel:

    def __init__(self, points: pl.DataFrame = None, nchunks: int = 1, chunk_column: str = 'phone_id'):
        self.chunk_column = chunk_column
        self.nchunks = nchunks
        self.points = points if points is not None else pl.DataFrame([])
        self.tracks = None

    def assign_chunks(self, chunk_column=None, nchunks=None, chunk_name='chunk'):
        """
        Assigns chunk indices to all DataFrame attributes containing the chunk_column.
        """
        chunk_column = chunk_column or self.chunk_column
        self.nchunks = nchunks

        # Trouver tous les attributs pl.DataFrame contenant chunk_column
        dfs = {}
        for attr in self.__dict__:
            value = getattr(self, attr)
            if isinstance(value, pl.DataFrame) and value.height > 0 and chunk_column in value.columns:
                dfs[attr] = value
        
        if not dfs:
            return

        # Calcul du mapping à partir de tous les phone_ids présents dans tous les df
        all_keys = set()
        for df in dfs.values():
            all_keys.update(df[chunk_column].unique().to_list())
        unique_keys = sorted(all_keys)
        from math import ceil
        chunk_size = ceil(len(unique_keys)/nchunks)
        chunk_map = {
            k: idx for idx, group in enumerate(
                [unique_keys[i * chunk_size:(i + 1) * chunk_size] for i in range(nchunks)]
            ) for k in group
        }
        self.chunk_map = chunk_map

        # Appliquer à tous les df trouvés
        for attr, df in dfs.items():
            new_df = df.with_columns(
                pl.col(chunk_column)
                .map_elements(lambda x: chunk_map.get(x, -1), return_dtype=pl.Int32)
                .alias(chunk_name)
            )
            setattr(self, attr, new_df)

    def filter_points(self, **kwargs):
        """
        Apply the filtering function to each chunk, result is merged with correct chunk assignment.
        """
        dfs = []
        for i in tqdm(range(self.nchunks)):
            pts_chunk = self.points.filter(pl.col("chunk") == i)
            if pts_chunk.height == 0:
                continue
            filtered = tracks.filtering(pts_chunk, **kwargs)
            if filtered.height > 0:
                dfs.append(filtered)
        self.points = pl.concat(dfs) if dfs else pl.DataFrame([])

    def build_tracks(self, **kwargs):
        """
        Build tracks per chunk, storing results in self.tracks with chunk column.
        Also updates self.points with possibly updated/enriched pts_aug (per chunk).
        """
        trks_list = []
        pts_new = []

        for i in tqdm(range(self.nchunks)):
            pts_chunk = self.points.filter(pl.col("chunk") == i)
            if pts_chunk.height == 0:
                continue
            pts_aug = tracks.build_tracked_points(pts_chunk, **kwargs)  # points enrichis
            pts_new.append(pts_aug)
            tracks_chunk = tracks.tracks_from_points_with_stops(pts_aug)
            if tracks_chunk.height > 0:
                trks_list.append(tracks_chunk.with_columns(pl.lit(i).alias('chunk')))
        self.tracks = pl.concat(trks_list) if trks_list else pl.DataFrame([])
        self.points = pl.concat(pts_new) if pts_new else pl.DataFrame([])  # met à jour les points
        # reset track ids to avoid duplicates
        self.tracks = self.tracks.with_columns(
            pl.arange(0, self.tracks.height).alias("track_id")
        )

    def analysis_tracks(self):
        """
        Analyze tracks per chunk, storing result in self.tracks.
        """
        dfs = []
        for i in tqdm(range(self.nchunks)):
            trks_chunk = self.tracks.filter(pl.col("chunk") == i)
            pts_chunk = self.points.filter(pl.col("chunk") == i)
            if trks_chunk.height > 0:
                analyzed = tracks.analysis_tracks(trks_chunk, pts_chunk)
                dfs.append(analyzed)
        self.tracks = pl.concat(dfs) if dfs else pl.DataFrame([])
        
    def get_points(self):
        """
        Returns full processed points table, including the chunk column.
        """
        return self.points

    def get_tracks(self):
        """
        Returns full processed tracks table, including the chunk column.
        """
        return self.tracks if self.tracks is not None else pl.DataFrame([])
    
    def cluster_home(self, **kwargs):
        """
        Return home centroid per phone_id per chunck.
        """
        dfs=[]
        for i in tqdm(range(self.nchunks)):
            pts_chunk = self.points.filter(pl.col("chunk") == i)
            if pts_chunk.height > 0:
                found_locs = clustering.cluster_home(pts_chunk, **kwargs)
                home_locations = clustering.add_missing_phone_ids(
                    pts_chunk["phone_id"].unique().to_list(), 
                    found_locs,
                )
                home_locations = home_locations.with_columns(pl.lit(i).alias("chunk"))
                dfs.append(home_locations)
        self.home_locations = pl.concat(dfs) if dfs else pl.DataFrame([])

    def cluster_work(self, **kwargs):
        """
        Return work centroid per phone_id per chunck.
        """
        dfs=[]
        for i in tqdm(range(self.nchunks)):
            pts_chunk = self.points.filter(pl.col("chunk") == i)
            if pts_chunk.height > 0:
                found_locs = clustering.cluster_work(pts_chunk, **kwargs)
                work_locations = clustering.add_missing_phone_ids(
                    pts_chunk["phone_id"].unique().to_list(), 
                    found_locs,
                )
                work_locations = work_locations.with_columns(pl.lit(i).alias("chunk"))
                dfs.append(work_locations)
        self.work_locations = pl.concat(dfs) if dfs else pl.DataFrame([])


    def resident_expansion(
        self,
        zoning: gpd.GeoDataFrame,
        phone_penetration_rate: float = 0.7,
        mobility_rate: float = 3.7
    ):
        """
        Return tracks with associated weight for each chunk
        
        Parameters
        ----------
        zoning : gpd.GeoDataFrame
            contains zones with 'zone_id' and 'population' column (and geometry)
        phone_penetration_rate : float
            part of population that has a smartphone -> helps finding the maximum captable population in each zone
        mobility_rate : float
            average number of journeys in a day
        """

        # 1. Par chunk : identification des résidents avec les home_locations
        residents = []
        for i in tqdm(range(self.nchunks)):
            home_location_chunk = self.home_locations.filter(pl.col('chunk') == i)
            if home_location_chunk.height == 0:
                continue
            residents_chunk = expansion.residents(
                home_location_chunk,
                zoning
            ).with_columns(pl.lit(i).alias("chunk"))
            residents.append(residents_chunk)
        self.residents = pl.concat(residents) if residents else pl.DataFrame([])

        # 2. Nombre de personnes identifiées comme résidents dans chacune des zones du zonage et calcul du ratio de représentativité associé
        self.residents_by_zone = expansion.compute_ratio_pop_zone(
            self.residents,
            zoning,
            phone_penetration_rate
        )
        
        # 3. Par chunk : 
        # attribution du ratio habitants réels / résidents identifiés et application à chaque phone_id
        # recollement dans tracks
        # extraction des éléments de tracks permettant de calculer le ratio correctif pour retomber sur un taux de mobilité correct
        residents, tracks_weight_pop, l_days, sum_weight = [], [], [], []
        for i in tqdm(range(self.nchunks)):
            residents_chunk = self.residents.filter(pl.col('chunk') == i)
            if residents_chunk.height == 0:
                continue
            residents_chunk = expansion.attribute_weight_zone(
                residents_chunk,
                self.residents_by_zone,
            )
            residents.append(residents_chunk)
            tracks_chunk = self.tracks.filter(pl.col('chunk') == i)
            tracks_chunk, l_days_chunk, sum_weight_zone_dep_chunk = expansion.extract_info_to_K(
                residents_chunk,
                tracks_chunk
            )
            tracks_weight_pop.append(tracks_chunk)
            l_days.append(l_days_chunk)
            sum_weight.append(sum_weight_zone_dep_chunk)
        self.residents = pl.concat(residents) if residents else pl.DataFrame([])
        self.tracks = pl.concat(tracks_weight_pop) if tracks_weight_pop else pl.DataFrame([])
        
        # 4. Calcul de K sur les données agrégées extraites des chunks
        total_days = pl.concat(l_days) if l_days else pl.DataFrame([])
        if total_days.height > 0:
            n_days = total_days['departure_day'].n_unique()
        else:
            n_days = 0
        total_weight_dep_residents = sum(sum_weight)
        K = expansion.compute_K(
            self.residents,
            total_weight_dep_residents,
            n_days,
            mobility_rate
        )
        
        # 5.Par chunk : application du poids final : 
        #   - pour les résidents = K * ratio_pop_zone
        #   - pour les non résidents = moyenne des ratio zonaux des résidents
        tracks_total_weight = []
        for i in tqdm(range(self.nchunks)):
            tracks_chunk = self.tracks.filter(pl.col('chunk') == i)
            tracks_chunk = expansion.compute_total_weight(
                tracks_chunk,
                self.residents_by_zone,
                K
            ).with_columns(pl.lit(i).alias("chunk"))
            tracks_total_weight.append(tracks_chunk)
        self.tracks = pl.concat(tracks_total_weight) if tracks_total_weight else pl.DataFrame([])

    def old_categorize_tracks(
        self,
        perimeter: gpd.GeoDataFrame
    ):
        """
        Return tracks with typology
        """
        all_tracks = []
        for i in tqdm(range(self.nchunks)):
            tracks_chunk = self.tracks.filter(pl.col("chunk") == i)
            if tracks_chunk.height == 0:
                continue
            typed_tracks_chunk = tracks.old_categorize(
                tracks=tracks_chunk,
                perimeter=perimeter,
            )
            all_tracks.append(typed_tracks_chunk)
        self.tracks = pl.concat(all_tracks) if all_tracks else pl.DataFrame([])

    def categorize_tracks(
        self,
        perimeter: gpd.GeoDataFrame,
        points_crs: str='EPSG:2154',
    ):
        """
        Return tracks with typology
        """
        bbox_ext, bbox_int = utils.build_bboxes(perimeter=perimeter)
        perim = perimeter.to_crs(points_crs).union_all()

        all_tracks = []
        for i in tqdm(range(self.nchunks)):
            tracks_chunk = self.tracks.filter(pl.col("chunk") == i)
            if tracks_chunk.height == 0:
                continue
            typed_tracks_chunk = tracks.categorize(
                tracks=tracks_chunk,
                perim=perim,
                bbox_ext=bbox_ext,
                bbox_int=bbox_int,
                points_crs=points_crs,
            )
            all_tracks.append(typed_tracks_chunk)
        self.tracks = pl.concat(all_tracks) if all_tracks else pl.DataFrame([])

    def prepare_tracks_to_mapmatch(
        self,
        bbox: tuple
    ):
        """
        Returns new pl.DataFrame with tracks to mapmatch (which is an extract ok tracks in the bounding box of the road network)
        """
        all_points = []
        tracks_to_mapmatch = []
        for i in tqdm(range(self.nchunks)):

            # Analysis points
            print('...analysing points...')
            points_chunk = self.points.filter(pl.col('chunk') == i)
            if points_chunk.height == 0:
                continue
            points_chunk = utils.point_in_bbox(points_chunk, bbox)
            all_points.append(points_chunk)
            points_chunk = points_chunk.rename({
                "t": "duration",
                "d": "length",
                "accuracy": "accuracy_median"
            })

            # Analysis tracks
            print('...analysing tracks...')
            tracks_chunk = self.tracks.filter(pl.col('chunk') == i)
            if tracks_chunk.height == 0:
                continue
            tracks_to_mapmatch_chunk = tracks.tracks_to_mapmatch(points_chunk, tracks_chunk)
            tracks_to_mapmatch.append(tracks_to_mapmatch_chunk)

        self.points = pl.concat(all_points) if all_points else pl.DataFrame([])
        self.tracks_to_mapmatch = pl.concat(tracks_to_mapmatch) if tracks_to_mapmatch else pl.DataFrame([])