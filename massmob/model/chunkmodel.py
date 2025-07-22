import os
import polars as pl
from math import ceil
from tqdm import tqdm
from massmob.engine import mode, tracks, stops, analysis, mapmatching, clustering, volumes


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
        for i in range(self.nchunks):
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

        for i in range(self.nchunks):
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

    def analysis_tracks(self):
        """
        Analyze tracks per chunk, storing result in self.tracks.
        """
        dfs = []
        for i in range(self.nchunks):
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