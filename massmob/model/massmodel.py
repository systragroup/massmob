import pandas as pd
import polars as pl
import os
import copy 
import pickle
from tqdm import tqdm
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor
from massmob.model import model, plotmodel, integritymodel, chunkmodel
from massmob.io import io
import json

def read_parquets(folder, omitted_attributes=(), only_attributes=None):
    files = [
        f for f in os.listdir(folder)
        if f.endswith('.parquet')
    ]
    keys = [f.split('.parquet')[0] for f in files]

    self = MassModel()

    iterator = tqdm(keys)
    for key in iterator:
        if key in omitted_attributes:
            continue
        if only_attributes is not None and key not in only_attributes:
            continue
        iterator.desc = key
        fpath = os.path.join(folder, f"{key}.parquet")
        setattr(self, key, pl.read_parquet(fpath))

    # lecture des args.json
    args_path = os.path.join(folder, "args.json")
    if os.path.exists(args_path):
        with open(args_path, "r", encoding="utf-8") as f:
            args = json.load(f)
        for k, v in args.items():
            setattr(self, k, v)

    return self

def from_singlespot_zip(zip_path, **kwargs):
    pts = io.singlespot_zip_to_points(zip_path, **kwargs)
    pts = pts.rename({'sptId':'phone_id'})
    return MassModel(pts)


class MassModel(
        model.Model,
        plotmodel.PlotModel,
        integritymodel.IntegrityModel,
        chunkmodel.ChunkModel
        ):
    
    def __init__(self, points=None, nchunks=1, **kwargs):
        self.nchunks = nchunks
        model.Model.__init__(self, points=points, **kwargs)
        chunkmodel.ChunkModel.__init__(self, points=points, nchunks=nchunks, **kwargs)

    def filter_points(self, **kwargs):
        """
        Dispatches to Model or ChunkModel depending on the chunk structure.
        """
        if getattr(self, "nchunks", 1) > 1:
            return chunkmodel.ChunkModel.filter_points(self, **kwargs)
        else:
            return model.Model.filter_points(self, **kwargs)
    
    def build_tracks(self, **kwargs):
        """
        Dispatches to Model or ChunkModel depending on the chunk structure.
        """
        if getattr(self, "nchunks", 1) > 1:
            return chunkmodel.ChunkModel.build_tracks(self, **kwargs)
        else:
            return model.Model.build_tracks(self, **kwargs)
    
    def analysis_tracks(self):
        """
        Dispatches to Model or ChunkModel depending on the chunk structure.
        """
        if getattr(self, "nchunks", 1) > 1:
            return chunkmodel.ChunkModel.analysis_tracks(self)
        else:
            return model.Model.analysis_tracks(self)
        
    def describe(self):
        """
        Generate summary statistics about the main data attributes.

        Returns
        -------
        dict
            Dictionary with formatted counts of points, unique phones, and tracks.

        Notes
        -----
        - Uses Polars-native commands for performance.
        - Returns a dictionary, Polars-first (not a pandas Series).
        """
        # Compute the number of points in the main DataFrame
        points_count = self.points.height
        # Compute the number of unique phone IDs in the main DataFrame
        unique_phones = self.points["phone_id"].n_unique()

        results = {
            "Points": f"{points_count:,}",
            "Unique phones": f"{unique_phones:,}",
        }

        # If self.tracks exists and is not empty, add the track count
        if hasattr(self, "tracks") and self.tracks is not None and self.tracks.height > 0:
            results["Tracks"] = f"{self.tracks.height:,}"

        return results


    def to_parquets(
        self,
        folder,
        omitted_attributes=(),
        only_attributes=None,
        max_workers=1,
        remove_first=True
    ):
        if remove_first:
            shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

        args = {}

        def export_parquet(key, value):
            fpath = os.path.join(folder, f"{key}.parquet")
            # Pandas DataFrame
            if hasattr(value, "to_parquet"):
                value.to_parquet(fpath, index=False)
            # Polars DataFrame
            elif "polars" in str(type(value)).lower():
                value.write_parquet(fpath)
            # Autre → stocké dans args.json
            else:
                args[key] = value

        if max_workers == 1:
            iterator = tqdm(self.__dict__.items())
            for key, value in iterator:
                iterator.desc = key
                if key in omitted_attributes:
                    continue
                if only_attributes is not None and key not in only_attributes:
                    continue
                export_parquet(key, value)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for key, value in self.__dict__.items():
                    if key in omitted_attributes:
                        continue
                    if only_attributes is not None and key not in only_attributes:
                        continue
                    executor.submit(export_parquet, key, value)

        # Sauvegarde des args en JSON
        if args:
            with open(os.path.join(folder, "args.json"), "w", encoding="utf-8") as f:
                json.dump(args, f, ensure_ascii=False, indent=2)


    def copy(self):
        return copy.deepcopy(self)