import pandas as pd
import polars as pl
import os
import copy 
import pickle
from tqdm import tqdm
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor
from massmob.model import model, plotmodel
from massmob.io import io


def read_parquets(folder, omitted_attributes=(), only_attributes=None):
    files = [
        f for f in os.listdir(folder)
        if f.endswith('.parquet')
    ]
    keys = [f.split('.parquet')[0] for f in files]

    # init model
    self = MassModel()

    iterator = tqdm(keys)
    for key in iterator:
        if key in omitted_attributes:
            continue
        if only_attributes is not None and key not in only_attributes:
            continue
        iterator.desc = key
        fpath = os.path.join(folder, f"{key}.parquet")
        self.__setattr__(key, pl.read_parquet(fpath))
    return self

def from_singlespot_zip(zip_path, **kwargs):
    pts = io.singlespot_zip_to_points(zip_path, **kwargs)
    pts = pts.rename({'sptId':'phone_id'})
    return MassModel(pts)


class MassModel(
        model.Model,
        plotmodel.PlotModel
        ):

    def __init__(self, points=None, MAX_ACCURACY=50):
        """
        points : DataFrame with columns ['phone_id','latitude','logitude','eventDate','accuracy']
        Initialise l'objet MassModel avec les points bruts
        """
        self.points = points
    
        if points is not None and len(points):
            self.phones = pl.DataFrame({"phone_id": points['phone_id'].unique()})
    
    def describe(self):
        results = {
            'Points': f'{len(self.points):,}',
            'Unique phones': f'{len(self.points.phone_id.unique()):,}',
        }
        if hasattr(self, 'tracks') and self.tracks is not None:
            results.update({'Tracks': f'{len(self.tracks):,}'})
        return pd.Series(results)

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

        def export_parquet(key, value):
            fpath = os.path.join(folder, f"{key}.parquet")
            # Pandas DataFrame
            if hasattr(value, "to_parquet"):
                value.to_parquet(fpath, index=False)
            # Polars DataFrame
            elif "polars" in str(type(value)).lower():
                value.write_parquet(fpath)
            # Autre type : sauvegarde non implémentée
            else:
                print(f"Export failed {key} (type: {type(value)})")

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


    def copy(self):
        return copy.deepcopy(self)