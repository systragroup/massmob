import pandas as pd
import os
import copy 
import pickle
from tqdm import tqdm
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor
from fcdmodel.model import model, plotmodel
from fcdmodel.io import io


def read_zippedpickles(folder, omitted_attributes=(), only_attributes=None):
    files = os.listdir(folder)
    keys = [
        file.split('.zippedpickle')[0]
        for file in files
        if '.zippedpickle' in file
    ]
    self = FCDModel()
    iterator = tqdm(keys)
    for key in iterator:
        if key in omitted_attributes:
            continue
        if only_attributes is not None and key not in only_attributes:
            continue

        iterator.desc = key
        with open('%s/%s.zippedpickle' % (folder, key), 'rb') as file:
            buffer = file.read()
            bigbuffer = zlib.decompress(buffer)
            self.__setattr__(key, pickle.loads(bigbuffer))
    return self


class FCDModel(
        model.Model,
        plotmodel.PlotModel
        ):

    def __init__(self, points=None, MAX_ACCURACY=100):
        """
        points : DataFrame with columns ['phone_id','latitude','logitude','eventDate','accuracy']
        Initialise l'objet FCData avec les points bruts
        """
        self.points = points
    
        if points is not None and len(points):
            if 'sptId' in points.columns:
                points.rename(columns={'sptId':'phone_id'}, inplace=True)
            self.phones = pd.DataFrame(points['phone_id'].unique()).rename(columns={0:'phone_id'})
    
    def describe(self):
        results = {
            'Points': f'{len(self.points):,}',
            'Unique phones': f'{len(self.points.phone_id.unique()):,}',
        }
        if hasattr(self, 'tracks') and self.tracks is not None:
            results.update({'Tracks': f'{len(self.tracks):,}'})
        return pd.Series(results)

    def to_zippedpickles(
        self,
        folder,
        omitted_attributes=(),
        only_attributes=None,
        max_workers=1,
        complevel=-1,
        remove_first=True
    ):
        if remove_first:
            shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        if max_workers == 1:
            iterator = tqdm(self.__dict__.items())
            for key, value in iterator:
                iterator.desc = key
                if key in omitted_attributes:
                    continue
                if only_attributes is not None and key not in only_attributes:
                    continue
                io.to_zippedpickle(
                    value, '%s/%s.zippedpickle' % (folder, key),
                    complevel=complevel
                )
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for key, value in self.__dict__.items():
                    if key in omitted_attributes:
                        continue
                    if only_attributes is not None and key not in only_attributes:
                        continue
                    executor.submit(
                        io.to_zippedpickle,
                        value,
                        r'%s/%s.zippedpickle' % (folder, key),
                        complevel=complevel

    
                    )

    def copy(self):
        return copy.deepcopy(self)