
import pandas as pd
import polars as pl
import numpy as np
import os
import copy 
import pickle
from tqdm import tqdm
from typing import List, Any
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor
from massmob.model import massmodel
from massmob.engine import mode, tracks, stops, analysis, mapmatching, clustering, volumes


class Model():

    def __init__(self, points: pl.DataFrame = None):
        self.points = points if points is not None else pl.DataFrame([])

    def filter_points(self, **kwargs):
        """
        Filter points to retain only those associated with active phones, based on spatial accuracy and dispersion.

        This function applies a two-stage filter:
        1. Discards points whose spatial accuracy ("accuracy" column) exceeds the `MAX_ACCURACY` parameter threshold.
        2. For each group of points sharing the same phone identifier (`phone_id_column`), computes the maximum distance between any two points in the group.
            Groups for which this maximum distance is less than or equal to `INACTIVE_PHONE_AREA_SIDE_METERS` parameter
                are considered "inactive," and all their points are removed from the dataset.

        Parameters
        ----------
        pts : pl.DataFrame
            Polars DataFrame containing the points to filter. Must include columns: `phone_id_column`, 'x', 'y' (coordinates), and 'accuracy'.
        phone_id_column : str, optional
            Name of the column identifying each phone (default is 'phone_id').
        INACTIVE_PHONE_AREA_SIDE_METERS : float, optional
            Maximum spatial spread (in meters) below which a phone is considered inactive.
        MAX_ACCURACY : float, optional
            Maximum allowed value for the 'accuracy' column.
        """
        self.points = tracks.filtering(self.points, **kwargs)

    def build_tracks(self, **kwargs):
        """
        Construit les traces à partir des points filtrés, en mettant à jour les points avec les identifiants de traces,
        et en construisant l'objest "traces" qui contient les traces sous forme de Linetring.
        kwargs:
            MAX_SECONDS_DELAY_BETWEEN_POINTS =  60 * 60 , # délai maximum en minutes entre deux points consécutifs pouvant appartenir à une même trace
            STOP_SPEED_THRESHOLD_KMH = 1,
            IDLING_PHONE_METERS_DISTANCE = 200,
            MAKING_A_STOP_SECONDS_DELAY = 10 * 60,                    
            MIN_TRIP_DURATION_SECONDS = 60 * 2,   # durée minimale en seconds d'une trace (non conservée en dessous)
            MIN_TRIP_DISTANCE_METERS = 200
        
        >Nécessite d'utiliser .filtering() avant.
        """
        assert 'ts' in self.points.columns, 'Points are not pre-filtered, use methode .filtering() first'
                        
        self.points = tracks.build_tracked_points(self.points, **kwargs)
        self.tracks = tracks.tracks_from_points_with_stops(self.points)
    
    def analysis_tracks(self):
        """
        Compute and append summary statistics for each track in the dataset.

        This method enriches the `tracks` attribute with aggregated statistical columns
        (e.g., accuracy, durations, distances, speed, timestamps) by analyzing the collection
        of points associated with each track.
        The departure and end points are also extracted as coordinate tuples from the 
        'coordinates' column of each track.

        Preconditions
        -------------
        - `self.tracks` must be initialized and non-empty, containing at least the columns:
            - 'point_ids' : a list of point identifiers for each track
            - 'coordinates' : a list of coordinate tuples for each track

        Effects
        -------
        Updates the `tracks` attribute in-place, attaching new columns:
            - accuracy_max, accuracy_moy,
            - sampling_duration_max, sampling_duration_moy,
            - sampling_distance_max, sampling_distance_moy,
            - speed_max, speed_median, speed_95th,
            - first_ts, last_ts,
            - departure_point, end_point (tuple (x, y))

        Returns
        -------
        None
            The method updates `self.tracks` in-place.

        Raises
        ------
        AssertionError
            If `self.tracks` is not initialized.

        Example
        -------
        >>> my_analyzer.analysis_tracks()
        >>> print(my_analyzer.tracks.columns)
        """
        assert self.tracks is not None, 'Tracks are not built, build them first'
        tracks.analysis_tracks(self.tracks, self.points)

    def restrict_to_tracks(self, track_ids: List[Any]) -> "Model":
        """
        Returns a copy of the object restricted to the given track IDs. 
        Only the tracks whose 'track_id' is in track_ids and the points used by those tracks 
        are kept in the returned object.

        Parameters
        ----------
        track_ids : List[Any]
            List of track IDs to retain in the object.

        Returns
        -------
        Model
            A copy of the object with filtered tracks and points.
        """
        # 1. Filter tracks to keep only the selected track IDs
        tracks_restricted = self.tracks.filter(
            pl.col('track_id').is_in(track_ids)
        )
        
        # 2. Gather all point IDs used in the selected tracks (assumes 'point_ids' is a list column)
        # We explode 'point_ids' to flatten, then select unique ids
        point_ids_in_tracks = (
            tracks_restricted
            .explode('point_ids')
            .select('point_ids')
            .unique()
            .to_series()
            .to_list()
        )

        # 3. Filter points DataFrame to keep only those present in the above list
        points_restricted = self.points.filter(
            pl.col('point_id').is_in(point_ids_in_tracks)
        )
        
        # 4. Create and return a copy of the object with filtered tracks and points
        restricted = self.copy()
        restricted.tracks = tracks_restricted
        restricted.points = points_restricted

        return restricted

    def analysis_points(self):
        """
        Renvoie des indicateurs et des graphs décrivant le jeu de données de points filtrés. 
        
        >Nécessite d'utiliser .filtering() avant.
        """
        assert 'ts' in self.points.columns, 'Points are not pre-filtered, use methode .filtering() first'
        analysis.analysis_points(self.points)
    
    def mapmatching(self, **kwargs):
        """
        Mapmatch les traces sur le réseau routier OSM.
        >Nécessite d'utiliser .build_tracks() avant.
        >Nécessite d'utiliser .extract_road_network() avant.
        >Nécessite d'utiliser .filtering() avant.
        """
        assert self.road_nodes is not None, 'Road network is not extracted, use methode .extract_road_network() first'
        assert self.tracks is not None, 'Tracks are not built, use methode .build_tracks() first'
        self.tracks_mapmatched = mapmatching.mapmatching_parallel(
            self.tracks[['track_id','geometry']],
            self.road_nodes,self.road_links, 
            **kwargs
        )
    
    def loaded_network(self):
        """
        Calclul lacharge du réseau routier de la zone, en comptant le nombre de traces par lien routier.
        >Nécessite d'utiliser .mapmatching() avant.
        >Nécessite d'utiliser .extract_road_network() avant.
        >Nécessite d'utiliser .build_tracks() avant.
        >Nécessite d'utiliser .filtering() avant."""
        assert self.tracks_mapmatched is not None, 'Tracks are not mapmatched, use methode .mapmatching() first'
        assigned_tracks = mapmatching.fast_assign(np.ones(len(self.tracks_mapmatched)), self.tracks_mapmatched['road_link_list'].values)
        self.road_links['tracks_count'] = assigned_tracks
    
    def inference_mode_hybrid(self,model_classif):
        assert self.rail_network is not None, 'Rail network is not loaded, use methode .set_rail_network() first'     
        assert self.tracks is not None, 'Tracks are not built, use methode .build_tracks() first'
        traces_mode = mode.inference_mode_hybrid(self.tracks, self.rail_network, model_classif)
        self.tracks = self.tracks.merge(traces_mode, how='left', on='track_id')


    def inference_mode_logic_rules(self,
                       RAYON_DETECTION_TRAIN = 200,
                       PROPORTION_IN_RAIL_BUFFER = 0.7,
                       PROPORTION_IN_METRO_BUFFER = 0.7,
                       V_MAX_BIKE = 30,
                       V_MOY_MAX_BIKE = 20,
                       V_MAX_WALK = 7,
                       V_MOY_MAX_WALK = 5,
                       DISTANCE_MAX_MOTOR = 15000):
        """
        Attribut un mode de déplacemenent à chaque trace selon des règles logiques.
        >Nécessite d'utiliser .set_rail_network() avant.
        >Nécessite d'utiliser .build_tracks() avant.
        >Nécessite d'utiliser .filtering() avant.
        """
        
        assert self.rail_network is not None, 'Rail network is not loaded, use methode .set_rail_network() first'     
        assert self.tracks is not None, 'Tracks are not built, use methode .build_tracks() first'
        self.tracks = mode.inference_mode_logic_rules(
            self.tracks,self.rail_network,
            RAYON_DETECTION_TRAIN,
            PROPORTION_IN_RAIL_BUFFER,
            PROPORTION_IN_METRO_BUFFER,
            V_MAX_BIKE,
            V_MOY_MAX_BIKE,
            V_MAX_WALK,
            V_MOY_MAX_WALK,
            DISTANCE_MAX_MOTOR
            )
    
    def set_zoning(self,zones):
        """ 
        Initialise le zonage qui sera utilisé pour les analyses (communes, départements, iris, etc.)
        """
        zones.rename(columns={'insee':'zone_id'}, inplace=True)
        zones.to_crs(epsg=2154,inplace=True)
        self.zones = zones

    def get_home_place(self):
        """Attribut un domicile à chaque télephone selon ses emplacement dans la journée.
        > Nécessite d'utiliser .set_zoning() avant.
        >Nécessite d'utiliser .filtering() avant."""
        assert self.zones is not None, 'Zones are not set, use methode .set_zoning() first'
        if 'domicile' in self.phones.columns:
            print('Home places already computed')
        else:
            domiciles = clustering.cluster_home(self.points, self.zones,NOMBRE_MIN_POINT_PAR_CLUSTER = 3,RAYON_DE_PRISE_EN_COMPTE_DU_CLUSTER = 50)
            self.phones= self.phones.merge(domiciles,how='left',on='phone_id')
            homes_in_zones = analysis.number_by_zone(self.zones,self.phones,'domicile')
            self.zones = self.zones.merge(homes_in_zones,how='left',on='zone_id')
        
    def get_work_place(self):
        """Attribut un lieu d'emploi à chaque télephone selon ses emplacement dans la journée.
        > Nécessite d'utiliser .set_zoning() avant.
        >Nécessite d'utiliser .filtering() avant.
        """
                 
        assert self.zones is not None, 'Zones are not set, use methode .set_zoning() first'
        if 'emploi' in self.phones.columns:
            print('Work places already computed')
        else:
            work = clustering.cluster_work(self.points, self.zones,NOMBRE_MIN_POINT_PAR_CLUSTER = 3,RAYON_DE_PRISE_EN_COMPTE_DU_CLUSTER = 50)
            self.phones= self.phones.merge(work,how='left',on='phone_id')
            work_in_zones = analysis.number_by_zone(self.zones,self.phones,'emploi')
            self.zones = self.zones.merge(work_in_zones,how='left',on='zone_id')

    def get_volumes(self):
        self.volumes= volumes.build_od_matrix(self.tracks, self.zones)
    
    def phones_dataset_analysis(self):
        self.phones = analysis.phones_dataset_analysis(self.points, self.phones)