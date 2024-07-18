import pandas as pd
import numpy as np
import os
import copy 
import pickle
from tqdm import tqdm
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor
from massmob.model import massmodel
from massmob.engine import mode, tracks, stops, analysis, mapmatching, clustering, volumes


class Model():

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
        self.tracks = tracks.points_to_tracks(self.points)
    
    def analysis_points(self):
        """
        Renvoie des indicateurs et des graphs décrivant le jeu de données de points filtrés. 
        
        >Nécessite d'utiliser .filtering() avant.
        """
        assert 'ts' in self.points.columns, 'Points are not pre-filtered, use methode .filtering() first'
        analysis.analysis_points(self.points)
    
    def analysis_tracks(self):
        """
        Renvoie des indicateurs et des graphs décrivnat le jeu de données de traces.
        >Nécessite d'utiliser .build_tracks() avant.
        >Nécessite d'utiliser .filtering() avant.
        """
        assert self.tracks is not None, 'Tracks are not built, use methode .build_tracks() first'
        analysis.analysis_tracks(self.tracks)
    
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