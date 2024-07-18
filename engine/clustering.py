import pandas as pd
import geopandas as gpd
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN


def _calc_time_elapsed(row):
    """
    Calcule le temps écoulé entre le point et le point précédent en seconde
    :param row: ligne du dataframe
    :type: pandas dataframe row
    :return: temps écoulé en seconde
    """
    try:
        time_elapsed = (row['eventDate'] - row['time_prec']).total_seconds()
        if time_elapsed < 0:
            time_elapsed = 0
        return time_elapsed
    except:
        time_elapsed = 0
        return time_elapsed

def _calc_dist_from_prec(row):
    """
    Calcule la distance entre le point et le point précédent en mètre
    :param row: ligne du dataframe
    :type: pandas dataframe row
    :return: distance en mètre
    """
    try:
        dist = row['geometry'].distance(row['loc_prec'])
        if dist < 0:
            dist = 0
        return dist
    except:
        dist = 0
        return dist

def _calc_v_instant(row):
    """
    Calcule la vitesse instantanée en km/h
    :param row: ligne du dataframe
    :type: pandas dataframe row
    :return: vitesse instantanée en km/h"""
    try:
        v = row['dist_from_prec'] / row['time_elapsed'] * 3,6
        if v < 0:
            v = 0
        return v
    except:
        v = 0
        return v

def cluster_home(points, zoning, NOMBRE_MIN_POINT_PAR_CLUSTER=2, RAYON_DE_PRISE_EN_COMPTE_DU_CLUSTER=100):
    """
    Attribut un lieu de domicile à chaque telephone
    :param points: dataframe des points de geolocalisation
    :type: pandas dataframe des points avec les colonnes 'eventDate', 'phone_id'
    :param zoning: dataframe des zones 
    :type: geopandas dataframe du zoning
    :return: dataframe des domiciles"""
    communes = zoning.copy()
    communes.astype({'zone_id': 'str'})
    
   

    print('Nombre de points :', len(points))
    print('Nombre telephone :', len(points.phone_id.unique()))
    ech_tel = points['phone_id'].unique()
    ech_points = points.loc[points['phone_id'].isin(ech_tel)]


    ech_points = gpd.GeoDataFrame(ech_points, geometry=gpd.points_from_xy(ech_points.longitude, ech_points.latitude), crs= 'EPSG:4326')
    ech_points = ech_points.to_crs('EPSG:2154')
    ech_points['time_prec'] = ech_points['eventDate'].shift(1)
    ech_points['loc_prec'] = ech_points['geometry'].shift(1)
    ech_points['time_elapsed'] = ech_points.apply(_calc_time_elapsed, axis=1)
    ech_points['dist_from_prec'] = ech_points.apply(_calc_dist_from_prec, axis=1)
    ech_points['v_instant'] = ech_points.apply(_calc_v_instant, axis=1)

    ech_points = ech_points[ech_points['accuracy'] < 50]
    ech_points = ech_points[ech_points['eventDate'].apply(lambda x :x.hour).isin([20,21,22,23,0,1,2,3,4,5,6,7])]
    ech_points = ech_points[ech_points['v_instant'] < 2]
    ech_points_group_by_tel = ech_points.groupby('phone_id')

    liste_dom=[]
    i=0
    for tel, points_tel in ech_points_group_by_tel:
        print(f'phone {i/len(ech_points_group_by_tel)} ')
        liste_points = np.array([[point.coords[:][0][0],point.coords[:][0][1]] for point in list(points_tel['geometry'])])
        try:
            clustering = DBSCAN(eps=100, min_samples=2).fit(liste_points) 
        except ValueError:
        
            print('ValueError')
            i+=1
            continue
        labels = clustering.labels_
        compteur = Counter(labels)
        points_tel['cluster'] = labels
        valeur_plus_frequente = compteur.most_common(1)[0][0]
        domicile = points_tel[points_tel['cluster'] == valeur_plus_frequente].unary_union.centroid
        output = {'phone_id':tel, 'domicile':domicile}
        liste_dom.append(output)
        i+=1
    domicile_df = pd.DataFrame(liste_dom)
    return domicile_df

def cluster_work(points,zoning,NOMBRE_MIN_POINT_PAR_CLUSTER=1,RAYON_DE_PRISE_EN_COMPTE_DU_CLUSTER=100):
    """
    Attribut un lieu de travail à chaque telephone
    :param points: dataframe des points de geolocalisation
    :type: pandas dataframe des points avec les colonnes 'eventDate', 'phone_id'
    :param zoning: dataframe des zones de travail
    :type: geopandas dataframe du zoning 
    :param NOMBRE_MIN_POINT_PAR_CLUSTER: nombre minimum de point pour former un cluster
    :type: int
    :param RAYON_DE_PRISE_EN_COMPTE_DU_CLUSTER: rayon de prise en compte du cluster
    :type: int
    :return: dataframe des emplois par phone_id
    """
    communes = zoning.copy()
    communes.astype({'zone_id': 'str'})
    
   

    print('Nombre de points :', len(points))
    print('Nombre telephone :', len(points.phone_id.unique()))
    ech_tel = points['phone_id'].unique()
    ech_points = points.loc[points['phone_id'].isin(ech_tel)]
    ech_points = gpd.GeoDataFrame(ech_points, geometry=gpd.points_from_xy(ech_points.longitude, ech_points.latitude), crs= 'EPSG:4326')
    ech_points = ech_points.to_crs('EPSG:2154')
    ech_points['time_prec'] = ech_points['eventDate'].shift(1)
    ech_points['loc_prec'] = ech_points['geometry'].shift(1)
    ech_points['time_elapsed'] = ech_points.apply(_calc_time_elapsed, axis=1)
    ech_points['dist_from_prec'] = ech_points.apply(_calc_dist_from_prec, axis=1)
    ech_points['v_instant'] = ech_points.apply(_calc_v_instant, axis=1)

    ech_points = ech_points[ech_points['accuracy'] < 50]
    ech_points = ech_points[~ech_points['eventDate'].apply(lambda x :x.day).isin([19,20,26,27])]
    ech_points = ech_points[ech_points['eventDate'].apply(lambda x :x.hour).isin([9,10,11,12,13,14,15,16,17,18,19])]
    ech_points = ech_points[ech_points['v_instant'] < 2]
    ech_points_group_by_tel = ech_points.groupby('phone_id')

    liste_dom=[]
    i=0
    for tel, points_tel in ech_points_group_by_tel:
        print(f'phone {i/len(ech_points_group_by_tel)}')
        liste_points = np.array([[point.coords[:][0][0],point.coords[:][0][1]] for point in list(points_tel['geometry'])])
        try:
            clustering = DBSCAN(eps=100, min_samples=2).fit(liste_points) 
        except ValueError:
        
            print('ValueError')
            i+=1
            continue
        labels = clustering.labels_
        compteur = Counter(labels)
        points_tel['cluster'] = labels
        valeur_plus_frequente = compteur.most_common(1)[0][0]
        emploi = points_tel[points_tel['cluster'] == valeur_plus_frequente].unary_union.centroid
        output = {'phone_id':tel, 'emploi':emploi}
        liste_dom.append(output)
        i+=1
    domicile_df = pd.DataFrame(liste_dom)
    return domicile_df