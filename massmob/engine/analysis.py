import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import shapely
from pathlib import Path
import seaborn as sns
sns.set_style("whitegrid")


def hist_plot(serie, unit, bins, range):
    name = serie.name
    liste = np.array(serie.values)

    liste_sorted = np.sort(liste)

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax2 = ax1.twinx()

    ax1.plot(liste_sorted, np.linspace(0, 1, len(liste_sorted), endpoint=False))
    ax2.hist(liste, bins=bins,color='grey',alpha=0.5)
    ax1.set_xlabel(name)
    ax1.set_ylabel('Number of tracks')
    ax2.set_ylabel('Cumulative distribution')
    mediane = round(np.median(liste),1)
    moyenne = round(np.mean(liste),1)
    ax1.axvline(mediane, color='green', linestyle='dashed', linewidth=1, label=f'Median : {mediane}')
    ax1.axvline(moyenne, color='red', linestyle='dotted', linewidth=1, label=f'Mean : {moyenne}')
    ax1.legend(loc='center right')
    ax2.legend()
    plt.title(f'Distribution of {name}')



def analysis_tracks(tracks):

    """
    :param tracks: dataframe of tracks 
    :return: plots of the tracks and indicators

    """
    ech_point_tracks = tracks
    liste_acc = np.array(ech_point_tracks['accuracy_moy'])
    liste_acc.sort()

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax2 = ax1.twinx()

    ax1.plot(liste_acc, np.linspace(0, 1, len(liste_acc), endpoint=False))
    ax2.hist(liste_acc, bins=range(0,50,1),color='grey',alpha=0.5)
    fig.legend(['Cumulative distribution','Histogram'])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    ax1.set_title('Cumulative distribution and histogram of the average accuracy of the traces')
    ax1.set_xlabel('Accuracy (m)')
    ax1.set_ylabel('Cumulative distribution')
    ax2.set_ylabel('Number of traces')
    plt.xlim(0,50)

    plt.figure(figsize=(10,5))
    plt.hist(np.array(ech_point_tracks['accuracy_max']), bins=range(0,200,1),color='green',alpha=0.5, edgecolor='black')
    plt.title('Histogram of the maximum accuracy of the traces')
    plt.xlabel('Accuracy (m)')
    plt.ylabel('Number of traces')
    plt.xlim(0,200)

    liste_duration = np.array(ech_point_tracks['duration'])
    liste_duration.sort()

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax2 = ax1.twinx()

    ax1.plot(liste_duration, np.linspace(0, 1, len(liste_duration), endpoint=False), color='red')
    ax2.hist(liste_duration, bins=range(0,20000,200),color='grey',alpha=0.5)
    fig.legend(['Cumulative distribution','Histogram'])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    ax1.set_title('Cumulative distribution and histogram of the duration of the traces')
    ax1.set_xlabel('Duration (s)')
    ax1.set_ylabel('Cumulative distribution')
    ax2.set_ylabel('Number of traces')
    plt.xlim(0,20000)

    liste_distance = np.array(ech_point_tracks['track_length'])
    liste_distance.sort()

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax2 = ax1.twinx()

    ax1.plot(liste_distance, np.linspace(0, 1, len(liste_distance), endpoint=False), color='green')
    ax2.hist(liste_distance, bins=range(0,40000,500),color='grey',alpha=0.5)
    fig.legend(['Cumulative distribution','Histogram'])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    ax1.set_title('Cumulative distribution and histogram of the total distance of the traces')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Cumulative distribution')
    ax2.set_ylabel('Number of traces')
    plt.xlim(0,40000)

    liste_tech_moy = np.array(ech_point_tracks['time_prec_moy'])
    liste_tech_moy.sort()

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax2 = ax1.twinx()

    ax1.plot(liste_tech_moy, np.linspace(0, 1, len(liste_tech_moy), endpoint=False), color='black')
    ax2.hist(liste_tech_moy, bins=range(0,500,10),color='grey',alpha=0.5)
    fig.legend(['Cumulative distribution','Histogram'])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    ax1.set_title('Cumulative distribution and histogram of the average sampling time of the traces')
    ax1.set_xlabel('Sampling time (m)')
    ax1.set_ylabel('Cumulative distribution')
    ax2.set_ylabel('Number of traces')
    plt.xlim(0,500)

    liste_tech_max = np.array(ech_point_tracks['time_prec_max'])
    liste_tech_max.sort()

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax2 = ax1.twinx()

    ax1.plot(liste_tech_max, np.linspace(0, 1, len(liste_tech_max), endpoint=False), color='black')
    ax2.hist(liste_tech_max, bins=range(0,700,10),color='grey',alpha=0.5)
    fig.legend(['Cumulative distribution','Histogram'])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    ax1.set_title('Cumulative distribution and histogram of the maximum sampling time of the traces')
    ax1.set_xlabel('Sampling time (m)')
    ax1.set_ylabel('Cumulative distribution')
    ax2.set_ylabel('Number of traces')
    plt.xlim(0,700)

    liste_dist_from_prec_moy = np.array(ech_point_tracks['dist_from_prec_moy'])
    liste_dist_from_prec_moy.sort()

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax2 = ax1.twinx()

    ax1.plot(liste_dist_from_prec_moy, np.linspace(0, 1, len(liste_dist_from_prec_moy), endpoint=False), color='orange')
    ax2.hist(liste_dist_from_prec_moy, bins=range(0,2000,10),color='grey',alpha=0.5)
    fig.legend(['Cumulative distribution','Histogram'])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    ax1.set_title('Cumulative distribution and histogram of the average distance from previous point time of the traces')
    ax1.set_xlabel('Average distance from previous point (m)')
    ax1.set_ylabel('Cumulative distribution')
    ax2.set_ylabel('Number of traces')
    plt.xlim(0,2000)
    liste_v_max = np.array(ech_point_tracks['v_instant_max']*3.6)
    liste_v_max.sort()

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax2 = ax1.twinx()

    ax1.plot(liste_v_max, np.linspace(0, 1, len(liste_v_max), endpoint=False), color='pink')
    ax2.hist(liste_v_max, bins=range(0,160,1),color='grey',alpha=0.5)
    fig.legend(['Cumulative distribution','Histogram'])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    ax1.set_title('Cumulative distribution and histogram of the maximum instant speed point time of the traces')
    ax1.set_xlabel('Maximum instant speed (km/h)')
    ax1.set_ylabel('Cumulative distribution')
    ax2.set_ylabel('Number of traces')
    plt.xlim(0,160)

    liste_v_moy = np.array(ech_point_tracks['v_instant_moy']*3.6)
    liste_v_moy.sort()

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax2 = ax1.twinx()

    ax1.plot(liste_v_moy, np.linspace(0, 1, len(liste_v_moy), endpoint=False), color='pink')
    ax2.hist(liste_v_moy, bins=range(0,160,1),color='grey',alpha=0.5)
    fig.legend(['Cumulative distribution','Histogram'])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    ax1.set_title('Cumulative distribution and histogram of the average instant speed point time of the traces')
    ax1.set_xlabel('Average instant speed (km/h)')
    ax1.set_ylabel('Cumulative distribution')
    ax2.set_ylabel('Number of traces')
    plt.xlim(0,160)


    plt.rcParams['figure.figsize'] = [6, 3]
    traces = tracks
    
    
    plt.show()
    trace_duration = traces['duration']
    trace_distances = traces['track_length']
    trace_speeds = tracks['v_instant_moy']
    hist_plot((trace_duration / 60), 'minutes', bins=30, range=(0, 120))

    plt.show()
    hist_plot((trace_distances / 1000), 'km', bins=50, range=(0, 50))

    plt.show()
    hist_plot(trace_speeds.loc[trace_speeds<1e9], 'km/h', bins=75, range=(0, 100))

    plt.show()
    traces.groupby('day')['phone_id'].nunique().plot.bar()

    plt.show()
    traces.groupby('day')['track_id'].nunique().plot.bar()
    plt.show()
    return  


def phones_dataset_analysis(points,phones):
    nbr_points_phone = points.copy()
    nbr_points_phone['accuracy_max'] = nbr_points_phone['accuracy']
    nbr_points_phone['accuracy_moy'] = nbr_points_phone['accuracy']
    nbr_points_phone['accuracy_min'] = nbr_points_phone['accuracy']
    nbr_points_phone = nbr_points_phone.groupby(['phone_id']).agg({'accuracy_max':'max','accuracy_moy':'mean','accuracy_min':'min','latitude':'count'}).reset_index()
    nbr_points_phone = nbr_points_phone.rename(columns={'latitude':'nbr_points'})
    nbr_points_phone=nbr_points_phone[['phone_id','nbr_points','accuracy_max','accuracy_moy','accuracy_min']]
    print('Accuracy indicator computed, number of points per phone computed')
    gpd_points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.x, points.y))
    print('GDF conversion done')
    points_centroid_by_phone = gpd_points.groupby('phone_id').agg({'geometry': lambda x: shapely.geometry.MultiPoint(x.tolist()).centroid})
    points_centroid_by_phone.reset_index(inplace=True)
    print('Centroid computed')
    points_centroid_by_phone = points_centroid_by_phone.rename(columns={'geometry':'centroid'})
    points_centroid_by_phone = points_centroid_by_phone.merge(nbr_points_phone, on='phone_id')
    phones = phones.merge(points_centroid_by_phone, on='phone_id')
    return phones
    

def analysis_points(points):
    """
    :param points: dataframe of points with columns ['phone_id', 'phoneId', 'datetime', 'accuracy', 'v_instant', 'geometry']
    :return: plots of the points and indicators
    """
    telephones = points['phone_id'].unique()
    print('Nombre de points: ', points.shape[0])
    print('Nombre de télephones: ', len(telephones))

    liste_accuracy = np.array(points['accuracy'])
    liste_accuracy.sort()

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax2 = ax1.twinx()

    ax1.plot(liste_accuracy, np.linspace(0, 1, len(liste_accuracy), endpoint=False), color='red')
    ax2.hist(liste_accuracy, bins=range(0,2000,5),color='blue',alpha=0.2)
    fig.legend(['Cumulative distribution','Histogram'])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    ax1.set_title('Cumulative distribution and histogram of the accuracy of the points')
    ax1.set_xlabel('Duration (s)')
    ax1.set_ylabel('Cumulative distribution')
    ax2.set_ylabel('Number of traces')
    plt.xlim(0,500)

    if 'os' in points.columns:
        points_os = points.copy()
        points_os['count'] = 1
        points_os['accuracy_max'] = points_os['accuracy']
        points_os['accuracy_moy'] = points_os['accuracy']
        points_os = points_os.groupby(['os']).agg({'accuracy_max':'max', 'accuracy_moy':'mean','count':np.sum}).reset_index()
        points_os.set_index('os', inplace=True)
        points_os['accuracy_max'].plot(kind='bar', figsize=(5,5),xlabel='OS', ylabel='Max accuracy (m)', title='Max accuracy by OS')
    return


def number_by_zone(zoning,phones, label):
    """
    Plot the number of points by zone, works for "domicile" and "emplois" for example
    :param zoning: zoning
    :type zoning: geopandas.GeoDataFrame
    :param phones: dataframe of phones with columns ['phone_id', 'domicile','emplois']
    :type phones: pandas.DataFrame
    :param label: column of phones to plot
    :type label: string
    :return: plot

    """
    corr = {'emploi':'work', 'domicile':'home'}
    phones_filtered = phones[['phone_id',f'{label}']]

    phones_filtered = gpd.GeoDataFrame(phones_filtered,geometry=phones_filtered[f'{label}'],crs=2154)


    zones = phones_filtered.sjoin(zoning, predicate='intersects')

    zones = zones.groupby('zone_id').count().reset_index()
    zones = zoning.merge(pd.DataFrame(zones)[['zone_id', 'phone_id']], on='zone_id', how='left')
    zones.rename(columns={'phone_id': f'nbr_{label}'}, inplace=True)
    zones[f'nbr_{label}'] = zones[f'nbr_{label}'].fillna(0)
    zones =pd.DataFrame(zones)[['zone_id', f'nbr_{label}']]
    
    zoning = zoning.merge(zones, on='zone_id', how='left')
    zoning = zoning[['zone_id', f'nbr_{label}','population_totale']]
    zoning[f'repr_rate_{corr[label]}'] = zoning.apply(lambda x: x['population_totale'] / x[f'nbr_{label}'] if x[f'nbr_{label}'] != 0 else 0, axis=1)
    zoning.drop(columns=['population_totale'], inplace=True)

    return zoning
    


