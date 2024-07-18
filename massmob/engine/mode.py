from pathlib import Path
import pandas as pd
import geopandas as gpd
import shapely
import matplotlib.pyplot as plt
import numpy as np
import folium
import pickle
import tqdm
import datetime 


def proportion_in_buffer(trace, buffer):
    """
    Calcule la proportion d'une trace qui est dans un buffer
    :param trace: trace
    :type trace: shapely.geometry.LineString
    :param buffer: buffer
    :type buffer: shapely.geometry.Polygon
    :return: proportion

    """
    # extraire points de la trace
    # boucle sur les points / part des points dans trace(buffer)
    # affinage sur la distance (efficace)

    prop = trace.intersection(buffer).length / trace.length
    return prop

def inference_mode_logic_rules(tracks,
                   rail_network,
                   RAYON_DETECTION_TRAIN,
                   PROPORTION_IN_RAIL_BUFFER,
                   PROPORTION_IN_METRO_BUFFER,
                   V_MAX_BIKE,
                   V_MOY_MAX_BIKE,
                   V_MAX_WALK,
                   V_MOY_MAX_WALK,
                   DISTANCE_MAX_MOTOR):
    """Associe le mode à une trace selon des règles logiques
    :param tracks: traces
    :type tracks: pandas. 
    :param rail_network: réseau ferré
    :type rail_network: geopandas.GeoDataFrame
    :param RAYON_DETECTION_TRAIN: rayon de détection du train
    :type RAYON_DETECTION_TRAIN: float
    :return: traces avec le mode associé"""

    # railways_geometry = rail_network.geometry.buffer(RAYON_DETECTION_TRAIN).unary_union
    metro_network_geometry = rail_network[rail_network['type_TC'] == 'metro'].geometry.buffer(RAYON_DETECTION_TRAIN).unary_union
    rer_network_geometry = rail_network[rail_network['type_TC'] == 'rer'].geometry.buffer(RAYON_DETECTION_TRAIN).unary_union
    train_network_geometry = rail_network[rail_network['type_TC'] == 'train'].geometry.buffer(RAYON_DETECTION_TRAIN).unary_union

    # traces_well_sampled = tracks[tracks['time_prec_max']<1200]
    traces_well_sampled = tracks
    traces_well_sampled = traces_well_sampled[traces_well_sampled['time_prec_max'] > 0]

    # traces_well_sampled['proportion_in_rail'] = 0
    traces_well_sampled['proportion_in_metro'] = 0
    traces_well_sampled['proportion_in_rer'] = 0
    traces_well_sampled['proportion_in_train'] = 0    

    # traces_well_sampled['proportion_in_rail'] = traces_well_sampled.apply(lambda x: proportion_in_buffer(x['geometry'], railways_geometry), axis=1)
    traces_well_sampled['proportion_in_metro'] = traces_well_sampled.apply(lambda x: proportion_in_buffer(x['geometry'], metro_network_geometry), axis=1)
    traces_well_sampled['proportion_in_rer'] = traces_well_sampled.apply(lambda x: proportion_in_buffer(x['geometry'], rer_network_geometry), axis=1)
    traces_well_sampled['proportion_in_train'] = traces_well_sampled.apply(lambda x: proportion_in_buffer(x['geometry'], train_network_geometry), axis=1)
    
    traces_well_sampled['mode'] = 'walk'

    traces_well_sampled.loc[traces_well_sampled['v_instant_moy'] > V_MOY_MAX_WALK, 'mode'] = 'bike'
    traces_well_sampled.loc[traces_well_sampled['v_instant_max'] > V_MAX_WALK, 'mode'] = 'bike'
    traces_well_sampled.loc[traces_well_sampled['v_instant_moy'] > V_MOY_MAX_BIKE, 'mode'] = 'motorized'
    traces_well_sampled.loc[traces_well_sampled['track_length'] > DISTANCE_MAX_MOTOR, 'mode'] = 'motorized'
    traces_well_sampled.loc[traces_well_sampled['v_instant_max'] > V_MAX_BIKE, 'mode'] = 'motorized'

    traces_well_sampled.loc[traces_well_sampled['proportion_in_metro'] > PROPORTION_IN_METRO_BUFFER, 'mode'] = 'metro'
    traces_well_sampled.loc[traces_well_sampled['proportion_in_rer'] > PROPORTION_IN_RAIL_BUFFER, 'mode'] = 'rer'
    traces_well_sampled.loc[traces_well_sampled['proportion_in_train'] > PROPORTION_IN_RAIL_BUFFER, 'mode'] = 'train'

    # traces_well_sampled.loc[traces_well_sampled['dist_departure_arrival']<100,'mode'] = 'motorized'
    
    # convert into str
    traces_output_gpd = gpd.GeoDataFrame(traces_well_sampled, geometry=traces_well_sampled['geometry'], crs=2154)
    traces_output_gpd.reset_index(inplace=True)
    traces_full = traces_output_gpd[['track_id', 'geometry', 'departure_point', 'arrival_point', 'mode']]

    traces_mode = traces_full[['track_id', 'mode']]
    traces_mode = tracks.merge(traces_mode, on='track_id', how='left')
    return traces_mode

def inference_mode_hybrid(traces, rail_network, model_classif):

    traces_mode = inference_mode_logic_rules(tracks = traces,
                                            rail_network = rail_network,
                                            RAYON_DETECTION_TRAIN = 200,
                                            PROPORTION_IN_RAIL_BUFFER = 0.6,
                                            PROPORTION_IN_METRO_BUFFER = 0.7,
                                            V_MAX_BIKE = 40,
                                            V_MOY_MAX_BIKE = 30,
                                            V_MAX_WALK = 15,
                                            V_MOY_MAX_WALK = 7,
                                            DISTANCE_MAX_MOTOR = 20000)

    id_traces_train = traces_mode[traces_mode['mode'] == 'train']['track_id'].unique()

    def istrain(row):
        if row['track_id'] in id_traces_train and row['track_length'] > 10000:
            return 'train'
        else:
            return row['mode_computed']

    def pasnan(x):
        if x > 1 or x < 0 or np.isnan(x):
            return 0
        else:
            return x
        
    def motorizedornot(row):
        if row['track_length'] > 15000 or row['v_instant_moy'] > 25:
            return 'motorized'
        elif row['mode_computed'] == 'car' or row['mode_computed'] == 'bus':
            return 'motorized'
        else:
            return row['mode_computed']

    def find_switch_points(list_x, list_t, seuil):
                seuil = float(seuil)
                liste_switch_points = []
                for i in range(1, len(list_x)):
                    x1 = list_x[i-1]
                    x2 = list_x[i]
                    t1 = list_t[i-1]
                    t2 = list_t[i]
                    if (float(x1) - float(seuil)) * (float(x2) - float(seuil)) < 0:
                        t0 = (t2 - t1) / (x2 - x1) * (seuil - x1) + t1
                        liste_switch_points.append(t0)
                return liste_switch_points

    def analyse_speed_prop_10(trace_test):
        
        if True:
                
            liste_switch_points_5 = find_switch_points(trace_test['list_v_instant'], trace_test['list_time_elapsed_cumul'], 5)
            liste_switch_points_10 = find_switch_points(trace_test['list_v_instant'], trace_test['list_time_elapsed_cumul'], 10)
            liste_switch_points_20 = find_switch_points(trace_test['list_v_instant'], trace_test['list_time_elapsed_cumul'], 20)
            liste_switch_points_30 = find_switch_points(trace_test['list_v_instant'], trace_test['list_time_elapsed_cumul'], 30)
            liste_switch_points_40 = find_switch_points(trace_test['list_v_instant'], trace_test['list_time_elapsed_cumul'], 40)
            liste_switch_points_50 = find_switch_points(trace_test['list_v_instant'], trace_test['list_time_elapsed_cumul'], 50)

            def time_above_x(liste_switch_points, list_t):
                tps_total = 0
                if len(liste_switch_points)%2 == 0:
                    for i in range(0, len(liste_switch_points), 2):
                        tps_itv = liste_switch_points[i + 1] - liste_switch_points[i]
                        tps_total += tps_itv

                else:
                    for i in range(0, len(liste_switch_points)-3, 2):
                        tps_itv = liste_switch_points[i + 1] - liste_switch_points[i]
                        tps_total += tps_itv
                    tps_total += list_t[-1]-liste_switch_points[-1]
                return tps_total
            col = {'walk':'lightcoral', 'bike':'lightblue', 'car':'lightgreen', 'bus':'lightpink'}

            list_t = trace_test['list_time_elapsed_cumul']
            tps_above_5 = time_above_x(liste_switch_points_5, list_t)
            tps_above_10 = time_above_x(liste_switch_points_10, list_t)
            tps_above_20 = time_above_x(liste_switch_points_20, list_t)
            tps_above_30 = time_above_x(liste_switch_points_30, list_t)
            tps_above_40 = time_above_x(liste_switch_points_40, list_t)
            tps_above_50 = time_above_x(liste_switch_points_50, list_t)

            tps_total = trace_test['list_time_elapsed_cumul'][-1] - trace_test['list_time_elapsed_cumul'][0]
            tps_0_5 = tps_total - tps_above_5
            tps_5_10 = tps_above_5 - tps_above_10
            tps_10_20 = tps_above_10 - tps_above_20
            tps_20_30 = tps_above_20 - tps_above_30
            tps_30_40 = tps_above_30 - tps_above_40
            tps_40_50 = tps_above_40 - tps_above_50
            tps_50 = tps_above_50

            prop_0_5, prop_5_10, prop_10_20, prop_20_30, prop_30_40, prop_40_50, prop_50 = tps_0_5/tps_total, tps_5_10/tps_total, tps_10_20/tps_total, tps_20_30/tps_total, tps_30_40/tps_total, tps_40_50/tps_total, tps_50/tps_total

        return(prop_0_5, prop_5_10, prop_10_20, prop_20_30, prop_30_40, prop_40_50, prop_50)


    liste_vmoy_fscore_bike = [-100000, 0.4105151041227126, 1.08700177789941, 3.0010020148977397, 3.676763741954413, 100000]
    liste_vmoy_fscore_walk = [-100000, 0.2799848679908142, 0.6137479298100725, 2.2943924935411273, 3.6191130058762884, 100000]
    liste_vmoy_fscore_car = [-100000, 3.0998198980542875, 5.320843845080464, 9.299098830321956, 10.844131096132337, 100000]
    liste_vmoy_fscore_bus = [-100000, 0.686809605902677, 1.6506549219792555, 3.9107762961126458, 5.62687494119651, 100000]
    liste_vmax_fscore_bike = [-100000, 10.754348907073089, 12.88238054187735, 18.444053878802357, 23.179953871428594, 100000]
    liste_vmax_fscore_walk = [-100000, 3.966656345267308, 4.800263925145154, 23.49551002742734, 37.11044663167593, 100000]
    liste_vmax_fscore_car = [-100000, 25.140868761081073, 36.0531888781126, 62.2208775038897, 67.38936552111863, 100000]
    liste_vmax_fscore_bus = [-100000, 11.314426448020688, 16.82439832785701, 31.26118608952548, 46.26993253993071, 100000]
    liste_vmedian_fscore_bike = [-100000, 3.7512221349662696, 7.272781328957448, 12.443036982073512, 14.118642937949335, 100000]
    liste_vmedian_fscore_walk = [-100000, 1.4189429035146086, 2.835524542059395, 5.227157025962203, 11.50872300219276, 100000]
    liste_vmedian_fscore_car = [-100000, 4.759093309441486, 8.842268453744229, 21.524034683334616, 29.614821050280344, 100000]
    liste_vmedian_fscore_bus = [-100000, 2.798782917202987, 4.405103986829141, 12.721424302728042, 16.419677652130545, 100000]
    liste_v95th_fscore_bike = [-100000, 10.364603243916969, 12.231276255122697, 16.645251869509575, 19.05927240962025, 100000]
    liste_v95th_fscore_walk = [-100000, 3.79196803179575, 4.574600519266477, 19.176914188889707, 29.20767566848131, 100000]
    liste_v95th_fscore_car = [-100000, 24.124619715829983, 33.58587484772206, 55.50664873265645, 59.89945629832348, 100000]
    liste_v95th_fscore_bus = [-100000, 9.98580631682779, 15.444884549017818, 26.358693070050357, 40.13670887064279, 100000]
    liste_length_fscore_bike = [-3000000, 1202.1530531957007, 2001.3344584397887, 4099.803351181057, 8797.354479525084, 3000000]
    liste_length_fscore_walk = [-3000000, 529.4410066754036, 743.4444911452108, 5983.928495867245, 11034.431686112946, 3000000]
    liste_length_fscore_car = [-3000000, 5729.860838691618, 10130.245444226382, 13701.275393358259, 14470.842217120684, 3000000]
    liste_length_fscore_bus = [-3000000, 2275.8263342141668, 3906.6943712744946, 9507.524411255992, 13060.900134875914, 3000000]
    liste_prop1_fscore_bike = [-100000, -38.55300338653714, -23.941383582323382, 0.4000951670950543, 0.7273085991138788, 100000]
    liste_prop1_fscore_walk = [-100000, -13.835449572346946, -0.06488636556079215, 1.0, 1.0, 100000]
    liste_prop1_fscore_car = [-100000, -18.64386556678606, -14.567283925318137, 0.06716514408244699, 0.27787040963564824, 100000]
    liste_prop1_fscore_bus = [-100000, -18.926561107950842, -9.81834485027646, 0.4875686673245967, 0.8423195489203461, 100000]
    liste_prop2_fscore_bike = [-100000, 0.038774419375185394, 0.14434043609951075, 1.7112205584820057, 29.4412739176404, 100000]
    liste_prop2_fscore_walk = [-100000, 0.0, 0.0, 0.26658642512338326, 6.785856444264282, 100000]
    liste_prop2_fscore_car = [-100000, 0.033448390945042504, 0.05120049462118153, 0.16127592806759183, 7.31133591950949, 100000]
    liste_prop2_fscore_bus = [-100000, 0.04015112335906942, 0.07548385843793279, 0.42110745417217144, 8.96552356130289, 100000]
    liste_prop3_fscore_bike = [-100000, 0.02780680111053917, 0.1463547730171308, 0.8442728343167547, 27.940325061995033, 100000]
    liste_prop3_fscore_walk = [-100000, 0.0, 0.0, 0.20709790977476378, 0.6188848805857856, 100000]
    liste_prop3_fscore_car = [-100000, 0.06637865763403636, 0.10449213357670438, 0.4211830081595799, 12.056333331333557, 100000]
    liste_prop3_fscore_bus = [-100000, 0.013153286983407739, 0.12158921749387544, 0.5831173137699325, 10.728594559000319, 100000]
    liste_prop4_fscore_bike = [-100000, 0.0, 0.0, 0.0, 0.016207726732776387, 100000]
    liste_prop4_fscore_walk = [-100000, 0.0, 0.0, 0.012580200858416404, 0.11340666992621616, 100000]
    liste_prop4_fscore_car = [-100000, 0.06447190756834761, 0.11653454155321566, 0.41998931588817057, 12.733399249840005, 100000]
    liste_prop4_fscore_bus = [-100000, 0.0, 0.0, 0.14344930792329944, 0.29842946197423914, 100000]
    liste_prop5_fscore_bike = [-100000, 0.0, 0.0, 0.0, 0.0, 100000]
    liste_prop5_fscore_walk = [-100000, 0.0, 0.0, 0.0, 0.02743294573550256, 100000]
    liste_prop5_fscore_car = [-100000, 0.0, 0.05913627793555022, 0.19754951646714392, 10.249626042793171, 100000]
    liste_prop5_fscore_bus = [-100000, 0.0, 0.0, 0.007023929398621709, 0.11532870030738979, 100000]

    traces['v_moy'] = traces['track_length'] / traces['duration']
    traces['list_time_elapsed_cumul'] = traces['list_time_elapsed'].apply(lambda x: list(np.cumsum(x)))
    from scipy.interpolate import interp1d
    y = [0, 0, 1, 1, 0, 0]
    finterpol_vmoy_bike = interp1d(liste_vmoy_fscore_bike,y,kind='linear')
    finterpol_vmoy_walk = interp1d(liste_vmoy_fscore_walk,y,kind='linear')
    finterpol_vmoy_car = interp1d(liste_vmoy_fscore_car,y,kind='linear')
    finterpol_vmoy_bus = interp1d(liste_vmoy_fscore_bus,y,kind='linear')

    finterpol_vmax_bike = interp1d(liste_vmax_fscore_bike,y,kind='linear')
    finterpol_vmax_walk = interp1d(liste_vmax_fscore_walk,y,kind='linear')
    finterpol_vmax_car = interp1d(liste_vmax_fscore_car,y,kind='linear')
    finterpol_vmax_bus = interp1d(liste_vmax_fscore_bus,y,kind='linear')

    finterpol_vmedian_bike = interp1d(liste_vmedian_fscore_bike,y,kind='linear')
    finterpol_vmedian_walk = interp1d(liste_vmedian_fscore_walk,y,kind='linear')
    finterpol_vmedian_car = interp1d(liste_vmedian_fscore_car,y,kind='linear')
    finterpol_vmedian_bus = interp1d(liste_vmedian_fscore_bus,y,kind='linear')

    finterpol_v95th_bike = interp1d(liste_v95th_fscore_bike,y,kind='linear')
    finterpol_v95th_walk = interp1d(liste_v95th_fscore_walk,y,kind='linear')
    finterpol_v95th_car = interp1d(liste_v95th_fscore_car,y,kind='linear')
    finterpol_v95th_bus = interp1d(liste_v95th_fscore_bus,y,kind='linear')

    finterpol_length_bike = interp1d(liste_length_fscore_bike,y,kind='linear')
    finterpol_length_walk = interp1d(liste_length_fscore_walk,y,kind='linear')
    finterpol_length_car = interp1d(liste_length_fscore_car,y,kind='linear')
    finterpol_length_bus = interp1d(liste_length_fscore_bus,y,kind='linear')

    finterpole_prop1_bike = interp1d(liste_prop1_fscore_bike,y,kind='linear')
    finterpole_prop1_walk = interp1d(liste_prop1_fscore_walk,y,kind='linear')
    finterpole_prop1_car = interp1d(liste_prop1_fscore_car,y,kind='linear')
    finterpole_prop1_bus = interp1d(liste_prop1_fscore_bus,y,kind='linear')

    finterpole_prop2_bike = interp1d(liste_prop2_fscore_bike,y,kind='linear')
    finterpole_prop2_walk = interp1d(liste_prop2_fscore_walk,y,kind='linear')
    finterpole_prop2_car = interp1d(liste_prop2_fscore_car,y,kind='linear')
    finterpole_prop2_bus = interp1d(liste_prop2_fscore_bus,y,kind='linear')

    finterpole_prop3_bike = interp1d(liste_prop3_fscore_bike,y,kind='linear')
    finterpole_prop3_walk = interp1d(liste_prop3_fscore_walk,y,kind='linear')
    finterpole_prop3_car = interp1d(liste_prop3_fscore_car,y,kind='linear')
    finterpole_prop3_bus = interp1d(liste_prop3_fscore_bus,y,kind='linear')

    finterpole_prop4_bike = interp1d(liste_prop4_fscore_bike,y,kind='linear')
    finterpole_prop4_walk = interp1d(liste_prop4_fscore_walk,y,kind='linear')
    finterpole_prop4_car = interp1d(liste_prop4_fscore_car,y,kind='linear')
    finterpole_prop4_bus = interp1d(liste_prop4_fscore_bus,y,kind='linear')

    finterpole_prop5_bike = interp1d(liste_prop5_fscore_bike,y,kind='linear')
    finterpole_prop5_walk = interp1d(liste_prop5_fscore_walk,y,kind='linear')
    finterpole_prop5_car = interp1d(liste_prop5_fscore_car,y,kind='linear')
    finterpole_prop5_bus = interp1d(liste_prop5_fscore_bus,y,kind='linear')



    traces['prop1'] = traces.apply(lambda x: analyse_speed_prop_10(x)[0],axis=1)
    traces['prop2'] = traces.apply(lambda x: analyse_speed_prop_10(x)[1],axis=1)
    traces['prop3'] = traces.apply(lambda x: analyse_speed_prop_10(x)[2],axis=1)
    traces['prop4'] = traces.apply(lambda x: analyse_speed_prop_10(x)[3],axis=1)
    traces['prop5'] = traces.apply(lambda x: analyse_speed_prop_10(x)[4],axis=1)
    traces['prop6'] = traces.apply(lambda x: analyse_speed_prop_10(x)[5],axis=1)

    traces['prop1'] = traces.apply(lambda x: analyse_speed_prop_10(x)[0],axis=1)
    traces['prop2'] = traces.apply(lambda x: analyse_speed_prop_10(x)[1],axis=1)
    traces['prop3'] = traces.apply(lambda x: analyse_speed_prop_10(x)[2],axis=1)
    traces['prop4'] = traces.apply(lambda x: analyse_speed_prop_10(x)[3],axis=1)
    traces['prop5'] = traces.apply(lambda x: analyse_speed_prop_10(x)[4],axis=1)
    traces['prop6'] = traces.apply(lambda x: analyse_speed_prop_10(x)[5],axis=1)

    traces['score_vmoy_bike'] = traces.apply(lambda x: finterpol_vmoy_bike(x['v_moy']),axis=1)
    traces['score_vmoy_walk'] = traces.apply(lambda x: finterpol_vmoy_walk(x['v_moy']),axis=1)
    traces['score_vmoy_car'] = traces.apply(lambda x: finterpol_vmoy_car(x['v_moy']),axis=1)
    traces['score_vmoy_bus'] = traces.apply(lambda x: finterpol_vmoy_bus(x['v_moy']),axis=1)

    traces['score_vmax_bike'] = traces.apply(lambda x: finterpol_vmax_bike(x['v_instant_max']),axis=1)
    traces['score_vmax_walk'] = traces.apply(lambda x: finterpol_vmax_walk(x['v_instant_max']),axis=1)
    traces['score_vmax_car'] = traces.apply(lambda x: finterpol_vmax_car(x['v_instant_max']),axis=1)
    traces['score_vmax_bus'] = traces.apply(lambda x: finterpol_vmax_bus(x['v_instant_max']),axis=1)

    traces['score_vmedian_bike'] = traces.apply(lambda x: finterpol_vmedian_bike(x['v_median']),axis=1)
    traces['score_vmedian_walk'] = traces.apply(lambda x: finterpol_vmedian_walk(x['v_median']),axis=1)
    traces['score_vmedian_car'] = traces.apply(lambda x: finterpol_vmedian_car(x['v_median']),axis=1)
    traces['score_vmedian_bus'] = traces.apply(lambda x: finterpol_vmedian_bus(x['v_median']),axis=1)

    traces['score_v95th_bike'] = traces.apply(lambda x: finterpol_v95th_bike(x['v_95th_percentile']),axis=1)
    traces['score_v95th_walk'] = traces.apply(lambda x: finterpol_v95th_walk(x['v_95th_percentile']),axis=1)
    traces['score_v95th_car'] = traces.apply(lambda x: finterpol_v95th_car(x['v_95th_percentile']),axis=1)
    traces['score_v95th_bus'] = traces.apply(lambda x: finterpol_v95th_bus(x['v_95th_percentile']),axis=1)

    traces['score_length_bike'] = traces.apply(lambda x: finterpol_length_bike(x['track_length']),axis=1)
    traces['score_length_walk'] = traces.apply(lambda x: finterpol_length_walk(x['track_length']),axis=1)
    traces['score_length_car'] = traces.apply(lambda x: finterpol_length_car(x['track_length']),axis=1)
    traces['score_length_bus'] = traces.apply(lambda x: finterpol_length_bus(x['track_length']),axis=1)

    traces['score_prop1_bike'] = traces.apply(lambda x: finterpole_prop1_bike(x['prop1']),axis=1)
    traces['score_prop1_walk'] = traces.apply(lambda x: finterpole_prop1_walk(x['prop1']),axis=1)
    traces['score_prop1_car'] = traces.apply(lambda x: finterpole_prop1_car(x['prop1']),axis=1)
    traces['score_prop1_bus'] = traces.apply(lambda x: finterpole_prop1_bus(x['prop1']),axis=1)

    traces['score_prop2_bike'] = traces.apply(lambda x: finterpole_prop2_bike(x['prop2']),axis=1)
    traces['score_prop2_walk'] = traces.apply(lambda x: finterpole_prop2_walk(x['prop2']),axis=1)
    traces['score_prop2_car'] = traces.apply(lambda x: finterpole_prop2_car(x['prop2']),axis=1)
    traces['score_prop2_bus'] = traces.apply(lambda x: finterpole_prop2_bus(x['prop2']),axis=1)

    traces['score_prop3_bike'] = traces.apply(lambda x: finterpole_prop3_bike(x['prop3']),axis=1)
    traces['score_prop3_walk'] = traces.apply(lambda x: finterpole_prop3_walk(x['prop3']),axis=1)
    traces['score_prop3_car'] = traces.apply(lambda x: finterpole_prop3_car(x['prop3']),axis=1)
    traces['score_prop3_bus'] = traces.apply(lambda x: finterpole_prop3_bus(x['prop3']),axis=1)

    traces['score_prop4_bike'] = traces.apply(lambda x: finterpole_prop4_bike(x['prop4']),axis=1)
    traces['score_prop4_walk'] = traces.apply(lambda x: finterpole_prop4_walk(x['prop4']),axis=1)
    traces['score_prop4_car'] = traces.apply(lambda x: finterpole_prop4_car(x['prop4']),axis=1)
    traces['score_prop4_bus'] = traces.apply(lambda x: finterpole_prop4_bus(x['prop4']),axis=1)

    traces['score_prop5_bike'] = traces.apply(lambda x: finterpole_prop5_bike(x['prop5']),axis=1)
    traces['score_prop5_walk'] = traces.apply(lambda x: finterpole_prop5_walk(x['prop5']),axis=1)
    traces['score_prop5_car'] = traces.apply(lambda x: finterpole_prop5_car(x['prop5']),axis=1)
    traces['score_prop5_bus'] = traces.apply(lambda x: finterpole_prop5_bus(x['prop5']),axis=1)


    # remplace les nan par 0 d'une manière pas propre mais qui marche (dropna ne marche pas etrangement)


    traces['score_vmoy_bike'] = traces['score_vmoy_bike'].apply(lambda x: pasnan(x))
    traces['score_vmoy_walk'] = traces['score_vmoy_walk'].apply(lambda x: pasnan(x))
    traces['score_vmoy_car'] = traces['score_vmoy_car'].apply(lambda x: pasnan(x))
    traces['score_vmoy_bus'] = traces['score_vmoy_bus'].apply(lambda x: pasnan(x))

    traces['score_vmax_bike'] = traces['score_vmax_bike'].apply(lambda x: pasnan(x))
    traces['score_vmax_walk'] = traces['score_vmax_walk'].apply(lambda x: pasnan(x))
    traces['score_vmax_car'] = traces['score_vmax_car'].apply(lambda x: pasnan(x))
    traces['score_vmax_bus'] = traces['score_vmax_bus'].apply(lambda x: pasnan(x))

    traces['score_vmedian_bike'] = traces['score_vmedian_bike'].apply(lambda x: pasnan(x))
    traces['score_vmedian_walk'] = traces['score_vmedian_walk'].apply(lambda x: pasnan(x))
    traces['score_vmedian_car'] = traces['score_vmedian_car'].apply(lambda x: pasnan(x))
    traces['score_vmedian_bus'] = traces['score_vmedian_bus'].apply(lambda x: pasnan(x))

    traces['score_v95th_bike'] = traces['score_v95th_bike'].apply(lambda x: pasnan(x))
    traces['score_v95th_walk'] = traces['score_v95th_walk'].apply(lambda x: pasnan(x))

    traces['score_v95th_car'] = traces['score_v95th_car'].apply(lambda x: pasnan(x))
    traces['score_v95th_bus'] = traces['score_v95th_bus'].apply(lambda x: pasnan(x))

    traces['score_length_bike'] = traces['score_length_bike'].apply(lambda x: pasnan(x))
    traces['score_length_walk'] = traces['score_length_walk'].apply(lambda x: pasnan(x))
    traces['score_length_car'] = traces['score_length_car'].apply(lambda x: pasnan(x))
    traces['score_length_bus'] = traces['score_length_bus'].apply(lambda x: pasnan(x))

    traces['score_prop1_bike'] = traces['score_prop1_bike'].apply(lambda x: pasnan(x))
    traces['score_prop1_walk'] = traces['score_prop1_walk'].apply(lambda x: pasnan(x))
    traces['score_prop1_car'] = traces['score_prop1_car'].apply(lambda x: pasnan(x))
    traces['score_prop1_bus'] = traces['score_prop1_bus'].apply(lambda x: pasnan(x))

    traces['score_prop2_bike'] = traces['score_prop2_bike'].apply(lambda x: pasnan(x))
    traces['score_prop2_walk'] = traces['score_prop2_walk'].apply(lambda x: pasnan(x))
    traces['score_prop2_car'] = traces['score_prop2_car'].apply(lambda x: pasnan(x))
    traces['score_prop2_bus'] = traces['score_prop2_bus'].apply(lambda x: pasnan(x))

    traces['score_prop3_bike'] = traces['score_prop3_bike'].apply(lambda x: pasnan(x))
    traces['score_prop3_walk'] = traces['score_prop3_walk'].apply(lambda x: pasnan(x))
    traces['score_prop3_car'] = traces['score_prop3_car'].apply(lambda x: pasnan(x))
    traces['score_prop3_bus'] = traces['score_prop3_bus'].apply(lambda x: pasnan(x))

    traces['score_prop4_bike'] = traces['score_prop4_bike'].apply(lambda x: pasnan(x))
    traces['score_prop4_walk'] = traces['score_prop4_walk'].apply(lambda x: pasnan(x))
    traces['score_prop4_car'] = traces['score_prop4_car'].apply(lambda x: pasnan(x))
    traces['score_prop4_bus'] = traces['score_prop4_bus'].apply(lambda x: pasnan(x))

    traces['score_prop5_bike'] = traces['score_prop5_bike'].apply(lambda x: pasnan(x))
    traces['score_prop5_walk'] = traces['score_prop5_walk'].apply(lambda x: pasnan(x))
    traces['score_prop5_car'] = traces['score_prop5_car'].apply(lambda x: pasnan(x))
    traces['score_prop5_bus'] = traces['score_prop5_bus'].apply(lambda x: pasnan(x))


    model_classif = pickle.load(open('../inputs/model_classif.pkl','rb'))
    traces['mode_computed_hybrid'] = model_classif.predict(traces[['score_vmoy_bike', 'score_vmoy_walk', 'score_vmoy_car', 'score_vmoy_bus', 'score_vmax_bike', 'score_vmax_walk', 'score_vmax_car', 'score_vmax_bus', 'score_vmedian_bike', 'score_vmedian_walk', 'score_vmedian_car', 'score_vmedian_bus', 'score_v95th_bike', 'score_v95th_walk', 'score_v95th_car', 'score_v95th_bus', 'score_length_bike', 'score_length_walk', 'score_length_car', 'score_length_bus', 'score_prop1_bike', 'score_prop1_walk', 'score_prop1_car', 'score_prop1_bus', 'score_prop2_bike', 'score_prop2_walk', 'score_prop2_car', 'score_prop2_bus', 'score_prop3_bike', 'score_prop3_walk','score_prop3_car','score_prop3_bus','score_prop4_bike','score_prop4_walk','score_prop4_car','score_prop4_bus','score_prop5_bike','score_prop5_walk','score_prop5_car','score_prop5_bus']])

    traces['mode_computed_hybrid'] = traces.apply(lambda x: motorizedornot(x), axis=1)
    traces['mode_computed_hybrid'] = traces.apply(lambda x: istrain(x), axis=1)

    return traces[['track_id', 'mode_computed_hybrid']]