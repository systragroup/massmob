import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import geopandas as gpd


def plot_hist_cumul(tracks, field,xmax,color,bins):
    ech_point_tracks = tracks
    liste_acc = np.array(ech_point_tracks[field])
    liste_acc.sort()

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax2 = ax1.twinx()

    ax1.plot(liste_acc,np.linspace(0, 1, len(liste_acc)),color=color)
    ax2.hist(liste_acc, bins = bins,color='grey', alpha=0.5)
    fig.legend(['Cumulative distribution','Histogram'])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    ax1.set_title(f'Cumulative distribution and histogram of{field}')
    ax1.set_xlabel(f'{field}')
    ax1.set_ylabel('Cumulative distribution')
    ax2.set_ylabel('Number of traces')
    plt.xlim(0,xmax)

def plot_attraction_zone(flows, zoning, specific_zone, tracks):
    nom_commune = specific_zone

    code_insee_commune = zoning[zoning['zone_id'] == nom_commune]['zone_id'].values[0]

    OD_emissions = flows[flows['commune_origin'] == code_insee_commune]
    OD_emissions.rename(columns={'counts':'destination_counts'}, inplace=True)
    OD_attractions = flows[flows['commune_destination'] == code_insee_commune]
    OD_attractions.rename(columns={'counts':'origin_counts'}, inplace=True)

    zoning_update = zoning.merge(OD_emissions[['commune_destination','destination_counts']], left_on='zone_id', right_on='commune_destination', how='left')
    zoning_update = zoning_update.merge(OD_attractions[['commune_origin','origin_counts']], left_on='zone_id', right_on='commune_origin', how='left')

    zoning_non_intra = zoning_update.loc[zoning_update.commune_destination != code_insee_commune]
    zoning_non_intra = zoning_update.loc[zoning_update.commune_origin!= code_insee_commune]
    zoning_non_intra['destination_counts'].fillna(0, inplace=True)
    zoning_non_intra['origin_counts'].fillna(0, inplace=True)

    fig, ax = plt.subplots(1) 
    zoning_non_intra.plot(ax=ax,column='origin_counts', legend=True)
    zoning_non_intra[zoning_non_intra['origin_counts']==0].plot(ax=ax, color='grey')
    ax.set_title(f'Origines des traces depuis {nom_commune}')

    return

def plot_emission_zone(flows, zoning, specific_zone, tracks):
    nom_commune = specific_zone

    code_insee_commune = zoning[zoning['zone_id'] == nom_commune]['zone_id'].values[0]
    OD_emissions = flows[flows['commune_origin'] == code_insee_commune]
    OD_emissions.rename(columns={'counts':'destination_counts'}, inplace=True)
    OD_attractions = flows[flows['commune_destination'] == code_insee_commune]
    OD_attractions.rename(columns={'counts':'origin_counts'}, inplace=True)
    zoning_update = zoning.merge(OD_emissions[['commune_destination','destination_counts']], left_on='zone_id', right_on='commune_destination', how='left')
    zoning_update = zoning_update.merge(OD_attractions[['commune_origin','origin_counts']], left_on='zone_id', right_on='commune_origin', how='left')
    zoning_non_intra = zoning_update.loc[zoning_update.commune_destination != code_insee_commune]
    zoning_non_intra = zoning_update.loc[zoning_update.commune_origin!= code_insee_commune]
    zoning_non_intra['destination_counts'].fillna(0, inplace=True)
    zoning_non_intra['origin_counts'].fillna(0, inplace=True)
    fig, ax = plt.subplots(1) 
    zoning_non_intra.plot(ax=ax,column='destination_counts', legend=True)
    zoning_non_intra[zoning_non_intra['destination_counts']==0].plot(ax=ax, color='grey')
    ax.set_title(f'Destination des traces depuis {nom_commune}')
    return

def plot_loaded_network(road_links,zoning,tracks):
    bbox = tracks.to_crs(4326).total_bounds
    zoning.to_crs(epsg=4326, inplace=True)
    fig,ax = plt.subplots()
    road_links.plot(ax=ax,legend=True, figsize=(20,20), linewidth=road_links['tracks_count']/300)
    zoning.plot(ax=ax, color='white', edgecolor='black')
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    # ax.legend('tracks_counts')
    fig.set_size_inches(10,10)
    ax.set_title('Loaded network')
    return

def plot_homeplace_map(phones,zoning,tracks):
    bbox = tracks.total_bounds
    fig, ax = plt.subplots()
    zoning.plot(ax=ax,column ='nbr_domicile' ,figsize=(10,10), legend=True, edgecolor='black', colormap='Greens')
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_title('Home location')
    ax.set_ylim(bbox[1], bbox[3])
    fig.set_size_inches(10,10)
    return

def plot_workplace_map(phones,zoning,tracks):
    bbox = tracks.total_bounds
    fig, ax = plt.subplots()
    zoning.plot(ax=ax,column ='nbr_emploi' ,figsize=(10,10), legend=True, edgecolor='black', colormap='Greens')
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_title('Home location')
    ax.set_ylim(bbox[1], bbox[3])
    fig.set_size_inches(10,10)
    return

def scatter_rep_rate_home(zones):
    assert 'repr_rate_home' in zones.columns, 'Representativity rates for homes not computed'
    
    zones_positif = zones[zones['repr_rate_home']>0]
    # sns.regplot(x='nbr_domicile', y='population_totale', data=zones_positif, fit_reg=False)
    # slope, intercept, r_value, p_value, std_err = stats.linregress(zones_positif['nbr_domicile'], zones_positif['population_totale'])
    # print(f"Equation de la droite: y = {slope}x + {intercept}")
    # print(f"R2: {r_value**2}")
    a = np.dot(zones_positif['nbr_domicile'], zones_positif['population_totale']) / np.dot(zones_positif['nbr_domicile'], zones_positif['nbr_domicile'])
    equation = f"Y = {a:.2f} * X"
    print("Équation de la droite de régression:", equation)
    Y_pred = a * zones_positif['nbr_domicile']
    r_squared = r2_score(zones_positif['population_totale'], Y_pred)
    r_squared_text = f"R² = {r_squared:.2f}"
    print("Coefficient de détermination R²:", r_squared)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(zones_positif['nbr_domicile'], zones_positif['population_totale'], color='blue', alpha=0.5)
    ax.plot(zones_positif['nbr_domicile'], Y_pred, color='red', linewidth=2, label=r_squared_text)
    return

def scatter_rep_rate_work(zones):
    assert 'repr_rate_work' in zones.columns, 'Representativity rates for works not computed'
    
    zones_positif = zones[zones['repr_rate_work']>0]
    # sns.regplot(x='nbr_emploi', y='emplois', data=zones_positif, fit_reg=False)
    # slope, intercept, r_value, p_value, std_err = stats.linregress(zones_positif['nbr_emploi'], zones_positif['emplois'])
    # print(f"Equation de la droite: y = {slope}x + {intercept}")
    # print(f"R2: {r_value**2}")
    a = np.dot(zones_positif['nbr_emploi'], zones_positif['emplois']) / np.dot(zones_positif['nbr_emploi'], zones_positif['nbr_emploi'])
    equation = f"Y = {a:.2f} * X"
    print("Équation de la droite de régression:", equation)
    Y_pred = a * zones_positif['nbr_emploi']
    r_squared = r2_score(zones_positif['emplois'], Y_pred)
    r_squared_text = f"R² = {r_squared:.2f}"
    print("Coefficient de détermination R²:", r_squared)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(zones_positif['nbr_emploi'], zones_positif['emplois'], color='blue', alpha=0.5)
    ax.plot(zones_positif['nbr_emploi'], Y_pred, color='red', linewidth=2, label=r_squared_text)
    return

def int_to_h_m(x):
    return round(x//3600), round((x%3600)/60)

def hist_plot(serie, unit, bins, range, color_bars='grey'):
    name = serie.name
    liste = np.array(serie.values)

    liste_sorted = np.sort(liste)

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax2 = ax1.twinx()

    ax1.plot(liste_sorted, np.linspace(0, 1, len(liste_sorted), endpoint=False))
    ax2.hist(liste, bins=bins, color=color_bars, alpha=0.5, label='_cumul')
    ax1.set_xlabel(name)
    ax2.set_ylabel('Number of tracks')
    ax1.set_ylabel('Cumulative distribution')
    mediane = round(np.median(liste),1)
    moyenne = round(np.mean(liste),1)
    if unit=='m':
        mediane = round(np.median(liste), 1)
        moyenne = round(np.mean(liste), 1)
        ax1.axvline(mediane, color='green', linestyle='dashed', linewidth=1, label=f'Median : {round(mediane/1000, 1)} km')
        ax1.axvline(moyenne, color='red', linestyle='dotted', linewidth=1, label=f'Mean : {round(moyenne/1000, 1)} km')
    elif unit=='s':
        mediane = round(np.median(liste),1)
        moyenne = round(np.mean(liste),1)
        ax1.axvline(mediane, color='green', linestyle='dashed', linewidth=1, label='Median : {}h{}min'.format(*int_to_h_m(mediane)))
        ax1.axvline(moyenne, color='red', linestyle='dotted', linewidth=1, label='Mean : {}h{}min'.format(*int_to_h_m(moyenne)))
    ax1.legend(loc='center right')
    ax2.legend()
    ax1.set_xlim(min(range), max(range))
    ax2.set_xlim(min(range), max(range))
    ax1.set_title(f'Distribution of {name} ({unit})')

    return fig, ax1, ax2


def gdf_serialize(gdf, ommited_columns=[]):
    for col in set(gdf.columns) - set(ommited_columns).union(set(['geometry'])):
        gdf[col] = gdf[col].apply(str)
    return gdf
    

def interactive_plot(fcd, phone_id=None, track_id=None, track_column='first_ts', point_column='ts', m=None):
    
    # track plot
    tracks = fcd.tracks.loc[fcd.tracks['phone_id'] == phone_id].copy()
    if track_id is not None:
        tracks = fcd.tracks.loc[fcd.tracks['track_id']==track_id]
    tracks = gdf_serialize(tracks, [track_column])
    m_tracks = tracks.explore(
        track_column, m=m,
        style_kwds={'weight': 5},
    )

    # point plot
    points = fcd.points.loc[fcd.points['phone_id'] == phone_id].copy()
    if track_id is not None:
        points = fcd.points.loc[fcd.points['track_id']==track_id]
    points = gpd.GeoDataFrame(points)
    points.set_crs(epsg=2154, inplace=True)
    points = gdf_serialize(points, [point_column])
    return points.explore(point_column, m=m_tracks, style_kwds={'weight': 5})


def base_plot(df, geographical_bounds=None, *args, **kwargs):
    """
    Create a basic plot of a GeoDataFrame with specific geographical bounds.
    """
    def _set_bandwidth_geographical_bounds(plot, xmin, ymin, xmax, ymax, offset=0.01):
        x_offset = (xmax - xmin) * offset
        y_offset = (ymax - ymin) * offset
        plot.set_xlim(xmin - x_offset, xmax + x_offset)
        plot.set_ylim(ymin - y_offset, ymax + y_offset)
    # Light geometry plot
    plot = gpd.GeoDataFrame(df).plot(linewidth=0.1, color='grey', *args, **kwargs)
    # Set bounds geographical bounds
    if geographical_bounds is not None:
        _set_bandwidth_geographical_bounds(plot, *geographical_bounds)
    return plot


def width_series(value_series, outer_average_width=5, max_value=None, method='linear'):
    """
    :param value_series: the pd.Series that contain the values
    :param outer_average_width: the average width of the width series to return
    :param max_value: value to use as the maximum when normalizing the series (to focus low values)
    :param method: linear or surface
    :return: width_series: pd.Series that contains the widths corresponding to the values
    :rtype: pd.Series
    """
    max_value = max_value if max_value else np.max(list(value_series.values))
    if method == 'linear':
        serie = value_series.apply(lambda x: x / max_value * outer_average_width)
    elif method == 'surface':
        serie = value_series.apply(lambda x: np.sqrt(x / max_value) * outer_average_width)
    return serie


def linewidth_from_data_units(linewidth, axis, reference='y'):
    """
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return linewidth * (length / value_range)


def bandwidth(
    gdf, value_column, power=1, legend=True, legend_values=None, legend_length=1 / 3,
    label_column=None, max_linewidth_meters=100, variable_width=True, line_offset=True, cmap='Spectral',
    geographical_bounds=None, label_kwargs={'size': 12}, vmin=None, vmax=None, *args, **kwargs
):

    # Can only plot valid LineString
    df = gdf[gdf.length > 0].sort_values(value_column).copy()
    df = df[df.geometry.type == 'LineString']


    plot = base_plot(df, geographical_bounds, *args, **kwargs)
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    divider = make_axes_locatable(plot)
    cax = divider.append_axes('right', size="2%", pad=0.05)
    
    # Plot values
    # Power scale
    df['power'] = np.power(df[value_column], power)
    power_max_value = np.power(vmax, power) if vmax else df['power'].max()
    # Linewidth
    df['geographical_width'] = width_series(df['power'], max_linewidth_meters, max_value=power_max_value)
    df['linewidth'] = linewidth_from_data_units(df['geographical_width'].values, plot)
    # Offset
    if line_offset:
        df['geometry'] = df.apply(
            lambda x: x['geometry'].parallel_offset(x['geographical_width'] * 0.5), 1
        )
        df = df[df.geometry.type == 'LineString']
        df = df[df.length > 0]  # offset can create empty LineString

    # Plot
    #return df
    df.plot(
        column=value_column, linewidth=df['linewidth'], ax=plot, legend=True, 
        vmin=vmin, vmax=vmax, cax=cax, cmap=cmap)
    plot.set_yticks([])
    plot.set_xticks([])
    return plot