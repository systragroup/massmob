import pandas as pd
import geopandas as gpd
import polars as pl
import copy
from shapely.geometry import box, Point, Polygon
import numpy as np
from typing import List
import time

def df_explode(df, column_to_explode):
    """
    Take a column with iterable elements, and flatten the iterable to one element
    per observation in the output table.
    Slow and therefore not adapted to huge df.

    :param df: A dataframe to explod
    :type df: pandas.DataFrame
    :param column_to_explode:
    :type column_to_explode: str
    :return: An exploded data frame
    :rtype: pandas.DataFrame
    """
    # Create a list of new observations
    new_observations = list()

    # Iterate through existing observations
    for row in df.to_dict(orient='records'):
        # Take out the exploding iterable
        explode_values = row[column_to_explode]
        del row[column_to_explode]
        # Create a new observation for every entry in the exploding iterable & add all of the other columns
        for explode_value in explode_values:
            # Deep copy existing observation
            new_observation = copy.deepcopy(row)
            # Add one (newly flattened) value from exploding iterable
            new_observation[column_to_explode] = explode_value
            # Add to the list of new observations
            new_observations.append(new_observation)
    # Create a DataFrame
    return_df = pd.DataFrame(new_observations)
    # Return
    return return_df


def add_geometry_coordinates(df, columns=['x_geometry', 'y_geometry']):
    df = df.copy()

    # if the geometry is not a point...
    centroids = df['geometry'].apply(lambda g: g.centroid)

    df[columns[0]] = centroids.apply(lambda g: g.coords[0][0])
    df[columns[1]] = centroids.apply(lambda g: g.coords[0][1])
    return df


def generate_centers(
    polygon: Polygon, 
    n: float
) -> List[Point]:
    """
    Génère jusqu'à n points répartis régulièrement à l'intérieur d'un polygone Shapely.
    """
    minx, miny, maxx, maxy = polygon.bounds
    # On fait une grille pas trop fine, pour être sûr d'avoir assez de points
    grid_size = int(np.ceil(np.sqrt(n)*1.5))
    x = np.linspace(minx, maxx, grid_size)
    y = np.linspace(miny, maxy, grid_size)
    xx, yy = np.meshgrid(x, y)
    pts = [Point(float(px), float(py)) for px, py in zip(xx.ravel(), yy.ravel())
           if polygon.contains(Point(float(px), float(py)))]
    if len(pts) < n:
        print(f"Attention : grille trop grossière ou polygone trop petit, seulement {len(pts)} points trouvés.")
    return pts[:n]

def find_bbox_inner(
    perimeter: Polygon, 
    center: Point, 
    tol: float = 1e-6, 
    max_iter: int = 50
) -> Polygon: 
    """
    Approxime le plus grand rectangle centré sur `center` et inclus dans `perimeter` par dichotomie sur largeur et hauteur
    
    Parameters
    ----------
    perimeter : Polygon Shapely
    center : Point Shapely (par ex. perimeter.centroid ou Point milieu du bbox)
    tol : float
        tolérance de convergence
    max_iter : int
        maximum d'itérations de la dichotomie
    """

    geom_perim = perimeter.geometry.union_all()
    xmin, ymin, xmax, ymax = geom_perim.bounds
    cx, cy = center.x, center.y

    # Maximum possible basé sur la bbox externe
    max_left = cx - xmin
    max_right = xmax - cx
    max_down = cy - ymin
    max_up = ymax - cy

    max_width = 2 * min(max_left, max_right)
    max_height = 2 * min(max_down, max_up)

    best_rect = None
    best_area = 0

    # Parcourir hauteur (dichotomie)
    h_low, h_high = 0, max_height
    for _ in range(max_iter):
        h_test = (h_low + h_high) / 2

        # Pour chaque hauteur testée, rechercher la largeur maximale possible (dichotomie)
        w_low, w_high = 0, max_width
        w_temp = 0
        rect_temp = None
        for _ in range(max_iter):
            w_test = (w_low + w_high) / 2
            rect = box(
                cx - w_test/2, cy - h_test/2,
                cx + w_test/2, cy + h_test/2
            )
            if geom_perim.contains(rect):
                w_temp = w_test
                rect_temp = rect
                w_low = w_test
            else:
                w_high = w_test
            if w_high - w_low < tol:
                break

        # Après recherche largeur, si le rectangle trouvé a une surface record, on garde
        if rect_temp is not None:
            area = rect_temp.area
            if area > best_area:
                best_area = area
                best_rect = rect_temp
            h_low = h_test  # On peut augmenter la hauteur
        else:
            h_high = h_test
        if h_high - h_low < tol:
            break

    return best_rect

def find_largest_inner_bbox(
    perimeter: Polygon,
    points: List[Point],
    tol: float = 1e-6,
    max_iter: int = 50
) -> Polygon: 
    """
    Retourne le plus grand rectangle (en surface) trouvé par dichotomie, centré autour de la liste points proposés.
    """
    best_rect = None
    best_area = 0
    for center_point in points:
        rect = find_bbox_inner(perimeter, center_point, tol=tol, max_iter=max_iter)
        if rect is not None:
            area = rect.area
            if area > best_area:
                best_area = area
                best_rect = rect
    return best_rect

def grow_rectangle_in_all_directions(
    perimeter: Polygon, 
    rect: Polygon, 
    step: float = 50
) -> Polygon:
    """
    Agrandit le rectangle trouvé à la fonction précédente dans les 4 directions
    """
    geom_perim = perimeter.geometry.union_all()
    minx, miny, maxx, maxy = rect.bounds
    directions = ["minx", "maxx", "miny", "maxy"]
    grew = True
    while grew:
        grew = False
        # Essai d’agrandissement sur chaque côté
        for side in directions:
            if side == "minx":
                minx_new = minx - step
                new_rect = box(minx_new, miny, maxx, maxy)
                if geom_perim.contains(new_rect):
                    minx = minx_new
                    grew = True
            elif side == "maxx":
                maxx_new = maxx + step
                new_rect = box(minx, miny, maxx_new, maxy)
                if geom_perim.contains(new_rect):
                    maxx = maxx_new
                    grew = True
            elif side == "miny":
                miny_new = miny - step
                new_rect = box(minx, miny_new, maxx, maxy)
                if geom_perim.contains(new_rect):
                    miny = miny_new
                    grew = True
            elif side == "maxy":
                maxy_new = maxy + step
                new_rect = box(minx, miny, maxx, maxy_new)
                if geom_perim.contains(new_rect):
                    maxy = maxy_new
                    grew = True
        # On refait le tour tant qu’au moins une direction a pu produire un agrandissement
    return box(minx, miny, maxx, maxy)

def build_bboxes(
    perimeter: gpd.GeoDataFrame,
    grid_n: float = 50,
    step_bbox_int: float=50,
    points_crs: str = 'EPSG:2154'
) -> List[gpd.GeoDataFrame]:
    
    # bbox_extern
    poly = perimeter.geometry.union_all()
    bbox_ext = box(*poly.bounds)

    # bbox_intern
    print('... building bbox_int ...')
    points = generate_centers(poly, n=grid_n)
    bbox_inner = find_largest_inner_bbox(perimeter, points)
    bbox_int = grow_rectangle_in_all_directions(perimeter, bbox_inner, step=step_bbox_int)

    # passage dans le bon crs
    bbox_ext = gpd.GeoDataFrame(geometry=[bbox_ext], crs=perimeter.crs).to_crs(points_crs).geometry.values[0]
    bbox_int = gpd.GeoDataFrame(geometry=[bbox_int], crs=perimeter.crs).to_crs(points_crs).geometry.values[0]

    return bbox_ext, bbox_int

def point_within_zoning(
    points: np.ndarray,  # array shape (n, 2)
    bbox_ext: Polygon,
    bbox_int: Polygon,
    perim: Polygon,
    points_crs: str='EPSG:2154',
) -> np.ndarray:
    """
    Pour chaque point, retourne True si le point est dans le périmètre, False sinon.
    Méthode optimisée :
        - Si point hors bbox_ext -> False immédiat
        - Si point dans bbox_int -> True immédiat
        - Sinon, test précis d'appartenance au polygone
    
    Parameters
    ----------
    points : np.ndarray of shape (n, 2)
        Tableau des coordonnées (x, y) des points à tester.
    bbox_ext : Polygon
        Bbox englobante (externe, pour exclusion rapide)
    bbox_int : Polygon
        Petite bbox interne (inclusion rapide)
    perimeter : Polygon
        Polygone cible
    
    Returns
    -------
    np.ndarray, dtype=bool
        Tableau de booléens : True si dans le périmètre, False sinon.
    """
    t0 = time.time()
    minx_ext, miny_ext, maxx_ext, maxy_ext = bbox_ext.bounds
    minx_int, miny_int, maxx_int, maxy_int = bbox_int.bounds
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    out = np.empty(len(xs), dtype="object")
    in_ext = (xs >= minx_ext) & (xs <= maxx_ext) & (ys >= miny_ext) & (ys <= maxy_ext)
    in_int = (xs >= minx_int) & (xs <= maxx_int) & (ys >= miny_int) & (ys <= maxy_int)

    out[~in_ext] = False
    out[in_int] = True

    n_ext = np.count_nonzero(~in_ext)
    n_int = np.count_nonzero(in_int)
    need_geo_test = in_ext & (~in_int)
    n_geo_test = np.count_nonzero(need_geo_test)
    
    print(f"[bbox exclusion] {n_ext} points hors bbox_ext (exclusion immédiate)")
    print(f"[bbox inclusion] {n_int} points dans bbox_int (inclusion immédiate)")
    print(f"[geo test] {n_geo_test} points à tester géométriquement")
    print(f"Temps bbox: {time.time()-t0:.3f} s")

    if np.any(need_geo_test):
        t1 = time.time()
        gdf_test = gpd.GeoDataFrame(geometry = [Point(x, y) for x, y, b in zip(xs, ys, need_geo_test) if b], crs=points_crs)
        in_poly = gdf_test.within(perim).to_numpy()
        ind = np.where(need_geo_test)[0]
        out[ind] = in_poly
        print(f"Temps test géométrique: {time.time()-t1:.3f} s")

    print(f"Temps total: {time.time()-t0:.3f} s")
    return out.astype(bool)


def point_in_bbox(
    points: pl.DataFrame,
    bbox: tuple
) -> pl.DataFrame:
    """
    Return True if point in bbox else False
    bbox is in crs (lat,lon) 4326
    """
    minx, miny, maxx, maxy = bbox
    points = points.with_columns([
        ((pl.col('latitude') <= maxy) & (pl.col('latitude') >= miny) & (pl.col('longitude') >= minx) & (pl.col('longitude') <= maxx))
        .alias("point_in_bbox")
    ])
    return points