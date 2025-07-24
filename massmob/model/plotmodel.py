from massmob.io import plot
import polars as pl
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

def polars_tracks_to_geodataframe(df: pl.DataFrame, crs="EPSG:2154"):
    """
    Transform a Polars DataFrame with a 'coordinates' column
    (list of dicts or structs with 'x', 'y') into a GeoPandas GeoDataFrame,
    using LineString geometry.
    """
    # Convert Polars DataFrame to Pandas DataFrame
    df = df.to_pandas()

    # Generate geometry from coordinates_list (as LineString)
    def coords_to_linestring(coords):
        # Accepts list of dict or list of Polars structs
        return LineString([(pt['x'], pt['y']) for pt in coords])

    # Apply the conversion to each row
    df['geometry'] = df['coordinates'].apply(coords_to_linestring)

    # Build the GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
    return gdf

def polars_points_to_geodataframe(df: pl.DataFrame, coord_x="x", coord_y="y", crs="EPSG:2154"):
    """
    Convert a Polars DataFrame with point coordinates to a GeoPandas GeoDataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame containing point coordinates.
    coord_x : str, default "x"
        Name of the column containing X coordinates (or longitude).
    coord_y : str, default "y"
        Name of the column containing Y coordinates (or latitude).
    crs : str, default "EPSG:3857"
        Coordinate reference system for the GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with Point geometries.
    """
    pdf = df.to_pandas()
    pdf["geometry"] = pdf.apply(lambda row: Point(row[coord_x], row[coord_y]), axis=1)
    gdf = gpd.GeoDataFrame(pdf, geometry="geometry", crs=crs)
    return gdf


class PlotModel():

    def explore_tracks(self, expr, force: bool = False, **kwargs):
        """
        Interactively explore selected tracks as a map.

        Applies a filter expression to the `tracks` Polars DataFrame, converts the resulting subset into
        a GeoDataFrame, and opens an interactive map visualization (using GeoPandas' .explore method).

        By default, limits the display to 1000 tracks to avoid performance issues.
        To override this and display more than 1000 tracks, set `force=True`.

        Parameters
        ----------
        expr : polars.Expr or Polars Series-like
            Boolean expression or mask to filter the tracks.
            For example: pl.col("average_speed") > 1.0
        force : bool, default False
            If True, disables the 1000 tracks display safety limit.
            If False, displays only the first 1000 tracks matching the filter.
        **kwargs : dict, optional
            Additional arguments passed to GeoDataFrame.explore() for customizing
            the map (such as color, column, cmap, tooltip, etc.).

        Returns
        -------
        folium.Map
            Interactive map object displaying the selected tracks.

        Examples
        --------
        >>> m = instance.explore_tracks(pl.col("track_id") == 4, color="average_speed")
        >>> m = instance.explore_tracks(pl.col("average_speed") > 1, force=True, column="track_id")
        >>> m.save("my_tracks_map.html")
        """
        filtered = self.tracks.filter(expr)
        if not force and filtered.height > 1000:
            filtered = filtered.head(1000)
        return polars_tracks_to_geodataframe(filtered).explore(**kwargs)
    
    def explore_points(self, expr, force: bool = False, coord_x="x", coord_y="y", **kwargs):
        """
        Interactively explore selected points as a map.

        Applies a filter expression to the `points` Polars DataFrame, converts the resulting subset into
        a GeoDataFrame (with Point geometry), and displays an interactive map (via GeoPandas' .explore).

        By default, the display is limited to 1000 points for performance reasons,
        unless `force=True` is specified.

        Parameters
        ----------
        expr : polars.Expr or Polars Series-like
            Boolean expression or mask to filter the points.
            For example: pl.col("accuracy_median") < 10
        force : bool, default False
            If True, disables the 1000 points display limit.
        coord_x : str, default "x"
            Name of the column containing X coordinates.
        coord_y : str, default "y"
            Name of the column containing Y coordinates.
        **kwargs : dict, optional
            Additional arguments passed to GeoDataFrame.explore() for customizing
            the map (such as color, column, cmap, tooltip, etc.).

        Returns
        -------
        folium.Map
            Interactive map object displaying the selected points.

        Examples
        --------
        >>> m = instance.explore_points(pl.col("accuracy_median") < 5, color="accuracy_median")
        >>> m = instance.explore_points(pl.lit(True), force=True, coord_x="longitude", coord_y="latitude")
        >>> m.save("my_points_map.html")
        """
        filtered = self.points.filter(expr)
        if not force and filtered.height > 10000:
            filtered = filtered.head(10000)
        return polars_points_to_geodataframe(filtered, coord_x=coord_x, coord_y=coord_y).explore(**kwargs)

    def plot_attraction_zone(self, specific_zone):
        plot.plot_attraction_zone(self.volumes, self.zones, specific_zone, self.tracks)
        return
    
    def plot_emission_zone(self, specific_zone):
        plot.plot_emission_zone(self.volumes, self.zones, specific_zone, self.tracks)
        return  
    
    def plot_loaded_network(self):
        plot.plot_loaded_network(self.road_links, self.zones, self.tracks)
        return
    
    def plot_homeplace_map(self):
        plot.plot_homeplace_map(self.phones, self.zones, self.tracks)
        return
    
    def plot_workplace_map(self):
        plot.plot_workplace_map(self.phones, self.zones, self.tracks)
        return

    def scatter_representativity_rate_home(self):
        plot.scatter_rep_rate_home(self.zones)
        return
    
    def scatter_representativity_rate_work(self):
        plot.scatter_rep_rate_work(self.zones)
        return
    
    def interactive_plot(self, **kwargs):
        return plot.interactive_plot(self, **kwargs)
    
    def plot_bandwidth(self, gdf, **kwargs):
        return plot.bandwidth(gdf, **kwargs)
