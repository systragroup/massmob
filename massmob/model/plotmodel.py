from massmob.io import plot


class PlotModel():

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
