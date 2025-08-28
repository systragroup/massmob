import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from tqdm import tqdm
import numpy as np
from massmob.engine import tracks

class IntegrityModel:

    def test_phone_ids_integrity(self):
        """
        Checks that all phone_id values present in points are referenced in the phone_id column itself (self-integrity).
        Returns True if OK, False otherwise.
        """
        # Get all unique phone_ids in the dataframe
        phone_ids = self.points['phone_id'].unique()
        # Filter rows where phone_id is in the list of unique phone_ids
        filtered_points = self.points.filter(
            pl.col('phone_id').is_in(phone_ids)
        )
        # Compare the number of original rows and filtered rows
        return len(self.points) == len(filtered_points)
    

    def integrity_test_all(self):
        """
        Runs all integrity test methods (starting with 'test_') and returns a dictionary of results.
        """
        results = {}
        # Loop through all attributes, find methods that start with 'test_'
        for attr in dir(self):
            if attr.startswith('test_') and callable(getattr(self, attr)):
                method = getattr(self, attr)
                try:
                    results[attr] = method()
                except Exception as e:
                    results[attr] = f"Error: {e}"
        return results
    
    def tracks_sensitivity_analysis(
            self,
            base_kwargs: dict = None,
            param_grid: dict = None,
            n_samples_per_param: int = 5,
            sample_frac: float = 0.01,
            phone_column: str = "phone_id",
            random_state: int = 42
        ) -> pl.DataFrame:
        """
        Performs a one-factor-at-a-time sensitivity analysis on the main parameters impacting tracks construction,
        using only a random sample of phones (default: 1%).

        For each parameter of interest, iterates over a defined range while keeping all other parameters at their default value,
        invokes the tracks building pipeline, and stores the resulting number of tracks (n_tracks) for each parameter setting.

        Args:
            base_kwargs (dict, optional): 
                Base/default keyword arguments for tracks construction. 
                If provided, will override default parameters.
            param_grid (dict, optional): 
                Custom grid of values for each parameter to test (param_name: np.array/list of values).
                If not provided, uses standard ranges for the 3 main parameters:
                - making_a_stop_seconds_delay: 3min to 3h (180 to 10800s)
                - stop_speed_threshold_kmh: 0.5 to 4
                - idling_phone_meters_distance: 10 to 300
            n_samples_per_param (int, optional): 
                Number of values to use per parameter if param_grid is not provided. Default is 5.
            sample_frac (float, optional):
                Fraction of phones to sample (default: 0.01, or 1%).
            phone_column (str, optional):
                Name of column identifying phones for sampling (default: 'phone_id').
            random_state (int, optional):
                Random seed for reproducibility.

        Returns:
            pl.DataFrame: Table of sensitivity analysis results (columns: param, value, n_tracks)
        """

        # Default construction parameters (can be overridden)
        default_kwargs = dict(
            max_seconds_delay_between_points = 60 * 60,
            stop_speed_threshold_kmh = 1,
            idling_phone_meters_distance = 200,
            making_a_stop_seconds_delay = 10 * 60,
            min_trip_duration_seconds = 60 * 2,
            min_trip_distance_meters = 200
        )
        # Override defaults if specified
        if base_kwargs:
            default_kwargs.update(base_kwargs)

        # Define parameter value ranges to test
        if param_grid:
            param_ranges = param_grid
        else:
            param_ranges = {
                "making_a_stop_seconds_delay": np.linspace(3*60, 3*60*60, n_samples_per_param, dtype=int),
                "stop_speed_threshold_kmh": np.linspace(0.5, 4.0, n_samples_per_param),
                "idling_phone_meters_distance": np.linspace(10, 300, n_samples_per_param, dtype=int),
            }

        # === Phone sampling block ===
        points_df = self.points  # Assume Polars DataFrame
        phone_ids = points_df.select(phone_column).unique().to_series().to_numpy()
        n_total = len(phone_ids)
        n_sample = max(1, int(sample_frac * n_total))
        rng = np.random.default_rng(seed=random_state)
        sampled_ids = rng.choice(phone_ids, size=n_sample, replace=False)
        print(f"Sampled {len(sampled_ids)} phones out of {n_total} ({sample_frac*100:.1f}%)")
        # Only retain sampled phones for the analysis (in polars)
        points_sampled = points_df.filter(pl.col(phone_column).is_in(sampled_ids))

        results = []
        total_tests = sum(len(vals) for vals in param_ranges.values())

        with tqdm(total=total_tests, desc="Sensitivity analysis") as pbar:
            for param, values in param_ranges.items():
                for val in values:
                    # Update parameters for this run, varying only one at a time
                    kwargs = default_kwargs.copy()
                    kwargs[param] = val
                    # Use only sampled points for this run
                    points_cpy = points_sampled.clone()
                    # Build tracked points with the current set of parameters
                    tracked_points = tracks.build_tracked_points(points_cpy, **kwargs)
                    # Construct tracks from the tracked points
                    tracks_result = tracks.tracks_from_points_with_stops(tracked_points)
                    # KPI: number of tracks generated
                    n_tracks = tracks_result.height
                    # Record the parameter, value, and KPI for analysis
                    results.append({
                        "param": param,
                        "value": val,
                        "n_tracks": n_tracks,
                    })
                    pbar.update(1)

        self.sensitivity_results = pd.DataFrame(results)

    def plot_sensitivity_bars(self, results=None, figsize=(5, 4), single_plot: bool = True):
        """
        Plots the sensitivity of each tested parameter ('param') 
        on the number of tracks ('n_tracks'), either as grouped subplots (default)
        or as separate individual plots.

        Parameters
        ----------
        results : pd.DataFrame or polars.DataFrame
            The results table with columns: 'param', 'value', 'n_tracks'.
        figsize : tuple, optional
            Size of each subplot (default: (5, 4)).
        single_plot : bool, optional
            If True (default), show all parameters as subplots in a single figure.
            If False, show each parameter as a separate plot.

        Returns
        -------
        matplotlib.figure.Figure or None
            The Matplotlib figure when single_plot=True, otherwise None.
        """
        if results is None:
            results = self.sensitivity_results
        if not isinstance(results, pd.DataFrame):
            results = results.to_pandas()
        params_tested = results['param'].unique()
        n_params = len(params_tested)
        color_list = sns.color_palette("Set2", n_colors=n_params)

        if single_plot:
            ncols = min(3, n_params)
            nrows = math.ceil(n_params / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
            # Flatten axes for easy iteration
            axes = axes.flatten() if n_params > 1 else [axes]
            for idx, param in enumerate(params_tested):
                df_param = results[results['param'] == param].sort_values("value")
                # Print parameter name and type for each subplot
                print(f'Parameter #{idx+1}: {param!r} (type: {type(param).__name__})')
                if not df_param.empty:
                    sns.barplot(
                        data=df_param,
                        x="value",
                        y="n_tracks",
                        color=color_list[idx],
                        ax=axes[idx],
                        width=0.2
                    )
                    axes[idx].set_title(f"Sensitivity: {param}")
                    axes[idx].set_xlabel(param)
                    axes[idx].set_ylabel("Number of tracks")
                    axes[idx].tick_params(axis='x', rotation=45)
                    # Format tick labels for better display
                    xticklabels = [format_tick_label(t.get_text()) for t in axes[idx].get_xticklabels()]
                    axes[idx].set_xticklabels(xticklabels)
                else:
                    axes[idx].set_visible(False)
            # Hide unused axes
            for j in range(idx+1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            plt.show()
            return fig
        else:
            for idx, param in enumerate(params_tested):
                df_param = results[results['param'] == param].sort_values("value")
                print(f'Parameter #{idx+1}: {param!r} (type: {type(param).__name__})')
                plt.figure(figsize=figsize)
                sns.barplot(
                    data=df_param,
                    x="value",
                    y="n_tracks",
                    color=color_list[idx],
                    width=0.2
                )
                plt.title(f"Sensitivity: {param}")
                plt.xlabel(param)
                plt.ylabel("Number of tracks")
                plt.tight_layout()
                plt.show()



def format_tick_label(val):
    try:
        v = float(val)
        # Affiche comme entier si la valeur est entière, sinon 2 décimales
        if v.is_integer():
            return str(int(v))
        else:
            return f"{v:.2f}"
    except Exception:
        return str(val)