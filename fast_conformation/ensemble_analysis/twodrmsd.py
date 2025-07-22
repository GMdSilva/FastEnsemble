import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

from tqdm import tqdm

from fast_conformation.ensemble_analysis.analysis_utils import parabola
from fast_conformation.ensemble_analysis.rmsd import calculate_rmsd


TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

class TwodRMSD:
    """
    A class to perform 2D RMSD analysis on molecular dynamics simulations.

    Attributes:
    ----------
    prediction_dicts : dict
        A dictionary containing prediction data with associated MDAnalysis Universes.
    input_dict : dict
        A dictionary containing job-related metadata (jobname, analysis range, etc.).
    ref_gr : str or None, optional
        The reference structure file path for the first RMSD calculation (default is None).
    ref_alt : str or None, optional
        The reference structure file path for the second RMSD calculation (default is None).
    filtering_dict : dict
        A dictionary to store data related to the filtering of RMSD values.
    clustering_dict : dict
        A dictionary to store data related to the clustering of 2D RMSD values.
    widget : object
        A widget object to handle the plotting of the analysis results.

    Methods:
    -------
    calculate_2d_rmsd(trial):
        Calculate 2D RMSD for a given trial.
    fit_and_filter_data(rmsd_2d_data, n_stdevs):
        Fit a parabola to the 2D RMSD data and filter points based on the standard deviation threshold.
    show_filt_data(rmsd_2d_data):
        Plot the 2D RMSD data along with the fitted curve and filtered points.
    plot_filtering_data(rmsd_2d_data):
        Generate and save a plot of the filtered 2D RMSD data with the fitted curve.
    cluster_2d_data(rmsd_2d_data, n_clusters):
        Perform clustering on the filtered 2D RMSD data and store clustering results.
    plot_and_save_2d_data(output_path):
        Plot the clustered 2D RMSD data, save the plot, and return a DataFrame with the clustering information.
    get_2d_rmsd(rmsd_mode_df_path, n_stdevs, n_clusters, output_path):
        Execute the full 2D RMSD analysis for all trials, including fitting, filtering, clustering, and saving results.

    """

    def __init__(self, prediction_dicts, input_dict, widget, ref_gr=None, ref_alt=None):
        """
        Initialize the TwodRMSD class.

        Parameters:
        ----------
        prediction_dicts : dict
            A dictionary containing prediction data with associated MDAnalysis Universes.
        input_dict : dict
            A dictionary containing job-related metadata (jobname, analysis range, etc.).
        widget : object
            A widget object to handle the plotting of the analysis results.
        ref_gr : str or None, optional
            The reference structure file path for the first RMSD calculation (default is None).
        ref_alt : str or None, optional
            The reference structure file path for the second RMSD calculation (default is None).
        """
        self.prediction_dicts = prediction_dicts
        self.input_dict = input_dict
        self.ref_gr = ref_gr
        self.ref_alt = ref_alt
        self.filtering_dict = {}
        self.clustering_dict = {}
        self.widget = widget

    def calculate_2d_rmsd(self, trial):
        """
        Calculate 2D RMSD for a given trial.

        Parameters:
        ----------
        trial : str
            The identifier for the trial being analyzed.

        Returns:
        -------
        rmsd_2d_data : np.ndarray
            A 2D array of RMSD values against two reference structures.
        """
        universe = self.prediction_dicts[trial]['mda_universe']
        rmsd_gr = calculate_rmsd(universe,
                                 ref=self.ref_gr,
                                 align_range=self.input_dict['align_range'],
                                 analysis_range=self.input_dict['analysis_range'])

        rmsd_alt = calculate_rmsd(universe,
                                  ref=self.ref_alt,
                                  align_range=self.input_dict['align_range'],
                                  analysis_range=self.input_dict['analysis_range'])
        rmsd_2d_data = np.array([rmsd_gr[self.input_dict['analysis_range']],
                                 rmsd_alt[self.input_dict['analysis_range']]]).T

        return rmsd_2d_data

    def fit_and_filter_data(self, rmsd_2d_data, n_stdevs):
        """
        Fit a parabola to the 2D RMSD data and filter points based on the standard deviation threshold.

        Parameters:
        ----------
        rmsd_2d_data : np.ndarray
            A 2D array of RMSD values.
        n_stdevs : int
            Number of standard deviations to use for filtering the data.

        Returns:
        -------
        None
        """
        popt, _ = curve_fit(parabola, rmsd_2d_data[:, 0], rmsd_2d_data[:, 1])
        fit_x = np.linspace(min(rmsd_2d_data[:, 0]), max(rmsd_2d_data[:, 0]), 100)
        fit_y = parabola(fit_x, *popt)

        fitted_curve_values = parabola(rmsd_2d_data[:, 0], *popt)
        distances = np.abs(rmsd_2d_data[:, 1] - fitted_curve_values)

        threshold = np.mean(distances) + int(n_stdevs) * np.std(distances)

        close_points = distances < threshold
        x_close = rmsd_2d_data[close_points, 0]
        y_close = rmsd_2d_data[close_points, 1]

        bin_edges = np.linspace(min(x_close), max(x_close), 101)
        bins = np.digitize(x_close, bin_edges)

        unique_bins = np.unique(bins)
        ratio = len(unique_bins) / 100

        self.filtering_dict = {
            'fit_x': fit_x,
            'fit_y': fit_y,
            'close_points': close_points,
            'bins': bins,
            'unique_bins': unique_bins,
            'ratio': ratio,
            'bin_edges': bin_edges,
            'x_close': x_close,
            'y_close': y_close,
        }

    def show_filt_data(self, rmsd_2d_data):
        """
        Plot the 2D RMSD data along with the fitted curve and filtered points.

        Parameters:
        ----------
        rmsd_2d_data : np.ndarray
            A 2D array of RMSD values.

        Returns:
        -------
        None
        """
        title = (f"{self.input_dict['jobname']} "
                 f"{self.input_dict['max_seq']} "
                 f"{self.input_dict['extra_seq']}")
                 
        plotter = self.widget.add_plot(rmsd_2d_data[:, 0], rmsd_2d_data[:, 1], title=title,
                                       x_label='RMSD vs. Ref1 (Å)', y_label='RMSD vs. Ref2 (Å)', scatter=True)
        self.widget.add_line(plotter, self.filtering_dict['fit_x'], self.filtering_dict['fit_y'],
                             label='Fitted Curve', color='r')
        self.widget.add_scatter(plotter, self.filtering_dict['x_close'],
                                self.filtering_dict['y_close'],
                                label='Close Points',
                                color=[68, 1, 84, 255])

    def plot_filtering_data(self, rmsd_2d_data):
        """
        Generate and save a plot of the filtered 2D RMSD data with the fitted curve.

        Parameters:
        ----------
        rmsd_2d_data : np.ndarray
            A 2D array of RMSD values.

        Returns:
        -------
        None
        """
        plt.figure(figsize=(5, 4))
        plt.scatter(rmsd_2d_data[:, 0], rmsd_2d_data[:, 1], s=10)

        plt.plot(self.filtering_dict['fit_x'],
                 self.filtering_dict['fit_y'],
                 label='Fitted Curve', color='red')

        plt.scatter(self.filtering_dict['x_close'],
                    self.filtering_dict['y_close'],
                    label='Close Points',
                    color='green',
                    s=20)

        title = (f"{self.input_dict['jobname']} "
                 f"{self.input_dict['max_seq']} "
                 f"{self.input_dict['extra_seq']}")

        plt.title(title, fontsize=15)
        plt.xlabel('RMSD vs. Ref1 (Å)', fontsize=14)
        plt.ylabel('RMSD vs. Ref2 (Å)', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(loc='best')
        plt.tight_layout()

        plot_path = (f"{self.input_dict['output_path']}/"
                     f"{self.input_dict['jobname']}/"
                     f"analysis/"
                     f"rmsd_2d/"
                     f"{self.input_dict['jobname']}_"
                     f"{self.input_dict['max_seq']}_"
                     f"{self.input_dict['extra_seq']}_"
                     f"rmsd_2d_fit.png")

        plt.savefig(plot_path, dpi=300)
        plt.close()

    def cluster_2d_data(self, rmsd_2d_data, n_clusters):
        """
        Perform clustering on the filtered 2D RMSD data and store clustering results.

        Parameters:
        ----------
        rmsd_2d_data : np.ndarray
            A 2D array of RMSD values.
        n_clusters : int
            Number of clusters to form.

        Returns:
        -------
        None
        """
        kmeans = KMeans(n_clusters)
        close_points_2d = np.array([self.filtering_dict['x_close'],
                                    self.filtering_dict['y_close']]).T

        kmeans.fit(close_points_2d)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        sorted_indices = np.argsort(centroids[:, 0])
        correct_labels = np.zeros_like(labels)

        for correct_label, original_index in enumerate(sorted_indices):
            correct_labels[labels == original_index] = correct_label

        unique_labels = set(correct_labels)
        cluster_counts = {i: 0 for i in unique_labels}
        total_samples = close_points_2d.shape[0]
        outliers = rmsd_2d_data.shape[0] - total_samples
        outliers = (outliers / total_samples) * 100

        for k in unique_labels:
            class_member_mask = (correct_labels == k)
            xy = close_points_2d[class_member_mask]
            cluster_counts[k] = round((len(xy) / total_samples) * 100, 1)
        self.clustering_dict = {
            'labels': labels,
            'correct_labels': correct_labels,
            'centroids': centroids,
            'cluster_counts': cluster_counts,
            'outliers': outliers,
            'unique_labels': unique_labels,
            'close_points_2d': close_points_2d
        }

    def plot_and_save_2d_data(self, output_path):
        """
        Plot the clustered 2D RMSD data, save the plot, and return a DataFrame with the clustering information.

        Parameters:
        ----------
        output_path : str
            The path where the plot will be saved.

        Returns:
        -------
        df : pd.DataFrame
            A DataFrame containing the clustering information.
        """
        unique_labels = self.clustering_dict['unique_labels']
        correct_labels = self.clustering_dict['correct_labels']
        cluster_counts = self.clustering_dict['cluster_counts']
        centroids = self.clustering_dict['centroids']
        outliers = self.clustering_dict['outliers']
        title = (f"{self.input_dict['jobname']} "
                 f"{self.input_dict['max_seq']} "
                 f"{self.input_dict['extra_seq']} "
                 f"Score: {self.filtering_dict['ratio']:.2f}")
        colors = np.array([[68, 1, 84, 255], [58, 82, 139, 255], [32, 144, 140, 255], [94, 201, 97, 255], [253, 231, 37, 255]])
        if self.widget:
            plotter = self.widget.add_plot(centroids[:, 0], centroids[:, 1], title=title, x_label='RMSD vs. Ref1 (Å)', y_label='RMSD vs. Ref2 (Å)', label='Centroids', scatter=True)
            self.widget.add_line(plotter, self.filtering_dict['fit_x'], self.filtering_dict['fit_y'], label='Fitted Curve', color='r')
        
            for i in unique_labels:
                cluster_points = self.clustering_dict['close_points_2d'][correct_labels == i]
                self.widget.add_scatter(plotter, cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i} pop: {cluster_counts[i]}')
    
        if output_path:
            plt.figure(figsize=(5, 4))

            for i in unique_labels:
                cluster_points = self.clustering_dict['close_points_2d'][correct_labels == i]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                            label=f'Cluster {i} pop: {cluster_counts[i]}', alpha=0.6)
            plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='black', marker='X', label='Centroids')
            plt.plot(self.filtering_dict['fit_x'], self.filtering_dict['fit_y'], label='Fitted Curve', color='red')


            plt.title(title, fontsize=16)
            plt.legend()
            plt.xlabel('RMSD vs. Ref1 (Å)', fontsize=14)
            plt.ylabel('RMSD vs. Ref2 (Å)', fontsize=14)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tight_layout()

            plot_path = (f"{self.input_dict['output_path']}/"
                        f"{self.input_dict['jobname']}/"
                        f"analysis/"
                        f"rmsd_2d/"
                        f"{self.input_dict['jobname']}_"
                        f"{self.input_dict['max_seq']}_"
                        f"{self.input_dict['extra_seq']}_"
                        f"rmsd_2d_clustered.png")

            plt.savefig(plot_path, dpi=300)
            plt.close()

        records = []

        for i in unique_labels:
            records.append({
                'trial': self.input_dict['trial'],
                'analysis_range': self.input_dict['analysis_range'],
                'score': self.filtering_dict['ratio'],
                'cluster_label': i,
                'cluster_pop': cluster_counts[i],
                'centroid_values': centroids[i],
                '%_outliers': outliers
            })

        df = pd.DataFrame(records)
        return df

    def get_2d_rmsd(self, rmsd_mode_df_path, n_stdevs, n_clusters, output_path):
        """
        Execute the full 2D RMSD analysis for all trials, including fitting, filtering, clustering, and saving results.

        Parameters:
        ----------
        rmsd_mode_df_path : str
            The path to the RMSD mode data file.
        n_stdevs : int
            Number of standard deviations to use for filtering the data.
        n_clusters : int
            Number of clusters to form.
        output_path : str
            The path where the results will be saved.

        Returns:
        -------
        None
        """
        df = pd.read_csv(rmsd_mode_df_path)
        unique_trials = df['trial'].unique()

        df_all_trials = pd.DataFrame()
        with tqdm(total=len(self.prediction_dicts), bar_format=TQDM_BAR_FORMAT) as pbar:
            for trial in unique_trials:
                if not n_clusters:
                    unique_df = df[df['trial'] == trial]
                    n_clusters_trial = len(unique_df['mode_label']) + 1
                else:
                    n_clusters_trial = n_clusters
                pbar.set_description(f'Running 2D RMSD analysis for {trial}')
                self.input_dict['trial'] = trial
                self.input_dict['max_seq'] = self.prediction_dicts[trial]['max_seq']
                self.input_dict['extra_seq'] = self.prediction_dicts[trial]['extra_seq']

                rmsd_2d_data = self.calculate_2d_rmsd(trial)
                if len(rmsd_2d_data) > 0:
                    self.fit_and_filter_data(rmsd_2d_data, n_stdevs)
                    if output_path:
                        self.plot_filtering_data(rmsd_2d_data)
                    if self.widget:
                        self.show_filt_data(rmsd_2d_data)
                    self.cluster_2d_data(rmsd_2d_data, n_clusters_trial)
                    df_to_save = self.plot_and_save_2d_data(output_path)
                    df_all_trials = pd.concat([df_all_trials, df_to_save], ignore_index=True)
                pbar.update(n=1)

        csv_path = (f"{self.input_dict['output_path']}/"
                    f"{self.input_dict['jobname']}/"
                    f"analysis/"
                    f"rmsd_2d/"
                    f"{self.input_dict['jobname']}_"
                    f"clustering_analysis.csv")

        print(f"\nSaving {self.input_dict['jobname']} 2D RMSD analysis results to {csv_path}\n")
        df_all_trials.to_csv(csv_path, index=False)
