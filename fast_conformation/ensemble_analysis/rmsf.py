import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from scipy.signal import find_peaks

from MDAnalysis.analysis import rms, align
import pyqtgraph as pg
from tqdm import tqdm


TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

def calculate_rmsf_and_call_peaks(jobname, prediction_dicts, align_range, output_path, peak_width, prominence, threshold, widget):
    """
    Calculate Root Mean Square Fluctuation (RMSF) and detect peaks for multiple molecular dynamics predictions.

    Parameters:
    jobname (str): The name of the job or analysis.
    prediction_dicts (dict): A dictionary containing prediction data with associated MDAnalysis Universes.
    align_range (str): Atom selection string for alignment of trajectories (MDAnalysis selection syntax).
    output_path (str): Directory where the analysis results and plots will be saved.
    peak_width (int): Minimum width of peaks in the RMSF data to be considered.
    prominence (float): Prominence required for a peak in the RMSF data.
    threshold (float): Threshold for peak detection based on the standard deviation of RMSF values.
    widget (object): A widget object for displaying plots interactively.

    Returns:
    dict: The updated prediction_dicts with detected peaks added to each prediction's data.
    """
    with tqdm(total=len(prediction_dicts), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
        for result in prediction_dicts:
            pbar.set_description(f'Running RMSF peak analysis for {result}')
            u = prediction_dicts[result]['mda_universe']
            max_seq = prediction_dicts[result]['max_seq']
            extra_seq = prediction_dicts[result]['extra_seq']

            average = align.AverageStructure(u, u, select=align_range, ref_frame=0).run()
            ref = average.results.universe
            align.AlignTraj(u, ref, select=align_range, in_memory=True).run()
            atom_sel = u.select_atoms(align_range)
            r = rms.RMSF(atom_sel).run()

            rmsf_values = r.results.rmsf
            resids = atom_sel.resids

            mean_rmsf = np.mean(rmsf_values)
            std_rmsf = np.std(rmsf_values)
            threshold = mean_rmsf + threshold * std_rmsf

            # Detect peaks
            peaks, properties = find_peaks(rmsf_values, width=peak_width, prominence=prominence)

            # Plot RMSF with detected peaks
            x_label = 'Residue Number'
            y_label = 'RMSF (Å)'
            title = f'{jobname} {max_seq} {extra_seq}'

            plt.figure(figsize=(8, 3))
            plt.title(title, fontsize=16)
            plt.xlabel(x_label, fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            plt.tick_params(axis='both', which='major', labelsize=12)

            plt.plot(resids, rmsf_values)
            plt.plot(resids[peaks], rmsf_values[peaks], "x", color="C1")
            plt.vlines(x=resids[peaks], ymin=rmsf_values[peaks] - properties["prominences"],
                       ymax=rmsf_values[peaks], color="C1")
            plt.hlines(y=properties["width_heights"], xmin=resids[properties["left_ips"].astype(int)],
                       xmax=resids[properties["right_ips"].astype(int)], color="C1")

            plt.tight_layout()

            full_output_path = (f"{output_path}/"
                                f"{jobname}/"
                                f"analysis/"
                                f"mobile_detection/"
                                f"{jobname}_"
                                f"{max_seq}_"
                                f"{extra_seq}_"
                                f"rmsf_peaks.png")

            plt.savefig(full_output_path, dpi=300)

            # Extract and save detected peaks information
            detected_peaks = {}
            peak_counter = 0
            for i, peak in enumerate(peaks):
                left_ip = int(properties["left_ips"][i])
                right_ip = int(properties["right_ips"][i])
                peak_resids = resids[left_ip:right_ip + 1]

                peak_prop = {'starting_residue': peak_resids[0],
                             'ending_residue': peak_resids[-1],
                             'length': len(peak_resids),
                             'peak_value': rmsf_values[peak],
                             'prominence': properties["prominences"][i],
                             'width_height': properties["width_heights"][i]}

                peak_counter += 1
                detected_peaks[f'detected_peak_{peak_counter}'] = peak_prop

            prediction_dicts[result]['detected_peaks'] = detected_peaks
            pbar.update(n=1)

    return prediction_dicts


def calculate_rmsf_multiple(jobname, prediction_dicts, align_range, output_path, widget):
    """
    Calculate Root Mean Square Fluctuation (RMSF) for multiple molecular dynamics predictions and plot the results.

    Parameters:
    jobname (str): The name of the job or analysis.
    prediction_dicts (dict): A dictionary containing prediction data with associated MDAnalysis Universes.
    align_range (str): Atom selection string for alignment of trajectories (MDAnalysis selection syntax).
    output_path (str): Directory where the analysis results and plots will be saved.
    widget (object): A widget object for displaying plots interactively.

    Returns:
    None
    """
    labels = []
    x_label = 'Residue number'
    y_label = 'RMSF (Å)'
    title = f'{jobname}'
    
    if output_path:
        plt.figure(figsize=(8, 3))
        plt.title(title, fontsize=16)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)

    colors = ['blue', 'green', 'magenta', 'orange', 'grey', 'brown', 'cyan', 'purple']

    with tqdm(total=len(prediction_dicts), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
        plotter = None
        for idx, (result, data) in enumerate(prediction_dicts.items()):
            pbar.set_description(f'Measuring RMSF for {result}')
            u = data['mda_universe']
            average = align.AverageStructure(u, u, select=align_range, ref_frame=0).run()
            ref = average.results.universe

            align.AlignTraj(u, ref, select=align_range, in_memory=True).run()

            atom_sel = u.select_atoms(align_range)
            r = rms.RMSF(atom_sel).run()

            resids = atom_sel.resids

            if output_path:
                plt.plot(resids, r.results.rmsf, color=colors[idx % len(colors)], label=result)
            if widget:
                if plotter is None:
                    plotter = widget.add_plot(resids, r.results.rmsf, color=colors[idx % len(colors)], label=result, title=title, x_label=x_label, y_label=y_label)
                else:
                    widget.add_line(plotter, resids, r.results.rmsf, color=colors[idx % len(colors)], label=result)
            labels.append(result)
            pbar.update(n=1)
    
    if output_path:
        plt.legend()
        plt.tight_layout()

        full_output_path = (f"{output_path}/"
                            f"{jobname}/"
                            f"analysis/"
                            f"rmsf_plddt/"
                            f"{jobname}_rmsf_all.png")

        plt.savefig(full_output_path, dpi=300)
        plt.close()


def plot_plddt_rmsf_corr(jobname, prediction_dicts, plddt_dict, output_path, widget):
    """
    Plot the correlation between pLDDT scores and RMSF values for each prediction.

    Parameters:
    jobname (str): The name of the job or analysis.
    prediction_dicts (dict): A dictionary containing prediction data with associated MDAnalysis Universes.
    plddt_dict (dict): A dictionary containing pLDDT scores for each prediction.
    output_path (str): Directory where the analysis results and plots will be saved.
    widget (object): A widget object for displaying plots interactively.

    Returns:
    None
    """
    with tqdm(total=len(prediction_dicts), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
        plot_item = None

        for result in prediction_dicts:
            pbar.set_description(f'Running pLDDT/RMSF Correlation Analysis for {result}')
            max_seq = prediction_dicts[result]['max_seq']
            extra_seq = prediction_dicts[result]['extra_seq']
            plddt_data = plddt_dict[result]['all_plddts']

            arrays = np.array(plddt_data)
            plddt_avg = np.mean(arrays, axis=0)

            u = prediction_dicts[result]['mda_universe']
            average = align.AverageStructure(u, u, select='name CA', ref_frame=0).run()
            ref = average.results.universe
            align.AlignTraj(u, ref, select='name CA', in_memory=True).run()

            atom_sel = u.select_atoms('name CA')
            r = rms.RMSF(atom_sel).run()

            rmsf_values = r.results.rmsf
            resids = atom_sel.resids

            norm = Normalize(vmin=resids.min(), vmax=resids.max())
            cmap = plt.get_cmap('viridis')

            title = f'{jobname} {max_seq} {extra_seq}'
            x_label = 'C-Alpha RMSF (A)'
            y_label = 'Average pLDDT'
            if widget:
                plot_item = widget.add_plot(rmsf_values, plddt_avg, title=title, x_label=x_label, y_label=y_label, resids=resids, scatter=True, colorbar=True)

            if output_path:
                plt.scatter(rmsf_values, plddt_avg, c=[cmap(norm(resid)) for resid in resids])
                plt.title(title, fontsize=16)
                plt.xlabel(x_label, fontsize=14)
                plt.ylabel(y_label, fontsize=14)
                plt.tick_params(axis='both', which='major', labelsize=12)
                plt.tick_params(axis='both', which='minor', labelsize=12)
                plt.tight_layout()
                plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), label='Residue #')

                full_output_path = (f"{output_path}/"
                                    f"{jobname}/"
                                    f"analysis/"
                                    f"rmsf_plddt/"
                                    f"{jobname}_"
                                    f"{max_seq}_"
                                    f"{extra_seq}_"
                                    f"plddt_rmsf_corr.png")

                plt.savefig(full_output_path, dpi=300)
                plt.close()

            pbar.update(n=1)


def plot_plddt_line(jobname, plddt_dict, output_path, custom_start_residue, widget):
    """
    Plot pLDDT scores across residues for multiple predictions.

    Parameters:
    jobname (str): The name of the job or analysis.
    plddt_dict (dict): A dictionary containing pLDDT scores for each prediction.
    output_path (str): Directory where the analysis results and plots will be saved.
    custom_start_residue (int): A custom starting residue number for plotting.
    widget (object): A widget object for displaying plots interactively.

    Returns:
    None
    """
    plotter = None
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'grey', 'brown', 'cyan', 'magenta']
    x_label = 'Residue number'
    y_label = 'pLDDT'
    title = f'{jobname}'

    if output_path:
        plt.figure(figsize=(8, 3))
        plt.title(title, fontsize=16)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)

    for idx, result in enumerate(plddt_dict):
        plddt_data = plddt_dict[result]['all_plddts']
        arrays = np.array(plddt_data)
        plddt_avg = np.mean(arrays, axis=0)
        
        length_avg = plddt_avg.shape[0] if np.ndim(arrays) != 1 else arrays.shape[0]
        residue_numbers = np.arange(length_avg)
        if custom_start_residue is not None:
            residue_numbers += custom_start_residue
        if output_path:
            plt.plot(residue_numbers, plddt_avg, color=colors[idx % len(colors)], label=result)
        if plotter is None and widget:
            plotter = widget.add_plot(residue_numbers, plddt_avg, color=colors[idx % len(colors)], label=result, title=title, x_label=x_label, y_label=y_label)
        elif widget:
            widget.add_line(plotter, residue_numbers, plddt_avg, color=colors[idx % len(colors)], label=result)

    if output_path:
        plt.legend()
        plt.tight_layout()
        full_output_path = (f"{output_path}/"
                            f"{jobname}/"
                            f"analysis/"
                            f"rmsf_plddt/"
                            f"{jobname}_"
                            f"all_plddt.png")

        plt.savefig(full_output_path, dpi=300)
        plt.close()


def build_dataset_rmsf_peaks(jobname, results_dict, output_path, engine):
    """
    Build a dataset from detected RMSF peaks and save it as a CSV file.

    Parameters:
    jobname (str): The name of the job or analysis.
    results_dict (dict): A dictionary containing detected peaks for each prediction.
    output_path (str): Directory where the analysis results and dataset will be saved.
    engine (str): The name of the engine used for the analysis (e.g., AlphaFold2, OpenFold).

    Returns:
    None: The function saves the dataset as a CSV file in the specified output directory.
    """
    trials = []
    peak_labels = []
    peak_starting_residues = []
    peak_ending_residues = []
    peak_lengths = []
    peak_rmsf_values = []
    peak_prominences = []
    peak_width_heights = []

    for result in results_dict:
        peak_data = results_dict[result]['detected_peaks']
        for peak in peak_data:
            trials.append(result)
            peak_labels.append(peak)
            peak_starting_residues.append(results_dict[result]['detected_peaks'][peak]['starting_residue'])
            peak_ending_residues.append(results_dict[result]['detected_peaks'][peak]['ending_residue'])
            peak_rmsf_values.append(results_dict[result]['detected_peaks'][peak]['peak_value'])
            peak_prominences.append(results_dict[result]['detected_peaks'][peak]['prominence'])
            peak_lengths.append(results_dict[result]['detected_peaks'][peak]['length'])
            peak_width_heights.append(results_dict[result]['detected_peaks'][peak]['width_height'])

    df = pd.DataFrame({'peak_label': peak_labels,
                       'starting_residue': peak_starting_residues,
                       'ending_residue': peak_ending_residues,
                       'length': peak_lengths,
                       'rmsf_value': peak_rmsf_values,
                       'prominence': peak_prominences,
                       'width_height': peak_width_heights,
                       'trial': trials})

    full_output_path = (f"{output_path}/"
                        f"{jobname}/"
                        f"analysis/"
                        f"mobile_detection/"
                        f"{jobname}_"
                        f"rmsf_peak_calling_results.csv")

    df.to_csv(full_output_path, index=False)
    print(f"\nSaved peak calling output of all results at {output_path}/{jobname}/predictions/{engine}/ to "
          f"{full_output_path}\n")
