import os
import warnings
from fast_conformation.ensemble_analysis.analysis_utils import create_directory, load_predictions, load_config, auto_select_2d_references
from fast_conformation.ensemble_analysis.twodrmsd import TwodRMSD
import argparse

warnings.filterwarnings("ignore")


def run_2d_rmsd_analysis(config, widget=None):
    """
    Run 2D RMSD analysis based on the provided configuration.

    Parameters:
    config (dict): Configuration dictionary containing parameters for the analysis.
    widget (object, optional): Widget for displaying results (default is None).

    Raises:
    NotADirectoryError: If the specified output path is not a directory.
    """

    # Retrieve configuration values
    jobname = config.get('jobname')
    output_path = config.get('output_path')
    mode_results = config.get('mode_results')
    seq_pairs = config.get('seq_pairs')
    predictions_path = config.get('predictions_path')
    engine = config.get('engine')
    align_range = config.get('align_range')
    analysis_range = config.get('analysis_range')
    analysis_range_name = config.get('analysis_range_name')
    ref2d1 = config.get('ref2d1')
    ref2d2 = config.get('ref2d2')
    n_stdevs = config.get('n_stdevs')
    n_clusters = config.get('n_clusters')
    starting_residue = config.get('starting_residue')

    # Check if the output path is a valid directory
    if not os.path.isdir(output_path):
        raise NotADirectoryError(f"Output path {output_path} is not a directory")

    # Set default predictions path if not provided
    if not predictions_path:
        predictions_path = f'{output_path}/{jobname}/predictions/{engine}'

    # Set default mode results path if not provided
    if not mode_results:
        mode_results = f'{output_path}/{jobname}/analysis/rmsd_1d/{jobname}_rmsd_1d_analysis_results.csv'

    # Auto-select references if not provided
    if not ref2d1 and not ref2d2:
        ref2d1, ref2d2 = auto_select_2d_references(mode_results, 'RMSD')

    # Create necessary directories
    create_directory(f'{output_path}/{jobname}/analysis/rmsd_2d')

    # Display configurations
    print("\nConfigurations:")
    print("***************************************************************")
    print(f"Job Name: {jobname}")
    print(f"Output Path: {output_path}")
    print(f"max_seq:extra_seq Pairs: {seq_pairs}")
    print(f"Predictions Path: {predictions_path}")
    print(f"Engine: {engine}")
    print(f"Align Range: {align_range}")
    print(f"Analysis Range: {analysis_range_name} = {analysis_range}")
    print(f"Reference 1: {ref2d1}")
    print(f"Reference 2: {ref2d2}")
    print(f"Number of Standard Devs. to Consider Points Close to fit Line: {n_stdevs}")
    if n_clusters:
        print(f"Number of Clusters: {n_clusters}")
    else:
        print(f"Number of Clusters: Number of Detected 1D RMSD Modes + 1")
    if starting_residue:
        print(f"Starting Residue: {starting_residue}")
    print("***************************************************************\n")

    # Prepare input dictionary
    input_dict = {
        'jobname': jobname,
        'output_path': output_path,
        'seq_pairs': seq_pairs,
        'analysis_range': analysis_range,
        'analysis_range_name': analysis_range_name,
        'align_range': align_range,
    }

    # Load predictions to RAM
    pre_analysis_dict = load_predictions(predictions_path, seq_pairs, jobname, starting_residue)

    # Run 2D RMSD analysis
    twod = TwodRMSD(pre_analysis_dict, input_dict, widget, ref2d1, ref2d2)

    # Build and save results dataset
    twod.get_2d_rmsd(mode_results, n_stdevs, n_clusters, output_path)


def main():
    """
    Main function to parse arguments and run 2D RMSD analysis.
    """

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Runs state detection analysis using RMSD vs. two references (2D) "
                                                 "previously identified with rmsd_mode1d.py")

    parser.add_argument('--config_file', type=str, help="Path to the configuration file")
    parser.add_argument('--jobname', type=str, help="The job name")
    parser.add_argument('--output_path', type=str, help="Path to save results to")
    parser.add_argument('--mode_results', type=str,
                        help="Path to output .csv file from 1D mode/state detection tool "
                             "(if not provided, will search automatically based on other parameters)")
    parser.add_argument('--seq_pairs', type=str, help="A list of [max_seq, extra_seq] pairs "
                                                      "previously used to generate the predictions")
    parser.add_argument('--predictions_path', type=str,
                        help="Path to read PDB files of predictions, expects format: /jobname_maxseq_extraseq/ "
                             "(if not provided, will search automatically based on other parameters)")
    parser.add_argument('--engine', type=str, choices=['alphafold2', 'openfold'],
                        help="The engine previously used to generate predictions (AlphaFold2 or OpenFold), "
                             "used to find predictions if predictions_path is not supplied")
    parser.add_argument('--starting_residue', type=int,
                        help="Sets the starting residue for reindexing (predictions are usually 1-indexed)")
    parser.add_argument('--align_range', type=str, help="The atom alignment range for RMSF calculations "
                                                        "(MDAnalysis Syntax)")
    parser.add_argument('--analysis_range', type=str, help="The atom range for RMSD calculations "
                                                           "after alignment to --align_range")
    parser.add_argument('--analysis_range_name', type=str, help="The name of the atom range "
                                                                "(e.g. kinase core, helix 1, etc.)")
    parser.add_argument('--ref2d1', type=str, help="Path to the .pdb file defining "
                                                   "the first reference structure "
                                                   "for RMSD calculations "
                                                   "(if not provided, chooses from mode dataset)")
    parser.add_argument('--ref2d2', type=str, help="Path to the .pdb file defining"
                                                   "the second reference structure "
                                                   "for RMSD calculations "
                                                   "(if not provided, chooses from mode dataset)")
    parser.add_argument('--n_stdevs', type=str, help="Number of standard deviations "
                                                     "to consider when calculating close points to fit curve")
    parser.add_argument('--n_clusters', type=str, help="Number of clusters to consider for RMSD analysis")

    args = parser.parse_args()

    # Load configuration from file if provided
    config_file = args.config_file if args.config_file else 'config.json'
    config = load_config(config_file)

    # Override config with command line arguments if provided
    config.update({k: v for k, v in vars(args).items() if v is not None})

    # Run 2D RMSD analysis with the provided configuration
    run_2d_rmsd_analysis(config)


if __name__ == "__main__":
    main()
