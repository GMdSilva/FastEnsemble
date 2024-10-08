from fast_conformation.jackhmmer_msa import build_jackhmmer_msa
from fast_conformation.mmseqs2_msa import run_mmseqs2_msa
from fast_conformation.gui.widget_base import AnalysisWidgetBase, merge_configs
from PyQt5.QtWidgets import QFileDialog, QLabel, QVBoxLayout, QComboBox, QLineEdit, QPushButton, QCheckBox, QHBoxLayout
from PyQt5.QtCore import Qt

class MSAOptionsWidget(AnalysisWidgetBase):
    """
    MSAOptionsWidget provides a user interface for configuring and running multiple 
    sequence alignment (MSA) jobs using either jackhmmer or mmseqs2.

    Methods:
        init_ui: Initializes the UI components.
        select_sequence_path: Opens a file dialog to select the sequence file.
        select_output_path: Opens a file dialog to select the output directory.
        toggle_additional_options: Toggles visibility of additional options based on the selected MSA type.
        validate_inputs: Validates the user inputs.
        get_specific_options: Retrieves the specific options for MSA analysis.
        run_specific_analysis: Runs the specific MSA analysis.
        run_analysis: Validates inputs, merges configurations, and starts the MSA job.
    """
    def __init__(self, job_manager):
        """
        Initialize the MSAOptionsWidget with a job manager.

        Args:
            job_manager: The manager responsible for handling job execution.
        """
        super().__init__(job_manager)
        self.init_ui()

    def init_ui(self):
        """
        Initializes the user interface components for the widget.
        """
        layout = QVBoxLayout()
        self.jobname_label = QLabel("Job Name: ")
        self.jobname_input = QLineEdit("msa_job")

        # MSA build method
        self.type_label = QLabel("MSA type:")
        self.type_dropdown = QComboBox()
        self.output_path_input = QLineEdit("")
        self.output_path_button = QPushButton("Browse")
        self.type_dropdown.addItems(["jackhmmer", "mmseqs2"])
        self.type_dropdown.currentIndexChanged.connect(self.toggle_additional_options)
        self.output_path_label = QLabel("Output Path:")
        layout.addWidget(self.jobname_label)
        layout.addWidget(self.jobname_input)
        layout.addWidget(self.output_path_label)
        layout.addWidget(self.output_path_input)
        layout.addWidget(self.output_path_button)
        self.sequence_path_label = QLabel("Sequence Path:")
        self.sequence_path_input = QLineEdit()
        self.sequence_path_button = QPushButton("Select Sequence File")
        self.sequence_path_button.clicked.connect(self.select_sequence_path)
        self.output_path_button.clicked.connect(self.select_output_path)
        
        self.tmp_dir_label = QLabel("Temporary Directory:")
        self.tmp_dir_input = QLineEdit("tmp")
        
        self.homooligomers_label = QLabel("Homooligomers:")
        self.homooligomers_input = QLineEdit("1")
        
        self.use_ramdisk_label = QLabel("Use RAM Disk:")
        self.use_ramdisk_checkbox = QCheckBox()
        self.use_ramdisk_checkbox.setChecked(False)

        layout.addWidget(self.type_label)
        layout.addWidget(self.type_dropdown)
        layout.addWidget(self.sequence_path_label)
        layout.addWidget(self.sequence_path_input)
        layout.addWidget(self.sequence_path_button)
        layout.addWidget(self.tmp_dir_label)
        layout.addWidget(self.tmp_dir_input)
        layout.addWidget(self.homooligomers_label)
        layout.addWidget(self.homooligomers_input)
        layout.addWidget(self.use_ramdisk_label)
        layout.addWidget(self.use_ramdisk_checkbox)
        self.setLayout(layout)
        self.setWindowTitle("MSA Options")

        # Run Button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_button)

        self.toggle_additional_options()  # Initial call to set the correct visibility

    def select_sequence_path(self):
        """
        Opens a file dialog to allow the user to select a sequence file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Sequence File", "", "FASTA files (*.fasta *.fa)")
        if file_path:
            self.sequence_path_input.setText(file_path)
    
    def select_output_path(self):
        """
        Opens a file dialog to allow the user to select an output directory.
        """
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.output_path_input.setText(directory)

    def toggle_additional_options(self):
        """
        Toggles the visibility of additional options based on the selected MSA type.
        """
        is_jackhmmer_selected = self.type_dropdown.currentText() == "jackhmmer"
        self.tmp_dir_label.setVisible(is_jackhmmer_selected)
        self.tmp_dir_input.setVisible(is_jackhmmer_selected)
        self.homooligomers_label.setVisible(is_jackhmmer_selected)
        self.homooligomers_input.setVisible(is_jackhmmer_selected)
        self.use_ramdisk_label.setVisible(is_jackhmmer_selected)
        self.use_ramdisk_checkbox.setVisible(is_jackhmmer_selected)

    def validate_inputs(self):
        """
        Validates the user inputs to ensure they are correct before running the MSA job.

        Returns:
            A list of error messages if validation fails, otherwise an empty list.
        """
        errors = []
        if not self.sequence_path_input.text():
            errors.append("Sequence Path cannot be empty.")
        if not self.tmp_dir_input.text():
            errors.append("Temporary Directory cannot be empty.")
        if not self.homooligomers_input.text().isdigit():
            errors.append("Homooligomers must be a number.")
        return errors

    def get_specific_options(self):
        """
        Retrieves the specific options set by the user for MSA analysis.

        Returns:
            A dictionary containing the specific MSA options.
        """
        sequence_path = self.sequence_path_input.text()
        with open(sequence_path, 'r') as file:
            sequence_string = file.read()
        
        return {
            "jobname": self.jobname_input.text(),
            "sequence_path": sequence_path,
            "output_path": self.output_path_input.text(),
            "use_ramdisk": self.use_ramdisk_checkbox.isChecked(),
            "homooligomers": int(self.homooligomers_input.text()),
            "tmp_dir": self.tmp_dir_input.text(),
            "sequence_string": str(sequence_string),
            "msa_type": self.type_dropdown.currentText()
        }
    
    def run_specific_analysis(config):
        """
        Runs the specific MSA analysis based on the provided configuration.

        Args:
            config: A dictionary containing the configuration options for the MSA analysis.
        """
        run_msa_job(config)
    
    def run_analysis(self):
        """
        Validates the inputs, merges configurations, and starts the MSA job.
        """
        errors = self.validate_inputs()
        if errors:
            self.show_error_message(errors)
            return

        try:
            g_options = self.general_options_getter()
        except Exception:
            g_options = {}

        specific_options = self.get_specific_options()
        config = merge_configs(g_options, specific_options)

        job_id = self.job_manager.run_job(run_msa_job, (config,), config['jobname'])
        self.show_info_message(f"Job {config['jobname']} started.")

def run_msa_job(config):
    """
    Runs the MSA job based on the configuration provided.

    Args:
        config: A dictionary containing the configuration options for the MSA job.
    """
    print("starting")
    msa_type = config["msa_type"]
    if msa_type == "jackhmmer":
        build_jackhmmer_msa(config)
    elif msa_type == "mmseqs2":
        run_mmseqs2_msa(config)

    print(f"MSA run with type: {msa_type}")