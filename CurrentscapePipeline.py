import os
import numpy as np
import pandas as pd

from currentscape_calculator.CurrentscapeCalculator import CurrentscapeCalculator
from simulator.ModelSimulator import ModelSimulator
from preprocessor.Preprocessor import Preprocessor
from currentscape_visualization.currentscape import plot_currentscape


class CurrentscapePipeline:
    """
    A pipeline to simulate neuronal activity, preprocess currents,
    calculate currentscape, and generate currentscape visualizations.

    Attributes:
        output_dir (str): The directory where the preprocessed and output files will be saved.
        target (str): The neuronal compartment targeted for currentscape calculations.
        partitioning (str): Strategy used for partitioning currents ('type' or 'region').
        ca (bool): A boolean indicating if calcium channels are included in the model.
        stim_dend (int): Dendrite that is stimulated during the simulation.
        direction (str): Direction of stimulation in the model ('IN' or 'OUT').
        tstop (int): Total duration for the simulation in milliseconds.
        tmin (int): Start time for currentscape calculation in milliseconds.
        tmax (int): End time for currentscape calculation in milliseconds.
        nsyn (int): Number of synapses for stimulation.
        t_interval (float): Time interval between synaptic stimulations in milliseconds.
        onset (int): Time of stimulation onset in milliseconds.
        currentscape_filename (str): Output file name for the currentscape plot.
        simulation_data (dict): Dictionary holding the results of the simulation.
        taxis (array): Array representing the time axis of the simulation results.
    """
    def __init__(self, output_dir: str = 'output', target: str = 'soma', partitioning: str = 'type', ca: bool = True,
                 stim_dend: int = 108, direction: str = 'IN', tstop: int = 900, tmin: int = 280, tmax: int = 380,
                 nsyn: int = 8, t_interval: float = 0.3, onset: int = 300,
                 currentscape_filename: str = 'currentscape.pdf') -> None:

        self.output_dir = output_dir
        self.target = target
        self.partitioning = partitioning
        self.ca = ca
        self.stim_dend = stim_dend
        self.direction = direction
        self.tstop = tstop
        self.tmin = tmin
        self.tmax = tmax
        self.nsyn = nsyn
        self.tInterval = t_interval
        self.onset = onset
        self.currentscape_filename = currentscape_filename
        self.simulation_data = None
        self.taxis = None


    def run_simulation(self):
        """
        Runs the simulation using the ModelSimulator.

        This method builds a neuron model with the specified stimulated dendrite,
        executes the simulation with the configured parameters, and stores
        the simulation data including the time axis ('taxis').
        """
        simulator = ModelSimulator()
        model = simulator.build_model(self.ca, self.stim_dend, self.nsyn)
        self.simulation_data = simulator.run_simulation(model, self.nsyn, self.tInterval, self.onset,
                                                        self.direction, self.tstop)
        self.taxis = self.simulation_data['taxis']


    def preprocess(self):
        """
        Preprocesses simulation data for membrane and axial currents and saves
        the preprocessed data as CSV files.

        Args:
            simulation_data : dict
                The input simulation data to be processed.
            output_dir : str
                Path to the directory where the preprocessed files will be stored.
            im_path : str
                File path for the preprocessed membrane currents CSV.
            iax_path : str
                File path for the preprocessed axial currents CSV.
            im : DataFrame
                DataFrame containing the preprocessed membrane currents.
            iax : DataFrame
                DataFrame containing the preprocessed axial currents.
        """

        preprocessor = Preprocessor(self.simulation_data)
        self.im = preprocessor.preprocess_membrane_currents()
        self.iax = preprocessor.preprocess_axial_currents()

        pre_dir = os.path.join(self.output_dir, 'preprocessed')
        os.makedirs(pre_dir, exist_ok=True)
        self.im_path = os.path.join(pre_dir, 'im.csv')
        self.iax_path = os.path.join(pre_dir, 'iax.csv')

        self.im.to_csv(self.im_path)
        self.iax.to_csv(self.iax_path)
        np.save(os.path.join(self.output_dir, 'taxis.npy'), self.taxis)


    def calculate_currentscape(self):
        """
        Calculates the currentscape for a given target and partitioning strategy.

        This method creates an instance of the CurrentscapeCalculator class,
        specifies the directory containing the region list, and uses it to
        calculate the positive and negative partitioned currents based
        on the provided target, input files, and time constraints. The
        calculated values are stored in the attributes `part_pos` and
        `part_neg`.
        """
        region_list_dir = os.path.join('currentscape_calculator', 'region_list')
        calc = CurrentscapeCalculator(self.target, self.partitioning, region_list_dir)
        self.part_pos, self.part_neg = calc.calculate_currentscape(
            self.iax_path, self.im_path, self.taxis, self.tmin, self.tmax
        )

        res_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(res_dir, exist_ok=True)
        part_pos_path = os.path.join(res_dir, 'part_pos.csv')
        part_neg_path = os.path.join(res_dir, 'part_neg.csv')

        self.part_pos.to_csv(part_pos_path)
        self.part_neg.to_csv(part_neg_path)


    def visualize(self):
        """
        Generates a currentscape plot.

        This method processes the simulation data of membrane potential at a specific target,
        filters the time range, and plots the currentscape. It outputs the currentscape plot to a specified file.
        """
        v_idx = np.where(
            np.array(self.simulation_data['membrane_potential_data'][0]).astype(str) == f'{self.target}(0.5)'
        )
        v_target = np.array(self.simulation_data['membrane_potential_data'][1])[v_idx].squeeze()[
            np.flatnonzero((self.taxis > self.tmin) & (self.taxis < self.tmax))
        ]
        currentscape = plot_currentscape(
                                        self.part_pos, self.part_neg, v_target, self.taxis, self.tmin, self.tmax,
                                        return_segs=False, segments_preselected=False,
                                        partitionby=self.partitioning)
        currentscape.save(os.path.join(self.output_dir, self.currentscape_filename))
        print("Currentscape saved to " + os.path.join(self.output_dir, self.currentscape_filename))


    def run_full_pipeline(self):
        """
        Runs the entire data processing pipeline including simulation, preprocessing, calculation, and
        visualization stages. Each step is executed sequentially and is critical for the pipeline
        workflow. The method should be used to execute all stages in the correct order. This method
        does not take any arguments and does not return any value.
        """
        self.run_simulation()
        self.preprocess()
        self.calculate_currentscape()
        self.visualize()


    def results_exist(self) -> bool:
        """
        Checks if the currentscape results files exist in the specified output directory.
        """
        res_dir = os.path.join(self.output_dir, 'results')
        part_pos_path = os.path.join(res_dir, 'part_pos.csv')
        part_neg_path = os.path.join(res_dir, 'part_neg.csv')

        return os.path.exists(part_pos_path) and os.path.exists(part_neg_path)


    def load_results(self):
        """
        Loads existing results from the 'results' folder and sets up internal state for visualization.
        """
        res_dir = os.path.join(self.output_dir, 'results')
        part_pos_path = os.path.join(res_dir, 'part_pos.csv')
        part_neg_path = os.path.join(res_dir, 'part_neg.csv')

        # Load currentscape partition data
        self.part_pos = pd.read_csv(part_pos_path, index_col=0)
        self.part_neg = pd.read_csv(part_neg_path, index_col=0)
        self.part_pos.columns = self.part_pos.columns.astype(int)
        self.part_neg.columns = self.part_neg.columns.astype(int)

        # Load simulation time axis and membrane potential to visualize properly
        self.taxis = np.load(os.path.join(self.output_dir, 'taxis.npy'))