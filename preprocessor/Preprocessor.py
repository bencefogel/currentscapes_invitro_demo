import pandas as pd

from preprocessor.MembraneCurrentPreprocessor import MembraneCurrentPreprocessor
from preprocessor.AxialCurrentPreprocessor import AxialCurrentPreprocessor

class Preprocessor:
    """
    Controls preprocessing methods for axial, intrinsic, and synaptic currents.

    Args:
        simulation_data (dict): The simulation data used for preprocessing.
        target (str): The target section for current preprocessing.
    """
    def __init__(self, simulation_data: dict) -> None:
        """
        Initializes the Preprocessor with simulation data, target, partitioning strategy,
        membrane current preprocessor and axial current preprocessor.

        Args:
            simulation_data (dict): The simulation data used for preprocessing.
            target (str): The target section for current preprocessing.
            partitioning_strategy (str): Strategy for membrane current preprocessing.
            membrane_current_preprocessor (MembraneCurrentPreprocessor): An instance for processing membrane currents.
            axial_current_preprocessor (AxialCurrentPreprocessor): An instance for processing axial currents.
        """
        self.simulation_data = simulation_data
        self.target = 'soma'  # soma is always the target compartment for the preprocessing steps
        self.membrane_current_preprocessor = MembraneCurrentPreprocessor()
        self.axial_current_preprocessor = AxialCurrentPreprocessor()

    def preprocess_membrane_currents(self) -> pd.DataFrame:
        """
        Preprocesses membrane currents using the MembraneCurrentPreprocessor.
        Combines and merges membrane currents based on the target section.

        Returns:
            pd.DataFrame: A DataFrame containing processed membrane current data.
        """
        print("Preprocessing membrane currents...")
        self.membrane_current_preprocessor.combine_membrane_currents(self.simulation_data)
        im = self.membrane_current_preprocessor.merge_section_im(self.target)
        return im

    def preprocess_axial_currents(self) -> pd.DataFrame:
        """
        Preprocesses axial currents using the AxialCurrentPreprocessor.
        Calculates and merges axial currents based on the target section.

        Returns:
            pd.DataFrame: A DataFrame containing processed axial current data.
        """
        print('Preprocessing axial currents...')
        self.axial_current_preprocessor.calculate_axial_currents(self.simulation_data)
        iax = self.axial_current_preprocessor.merge_section_iax(self.target)
        return iax

