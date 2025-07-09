import pandas as pd
from neuron import h

import simulator.model.simulation as simulation
from simulator.model.ca1_model import CA1
from simulator.model.ca1_functions import init_activeCA1, add_syns
from simulator.model.ca1_functions import genDendLocs
from simulator.model.sim_functions import SIM_nsynIteration
from simulator.model.utils.extract_connections import get_external_connections, get_internal_connections, get_connections
from simulator.model.utils.extract_areas import get_segment_areas

class ModelSimulator:
    """
    ModelSimulator is a class that constructs a CA1 hippocampal model, configures synaptic inputs,
    and runs simulations. It manages both the model building process and the
    execution of simulations, returning the simulation data including membrane potentials,
    segment connections and segment areas.
    """

    def __init__(self):
        """
       Initialize the ModelSimulator object.

       Args:
           connections (dict): Stores internal and external segment connections for later processing.
           segment_areas (pd.DataFrame): DataFrame containing segment name and segment area information.
       """
        self.connections = {}
        self.segment_areas = pd.DataFrame()

    def build_model(self, ca: bool, stimulated_dend: int, nsyn: int) -> CA1:
        """
        Build the CA1 hippocampal model with synaptic inputs.

        Args:
            ca (bool): Whether to have R-type Ca2+ (and slow K+) active or not.
            stimulated_dend (int): The dendrite to stimulate.
            nsyn (int): The number of synaptic inputs to add to the model.

        Returns:
            CA1: A configured instance of the CA1 model.
        """
        print("Building CA1 model...")
        # Create and initialize the CA1 model
        model = CA1()
        init_activeCA1(model, ca)

        # Generate synapse locations on the dendrites and add synapses to the model
        Elocs = genDendLocs(stimulated_dend, nsyn)
        add_syns(model, Elocs)

        # Get connections and segment area data
        self.connections['external'] = get_external_connections()
        self.connections['internal'] = get_internal_connections()
        self.segment_areas = get_segment_areas()
        return model

    def run_simulation(self, model: CA1, nsyn: int, t_interval: float, onset: int, direction: str,
                       t_stop: int) -> dict:
        """
        Run a simulation with the specified parameters.

        This method executes a simulation for a given model and specified parameters,
        handling aspects such as interval timing, onset, and simulation direction.
        It retrieves simulation data along with specific connection and area information for further processing.

        Args:
            model (CA1): The biophysical model used for the simulation.
            max_nsyn (int): The maximum number of synaptic connections to consider.
            nsyn (int): The actual number of synaptic connections to simulate.
            t_interval (float): The interval time for the simulation in milliseconds.
            onset (int): The onset time for the start of the simulation in miliseconds.
            direction (str): The direction of the simulation (e.g., IN or OUT).
            t_stop (int): The stop time for the end of the simulation in milliseconds.

        Returns:
            dict: A dictionary containing simulation data, connections information,
            and segment area details.
        """
        print("Running simulation...")
        simulation_data = SIM_nsynIteration(model, nsyn=nsyn, t_interval=t_interval, onset=onset,
                                            direction=direction, t_stop=t_stop)
        simulation_data['connections'] = get_connections(self.connections['external'], self.connections['internal'])
        simulation_data['areas'] = self.segment_areas
        return simulation_data
