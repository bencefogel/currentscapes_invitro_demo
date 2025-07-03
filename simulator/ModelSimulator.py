import pandas as pd
import numpy as np
from neuron import h

import simulator.model.saveClass as sc
import simulator.model.simulation as simulation
from simulator.model.ca1_model import CA1
from simulator.model.ca1_functions import init_activeCA1, addClustLocs, genRandomLocs, add_syns
from simulator.model.sim_functions import SIM_nsynIteration
from simulator.model.utils.extract_connections import get_external_connections, get_internal_connections, get_connections
from simulator.model.utils.extract_areas import get_segment_areas

class ModelSimulator:
    """
    ModelSimulator is a class that constructs a CA1 hippocampal model, configures synaptic inputs,
    and runs simulations. It manages both the model building process and the
    execution of simulations, returning the simulation data including membrane potentials, membrane currents,
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

    def build_model(self, stimulated_dend) -> CA1:
        """
        Builds the CA1 model with synaptic connections.

        Args:
            cluster_seed (int): Seed for clustering excitatory synapses.
            random_seed (int): Seed for random synapse location generation.

        Returns:
            model (CA1): An instance of the CA1 model with configured synapses.
        """
        print("Building CA1 model...")
        # Initialize model
        model = CA1()
        init_activeCA1(model)

        # Generate excitatory synapse locations and cluster locations
        # Add synapses to model
        from simulator.model.ca1_functions import genDendLocs
        Elocs = genDendLocs(stimulated_dend)
        add_syns(model, Elocs)

        # Get section and segment connections
        self.connections['external'] = get_external_connections()
        self.connections['internal'] = get_internal_connections()
        self.segment_areas = get_segment_areas()
        return model

    def run_simulation(self, model: CA1, nsyn: int) -> dict:
        print("Running simulation...")
        simulation_data = SIM_nsynIteration(model, maxNsyn=30, nsyn=nsyn, tInterval=0.3, onset=0.3, direction='IN', tstop=900)
        simulation_data['connections'] = get_connections(self.connections['external'], self.connections['internal'])
        simulation_data['areas'] = self.segment_areas
        return simulation_data
