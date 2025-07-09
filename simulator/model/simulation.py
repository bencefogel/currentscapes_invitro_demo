from neuron import h, gui
import numpy as np

from simulator.model.ca1_model import CA1
from simulator.model.utils.record_intrinsic import record_intrinsic_currents, preprocess_intrinsic_data
from simulator.model.utils.record_synaptic import record_synaptic_currents, preprocess_synaptic_data
from simulator.model.utils.record_membrane_potential import record_membrane_potential, preprocess_membrane_potential_data


def simulate(model: CA1, tstop: float) -> dict:
    """
    Simulate the activity of a CA1 model.

    This function initializes and runs a simulation using the NEURON simulation environment.
    It records various physiological data, including membrane potential, intrinsic currents,
    and synaptic currents, for later analysis. The time series data is processed and
    downsampled to facilitate further evaluation.

    Parameters:
        model (CA1): The biophysical model to be simulated.
        tstop (float): The simulation end time in milliseconds.

    Returns:
        dict: A dictionary containing the processed simulation data. The dictionary keys
        are:
            - 'membrane_potential_data': A list with membrane potential segments and
              corresponding processed arrays.
            - 'intrinsic_data': A list with intrinsic current segments and corresponding
              processed arrays.
            - 'synaptic_data': A list with synaptic current segments and corresponding
              processed arrays.
            - 'taxis': A downsampled time axis array.
    """
    h.CVode().active(True)
    h.CVode().atol((1e-3))

    trec = h.Vector()
    trec.record(h._ref_t)

    v_segments, v = record_membrane_potential()
    intrinsic_segments, intrinsic_currents = record_intrinsic_currents()
    synaptic_segments, synaptic_currents = record_synaptic_currents(model)

    h.celsius = 35
    h.finitialize(-68.3)

    h.continuerun(tstop)

    taxis = np.array(trec)
    taxis_unique, index_unique = np.unique(taxis, return_index=True)
    x = int((max(taxis_unique)) * 5)
    taxis_downsampled = np.linspace(min(taxis_unique), max(taxis_unique), x)

    v_segments, v_arrays = preprocess_membrane_potential_data(v_segments, v, taxis_unique, index_unique)
    intrinsic_segments, intrinsic_arrays = preprocess_intrinsic_data(intrinsic_segments, intrinsic_currents, taxis_unique, index_unique)
    synaptic_segments, synaptic_arrays = preprocess_synaptic_data(synaptic_segments, synaptic_currents, taxis_unique, index_unique)

    simulation_data = {'membrane_potential_data': [v_segments, v_arrays],
                       'intrinsic_data': [intrinsic_segments, intrinsic_arrays],
                       'synaptic_data': [synaptic_segments, synaptic_arrays],
                       'taxis': taxis_downsampled}
    return simulation_data
