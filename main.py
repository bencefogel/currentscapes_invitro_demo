from currentscape_calculator.CurrentscapeCalculator import CurrentscapeCalculator
from datasaver.DataSaver import DataSaver
from simulator.ModelSimulator import ModelSimulator
from preprocessor.Preprocessor import Preprocessor
import os

# model parameters:
cluster_seed = 0
random_seed = 30
# simulation parameters:
e_input = 'E:/FBence/CA1_PFs/synaptic_input/Espikes_d10_Ne2000_Re0.5_rseed1_rep0.dat'
i_input = 'E:/FBence/CA1_PFs/synaptic_input/Ispikes_d10_Ni200_Ri7.4_rseed1_rep0.dat'
simulation_time = 1 * 1000
output_directory = 'output'
# partitioning parameters:
target = 'soma'
partitioning_strategy = 'type'

# generate simulation data
simulator = ModelSimulator()
model = simulator.build_model(cluster_seed, random_seed)
simulation_data = simulator.run_simulation(model, e_input, i_input, simulation_time)

# preprocessing (set target section)
preprocessor = Preprocessor(simulation_data, target)
im = preprocessor.preprocess_membrane_currents()
iax = preprocessor.preprocess_axial_currents()

# save results
preprocessed_im_directory = 'preprocessed/im'
preprocessed_iax_directory = 'preprocessed/iax'
preprocessed_datasaver = DataSaver(columns_in_chunk=1000)
preprocessed_datasaver.save_in_chunks(im, os.path.join(output_directory, preprocessed_im_directory), 'im')
preprocessed_datasaver.save_in_chunks(iax, os.path.join(output_directory, preprocessed_iax_directory), 'iax')

# partition axial currents of the target (can be type-or region-specific)
input_directory = os.path.join(output_directory, 'preprocessed')
currentscape_calculator = CurrentscapeCalculator(target, partitioning_strategy, input_directory, output_directory)
currentscape_calculator.calculate_currentscape()
