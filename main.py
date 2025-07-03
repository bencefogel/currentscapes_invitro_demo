import os
import numpy as np
from currentscape_calculator.CurrentscapeCalculator import CurrentscapeCalculator
from datasaver.DataSaver import DataSaver
from simulator.ModelSimulator import ModelSimulator
from preprocessor.Preprocessor import Preprocessor
from currentscape_visualization.currentscape import plot_currentscape


# model parameters:
output_directory = 'output'
# partitioning parameters:
target = 'soma'
partitioning_strategy = 'type'
stimulated_dend = 108

# generate simulation data
simulator = ModelSimulator()
model = simulator.build_model(stimulated_dend)
simulation_data = simulator.run_simulation(model, 8)

# preprocessing (set target section)
preprocessor = Preprocessor(simulation_data)
im = preprocessor.preprocess_membrane_currents()
iax = preprocessor.preprocess_axial_currents()

# save results
preprocessed_directory = os.path.join(output_directory, 'preprocessed')
if not os.path.exists(preprocessed_directory):
    os.makedirs(preprocessed_directory)
im.to_csv(os.path.join(preprocessed_directory, 'im.csv'))
iax.to_csv(os.path.join(preprocessed_directory, 'iax.csv'))


# partition axial currents of the target (can be type-or region-specific)
regions_list_directory = os.path.join('currentscape_calculator', 'region_list')
currentscape_calculator = CurrentscapeCalculator(target, partitioning_strategy, regions_list_directory)

im_fpath = os.path.join(preprocessed_directory, 'im.csv')
iax_fpath = os.path.join(preprocessed_directory, 'iax.csv')

part_pos, part_neg = currentscape_calculator.calculate_currentscape(iax_fpath, im_fpath, timepoints=None)

# run visualization
v_idx = np.where(np.array(simulation_data['membrane_potential_data'][0]).astype(str) == f'{target}(0.5)')
v_target = np.array(simulation_data['membrane_potential_data'][1])[v_idx].squeeze()
taxis = simulation_data['taxis']
filename = 'test.pdf'
tmin=0
tmax=98
plot_currentscape(part_pos, part_neg, v_target, taxis, tmin, tmax, filename, return_segs=False, segments_preselected=False, partitionby=partitioning_strategy)
