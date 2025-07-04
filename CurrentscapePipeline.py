import os
import numpy as np
from neuron import nrn_dll_sym_nt

from currentscape_calculator.CurrentscapeCalculator import CurrentscapeCalculator
from simulator.ModelSimulator import ModelSimulator
from preprocessor.Preprocessor import Preprocessor
from currentscape_visualization.currentscape import plot_currentscape


class CurrentscapePipeline:
    def __init__(
        self,
        output_dir='output',
        target='soma',
        partitioning='type',
        stim_dend=108,
        tmin=280,
        tmax=380,
        currentscape_filename='currentscape.pdf',
        maxNsyn=30,
        nsyn=8,
        tInterval=0.3,
        onset=300,
        direction='IN',
        tstop=900
    ):
        self.maxNsyn = maxNsyn
        self.nsyn = nsyn
        self.tInterval = tInterval
        self.onset = onset
        self.direction = direction
        self.tstop = tstop
        self.output_dir = output_dir
        self.target = target
        self.partitioning = partitioning
        self.stim_dend = stim_dend
        self.tmin = tmin
        self.tmax = tmax
        self.currentscape_filename = currentscape_filename
        self.simulation_data = None
        self.taxis = None

    def run_simulation(self):
        simulator = ModelSimulator()
        model = simulator.build_model(self.stim_dend)
        self.simulation_data = simulator.run_simulation(model, self.maxNsyn, self.nsyn,
                                                        self.tInterval, self.onset, self.direction, self.tstop)
        self.taxis = self.simulation_data['taxis']

    def preprocess(self):
        preprocessor = Preprocessor(self.simulation_data)
        self.im = preprocessor.preprocess_membrane_currents()
        self.iax = preprocessor.preprocess_axial_currents()
        pre_dir = os.path.join(self.output_dir, 'preprocessed')
        os.makedirs(pre_dir, exist_ok=True)
        self.im_path = os.path.join(pre_dir, 'im.csv')
        self.iax_path = os.path.join(pre_dir, 'iax.csv')
        self.im.to_csv(self.im_path)
        self.iax.to_csv(self.iax_path)

    def calculate_currentscape(self):
        region_list_dir = os.path.join('currentscape_calculator', 'region_list')
        calc = CurrentscapeCalculator(self.target, self.partitioning, region_list_dir)
        self.part_pos, self.part_neg = calc.calculate_currentscape(
            self.iax_path, self.im_path, self.taxis, self.tmin, self.tmax
        )

    def visualize(self):
        v_idx = np.where(
            np.array(self.simulation_data['membrane_potential_data'][0]).astype(str) == f'{self.target}(0.5)'
        )
        v_target = np.array(self.simulation_data['membrane_potential_data'][1])[v_idx].squeeze()[
            np.flatnonzero((self.taxis > self.tmin) & (self.taxis < self.tmax))
        ]
        plot_currentscape(
            self.part_pos, self.part_neg, v_target, self.taxis, self.tmin, self.tmax,
            self.currentscape_filename, return_segs=False, segments_preselected=False,
            partitionby=self.partitioning
        )

    def run_full_pipeline(self):
        self.run_simulation()
        self.preprocess()
        self.calculate_currentscape()
        self.visualize()
