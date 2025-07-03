import numpy as np

import simulator.model.simulation as simulation
from numpy import loadtxt


def SIM_nsynIteration(model, maxNsyn, nsyn, tInterval, onset, direction='IN', tstop=300):
    # next, synapses are activated together
    etimes = genDSinput(nsyn, maxNsyn, tInterval, onset * 1000, direction)
    fih = simulation.h.FInitializeHandler(1, lambda: initSpikes_dend(model, etimes))
    simulation_data = simulation.simulate(model, tstop)
    simulation_data['etimes'] = etimes
    return simulation_data


def genDSinput(nsyn, Nmax, tInterval, onset, direction='OUT'):
    # a single train with nsyn inputs - either in the in or in the out direction
    times = np.zeros([nsyn, 2])
    if (direction=='OUT'):
        times[:,0] = np.arange(0, nsyn)
    else:
        times[:,0] = np.arange(Nmax-1, Nmax-nsyn-1, -1)
    times[:,1] = np.arange(0, nsyn*tInterval, tInterval)[0:nsyn] + onset
    return times


def initSpikes_dend(model, etimes):
    if (len(etimes)>0):
        for s in etimes:
            model.ncAMPAlist[int(s[0])].event(float(s[1]))
            model.ncNMDAlist[int(s[0])].event(float(s[1]))
