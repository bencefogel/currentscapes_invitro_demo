import numpy as np
import simulator.model.simulation as simulation


def SIM_nsynIteration(model, nsyn, t_interval, onset, direction, t_stop):
    etimes = genDSinput(nsyn, t_interval, onset, direction)
    fih = simulation.h.FInitializeHandler(1, lambda: initSpikes_dend(model, etimes))
    simulation_data = simulation.simulate(model, t_stop)
    simulation_data['etimes'] = etimes
    return simulation_data


def genDSinput(nsyn, t_interval, onset, direction):
    # a single train with nsyn inputs - either in the in or in the out direction
    times = np.zeros([nsyn, 2])
    if direction == 'OUT':
        times[:, 0] = np.arange(0, nsyn)
    else:
        times[:, 0] = np.arange(nsyn-1, -1, -1)  # Reverse order for 'IN' direction
    times[:, 1] = np.arange(0, nsyn * t_interval, t_interval)[0:nsyn] + onset
    return times


def initSpikes_dend(model, etimes):
    if (len(etimes)>0):
        for s in etimes:
            model.ncAMPAlist[int(s[0])].event(float(s[1]))
            model.ncNMDAlist[int(s[0])].event(float(s[1]))
