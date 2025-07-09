import sys
import numpy as np

from neuron import h
from simulator.model.ca1_model import CA1
h('objref nil')

modpath = 'simulator/model/density_mechs'
h.nrn_load_dll(modpath + '\\nrnmech.dll')


def init_activeCA1(model: CA1, ca: bool) -> None:
    """
    Initializes active properties within a model neuron, setting up channel
    densities, locations, and dependencies on distance for various regions of
    the neuron including soma, axon, dendrites, apical and basal sections.

    Args:
        model: The model neuron object to be configured.
        ca: Whether to have R-type Ca2+ (and slow K+) active or not.

    Modifies:
        The provided `model` by inserting ion channels (e.g., 'nax', 'kdr',
        'kap', 'kad', 'nad', etc.), setting their densities for respective
        regions, and adjusting properties based on distance from the soma for
        apical dendrites and primary apical trunks.

    Note:
        The configuration includes:
        - Axonal initial segments and dendritic regions.
        - Distance-dependent A-type potassium channels ('kap' and 'kad').
        - Sodium channel ('nax') properties with distance scaling.
        - Specific conductances for calcium ('car') and slow potassium ('kslow')
          channels added to apical dendritic sections.
    """
    model.soma.insert('nax')
    model.soma.gbar_nax = model.gna_soma
    model.soma.insert('kdr')
    model.soma.gkdrbar_kdr = model.gkdr_soma
    model.soma.insert('kap')
    model.soma.gkabar_kap = model.gka

    model.hill.insert('nax')
    model.hill.gbar_nax = model.gna_axon
    model.hill.insert('kdr')
    model.hill.gkdrbar_kdr = model.gkdr_axon
    model.soma.insert('kap')
    model.soma.gkabar_kap = model.gka

    model.iseg.insert('nax')
    model.iseg.gbar_nax = model.gna_axon
    model.iseg.insert('kdr')
    model.iseg.gkdrbar_kdr = model.gkdr_axon
    model.iseg.insert('kap')
    model.soma.gkabar_kap = model.gka

    model.node[0].insert('nax')
    model.node[0].gbar_nax = model.gna_node
    model.node[0].insert('kdr')
    model.node[0].gkdrbar_kdr = model.gkdr_axon
    model.node[0].insert('kap')
    model.node[0].gkabar_kap = model.gka * 0.2

    model.node[1].insert('nax')
    model.node[1].gbar_nax = model.gna_node
    model.node[1].insert('kdr')
    model.node[1].gkdrbar_kdr = model.gkdr_axon
    model.node[1].insert('kap')
    model.node[1].gkabar_kap = model.gka * 0.2

    model.inode[0].insert('nax')
    model.inode[0].gbar_nax = model.gna_axon
    model.inode[0].insert('kdr')
    model.inode[0].gkdrbar_kdr = model.gkdr_axon
    model.inode[0].insert('kap')
    model.inode[0].gkabar_kap = model.gka * 0.2

    model.inode[1].insert('nax')
    model.inode[1].gbar_nax = model.gna_axon
    model.inode[1].insert('kdr')
    model.inode[1].gkdrbar_kdr = model.gkdr_axon
    model.inode[1].insert('kap')
    model.inode[1].gkabar_kap = model.gka * 0.2

    model.inode[2].insert('nax')
    model.inode[2].gbar_nax = model.gna_axon
    model.inode[2].insert('kdr')
    model.inode[2].gkdrbar_kdr = model.gkdr_axon
    model.inode[2].insert('kap')
    model.inode[2].gkabar_kap = model.gka * 0.2

    for d in model.dends:
        d.insert('nad')
        d.gbar_nad = model.gna
        d.insert('kdr')
        d.gkdrbar_kdr = model.gkdr
        d.insert('kap')
        d.gkabar_kap = 0
        d.insert('kad')
        d.gkabar_kad = 0

    h('access soma')
    h('distance()')

    ## for the apicals: KA-type depends on distance
    ## density is as in terminal branches - independent of the distance
    for sec in h.all_apicals:
        nseg = sec.nseg
        iseg = 0
        for seg in sec:
            xx = iseg * 1.0 / nseg + 1.0 / nseg / 2.0
            xdist = h.distance(xx, sec=sec)
            if (xdist > model.dprox):
                seg.gkabar_kad = model.gka
            else:
                seg.gkabar_kap = model.gka
            iseg = iseg + 1

    h('access soma')
    h('distance()')

    ## distance dependent A-channel densities in apical trunk dendrites
    ##      1. densities increase till 'dlimit' with dslope
    ##      2. proximal channels switch to distal at 'dprox'
    ##      3. sodium channel density also increases with distance
    for sec in h.primary_apical_list:
        nseg = sec.nseg
        sec.insert('nax')
        iseg = 0
        for seg in sec:
            # 0. calculate the distance from soma
            xx = iseg * 1.0 / nseg + 1.0 / nseg / 2.0
            xdist = h.distance(xx, sec=sec)
            # 1. densities increase till 'dlimit' with dslope
            if (xdist > model.dlimit):
                xdist = model.dlimit
            # 2. proximal channels switch to distal at 'dprox'
            if (xdist > model.dprox):
                seg.gkabar_kad = model.gka_trunk * (1 + xdist * model.dslope)
            else:
                seg.gkabar_kap = model.gka_trunk * (1 + xdist * model.dslope)
            iseg = iseg + 1
            # 3. sodiom channel density also increases with distance
            if (xdist > model.nalimit):
                xdist = model.nalimit
            seg.gbar_nax = model.gna_trunk * (1 + xdist * model.naslope)

            seg.gbar_nad = 0
            seg.gkdrbar_kdr = model.gkdr_trunk

    ## for the basals: all express proximal KA-type
    ## density does not increase with the distance
    for sec in h.all_basals:
        for seg in sec:
            seg.gkabar_kap = model.gka

    # Adding Ca and K_slow conductances
    if ca:
        for sec in h.all_apicals:
            sec.insert('car')
            sec.gmax_car = 0.006
            sec.insert('kslow')
            sec.gmax_kslow = 0.001


def add_syns(model: CA1, Elocs: list[list[int, float]]):
    """
    Adds AMPA and NMDA synapses to the specified locations on the model's neuronal structure.
    This function defines two types of synaptic channels (AMPA and NMDA) with their specific
    parameters and attaches them to the corresponding locations in the dendrites of
    the model.

    Parameters:
        model (CA1): The neuronal model to which the synapses will be added.
        Elocs (list[list[int, float]]): A list of locations where synapses will
            be added. Each location is represented as a list containing two elements:
            an integer representing the index of a dendritic section,
            and a float representing the position along the section.
    """

    model.AMPAlist = []
    model.ncAMPAlist = []
    AMPA_gmax = 0.6 / 1000.  # Set in nS and convert to muS

    model.NMDAlist = []
    model.ncNMDAlist = []
    NMDA_gmax = 0.8 / 1000.  # Set in nS and convert to muS

    for loc in Elocs:
        locInd = int(loc[0])
        if (locInd == -1):
            synloc = model.soma
        else:
            synloc = model.dends[int(loc[0])]
            synpos = float(loc[1])

        AMPA = h.Exp2Syn(synpos, sec=synloc)
        AMPA.tau1 = 0.1
        AMPA.tau2 = 1
        NC = h.NetCon(h.nil, AMPA, 0, 0, AMPA_gmax)  # NetCon(source, target, threshold, delay, weight)
        model.AMPAlist.append(AMPA)
        model.ncAMPAlist.append(NC)

        NMDA = h.Exp2SynNMDA(synpos, sec=synloc)
        NMDA.tau1 = 2
        NMDA.tau2 = 50
        NC = h.NetCon(h.nil, NMDA, 0, 0, NMDA_gmax)
        model.NMDAlist.append(NMDA)
        model.ncNMDAlist.append(NC)


def genDendLocs(stimulated_dend: int, nsyn: int, spread: list[float] = [0.4, 0.6]) -> list[list[int, float]]:
    """
    Generate dendritic locations for synapses.

    This function generates a list of dendritic locations for synaptic stimulation. Synapses
    are placed uniformly on a specific dendrite within a defined range (spread). Each location
    includes the dendrite identifier and the relative position along the dendrite.

    Parameters:
    stimulated_dend: int
        Identifier of the dendrite where synapses will be placed.
    nsyn: int
        The number of synapses to be placed on the dendrite.
    spread: list[float]
        A list with two float values defining the range for synapse placement along
        the dendrite, given as [start, end]. The default range is [0.4, 0.6].

    Returns:
    list[list[int, float]]
        A list of locations, where each location is represented as a list containing
        the dendrite identifier (int) and the relative position along the dendrite (float).
    """
    locs = []

    isd = (spread[1]-spread[0])/float(nsyn)
    pos = np.arange(spread[0], spread[1], isd)[0:nsyn]

    if (len(pos) != nsyn):
        # print ('error: synapse number mismatch, stop simulation! dend:', i_dend, 'created=', len(pos), '!=', nsyn_dend)
        sys.exit(1)
    for p in pos:
        locs.append([stimulated_dend, p])
    return locs
