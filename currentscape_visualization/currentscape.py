import numpy as np
import altair as alt
from currentscape_visualization.utils import *


def plot_currentscape(part_pos, part_neg, vm, taxis, tmin, tmax, filename, return_segs=False, segments_preselected=True,
                      vmin=-69, vmax=-65, partitionby='type'):
    if (segments_preselected):
        part_pos_seg = part_pos
        part_neg_seg = part_neg
        t_seg = taxis
        vm_seg = vm
    else:
        segment_indexes = np.flatnonzero((taxis > tmin) & (taxis < tmax))
        part_pos_seg = part_pos.iloc[:, segment_indexes]
        part_neg_seg = part_neg.iloc[:, segment_indexes]
        t_seg = taxis[segment_indexes]
        vm_seg = vm[segment_indexes]

    # Create charts
    totalpos = create_currsum_pos_chart(part_pos_seg, t_seg)
    currshares_pos, currshares_neg = create_currshares_chart(part_pos_seg, part_neg_seg, t_seg, partitionby)
    vm_chart = create_vm_chart(vm_seg, t_seg, vmin, vmax)

    # Create currentscape
    currentscape = combine_charts(vm_chart, totalpos, currshares_pos, currshares_neg)
    currentscape.save(filename)
    if (return_segs):
        return [part_neg_seg, part_pos_seg, vm_seg]