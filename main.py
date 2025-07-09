from CurrentscapePipeline import CurrentscapePipeline

output_dir = 'output'
target = 'soma'
partitioning = 'type'
ca = False
stim_dend = 108
direction = 'IN'
tstop = 380
tmin = 280
tmax = 380
nsyn = 8
t_interval = 0.3
onset = 300
currentscape_filename = f'currentscape_Fig3C_ca{ca}_{partitioning}_{nsyn}.pdf'


pipeline = CurrentscapePipeline(output_dir, target, partitioning, ca, stim_dend, direction, tstop, tmin, tmax, nsyn,
                                t_interval, onset, currentscape_filename)

pipeline.run_full_pipeline()
