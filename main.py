from CurrentscapePipeline import CurrentscapePipeline

output_dir = 'output'
target = 'soma'  # can be any dendrite too e.g. 'dend5_0'
partitioning = 'type'  # can be 'type' or 'region'
ca = False
stim_dend = 108
direction = 'IN'  # can be 'IN' or 'OUT'
tstop = 380
tmin = 280
tmax = 380
nsyn = 8  # nsyn values used in the article: 8, 10, 15, 20
t_interval = 0.3
onset = 300
currentscape_filename = f'currentscape_Fig3C_ca{ca}_{partitioning}_{nsyn}.pdf'


pipeline = CurrentscapePipeline(output_dir, target, partitioning, ca, stim_dend, direction, tstop, tmin, tmax, nsyn,
                                t_interval, onset, currentscape_filename)

if pipeline.results_exist():
    print("Results found. Loading and visualizing only...")
    pipeline.run_simulation()
    pipeline.load_results()
    pipeline.visualize()
else:
    print("Results not found. Running full pipeline...")
    pipeline.run_full_pipeline()
