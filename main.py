from CurrentscapePipeline import CurrentscapePipeline


pipeline = CurrentscapePipeline(
    output_dir='output',
    target='soma',
    partitioning='type',
    stim_dend=108,
    tmin=280,
    tmax=380,
    currentscape_filename='test.pdf',
    maxNsyn=30,
    nsyn=8,
    tInterval=0.3,
    onset=300,
    direction='IN',
    tstop=900
)
pipeline.run_full_pipeline()
