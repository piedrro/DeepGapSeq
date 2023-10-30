import os
from DeepGapSeq.simulation.deepgapseq_trace_generator import trace_generator

desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

generator = trace_generator(n_colors=1, 
                            n_states=2,
                            n_frames=500,
                            n_traces=100,
                            outdir=desktop,
                            clear_outdir=True,
                            export_name = "trace_dataset_example",
                            export_mode="ebfret",
)

training_data, training_labels = generator.generate_traces()