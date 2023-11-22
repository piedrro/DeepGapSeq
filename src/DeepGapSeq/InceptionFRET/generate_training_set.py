import os
from DeepGapSeq.simulation.deepgapseq_trace_generator import trace_generator
import importlib.resources as resources
import importlib


dataset_directory = resources.files(importlib.import_module(f'DeepGapSeq.InceptionFRET'))
dataset_directory = os.path.join(dataset_directory, 'dataset')

generator = trace_generator(n_colors=2,
                            n_states=2,
                            n_frames=500,
                            n_traces=100,
                            outdir=dataset_directory,
                            export_name = "dataset",
                            export_mode= "pickledict")



