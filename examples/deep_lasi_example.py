

from DeepGapSeq.simulation.deepgapseq_trace_generator import trace_generator
from DeepGapSeq.DeepLASI.wrapper import DeepLasiWrapper
import os


n_colors = 1
n_states = 2
n_frames = 500
n_traces = 100

desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

generator = trace_generator(
    n_colors=n_colors,
    n_states=n_states,
    n_frames=n_frames,
    n_traces=n_traces,
    outdir=desktop,
    export_name="trace_dataset_example",
    export_mode="pickledict")

traces, labels = generator.generate_traces()

deeplasi = evaluator = DeepLasiWrapper()

# an example of state prediction
states, confidence = deeplasi.predict_states(traces, n_colors=n_colors, n_states=n_states)

# an example of n-state prediction
n_states_list, confidence_list = evaluator.predict_n_states(traces, n_colors=n_colors)




