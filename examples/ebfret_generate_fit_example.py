import os
from DeepGapSeq.simulation.deepgapseq_trace_generator import trace_generator
from DeepGapSeq.GUI.ebfret_utils import ebFRET_controller

desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

generator = trace_generator(n_colors=1,
                            n_states=2,
                            n_frames=500,
                            n_traces=100,
                            outdir=desktop,
                            export_name = "trace_dataset_example",
                            export_mode="ebfret")

training_data, training_labels = generator.generate_traces()


#  ebFRET controller will accept a list of numpy arrays in the shape (N,2), (N,1) or (N,)

controller = ebFRET_controller()

controller.start_ebfret()

controller.python_import_ebfret_data(training_data)

ebfret_states = controller.run_ebfret_analysis(min_states=2, max_states=2)

controller.close_ebfret()


