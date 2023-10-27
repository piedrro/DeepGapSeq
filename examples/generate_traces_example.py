
from DeepBacSeg.simulation.deepgapseq_trace_generator import trace_generator

generator = trace_generator(n_colors=1, 
                            n_states=2,
                            n_frames=500,
                            n_traces=100,
                            export_mode="text_files")

training_data, training_labels = generator.generate_traces()