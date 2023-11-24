
import pickle

import numpy as np
from DeepGapSeq.simulation import training_data_1color, training_data_2color
from time import time
import os
import shutil
import datetime
import random
import string


from DeepGapSeq.DeepFretSimulate.lib.math import generate_traces





class trace_generator():
    
    def __init__(self,
                 n_traces = 100,
                 n_frames = 500,
                 n_colors = 2,
                 n_states = 2,
                 outdir = "",
                 export_mode = "",
                 export_name = "",
                 ):
        
        self.n_traces = n_traces
        self.n_frames = n_frames
        self.n_colors = n_colors
        self.n_states = n_states
        self.outdir = outdir
        self.export_mode = export_mode
        self.export_name = export_name
        
        self.check_outdir()

        assert n_colors in [1,2], "available colours: 1, 2"
        assert export_mode in ["","text_files", "pickledict","ebfret", "ebFRET_files"], "available export modes: '','text_files', 'pickledict', 'ebfret', 'ebFRET_files'"
        
    def check_outdir(self, overwrite=True, folder_name = "simulated_traces"):
    
        if os.path.exists(self.outdir) == False:
            self.outdir = os.getcwd()
        
        if folder_name != "":
            self.outdir = os.path.join(self.outdir, "deepgapseq_simulated_traces")
            
        if overwrite and os.path.exists(self.outdir):
                shutil.rmtree(self.outdir)

        if os.path.exists(self.outdir) == False:
            os.mkdir(self.outdir)


    def generate_traces(self):
        
        print("Generating traces...")
        start = time()
        
        df, matrices = generate_traces(
                    n_traces=self.n_traces,
                    trace_length=self.n_frames,
                    state_means="random",
                    random_k_states_max=self.n_states,
                    discard_unbleached=False,
                    return_matrix=True,
                    reduce_memory=False,
                    run_headless_parallel=False,
                    merge_labels=True,
                    merge_state_labels=False,
        )
        
        stop = time()
        duration = stop - start
        
        len_traces = len(df.name.unique())
        
        print(f"Spent {duration:.1f} s to generate {len_traces} traces")
        
        print(np.unique(df.label))
        
        df.columns = ['DD', 'DA', 'AA', 'DD-bg', 'DA-bg',
               'AA-bg', 'E', 'E_true', 'S', 'frame', 'name', 'label',
               '_bleaches_at', '_noise_level', '_min_state_diff', '_max_n_classes']
        
        trace_dict = []
        
        for name, data in df.groupby("name"):
            
            trace_dict.append(data.to_dict(orient="list"))
            
        return trace_dict
            
            

        


        


        
generator = trace_generator(n_colors=2,
                            n_states=2,
                            n_frames=500,
                            n_traces=100)


trace_dict = generator.generate_traces()



# simulated_dataset = []

# print(df.keys())

# n_colurs

# for name, data in df.groupby("name"):
    
#     pass
    
    
    