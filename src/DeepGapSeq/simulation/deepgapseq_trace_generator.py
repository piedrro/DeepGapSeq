import pickle

import numpy as np
from DeepGapSeq.simulation import training_data_1color, training_data_2color
from time import time
import os
import shutil
import datetime

class trace_generator():
    
    def __init__(self,
                 n_traces = 100,
                 n_frames = 500,
                 n_colors = 2,
                 n_states = 2,
                 balance_classes = False,
                 reduce_memory = True,
                 mode = "state_mode",
                 parallel_asynchronous = False,
                 outdir = "",
                 export_mode = "text_files",
                 export_name = "trace_dataset",
                 ):
        
        """
        Simulation scripts are inspired by the DeepLASI implementation of DeepFRET simulation scripts.
    
        n_traces: Number of traces
        n_timesteps: Number of frames per trace
        n_colors: Number of colors (1-color, 2-color or 3-color data possible)
        balance_classes: Balance classes based on minimum number of labeled frames
        reduce_memory: Include/exclude trace parameters beside countrates
        state_mode: Label dynamic traces according to state occupancy, used for training state classifiers
        n_states_model: Label each trace according to number of observed traces, used for number of states classifier
        parallel_asynchronous: parallel processing (faster)
        outdir: Output directory
        export_mode: export mode, more modes will be added over time
        """
        
        self.n_traces = n_traces
        self.n_frames = n_frames
        self.n_colors = n_colors
        self.n_states = n_states
        self.balance_classes = balance_classes
        self.reduce_memory = reduce_memory
        self.mode = mode
        self.parallel_asynchronous = parallel_asynchronous
        self.outdir = outdir
        self.export_mode = export_mode
        self.export_name = export_name
        
        self.check_mode()
        self.check_outdir()

        assert n_colors in [1,2], "available colours: 1, 2"
        assert export_mode in ["text_files", "pickledict"], "available export modes: 'text_files', 'pickledict'"
        
    def check_outdir(self, overwrite=True, folder_name = "simulated_traces"):
    
        if os.path.exists(self.outdir) == False:
            self.outdir = os.getcwd()
        
        if folder_name != "":
            self.outdir = os.path.join(self.outdir, "deepgapseq_simulated_traces")
            
        if overwrite and os.path.exists(self.outdir):
                shutil.rmtree(self.outdir)

        if os.path.exists(self.outdir) == False:
            os.mkdir(self.outdir)

        
    def check_mode(self):
        
        assert self.mode in ["state_mode", "n_states_mode"], "available modes: 'state_mode', 'n_states_mode'"
        
        if self.mode == "state_mode":
            self.state_mode = True
            self.n_states_mode = False
        else:
            self.state_mode = False
            self.n_states_mode = True
    
    def generate_single_colour_traces(self):
        
        traces = training_data_1color.simulate_1color_traces(
            n_traces=int(self.n_traces),
            max_n_states=self.n_states,
            n_frames=self.n_frames,
            state_mode=self.state_mode,
            n_states_mode=self.n_states_mode,
            reduce_memory=self.reduce_memory,
            parallel_asynchronous=self.parallel_asynchronous
        )
        
        training_data = []
        training_labels = []
        
        for trace in traces:
            
            training_labels.append(trace["label"].values)
            
            if self.reduce_memory:
                training_data.append(trace[["DD"]].values)
            else:
                training_data.append(trace[["DD", "DA", 
                                            "E", "E_true", 
                                            "label", "_noise_level", 
                                            "_min_E_diff", "trans_mean"]].values)
                
        return training_data, training_labels
    
    def generate_two_colour_traces(self):
        
        traces = training_data_2color.simulate_2color_traces(
            n_traces=int(self.n_traces),
            max_n_states=self.n_states,
            n_frames=self.n_frames,
            state_mode=self.state_mode,
            n_states_mode=self.n_states_mode,
            reduce_memory=self.reduce_memory,
            parallel_asynchronous=self.parallel_asynchronous,
        )
        
        training_data = []
        training_labels = []
        
        for trace in traces:
            
            training_labels.append(trace["label"].values)
            
            if self.reduce_memory:
                
                if self.state_mode or self.n_states_mode:
                    training_data.append(trace[["DD", "DA"]].values)
                else:
                    training_data.append(trace[["DD", "DA", "AA"]].values)
                    
            else:
                training_data.append(trace[["DD", "DA", 
                                            "AA", "E", 
                                            "E_true", "label", 
                                            "_noise_level", 
                                            "_min_E_diff", "trans_mean"]].values)
                
        return training_data, training_labels
        
    def export_traces(self, training_data, training_labels):

        date = datetime.datetime.now().strftime("%Y_%m_%d")
        
        if self.export_mode == "text_files":
            
            print(f"exporting txt files to: {self.outdir}")
            
            for index, (data, label) in enumerate(zip(training_data, training_labels)):
                
                label = np.expand_dims(label, 1)
            
                dat = np.hstack([data, label])
                
                file_path = os.path.join(self.outdir, f"{self.export_name}_{date}_{index}.csv")
                
                np.savetxt(file_path, dat, delimiter=",")

        if self.export_mode == "pickledict":

            trace_dictionary = {"data": [], "labels": [], "simulation_parameters": {}}

            trace_dictionary["simulation_parameters"]["n_traces"] = self.n_traces
            trace_dictionary["simulation_parameters"]["n_frames"] = self.n_frames
            trace_dictionary["simulation_parameters"]["n_colors"] = self.n_colors
            trace_dictionary["simulation_parameters"]["n_states"] = self.n_states
            trace_dictionary["simulation_parameters"]["balance_classes"] = self.balance_classes
            trace_dictionary["simulation_parameters"]["reduce_memory"] = self.reduce_memory
            trace_dictionary["simulation_parameters"]["mode"] = self.mode
            trace_dictionary["simulation_parameters"]["parallel_asynchronous"] = self.parallel_asynchronous

            for index, (data, label) in enumerate(zip(training_data, training_labels)):

                trace_dictionary["data"].append(data)
                trace_dictionary["labels"].append(label)



            file_path = os.path.join(self.outdir, f"{self.export_name}_{date}.pkl")

            with open(file_path, 'wb') as handle:
                pickle.dump(trace_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"exporting pickled dictionary to: {file_path}")

    def generate_traces(self):
        
        print("Generating traces...")
        start = time()
        
        if self.n_colors == 1 and not self.state_mode:
            
            training_data, training_labels = self.generate_single_colour_traces()
                
        elif self.n_colors == 2 or (self.n_colors == 1 and self.state_mode):
            
            training_data, training_labels = self.generate_two_colour_traces()
            
        stop = time()
        duration = stop - start
        
        unique_labels = np.unique(np.concatenate(training_labels))
        
        print(f"Spent {duration:.1f} s to generate {len(training_data)} traces")
        print("Labels: ", unique_labels)
        
        self.export_traces(training_data, training_labels)
        
        return training_data, training_labels
        



