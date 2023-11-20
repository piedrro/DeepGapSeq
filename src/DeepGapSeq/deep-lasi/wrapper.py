
import tensorflow as tf
import importlib
import importlib.resources as resources
import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import statistics
from DeepGapSeq.simulation.deepgapseq_trace_generator import trace_generator




class DeepLasiWrapper():

    def __init__(self,
            n_colors = 2,
            n_states = 2,
            model_type = "states",
            traces = []):
        
        
        self.n_colors = n_colors
        self.n_states = n_states
        self.model_type = model_type
        self.traces = traces
        
        self.check_gpu()

    def check_gpu(self):
        
        devices = tf.config.list_physical_devices('GPU')
        
        if len(devices) == 0:
            print("Tensorflow running on CPU")
            self.gpu = False
            self.device = None
        else:
            print("Tensorflow running on GPU")
            self.gpu = True
            self.device = devices[0]

    def check_data_format(self, traces):

        correct_format = True

        for index, arr in enumerate(traces):
            if not isinstance(arr, np.ndarray):  # Check if the element is a NumPy array
                correct_format = False
                break
            if len(arr.shape) == 1 and self.n_colors == 1:
                
                arr = np.expand_dims(arr, axis=-1)
                traces[index] = arr
                correct_format == True
            else:
                if arr.shape[-1] != self.n_colors:
                    correct_format = False
                    break

        if correct_format == False:
            print("input data must be a list of numpy arryas of shape (B,N,C)")

        return correct_format, traces

    def initialise_model(self, model_type="", n_colors="",n_states=""):
        
        if model_type != "":
            self.model_type = model_type
        if n_colors != "":
            self.n_colors = n_colors
        if n_states != "":
            self.n_states = n_states

        assert self.model_type in ["states","n_states","trace"], "model_type must be one of ['states','n_states','trace']"
        assert self.n_colors in [1,2,3], "n_colors must be one of [1,2,3]"
        assert self.n_states in [2,3,4], "n_states must be one of [2,3,4]"

        model_directory = resources.files(importlib.import_module(f'DeepGapSeq.deep-lasi'))
        
        if self.model_type == "states":
            model_name = "DeepLASI_{}color_{}state_classifier.h5".format(self.n_colors, self.n_states)
        elif self.model_type == "n_states":
            print(True)
            model_name = "DeepLASI_{}color_number_of_states_classifier.h5".format(self.n_colors)
        elif self.model_type == "trace":
            if self.n_colors == 1:
                model_name = "DeepLASI_1Color_trace_classifier.h5"
            elif self.n_colors == 2: 
                model_name = "DeepLASI_2color_nonALEX_trace_classifier.h5"
            elif self.n_colors == 3:
                model_name = "DeepLASI_3Color_trace_classifier.h5"
                
        model_path = os.path.join(model_directory,"models",model_name)
        
        if os.path.exists(model_path):
            print(f"loading model: {model_name}")
            
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = None
            print(f"model {model_name}, does not exist")

        return self.model


    def predict_states(self, traces = [], n_colors = None, n_states = None):
        
        if len(traces) > 0:
            self.traces = traces
        if n_colors in [1,2,3]:
            self.n_colors = n_colors
        if n_states in [2,3,4]:
            self.n_states = n_states

        states = []
        predictions = []

        correct_format, traces = self.check_data_format(traces)
        
        if correct_format:

            self.model = self.initialise_model()

            if self.model != None:

                traces = tf.convert_to_tensor(traces, dtype=tf.float32)

                predictions = self.model.predict(traces)

                states = np.argmax(predictions,axis=1)
                confidence = np.max(predictions,axis=1)

        return states, confidence
        
    def predict_n_states(self, traces = [], n_colors = None):

        if len(traces) > 0:
            self.traces = traces
        if n_colors in [1,2,3]:
            self.n_colors = n_colors

        correct_format, traces = self.check_data_format(traces)

        n_states_list = []
        confidence_list = []

        if correct_format:

            self.model = self.initialise_model(model_type="n_states", n_colors=self.n_colors)

            if self.model != None:

                traces = tf.convert_to_tensor(traces, dtype=tf.float32)

                predictions = self.model.predict(traces)

                for prediction in predictions:
                    n_states = statistics.mode(np.argmax(prediction,axis=1))
                    confidence = np.mean(prediction[:,n_states])

                    n_states_list.append(n_states+2)
                    confidence_list.append(confidence)
                
        return n_states_list, confidence_list
        
        
        
        
      



desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

generator = trace_generator(n_colors=1,
                            n_states=2,
                            n_frames=500,
                            n_traces=100,
                            outdir=desktop,
                            export_name = "trace_dataset_example",
                            export_mode="pickledict")

traces, labels = generator.generate_traces()        
      

evaluator = DeepLasiWrapper(n_colors=1)

n_states_list, confidence_list = evaluator.predict_n_states(traces)





# for trace, trace_prediction in zip(traces, predictions):
#
#     trace_prediction = np.argmax(trace_prediction,axis=1)
#
#     plt.plot(trace[:,0])
#     plt.plot(trace_prediction)
#     plt.show()





