import os
import sys
import subprocess
import atexit
import platform
import scipy
import numpy as np
import traceback
from DeepGapSeq._utils_worker import Worker
from functools import partial


class ebFRET_controller():

    def __init__(self,
                 ebfret_dir: str = "",
                 num_workers: int = 2,):

        self.engine = None
        atexit.register(self.cleanup)  # Register cleanup method to be called on exit

        self.ebfret_dir = ebfret_dir
        self.matlab_installed = False
        self.ebfret_instance = None
        self.ebfret_running = False
        self.num_workers = num_workers

    def check_ebfret_dir(self):

        directory_status = False

        if os.path.exists(self.ebfret_dir):
            for root, dir, files in os.walk(self.ebfret_dir):
                if "ebFRET.m" in files:
                    directory_status = True
            if directory_status == True:
                print("ebFRET directory found: " + self.ebfret_dir)
            else:
                print("ebFRET directory does not contain ebFRET.m: " + self.ebfret_dir)
        else:
            print("ebFRET directory not exist: " + self.ebfret_dir)

        return directory_status

    def check_matlab_engine_installed(self):
        try:
            import matlab.engine
            print("MATLAB engine API for Python is installed.")
            self.matlab_installed = True
            return True
        except ImportError:
            print("MATLAB engine API for Python is not installed.")
            return False

    def check_matlab_running(self):
        try:
            if platform.system() == "Windows":
                procs = subprocess.check_output("tasklist").decode("utf-8")
                return "MATLAB.exe" in procs
            else:  # Linux and macOS
                procs = subprocess.check_output(["ps", "aux"]).decode("utf-8")
                return "matlab" in procs.lower()
        except Exception as e:
            print(f"Error checking if MATLAB is running: {e}")
            return False

    def start_engine(self):

        if self.matlab_installed == True:
            try:
                import matlab.engine

                matlab_session = matlab.engine.find_matlab()

                if len(matlab_session) > 0:
                    try:
                        self.engine = matlab.engine.start_matlab(matlab_session[0])
                        print("Connected to existing MATLAB engine")
                    except:
                        self.engine = matlab.engine.start_matlab()
                        print("MATLAB engine started")

                else:
                    self.engine = matlab.engine.start_matlab()
                    print("MATLAB engine started")

                return True
            except Exception as e:
                print(f"Error starting MATLAB engine: {e}")
                self.close_engine()
                return False

    def start_parrallel_pool(self):

        try:

            if self.engine:
                self.engine.parpool('local', self.num_workers, nargout=0)
        except:
            self.close_engine()

    def stop_parrallel_pool(self):

        try:
            if self.engine:
                print("Stopping MATLAB parallel pool")
                self.engine.eval("poolobj = gcp('nocreate');", nargout=0)
                self.engine.eval("if ~isempty(poolobj), delete(poolobj); end", nargout=0)
                print("MATLAB parallel pool stopped")
        except:
            self.close_engine()

    def start_ebfret(self):

        if self.engine == None:
            self.start_engine()

        if self.engine:
            self.engine.cd(self.ebfret_dir, nargout=0)

            self.engine.eval("addpath(genpath('" + self.ebfret_dir + "'))", nargout=0)
            self.engine.addpath(self.engine.genpath("\python"), nargout=0)

            self.ebfret_instance = self.engine.ebFRET()

    def check_ebfret_running(self):
        self.ebfret_running = False
        if self.engine and self.ebfret_instance:
            self.ebfret_running = self.engine.ebfret.python.python_check_running(self.ebfret_instance)
        return self.ebfret_running

    def load_fret_data(self, data=[], file_name="temp.tif"):
        try:
            def check_data_format(input_list, min_length=5):
                if not isinstance(input_list, list):  # Check if the input is a list
                    return False
                for sublist in input_list:
                    if not (isinstance(sublist, list) and len(sublist) >= min_length):  # Check if each element is a list with a length of at least 5
                        return False
                return True

            # cast all values to floats
            data = [[float(y) for y in x] for x in data]

            data_min = np.min(data)
            data_max = np.max(data)
            data_shape = np.shape(data)

            # print(f"min: {data_min}, max: {data_max}, shape: {data_shape}")

            if self.engine and self.ebfret_instance:
                if check_data_format(data):
                    self.engine.ebfret.python.python_load_data(self.ebfret_instance, file_name, data, nargout=0)
        except:
            self.stop_parrallel_pool()
            self.close_ebfret()
            self.close_engine()
            print(traceback.format_exc())

    def run_ebfret_analysis(self, min_states=2, max_states=6):
        try:
            self.ebfret_states = []

            if self.engine and self.ebfret_instance:
                self.engine.ebfret.python.python_run_ebayes(self.ebfret_instance, min_states, max_states, nargout=0)
                self.ebfret_states = self.engine.ebfret.python.python_export_traces(self.ebfret_instance, min_states, max_states)

                self.ebfret_states = np.array(self.ebfret_states)

        except:
            self.close_engine()

        return self.ebfret_states



    def close_ebfret(self):
        if self.engine and self.ebfret_instance:
            self.engine.ebfret.python.python_close_ebfret(self.ebfret_instance, nargout=0)
            self.ebfret_instance = None

    def close_engine(self):
        if self.engine:
            try:
                if self.ebfret_instance:
                    self.close_ebfret()
                self.engine.quit()
                self.engine = None
                print("MATLAB engine closed")
            except Exception as e:
                print(f"Error closing MATLAB engine: {e}")

    def cleanup(self):
        print("Cleanup method called due to an error or normal termination.")
        if self.ebfret_instance:
            self.close_ebfret()
        self.close_engine()  # Close MATLAB engine if it is active


def gapseq_visualise_ebfret(self):

    try:

        if hasattr(self, "ebfret_states"):

            state = self.ebfret_visualisation_state.currentText()

            if state.isdigit():
                state = int(state)

                state_data = self.ebfret_states.copy()
                state_data_loc_num = self.ebfret_data_loc_number

                indices = np.where(state_data[:,0] == state)
                state_data = np.take(state_data, indices, axis=0)[0]

                meta = self.meta.copy()
                precomputed_trace_graph_data = self.precomputed_trace_graph_data.copy()

                localisation_data = meta["localisation_data"]

                plot_localisation_number = int(self.plot_localisation_number.value())

                layer_names = localisation_data.keys()

                for localisation_number in state_data_loc_num:

                    localisation_indices = np.where(state_data[:, 1] == localisation_number+1)
                    localisation_data = np.take(state_data, localisation_indices, axis=0)[0]
                    hmm_fit = localisation_data[:,2]

                    breakpoints = []
                    for index in range(len(hmm_fit) - 1):
                        if hmm_fit[index] != hmm_fit[index + 1]:
                            breakpoints.append(index)

                    breakpoints = [0] + breakpoints + [len(hmm_fit)-1]

                    for layer in layer_names:

                        self.meta["bounding_box_breakpoints"][layer][localisation_number] = breakpoints
                        self.meta["bounding_box_hmm_states"][layer][localisation_number] = hmm_fit

                    self.precompute_trace_graph_data(localisation_number=localisation_number)

                self.initialise_draw_trace_graph()

    except:
        print(traceback.format_exc())






def _run_ebfet_analysis(self, progress_callback = None):

    try:

        layer = self.ebfret_fit_dataset.currentText()

        channel_name = self.ebfret_fit_channel.currentText()
        channel_index = self.ebfret_fit_channel.currentIndex()

        metric_index = self.plot_metric.currentIndex()
        background_mode = self.plot_background_subtraction_mode.currentIndex()
        alex_pulse_duration = self.alex_pulse_duration.value()
        alex_firstframe_excitation = self.alex_firstframe_excitation.currentIndex()

        ebfret_min_states = self.ebfret_min_states.currentText()
        ebfret_max_states = self.ebfret_max_states.currentText()

        if ebfret_min_states.isdigit():
            ebfret_min_states = int(ebfret_min_states)
        else:
            ebfret_min_states = 2
        if ebfret_max_states.isdigit():
            ebfret_max_states = int(ebfret_max_states)
        else:
            ebfret_max_states = 6

        if ebfret_min_states > ebfret_max_states:
            ebfret_min_states == ebfret_max_states

        ebfret_data = []
        ebfret_data_loc_number = []

        box_num = len(self.box_layer.data.copy())
        meta = self.meta.copy()

        localisation_data = meta["localisation_data"]

        if layer in localisation_data.keys():

            for localisation_number in range(box_num):

                if "ALEX" not in channel_name:
                    trace_data = self.get_gapseq_trace_data(
                        layer,
                        channel_name,
                        localisation_number,
                        metric_index,
                        background_mode)

                else:
                    trace_data = self.get_alex_trace_data(
                        layer,
                        channel_name,
                        localisation_number,
                        metric_index,
                        background_mode,
                        alex_pulse_duration,
                        alex_firstframe_excitation)

                if "plot_data" in trace_data.keys():
                    if len(trace_data["plot_data"]) == 1:
                        ebfret_data.extend([trace_data["plot_data"][0]])
                    else:
                        ebfret_data.extend([trace_data["plot_data"][-1]])

                    ebfret_data_loc_number.append(localisation_number)

        if len(ebfret_data) > 0:
            self.ebFRET_controller.load_fret_data(ebfret_data)
            self.ebfret_states = self.ebFRET_controller.run_ebfret_analysis(ebfret_min_states, ebfret_max_states)
            self.ebfret_data_loc_number = ebfret_data_loc_number

            if len(self.ebfret_states) > 0:
                unique_states = np.unique(self.ebfret_states[:,0])
                self.ebfret_visualisation_state.clear()
                self.ebfret_visualisation_state.addItems([str(int(x)) for x in unique_states])

    except:
        print(traceback.format_exc())
        pass


def _disconnect_matlab(self, progress_callback=None):

    try:
        self.ebFRET_controller.close_ebfret()
        progress_callback.emit(33)
        self.ebFRET_controller.stop_parrallel_pool()
        progress_callback.emit(66)
        self.ebFRET_controller.close_engine()
        progress_callback.emit(100)
        progress_callback.emit(0)

    except:
        pass

def _connect_matlab(self, progress_callback=None):

    ebFRET_controller = None

    try:
        import importlib.resources

        gapseq_directory = os.path.dirname(os.path.realpath(__file__))

        # print(f"GapSeq install directory: {gapseq_directory}")

        if hasattr(self, "ebfret_path") == False:
            self.ebfret_path = os.path.join(gapseq_directory)

        if os.path.exists(self.ebfret_path) == True:

            progress_callback.emit(5)

            ebFRET_controller = ebFRET_controller(ebfret_dir=self.ebfret_path)
            progress_callback.emit(10)

            ebfret_dir_status = ebFRET_controller.check_ebfret_dir()
            progress_callback.emit(15)

            matlab_engine_status = ebFRET_controller.check_matlab_engine_installed()
            progress_callback.emit(20)

            if ebfret_dir_status and matlab_engine_status:
                ebFRET_controller.start_engine()
                progress_callback.emit(40)
                # ebFRET_controller.start_parrallel_pool()
                # progress_callback.emit(70)
                ebFRET_controller.start_ebfret()
                progress_callback.emit(100)
                progress_callback.emit(0)

        else:
            print("ebFRET directory does not exist: ", self.ebfret_path)
            progress_callback.emit(0)

    except:
        print(traceback.format_exc())
        progress_callback.emit(0)

    return ebFRET_controller




def _connect_matlab_cleanup(self, ebFRET_controller = None):

    try:

        if ebFRET_controller != None:

            self.ebFRET_controller = ebFRET_controller
            if ebFRET_controller.check_ebfret_running():
                self.ebfret_connect_matlab.setText(r"Close MATLAB/ebFRET")
            else:
                self.ebfret_connect_matlab.setText(r"Open MATLAB/ebFRET")
        else:
            self.ebFRET_controller = None
            self.ebfret_connect_matlab.setText(r"Open MATLAB/ebFRET")
    except:
        print(traceback.format_exc())
        pass

def print_directory():

    directory = os.path.dirname(os.path.realpath(__file__))

    print(directory)



def check_ebfret_directory(path):

    directory_status = False

    if os.path.exists(path):
        for root, dir, files in os.walk(path):
            if "ebFRET.m" in files:
                directory_status = True
        if directory_status == True:
            print("ebFRET directory found: " + path)
        else:
            print("ebFRET directory does not contain ebFRET.m: " + path)
    else:
        print("ebFRET directory not exist: " + path)

    return directory_status



def gapseq_run_ebfret_analysis(self):
    if hasattr(self, "ebFRET_controller"):
        if self.ebFRET_controller is not None:
            worker = Worker(self._run_ebfet_analysis)
            worker.signals.result.connect(self._connect_matlab_cleanup)
            self.threadpool.start(worker)

def connect_matlab(self):

    try:

        launch_ebfret = True
        if hasattr(self, "ebFRET_controller"):

            if hasattr(self.ebFRET_controller, "check_ebfret_running"):
                if self.ebFRET_controller.check_ebfret_running():
                    launch_ebfret = False

        if launch_ebfret:
            print("launching MATLAB/ebFRET")
            worker = Worker(self._connect_matlab)
            worker.signals.result.connect(self._connect_matlab_cleanup)
            worker.signals.progress.connect(partial(self.gapseq_progressbar, progressbar="ebfret_connect_matlab"))
            self.threadpool.start(worker)
        else:
            print("closing MATLAB/ebFRET")
            worker = Worker(self._disconnect_matlab)
            worker.signals.result.connect(self._connect_matlab_cleanup)
            worker.signals.progress.connect(partial(self.gapseq_progressbar, progressbar="ebfret_connect_matlab"))
            self.threadpool.start(worker)

    except:
        print(traceback.format_exc())