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
import threading
import time

class ebFRET_controller:

    def __init__(self,
                 ebfret_dir: str = os.path.dirname(os.path.realpath(__file__)),
                 num_workers: int = 2,):

        self.engine = None
        atexit.register(self.cleanup)  # Register cleanup method to be called on exit

        self.ebfret_dir = ebfret_dir
        self.matlab_installed = False

        self.ebfret_handle = None
        self.ebfret_running = False

        self.num_workers = num_workers

        self.ebfret_dir_status = self.check_ebfret_dir()
        self.matlab_engine_status = self.check_matlab_engine_installed()

        self.lock = threading.Lock()

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

    def start_ebfret(self, threaded = True):

        if self.engine == None:
            self.start_engine()

        if self.engine:
            self.engine.cd(self.ebfret_dir, nargout=0)

            self.engine.eval("addpath(genpath('" + self.ebfret_dir + "'))", nargout=0)
            self.engine.addpath(self.engine.genpath("\python"), nargout=0)

            self.ebfret_handle = self.engine.ebFRET()

            if threaded == True:

                while True:
                    with self.lock:
                        try:
                            self.ebfret_running = self.check_ebfret_running()
                            if not self.ebfret_running:
                                self.close_ebfret()
                                self.close_engine()
                                print("ebFRET GUI has been closed!")
                                break
                            time.sleep(1)
                        except:
                            self.close_ebfret()
                            self.close_engine()
                            print("Error checking ebFRET GUI state. Assuming it's closed.")
                            print(traceback.format_exc())
                            break

    def check_ebfret_running(self):

        ebfret_running = False
        if self.engine and self.ebfret_handle:
            try:
                ebfret_running = self.engine.isvalid(self.ebfret_handle)
            except:
                print("ebFRET closed")

        return ebfret_running

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

            if self.engine and self.ebfret_handle:
                if check_data_format(data):
                    self.engine.ebfret.python.python_load_data(self.ebfret_handle, file_name, data, nargout=0)
        except:
            self.stop_parrallel_pool()
            self.close_ebfret()
            self.close_engine()
            print(traceback.format_exc())

    def run_ebfret_analysis(self, min_states=2, max_states=6):
        try:
            self.ebfret_states = []

            if self.engine and self.ebfret_handle:
                self.engine.ebfret.python.python_run_ebayes(self.ebfret_handle, min_states, max_states, nargout=0)
                self.ebfret_states = self.engine.ebfret.python.python_export_traces(self.ebfret_handle, min_states, max_states)

                self.ebfret_states = np.array(self.ebfret_states)

        except:
            self.close_engine()

        return self.ebfret_states

    def close_ebfret(self):
        if self.engine and self.ebfret_handle:
            self.engine.ebfret.python.python_close_ebfret(self.ebfret_handle, nargout=0)
            self.ebfret_handle = None

    def close_engine(self):
        if self.engine:
            try:
                if self.ebfret_handle:
                    self.close_ebfret()
                self.engine.quit()
                self.engine = None
                print("MATLAB engine closed")
            except Exception as e:
                print(f"Error closing MATLAB engine: {e}")

    def cleanup(self):
        self.close_ebfret()
        self.close_engine()

def launch_ebfret_instance():

    controller = ebFRET_controller()
    controller.start_ebfret()

    return controller


