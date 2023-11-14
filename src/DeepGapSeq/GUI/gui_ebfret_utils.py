import time
import traceback
from DeepGapSeq.GUI.gui_worker import Worker
import os





class _ebFRET_methods:

    def _launch_ebFRET(self, progress_callback=None):

        ebFRET_controller = None

        try:
            from DeepGapSeq.ebFRET.ebfret_utils import ebFRET_controller

            ebFRET_controller = ebFRET_controller()
            progress_callback.emit(10)

            ebfret_dir_status = ebFRET_controller.check_ebfret_dir()
            progress_callback.emit(15)

            matlab_engine_status = ebFRET_controller.check_matlab_engine_installed()
            progress_callback.emit(20)

            if ebfret_dir_status and matlab_engine_status:
                ebFRET_controller.start_engine()
                progress_callback.emit(40)
                ebFRET_controller.start_ebfret()
                progress_callback.emit(100)
                progress_callback.emit(0)

        except:
            print(traceback.format_exc())
            progress_callback.emit(0)

        return ebFRET_controller


    def _close_ebFRET(self, progress_callback=None):
        """Close ebFRET GUI."""

        try:

            if hasattr(self, "ebFRET_controller"):
                if self.ebFRET_controller != None:

                    self.ebFRET_controller.close_ebfret()
                    progress_callback.emit(33)
                    self.ebFRET_controller.stop_parrallel_pool()
                    progress_callback.emit(66)
                    self.ebFRET_controller.close_engine()
                    progress_callback.emit(100)
                    progress_callback.emit(0)
        except:
            pass

    def _launch_ebFRET_cleanup(self, ebFRET_controller=None):

        """Cleanup after launching ebFRET GUI."""
        try:
            if ebFRET_controller != None:
                self.ebFRET_controller = ebFRET_controller
                if ebFRET_controller.check_ebfret_running():
                    self.fitting_window.ebfret_connect_matlab.setText(r"Close MATLAB/ebFRET")
                else:
                    self.fitting_window.ebfret_connect_matlab.setText(r"Open MATLAB/ebFRET")
            else:
                self.ebFRET_controller = None
                self.fitting_window.ebfret_connect_matlab.setText(r"Open MATLAB/ebFRET")
        except:
            print(traceback.format_exc())
            pass

    def launch_ebFRET(self):

        launch_ebfret = True
        if hasattr(self, "ebFRET_controller"):

            if hasattr(self.ebFRET_controller, "check_ebfret_running"):
                if self.ebFRET_controller.check_ebfret_running():
                    launch_ebfret = False

        if launch_ebfret:
            print("launching MATLAB/ebFRET")
            worker = Worker(self._launch_ebFRET)
            worker.signals.result.connect(self._launch_ebFRET_cleanup)
            self.threadpool.start(worker)
        else:
            print("closing MATLAB/ebFRET")
            worker = Worker(self._close_ebFRET)
            worker.signals.result.connect(self._launch_ebFRET_cleanup)
            self.threadpool.start(worker)