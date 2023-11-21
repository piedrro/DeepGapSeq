import copy
import traceback
import numpy as np
from functools import partial
from DeepGapSeq.GUI.gui_worker import Worker
from DeepGapSeq.DeepLASI.wrapper import DeepLasiWrapper

class _DeepLasi_methods:

    def populate_deeplasi_options(self):

        try:

            if self.data_dict != {}:

                self.fitting_window.deeplasi_fit_dataset.clear()

                dataset_names = list(self.data_dict.keys())

                self.fitting_window.deeplasi_fit_dataset.clear()
                self.fitting_window.deeplasi_fit_dataset.addItems(dataset_names)

                self.fitting_window.deeplasi_fit_data.clear()

                plot_names = []

                for dataset_name in self.data_dict.keys():
                    for plot_name, plot_value in self.data_dict[dataset_name][0].items():
                        if plot_name in ["donor", "acceptor", "efficiency", "DD", "AA", "DA", "AD"]:
                            if len(plot_value) > 0:
                                plot_names.append(plot_name)

                if "donor" in plot_names:
                    self.fitting_window.deeplasi_fit_data.addItem("Donor")
                if "acceptor" in plot_names:
                    self.fitting_window.deeplasi_fit_data.addItem("Acceptor")
                if set(["donor", "acceptor"]).issubset(plot_names):
                    self.fitting_window.deeplasi_fit_data.addItem("FRET Data")
                if set(["donor", "acceptor", "efficiency"]).issubset(plot_names):
                    self.fitting_window.deeplasi_fit_data.addItem("FRET Efficiency")
                if set(["DD", "AA", "DA", "AD"]).issubset(plot_names):
                    self.fitting_window.deeplasi_fit_data.addItem("DD")
                    self.fitting_window.deeplasi_fit_data.addItem("AA")
                    self.fitting_window.deeplasi_fit_data.addItem("DA")
                    self.fitting_window.deeplasi_fit_data.addItem("AD")
                if set(["DD", "AA", "DA", "AD", "efficiency"]).issubset(plot_names):
                    self.fitting_window.deeplasi_fit_data.addItem("ALEX Efficiency")

        except:
            print(traceback.format_exc())

    def build_deeplasi_dataset(self):

        deeplasi_dataset = []
        n_colors = None

        try:
            if self.data_dict != {}:

                dataset_name = self.fitting_window.deeplasi_fit_dataset.currentText()

                data_name = self.fitting_window.deeplasi_fit_data.currentText()
                crop_plots = self.fitting_window.deeplasi_crop_plots.isChecked()

                for localisation_index, localisation_data in enumerate(self.data_dict[dataset_name]):

                    if data_name == "Donor":
                        data = localisation_data["donor"]
                        n_colors=1
                    elif data_name == "Acceptor":
                        data = localisation_data["acceptor"]
                        n_colors = 1
                    elif data_name == "FRET Data":
                        data = np.array([localisation_data["donor"], localisation_data["acceptor"]])
                        n_colors = 2
                    elif "efficiency" in data_name.lower():
                        data = localisation_data["efficiency"]
                        n_colors = 1
                    elif data_name == "DD":
                        data = localisation_data["DD"]
                        n_colors = 1
                    elif data_name == "AA":
                        data = localisation_data["AA"]
                        n_colors = 1
                    elif data_name == "DA":
                        data = localisation_data["DA"]
                        n_colors = 1
                    elif data_name == "AD":
                        data = localisation_data["AD"]
                        n_colors = 1

                    crop_range = localisation_data["crop_range"]

                    if len(data.shape) == 1:
                        data = np.expand_dims(data, axis=0)

                    if crop_plots == True and len(crop_range) == 2:
                        crop_range = sorted(crop_range)
                        data = np.array(data)
                        data = data[:, crop_range[0]:crop_range[1]]

                    data = np.array(data).T

                    deeplasi_dataset.append(data)

        except:
            print(traceback.format_exc())
            pass

        return deeplasi_dataset, n_colors


    def _detect_deeplasi_states(self, progress_callback = None, deeplasi_dataset = []):

        detected_states = []

        try:

            deeplasi_dataset, n_colors = self.build_deeplasi_dataset()

            n_states = int(self.fitting_window.deeplasi_n_states.currentText())
            crop_data = self.fitting_window.deeplasi_crop_plots.isChecked()

            detected_states = []
            confidence_list = []

            wrapper = DeepLasiWrapper(n_colors=n_colors, n_states=n_states)

            wrapper.initialise_model()

            if crop_data:

                for index, data in enumerate(deeplasi_dataset):

                    state, confidence = wrapper.predict_states([data], n_colors=n_colors, n_states=n_states)

                    detected_states.append(state[0])
                    confidence_list.append(confidence[0])

            else:

                detected_states, confidence_list = wrapper.predict_states(deeplasi_dataset, n_colors=n_colors, n_states=n_states)

        except:
            print(traceback.format_exc())

        self.deeplasi_detected_states = detected_states


    def _detect_deeplasi_states_cleanup(self, detected_states=[]):

        try:

            dataset_name = self.fitting_window.deeplasi_fit_dataset.currentText()

            for localisation_number, state in enumerate(self.deeplasi_detected_states):

                localisation_data = self.data_dict[dataset_name][localisation_number]

                localisation_data["states"] = np.array(state).astype(int)

                # if localisation_number == 17:
                #     print(state)

                self.data_dict[dataset_name][localisation_number] = copy.deepcopy(localisation_data)

            self.compute_state_means(dataset_name=dataset_name)
            self.plot_traces(update_plot=True)

        except:
            print(traceback.format_exc())


    def detect_deeplasi_states(self):

        if self.data_dict != {}:

            worker = Worker(self._detect_deeplasi_states)
            worker.signals.finished.connect(self._detect_deeplasi_states_cleanup)
            self.threadpool.start(worker)


