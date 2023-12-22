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
                        if plot_name in ["Donor", "Acceptor", "FRET Efficiency", "ALEX Efficiency", "DD", "AA", "DA", "AD"]:
                            if len(plot_value) > 0:
                                plot_names.append(plot_name)

                if "Donor" in plot_names:
                    self.fitting_window.deeplasi_fit_data.addItem("Donor")
                if "Acceptor" in plot_names:
                    self.fitting_window.deeplasi_fit_data.addItem("Acceptor")
                if set(["Donor", "Acceptor"]).issubset(plot_names):
                    self.fitting_window.deeplasi_fit_data.addItem("FRET Data")
                if set(["Donor", "Acceptor", "FRET Efficiency"]).issubset(plot_names):
                    self.fitting_window.deeplasi_fit_data.addItem("FRET Efficiency")
                if set(["DD", "AA", "DA", "AD"]).issubset(plot_names):
                    self.fitting_window.deeplasi_fit_data.addItem("DD")
                    self.fitting_window.deeplasi_fit_data.addItem("AA")
                    self.fitting_window.deeplasi_fit_data.addItem("DA")
                    self.fitting_window.deeplasi_fit_data.addItem("AD")
                if set(["DD", "AA", "DA", "AD", "ALEX Efficiency"]).issubset(plot_names):
                    self.fitting_window.deeplasi_fit_data.addItem("ALEX Efficiency")

        except:
            print(traceback.format_exc())

    def build_deeplasi_dataset(self):

        deeplasi_dataset = {}
        n_colors = None

        try:
            if self.data_dict != {}:

                dataset_name = self.fitting_window.deeplasi_fit_dataset.currentText()

                data_name = self.fitting_window.deeplasi_fit_data.currentText()
                crop_plots = self.fitting_window.deeplasi_crop_plots.isChecked()

                for localisation_index, localisation_data in enumerate(self.data_dict[dataset_name]):

                    user_label = localisation_data["user_label"]
                    nucleotide_label = localisation_data["nucleotide_label"]
                    crop_range = localisation_data["crop_range"]

                    if self.get_filter_status("deeplasi", user_label, nucleotide_label) == False:

                        if data_name == "Donor":
                            data = localisation_data["Donor"]
                            n_colors=1
                        elif data_name == "Acceptor":
                            data = localisation_data["Acceptor"]
                            n_colors = 1
                        elif data_name == "FRET Data":
                            data = np.array([localisation_data["Donor"], localisation_data["Acceptor"]])
                            n_colors = 2
                        elif "FRET Efficiency" in data_name:
                            data = localisation_data["FRET Efficiency"]
                            n_colors = 1
                        elif "ALEX Efficiency" in data_name:
                            data = localisation_data["ALEX Efficiency"]
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

                        if len(data.shape) == 1:
                            data = np.expand_dims(data, axis=0)

                        if crop_plots == True and len(crop_range) == 2:
                            crop_range = sorted(crop_range)
                            data = np.array(data)
                            data = data[:, crop_range[0]:crop_range[1]]

                        data = np.array(data).T

                        deeplasi_dataset[localisation_index] = data

        except:
            print(traceback.format_exc())
            pass

        return deeplasi_dataset, n_colors


    def _detect_deeplasi_states(self, progress_callback = None, deeplasi_dataset = []):

        detected_states = []
        detected_labels = []

        try:

            self.deeplasi_datadict, n_colors = self.build_deeplasi_dataset()

            deeplasi_data = list(self.deeplasi_datadict.values())

            lengths = np.unique([len(data) for data in deeplasi_data])

            if len(lengths) == 1:
                deeplasi_mode = "fast"
            else:
                deeplasi_mode = "slow"

            wrapper = DeepLasiWrapper(parent=self, n_colors=n_colors)

            detected_states, detected_labels = wrapper.predict(
                deeplasi_data,
                n_colors=n_colors,
                deeplasi_mode=deeplasi_mode,
                progress_callback=progress_callback)

        except:
            print(traceback.format_exc())

        self.detected_states = detected_states
        self.detected_labels = detected_labels

        return detected_states, detected_labels


    def _detect_deeplasi_states_cleanup(self):

        try:

            dataset_name = self.fitting_window.deeplasi_fit_dataset.currentText()

            for localisation_index, localisation_number in enumerate(self.deeplasi_datadict.keys()):

                localisation_data = self.data_dict[dataset_name][localisation_number]
                state = self.detected_states[localisation_index]
                label = self.detected_labels[localisation_index]

                localisation_data["states"] = np.array(state).astype(int)

                self.data_dict[dataset_name][localisation_number] = copy.deepcopy(localisation_data)

            self.compute_state_means(dataset_name=dataset_name)
            self.plot_traces(update_plot=True)

            self.gui_progrssbar(progress=0, name="deeplasi")
            self.print_notification(f"DeepLasi prediction complete.")

        except:
            print(traceback.format_exc())


    def detect_deeplasi_states(self):

        if self.data_dict != {}:

            worker = Worker(self._detect_deeplasi_states)
            worker.signals.finished.connect(self._detect_deeplasi_states_cleanup)
            worker.signals.progress.connect(partial(self.gui_progrssbar,name="deeplasi"))
            self.threadpool.start(worker)


