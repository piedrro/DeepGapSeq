import os
import pickle
import numpy as np
from PyQt5.QtWidgets import QFileDialog
import traceback
import pandas as pd
import json
import copy

class _import_methods:

    def import_simulated_data(self):

        desktop = os.path.expanduser("~/Desktop")
        path, _ = QFileDialog.getOpenFileName(self, "Open Files", desktop, "DeepGapSeq Simulated Traces (*.pkl)")

        if os.path.isfile(path):

            self.data_dict = {}

            file_name = os.path.basename(path)

            self.print_notification("Loading simulated data from: " + file_name)

            with open(path, 'rb') as handle:
                trace_dictionary = pickle.load(handle)

            dataset_name = "Data"

            trace_data = trace_dictionary["data"]
            trace_labels = trace_dictionary["labels"]

            self.data_dict[dataset_name] = []

            for i in range(len(trace_data)):

                data = trace_data[i]

                if len(data.shape) == 1:
                    donor_data =[]
                    acceptor_data = []
                    efficiency_data = data
                elif data.shape[-1] == 2:
                    donor_data = data[:,0]
                    acceptor_data = data[:,1]
                    efficiency_data = np.divide(acceptor_data, donor_data)
                else:
                    donor_data = []
                    acceptor_data = []
                    efficiency_data = []

                labels = trace_labels[i]

                self.data_dict[dataset_name].append({"donor": donor_data,
                                                     "acceptor": acceptor_data,
                                                     "efficiency": efficiency_data,
                                                     "states": labels,
                                                     "filter": False,
                                                     "state_means": {},
                                                     "user_label" : 0,
                                                     "nucleotide_label" : 0,
                                                     "break_points" : [],
                                                     "crop_range" : [],
                                                     "gamma_ranges" : [],
                                                     "import_path" : path,
                                                     })


            self.compute_state_means()

            self.plot_data.clear()
            self.plot_data.addItems(list(self.data_dict.keys()))

            self.populate_plot_mode()

            self.plot_localisation_number.setValue(0)

            self.initialise_plot()


    def import_data_files(self):

        try:

            self.data_dict = {}
            n_traces = 0

            file_format = self.import_settings.import_data_file_format.currentText()
            traces_per_file = self.import_settings.import_data_traces_per_file.currentText()
            sep = self.import_settings.import_data_separator.currentText()
            column_format = self.import_settings.import_data_column_format.currentText()
            skiprows = int(self.import_settings.import_data_skip_rows.currentText())
            skipcolumns = int(self.import_settings.import_data_skip_columns.currentText())
            dataset_mode = int(self.import_settings.import_data_dataset_mode.currentIndex())

            if sep.lower() == "space":
                sep = " "
            if sep.lower() == "tab":
                sep == "\t"
            if sep.lower() == ",":
                sep == ","

            column_names = column_format.split("-")
            column_names = [col.lower() for col in column_names]

            desktop = os.path.expanduser("~/Desktop")

            paths, _ = QFileDialog.getOpenFileNames(self,
                "Open File(s)",
                desktop,
                f"Data Files (*{file_format})")

            if len(paths) > 0:

                n_traces = 0

                if dataset_mode == 1:
                    dataset_names = [os.path.basename(path) for path in paths]
                else:
                    dataset_names = [os.path.basename(paths[0])]*len(paths)

                for path_index, path in enumerate(paths):

                    dataset_name = dataset_names[path_index]

                    if dataset_name not in self.data_dict.keys():
                        self.data_dict[dataset_name] = []

                    if file_format in [".dat", ".txt"]:
                        data = pd.read_table(path, sep=sep,skiprows=skiprows)
                    elif file_format == ".csv":
                        data = pd.read_csv(path, skiprows=skiprows)
                    elif file_format == ".xlsx":
                        data = pd.read_excel(path, skiprows=skiprows)

                    if skipcolumns > 0:
                        data = data.iloc[:,skipcolumns:]

                    n_rows, n_columns = data.shape

                    if n_columns == 1:
                        self.print_notification(f"Importing {dataset_name} as a single columns, incorrect separator?")
                        import_error = True
                    elif traces_per_file == "Multiple" and n_columns % len(column_names) != 0:
                        self.print_notification(f"Importing {dataset_name} with {n_columns} columns, not divisible by {len(column_names)}")
                        import_error = True
                    else:
                        import_error = False

                    if import_error == False:

                        if traces_per_file == "Multiple":
                            import_range = range(0, n_columns, len(column_names))
                        else:
                            import_range = range(0, len(column_names))

                        for i in import_range:

                            loc_data = {"efficiency" : [], "states": [],
                                        "filter": False, "state_means": {},
                                        "user_label": 0, "nucleotide_label": 0,
                                        "break_points": [], "gamma_ranges": [],
                                        "crop_range" : [],
                                        "import_path" : path,
                                        }

                            # Select the current group of four columns
                            group = data.iloc[:, i:i + len(column_names)]

                            group.columns = column_names
                            group_dict = group.to_dict(orient="list")

                            if self.import_settings.import_data_alex.isChecked():
                                alex_dict = self.get_alex_data(group_dict["donor"], group_dict["acceptor"])

                                for key, value in alex_dict.items():
                                    loc_data[key] = value

                            else:

                                for key, value in group_dict.items():
                                    loc_data[key] = value

                                if loc_data["efficiency"] == []:
                                    loc_data["efficiency"] = self.calculate_fret_efficiency(loc_data["donor"], loc_data["acceptor"])

                            self.data_dict[dataset_name].append(loc_data)
                            n_traces += 1/len(column_names)

            if n_traces > 1:

                self.print_notification(f"Imported {int(n_traces)} traces")

                self.compute_state_means()

                self.plot_data.clear()
                if len(self.data_dict.keys()) == 1:
                    self.plot_data.addItems(list(self.data_dict.keys()))
                else:
                    self.plot_data.addItem("All Datasets")
                    self.plot_data.addItems(list(self.data_dict.keys()))

                self.populate_plot_mode()

                self.plot_localisation_number.setValue(0)

                self.initialise_plot()

        except:
            print(traceback.format_exc())
            pass

    def compute_state_means(self, dataset_name=None):

        def _compute_state_means(data, labels):
            state_means = labels.copy()
            for state in np.unique(labels):
                state_means[state_means == state] = np.mean(data[state_means == state])
            return state_means

        if dataset_name is None:
            dataset_names = self.data_dict.keys()
        else:
            dataset_names = [dataset_name]

        for dataset_name in dataset_names:
            for i in range(len(self.data_dict[dataset_name])):
                trace_data = self.data_dict[dataset_name][i]
                crop_range = copy.deepcopy(trace_data["crop_range"])

                labels = np.array(trace_data["states"])

                if "state_means" not in trace_data:
                    trace_data["state_means"] = {}

                for plot in ["donor", "acceptor", "efficiency", "DD", "AA", "DA", "AD"]:
                    if plot in trace_data.keys():
                        plot_data = np.array(trace_data[plot])

                        if len(plot_data) > 0 and len(labels) > 0:

                            if len(plot_data) == len(labels):

                                state_means_y = _compute_state_means(plot_data, labels)
                                state_means_x = np.arange(len(state_means_y))
                                trace_data["state_means"][plot] = [state_means_x, state_means_y]

                            else:
                                plot_data = plot_data[int(crop_range[0]):int(crop_range[1])]
                                state_means_y = _compute_state_means(plot_data, labels)
                                state_means_x = np.arange(int(crop_range[0]),int(crop_range[1]))
                                trace_data["state_means"][plot] = [state_means_x, state_means_y]

                        else:
                            trace_data["state_means"][plot] = [[], []]

                self.data_dict[dataset_name][i] = copy.deepcopy(trace_data)

    def calculate_fret_efficiency(self, donor, acceptor, gamma_correction=1):

        donor = np.array(donor)
        acceptor = np.array(acceptor)

        efficiency = acceptor / ((gamma_correction * donor) + acceptor)

        return efficiency

    def calculate_alex_efficiency(self, DD, DA, gamma_correction=1):

            DD = np.array(DD)
            DA = np.array(DA)

            efficiency = DA / ((gamma_correction * DD) + DA)

            return efficiency

    def get_alex_data(self, donor, acceptor):

        alex_first_frame = self.import_settings.alex_firstframe_excitation.currentIndex()

        donor = np.array(donor)
        acceptor = np.array(acceptor)

        if alex_first_frame == 0:
            "donor excitaton first"

            DD = donor[::2] #every second element, starting from 0
            AD = donor[1::2]  # every second element, starting from 1

            DA = acceptor[::2] #every second element, starting from 0
            AA = acceptor[1::2] #every second element, starting from 1

        else:
            "acceptor excitation first"

            AA = acceptor[::2] #every second element, starting from 0
            DA = acceptor[1::2]  # every second element, starting from 1

            AD = donor[::2] #every second element, starting from 0
            DD = donor[1::2] #every second element, starting from 1

        efficiency = self.calculate_alex_efficiency(DD, DA)

        alex_dict = {"DD": DD, "DA": DA,
                     "AA": AA, "AD": AD,
                     "efficiency": efficiency}

        return alex_dict

    def populate_plot_mode(self):

        self.plot_mode.clear()

        plot_names = []

        for dataset_name in self.data_dict.keys():
            for plot_name in self.data_dict[dataset_name][0].keys():
                if plot_name not in plot_names:
                    plot_names.append(plot_name)

        if self.import_settings.import_data_alex.isChecked() == False:
            if "donor" in plot_names:
                self.plot_mode.addItem("Donor")
            if "acceptor" in plot_names:
                self.plot_mode.addItem("Acceptor")
            if set(["donor", "acceptor"]).issubset(plot_names):
                self.plot_mode.addItem("FRET Data")
            if "efficiency" in plot_names:
                self.plot_mode.addItem("FRET Efficiency")
            if set(["donor", "acceptor", "efficiency"]).issubset(plot_names):
                self.plot_mode.addItem("FRET Data + FRET Efficiency")
        else:
            if set(["DD", "AA", "DA", "AD"]).issubset(plot_names):
                self.plot_mode.addItem("ALEX Data")
            if "efficiency" in plot_names:
                self.plot_mode.addItem("ALEX Efficiency")
            if set(["DD", "AA", "DA", "AD", "efficiency"]).issubset(plot_names):
                self.plot_mode.addItem("ALEX Data + ALEX Efficiency")


    def import_gapseq_json(self):

        try:
            expected_data = {"donor": np.array([]), "acceptor": np.array([]),
                             "efficiency": np.array([]), "states": np.array([]),
                             "DD": np.array([]), "DA": np.array([]),
                             "AA": np.array([]), "AD": np.array([]),
                             "filter": False, "state_means": {},
                             "user_label": 0, "nucleotide_label": 0,
                             "break_points": [], "crop_range": [],
                             "gamma_ranges": [], "import_path": "", }

            desktop = os.path.expanduser("~/Desktop")

            paths, _ = QFileDialog.getOpenFileNames(self,"Open File(s)", desktop,f"Data Files (*.json)")

            self.data_dict = {}
            #
            # paths = [r"C:/Users/turnerp/Desktop/deepgapseq_simulated_traces/trace_dataset_example_2023_11_09.json"]

            n_traces = 0

            for path in paths:

                import_data = json.load(open(path, "r"))

                for dataset_name, dataset_data in import_data["data"].items():

                    if dataset_name not in self.data_dict.keys():
                        self.data_dict[dataset_name] = []

                    for localisation_index, localisation_data in enumerate(dataset_data):

                        localisation_dict = {}

                        for key, value in localisation_data.items():


                            if key in expected_data.keys():
                                expected_type = type(expected_data[key])

                                if expected_type == type(np.array([])):
                                    value = np.array(value)

                                localisation_dict[key] = value

                        for key, value in expected_data.items():
                            if key not in localisation_dict.keys():
                                localisation_dict[key] = value

                        self.data_dict[dataset_name].append(localisation_dict)
                        n_traces += 1

            if n_traces > 0:

                self.compute_state_means()

                self.plot_data.clear()
                if len(self.data_dict.keys()) == 1:
                    self.plot_data.addItems(list(self.data_dict.keys()))
                else:
                    self.plot_data.addItem("All Datasets")
                    self.plot_data.addItems(list(self.data_dict.keys()))

                self.populate_plot_mode()

                self.plot_localisation_number.setValue(0)

                self.initialise_plot()

        except:
            print(traceback.format_exc())
            pass