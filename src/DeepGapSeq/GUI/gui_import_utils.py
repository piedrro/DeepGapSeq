import os
import pickle
import numpy as np
from PyQt5.QtWidgets import QFileDialog
import traceback
import pandas as pd
import json
import copy

class _import_methods:

    def populate_combos(self):

        self.updating_combos = True

        self.update_plot_data_combo()
        self.update_plot_mode_combo()

        self.populate_analysis_graph_combos()
        self.populate_deeplasi_options()

        self.updating_combos = False

    def import_simulated_data(self):

        try:

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
                        print(data.shape)
                        donor_data = []
                        acceptor_data = []
                        # efficiency_data = data
                    elif data.shape[-1] == 2:
                        donor_data = data[:,0]
                        acceptor_data = data[:,1]
                        # efficiency_data = np.divide(acceptor_data, donor_data)
                    else:
                        donor_data = []
                        acceptor_data = []
                        # efficiency_data = []

                    labels = trace_labels[i]

                    self.data_dict[dataset_name].append({"Donor": donor_data,
                                                         "Acceptor": acceptor_data,
                                                         # "FRET Efficiency": efficiency_data,
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

                self.compute_efficiencies()
                self.compute_state_means()

                self.populate_combos()

                self.plot_localisation_number.setValue(0)

                self.initialise_plot()
                self.initialise_analysis_plot()

        except:
            print(traceback.format_exc())


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
            column_names = [col for col in column_names]

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

                            loc_data = {"FRET Efficiency": [], "ALEX Efficiency": [],
                                        "states": [],"state_means": {},
                                        "user_label": 0, "nucleotide_label": 0,
                                        "break_points": [], "gamma_ranges": [],
                                        "crop_range" : [], "filter": False,
                                        "import_path" : path,
                                        }

                            # Select the current group of four columns
                            group = data.iloc[:, i:i + len(column_names)]

                            group.columns = column_names
                            group_dict = group.to_dict(orient="list")

                            if self.import_settings.import_data_alex.isChecked():
                                alex_dict = self.get_alex_data(group_dict["Donor"], group_dict["Acceptor"])

                                for key, value in alex_dict.items():
                                    loc_data[key] = value
                            else:
                                for key, value in group_dict.items():
                                    loc_data[key] = value

                            self.data_dict[dataset_name].append(loc_data)
                            n_traces += 1

            if n_traces > 1:

                self.print_notification(f"Imported {int(n_traces)} traces")

                self.compute_efficiencies()
                self.compute_state_means()

                self.populate_combos()

                self.plot_localisation_number.setValue(0)

                self.initialise_plot()
                self.initialise_analysis_plot()

        except:
            print(traceback.format_exc())
            pass

    def compute_state_means(self, dataset_name=None):

        def _compute_state_means(data, labels, print_data=False):

            unique_labels = np.unique(labels)

            labels = np.array(labels).astype(int).copy()
            data = np.array(data).astype(float).copy()

            if len(unique_labels) == 1:
                state_means = [np.mean(data)]*len(data)
            else:
                state_means = np.zeros(len(data))
                for state in np.unique(labels):
                    state_mean = np.mean(data[labels == state])
                    indices = np.where(labels == state)[0]
                    state_means[indices] = state_mean

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

                for plot in ["Donor", "Acceptor", "FRET Efficiency", "ALEX Efficiency", "DD", "AA", "DA", "AD"]:
                    if plot in trace_data.keys():
                        plot_data = np.array(trace_data[plot])

                        if len(plot_data) > 0 and len(labels) > 0:

                            if len(plot_data) == len(labels):

                                if plot == "Donor" and i == 17:
                                    print_data = True
                                else:
                                    print_data = False

                                state_means_y = _compute_state_means(plot_data, labels, print_data=print_data)
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

        alex_dict = {"DD": DD, "DA": DA,
                     "AA": AA, "AD": AD}

        return alex_dict

    def populate_trace_graph_combos(self):

        self.plot_mode.clear()

        plot_names = []

        for dataset_name in self.data_dict.keys():
            for plot_name, plot_value in self.data_dict[dataset_name][0].items():
                if plot_name in ["Donor", "Acceptor", "FRET Efficiency", "ALEX Efficiency", "DD", "AA", "DA", "AD"]:
                    if len(plot_value) > 0:
                        plot_names.append(plot_name)

        if "Donor" in plot_names:
            self.plot_mode.addItem("Donor")
        if "Acceptor" in plot_names:
            self.plot_mode.addItem("Acceptor")
        if set(["Donor", "Acceptor"]).issubset(plot_names):
            self.plot_mode.addItem("FRET Data")
        if set(["Donor", "Acceptor", "FRET Efficiency"]).issubset(plot_names):
            self.plot_mode.addItem("FRET Efficiency")
            self.plot_mode.addItem("FRET Data + FRET Efficiency")
        if set(["DD", "AA", "DA", "AD"]).issubset(plot_names):
            self.plot_mode.addItem("DD")
            self.plot_mode.addItem("AA")
            self.plot_mode.addItem("DA")
            self.plot_mode.addItem("AD")
            self.plot_mode.addItem("ALEX Data")
        if set(["DD", "AA", "DA", "AD", "ALEX Efficiency"]).issubset(plot_names):
            self.plot_mode.addItem("ALEX Efficiency")
            self.plot_mode.addItem("ALEX Data + ALEX Efficiency")

        self.plot_data.clear()
        if len(self.data_dict.keys()) == 1:
            self.plot_data.addItems(list(self.data_dict.keys()))
        else:
            self.plot_data.addItem("All Datasets")
            self.plot_data.addItems(list(self.data_dict.keys()))

    def update_plot_data_combo(self):

        try:

            plot_datasets = list(self.data_dict.keys())

            if len(plot_datasets) > 1:
                plot_datasets.insert(0, "All Datasets")

            self.plot_data.clear()
            self.plot_data.addItems(plot_datasets)

        except:
            print(traceback.format_exc())
            pass

    def update_plot_mode_combo(self):

        try:

            self.plot_mode.clear()

            plot_channels = []

            if self.plot_data.currentText() == "All Datasets":
                plot_datasets = list(self.data_dict.keys())
            else:
                plot_datasets = [self.plot_data.currentText()]

            for dataset_name in plot_datasets:
                if dataset_name in self.data_dict.keys():
                    for plot_name, plot_value in self.data_dict[dataset_name][0].items():
                        if plot_name in ["Donor", "Acceptor", "FRET Efficiency","ALEX Efficiency", "DD", "AA", "DA", "AD"]:
                            if len(plot_value) > 0:
                                plot_channels.append(plot_name)

            if set(["Donor", "Acceptor"]).issubset(plot_channels):
                plot_channels.insert(0, "FRET Data")
            if set(["Donor", "Acceptor", "FRET Efficiency"]).issubset(plot_channels):
                plot_channels.insert(0, "FRET Efficiency")
                plot_channels.insert(0, "FRET Data + FRET Efficiency")
            if set(["DD", "AA", "DA", "AD"]).issubset(plot_channels):
                plot_channels.insert(0, "ALEX Data")
            if set(["DD", "AA", "DA", "AD", "ALEX Efficiency"]).issubset(plot_channels):
                plot_channels.insert(0, "ALEX Efficiency")
                plot_channels.insert(0, "ALEX Data + ALEX Efficiency")

            if len(plot_channels) > 1:
                plot_channels.insert(0, "All Channels")

            self.plot_mode.addItems(plot_channels)

        except:
            print(traceback.format_exc())
            pass

    def populate_analysis_graph_combos(self):

        self.analysis_graph_mode.clear()

        plot_names = []

        for dataset_name in self.data_dict.keys():
            for plot_name, plot_value in self.data_dict[dataset_name][0].items():
                if plot_name in ["Donor", "Acceptor", "FRET Efficiency","ALEX Efficiency", "DD", "AA", "DA", "AD"]:
                    if len(plot_value) > 0:
                        plot_names.append(plot_name)

        if "Donor" in plot_names:
            self.analysis_graph_mode.addItem("Donor")
        if "Acceptor" in plot_names:
            self.analysis_graph_mode.addItem("Acceptor")
        if set(["Donor", "Acceptor", "FRET Efficiency"]).issubset(plot_names):
            self.analysis_graph_mode.addItem("FRET Efficiency")
        if set(["DD", "AA", "DA", "AD"]).issubset(plot_names):
            self.analysis_graph_mode.addItem("DD")
            self.analysis_graph_mode.addItem("AA")
            self.analysis_graph_mode.addItem("DA")
            self.analysis_graph_mode.addItem("AD")
        if set(["DD", "AA", "DA", "AD", "ALEX Efficiency"]).issubset(plot_names):
            self.analysis_graph_mode.addItem("ALEX Efficiency")

        self.analysis_graph_data.clear()
        if len(self.data_dict.keys()) == 1:
            self.analysis_graph_data.addItems(list(self.data_dict.keys()))
        else:
            self.analysis_graph_data.addItem("All Datasets")
            self.analysis_graph_data.addItems(list(self.data_dict.keys()))


    def import_gapseq_json(self):

        try:
            expected_data = {"Donor": np.array([]), "Acceptor": np.array([]),
                             "FRET Efficiency": np.array([]), "ALEX Efficiency": np.array([]),
                             "DD": np.array([]), "DA": np.array([]),
                             "AA": np.array([]), "AD": np.array([]),
                             "filter": False, "state_means": {}, "states": np.array([]),
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

                        localisation_dict["import_path"] = path

                        self.data_dict[dataset_name].append(localisation_dict)
                        n_traces += 1

            if n_traces > 0:

                self.compute_efficiencies()
                self.compute_state_means()

                self.populate_combos()

                self.plot_localisation_number.setValue(0)

                self.initialise_plot()
                self.initialise_analysis_plot()

        except:
            print(traceback.format_exc())
            pass

    def compute_efficiencies(self):

        try:

            if self.data_dict != {}:

                for dataset_name, dataset_data in self.data_dict.items():

                    channel_names = dataset_data[0].keys()
                    channel_names = [channel for channel in channel_names if channel in ["Donor", "Acceptor", "DD", "AA", "DA", "AD"]]
                    channel_names = [channel for channel in channel_names if len(dataset_data[0][channel]) > 0]

                    print(dataset_name, channel_names)

                    if set(["Donor", "Acceptor"]).issubset(channel_names):
                        self.compute_fret_efficiency(dataset_name)
                    if set(["DD", "AA", "DA", "AD"]).issubset(channel_names):
                        self.compute_alex_efficiency(dataset_name)

        except:
            print(traceback.format_exc())
            pass

    def compute_fret_efficiency(self, dataset_name):

        try:

            dataset_dict = self.data_dict[dataset_name]

            for localisation_index, localisation_dict in enumerate(dataset_dict):

                donor = localisation_dict["Donor"]
                acceptor = localisation_dict["Acceptor"]

                fret_efficiency = acceptor / (acceptor + donor)

                localisation_dict["FRET Efficiency"] = np.array(fret_efficiency)

                dataset_dict[localisation_index] = localisation_dict

        except:
            print(traceback.format_exc())
            pass

    def compute_alex_efficiency(self, dataset_name):

        try:

            dataset_dict = self.data_dict[dataset_name]

            for localisation_index, localisation_dict in enumerate(dataset_dict):

                dd = localisation_dict["DD"]
                da = localisation_dict["DA"]

                alex_efficiency = da / (da + dd)

                localisation_dict["ALEX Efficiency"] = np.array(alex_efficiency)

                dataset_dict[localisation_index] = localisation_dict

        except:
            print(traceback.format_exc())
            pass

