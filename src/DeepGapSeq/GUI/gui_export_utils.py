
import json
import os.path
import traceback
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import pandas as pd
import originpro as op
from functools import partial


class _export_methods:

    def populate_export_combos(self):

            try:

                self.update_export_dataset_selection("export_dataset_selection")
                self.update_export_dataset_selection("excel_export_dataset_selection")
                self.update_export_dataset_selection("origin_export_dataset_selection")

                self.update_export_channel_selection("export_dataset_selection", "export_channel_selection")
                self.update_export_channel_selection("excel_export_dataset_selection", "excel_export_channel_selection")
                self.update_export_channel_selection("origin_export_dataset_selection", "origin_export_channel_selection")

                for dataset_combo_name in ["export_dataset_selection", "excel_export_dataset_selection", "origin_export_dataset_selection"]:
                    try:
                        dataset_combo = getattr(self.export_settings, dataset_combo_name)
                        dataset_name = dataset_combo_name
                        channel_name = dataset_combo_name.replace("dataset", "channel")
                        dataset_combo_func = partial(self.update_export_channel_selection, dataset_combo_name, channel_name)
                        dataset_combo.currentIndexChanged.connect(dataset_combo_func)
                    except:
                        print(traceback.format_exc())

            except:
                print(traceback.format_exc())

    def update_export_dataset_selection(self, dataset_combo ="export_dataset_selection"):

        try:

            if len(self.data_dict.keys()) > 0:

                dataset_combo = getattr(self.export_settings, dataset_combo)
                dataset_names = list(self.data_dict.keys())

                if len(dataset_names) > 1:
                    dataset_names.insert(0, "All Datasets")

                dataset_combo.blockSignals(True)
                dataset_combo.clear()
                dataset_combo.addItems(dataset_names)
                dataset_combo.blockSignals(False)

        except:
            print(traceback.format_exc())

    def update_export_channel_selection(self, dataset_combo_name ="export_dataset_selection",
                  channel_combo_name = "export_channel_selection"):

        try:

            if len(self.data_dict.keys()) > 0:

                channel_combo = getattr(self.export_settings, channel_combo_name)
                dataset_combo = getattr(self.export_settings, dataset_combo_name)

                export_dataset = dataset_combo.currentText()

                if export_dataset == "All Datasets":
                    dataset_list = list(self.data_dict.keys())
                else:
                    dataset_list = [export_dataset]

                all_export_names = []
                combo_options = []
                self.export_selection_dict = {}

                for dataset_name in dataset_list:
                    for channel_name in self.data_dict[dataset_name][0].keys():
                        if channel_name in ["Donor", "Acceptor", "FRET Efficiency","ALEX Efficiency","DD", "AA", "DA", "AD"]:
                            data_length = len(self.data_dict[dataset_name][0][channel_name])
                            if data_length > 1:
                                if channel_name not in all_export_names:
                                    all_export_names.append(channel_name)

                if set(["Donor", "Acceptor"]).issubset(all_export_names):
                    combo_options.append("FRET Data")
                    self.export_selection_dict["FRET Data"] = ["Donor", "Acceptor"]
                if "FRET Efficiency" in all_export_names:
                    combo_options.append("FRET Efficiency")
                    self.export_selection_dict["FRET Efficiency"] = ["FRET Efficiency"]
                if set(["Donor", "Acceptor", "FRET Efficiency"]).issubset(all_export_names):
                    combo_options.append("FRET Data + FRET Efficiency")
                    self.export_selection_dict["FRET Data + FRET Efficiency"] = ["Donor", "Acceptor", "FRET Efficiency"]
                if "ALEX Efficiency" in all_export_names:
                    combo_options.append("ALEX Efficiency")
                    self.export_selection_dict["ALEX Efficiency"] = ["ALEX Efficiency"]
                if set(["DD","DA","AD","AA"]).issubset(all_export_names):
                    combo_options.append("ALEX Data")
                    self.export_selection_dict["ALEX Data"] = ["DD","DA","AD","AA"]
                if set(["DD","DA","AD","AA","ALEX Efficiency"]).issubset(all_export_names):
                    combo_options.append("ALEX Data + ALEX Efficiency")
                    self.export_selection_dict["ALEX Data + ALEX Efficiency"] = ["DD","DA","AD","AA","ALEX Efficiency"]
                if "Donor" in all_export_names:
                    combo_options.append("Donor")
                    self.export_selection_dict["Donor"] = ["Donor"]
                if "Acceptor" in all_export_names:
                    combo_options.append("Acceptor")
                    self.export_selection_dict["Acceptor"] = ["Acceptor"]
                if "DD" in all_export_names:
                    combo_options.append("DD")
                    self.export_selection_dict["DD"] = ["DD"]
                if "AA" in all_export_names:
                    combo_options.append("AA")
                    self.export_selection_dict["AA"] = ["AA"]
                if "DA" in all_export_names:
                    combo_options.append("DA")
                    self.export_selection_dict["DA"] = ["DA"]
                if "AD" in all_export_names:
                    combo_options.append("AD")
                    self.export_selection_dict["AD"] = ["AD"]

                channel_combo.blockSignals(True)
                channel_combo.clear()
                channel_combo.addItems(combo_options)
                channel_combo.blockSignals(False)

        except:
            print(traceback.format_exc())

    def initialise_excel_export(self):

        if self.data_dict != {}:

            export_location = self.export_settings.excel_export_location.currentText()
            split_datasets = self.export_settings.excel_export_split_datasets.isChecked()
            export_dataset_name = self.export_settings.excel_export_dataset_selection.currentText()
            export_channel_name = self.export_settings.excel_export_channel_selection.currentText()
            crop_mode = self.export_settings.excel_export_crop_data.isChecked()
            export_states = self.export_settings.excel_export_fitted_states.isChecked()

            export_paths = self.get_export_paths(extension="xlsx")

            if export_location == "Select Directory":
                export_dir = os.path.dirname(export_paths[0])

                export_dir = QFileDialog.getExistingDirectory(self, "Select Directory", export_dir)

                if export_dir != "":
                    export_paths = [os.path.join(export_dir, os.path.basename(export_path)) for export_path in export_paths]
                    export_paths = [os.path.abspath(export_path) for export_path in export_paths]

            self.export_excel_data(export_dataset_name, export_channel_name,
                crop_mode,export_states, export_paths, split_datasets)

    def initialise_origin_export(self):

        if self.data_dict != {}:

            export_location = self.export_settings.origin_export_location.currentText()
            split_datasets = self.export_settings.origin_export_split_datasets.isChecked()
            export_dataset_name = self.export_settings.origin_export_dataset_selection.currentText()
            export_channel_name = self.export_settings.origin_export_channel_selection.currentText()
            crop_mode = self.export_settings.origin_export_crop_data.isChecked()
            export_states = self.export_settings.origin_export_fitted_states.isChecked()

            export_paths = self.get_export_paths(extension="opju")

            if export_location == "Select Directory":
                export_dir = os.path.dirname(export_paths[0])

                export_dir = QFileDialog.getExistingDirectory(self, "Select Directory", export_dir)

                if export_dir != "":
                    export_paths = [os.path.join(export_dir, os.path.basename(export_path)) for export_path in export_paths]
                    export_paths = [os.path.abspath(export_path) for export_path in export_paths]

            self.export_origin_data(export_dataset_name, export_channel_name,
                crop_mode, export_states, export_paths, split_datasets)

    def initialise_json_export(self):

        if self.data_dict != {}:

            export_location = self.export_settings.json_export_location.currentText()

            export_path = self.get_export_paths(extension="json")[0]

            if export_location == "Select Directory":
                export_path, _ = QFileDialog.getSaveFileName(self, "Select Directory", export_path)

            export_dir = os.path.dirname(export_path)

            if os.path.isdir(export_dir) == True:

                self.export_gapseq_json(export_path)

    def initialise_file_export(self):

        if self.data_dict != {}:

            export_mode = self.export_settings.export_mode.currentText()
            export_location = self.export_settings.export_location.currentText()
            split_datasets = self.export_settings.export_split_datasets.isChecked()
            export_dataset_name = self.export_settings.export_dataset_selection.currentText()
            export_channel_name = self.export_settings.export_channel_selection.currentText()
            crop_mode = self.export_settings.export_crop_data.isChecked()
            data_separator = self.export_settings.export_separator.currentText()
            export_states = self.export_settings.excel_export_fitted_states.isChecked()

            if export_mode == "Dat (.dat)":
                export_paths = self.get_export_paths(extension="dat")
            if export_mode == "Text (.txt)":
                export_paths = self.get_export_paths(extension="txt")
            if export_mode == "CSV (.csv)":
                export_paths = self.get_export_paths(extension="csv")

            if data_separator.lower() == "space":
                data_separator = " "
            elif data_separator.lower() == "tab":
                data_separator = "\t"
            elif data_separator.lower() == "comma":
                data_separator = ","

            if export_location == "Select Directory":
                export_dir = os.path.dirname(export_paths[0])

                export_dir = QFileDialog.getExistingDirectory(self, "Select Directory", export_dir)

                if export_dir != "":
                    export_paths = [os.path.join(export_dir, os.path.basename(export_path)) for export_path in export_paths]
                    export_paths = [os.path.abspath(export_path) for export_path in export_paths]

            self.export_dat(export_dataset_name, export_channel_name,
                crop_mode, export_states, data_separator, export_paths, split_datasets)

    def get_export_paths(self, extension="json"):

        export_paths = []

        import_paths = [value[0]["import_path"] for key, value in self.data_dict.items()]

        for import_path in import_paths:

            export_filename = os.path.basename(import_path)
            export_dir = os.path.dirname(import_path)
            if "_gapseq" not in export_filename:
                export_filename = export_filename.split(".")[0] + f"_gapseq.{extension}"
            else:
                export_filename = export_filename.split(".")[0] + f".{extension}"

            export_path = os.path.join(export_dir, export_filename)
            export_path = os.path.abspath(export_path)

            export_paths.append(export_path)

        return export_paths

    def export_excel_data(self,export_dataset_name, export_channel_name,
            crop_mode, export_states=False, export_paths = [], split_datasets = False):

        try:
            if self.data_dict != {}:

                if split_datasets == False:

                    export_path = export_paths[0]

                    export_data_dict = self.get_export_data("excel",
                        export_dataset_name, export_channel_name, crop_mode, export_states)

                    export_data = export_data_dict["data"]

                    max_length = max([len(data) for data in export_data])

                    export_data = [np.pad(data, (0, max_length - len(data)), mode="constant", constant_values=np.nan) for data in export_data]

                    export_data = np.stack(export_data, axis=0).T

                    export_data = pd.DataFrame(export_data)

                    export_data.columns = [export_data_dict["index"],
                                           export_data_dict["dataset"],
                                           export_data_dict["data_name"],
                                           export_data_dict["user_label"],
                                           export_data_dict["nucleotide_label"]]

                    export_data.columns.names = ['Index', 'Dataset', 'Data', 'Class', 'Nucleotide']

                    with pd.ExcelWriter(export_path) as writer:
                        export_data.to_excel(writer, sheet_name='Trace Data', index=True, startrow=1, startcol=1)

                    self.print_notification(f"Exported data to {export_path}")

                else:

                    for export_dataset_name, export_path in zip(self.data_dict.keys(), export_paths):

                        export_data_dict = self.get_export_data("excel",
                            export_dataset_name, export_channel_name, crop_mode, export_states, )

                        export_data = np.stack(export_data_dict["data"], axis=0).T

                        export_data = pd.DataFrame(export_data)

                        export_data.columns = [export_data_dict["index"],
                                               export_data_dict["dataset"],
                                               export_data_dict["data_name"],
                                               export_data_dict["user_label"],
                                               export_data_dict["nucleotide_label"]]

                        export_data.columns.names = ['Index', 'Dataset', 'Data', 'Class', 'Nucleotide']

                        with pd.ExcelWriter(export_path) as writer:
                            export_data.to_excel(writer, sheet_name='Trace Data', index=True, startrow=1, startcol=1)

        except:
            print(traceback.format_exc())

    def export_origin_data(self, export_dataset_name, export_channel_name,
            crop_mode, export_states=False, export_paths = [], split_datasets = False):

         try:

             if self.data_dict != {}:
                 if split_datasets == False:
                     export_path = export_paths[0]

                     export_data_dict = self.get_export_data("origin",
                         export_dataset_name, export_channel_name, crop_mode, export_states, )

                     export_data = export_data_dict["data"]

                     max_length = max([len(data) for data in export_data])

                     export_data = [np.pad(data, (0, max_length - len(data)), mode="constant", constant_values=np.nan) for data in export_data]

                     export_data = np.stack(export_data, axis=0).T

                     export_data = pd.DataFrame(export_data)

                     export_data.columns = export_data_dict["data_name"]

                     if os.path.exists(export_path):
                         os.remove(export_path)

                     if op.oext:
                         op.set_show(False)

                     op.new()

                     wks = op.new_sheet()
                     wks.cols_axis('YY')
                     wks.from_df(export_data, addindex=True)

                     for i in range(len(export_data_dict["data_name"])):

                         index = export_data_dict["index"][i]
                         dataset = export_data_dict["dataset"][i]
                         user_label = export_data_dict["user_label"][i]
                         nucleotide_label = export_data_dict["nucleotide_label"][i]

                         wks.set_label(i, dataset, 'Dataset')
                         wks.set_label(i, index, 'Index')
                         wks.set_label(i, user_label, 'User Label')
                         wks.set_label(i, nucleotide_label, 'Nucleotide Label')

                     op.save(export_path)

                     if op.oext:
                         op.exit()

                     self.print_notification(f"Exported data to {export_path}")

                 else:

                     for export_dataset_name, export_path in zip(self.data_dict.keys(), export_paths):

                         export_data_dict = self.get_export_data("origin",
                             export_dataset_name, export_channel_name, crop_mode, export_states, )

                         export_dataset = np.stack(export_data_dict["data"], axis=0).T

                         export_dataset = pd.DataFrame(export_dataset)

                         export_dataset.columns = export_data_dict["data_name"]

                         export_dataset.columns = export_data_dict["data_name"]

                         if os.path.exists(export_path):
                             os.remove(export_path)

                         if op.oext:
                             op.set_show(False)

                         op.new()

                         wks = op.new_sheet()
                         wks.cols_axis('YY')
                         wks.from_df(export_dataset, addindex=True)

                         for i in range(len(export_data_dict["data_name"])):
                             index = export_data_dict["index"][i]
                             dataset = export_data_dict["dataset"][i]
                             user_label = export_data_dict["user_label"][i]
                             nucleotide_label = export_data_dict["nucleotide_label"][i]

                             wks.set_label(i, dataset, 'Dataset')
                             wks.set_label(i, index, 'Index')
                             wks.set_label(i, user_label, 'User Label')
                             wks.set_label(i, nucleotide_label, 'Nucleotide Label')

                         op.save(export_path)

                         if op.oext:
                             op.exit()

                         self.print_notification(f"Exported data to {export_path}")



         except:
             print(traceback.format_exc())

    def export_dat(self, export_dataset_name, export_channel_name, crop_mode,
            export_states = False, data_separator=",", export_paths = [], split_datasets = False):

            try:

                if self.data_dict != {}:
                    if split_datasets == False:

                        export_path = export_paths[0]

                        export_data_dict = self.get_export_data("data",
                            export_dataset_name, export_channel_name, crop_mode, export_states, )

                        export_dataset = np.stack(export_data_dict["data"], axis=0).T

                        export_dataset = pd.DataFrame(export_dataset)

                        export_dataset.columns = [export_data_dict["index"],
                                                  export_data_dict["dataset"],
                                                  export_data_dict["data_name"],
                                                  export_data_dict["user_label"],
                                                  export_data_dict["nucleotide_label"]]

                        export_dataset.to_csv(export_path, sep=data_separator, index=False, header=True)

                        self.print_notification(f"Exported data to {export_path}")

                    else:

                        for export_dataset_name, export_path in zip(self.data_dict.keys(), export_paths):

                            export_data_dict = self.get_export_data("data",
                                export_dataset_name, export_channel_name, crop_mode, export_states, )

                            export_dataset = np.stack(export_data_dict["data"], axis=0).T

                            export_dataset = pd.DataFrame(export_dataset)

                            export_dataset.columns = [export_data_dict["index"],
                                                      export_data_dict["dataset"],
                                                      export_data_dict["data_name"],
                                                      export_data_dict["user_label"],
                                                      export_data_dict["nucleotide_label"]]

                            export_dataset.to_csv(export_path, sep=data_separator, index=False, header=True)

                            self.print_notification(f"Exported data to {export_path}")

            except:
                print(traceback.format_exc())

    def get_export_data(self, export_mode, export_dataset_name, export_channel_name,
            crop_data, export_states, pad_data = True, pad_value = np.nan):

        loc_index = []
        loc_dataset = []
        loc_user_label = []
        loc_nucleotide_label = []
        loc_data = []
        loc_data_name = []

        if export_dataset_name == "All Datasets":
            dataset_list = list(self.data_dict.keys())
        else:
            dataset_list = [export_dataset_name]

        for dataset_name in dataset_list:

            dataset_data = self.data_dict[dataset_name]

            for localisation_number, localisation_data in enumerate(dataset_data):

                user_label = localisation_data["user_label"]
                nucleotide_label = localisation_data["nucleotide_label"]
                crop_range = localisation_data["crop_range"]

                if self.get_filter_status(export_mode, user_label, nucleotide_label) == False:

                    for data_name in self.export_selection_dict[export_channel_name]:

                        data = localisation_data[data_name]
                        state_means_x, state_means_y = localisation_data["state_means"][data_name]

                        if crop_data == True and len(crop_range) == 2:
                            crop_range = [int(crop_range[0]), int(crop_range[1])]
                            crop_range = sorted(crop_range)
                            if crop_range[0] > 0 and crop_range[1] < len(data):
                                data = data[crop_range[0]:crop_range[1]]
                                state_means_y = state_means_y[crop_range[0]:crop_range[1]]
                        else:
                            if len(state_means_y) < len(data):
                                state_indeces = state_means_x
                                padded_state_means_y = [pad_value] * len(data)
                                for index, value in zip(state_indeces, state_means_y):
                                    padded_state_means_y[index] = value
                                state_means_y = padded_state_means_y

                        loc_index.append(localisation_number)
                        loc_dataset.append(dataset_name)
                        loc_user_label.append(user_label)
                        loc_nucleotide_label.append(nucleotide_label)
                        loc_data.append(data)
                        loc_data_name.append(data_name)

                        if export_states:

                            loc_index.append(localisation_number)
                            loc_dataset.append(dataset_name)
                            loc_user_label.append(user_label)
                            loc_nucleotide_label.append(nucleotide_label)
                            loc_data.append(state_means_y)
                            loc_data_name.append(data_name+"_states")

        export_data_dict = {"index": loc_index,
                            "dataset": loc_dataset,
                            "user_label": loc_user_label,
                            "nucleotide_label": loc_nucleotide_label,
                            "data": loc_data,
                            "data_name": loc_data_name}

        if pad_data == True:

            data = export_data_dict["data"]

            max_length = max([len(data) for data in data])

            padded_data = [np.pad(data, (0, max_length - len(data)), mode="constant", constant_values=pad_value) for data in data]

            export_data_dict["data"] = padded_data

        return export_data_dict

    def export_gapseq_json(self, export_path):

        try:

            # export gapseq data as a json file

            if self.data_dict != {}:

                dataset_names = self.data_dict.keys()

                json_dataset_dict = self.build_json_dict(dataset_names=dataset_names)

                with open(export_path, "w") as f:
                    json.dump(json_dataset_dict, f, cls=npEncoder)

                self.print_notification(f"Exported data to {export_path}")

        except:
            print(traceback.format_exc())

    def get_filter_status(self, export_mode = "data", user_label = "", nucleotide_label= ""):

        if export_mode.lower() == "data":
            user_filter = self.export_settings.export_user_filter.currentText()
            nucleotide_filter = self.export_settings.export_nucleotide_filter.currentText()
        elif export_mode.lower() == "excel":
            user_filter = self.export_settings.excel_export_user_filter.currentText()
            nucleotide_filter = self.export_settings.excel_export_nucleotide_filter.currentText()
        elif export_mode.lower() == "origin":
            user_filter = self.export_settings.origin_export_user_filter.currentText()
            nucleotide_filter = self.export_settings.origin_export_nucleotide_filter.currentText()
        elif export_mode.lower() == "ebfret":
            user_filter = self.fitting_window.ebfret_user_filter.currentText()
            nucleotide_filter = self.fitting_window.ebfret_nucleotide_filter.currentText()
        elif export_mode.lower() == "deeplasi":
            user_filter = self.fitting_window.deeplasi_user_filter.currentText()
            nucleotide_filter = self.fitting_window.deeplasi_nucleotide_filter.currentText()
        elif export_mode.lower() == "analysis":
            user_filter = self.analysis_user_filter.currentText()
            nucleotide_filter = self.analysis_nucleotide_filter.currentText()
        elif export_mode.lower() == "inceptiontime":
            user_filter = self.detect_window.detect_user_filter.currentText()
            nucleotide_filter = self.detect_window.detect_nucleotide_filter.currentText()

        filter = False

        if user_filter != "None" and nucleotide_filter == "None":
            if user_label != user_filter:
                filter = True
        elif user_filter == "None" and nucleotide_filter != "None":
            if nucleotide_label != nucleotide_filter:
                filter = True
        elif user_filter != "None" and nucleotide_filter != "None":
            if user_label != user_filter or nucleotide_label != nucleotide_filter:
                filter = True

        return filter

    def json_dict_report(self, json_dataset):

        try:

            if json_dataset != {}:

                json_dataset = json_dataset.copy()

                json_report = {}
                dataset_traces = {}

                if "data" in json_dataset.keys():
                    data_dict = json_dataset["data"]
                else:
                    data_dict = json_dataset

                for dataset in data_dict.keys():

                    data = data_dict[dataset]

                    if dataset not in json_report.keys():
                        json_report[dataset] = {}

                    dataset_traces[dataset] = len(data)

                    for json_dict in data:

                        json_dict_keys = json_dict.keys()

                        for key in json_dict_keys:

                            if key not in json_report[dataset].keys():
                                json_report[dataset][key] = 0

                            json_report[dataset][key] += 1

                n_datasets = len(json_report.keys())
                unique_channels = list(set([key for dataset in json_report.keys() for key in json_report[dataset].keys()]))
                unique_n_traces = np.unique([value for dataset in json_report.keys() for value in json_report[dataset].values()])
                total_traces = sum([value for dataset in json_report.keys() for value in json_report[dataset].values()])

                try:
                    # size of json_dataset
                    json_dataset_size = len(json.dumps(json_dataset, indent=4, cls=npEncoder))
                    json_dataset_size_mb = json_dataset_size / 1000000
                except:
                    json_dataset_size_mb = 0

                print(f"JSON Dataset report:")
                print(f" N datasets: {n_datasets}")
                print(f" Dataset traces: {list(dataset_traces.values())}")
                print(f" Unique channels: {unique_channels}")
                print(f" N traces: {unique_n_traces}")
                print(f" Total traces: {total_traces}")
                print(f" Size: {json_dataset_size_mb} MB")

        except:
            print(traceback.format_exc())


    def build_json_dict(self, dataset_names = []):

        try:

            json_dataset_dict = {"metadata":{}, "data":{}}

            json_list_keys = ["Donor", "Acceptor", "FRET Efficiency", "ALEX Efficiency",
                              "DD", "AA", "DA", "AD", "states", "break_points", "crop_range", "gamma_ranges",
                              'gap_label', 'sequence_label', 'picasso_loc']

            json_var_keys = ["user_label", "nucleotide_label", "import_path"]

            if len(dataset_names) == 0:
                dataset_names = self.data_dict.keys()

            for dataset_name in dataset_names:

                dataset_data = self.data_dict[dataset_name]

                if dataset_name not in json_dataset_dict.keys():
                    json_dataset_dict["data"][dataset_name] = []

                for localisation_number, localisation_data in enumerate(dataset_data):

                    json_localisation_dict = {}

                    for key, value in localisation_data.items():

                        if key in json_list_keys:
                            if key in ["gap_label", "sequence_label"]:
                                json_localisation_dict[key] = str(value)
                            else:
                                json_localisation_dict[key] = list(value)

                        if key in json_var_keys:
                            json_localisation_dict[key] = value

                    json_dataset_dict["data"][dataset_name].append(json_localisation_dict)

        except:
            print(traceback.format_exc())

        return json_dataset_dict



class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)