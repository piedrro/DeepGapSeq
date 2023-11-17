
import json
import os.path
import traceback
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import pandas as pd


class _export_methods:


    def populate_export_selection(self):

        self.export_settings.export_data_selection.clear()

        all_export_names = []
        self.export_selection_dict = {}

        for dataset_name in self.data_dict.keys():
            for plot_name in self.data_dict[dataset_name][0].keys():
                if plot_name not in all_export_names:
                    all_export_names.append(plot_name)

        if self.import_settings.import_data_alex.isChecked() == False:
            if "donor" in all_export_names:
                self.export_settings.export_data_selection.addItem("Donor")
                self.export_selection_dict["Donor"] = ["donor"]
            if "acceptor" in all_export_names:
                self.export_settings.export_data_selection.addItem("Acceptor")
                self.export_selection_dict["Acceptor"] = ["acceptor"]
            if set(["donor", "acceptor"]).issubset(all_export_names):
                self.export_settings.export_data_selection.addItem("FRET Data")
                self.export_selection_dict["FRET Data"] = ["donor", "acceptor"]
            if "efficiency" in all_export_names:
                self.export_settings.export_data_selection.addItem("FRET Efficiency")
                self.export_selection_dict["FRET Efficiency"] = ["efficiency"]
            if set(["donor", "acceptor", "efficiency"]).issubset(all_export_names):
                self.export_settings.export_data_selection.addItem("FRET Data + FRET Efficiency")
                self.export_selection_dict["FRET Data + FRET Efficiency"] = ["donor", "acceptor", "efficiency"]
        else:
            if set(["DD", "AA", "DA", "AD"]).issubset(all_export_names):
                self.export_settings.export_data_selection.addItem("ALEX Data")
                self.export_selection_dict["ALEX Data"] = ["DD", "AA", "DA", "AD"]
            if "efficiency" in all_export_names:
                self.plot_mode.addItem("ALEX Efficiency")
            if set(["DD", "AA", "DA", "AD", "efficiency"]).issubset(all_export_names):
                self.export_settings.export_data_selection.addItem("ALEX Data + ALEX Efficiency")
                self.export_selection_dict["ALEX Data + ALEX Efficiency"] = ["DD", "AA", "DA", "AD", "efficiency"]


    def initialise_json_export(self):

        export_location = self.export_settings.json_export_location.currentText()

        export_path = self.get_export_paths(extension="json")[0]

        if export_location == "Select Directory":

            export_path, _ = QFileDialog.getSaveFileName(self, "Select Directory", export_path)

        export_dir = os.path.dirname(export_path)

        if os.path.isdir(export_dir) == True:

            self.export_gapseq_json(export_path)

    def initialise_export(self):

        export_mode = self.export_settings.export_mode.currentText()
        export_data_selection = self.export_settings.export_data_selection.currentText()
        export_location = self.export_settings.export_location.currentText()
        split_datasets = self.export_settings.export_split_datasets.isChecked()

        if export_mode == "GapSeq (.json)":
            export_paths = self.get_export_paths(extension="json")
        elif export_mode == "Excel (.xlsx)":
            export_paths = self.get_export_paths(extension="xlsx")
        elif export_mode == "Dat (.dat)":
            export_paths = self.get_export_paths(extension="dat")

        n_datasets = len(self.data_dict.keys())

        if export_location == "Select Directory":
            export_dir = os.path.dirname(export_paths[0])

            export_dir = QFileDialog.getExistingDirectory(self, "Select Directory", export_dir)

            if export_dir != "":
                export_paths = [os.path.join(export_dir, os.path.basename(export_path)) for export_path in export_paths]
                export_paths = [os.path.abspath(export_path) for export_path in export_paths]

        elif export_mode == "Excel (.xlsx)":

            self.export_excel(export_paths, split_datasets)

        elif export_mode == "Dat (.dat)":

            self.export_dat(export_paths, split_datasets)


    def get_export_paths(self, extension="json"):

        export_paths = []

        import_paths = [value[0]["import_path"] for key, value in self.data_dict.items()]

        for import_path in import_paths:

            export_filename = os.path.basename(import_path)
            export_dir = os.path.dirname(import_path)
            export_filename = export_filename.split(".")[0] + f"_gapseq.{extension}"

            export_path = os.path.join(export_dir, export_filename)
            export_path = os.path.abspath(export_path)

            export_paths.append(export_path)

        return export_paths


    def export_excel(self,export_paths = [], split_datasets = False):

        try:
            if self.data_dict != {}:

                if split_datasets == False:

                    export_path = export_paths[0]

                    export_data_dict = self.get_export_data()

                    export_dataset = np.stack(export_data_dict["data"], axis=0).T

                    export_dataset = pd.DataFrame(export_dataset)

                    export_dataset.columns = [export_data_dict["index"],
                                              export_data_dict["dataset"],
                                              export_data_dict["data_name"],
                                              export_data_dict["user_label"],
                                              export_data_dict["nucleotide_label"]]

                    export_dataset.columns.names = ['Index', 'Dataset', 'Data', 'Class', 'Nucleotide']

                    with pd.ExcelWriter(export_path) as writer:
                        export_dataset.to_excel(writer, sheet_name='Trace Data', index=True, startrow=1, startcol=1)

                    self.print_notification(f"Exported data to {export_path}")

                else:

                    for dataset_name, export_path in zip(self.data_dict.keys(), export_paths):

                        export_data_dict = self.get_export_data([dataset_name])

                        export_dataset = np.stack(export_data_dict["data"], axis=0).T

                        export_dataset = pd.DataFrame(export_dataset)

                        export_dataset.columns = [export_data_dict["index"],
                                                  export_data_dict["dataset"],
                                                  export_data_dict["data_name"],
                                                  export_data_dict["user_label"],
                                                  export_data_dict["nucleotide_label"]]

                        export_dataset.columns.names = ['Index', 'Dataset', 'Data', 'Class', 'Nucleotide']

                        with pd.ExcelWriter(export_path) as writer:
                            export_dataset.to_excel(writer, sheet_name='Trace Data', index=True, startrow=1, startcol=1)

                        self.print_notification(f"Exported data to {export_path}")

        except:
            print(traceback.format_exc())


    def export_dat(self,export_paths = [], split_datasets = False):

        try:

            if self.data_dict != {}:
                if split_datasets == False:

                    export_path = export_paths[0]

                    export_data_dict = self.get_export_data()

                    export_dataset = np.stack(export_data_dict["data"], axis=0).T

                    export_dataset = pd.DataFrame(export_dataset)

                    export_dataset.to_csv(export_path, sep=" ", index=False, header=False)

                    self.print_notification(f"Exported data to {export_path}")

                else:

                    for dataset_name, export_path in zip(self.data_dict.keys(), export_paths):

                        export_data_dict = self.get_export_data([dataset_name])

                        export_dataset = np.stack(export_data_dict["data"], axis=0).T

                        export_dataset = pd.DataFrame(export_dataset)

                        export_dataset.to_csv(export_path, sep=" ", index=False, header=False)

                        self.print_notification(f"Exported data to {export_path}")

        except:
            print(traceback.format_exc())



    def get_export_data(self, dataset_names = []):

        export_selection = self.export_settings.export_data_selection.currentText()

        loc_index = []
        loc_dataset = []
        loc_user_label = []
        loc_nucleotide_label = []
        loc_data = []
        loc_data_name = []

        if dataset_names == []:
            dataset_names = self.data_dict.keys()

        for dataset_name in dataset_names:

            dataset_data = self.data_dict[dataset_name]

            for localisation_number, localisation_data in enumerate(dataset_data):

                user_label = localisation_data["user_label"]
                nucleotide_label = localisation_data["nucleotide_label"]

                if self.get_filter_status(user_label, nucleotide_label) == False:

                    for data_name in self.export_selection_dict[export_selection]:

                        loc_index.append(localisation_number)
                        loc_dataset.append(dataset_name)
                        loc_user_label.append(user_label)
                        loc_nucleotide_label.append(nucleotide_label)
                        loc_data.append(localisation_data[data_name])
                        loc_data_name.append(data_name)

        export_data_dict = {"index": loc_index,
                            "dataset": loc_dataset,
                            "user_label": loc_user_label,
                            "nucleotide_label": loc_nucleotide_label,
                            "data": loc_data,
                            "data_name": loc_data_name}

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

    def get_filter_status(self, user_label, nucleotide_label):

        user_filter = self.export_settings.export_user_filter.currentText()
        nucleotide_filter = self.export_settings.export_nucleotide_filter.currentText()

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

    def build_json_dict(self, dataset_names = []):

        try:

            json_dataset_dict = {"metadata":{}, "data":{}}

            json_list_keys = ["donor", "acceptor", "efficiency", "DD", "AA", "DA", "AD", "states", "break_points", "crop_range", "gamma_ranges"]

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
                            json_localisation_dict[str(key)] = list(value)

                        elif key in json_var_keys:
                            json_localisation_dict[key] = value

                        else:

                            for key in json_list_keys:
                                json_localisation_dict[key] = []

                            for key in json_var_keys:
                                json_localisation_dict[key] = None

                    json_dataset_dict["data"][dataset_name].append(json_localisation_dict)

        except:
            print(traceback.format_exc())

        return json_dataset_dict



class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)