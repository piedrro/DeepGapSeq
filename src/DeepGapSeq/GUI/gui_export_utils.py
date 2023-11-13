
import json
import os.path
import traceback
from PyQt5.QtWidgets import QFileDialog

class _export_methods:


    def initialise_export(self):

        export_mode = self.export_settings.export_mode.currentText()
        export_location = self.export_settings.export_location.currentText()
        split_datasets = self.export_settings.export_split_datasets.isChecked()
        export_paths = self.get_export_paths()

        n_datasets = len(self.data_dict.keys())

        if export_location == "Select Directory":
            export_dir = os.path.dirname(export_paths[0])

            export_dir = QFileDialog.getExistingDirectory(self, "Select Directory", export_dir)

            if export_dir != "":
                export_paths = [os.path.join(export_dir, os.path.basename(export_path)) for export_path in export_paths]
                export_paths = [os.path.abspath(export_path) for export_path in export_paths]

        if export_mode == "GapSeq (.json)":

                self.export_gapseq_json(export_paths, split_datasets)


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



    def export_gapseq_json(self, export_paths = [], split_datasets = False):

        try:

            # export gapseq data as a json file

            if self.data_dict != {}:

                if split_datasets == False:

                    json_dataset_dict = self.build_json_dict()

                    export_path = export_paths[0]

                    with open(export_path, "w") as f:
                        json.dump(json_dataset_dict, f)

                    self.print_notification(f"Exported data to {export_path}")

                else:

                    for dataset_name, export_path in zip(self.data_dict.keys(),export_paths):

                        json_dataset_dict = self.build_json_dict(dataset_names=[dataset_name])

                        with open(export_path, "w") as f:
                            json.dump(json_dataset_dict, f)

                        self.print_notification(f"Exported data to {export_path}")

        except:
            print(traceback.format_exc())

    def build_json_dict(self, dataset_names = []):

        json_dataset_dict = {}

        json_list_keys = ["donor", "acceptor", "efficiency", "DD", "AA", "DA", "AD", "states", "break_points", "crop_range", "gamma_ranges"]

        json_var_keys = ["user_label", "nucleotide_label", "import_path"]

        if len(dataset_names) == 0:
            dataset_names = self.data_dict.keys()

        user_filter = self.export_settings.export_user_filter.currentText()
        nucleotide_filter = self.export_settings.export_nucleotide_filter.currentText()

        for dataset_name in dataset_names:

            dataset_data = self.data_dict[dataset_name]

            if dataset_name not in json_dataset_dict.keys():
                json_dataset_dict[dataset_name] = []

            for localisation_number, localisation_data in enumerate(dataset_data):

                json_localisation_dict = {}

                user_label = localisation_data["user_label"]
                nucleotide_label = localisation_data["nucleotide_label"]

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

                if filter == False:

                    for key, value in localisation_data.items():
                        if key in json_list_keys:
                            json_localisation_dict[key] = list(value)

                        elif key in json_var_keys:
                            json_localisation_dict[key] = value

                else:

                    for key in json_list_keys:
                        json_localisation_dict[key] = []

                    for key in json_var_keys:
                        json_localisation_dict[key] = None

                json_dataset_dict[dataset_name].append(json_localisation_dict)


        return json_dataset_dict