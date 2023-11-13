
import json
import os.path
import traceback
from PyQt5.QtWidgets import QFileDialog

class _export_methods:


    def get_export_path(self, extension="json"):

        import_paths = [value[0]["import_path"] for key, value in self.data_dict.items()]

        export_filename = os.path.basename(import_paths[0])
        export_dir = os.path.dirname(import_paths[0])
        export_filename = export_filename.split(".")[0] + f"_deepgapseq.{extension}"

        export_path = os.path.join(export_dir, export_filename)

        return export_path

    def export_gapseq_json(self):

        try:

            # export gapseq data as a json file

            if self.data_dict != {}:

                desktop = os.path.expanduser("~/Desktop")
                export_path = self.get_export_path()
                export_path, _ = QFileDialog.getSaveFileName(self, "Save file", export_path, "json (*.json)")

                directory = os.path.dirname(export_path)
                file_name = os.path.basename(export_path)

                if export_path != "" and os.path.isdir(directory):

                    json_dataset_dict = {}

                    json_list_keys = ["donor", "acceptor", "efficiency",
                                      "DD", "AA", "DA", "AD",
                                      "states", "break_points",
                                      "crop_range","gamma_ranges"]

                    json_var_keys = ["user_label",
                                     "nucleotide_label",
                                     "import_path"]

                    for dataset_name, dataset_data in self.data_dict.items():

                        if dataset_name not in json_dataset_dict.keys():
                            json_dataset_dict[dataset_name] = []

                        for localisation_number, localisation_data in enumerate(dataset_data):

                            json_localisation_dict = {}

                            for key, value in localisation_data.items():

                                if key in json_list_keys:

                                    json_localisation_dict[key] = [list(value)]

                                elif key in json_var_keys:

                                    json_localisation_dict[key] = value

                            json_dataset_dict[dataset_name].append(json_localisation_dict)

                    with open(export_path, "w") as f:
                        json.dump(json_dataset_dict, f)

                    self.print_notification(f"Exported data to {export_path}")

        except:
            print(traceback.format_exc())
