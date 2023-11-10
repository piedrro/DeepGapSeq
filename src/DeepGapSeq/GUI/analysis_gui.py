import sys
import pickle
from glob2 import glob
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
from DeepGapSeq.GUI.mainwindow_gui import Ui_MainWindow
from DeepGapSeq.GUI.plotsettings_gui import Ui_Form as plotsettings_gui
from DeepGapSeq.GUI.importwindow_gui import Ui_Form as importwindow_gui
import pyqtgraph as pg
from qtpy.QtWidgets import (QWidget, QDialog, QVBoxLayout, QSizePolicy,QSlider, QLabel, QFileDialog)
import numpy as np
import traceback
from functools import partial
import os
import pandas as pd
import uuid

class ImportSettingsWindow(QDialog, importwindow_gui):

    def __init__(self, parent):
        super(ImportSettingsWindow, self).__init__()
        self.setupUi(self)  # Set up the user interface from Designer.
        self.setWindowTitle("Import Settings")  # Set the window title

        self.AnalysisGUI = parent

    def keyPressEvent(self, event):
        try:
            if event.key() == Qt.Key_I:
                self.close()
            else:
                super().keyPressEvent(event)
        except:
            pass

class PlotSettingsWindow(QDialog, plotsettings_gui):

    def __init__(self, parent):
        super(PlotSettingsWindow, self).__init__()
        self.setupUi(self)  # Set up the user interface from Designer.
        self.setWindowTitle("Plot Settings")  # Set the window title

        self.AnalysisGUI = parent

    def keyPressEvent(self, event):
        try:
            if event.key() in [Qt.Key_A,Qt.Key_T,Qt.Key_C,Qt.Key_G]:
                self.AnalysisGUI.classify_traces(mode = "nucleotide", key = chr(event.key()))
            elif event.key() in [Qt.Key_1,Qt.Key_2,Qt.Key_3,Qt.Key_4,Qt.Key_5,Qt.Key_6,Qt.Key_7,Qt.Key_8,Qt.Key_9]:
                self.AnalysisGUI.classify_traces(mode = "user", key = chr(event.key()))
            elif event.key() in [Qt.Key_Left,Qt.Key_Right]:
                self.AnalysisGUI.update_localisation_number(event.key())
            elif event.key() == Qt.Key_X:
                self.AnalysisGUI.toggle_checkbox(self.AnalysisGUI.plot_settings.plot_showx)
            elif event.key() == Qt.Key_Y:
                self.AnalysisGUI.toggle_checkbox(self.AnalysisGUI.plot_settings.plot_showy)
            elif event.key() == Qt.Key_N:
                self.AnalysisGUI.toggle_checkbox(self.AnalysisGUI.plot_settings.plot_normalise)
            elif event.key() == Qt.Key_S:
                self.AnalysisGUI.toggle_checkbox(self.AnalysisGUI.plot_settings.plot_split_lines)
            elif event.key() == Qt.Key_Space:
                self.close()
            else:
                super().keyPressEvent(event)
        except:
            pass


class AnalysisGUI(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(AnalysisGUI, self).__init__()

        self.setupUi(self)  # Set up the user interface from Designer.
        self.setFocusPolicy(Qt.StrongFocus)

        self.plot_settings = PlotSettingsWindow(self)
        self.import_settings = ImportSettingsWindow(self)

        self.graph_container = self.findChild(QWidget, "graph_container")
        self.setWindowTitle("DeepGapSeq-Analysis")  # Set the window title

        #create matplotib plot graph
        self.graph_container.setLayout(QVBoxLayout())
        self.graph_container.setMinimumWidth(100)

        self.graph_canvas = CustomGraphicsLayoutWidget(self)
        self.graph_container.layout().addWidget(self.graph_canvas)
        self.graph_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.plotsettings_button = self.findChild(QtWidgets.QPushButton, "plotsettings_button")
        self.plotsettings_button.clicked.connect(self.toggle_plot_settings)

        self.import_settings.import_simulated.clicked.connect(self.import_simulated_data)
        self.actionImport_I.triggered.connect(self.toggle_import_settings)

        self.data_dict = {}

        self.plot_localisation_number.valueChanged.connect(lambda: self.update_slider_label("plot_localisation_number"))
        self.plot_localisation_number.valueChanged.connect(partial(self.plot_traces, update_plot=False))

        self.plot_mode.currentIndexChanged.connect(self.initialise_plot)
        self.plot_data.currentIndexChanged.connect(self.initialise_plot)

        self.plot_settings.plot_split_lines.stateChanged.connect(partial(self.plot_traces, update_plot=True))
        self.plot_settings.plot_showx.stateChanged.connect(partial(self.plot_traces, update_plot=True))
        self.plot_settings.plot_showy.stateChanged.connect(partial(self.plot_traces, update_plot=True))
        self.plot_settings.plot_user_filter.currentIndexChanged.connect(self.initialise_plot)
        self.plot_settings.plot_nucleotide_filter.currentIndexChanged.connect(self.initialise_plot)

        self.plot_settings.show_crop_range.stateChanged.connect(partial(self.plot_traces, update_plot=False))
        self.plot_settings.crop_plots.stateChanged.connect(partial(self.plot_traces, update_plot=False))

        self.plot_settings.plot_normalise.stateChanged.connect(partial(self.plot_traces, update_plot=False))
        self.plot_settings.show_plot_details.stateChanged.connect(partial(self.plot_traces, update_plot=False))
        self.plot_settings.show_correction_factor_ranges.stateChanged.connect(partial(self.plot_traces, update_plot=False))
        self.plot_settings.show_detected_states.stateChanged.connect(partial(self.plot_traces, update_plot=False))
        self.plot_settings.show_breakpoints.stateChanged.connect(partial(self.plot_traces, update_plot=False))

        self.import_settings.import_data.clicked.connect(self.import_data_files)

        # Set the color of the status bar text
        self.statusBar().setStyleSheet("QStatusBar{color: red;}")

    def update_crop_range(self, event, mode ="click"):

        slider_value = self.plot_localisation_number.value()
        localisation_number = self.localisation_numbers[slider_value]

        if mode == "click":

            for dataset_name in self.data_dict.keys():
                crop_range = self.data_dict[dataset_name][localisation_number]["crop_range"]

                crop_range.append(event)

                if len(crop_range) > 2:
                    crop_range.pop(0)

                self.data_dict[dataset_name][localisation_number]["crop_range"] = crop_range

                self.plot_traces(update_plot=False)

        elif mode == "drag":

            crop_range = list(event.getRegion())

            for region in self.unique_crop_regions:
                if region != event:
                    region.setRegion(crop_range)

            for dataset_name in self.data_dict.keys():
                self.data_dict[dataset_name][localisation_number]["crop_range"] = crop_range

    def calculate_fret_efficiency(self, donor, acceptor, gamma_correction =1):

        donor = np.array(donor)
        acceptor = np.array(acceptor)

        efficiency = acceptor / ((gamma_correction * donor) + acceptor)

        return efficiency

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

                        for i in range(0, len(data.columns), len(column_names)):

                            loc_data = {"donor": [], "acceptor": [],
                                        "efficiency": [], "states": [],
                                        "filter": False, "state_means": {},
                                        "user_label": 0, "nucleotide_label": 0,
                                        "break_points": [], "gamma_correction_ranges": [],
                                        "crop_range" : [],}

                            # Select the current group of four columns
                            group = data.iloc[:, i:i + len(column_names)]

                            group.columns = column_names
                            group_dict = group.to_dict(orient="list")

                            for key, value in group_dict.items():
                                loc_data[key] = value

                            if loc_data["efficiency"] == []:
                                loc_data["efficiency"] = self.calculate_fret_efficiency(loc_data["donor"], loc_data["acceptor"])

                            self.data_dict[dataset_name].append(loc_data)
                            n_traces += 1

            if n_traces > 1:

                self.print_notification(f"Imported {n_traces} traces")

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

    def closeEvent(self, event):
        self.plot_settings.close()
        self.import_settings.close()

    def keyPressEvent(self, event):
        try:

            if event.key() in [Qt.Key_A,Qt.Key_T,Qt.Key_C,Qt.Key_G]:
                self.classify_traces(mode = "nucleotide", key = chr(event.key()))
            elif event.key() in [Qt.Key_1,Qt.Key_2,Qt.Key_3,Qt.Key_4,Qt.Key_5,Qt.Key_6,Qt.Key_7,Qt.Key_8,Qt.Key_9]:
                self.classify_traces(mode = "user", key = chr(event.key()))
            elif event.key() in [Qt.Key_Left,Qt.Key_Right]:
                self.update_localisation_number(event.key())
            elif event.key() == Qt.Key_Space:
                self.toggle_plot_settings()
            elif event.key() == Qt.Key_I:
                self.toggle_import_settings()
            elif event.key() == Qt.Key_X:
                self.toggle_checkbox(self.plot_settings.plot_showx)
            elif event.key() == Qt.Key_Y:
                self.toggle_checkbox(self.plot_settings.plot_showy)
            elif event.key() == Qt.Key_N:
                self.toggle_checkbox(self.plot_settings.plot_normalise)
            elif event.key() == Qt.Key_S:
                self.toggle_checkbox(self.plot_settings.plot_split_lines)
            elif event.key() == Qt.Key_Escape:
                self.close()  # Close the application on pressing the Escape key
            # Add more key handling as needed
            else:
                super().keyPressEvent(event)  # Important to allow unhandled key events to propagate
        except:
            print(traceback.format_exc())
            pass




    def toggle_plot_settings(self):

        if self.plot_settings.isHidden() or self.plot_settings.isActiveWindow() == False:
            self.plot_settings.show()
            self.plot_settings.raise_()
            self.plot_settings.activateWindow()
            self.plot_settings.setFocus()

        else:
            self.plot_settings.hide()
            self.activateWindow()

    def toggle_import_settings(self):

        if self.import_settings.isHidden() or self.import_settings.isActiveWindow() == False:
            self.import_settings.show()
            self.import_settings.raise_()
            self.import_settings.activateWindow()
            self.import_settings.setFocus()
        else:
            self.import_settings.hide()
            self.activateWindow()

    def print_notification(self, message):
        self.statusBar().showMessage(message, 2000)

    def toggle_checkbox(self, checkbox):
        checkbox.setChecked(not checkbox.isChecked())

    def classify_traces(self, mode = "nucleotide", key= ""):

        if self.data_dict != {}:

            slider_value = self.plot_localisation_number.value()
            localisation_number = self.localisation_numbers[slider_value]

            for dataset_name in self.data_dict.keys():

                if mode == "user":
                    self.data_dict[dataset_name][localisation_number]["user_label"] = key
                else:
                    self.data_dict[dataset_name][localisation_number]["nucleotide_label"] = key

            self.plot_traces(update_plot=False)

    def update_localisation_number(self, key):

        if self.data_dict != {}:

            localisation_number = self.plot_localisation_number.value()
            dataset_name = self.plot_data.currentText()

            if key == Qt.Key_Left:
                new_localisation_number = localisation_number - 1
            elif key == Qt.Key_Right:
                new_localisation_number = localisation_number + 1
            else:
                new_localisation_number = localisation_number

            if new_localisation_number >= 0 and new_localisation_number < len(self.data_dict[dataset_name]):
                self.plot_localisation_number.setValue(new_localisation_number)
                self.plot_traces(update_plot=True)


    def filter_data_dict(self):

        user_filter = self.plot_settings.plot_user_filter.currentText()
        nucleotide_filter = self.plot_settings.plot_nucleotide_filter.currentText()

        if user_filter != "None":
            user_filter = int(user_filter)

        localisation_numbers = []

        if user_filter != "None" or nucleotide_filter != "None":

            for dataset_name, dataset_data in self.data_dict.items():

                if dataset_name != "All Datasets":

                    for localisation_number in range(len(dataset_data)):
                        user_label = int(dataset_data[localisation_number]["user_label"])
                        nucleotide_label = dataset_data[localisation_number]["nucleotide_label"]

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


                        if filter == True:
                            dataset_data[localisation_number]["filter"] = True
                        else:
                            localisation_numbers.append(localisation_number)

            self.localisation_numbers = np.unique(localisation_numbers)
            self.n_traces = len(self.localisation_numbers)

        else:
            self.n_traces = np.max([len(self.data_dict[dataset]) - 1 for dataset in self.plot_datasets])
            self.localisation_numbers = list(range(self.n_traces))

        return self.localisation_numbers, self.n_traces




    def update_slider_label(self, slider_name):

        label_name = slider_name + "_label"

        self.slider = self.findChild(QSlider, slider_name)
        self.label = self.findChild(QLabel, label_name)

        slider_value = self.slider.value()

        if slider_name == "localisation_aspect_ratio":
            slider_value = self.slider.value() / 10

        self.label.setText(str(slider_value))


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
                                                     "gamma_correction_ranges" : [],})

            self.compute_state_means()

            self.plot_data.clear()
            self.plot_data.addItems(list(self.data_dict.keys()))

            self.populate_plot_mode()

            self.plot_localisation_number.setValue(0)

            self.initialise_plot()

    def populate_plot_mode(self):

        self.plot_mode.clear()

        plot_names = []

        for dataset_name in self.data_dict.keys():
            for plot_name in self.data_dict[dataset_name][0].keys():
                if plot_name not in plot_names:
                    plot_names.append(plot_name)

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

    def compute_state_means(self):

        def _compute_state_means(data, labels):
            state_means = labels.copy()
            for state in np.unique(labels):
                state_means[state_means == state] = np.mean(data[state_means == state])
            return state_means

        for dataset_name, dataset_data in self.data_dict.items():
            for i, trace_data in enumerate(dataset_data):
                labels = trace_data["states"]
                for plot in ["donor", "acceptor", "efficiency"]:
                    plot_data = trace_data[plot]
                    if len(plot_data) > 0:
                        self.data_dict[dataset_name][i]["state_means"][plot] = _compute_state_means(plot_data, labels)

    def initialise_plot(self):

        try:

            plot_data = self.plot_data.currentText()
            plot_mode = self.plot_mode.currentText()
            
            if plot_data == "All Datasets":
                self.plot_datasets = list(self.data_dict.keys())
            elif plot_data != "" :
                self.plot_datasets = [plot_data]
            else:
                self.plot_datasets = []

            if len(self.plot_datasets) > 0:

                plot_labels = list(self.data_dict[self.plot_datasets[0]][0].keys())

                plot = False

                if plot_mode == "Donor" and set(["donor"]).issubset(plot_labels):
                    plot = True
                    self.n_plot_lines = 1
                    self.plot_line_labels = ["donor"]
                elif plot_mode == "Acceptor" and set(["acceptor"]).issubset(plot_labels):
                    plot = True
                    self.n_plot_lines = 1
                    self.plot_line_labels = ["acceptor"]
                elif plot_mode == "FRET Data" and set(["donor","acceptor"]).issubset(plot_labels):
                    plot = True
                    self.n_plot_lines = 2
                    self.plot_line_labels = ["donor","acceptor"]
                elif plot_mode == "FRET Efficiency" and set(["efficiency"]).issubset(plot_labels):
                    plot = True
                    self.n_plot_lines = 1
                    self.plot_line_labels = ["efficiency"]
                elif plot_mode == "FRET Data + FRET Efficiency" and set(["donor","acceptor","efficiency"]).issubset(plot_labels):
                    plot = True
                    self.n_plot_lines = 3
                    self.plot_line_labels = ["donor","acceptor","efficiency"]

                if plot == True:

                    self.localisation_numbers, self.n_traces = self.filter_data_dict()

                    if self.n_traces > 0:

                        slider_value = self.plot_localisation_number.value()
                        if slider_value >= self.n_traces:
                            slider_value = self.n_traces - 1
                            self.plot_localisation_number.setValue(slider_value)

                        self.plot_localisation_number.setMinimum(0)
                        self.plot_localisation_number.setMaximum(self.n_traces -1)

                        self.plot_traces(update_plot = True)

                    else:
                        self.print_notification("No traces to plot")

        except:
            print(traceback.format_exc())
            pass

    def check_plot_item_exists(self, plot, item):

        for item in plot.items:
            if item is item:
                return True

        return False

    def get_plot_item_instance(self, plot, instance):

        for item in plot.items:
            if isinstance(item, instance):
                return item

        return None

    def remove_plot_instance(self, plot, instance):

        for item in plot.items:
            if isinstance(item, instance):
                plot.removeItem(item)



    def plot_traces(self, update_plot = False):

        try:

            if hasattr(self, "plot_grid") == False or update_plot == True:
                self.plot_grid = self.update_plot_layout()

            if self.plot_grid != {}:

                plot_mode = self.plot_mode.currentText()

                slider_value = self.plot_localisation_number.value()
                crop_plots = self.plot_settings.crop_plots.isChecked()
                show_crop_range = self.plot_settings.show_crop_range.isChecked()
                localisation_number = self.localisation_numbers[slider_value]

                for plot_index, grid in enumerate(self.plot_grid.values()):

                    plot_dataset = grid["plot_dataset"]
                    sub_axes = grid["sub_axes"]
                    plot_lines = grid["plot_lines"]
                    plot_lines_labels = grid["plot_lines_labels"]
                    title_plot = grid["title_plot"]
                    efficiency_plot = grid["efficiency_plot"]

                    user_label = self.data_dict[plot_dataset][localisation_number]["user_label"]
                    nucleotide_label = self.data_dict[plot_dataset][localisation_number]["nucleotide_label"]
                    crop_range = self.data_dict[plot_dataset][localisation_number]["crop_range"]

                    if crop_plots == True and len(crop_range) == 2:
                        crop_range = sorted(crop_range)
                        crop_range = [int(crop_range[0]), int(crop_range[1])]

                        plot_details = f"{plot_dataset} [#:{localisation_number} C:{user_label}  N:{nucleotide_label} Cropped:{True}]"

                    else:
                        plot_details = f"{plot_dataset} [#:{localisation_number} C:{user_label}  N:{nucleotide_label} Cropped:{False}]"

                    title_plot.setTitle(plot_details)

                    for plot in sub_axes:
                        for item in plot.items:
                            pass
                            # if item == self.crop_region:
                            #     self.crop_region = plot.removeItem(item)
                            # if isinstance(item, pg.LinearRegionItem):
                            #     plot.removeItem(item)
                            # if isinstance(item, pg.PlotDataItem):
                            #     if item.name() == "hmm_mean":
                            #         plot.removeItem(item)
                            # elif item.name() == "hmm_mean":
                            #     plot.removeItem(item)

                    for line_index, (plot, line,  plot_label) in enumerate(zip(sub_axes, plot_lines, plot_lines_labels)):

                        legend = plot.legend
                        data = self.data_dict[plot_dataset][localisation_number][plot_label]

                        break_points = self.data_dict[plot_dataset][localisation_number]["break_points"]
                        state_means = self.data_dict[plot_dataset][localisation_number]["state_means"][plot_label]
                        gamma_correction_ranges = self.data_dict[plot_dataset][localisation_number]["gamma_correction_ranges"]

                        if crop_plots == True and len(crop_range) == 2:
                            crop_range = sorted(crop_range)
                            crop_range = [int(crop_range[0]), int(crop_range[1])]

                            data = data[crop_range[0]:crop_range[1]]
                            state_means = state_means[crop_range[0]:crop_range[1]]

                        if self.plot_settings.plot_normalise.isChecked() and "Efficiency" not in plot_label:
                            data = (data - np.min(data)) / (np.max(data) - np.min(data))

                        plot_line = plot_lines[line_index]
                        plot_line.setData(data)
                        plot.enableAutoRange()
                        plot.autoRange()

                        if self.plot_settings.show_detected_states.isChecked() and len(state_means) > 0:

                            if self.plot_settings.plot_normalise.isChecked():
                                state_means = (state_means - np.min(state_means)) / (np.max(state_means) - np.min(state_means))

                            if "Efficiency" in plot_mode:
                                if line_index == len(sub_axes) - 1:
                                    plot.plot(state_means, pen=pg.mkPen('b', width=3), name="hmm_mean")
                                    legend.removeItem("hmm_mean")
                            else:
                                if line_index == 0:
                                    plot.plot(state_means, pen=pg.mkPen('b', width=3), name="hmm_mean")
                                    legend.removeItem("hmm_mean")

                        if crop_plots == False and show_crop_range == True and len(crop_range) == 2:

                            try:
                                # random name for crop region
                                crop_region_name = str(uuid.uuid4())

                                if hasattr(self, crop_region_name) == False:
                                    setattr(self, crop_region_name, pg.LinearRegionItem(brush=pg.mkBrush(255, 0, 0, 50)))

                                crop_region = getattr(self, crop_region_name)

                                if self.get_plot_item_instance(plot, pg.LinearRegionItem) is None:
                                    plot.addItem(crop_region)
                                    crop_region.setRegion(crop_range)
                                    self.unique_crop_regions.append(crop_region)
                                else:
                                    crop_region.setRegion(crop_range)
                            except:
                                pass
                        else:
                            self.remove_plot_instance(plot, pg.LinearRegionItem)

                        if len(break_points) > 2 and self.show_cpd_breakpoints.isChecked():

                            break_points = np.unique(break_points).tolist()

                            for break_point in break_points[1:-1]:
                                bp = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('g', width=3))
                                plot.addItem(bp)
                                bp.setPos(break_point)

                        if efficiency_plot == True:

                            if "Efficiency" not in plot_label:

                                if len(gamma_correction_ranges) > 0:

                                    for gamma_correction_range in gamma_correction_ranges:
                                        gc = pg.LinearRegionItem(values=gamma_correction_range, brush=pg.mkBrush(0, 0, 255, 50))
                                        plot.addItem(gc)


                for crop_region in self.unique_crop_regions:
                    crop_region.sigRegionChanged.connect(partial(self.update_crop_range, mode="drag"))

        except:
            print(traceback.format_exc())
            pass






    def update_plot_layout(self):

        try:

            self.plot_grid = {}
            self.unique_crop_regions = []

            self.graph_canvas.clear()

            plot_mode = self.plot_mode.currentText()
            split = self.plot_settings.plot_split_lines.isChecked()

            efficiency_plot = False

            for plot_index, plot_dataset in enumerate(self.plot_datasets):

                if plot_mode == "FRET Data + FRET Efficiency" and split==False:

                    layout = pg.GraphicsLayout()
                    self.graph_canvas.addItem(layout, row=plot_index, col=0)

                    sub_plots = []

                    for line_index in range(2):
                        p = CustomPlot()

                        layout.addItem(p, row=line_index, col=0)

                        if self.plot_settings.plot_showy.isChecked() == False:
                            p.hideAxis('left')

                        if self.plot_settings.plot_showx.isChecked() == False:
                            p.hideAxis('bottom')
                        elif line_index != 1:
                            p.hideAxis('bottom')

                        sub_plots.append(p)

                    sub_plots = [sub_plots[0] for i in range(self.n_plot_lines - 1)] + [sub_plots[1]]

                    for j in range(1, len(sub_plots)):
                        sub_plots[j].setXLink(sub_plots[0])

                    efficiency_plot = True

                elif split == True and self.n_plot_lines > 1:

                    layout = pg.GraphicsLayout()
                    self.graph_canvas.addItem(layout, row=plot_index, col=0)

                    sub_plots = []

                    for line_index in range(self.n_plot_lines):
                        p = CustomPlot()

                        layout.addItem(p, row=line_index, col=0)

                        if self.plot_settings.plot_showy.isChecked() == False:
                            p.hideAxis('left')

                        if self.plot_settings.plot_showx.isChecked() == False:
                            p.hideAxis('bottom')
                        if line_index != self.n_plot_lines - 1:
                            p.hideAxis('bottom')

                        sub_plots.append(p)

                    for j in range(1, len(sub_plots)):
                        sub_plots[j].setXLink(sub_plots[0])

                else:
                    layout = self.graph_canvas

                    p = CustomPlot()

                    p.hideAxis('top')
                    p.hideAxis('right')

                    if self.plot_settings.plot_showy.isChecked() == False:
                        p.hideAxis('left')
                    if self.plot_settings.plot_showx.isChecked() == False:
                        p.hideAxis('bottom')

                    layout.addItem(p, row=plot_index, col=0)

                    sub_plots = []

                    for line_index in range(self.n_plot_lines):
                        sub_plots.append(p)

                localisation_number = self.plot_localisation_number.value()

                user_label = self.data_dict[plot_dataset][localisation_number]["user_label"]
                nucleotide_label = self.data_dict[plot_dataset][localisation_number]["nucleotide_label"]

                plot_lines = []
                plot_lines_labels = []

                for axes_index, plot in enumerate(sub_plots):
                    line_label = self.plot_line_labels[axes_index]
                    line_format = pg.mkPen(color=100 + axes_index * 100, width=2)
                    plot_line = plot.plot(np.zeros(10), pen=line_format, name=line_label)
                    plot.enableAutoRange()
                    plot.autoRange()

                    legend = plot.legend
                    legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))

                    plot_details = f"{plot_dataset}   #:{localisation_number} C:{user_label}  N:{nucleotide_label}"

                    if axes_index == 0:
                        plot.setTitle(plot_details)
                        title_plot = plot

                    plotmeta = plot.metadata
                    plotmeta[axes_index] = {"plot_dataset": plot_dataset, "line_label": line_label}

                    plot_lines.append(plot_line)
                    plot_lines_labels.append(line_label)

                    self.plot_grid[plot_index] = {
                        "sub_axes": sub_plots,
                        "title_plot": title_plot,
                        "plot_lines": plot_lines,
                        "plot_dataset": plot_dataset,
                        "plot_index": plot_index,
                        "n_plot_lines": self.n_plot_lines,
                        "split": split,
                        "plot_lines_labels": plot_lines_labels,
                        "efficiency_plot": efficiency_plot,
                        }

            plot_list = []
            for plot_index, grid in enumerate(self.plot_grid.values()):
                sub_axes = grid["sub_axes"]
                sub_plots = []
                for plot in sub_axes:
                    sub_plots.append(plot)
                    plot_list.append(plot)
            for i in range(1, len(plot_list)):
                plot_list[i].setXLink(plot_list[0])
            plot.getViewBox().sigXRangeChanged.connect(lambda: auto_scale_y(plot_list))


        except:
            print(traceback.format_exc())
            pass

        return self.plot_grid





    # Slot method to handle the menu selection event
    def test_event(self):
        print(True)
        print(False)
        print("test_event")




def start_gui(blocking=True):

    dark_stylesheet = """
        QMainWindow {background-color: #2e2e2e;}
        QMenuBar {background-color: #2e2e2e;}
        QMenuBar::item {background-color: #2e2e2e;color: #ffffff;}
        QMenu {background-color: #2e2e2e;border: 1px solid #1e1e1e;}
        QMenu::item {color: #ffffff;}
        QMenu::item:selected {background-color: #5e5e5e;}
    """

    # to launch the GUI from the console such that it is editable:
    # from DeepGapSeq.GUI.analysis_gui import start_gui
    # gui = start_gui(False)

    app = QtWidgets.QApplication.instance()  # Check if QApplication already exists
    if not app:  # Create QApplication if it doesn't exist
        app = QtWidgets.QApplication(sys.argv)

    app.setStyleSheet(dark_stylesheet)  # Apply the dark theme
    mainwindow = AnalysisGUI()
    mainwindow.show()

    if blocking:
        app.exec()  # Start the event loop

    return mainwindow









def auto_scale_y(sub_plots):

    try:

        for p in sub_plots:
            data_items = p.listDataItems()

            if not data_items:
                return

            y_min = np.inf
            y_max = -np.inf

            # Get the current x-range of the plot
            plot_x_min, plot_x_max = p.getViewBox().viewRange()[0]

            for index, item in enumerate(data_items):
                y_data = item.yData
                x_data = item.xData

                # Get the indices of y_data that lies within the current x-range
                idx = np.where((x_data >= plot_x_min) & (x_data <= plot_x_max))

                if len(idx[0]) > 0:  # If there's any data within the x-range
                    y_min = min(y_min, y_data[idx].min())
                    y_max = max(y_max, y_data[idx].max())

                if plot_x_min < 0:
                    x_min = 0
                else:
                    x_min = plot_x_min

                if plot_x_max > x_data.max():
                    x_max = x_data.max()
                else:
                    x_max = plot_x_max

            p.getViewBox().setYRange(y_min, y_max, padding=0)
            p.getViewBox().setXRange(x_min, x_max, padding=0)

    except:
        pass


class CustomPlot(pg.PlotItem):

    def __init__(self, title="", colour="", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metadata = {}

        self.setMenuEnabled(False)

        legend = self.addLegend(offset=(10, 10))
        legend.setBrush('w')
        legend.setLabelTextSize("10pt")
        self.hideAxis('top')
        self.hideAxis('right')

        self.title = title
        self.colour = colour

        if self.title != "":
            self.setLabel('top', text=title)

    def setMetadata(self, metadata_dict):
        self.metadata = metadata_dict

    def getMetadata(self):
        return self.metadata


class CustomGraphicsLayoutWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.frame_position_memory = {}
        self.frame_position = None

    def mousePressEvent(self, event):

        if hasattr(self.parent, "plot_grid"):

            if event.modifiers() & Qt.ControlModifier:

                xpos = self.get_event_x_postion(event, mode="click")
                self.parent.update_crop_range(xpos)

            super().mousePressEvent(event)  # Process the event further

    def keyPressEvent(self, event):

        if hasattr(self.parent, "plot_grid"):

            pass

            super().keyPressEvent(event)  # Process the event further

    def get_event_x_postion(self, event,  mode="click"):

        self.xpos = None

        if hasattr(self.parent, "plot_grid"):

            if mode == "click":
                pos = event.pos()
                self.scene_pos = self.mapToScene(pos)
            else:
                pos = QCursor.pos()
                self.scene_pos = self.mapFromGlobal(pos)

            # print(f"pos: {pos}, scene_pos: {scene_pos}")

            # Iterate over all plots
            plot_grid = self.parent.plot_grid

            for plot_index, grid in enumerate(plot_grid.values()):
                sub_axes = grid["sub_axes"]

                for axes_index in range(len(sub_axes)):
                    plot = sub_axes[axes_index]

                    viewbox = plot.vb
                    plot_coords = viewbox.mapSceneToView(self.scene_pos)

            self.xpos = plot_coords.x()

        return self.xpos

