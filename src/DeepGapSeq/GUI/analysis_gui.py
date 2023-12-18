import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from qtpy.QtCore import QThreadPool

from DeepGapSeq.GUI.GUI_windows.mainwindow_gui import Ui_MainWindow
from DeepGapSeq.GUI.GUI_windows.plotsettings_gui import Ui_Form as plotsettings_gui
from DeepGapSeq.GUI.GUI_windows.importwindow_gui import Ui_Form as importwindow_gui
from DeepGapSeq.GUI.GUI_windows.exportsettings_gui import Ui_Form as exportwindow_gui
from DeepGapSeq.GUI.GUI_windows.fittingwindow_gui import Ui_Form as fittingwindow_gui

from qtpy.QtWidgets import (QWidget, QDialog, QVBoxLayout, QSizePolicy,QSlider, QLabel)
import traceback
from functools import partial

from DeepGapSeq.GUI.gui_trace_plot_utils import CustomPyQTGraphWidget, _trace_plotting_methods
from DeepGapSeq.GUI.gui_analysis_plot_utils import  CustomMatplotlibWidget, _analysis_plotting_methods
from DeepGapSeq.GUI.gui_import_utils import _import_methods
from DeepGapSeq.GUI.gui_export_utils import _export_methods
from DeepGapSeq.GUI.gui_ebfret_utils import _ebFRET_methods
from DeepGapSeq.GUI.gui_deeplasi_utils import _DeepLasi_methods


class FittingWindow(QDialog, fittingwindow_gui):

    def __init__(self, parent):
        super(FittingWindow, self).__init__()

        self.setupUi(self)  # Set up the user interface from Designer.
        self.setWindowTitle("Analysis Settings")  # Set the window title

        self.AnalysisGUI = parent

    def keyPressEvent(self, event):
        try:
            if event.key() == Qt.Key_F:
                self.close()
            else:
                super().keyPressEvent(event)
        except:
            pass

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

class ExportSettingsWindow(QDialog, exportwindow_gui):

    def __init__(self, parent):
        super(ExportSettingsWindow, self).__init__()
        self.setupUi(self)  # Set up the user interface from Designer.
        self.setWindowTitle("Export Settings")  # Set the window title

        self.AnalysisGUI = parent

    def keyPressEvent(self, event):
        try:
            if event.key() == Qt.Key_E:
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



class AnalysisGUI(QtWidgets.QMainWindow,
    Ui_MainWindow, _trace_plotting_methods,
    _import_methods, _export_methods,
    _ebFRET_methods, _DeepLasi_methods,
    _analysis_plotting_methods):

    def __init__(self):
        super(AnalysisGUI, self).__init__()

        self.setupUi(self)  # Set up the user interface from Designer.
        self.setFocusPolicy(Qt.StrongFocus)

        self.plot_settings = PlotSettingsWindow(self)
        self.import_settings = ImportSettingsWindow(self)
        self.export_settings = ExportSettingsWindow(self)
        self.fitting_window = FittingWindow(self)


        self.setWindowTitle("DeepGapSeq-Analysis")  # Set the window title

        #create pyqt graph container
        self.graph_container = self.findChild(QWidget, "graph_container")
        self.graph_container.setLayout(QVBoxLayout())
        self.graph_container.setMinimumWidth(100)

        self.graph_canvas = CustomPyQTGraphWidget(self)
        self.graph_container.layout().addWidget(self.graph_canvas)
        self.graph_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # create matlotlib graph container
        self.analysis_graph_container = self.findChild(QWidget, "analysis_graph_container")
        self.analysis_graph_container.setLayout(QVBoxLayout())
        self.analysis_graph_container.setMinimumWidth(100)

        self.analysis_graph_canvas = CustomMatplotlibWidget(self)
        self.analysis_graph_container.layout().addWidget(self.analysis_graph_canvas)
        self.analysis_graph_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        self.plotsettings_button = self.findChild(QtWidgets.QPushButton, "plotsettings_button")
        self.plotsettings_button.clicked.connect(self.toggle_plot_settings)

        self.import_settings.import_simulated.clicked.connect(self.import_simulated_data)
        self.actionImport_I.triggered.connect(self.toggle_import_settings)
        self.actionExport_E.triggered.connect(self.toggle_export_settings)
        self.actionFit_Hidden_States_F.triggered.connect(self.toggle_fitting_window)

        self.export_settings.export_gapseq.clicked.connect(self.initialise_file_export)
        self.export_settings.export_json.clicked.connect(self.initialise_json_export)
        self.export_settings.export_excel.clicked.connect(self.initialise_excel_export)
        self.export_settings.export_origin.clicked.connect(self.initialise_origin_export)

        self.fitting_window.ebfret_connect_matlab.clicked.connect(self.launch_ebFRET)
        self.fitting_window.ebfret_run_analysis.clicked.connect(self.run_ebFRET_analysis)
        self.fitting_window.ebfret_visualisation_state.currentIndexChanged.connect(self.gapseq_visualise_ebfret)

        self.fitting_window.deeplasi_detect_states.clicked.connect(self.detect_deeplasi_states)

        self.plot_settings.crop_reset_active.clicked.connect(partial(self.reset_crop_ranges, mode = "active"))
        self.plot_settings.crop_reset_all.clicked.connect(partial(self.reset_crop_ranges, mode = "all"))

        self.data_dict = {}

        self.plot_localisation_number.valueChanged.connect(lambda: self.update_slider_label("plot_localisation_number"))
        self.plot_localisation_number.valueChanged.connect(partial(self.plot_traces, update_plot=False))

        self.plot_mode.currentIndexChanged.connect(self.initialise_plot)
        self.plot_data.currentIndexChanged.connect(self.initialise_plot)

        self.analysis_graph_data.currentIndexChanged.connect(self.initialise_analysis_plot)
        self.analysis_graph_mode.currentIndexChanged.connect(self.initialise_analysis_plot)
        self.analysis_nucleotide_filter.currentIndexChanged.connect(self.initialise_analysis_plot)
        self.analysis_user_filter.currentIndexChanged.connect(self.initialise_analysis_plot)
        self.analysis_crop_traces.stateChanged.connect(self.initialise_analysis_plot)
        self.analysis_histogram_dataset.currentIndexChanged.connect(self.initialise_analysis_plot)
        self.analysis_histogram_mode.currentIndexChanged.connect(self.initialise_analysis_plot)
        self.analysis_histogram_bin_size.currentIndexChanged.connect(self.initialise_analysis_plot)


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
        self.plot_settings.plot_downsample.currentIndexChanged.connect(partial(self.plot_traces, update_plot=False))

        self.plot_data.currentIndexChanged.connect(self.update_plot_mode_combo)

        self.import_settings.import_data.clicked.connect(self.import_data_files)
        self.import_settings.import_json.clicked.connect(self.import_gapseq_json)

        # Set the color of the status bar text
        self.statusBar().setStyleSheet("QStatusBar{color: red;}")

        self.threadpool = QThreadPool()

        self.export_settings.export_mode.currentIndexChanged.connect(self.format_export_settings)

        self.format_export_settings()

        self.current_dialog = None
        self.updating_combos = False

    def dev_function(self):

        self.populate_combos()

    def gui_progrssbar(self,progress, name):

        if name.lower() == "deeplasi":
            self.fitting_window.deeplasi_progressbar.setValue(progress)


    def format_export_settings(self):

        export_mode = self.export_settings.export_mode.currentText()

        if "csv" in export_mode.lower():

            self.export_settings.export_separator.clear()
            self.export_settings.export_separator.addItems(["Comma"])

        else:
            self.export_settings.export_separator.clear()
            self.export_settings.export_separator.addItems(["Tab","Comma", "Space"])


    def closeEvent(self, event):
        self._close_ebFRET()
        self.plot_settings.close()
        self.import_settings.close()
        self.export_settings.close()
        self.fitting_window.close()

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
            elif event.key() == Qt.Key_E:
                self.toggle_export_settings()
            elif event.key() == Qt.Key_F:
                self.toggle_fitting_window()
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
            elif event.key() == Qt.Key_D:
                self.dev_function()
            # Add more key handling as needed
            else:
                super().keyPressEvent(event)  # Important to allow unhandled key events to propagate
        except:
            print(traceback.format_exc())
            pass


    def toggle_fitting_window(self):

        if self.fitting_window.isHidden() or self.fitting_window.isActiveWindow() == False:

            self.populate_ebFRET_options()

            self.fitting_window.show()
            self.fitting_window.raise_()
            self.fitting_window.activateWindow()
            self.fitting_window.setFocus()

            # self.fitting_window.exec_()

            self.current_dialog = self.fitting_window
        else:
            self.fitting_window.hide()
            self.activateWindow()

    def toggle_plot_settings(self):

        if self.plot_settings.isHidden() or self.plot_settings.isActiveWindow() == False:
            self.plot_settings.show()
            self.plot_settings.raise_()
            self.plot_settings.activateWindow()
            self.plot_settings.setFocus()
            # self.plot_settings.exec_()

            self.current_dialog = self.fitting_window

        else:
            self.plot_settings.hide()
            self.activateWindow()

    def toggle_import_settings(self):

        if self.import_settings.isHidden() or self.import_settings.isActiveWindow() == False:
            self.import_settings.show()
            self.import_settings.raise_()
            self.import_settings.activateWindow()
            self.import_settings.setFocus()
            # self.import_settings.exec_()

            self.current_dialog = self.fitting_window
        else:
            self.import_settings.hide()
            self.activateWindow()

    def toggle_export_settings(self):

        if self.export_settings.isHidden() or self.export_settings.isActiveWindow() == False:

            self.export_settings.show()
            self.export_settings.raise_()
            self.export_settings.activateWindow()
            self.export_settings.setFocus()
            # self.export_settings.exec_()
        else:
            self.export_settings.hide()
            self.activateWindow()


    def print_notification(self, message):
        self.statusBar().showMessage(message, 5000)
        print(message)

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


    def update_slider_label(self, slider_name):

        label_name = slider_name + "_label"

        self.slider = self.findChild(QSlider, slider_name)
        self.label = self.findChild(QLabel, label_name)

        slider_value = self.slider.value()
        self.label.setText(str(slider_value))





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

