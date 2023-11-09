import sys
from PyQt5 import QtWidgets, uic
from DeepGapSeq.GUI.mainwindow_gui import Ui_MainWindow
import pyqtgraph as pg
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QSizePolicy)



class CustomGraphicsLayoutWidget(pg.GraphicsLayoutWidget):

    def __init__(self, my_plugin_instance):
        super().__init__()
        self.my_plugin_instance = my_plugin_instance
        self.frame_position_memory = {}
        self.frame_position = None


class AnalysisGUI(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(AnalysisGUI, self).__init__()
        self.setupUi(self)  # Set up the user interface from Designer.

        self.graph_container = self.findChild(QWidget, "graph_container")
        self.setWindowTitle("DeepGapSeq-Analysis")  # Set the window title

        #create matplotib plot graph
        self.graph_container.setLayout(QVBoxLayout())
        self.graph_container.setMinimumWidth(100)

        self.trace_graph_canvas = CustomGraphicsLayoutWidget(self)
        self.graph_container.layout().addWidget(self.trace_graph_canvas)
        self.trace_graph_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.actionGapSeq.triggered.connect(self.test_event)

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
    # % gui qt
    # from DeepGapSeq.GUI.analysis_gui import start_gui
    # gui = start_gui(False)

    app = QtWidgets.QApplication.instance()  # Check if QApplication already exists
    if not app:  # Create QApplication if it doesn't exist
        app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(dark_stylesheet)  # Apply the dark theme
    window = AnalysisGUI()
    window.show()

    if blocking:
        app.exec()  # Start the event loop

    return window








