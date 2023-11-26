from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget

class _analysis_plotting_methods:

    def draw_analysis_graph(self):

        print(True)

















class CustomMatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(CustomMatplotlibWidget, self).__init__(parent)

        # Initialize matplotlib figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)

        # Setup layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Example plot
        self.axes.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 16])

    def mousePressEvent(self, event):
        # Handle mouse press event
        if event.modifiers():  # Check for any specific modifiers you need
            click_position = event.pos()
            # Convert click position to axes coordinates
            axes_coord = self.axes.transData.inverted().transform((click_position.x(), click_position.y()))
            print("Clicked at:", axes_coord)
            # You can add your logic here

        super().mousePressEvent(event)