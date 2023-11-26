from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import traceback
import numpy as np
from DeepGapSeq._utils_worker import Worker


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

    def mousePressEvent(self, event):
        # Handle mouse press event
        if event.modifiers():  # Check for any specific modifiers you need
            click_position = event.pos()
            # Convert click position to axes coordinates
            axes_coord = self.axes.transData.inverted().transform((click_position.x(), click_position.y()))
            print("Clicked at:", axes_coord)
            # You can add your logic here

        super().mousePressEvent(event)


class _analysis_plotting_methods:

    def initialise_analysis_plot(self):

        try:

            if self.updating_combos == False and self.data_dict != {}:

                worker = Worker(self.compute_trace_analysis)
                worker.signals.result.connect(self.update_analysis_plot)
                self.threadpool.start(worker)

        except:
            print(traceback.format_exc())


    def compute_histograms(self, trace_data, state_data):

        histogram_data = {"intensity": {}, "centres": {}, "noise": {}}

        try:

            for trace, states in zip(trace_data, state_data):

                trace = np.array(trace)
                states = np.array(states)

                for state in np.unique(states):

                    if state not in histogram_data["intensity"].keys():
                        histogram_data["intensity"][state] = []
                        histogram_data["centres"][state] = []
                        histogram_data["noise"][state] = []

                    state_intensity_data = trace[states == state].tolist()

                    histogram_data["intensity"][state].extend(state_intensity_data)
                    histogram_data["centres"][state].append(np.mean(state_intensity_data))
                    histogram_data["noise"][state].append(np.std(state_intensity_data))


        except:
            print(traceback.format_exc())
            pass

        return histogram_data

    def compute_trace_analysis(self, progress_callback=None):

        try:

            dataset_name = self.analysis_graph_data.currentText()
            mode = self.analysis_graph_mode.currentText()

            if "efficiency" in mode.lower():
                mode = "efficiency"

            trace_data_list = []
            state_data_list = []

            for localisation_index, localisation_data in enumerate(self.data_dict[dataset_name]):

                user_label = localisation_data["user_label"]
                nucleotide_label = localisation_data["nucleotide_label"]

                if self.get_filter_status("analysis", user_label, nucleotide_label) == False:

                    trace_data = localisation_data[mode.lower()]
                    state_data = localisation_data["states"]

                    if len(trace_data) == len(state_data):
                        trace_data_list.append(trace_data)
                        state_data_list.append(state_data)

        except:
            print(traceback.format_exc())
            pass

        histogram_data = self.compute_histograms(trace_data_list, state_data_list)

        return histogram_data




    def update_analysis_plot(self, histogram_data):

        histogram_dataset = self.analysis_histogram_dataset.currentText().lower()
        histogram_mode = self.analysis_histogram_mode.currentText().lower()
        bin_size = int(self.analysis_histogram_bin_size.currentText())
        plot_dataset = self.analysis_graph_data.currentText()
        plot_mode = self.analysis_graph_mode.currentText()

        if histogram_mode.lower() == "frequency":
            xlabel = plot_mode.capitalize() + " " + histogram_dataset.capitalize()
            ylabel = "Frequency"
            density = False
        else:
            xlabel = plot_mode.capitalize() + " " + histogram_dataset.capitalize()
            ylabel = "Probability"
            density = True

        try:

            self.analysis_graph_canvas.axes.clear()

            all_data = []

            for state in histogram_data[histogram_dataset].keys():

                plot_label = f"State {int(state)}"

                histogram_values = histogram_data[histogram_dataset][state]

                all_data.extend(histogram_values)

                self.analysis_graph_canvas.axes.hist(histogram_values,
                    bins=bin_size,
                    alpha=0.5,
                    label=plot_label,
                    density=density)

            self.analysis_graph_canvas.axes.legend()

            self.analysis_graph_canvas.axes.set_xlabel(xlabel)
            self.analysis_graph_canvas.axes.set_ylabel(ylabel)

            lower_limit, upper_limit = np.percentile(all_data, [0.1, 99.9])
            lower_limit = lower_limit - (upper_limit - lower_limit) * 0.1
            upper_limit = upper_limit + (upper_limit - lower_limit) * 0.1

            self.analysis_graph_canvas.axes.set_xlim(lower_limit, upper_limit)

            self.analysis_graph_canvas.canvas.draw()
        except:
            print(traceback.format_exc())
            pass



