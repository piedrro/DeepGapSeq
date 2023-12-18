from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import traceback
import numpy as np
from DeepGapSeq._utils_worker import Worker


class CustomMatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

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

        histogram_data = {}

        try:

            if len(trace_data) == len(state_data) and len(trace_data) > 0:

                histogram_data = {"intensity": {}, "centres": {}, "noise": {}, "dwell_times": {}}

                trace_data = np.concatenate(trace_data)
                state_data = np.concatenate(state_data)

                change_indices = np.where(np.diff(state_data) != 0)[0] + 1

                split_trace_data = np.split(trace_data, change_indices)
                split_state_data = np.split(state_data, change_indices)

                for data, state in zip(split_trace_data, split_state_data):

                    state = state[0]

                    if state not in histogram_data["intensity"].keys():
                        histogram_data["intensity"][state] = []
                        histogram_data["centres"][state] = []
                        histogram_data["noise"][state] = []
                        histogram_data["dwell_times"][state] = []

                    instensity = data.tolist()
                    centres = [np.mean(data)]*len(data)
                    noise = [np.std(data)]*len(data)
                    dwell_time = len(data)

                    histogram_data["intensity"][state].extend(instensity)
                    histogram_data["centres"][state].extend(centres)
                    histogram_data["noise"][state].extend(noise)
                    histogram_data["dwell_times"][state].append(dwell_time)

        except:
            print(traceback.format_exc())
            pass

        return histogram_data

    def compute_trace_analysis(self, progress_callback=None):

        try:

            analysis_dataset = self.analysis_graph_data.currentText()
            mode = self.analysis_graph_mode.currentText()

            if analysis_dataset == "All Datasets":
                dataset_list = list(self.data_dict.keys())
            else:
                dataset_list = [analysis_dataset]

            if "Efficiency" in mode:
                mode = "Efficiency"

            trace_data_list = []
            state_data_list = []

            for dataset in dataset_list:
                for localisation_index, localisation_data in enumerate(self.data_dict[dataset]):

                    user_label = localisation_data["user_label"]
                    nucleotide_label = localisation_data["nucleotide_label"]

                    if self.get_filter_status("analysis", user_label, nucleotide_label) == False:

                        trace_data = localisation_data[mode]
                        state_data = localisation_data["states"]

                        if len(trace_data) == len(state_data) and len(trace_data) > 0:

                            if len(trace_data) == len(state_data):
                                trace_data_list.append(trace_data)
                                state_data_list.append(state_data)

        except:
            print(traceback.format_exc())
            pass

        histogram_data = self.compute_histograms(trace_data_list, state_data_list)

        return histogram_data




    def update_analysis_plot(self, histogram_data):

        if histogram_data != {}:

            histogram_dataset = self.analysis_histogram_dataset.currentText().lower()
            histogram_mode = self.analysis_histogram_mode.currentText().lower()
            bin_size = self.analysis_histogram_bin_size.currentText()
            plot_dataset = self.analysis_graph_data.currentText()
            plot_mode = self.analysis_graph_mode.currentText()

            if bin_size.isdigit():
                bin_size = int(bin_size)
            else:
                bin_size = "auto"

            if histogram_dataset == "dwell times":
                histogram_dataset = "dwell_times"

            if histogram_mode.lower() == "frequency":
                xlabel = plot_mode.capitalize() + " " + histogram_dataset.capitalize().replace("_", " ")
                ylabel = "Frequency"
                density = False
            else:
                xlabel = plot_mode.capitalize() + " " + histogram_dataset.capitalize().replace("_", " ")
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



