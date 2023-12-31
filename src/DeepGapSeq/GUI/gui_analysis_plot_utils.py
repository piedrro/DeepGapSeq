from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import traceback
import numpy as np
from DeepGapSeq._utils_worker import Worker
import threading
import queue
import traceback
import numpy as np
import matplotlib.pyplot as plt


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

            self.print_notification("Drawing analysis plot...")

            if self.data_dict != {}:

                worker = Worker(self._initialise_analysis_plot)
                worker.signals.finished.connect(self.update_analysis_graph_canvas)
                self.threadpool.start(worker)
                pass

        except:
            print(traceback.format_exc())


    def _initialise_analysis_plot(self, progress_callback=None):

        try:

            self.analysis_graph_canvas.axes.clear()

            dataset = self.analysis_graph_data.currentText()
            mode = self.analysis_graph_mode.currentText()

            trace_data_list = []
            state_data_list = []

            for localisation_index, localisation_data in enumerate(self.data_dict[dataset]):

                user_label = localisation_data["user_label"]
                nucleotide_label = localisation_data["nucleotide_label"]

                if self.get_filter_status("analysis", user_label, nucleotide_label) == False:

                    trace_data = localisation_data[mode]
                    state_data = localisation_data["states"]

                    if len(trace_data) > 0:

                        trace_data_list.append(trace_data)
                        state_data_list.append(state_data)

        except:
            print(traceback.format_exc())
            pass

        histogram_data = self.compute_histograms(trace_data_list, state_data_list)
        self.plot_analysis_histogram(histogram_data)

    def compute_histograms(self, trace_data, state_data):

        histogram_data = {}

        try:

            if len(trace_data) > 0:

                histogram_data = {"data_intensity": {}, "states_intensity": {},
                                  "states_centres": {}, "states_noise": {}, "states_dwell_times": {}}

                histogram_data["data_intensity"][0] = np.concatenate(trace_data).tolist()

                if len(state_data) == len(trace_data) and len(np.unique(state_data)) > 1:

                    trace_data = np.concatenate(trace_data)
                    state_data = np.concatenate(state_data)

                    change_indices = np.where(np.diff(state_data) != 0)[0] + 1

                    split_trace_data = np.split(trace_data, change_indices)
                    split_state_data = np.split(state_data, change_indices)

                    for data, state in zip(split_trace_data, split_state_data):

                        state = state[0]

                        if state not in histogram_data["states_intensity"].keys():
                            histogram_data["states_intensity"][state] = []
                            histogram_data["states_centres"][state] = []
                            histogram_data["states_noise"][state] = []
                            histogram_data["states_dwell_times"][state] = []

                        instensity = data.tolist()
                        centres = [np.mean(data)]*len(data)
                        noise = [np.std(data)]*len(data)
                        dwell_time = len(data)

                        histogram_data["states_intensity"][state].extend(instensity)
                        histogram_data["states_centres"][state].extend(centres)
                        histogram_data["states_noise"][state].extend(noise)
                        histogram_data["states_dwell_times"][state].append(dwell_time)

        except:
            print(traceback.format_exc())
            pass

        return histogram_data

    def plot_analysis_histogram(self, histogram_data):

        try:

            if histogram_data != {}:

                histogram_dataset = self.analysis_histogram_dataset.currentText()
                histogram_mode = self.analysis_histogram_mode.currentText().lower()
                bin_size = self.analysis_histogram_bin_size.currentText()
                plot_dataset = self.analysis_graph_data.currentText()
                plot_mode = self.analysis_graph_mode.currentText()

                if bin_size.isdigit():
                    bin_size = int(bin_size)
                else:
                    bin_size = "auto"

                if histogram_dataset == "Raw Data: Intensity":
                    histogram_key = "data_intensity"
                elif histogram_dataset == "Fitted States: Intensity":
                    histogram_key = "states_intensity"
                elif histogram_dataset == "Fitted States: Centres":
                    histogram_key = "states_centres"
                elif histogram_dataset == "Fitted States: Noise":
                    histogram_key = "states_noise"
                elif histogram_dataset == "Fitted States: Dwell Times":
                    histogram_key = "states_dwell_times"

                if histogram_mode.lower() == "frequency":
                    ylabel = "Frequency"
                    density = False
                else:
                    ylabel = "Probability"
                    density = True

                fig, ax = plt.subplots()

                all_data = []

                for state in histogram_data[histogram_key].keys():

                    if histogram_dataset != "Raw Data: Intensity":
                        plot_label = f"{plot_mode}: State {int(state)}"
                    else:
                        plot_label = plot_mode

                    histogram_values = histogram_data[histogram_key][state]

                    all_data.extend(histogram_values)

                    if len(histogram_values) > 0:

                        counts, bin_edges = np.histogram(histogram_values, bins=bin_size, density=density)
                        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                        # Plot using matplotlib's bar function
                        width = np.diff(bin_edges)
                        ax.bar(bin_centers, counts, align='center', width=width, alpha=0.5, label=plot_label)

                ax.legend()
                ax.set_xlabel(histogram_dataset)
                ax.set_ylabel(ylabel)

                lower_limit, upper_limit = np.percentile(all_data, [1, 99])
                lower_limit = lower_limit - (upper_limit - lower_limit) * 0.1
                upper_limit = upper_limit + (upper_limit - lower_limit) * 0.1

                ax.set_xlim(lower_limit, upper_limit)
                ax.autoscale(enable=True, axis='y')

                fig.canvas.draw()
                self.plot_queue.put((fig, ax))

        except:
            print(traceback.format_exc())
            pass

    def update_analysis_graph_canvas(self):

        try:
            fig, ax = self.plot_queue.get_nowait()

            # Replace the figure in the canvas
            self.analysis_graph_canvas.canvas.figure = fig

            # Update the canvas
            self.analysis_graph_canvas.canvas.draw()

            self.print_notification("Analysis graph updated")

        except queue.Empty:
            # Handle the case where the queue is empty
            pass
        except Exception as e:
            print(f"Error updating plot: {e}")
            traceback.print_exc()
