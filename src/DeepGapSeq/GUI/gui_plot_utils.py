from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg
import traceback
from functools import partial
import uuid

class _plotting_methods:

    def reset_crop_ranges(self, mode="active"):
        slider_value = self.plot_localisation_number.value()
        localisation_number = self.localisation_numbers[slider_value]

        if mode == "active":
            for dataset_name in self.data_dict.keys():
                self.data_dict[dataset_name][localisation_number]["crop_range"] = []
        else:
            for dataset_name, dataset_data in self.data_dict.items():
                for localisation_number in range(len(dataset_data)):
                    self.data_dict[dataset_name][localisation_number]["crop_range"] = []

        self.plot_traces(update_plot=False)

    def initialise_crop_region(self, colour = (255, 0, 0, 50)):

        crop_region_name = str(uuid.uuid4())
        setattr(self, crop_region_name, pg.LinearRegionItem(brush=colour))
        crop_region = getattr(self, crop_region_name)
        crop_region.sigRegionChanged.connect(partial(self.update_crop_range, mode="drag"))

        self.unique_crop_regions.append(crop_region)

        return crop_region

    def initialise_gamma_ranges(self, colour = (255, 0, 255, 50)):

        gamma_range1_name = str(uuid.uuid4())
        gamma_range2_name = str(uuid.uuid4())

        setattr(self, gamma_range1_name, pg.LinearRegionItem(brush=pg.mkBrush(*colour)))
        setattr(self, gamma_range2_name, pg.LinearRegionItem(brush=pg.mkBrush(*colour)))

        gamma_range1 = getattr(self, gamma_range1_name)
        gamma_range2 = getattr(self, gamma_range2_name)

        gamma_range1.sigRegionChanged.connect(partial(self.update_gamma_ranges, mode="drag"))
        gamma_range2.sigRegionChanged.connect(partial(self.update_gamma_ranges, mode="drag"))

        self.unique_gamma_ranges.append(gamma_range1)
        self.unique_gamma_ranges.append(gamma_range2)

        return [gamma_range1, gamma_range2]

    def update_gamma_ranges(self, event, mode = "click"):

        slider_value = self.plot_localisation_number.value()
        localisation_number = self.localisation_numbers[slider_value]

        if mode == "click":

            for dataset_name in self.data_dict.keys():
                gamma_ranges = self.data_dict[dataset_name][localisation_number]["gamma_ranges"]

                if len(gamma_ranges) == 0:
                    gamma_ranges.append(event)
                else:
                    gamma_distances = [abs(event - range) for range in gamma_ranges]

                    if len(gamma_ranges) == 2:
                        if np.min(gamma_distances) < 5:
                            gamma_ranges.pop(gamma_distances.index(np.min(gamma_distances)))
                        else:
                            gamma_ranges.pop(gamma_distances.index(np.min(gamma_distances)))
                            gamma_ranges.append(event)
                    else:
                        gamma_ranges.append(event)

                self.data_dict[dataset_name][localisation_number]["gamma_ranges"] = gamma_ranges

                self.plot_traces(update_plot=False)

        else:

            correction_x = np.mean(list(event.getRegion()))

            for dataset_name in self.data_dict.keys():
                gamma_ranges = self.data_dict[dataset_name][localisation_number]["gamma_ranges"]

                if len(gamma_ranges) != 0:
                    gamma_ranges = self.data_dict[dataset_name][localisation_number]["gamma_ranges"]

                    nearest_index = (np.abs(gamma_ranges - correction_x)).argmin()
                    gamma_ranges[nearest_index] = correction_x

                    self.data_dict[dataset_name][localisation_number]["gamma_ranges"] = gamma_ranges

            self.plot_traces(update_plot=False)

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

    def get_plot_item(self, plot, name):

        for index, item in enumerate(plot.items):
            if item.name() == name:
                return item

        return None

    def remove_plot_item(self, plot, item):

        try:

            if type(item) == str:
                for item in plot.items:
                    if item.name() == "hmm_mean":
                        plot.removeItem(item)
            else:
                for item in plot.items:
                    if isinstance(item, item):
                        plot.removeItem(item)
        except:
            pass

        return plot

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

    def initialise_plot(self):
        try:
            plot_data = self.plot_data.currentText()
            plot_mode = self.plot_mode.currentText()
            show_detected_states = self.plot_settings.show_detected_states.isChecked()

            if plot_data == "All Datasets":
                self.plot_datasets = list(self.data_dict.keys())
            elif plot_data != "":
                self.plot_datasets = [plot_data]
            else:
                self.plot_datasets = []

            alex_channels = []

            if self.plot_settings.alex_show_DD.isChecked():
                alex_channels.append("DD")
            if self.plot_settings.alex_show_DA.isChecked():
                alex_channels.append("DA")
            if self.plot_settings.alex_show_AD.isChecked():
                alex_channels.append("AD")
            if self.plot_settings.alex_show_AA.isChecked():
                alex_channels.append("AA")

            self.n_plot_lines = 0

            if len(self.plot_datasets) > 0:
                plot_labels = list(self.data_dict[self.plot_datasets[0]][0].keys())

                plot = False

                if plot_mode == "Donor" and set(["donor"]).issubset(plot_labels):
                    plot = True
                    self.plot_line_labels = ["donor"]
                    self.n_plot_lines = len(self.plot_line_labels)
                elif plot_mode == "Acceptor" and set(["acceptor"]).issubset(plot_labels):
                    plot = True
                    self.plot_line_labels = ["acceptor"]
                    self.n_plot_lines = len(self.plot_line_labels)
                elif plot_mode == "FRET Data" and set(["donor", "acceptor"]).issubset(plot_labels):
                    plot = True
                    self.plot_line_labels = ["donor", "acceptor"]
                    self.n_plot_lines = len(self.plot_line_labels)
                elif plot_mode == "FRET Efficiency" and set(["efficiency"]).issubset(plot_labels):
                    plot = True
                    self.plot_line_labels = ["efficiency"]
                    self.n_plot_lines = len(self.plot_line_labels)
                elif plot_mode == "FRET Data + FRET Efficiency" and set(["donor", "acceptor", "efficiency"]).issubset(plot_labels):
                    plot = True
                    self.plot_line_labels = ["donor", "acceptor", "efficiency"]
                    self.n_plot_lines = len(self.plot_line_labels)
                elif plot_mode == "ALEX Data" and set(["DD", "AA", "DA", "AD"]).issubset(plot_labels):
                    plot = True
                    self.plot_line_labels = alex_channels
                    self.n_plot_lines = len(self.plot_line_labels)
                elif plot_mode == "ALEX Efficiency" and set(["efficiency"]).issubset(plot_labels):
                    plot = True
                    self.plot_line_labels = ["efficiency"]
                    self.n_plot_lines = len(self.plot_line_labels)
                elif plot_mode == "ALEX Data + ALEX Efficiency" and set(["DD", "AA", "DA", "AD", "efficiency"]).issubset(plot_labels):
                    plot = True
                    self.plot_line_labels = alex_channels + ["efficiency"]
                    self.n_plot_lines = len(self.plot_line_labels)
                else:
                    plot = False

                if plot == True and len(self.plot_line_labels) > 0:

                    self.export_settings.export_data_selection.clear()
                    self.export_settings.export_data_selection.addItems(self.plot_line_labels)

                    self.localisation_numbers, self.n_traces = self.filter_data_dict()

                    if self.n_traces > 0:
                        slider_value = self.plot_localisation_number.value()
                        if slider_value >= self.n_traces:
                            slider_value = self.n_traces - 1
                            self.plot_localisation_number.setValue(slider_value)

                        self.plot_localisation_number.setMinimum(0)
                        self.plot_localisation_number.setMaximum(self.n_traces - 1)

                        self.plot_traces(update_plot=True)

                    else:
                        self.print_notification("No traces to plot")

        except:
            print(traceback.format_exc())
            pass

    def update_plot_layout(self):

        try:

            self.plot_grid = {}
            self.unique_crop_regions = []
            self.unique_gamma_ranges = []

            self.graph_canvas.clear()

            plot_mode = self.plot_mode.currentText()
            split = self.plot_settings.plot_split_lines.isChecked()

            efficiency_plot = False

            for plot_index, plot_dataset in enumerate(self.plot_datasets):

                sub_plots = []
                sub_plot_gamma_ranges = []
                sub_plot_crop_regions = []

                if plot_mode == "FRET Data + FRET Efficiency" and split==False:

                    layout = pg.GraphicsLayout()
                    self.graph_canvas.addItem(layout, row=plot_index, col=0)

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

                        crop_region = self.initialise_crop_region()
                        sub_plot_crop_regions.append(crop_region)

                        gamma_ranges = self.initialise_gamma_ranges()
                        sub_plot_gamma_ranges.append(gamma_ranges)

                    for j in range(1, len(sub_plots)):
                        sub_plots[j].setXLink(sub_plots[0])

                    sub_plots = [sub_plots[0] for i in range(self.n_plot_lines - 1)] + [sub_plots[1]]
                    sub_plot_crop_regions = [sub_plot_crop_regions[0] for i in range(self.n_plot_lines - 1)] + [sub_plot_crop_regions[1]]
                    sub_plot_gamma_ranges = [sub_plot_gamma_ranges[0] for i in range(self.n_plot_lines - 1)] + [sub_plot_gamma_ranges[1]]

                    efficiency_plot = True

                elif split == True and self.n_plot_lines > 1:

                    layout = pg.GraphicsLayout()
                    self.graph_canvas.addItem(layout, row=plot_index, col=0)

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

                        crop_region = self.initialise_crop_region()
                        sub_plot_crop_regions.append(crop_region)

                        gamma_ranges = self.initialise_gamma_ranges()
                        sub_plot_gamma_ranges.append(gamma_ranges)

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

                    for line_index in range(self.n_plot_lines):
                        sub_plots.append(p)

                        crop_region = self.initialise_crop_region()
                        sub_plot_crop_regions.append(crop_region)

                        gamma_ranges = self.initialise_gamma_ranges()
                        sub_plot_gamma_ranges.append(gamma_ranges)

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
                        "sub_plot_crop_regions": sub_plot_crop_regions,
                        "sub_plot_gamma_ranges": sub_plot_gamma_ranges,
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


            # print(f"len unique crop regions: {len(self.unique_crop_regions)}")

        except:
            print(traceback.format_exc())
            pass

        return self.plot_grid

    def plot_traces(self, update_plot = False):

        try:

            if hasattr(self, "plot_grid") == False or update_plot == True:
                self.plot_grid = self.update_plot_layout()

            if self.plot_grid != {}:

                plot_mode = self.plot_mode.currentText()

                slider_value = self.plot_localisation_number.value()
                crop_plots = self.plot_settings.crop_plots.isChecked()
                show_crop_range = self.plot_settings.show_crop_range.isChecked()
                show_gamma = self.plot_settings.show_correction_factor_ranges.isChecked()
                localisation_number = self.localisation_numbers[slider_value]
                downsample = int(self.plot_settings.plot_downsample.currentText())

                for plot_index, grid in enumerate(self.plot_grid.values()):

                    plot_dataset = grid["plot_dataset"]
                    sub_axes = grid["sub_axes"]
                    crop_regions = grid["sub_plot_crop_regions"]
                    plot_gamma_ranges = grid["sub_plot_gamma_ranges"]
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
                        self.remove_plot_item(plot, "hmm_mean")

                    for line_index, (plot, crop_region, gamma_ranges, line,  plot_label) in enumerate(zip(sub_axes, crop_regions, plot_gamma_ranges, plot_lines, plot_lines_labels)):

                        legend = plot.legend
                        data = self.data_dict[plot_dataset][localisation_number][plot_label]

                        break_points = self.data_dict[plot_dataset][localisation_number]["break_points"]
                        state_means = self.data_dict[plot_dataset][localisation_number]["state_means"][plot_label]

                        gamma_correction_ranges = self.data_dict[plot_dataset][localisation_number]["gamma_ranges"]

                        if crop_plots == True and len(crop_range) == 2:
                            crop_range = sorted(crop_range)
                            crop_range = [int(crop_range[0]), int(crop_range[1])]

                            data = data[crop_range[0]:crop_range[1]]
                            state_means = state_means[crop_range[0]:crop_range[1]]

                        if self.plot_settings.plot_normalise.isChecked() and "Efficiency" not in plot_label:
                            data = (data - np.min(data)) / (np.max(data) - np.min(data))

                        plot_line = plot_lines[line_index]
                        plot_line.setData(data)
                        plot_line.setDownsampling(ds = downsample)
                        plot.enableAutoRange()
                        plot.autoRange()

                        if self.plot_settings.show_detected_states.isChecked() and len(state_means) > 0:

                            if self.plot_settings.plot_normalise.isChecked():
                                state_means = (state_means - np.min(state_means)) / (np.max(state_means) - np.min(state_means))

                            if "Efficiency" in plot_mode:
                                if line_index == len(sub_axes) - 1:
                                    plot.plot(state_means, pen=pg.mkPen('b', width=3), name="hmm_mean", downsample=downsample)
                                    legend.removeItem("hmm_mean")
                            else:
                                if line_index == 0:
                                    plot.plot(state_means, pen=pg.mkPen('b', width=3), name="hmm_mean", downsample=downsample)
                                    legend.removeItem("hmm_mean")

                        if crop_plots == False and show_crop_range == True and len(crop_range) == 2:

                            if self.get_plot_item_instance(plot, pg.LinearRegionItem) is None:
                                plot.addItem(crop_region)
                                crop_region.setRegion(crop_range)
                                self.unique_crop_regions.append(crop_region)

                        else:
                            for crop_region in crop_regions:
                                plot.removeItem(crop_region)

                        if show_gamma == True and len(gamma_correction_ranges) > 0:

                            for gamma_range in gamma_ranges:
                                if gamma_range in plot.items:
                                    plot.removeItem(gamma_range)

                            for gamma_index, correction_x in enumerate(gamma_correction_ranges):

                                gamma_range = gamma_ranges[gamma_index]

                                if gamma_range not in plot.items:
                                    plot.addItem(gamma_range)
                                    gamma_range.setRegion([correction_x-10, correction_x+10])
                                else:
                                    gamma_range.setRegion([correction_x-10, correction_x+10])
                        else:
                            for gamma_range in gamma_ranges:
                                if gamma_range in plot.items:
                                    plot.removeItem(gamma_range)



                        if len(break_points) > 2 and self.show_cpd_breakpoints.isChecked():

                            break_points = np.unique(break_points).tolist()

                            for break_point in break_points[1:-1]:
                                bp = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('g', width=3))
                                plot.addItem(bp)
                                bp.setPos(break_point)

        except:
            print(traceback.format_exc())
            pass






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

            elif event.modifiers() & Qt.AltModifier:

                xpos = self.get_event_x_postion(event, mode="click")
                self.parent.update_gamma_ranges(xpos)

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

