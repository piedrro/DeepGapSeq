# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(868, 604)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.mainwindow_tab = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mainwindow_tab.sizePolicy().hasHeightForWidth())
        self.mainwindow_tab.setSizePolicy(sizePolicy)
        self.mainwindow_tab.setObjectName("mainwindow_tab")
        self.traces_tab = QtWidgets.QWidget()
        self.traces_tab.setObjectName("traces_tab")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.traces_tab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.graph_container = QtWidgets.QWidget(self.traces_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graph_container.sizePolicy().hasHeightForWidth())
        self.graph_container.setSizePolicy(sizePolicy)
        self.graph_container.setObjectName("graph_container")
        self.verticalLayout_2.addWidget(self.graph_container)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_3 = QtWidgets.QLabel(self.traces_tab)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 0, 0, 1, 1)
        self.plot_localisation_number = QtWidgets.QSlider(self.traces_tab)
        self.plot_localisation_number.setOrientation(QtCore.Qt.Horizontal)
        self.plot_localisation_number.setObjectName("plot_localisation_number")
        self.gridLayout_4.addWidget(self.plot_localisation_number, 0, 1, 1, 1)
        self.plot_localisation_number_label = QtWidgets.QLabel(self.traces_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_localisation_number_label.sizePolicy().hasHeightForWidth())
        self.plot_localisation_number_label.setSizePolicy(sizePolicy)
        self.plot_localisation_number_label.setMinimumSize(QtCore.QSize(20, 0))
        self.plot_localisation_number_label.setObjectName("plot_localisation_number_label")
        self.gridLayout_4.addWidget(self.plot_localisation_number_label, 0, 2, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_4)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_61 = QtWidgets.QLabel(self.traces_tab)
        self.label_61.setObjectName("label_61")
        self.gridLayout.addWidget(self.label_61, 0, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.traces_tab)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.plot_data = QtWidgets.QComboBox(self.traces_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_data.sizePolicy().hasHeightForWidth())
        self.plot_data.setSizePolicy(sizePolicy)
        self.plot_data.setObjectName("plot_data")
        self.plot_data.addItem("")
        self.gridLayout.addWidget(self.plot_data, 0, 1, 1, 1)
        self.plot_mode = QtWidgets.QComboBox(self.traces_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_mode.sizePolicy().hasHeightForWidth())
        self.plot_mode.setSizePolicy(sizePolicy)
        self.plot_mode.setObjectName("plot_mode")
        self.plot_mode.addItem("")
        self.plot_mode.addItem("")
        self.plot_mode.addItem("")
        self.plot_mode.addItem("")
        self.plot_mode.addItem("")
        self.gridLayout.addWidget(self.plot_mode, 0, 3, 1, 1)
        self.plotsettings_button = QtWidgets.QPushButton(self.traces_tab)
        self.plotsettings_button.setMinimumSize(QtCore.QSize(50, 0))
        self.plotsettings_button.setObjectName("plotsettings_button")
        self.gridLayout.addWidget(self.plotsettings_button, 0, 4, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.plot_checkbox_qgrid = QtWidgets.QGridLayout()
        self.plot_checkbox_qgrid.setObjectName("plot_checkbox_qgrid")
        self.verticalLayout_2.addLayout(self.plot_checkbox_qgrid)
        self.mainwindow_tab.addTab(self.traces_tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_17 = QtWidgets.QLabel(self.tab_2)
        self.label_17.setObjectName("label_17")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.analysis_user_filter = QtWidgets.QComboBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.analysis_user_filter.sizePolicy().hasHeightForWidth())
        self.analysis_user_filter.setSizePolicy(sizePolicy)
        self.analysis_user_filter.setMinimumSize(QtCore.QSize(10, 0))
        self.analysis_user_filter.setObjectName("analysis_user_filter")
        self.analysis_user_filter.addItem("")
        self.analysis_user_filter.addItem("")
        self.analysis_user_filter.addItem("")
        self.analysis_user_filter.addItem("")
        self.analysis_user_filter.addItem("")
        self.analysis_user_filter.addItem("")
        self.analysis_user_filter.addItem("")
        self.analysis_user_filter.addItem("")
        self.analysis_user_filter.addItem("")
        self.analysis_user_filter.addItem("")
        self.analysis_user_filter.addItem("")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.analysis_user_filter)
        self.label_26 = QtWidgets.QLabel(self.tab_2)
        self.label_26.setObjectName("label_26")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_26)
        self.analysis_nucleotide_filter = QtWidgets.QComboBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.analysis_nucleotide_filter.sizePolicy().hasHeightForWidth())
        self.analysis_nucleotide_filter.setSizePolicy(sizePolicy)
        self.analysis_nucleotide_filter.setMinimumSize(QtCore.QSize(10, 0))
        self.analysis_nucleotide_filter.setObjectName("analysis_nucleotide_filter")
        self.analysis_nucleotide_filter.addItem("")
        self.analysis_nucleotide_filter.addItem("")
        self.analysis_nucleotide_filter.addItem("")
        self.analysis_nucleotide_filter.addItem("")
        self.analysis_nucleotide_filter.addItem("")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.analysis_nucleotide_filter)
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.analysis_graph_data = QtWidgets.QComboBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.analysis_graph_data.sizePolicy().hasHeightForWidth())
        self.analysis_graph_data.setSizePolicy(sizePolicy)
        self.analysis_graph_data.setMinimumSize(QtCore.QSize(10, 0))
        self.analysis_graph_data.setObjectName("analysis_graph_data")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.analysis_graph_data)
        self.label_63 = QtWidgets.QLabel(self.tab_2)
        self.label_63.setObjectName("label_63")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_63)
        self.analysis_graph_mode = QtWidgets.QComboBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.analysis_graph_mode.sizePolicy().hasHeightForWidth())
        self.analysis_graph_mode.setSizePolicy(sizePolicy)
        self.analysis_graph_mode.setMinimumSize(QtCore.QSize(10, 0))
        self.analysis_graph_mode.setObjectName("analysis_graph_mode")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.analysis_graph_mode)
        self.verticalLayout_3.addLayout(self.formLayout_2)
        self.analysis_crop_traces = QtWidgets.QCheckBox(self.tab_2)
        self.analysis_crop_traces.setObjectName("analysis_crop_traces")
        self.verticalLayout_3.addWidget(self.analysis_crop_traces)
        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_3.addItem(spacerItem)
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_6 = QtWidgets.QLabel(self.tab_2)
        self.label_6.setObjectName("label_6")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.analysis_histogram_dataset = QtWidgets.QComboBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.analysis_histogram_dataset.sizePolicy().hasHeightForWidth())
        self.analysis_histogram_dataset.setSizePolicy(sizePolicy)
        self.analysis_histogram_dataset.setMinimumSize(QtCore.QSize(10, 0))
        self.analysis_histogram_dataset.setObjectName("analysis_histogram_dataset")
        self.analysis_histogram_dataset.addItem("")
        self.analysis_histogram_dataset.addItem("")
        self.analysis_histogram_dataset.addItem("")
        self.analysis_histogram_dataset.addItem("")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.analysis_histogram_dataset)
        self.label_7 = QtWidgets.QLabel(self.tab_2)
        self.label_7.setObjectName("label_7")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.analysis_histogram_bin_size = QtWidgets.QComboBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.analysis_histogram_bin_size.sizePolicy().hasHeightForWidth())
        self.analysis_histogram_bin_size.setSizePolicy(sizePolicy)
        self.analysis_histogram_bin_size.setMinimumSize(QtCore.QSize(10, 0))
        self.analysis_histogram_bin_size.setEditable(True)
        self.analysis_histogram_bin_size.setObjectName("analysis_histogram_bin_size")
        self.analysis_histogram_bin_size.addItem("")
        self.analysis_histogram_bin_size.addItem("")
        self.analysis_histogram_bin_size.addItem("")
        self.analysis_histogram_bin_size.addItem("")
        self.analysis_histogram_bin_size.addItem("")
        self.analysis_histogram_bin_size.addItem("")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.analysis_histogram_bin_size)
        self.label_9 = QtWidgets.QLabel(self.tab_2)
        self.label_9.setObjectName("label_9")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.analysis_histogram_mode = QtWidgets.QComboBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.analysis_histogram_mode.sizePolicy().hasHeightForWidth())
        self.analysis_histogram_mode.setSizePolicy(sizePolicy)
        self.analysis_histogram_mode.setMinimumSize(QtCore.QSize(10, 0))
        self.analysis_histogram_mode.setObjectName("analysis_histogram_mode")
        self.analysis_histogram_mode.addItem("")
        self.analysis_histogram_mode.addItem("")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.analysis_histogram_mode)
        self.verticalLayout_3.addLayout(self.formLayout_3)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem1)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.analysis_graph_container = QtWidgets.QWidget(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.analysis_graph_container.sizePolicy().hasHeightForWidth())
        self.analysis_graph_container.setSizePolicy(sizePolicy)
        self.analysis_graph_container.setObjectName("analysis_graph_container")
        self.horizontalLayout.addWidget(self.analysis_graph_container)
        self.mainwindow_tab.addTab(self.tab_2, "")
        self.verticalLayout.addWidget(self.mainwindow_tab)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 868, 21))
        self.menubar.setObjectName("menubar")
        self.menu_File = QtWidgets.QMenu(self.menubar)
        self.menu_File.setObjectName("menu_File")
        self.menuAnalysis_2 = QtWidgets.QMenu(self.menubar)
        self.menuAnalysis_2.setObjectName("menuAnalysis_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionGapSeq = QtWidgets.QAction(MainWindow)
        self.actionGapSeq.setObjectName("actionGapSeq")
        self.actionebFRET = QtWidgets.QAction(MainWindow)
        self.actionebFRET.setObjectName("actionebFRET")
        self.actionXXX = QtWidgets.QAction(MainWindow)
        self.actionXXX.setObjectName("actionXXX")
        self.actionPlot_Settings = QtWidgets.QAction(MainWindow)
        self.actionPlot_Settings.setObjectName("actionPlot_Settings")
        self.actionFit_Settings = QtWidgets.QAction(MainWindow)
        self.actionFit_Settings.setObjectName("actionFit_Settings")
        self.action_Plot_Settings = QtWidgets.QAction(MainWindow)
        self.action_Plot_Settings.setCheckable(True)
        self.action_Plot_Settings.setObjectName("action_Plot_Settings")
        self.actionBreak_Point_Analysis = QtWidgets.QAction(MainWindow)
        self.actionBreak_Point_Analysis.setObjectName("actionBreak_Point_Analysis")
        self.actionDetect_Hidden_States = QtWidgets.QAction(MainWindow)
        self.actionDetect_Hidden_States.setObjectName("actionDetect_Hidden_States")
        self.actionDetect_Break_Points = QtWidgets.QAction(MainWindow)
        self.actionDetect_Break_Points.setObjectName("actionDetect_Break_Points")
        self.actionHMM = QtWidgets.QAction(MainWindow)
        self.actionHMM.setObjectName("actionHMM")
        self.actionebFRET_2 = QtWidgets.QAction(MainWindow)
        self.actionebFRET_2.setObjectName("actionebFRET_2")
        self.actionDeep_Learning = QtWidgets.QAction(MainWindow)
        self.actionDeep_Learning.setObjectName("actionDeep_Learning")
        self.actionImport_I = QtWidgets.QAction(MainWindow)
        self.actionImport_I.setObjectName("actionImport_I")
        self.actionExport_E = QtWidgets.QAction(MainWindow)
        self.actionExport_E.setObjectName("actionExport_E")
        self.actionFit_Hidden_States_F = QtWidgets.QAction(MainWindow)
        self.actionFit_Hidden_States_F.setObjectName("actionFit_Hidden_States_F")
        self.actionDetect_Binding_Events_D = QtWidgets.QAction(MainWindow)
        self.actionDetect_Binding_Events_D.setObjectName("actionDetect_Binding_Events_D")
        self.menu_File.addAction(self.actionImport_I)
        self.menu_File.addAction(self.actionExport_E)
        self.menuAnalysis_2.addAction(self.actionDetect_Binding_Events_D)
        self.menuAnalysis_2.addAction(self.actionFit_Hidden_States_F)
        self.menubar.addAction(self.menu_File.menuAction())
        self.menubar.addAction(self.menuAnalysis_2.menuAction())

        self.retranslateUi(MainWindow)
        self.mainwindow_tab.setCurrentIndex(0)
        self.analysis_histogram_bin_size.setCurrentIndex(5)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "Localisation Number"))
        self.plot_localisation_number_label.setText(_translate("MainWindow", "0"))
        self.label_61.setText(_translate("MainWindow", "Plot Mode"))
        self.label.setText(_translate("MainWindow", "Plot Data"))
        self.plot_data.setItemText(0, _translate("MainWindow", "All Datasets"))
        self.plot_mode.setItemText(0, _translate("MainWindow", "Donor"))
        self.plot_mode.setItemText(1, _translate("MainWindow", "Acceptor"))
        self.plot_mode.setItemText(2, _translate("MainWindow", "FRET Data"))
        self.plot_mode.setItemText(3, _translate("MainWindow", "FRET Efficiency"))
        self.plot_mode.setItemText(4, _translate("MainWindow", "FRET Data + FRET Efficiency"))
        self.plotsettings_button.setText(_translate("MainWindow", "Plot Settings [SPACE]"))
        self.mainwindow_tab.setTabText(self.mainwindow_tab.indexOf(self.traces_tab), _translate("MainWindow", "Trace Graphs"))
        self.label_2.setText(_translate("MainWindow", "Data Selection"))
        self.label_17.setText(_translate("MainWindow", "Localisation Filter"))
        self.analysis_user_filter.setItemText(0, _translate("MainWindow", "None"))
        self.analysis_user_filter.setItemText(1, _translate("MainWindow", "0"))
        self.analysis_user_filter.setItemText(2, _translate("MainWindow", "1"))
        self.analysis_user_filter.setItemText(3, _translate("MainWindow", "2"))
        self.analysis_user_filter.setItemText(4, _translate("MainWindow", "3"))
        self.analysis_user_filter.setItemText(5, _translate("MainWindow", "4"))
        self.analysis_user_filter.setItemText(6, _translate("MainWindow", "5"))
        self.analysis_user_filter.setItemText(7, _translate("MainWindow", "6"))
        self.analysis_user_filter.setItemText(8, _translate("MainWindow", "7"))
        self.analysis_user_filter.setItemText(9, _translate("MainWindow", "8"))
        self.analysis_user_filter.setItemText(10, _translate("MainWindow", "9"))
        self.label_26.setText(_translate("MainWindow", "Nucleotide Filter"))
        self.analysis_nucleotide_filter.setItemText(0, _translate("MainWindow", "None"))
        self.analysis_nucleotide_filter.setItemText(1, _translate("MainWindow", "A"))
        self.analysis_nucleotide_filter.setItemText(2, _translate("MainWindow", "T"))
        self.analysis_nucleotide_filter.setItemText(3, _translate("MainWindow", "C"))
        self.analysis_nucleotide_filter.setItemText(4, _translate("MainWindow", "G"))
        self.label_5.setText(_translate("MainWindow", "Plot Data"))
        self.label_63.setText(_translate("MainWindow", "Plot Mode"))
        self.analysis_crop_traces.setText(_translate("MainWindow", "Crop Traces"))
        self.label_4.setText(_translate("MainWindow", "Histogram Settings"))
        self.label_6.setText(_translate("MainWindow", "Histogram"))
        self.analysis_histogram_dataset.setItemText(0, _translate("MainWindow", "Intensity"))
        self.analysis_histogram_dataset.setItemText(1, _translate("MainWindow", "Centres"))
        self.analysis_histogram_dataset.setItemText(2, _translate("MainWindow", "Noise"))
        self.analysis_histogram_dataset.setItemText(3, _translate("MainWindow", "Dwell Times"))
        self.label_7.setText(_translate("MainWindow", "Bin Size"))
        self.analysis_histogram_bin_size.setItemText(0, _translate("MainWindow", "10"))
        self.analysis_histogram_bin_size.setItemText(1, _translate("MainWindow", "20"))
        self.analysis_histogram_bin_size.setItemText(2, _translate("MainWindow", "30"))
        self.analysis_histogram_bin_size.setItemText(3, _translate("MainWindow", "40"))
        self.analysis_histogram_bin_size.setItemText(4, _translate("MainWindow", "50"))
        self.analysis_histogram_bin_size.setItemText(5, _translate("MainWindow", "100"))
        self.label_9.setText(_translate("MainWindow", "Histogram Mode"))
        self.analysis_histogram_mode.setItemText(0, _translate("MainWindow", "Frequency"))
        self.analysis_histogram_mode.setItemText(1, _translate("MainWindow", "Probability"))
        self.mainwindow_tab.setTabText(self.mainwindow_tab.indexOf(self.tab_2), _translate("MainWindow", "State Histograms"))
        self.menu_File.setTitle(_translate("MainWindow", "&File"))
        self.menuAnalysis_2.setTitle(_translate("MainWindow", "Analyse"))
        self.actionGapSeq.setText(_translate("MainWindow", "DeepGapSeq"))
        self.actionebFRET.setText(_translate("MainWindow", "ebFRET"))
        self.actionXXX.setText(_translate("MainWindow", "XXX"))
        self.actionPlot_Settings.setText(_translate("MainWindow", "Plot Settings"))
        self.actionFit_Settings.setText(_translate("MainWindow", "Fit Settings"))
        self.action_Plot_Settings.setText(_translate("MainWindow", "Plot Settings"))
        self.actionBreak_Point_Analysis.setText(_translate("MainWindow", "Detect Break Points"))
        self.actionDetect_Hidden_States.setText(_translate("MainWindow", "Detect Hidden States"))
        self.actionDetect_Break_Points.setText(_translate("MainWindow", "Fit Hidden States [F]"))
        self.actionHMM.setText(_translate("MainWindow", "HMM"))
        self.actionebFRET_2.setText(_translate("MainWindow", "ebFRET (MatLAB)"))
        self.actionDeep_Learning.setText(_translate("MainWindow", "Deep Learning"))
        self.actionImport_I.setText(_translate("MainWindow", "Import [I]"))
        self.actionExport_E.setText(_translate("MainWindow", "Export [E]"))
        self.actionFit_Hidden_States_F.setText(_translate("MainWindow", "Fit Hidden States [F]"))
        self.actionDetect_Binding_Events_D.setText(_translate("MainWindow", "Detect Binding Events [D]"))
