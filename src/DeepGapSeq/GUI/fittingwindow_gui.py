# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fittingwindow_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(354, 399)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.ebfret_connect_matlab = QtWidgets.QPushButton(self.tab)
        self.ebfret_connect_matlab.setObjectName("ebfret_connect_matlab")
        self.verticalLayout_2.addWidget(self.ebfret_connect_matlab)
        self.ebfret_connect_matlab_progress = QtWidgets.QProgressBar(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ebfret_connect_matlab_progress.sizePolicy().hasHeightForWidth())
        self.ebfret_connect_matlab_progress.setSizePolicy(sizePolicy)
        self.ebfret_connect_matlab_progress.setMaximumSize(QtCore.QSize(16777215, 10))
        self.ebfret_connect_matlab_progress.setProperty("value", 0)
        self.ebfret_connect_matlab_progress.setObjectName("ebfret_connect_matlab_progress")
        self.verticalLayout_2.addWidget(self.ebfret_connect_matlab_progress)
        spacerItem = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.label_2 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.ebfret_fit_dataset = QtWidgets.QComboBox(self.tab)
        self.ebfret_fit_dataset.setObjectName("ebfret_fit_dataset")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.ebfret_fit_dataset)
        self.label_5 = QtWidgets.QLabel(self.tab)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.ebfret_fit_data = QtWidgets.QComboBox(self.tab)
        self.ebfret_fit_data.setObjectName("ebfret_fit_data")
        self.ebfret_fit_data.addItem("")
        self.ebfret_fit_data.addItem("")
        self.ebfret_fit_data.addItem("")
        self.ebfret_fit_data.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.ebfret_fit_data)
        self.label_17 = QtWidgets.QLabel(self.tab)
        self.label_17.setObjectName("label_17")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.ebfret_user_filter = QtWidgets.QComboBox(self.tab)
        self.ebfret_user_filter.setObjectName("ebfret_user_filter")
        self.ebfret_user_filter.addItem("")
        self.ebfret_user_filter.addItem("")
        self.ebfret_user_filter.addItem("")
        self.ebfret_user_filter.addItem("")
        self.ebfret_user_filter.addItem("")
        self.ebfret_user_filter.addItem("")
        self.ebfret_user_filter.addItem("")
        self.ebfret_user_filter.addItem("")
        self.ebfret_user_filter.addItem("")
        self.ebfret_user_filter.addItem("")
        self.ebfret_user_filter.addItem("")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.ebfret_user_filter)
        self.label_26 = QtWidgets.QLabel(self.tab)
        self.label_26.setObjectName("label_26")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_26)
        self.ebfret_nucleotide_filter = QtWidgets.QComboBox(self.tab)
        self.ebfret_nucleotide_filter.setObjectName("ebfret_nucleotide_filter")
        self.ebfret_nucleotide_filter.addItem("")
        self.ebfret_nucleotide_filter.addItem("")
        self.ebfret_nucleotide_filter.addItem("")
        self.ebfret_nucleotide_filter.addItem("")
        self.ebfret_nucleotide_filter.addItem("")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.ebfret_nucleotide_filter)
        self.verticalLayout_2.addLayout(self.formLayout)
        self.gridLayout_17 = QtWidgets.QGridLayout()
        self.gridLayout_17.setObjectName("gridLayout_17")
        self.ebfret_max_states = QtWidgets.QComboBox(self.tab)
        self.ebfret_max_states.setObjectName("ebfret_max_states")
        self.ebfret_max_states.addItem("")
        self.ebfret_max_states.addItem("")
        self.ebfret_max_states.addItem("")
        self.ebfret_max_states.addItem("")
        self.ebfret_max_states.addItem("")
        self.gridLayout_17.addWidget(self.ebfret_max_states, 0, 3, 1, 1)
        self.ebfret_min_states = QtWidgets.QComboBox(self.tab)
        self.ebfret_min_states.setObjectName("ebfret_min_states")
        self.ebfret_min_states.addItem("")
        self.ebfret_min_states.addItem("")
        self.ebfret_min_states.addItem("")
        self.ebfret_min_states.addItem("")
        self.ebfret_min_states.addItem("")
        self.gridLayout_17.addWidget(self.ebfret_min_states, 0, 1, 1, 1)
        self.label_38 = QtWidgets.QLabel(self.tab)
        self.label_38.setObjectName("label_38")
        self.gridLayout_17.addWidget(self.label_38, 0, 0, 1, 1)
        self.label_39 = QtWidgets.QLabel(self.tab)
        self.label_39.setObjectName("label_39")
        self.gridLayout_17.addWidget(self.label_39, 0, 2, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_17)
        self.ebfret_crop_plots = QtWidgets.QCheckBox(self.tab)
        self.ebfret_crop_plots.setObjectName("ebfret_crop_plots")
        self.verticalLayout_2.addWidget(self.ebfret_crop_plots)
        self.ebfret_run_analysis = QtWidgets.QPushButton(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ebfret_run_analysis.sizePolicy().hasHeightForWidth())
        self.ebfret_run_analysis.setSizePolicy(sizePolicy)
        self.ebfret_run_analysis.setObjectName("ebfret_run_analysis")
        self.verticalLayout_2.addWidget(self.ebfret_run_analysis)
        spacerItem1 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.label_4 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.formLayout_26 = QtWidgets.QFormLayout()
        self.formLayout_26.setObjectName("formLayout_26")
        self.label_66 = QtWidgets.QLabel(self.tab)
        self.label_66.setObjectName("label_66")
        self.formLayout_26.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_66)
        self.ebfret_visualisation_state = QtWidgets.QComboBox(self.tab)
        self.ebfret_visualisation_state.setObjectName("ebfret_visualisation_state")
        self.ebfret_visualisation_state.addItem("")
        self.ebfret_visualisation_state.addItem("")
        self.ebfret_visualisation_state.addItem("")
        self.ebfret_visualisation_state.addItem("")
        self.ebfret_visualisation_state.addItem("")
        self.formLayout_26.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.ebfret_visualisation_state)
        self.verticalLayout_2.addLayout(self.formLayout_26)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_9 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_3.addWidget(self.label_9)
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_6 = QtWidgets.QLabel(self.tab_2)
        self.label_6.setObjectName("label_6")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.deeplasi_fit_dataset = QtWidgets.QComboBox(self.tab_2)
        self.deeplasi_fit_dataset.setObjectName("deeplasi_fit_dataset")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.deeplasi_fit_dataset)
        self.label_7 = QtWidgets.QLabel(self.tab_2)
        self.label_7.setObjectName("label_7")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.deeplasi_fit_data = QtWidgets.QComboBox(self.tab_2)
        self.deeplasi_fit_data.setObjectName("deeplasi_fit_data")
        self.deeplasi_fit_data.addItem("")
        self.deeplasi_fit_data.addItem("")
        self.deeplasi_fit_data.addItem("")
        self.deeplasi_fit_data.addItem("")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.deeplasi_fit_data)
        self.label_18 = QtWidgets.QLabel(self.tab_2)
        self.label_18.setObjectName("label_18")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.deeplasi_user_filter = QtWidgets.QComboBox(self.tab_2)
        self.deeplasi_user_filter.setObjectName("deeplasi_user_filter")
        self.deeplasi_user_filter.addItem("")
        self.deeplasi_user_filter.addItem("")
        self.deeplasi_user_filter.addItem("")
        self.deeplasi_user_filter.addItem("")
        self.deeplasi_user_filter.addItem("")
        self.deeplasi_user_filter.addItem("")
        self.deeplasi_user_filter.addItem("")
        self.deeplasi_user_filter.addItem("")
        self.deeplasi_user_filter.addItem("")
        self.deeplasi_user_filter.addItem("")
        self.deeplasi_user_filter.addItem("")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.deeplasi_user_filter)
        self.label_27 = QtWidgets.QLabel(self.tab_2)
        self.label_27.setObjectName("label_27")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_27)
        self.deeplasi_nucleotide_filter = QtWidgets.QComboBox(self.tab_2)
        self.deeplasi_nucleotide_filter.setObjectName("deeplasi_nucleotide_filter")
        self.deeplasi_nucleotide_filter.addItem("")
        self.deeplasi_nucleotide_filter.addItem("")
        self.deeplasi_nucleotide_filter.addItem("")
        self.deeplasi_nucleotide_filter.addItem("")
        self.deeplasi_nucleotide_filter.addItem("")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.deeplasi_nucleotide_filter)
        self.verticalLayout_3.addLayout(self.formLayout_2)
        self.deeplasi_crop_plots = QtWidgets.QCheckBox(self.tab_2)
        self.deeplasi_crop_plots.setObjectName("deeplasi_crop_plots")
        self.verticalLayout_3.addWidget(self.deeplasi_crop_plots)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem2)
        self.deeplasi_detect_states = QtWidgets.QPushButton(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.deeplasi_detect_states.sizePolicy().hasHeightForWidth())
        self.deeplasi_detect_states.setSizePolicy(sizePolicy)
        self.deeplasi_detect_states.setObjectName("deeplasi_detect_states")
        self.verticalLayout_3.addWidget(self.deeplasi_detect_states)
        self.deeplasi_progressbar = QtWidgets.QProgressBar(self.tab_2)
        self.deeplasi_progressbar.setMaximumSize(QtCore.QSize(16777215, 10))
        self.deeplasi_progressbar.setProperty("value", 0)
        self.deeplasi_progressbar.setObjectName("deeplasi_progressbar")
        self.verticalLayout_3.addWidget(self.deeplasi_progressbar)
        self.tabWidget.addTab(self.tab_2, "")
        self.verticalLayout.addWidget(self.tabWidget)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        self.ebfret_max_states.setCurrentIndex(4)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Connect MATLAB/ebFRET"))
        self.ebfret_connect_matlab.setText(_translate("Form", "Open MATLAB/ebFRET"))
        self.label_2.setText(_translate("Form", "ebFRET Analysis"))
        self.label_3.setText(_translate("Form", "Fit Dataset"))
        self.label_5.setText(_translate("Form", "Fit Data"))
        self.ebfret_fit_data.setItemText(0, _translate("Form", "Donor"))
        self.ebfret_fit_data.setItemText(1, _translate("Form", "Acceptor"))
        self.ebfret_fit_data.setItemText(2, _translate("Form", "FRET Efficiency"))
        self.ebfret_fit_data.setItemText(3, _translate("Form", "ALEX Efficiency"))
        self.label_17.setText(_translate("Form", "Localisation Filter"))
        self.ebfret_user_filter.setItemText(0, _translate("Form", "None"))
        self.ebfret_user_filter.setItemText(1, _translate("Form", "0"))
        self.ebfret_user_filter.setItemText(2, _translate("Form", "1"))
        self.ebfret_user_filter.setItemText(3, _translate("Form", "2"))
        self.ebfret_user_filter.setItemText(4, _translate("Form", "3"))
        self.ebfret_user_filter.setItemText(5, _translate("Form", "4"))
        self.ebfret_user_filter.setItemText(6, _translate("Form", "5"))
        self.ebfret_user_filter.setItemText(7, _translate("Form", "6"))
        self.ebfret_user_filter.setItemText(8, _translate("Form", "7"))
        self.ebfret_user_filter.setItemText(9, _translate("Form", "8"))
        self.ebfret_user_filter.setItemText(10, _translate("Form", "9"))
        self.label_26.setText(_translate("Form", "Nucleotide Filter"))
        self.ebfret_nucleotide_filter.setItemText(0, _translate("Form", "None"))
        self.ebfret_nucleotide_filter.setItemText(1, _translate("Form", "A"))
        self.ebfret_nucleotide_filter.setItemText(2, _translate("Form", "T"))
        self.ebfret_nucleotide_filter.setItemText(3, _translate("Form", "C"))
        self.ebfret_nucleotide_filter.setItemText(4, _translate("Form", "G"))
        self.ebfret_max_states.setItemText(0, _translate("Form", "2"))
        self.ebfret_max_states.setItemText(1, _translate("Form", "3"))
        self.ebfret_max_states.setItemText(2, _translate("Form", "4"))
        self.ebfret_max_states.setItemText(3, _translate("Form", "5"))
        self.ebfret_max_states.setItemText(4, _translate("Form", "6"))
        self.ebfret_min_states.setItemText(0, _translate("Form", "2"))
        self.ebfret_min_states.setItemText(1, _translate("Form", "3"))
        self.ebfret_min_states.setItemText(2, _translate("Form", "4"))
        self.ebfret_min_states.setItemText(3, _translate("Form", "5"))
        self.ebfret_min_states.setItemText(4, _translate("Form", "6"))
        self.label_38.setText(_translate("Form", "Min States:"))
        self.label_39.setText(_translate("Form", "Max States:"))
        self.ebfret_crop_plots.setText(_translate("Form", "Crop Plots"))
        self.ebfret_run_analysis.setText(_translate("Form", "Run ebFRET analysis"))
        self.label_4.setText(_translate("Form", "ebFRET Visualisation"))
        self.label_66.setText(_translate("Form", "ebFRET fitted States:"))
        self.ebfret_visualisation_state.setItemText(0, _translate("Form", "2"))
        self.ebfret_visualisation_state.setItemText(1, _translate("Form", "3"))
        self.ebfret_visualisation_state.setItemText(2, _translate("Form", "4"))
        self.ebfret_visualisation_state.setItemText(3, _translate("Form", "5"))
        self.ebfret_visualisation_state.setItemText(4, _translate("Form", "6"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "ebFRET"))
        self.label_9.setText(_translate("Form", "Settings"))
        self.label_6.setText(_translate("Form", "Fit Dataset"))
        self.label_7.setText(_translate("Form", "Fit Data"))
        self.deeplasi_fit_data.setItemText(0, _translate("Form", "Donor"))
        self.deeplasi_fit_data.setItemText(1, _translate("Form", "Acceptor"))
        self.deeplasi_fit_data.setItemText(2, _translate("Form", "FRET Efficiency"))
        self.deeplasi_fit_data.setItemText(3, _translate("Form", "ALEX Efficiency"))
        self.label_18.setText(_translate("Form", "Localisation Filter"))
        self.deeplasi_user_filter.setItemText(0, _translate("Form", "None"))
        self.deeplasi_user_filter.setItemText(1, _translate("Form", "0"))
        self.deeplasi_user_filter.setItemText(2, _translate("Form", "1"))
        self.deeplasi_user_filter.setItemText(3, _translate("Form", "2"))
        self.deeplasi_user_filter.setItemText(4, _translate("Form", "3"))
        self.deeplasi_user_filter.setItemText(5, _translate("Form", "4"))
        self.deeplasi_user_filter.setItemText(6, _translate("Form", "5"))
        self.deeplasi_user_filter.setItemText(7, _translate("Form", "6"))
        self.deeplasi_user_filter.setItemText(8, _translate("Form", "7"))
        self.deeplasi_user_filter.setItemText(9, _translate("Form", "8"))
        self.deeplasi_user_filter.setItemText(10, _translate("Form", "9"))
        self.label_27.setText(_translate("Form", "Nucleotide Filter"))
        self.deeplasi_nucleotide_filter.setItemText(0, _translate("Form", "None"))
        self.deeplasi_nucleotide_filter.setItemText(1, _translate("Form", "A"))
        self.deeplasi_nucleotide_filter.setItemText(2, _translate("Form", "T"))
        self.deeplasi_nucleotide_filter.setItemText(3, _translate("Form", "C"))
        self.deeplasi_nucleotide_filter.setItemText(4, _translate("Form", "G"))
        self.deeplasi_crop_plots.setText(_translate("Form", "Crop Plots"))
        self.deeplasi_detect_states.setText(_translate("Form", "Detect States"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "DeepLASI"))
