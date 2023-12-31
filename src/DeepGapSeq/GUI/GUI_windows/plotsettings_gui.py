# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'plotsettings_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(339, 403)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.gridLayout_16 = QtWidgets.QGridLayout()
        self.gridLayout_16.setObjectName("gridLayout_16")
        self.plot_normalise = QtWidgets.QCheckBox(Form)
        self.plot_normalise.setObjectName("plot_normalise")
        self.gridLayout_16.addWidget(self.plot_normalise, 1, 1, 1, 1)
        self.plot_split_lines = QtWidgets.QCheckBox(Form)
        self.plot_split_lines.setObjectName("plot_split_lines")
        self.gridLayout_16.addWidget(self.plot_split_lines, 1, 0, 1, 1)
        self.plot_showx = QtWidgets.QCheckBox(Form)
        self.plot_showx.setChecked(True)
        self.plot_showx.setObjectName("plot_showx")
        self.gridLayout_16.addWidget(self.plot_showx, 0, 0, 1, 1)
        self.plot_showy = QtWidgets.QCheckBox(Form)
        self.plot_showy.setChecked(True)
        self.plot_showy.setObjectName("plot_showy")
        self.gridLayout_16.addWidget(self.plot_showy, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_16)
        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem)
        self.label_6 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setObjectName("label_7")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.plot_downsample = QtWidgets.QComboBox(Form)
        self.plot_downsample.setObjectName("plot_downsample")
        self.plot_downsample.addItem("")
        self.plot_downsample.addItem("")
        self.plot_downsample.addItem("")
        self.plot_downsample.addItem("")
        self.plot_downsample.addItem("")
        self.plot_downsample.addItem("")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.plot_downsample)
        self.verticalLayout.addLayout(self.formLayout_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem1)
        self.label_4 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.crop_reset_active = QtWidgets.QPushButton(Form)
        self.crop_reset_active.setObjectName("crop_reset_active")
        self.gridLayout.addWidget(self.crop_reset_active, 1, 0, 1, 1)
        self.crop_reset_all = QtWidgets.QPushButton(Form)
        self.crop_reset_all.setObjectName("crop_reset_all")
        self.gridLayout.addWidget(self.crop_reset_all, 1, 1, 1, 1)
        self.show_crop_range = QtWidgets.QCheckBox(Form)
        self.show_crop_range.setChecked(True)
        self.show_crop_range.setObjectName("show_crop_range")
        self.gridLayout.addWidget(self.show_crop_range, 0, 0, 1, 1)
        self.crop_plots = QtWidgets.QCheckBox(Form)
        self.crop_plots.setObjectName("crop_plots")
        self.gridLayout.addWidget(self.crop_plots, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        spacerItem2 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem2)
        self.label_2 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.gridLayout_25 = QtWidgets.QGridLayout()
        self.gridLayout_25.setObjectName("gridLayout_25")
        self.show_plot_details = QtWidgets.QCheckBox(Form)
        self.show_plot_details.setChecked(True)
        self.show_plot_details.setObjectName("show_plot_details")
        self.gridLayout_25.addWidget(self.show_plot_details, 0, 0, 1, 1)
        self.show_detected_states = QtWidgets.QCheckBox(Form)
        self.show_detected_states.setChecked(True)
        self.show_detected_states.setObjectName("show_detected_states")
        self.gridLayout_25.addWidget(self.show_detected_states, 0, 1, 1, 1)
        self.show_breakpoints = QtWidgets.QCheckBox(Form)
        self.show_breakpoints.setChecked(False)
        self.show_breakpoints.setObjectName("show_breakpoints")
        self.gridLayout_25.addWidget(self.show_breakpoints, 1, 0, 1, 1)
        self.show_correction_factor_ranges = QtWidgets.QCheckBox(Form)
        self.show_correction_factor_ranges.setChecked(False)
        self.show_correction_factor_ranges.setObjectName("show_correction_factor_ranges")
        self.gridLayout_25.addWidget(self.show_correction_factor_ranges, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_25)
        spacerItem3 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.verticalLayout.addItem(spacerItem3)
        self.label_3 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_17 = QtWidgets.QLabel(Form)
        self.label_17.setObjectName("label_17")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.plot_user_filter = QtWidgets.QComboBox(Form)
        self.plot_user_filter.setObjectName("plot_user_filter")
        self.plot_user_filter.addItem("")
        self.plot_user_filter.addItem("")
        self.plot_user_filter.addItem("")
        self.plot_user_filter.addItem("")
        self.plot_user_filter.addItem("")
        self.plot_user_filter.addItem("")
        self.plot_user_filter.addItem("")
        self.plot_user_filter.addItem("")
        self.plot_user_filter.addItem("")
        self.plot_user_filter.addItem("")
        self.plot_user_filter.addItem("")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.plot_user_filter)
        self.label_26 = QtWidgets.QLabel(Form)
        self.label_26.setObjectName("label_26")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_26)
        self.plot_nucleotide_filter = QtWidgets.QComboBox(Form)
        self.plot_nucleotide_filter.setObjectName("plot_nucleotide_filter")
        self.plot_nucleotide_filter.addItem("")
        self.plot_nucleotide_filter.addItem("")
        self.plot_nucleotide_filter.addItem("")
        self.plot_nucleotide_filter.addItem("")
        self.plot_nucleotide_filter.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.plot_nucleotide_filter)
        self.verticalLayout.addLayout(self.formLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "General Settings"))
        self.plot_normalise.setText(_translate("Form", "Normalise Data [N]"))
        self.plot_split_lines.setText(_translate("Form", "Split Plots With Multiple Lines [S]"))
        self.plot_showx.setText(_translate("Form", "Show X axis [X]"))
        self.plot_showy.setText(_translate("Form", "Show Y axis [Y]"))
        self.label_6.setText(_translate("Form", "Downsample Settings"))
        self.label_7.setText(_translate("Form", "Downsample"))
        self.plot_downsample.setItemText(0, _translate("Form", "1"))
        self.plot_downsample.setItemText(1, _translate("Form", "2"))
        self.plot_downsample.setItemText(2, _translate("Form", "3"))
        self.plot_downsample.setItemText(3, _translate("Form", "4"))
        self.plot_downsample.setItemText(4, _translate("Form", "5"))
        self.plot_downsample.setItemText(5, _translate("Form", "10"))
        self.label_4.setText(_translate("Form", "Crop Settings"))
        self.crop_reset_active.setText(_translate("Form", "Reset Crop Ranges (Active Plot)"))
        self.crop_reset_all.setText(_translate("Form", "Reset Crop Ranges (All Plots)"))
        self.show_crop_range.setText(_translate("Form", "Show Crop Ranges"))
        self.crop_plots.setText(_translate("Form", "Crop Plots"))
        self.label_2.setText(_translate("Form", "Visualisation Settings"))
        self.show_plot_details.setText(_translate("Form", "Show Plot Details Overlay"))
        self.show_detected_states.setText(_translate("Form", "Show Detected States"))
        self.show_breakpoints.setText(_translate("Form", "Show Break Points"))
        self.show_correction_factor_ranges.setText(_translate("Form", "Show FRET Ranges"))
        self.label_3.setText(_translate("Form", "Filter Settings"))
        self.label_17.setText(_translate("Form", "Localisation Filter"))
        self.plot_user_filter.setItemText(0, _translate("Form", "None"))
        self.plot_user_filter.setItemText(1, _translate("Form", "0"))
        self.plot_user_filter.setItemText(2, _translate("Form", "1"))
        self.plot_user_filter.setItemText(3, _translate("Form", "2"))
        self.plot_user_filter.setItemText(4, _translate("Form", "3"))
        self.plot_user_filter.setItemText(5, _translate("Form", "4"))
        self.plot_user_filter.setItemText(6, _translate("Form", "5"))
        self.plot_user_filter.setItemText(7, _translate("Form", "6"))
        self.plot_user_filter.setItemText(8, _translate("Form", "7"))
        self.plot_user_filter.setItemText(9, _translate("Form", "8"))
        self.plot_user_filter.setItemText(10, _translate("Form", "9"))
        self.label_26.setText(_translate("Form", "Nucleotide Filter"))
        self.plot_nucleotide_filter.setItemText(0, _translate("Form", "None"))
        self.plot_nucleotide_filter.setItemText(1, _translate("Form", "A"))
        self.plot_nucleotide_filter.setItemText(2, _translate("Form", "T"))
        self.plot_nucleotide_filter.setItemText(3, _translate("Form", "C"))
        self.plot_nucleotide_filter.setItemText(4, _translate("Form", "G"))
