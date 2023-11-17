# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'exportsettings_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(415, 316)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.export_location = QtWidgets.QComboBox(self.tab)
        self.export_location.setObjectName("export_location")
        self.export_location.addItem("")
        self.export_location.addItem("")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.export_location)
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.export_mode = QtWidgets.QComboBox(self.tab)
        self.export_mode.setObjectName("export_mode")
        self.export_mode.addItem("")
        self.export_mode.addItem("")
        self.export_mode.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.export_mode)
        self.export_data_selection_label = QtWidgets.QLabel(self.tab)
        self.export_data_selection_label.setObjectName("export_data_selection_label")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.export_data_selection_label)
        self.export_data_selection = QtWidgets.QComboBox(self.tab)
        self.export_data_selection.setObjectName("export_data_selection")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.export_data_selection)
        self.export_user_filter_label = QtWidgets.QLabel(self.tab)
        self.export_user_filter_label.setObjectName("export_user_filter_label")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.export_user_filter_label)
        self.export_user_filter = QtWidgets.QComboBox(self.tab)
        self.export_user_filter.setObjectName("export_user_filter")
        self.export_user_filter.addItem("")
        self.export_user_filter.addItem("")
        self.export_user_filter.addItem("")
        self.export_user_filter.addItem("")
        self.export_user_filter.addItem("")
        self.export_user_filter.addItem("")
        self.export_user_filter.addItem("")
        self.export_user_filter.addItem("")
        self.export_user_filter.addItem("")
        self.export_user_filter.addItem("")
        self.export_user_filter.addItem("")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.export_user_filter)
        self.export_nucleotide_filter_label = QtWidgets.QLabel(self.tab)
        self.export_nucleotide_filter_label.setObjectName("export_nucleotide_filter_label")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.export_nucleotide_filter_label)
        self.export_nucleotide_filter = QtWidgets.QComboBox(self.tab)
        self.export_nucleotide_filter.setObjectName("export_nucleotide_filter")
        self.export_nucleotide_filter.addItem("")
        self.export_nucleotide_filter.addItem("")
        self.export_nucleotide_filter.addItem("")
        self.export_nucleotide_filter.addItem("")
        self.export_nucleotide_filter.addItem("")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.export_nucleotide_filter)
        self.label_9 = QtWidgets.QLabel(self.tab)
        self.label_9.setObjectName("label_9")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.export_separator = QtWidgets.QComboBox(self.tab)
        self.export_separator.setObjectName("export_separator")
        self.export_separator.addItem("")
        self.export_separator.addItem("")
        self.export_separator.addItem("")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.export_separator)
        self.verticalLayout_2.addLayout(self.formLayout)
        self.export_crop_data = QtWidgets.QCheckBox(self.tab)
        self.export_crop_data.setObjectName("export_crop_data")
        self.verticalLayout_2.addWidget(self.export_crop_data)
        self.export_fitted_states = QtWidgets.QCheckBox(self.tab)
        self.export_fitted_states.setObjectName("export_fitted_states")
        self.verticalLayout_2.addWidget(self.export_fitted_states)
        self.export_split_datasets = QtWidgets.QCheckBox(self.tab)
        self.export_split_datasets.setObjectName("export_split_datasets")
        self.verticalLayout_2.addWidget(self.export_split_datasets)
        self.export_gapseq = QtWidgets.QPushButton(self.tab)
        self.export_gapseq.setObjectName("export_gapseq")
        self.verticalLayout_2.addWidget(self.export_gapseq)
        self.tabWidget.addTab(self.tab, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.formLayout_5 = QtWidgets.QFormLayout()
        self.formLayout_5.setObjectName("formLayout_5")
        self.label_6 = QtWidgets.QLabel(self.tab_3)
        self.label_6.setObjectName("label_6")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.excel_export_location = QtWidgets.QComboBox(self.tab_3)
        self.excel_export_location.setObjectName("excel_export_location")
        self.excel_export_location.addItem("")
        self.excel_export_location.addItem("")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.excel_export_location)
        self.export_data_selection_label_3 = QtWidgets.QLabel(self.tab_3)
        self.export_data_selection_label_3.setObjectName("export_data_selection_label_3")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.export_data_selection_label_3)
        self.excel_export_data_selection = QtWidgets.QComboBox(self.tab_3)
        self.excel_export_data_selection.setObjectName("excel_export_data_selection")
        self.excel_export_data_selection.addItem("")
        self.excel_export_data_selection.addItem("")
        self.excel_export_data_selection.addItem("")
        self.excel_export_data_selection.addItem("")
        self.excel_export_data_selection.addItem("")
        self.excel_export_data_selection.addItem("")
        self.excel_export_data_selection.addItem("")
        self.excel_export_data_selection.addItem("")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.excel_export_data_selection)
        self.export_user_filter_label_3 = QtWidgets.QLabel(self.tab_3)
        self.export_user_filter_label_3.setObjectName("export_user_filter_label_3")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.export_user_filter_label_3)
        self.excel_export_user_filter = QtWidgets.QComboBox(self.tab_3)
        self.excel_export_user_filter.setObjectName("excel_export_user_filter")
        self.excel_export_user_filter.addItem("")
        self.excel_export_user_filter.addItem("")
        self.excel_export_user_filter.addItem("")
        self.excel_export_user_filter.addItem("")
        self.excel_export_user_filter.addItem("")
        self.excel_export_user_filter.addItem("")
        self.excel_export_user_filter.addItem("")
        self.excel_export_user_filter.addItem("")
        self.excel_export_user_filter.addItem("")
        self.excel_export_user_filter.addItem("")
        self.excel_export_user_filter.addItem("")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.excel_export_user_filter)
        self.export_nucleotide_filter_label_3 = QtWidgets.QLabel(self.tab_3)
        self.export_nucleotide_filter_label_3.setObjectName("export_nucleotide_filter_label_3")
        self.formLayout_5.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.export_nucleotide_filter_label_3)
        self.excel_export_nucleotide_filter = QtWidgets.QComboBox(self.tab_3)
        self.excel_export_nucleotide_filter.setObjectName("excel_export_nucleotide_filter")
        self.excel_export_nucleotide_filter.addItem("")
        self.excel_export_nucleotide_filter.addItem("")
        self.excel_export_nucleotide_filter.addItem("")
        self.excel_export_nucleotide_filter.addItem("")
        self.excel_export_nucleotide_filter.addItem("")
        self.formLayout_5.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.excel_export_nucleotide_filter)
        self.verticalLayout_4.addLayout(self.formLayout_5)
        self.excel_export_crop_data = QtWidgets.QCheckBox(self.tab_3)
        self.excel_export_crop_data.setObjectName("excel_export_crop_data")
        self.verticalLayout_4.addWidget(self.excel_export_crop_data)
        self.excel_export_fitted_states = QtWidgets.QCheckBox(self.tab_3)
        self.excel_export_fitted_states.setObjectName("excel_export_fitted_states")
        self.verticalLayout_4.addWidget(self.excel_export_fitted_states)
        self.excel_export_split_datasets = QtWidgets.QCheckBox(self.tab_3)
        self.excel_export_split_datasets.setObjectName("excel_export_split_datasets")
        self.verticalLayout_4.addWidget(self.excel_export_split_datasets)
        self.export_excel = QtWidgets.QPushButton(self.tab_3)
        self.export_excel.setObjectName("export_excel")
        self.verticalLayout_4.addWidget(self.export_excel)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.tab_4)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.formLayout_7 = QtWidgets.QFormLayout()
        self.formLayout_7.setObjectName("formLayout_7")
        self.label_11 = QtWidgets.QLabel(self.tab_4)
        self.label_11.setObjectName("label_11")
        self.formLayout_7.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.origin_export_location = QtWidgets.QComboBox(self.tab_4)
        self.origin_export_location.setObjectName("origin_export_location")
        self.origin_export_location.addItem("")
        self.origin_export_location.addItem("")
        self.formLayout_7.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.origin_export_location)
        self.export_data_selection_label_6 = QtWidgets.QLabel(self.tab_4)
        self.export_data_selection_label_6.setObjectName("export_data_selection_label_6")
        self.formLayout_7.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.export_data_selection_label_6)
        self.origin_export_data_selection = QtWidgets.QComboBox(self.tab_4)
        self.origin_export_data_selection.setObjectName("origin_export_data_selection")
        self.origin_export_data_selection.addItem("")
        self.origin_export_data_selection.addItem("")
        self.origin_export_data_selection.addItem("")
        self.origin_export_data_selection.addItem("")
        self.origin_export_data_selection.addItem("")
        self.origin_export_data_selection.addItem("")
        self.origin_export_data_selection.addItem("")
        self.origin_export_data_selection.addItem("")
        self.formLayout_7.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.origin_export_data_selection)
        self.export_user_filter_label_6 = QtWidgets.QLabel(self.tab_4)
        self.export_user_filter_label_6.setObjectName("export_user_filter_label_6")
        self.formLayout_7.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.export_user_filter_label_6)
        self.origin_export_user_filter = QtWidgets.QComboBox(self.tab_4)
        self.origin_export_user_filter.setObjectName("origin_export_user_filter")
        self.origin_export_user_filter.addItem("")
        self.origin_export_user_filter.addItem("")
        self.origin_export_user_filter.addItem("")
        self.origin_export_user_filter.addItem("")
        self.origin_export_user_filter.addItem("")
        self.origin_export_user_filter.addItem("")
        self.origin_export_user_filter.addItem("")
        self.origin_export_user_filter.addItem("")
        self.origin_export_user_filter.addItem("")
        self.origin_export_user_filter.addItem("")
        self.origin_export_user_filter.addItem("")
        self.formLayout_7.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.origin_export_user_filter)
        self.export_nucleotide_filter_label_6 = QtWidgets.QLabel(self.tab_4)
        self.export_nucleotide_filter_label_6.setObjectName("export_nucleotide_filter_label_6")
        self.formLayout_7.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.export_nucleotide_filter_label_6)
        self.origin_export_nucleotide_filter = QtWidgets.QComboBox(self.tab_4)
        self.origin_export_nucleotide_filter.setObjectName("origin_export_nucleotide_filter")
        self.origin_export_nucleotide_filter.addItem("")
        self.origin_export_nucleotide_filter.addItem("")
        self.origin_export_nucleotide_filter.addItem("")
        self.origin_export_nucleotide_filter.addItem("")
        self.origin_export_nucleotide_filter.addItem("")
        self.formLayout_7.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.origin_export_nucleotide_filter)
        self.verticalLayout_5.addLayout(self.formLayout_7)
        self.origin_export_crop_data = QtWidgets.QCheckBox(self.tab_4)
        self.origin_export_crop_data.setObjectName("origin_export_crop_data")
        self.verticalLayout_5.addWidget(self.origin_export_crop_data)
        self.origin_export_fitted_states = QtWidgets.QCheckBox(self.tab_4)
        self.origin_export_fitted_states.setObjectName("origin_export_fitted_states")
        self.verticalLayout_5.addWidget(self.origin_export_fitted_states)
        self.origin_export_split_datasets = QtWidgets.QCheckBox(self.tab_4)
        self.origin_export_split_datasets.setObjectName("origin_export_split_datasets")
        self.verticalLayout_5.addWidget(self.origin_export_split_datasets)
        self.export_origin = QtWidgets.QPushButton(self.tab_4)
        self.export_origin.setObjectName("export_origin")
        self.verticalLayout_5.addWidget(self.export_origin)
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setObjectName("label_5")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.json_export_location = QtWidgets.QComboBox(self.tab_2)
        self.json_export_location.setObjectName("json_export_location")
        self.json_export_location.addItem("")
        self.json_export_location.addItem("")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.json_export_location)
        self.verticalLayout_3.addLayout(self.formLayout_3)
        self.export_json = QtWidgets.QPushButton(self.tab_2)
        self.export_json.setObjectName("export_json")
        self.verticalLayout_3.addWidget(self.export_json)
        self.tabWidget.addTab(self.tab_2, "")
        self.verticalLayout.addWidget(self.tabWidget)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_2.setText(_translate("Form", "Export Location"))
        self.export_location.setItemText(0, _translate("Form", "Select Directory"))
        self.export_location.setItemText(1, _translate("Form", "Import Directory"))
        self.label.setText(_translate("Form", "Export Mode"))
        self.export_mode.setItemText(0, _translate("Form", "Dat (.dat)"))
        self.export_mode.setItemText(1, _translate("Form", "Text (.txt)"))
        self.export_mode.setItemText(2, _translate("Form", "CSV (.csv)"))
        self.export_data_selection_label.setText(_translate("Form", "Export Data"))
        self.export_data_selection.setItemText(0, _translate("Form", "Donor"))
        self.export_data_selection.setItemText(1, _translate("Form", "Acceptor"))
        self.export_data_selection.setItemText(2, _translate("Form", "FRET Data"))
        self.export_data_selection.setItemText(3, _translate("Form", "FRET Efficiency"))
        self.export_data_selection.setItemText(4, _translate("Form", "FRET Data + FRET Efficiency"))
        self.export_data_selection.setItemText(5, _translate("Form", "ALEX Data"))
        self.export_data_selection.setItemText(6, _translate("Form", "ALEX Efficiency"))
        self.export_data_selection.setItemText(7, _translate("Form", "ALEX Data + ALEX Efficiency"))
        self.export_user_filter_label.setText(_translate("Form", "Localisation Filter"))
        self.export_user_filter.setItemText(0, _translate("Form", "None"))
        self.export_user_filter.setItemText(1, _translate("Form", "0"))
        self.export_user_filter.setItemText(2, _translate("Form", "1"))
        self.export_user_filter.setItemText(3, _translate("Form", "2"))
        self.export_user_filter.setItemText(4, _translate("Form", "3"))
        self.export_user_filter.setItemText(5, _translate("Form", "4"))
        self.export_user_filter.setItemText(6, _translate("Form", "5"))
        self.export_user_filter.setItemText(7, _translate("Form", "6"))
        self.export_user_filter.setItemText(8, _translate("Form", "7"))
        self.export_user_filter.setItemText(9, _translate("Form", "8"))
        self.export_user_filter.setItemText(10, _translate("Form", "9"))
        self.export_nucleotide_filter_label.setText(_translate("Form", "Nucleotide Filter"))
        self.export_nucleotide_filter.setItemText(0, _translate("Form", "None"))
        self.export_nucleotide_filter.setItemText(1, _translate("Form", "A"))
        self.export_nucleotide_filter.setItemText(2, _translate("Form", "T"))
        self.export_nucleotide_filter.setItemText(3, _translate("Form", "C"))
        self.export_nucleotide_filter.setItemText(4, _translate("Form", "G"))
        self.label_9.setText(_translate("Form", "Export Separator"))
        self.export_separator.setItemText(0, _translate("Form", "Tab"))
        self.export_separator.setItemText(1, _translate("Form", "Comma"))
        self.export_separator.setItemText(2, _translate("Form", "Space"))
        self.export_crop_data.setText(_translate("Form", "Crop Data using Crop Ranges"))
        self.export_fitted_states.setText(_translate("Form", "Export Fitted States Data"))
        self.export_split_datasets.setText(_translate("Form", "Split Datasets into seperate files"))
        self.export_gapseq.setText(_translate("Form", "Export Files"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "Export Files"))
        self.label_6.setText(_translate("Form", "Export Location"))
        self.excel_export_location.setItemText(0, _translate("Form", "Select Directory"))
        self.excel_export_location.setItemText(1, _translate("Form", "Import Directory"))
        self.export_data_selection_label_3.setText(_translate("Form", "Export Data"))
        self.excel_export_data_selection.setItemText(0, _translate("Form", "Donor"))
        self.excel_export_data_selection.setItemText(1, _translate("Form", "Acceptor"))
        self.excel_export_data_selection.setItemText(2, _translate("Form", "FRET Data"))
        self.excel_export_data_selection.setItemText(3, _translate("Form", "FRET Efficiency"))
        self.excel_export_data_selection.setItemText(4, _translate("Form", "FRET Data + FRET Efficiency"))
        self.excel_export_data_selection.setItemText(5, _translate("Form", "ALEX Data"))
        self.excel_export_data_selection.setItemText(6, _translate("Form", "ALEX Efficiency"))
        self.excel_export_data_selection.setItemText(7, _translate("Form", "ALEX Data + ALEX Efficiency"))
        self.export_user_filter_label_3.setText(_translate("Form", "Localisation Filter"))
        self.excel_export_user_filter.setItemText(0, _translate("Form", "None"))
        self.excel_export_user_filter.setItemText(1, _translate("Form", "0"))
        self.excel_export_user_filter.setItemText(2, _translate("Form", "1"))
        self.excel_export_user_filter.setItemText(3, _translate("Form", "2"))
        self.excel_export_user_filter.setItemText(4, _translate("Form", "3"))
        self.excel_export_user_filter.setItemText(5, _translate("Form", "4"))
        self.excel_export_user_filter.setItemText(6, _translate("Form", "5"))
        self.excel_export_user_filter.setItemText(7, _translate("Form", "6"))
        self.excel_export_user_filter.setItemText(8, _translate("Form", "7"))
        self.excel_export_user_filter.setItemText(9, _translate("Form", "8"))
        self.excel_export_user_filter.setItemText(10, _translate("Form", "9"))
        self.export_nucleotide_filter_label_3.setText(_translate("Form", "Nucleotide Filter"))
        self.excel_export_nucleotide_filter.setItemText(0, _translate("Form", "None"))
        self.excel_export_nucleotide_filter.setItemText(1, _translate("Form", "A"))
        self.excel_export_nucleotide_filter.setItemText(2, _translate("Form", "T"))
        self.excel_export_nucleotide_filter.setItemText(3, _translate("Form", "C"))
        self.excel_export_nucleotide_filter.setItemText(4, _translate("Form", "G"))
        self.excel_export_crop_data.setText(_translate("Form", "Crop Data using Crop Ranges"))
        self.excel_export_fitted_states.setText(_translate("Form", "Export Fitted States Data"))
        self.excel_export_split_datasets.setText(_translate("Form", "Split Datasets into seperate files"))
        self.export_excel.setText(_translate("Form", "Export Excel"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("Form", "Export Excel"))
        self.label_11.setText(_translate("Form", "Export Location"))
        self.origin_export_location.setItemText(0, _translate("Form", "Select Directory"))
        self.origin_export_location.setItemText(1, _translate("Form", "Import Directory"))
        self.export_data_selection_label_6.setText(_translate("Form", "Export Data"))
        self.origin_export_data_selection.setItemText(0, _translate("Form", "Donor"))
        self.origin_export_data_selection.setItemText(1, _translate("Form", "Acceptor"))
        self.origin_export_data_selection.setItemText(2, _translate("Form", "FRET Data"))
        self.origin_export_data_selection.setItemText(3, _translate("Form", "FRET Efficiency"))
        self.origin_export_data_selection.setItemText(4, _translate("Form", "FRET Data + FRET Efficiency"))
        self.origin_export_data_selection.setItemText(5, _translate("Form", "ALEX Data"))
        self.origin_export_data_selection.setItemText(6, _translate("Form", "ALEX Efficiency"))
        self.origin_export_data_selection.setItemText(7, _translate("Form", "ALEX Data + ALEX Efficiency"))
        self.export_user_filter_label_6.setText(_translate("Form", "Localisation Filter"))
        self.origin_export_user_filter.setItemText(0, _translate("Form", "None"))
        self.origin_export_user_filter.setItemText(1, _translate("Form", "0"))
        self.origin_export_user_filter.setItemText(2, _translate("Form", "1"))
        self.origin_export_user_filter.setItemText(3, _translate("Form", "2"))
        self.origin_export_user_filter.setItemText(4, _translate("Form", "3"))
        self.origin_export_user_filter.setItemText(5, _translate("Form", "4"))
        self.origin_export_user_filter.setItemText(6, _translate("Form", "5"))
        self.origin_export_user_filter.setItemText(7, _translate("Form", "6"))
        self.origin_export_user_filter.setItemText(8, _translate("Form", "7"))
        self.origin_export_user_filter.setItemText(9, _translate("Form", "8"))
        self.origin_export_user_filter.setItemText(10, _translate("Form", "9"))
        self.export_nucleotide_filter_label_6.setText(_translate("Form", "Nucleotide Filter"))
        self.origin_export_nucleotide_filter.setItemText(0, _translate("Form", "None"))
        self.origin_export_nucleotide_filter.setItemText(1, _translate("Form", "A"))
        self.origin_export_nucleotide_filter.setItemText(2, _translate("Form", "T"))
        self.origin_export_nucleotide_filter.setItemText(3, _translate("Form", "C"))
        self.origin_export_nucleotide_filter.setItemText(4, _translate("Form", "G"))
        self.origin_export_crop_data.setText(_translate("Form", "Crop Data using Crop Ranges"))
        self.origin_export_fitted_states.setText(_translate("Form", "Export Fitted States Data"))
        self.origin_export_split_datasets.setText(_translate("Form", "Split Datasets into seperate files"))
        self.export_origin.setText(_translate("Form", "Export OriginLab"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("Form", "Export OriginLab"))
        self.label_5.setText(_translate("Form", "Export Location"))
        self.json_export_location.setItemText(0, _translate("Form", "Select Directory"))
        self.json_export_location.setItemText(1, _translate("Form", "Import Directory"))
        self.export_json.setText(_translate("Form", "Export JSON Dataset"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "Export JSON Dataset"))
