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
        Form.resize(354, 197)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.export_location = QtWidgets.QComboBox(Form)
        self.export_location.setObjectName("export_location")
        self.export_location.addItem("")
        self.export_location.addItem("")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.export_location)
        self.label_17 = QtWidgets.QLabel(Form)
        self.label_17.setObjectName("label_17")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.export_user_filter = QtWidgets.QComboBox(Form)
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
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.export_user_filter)
        self.label_26 = QtWidgets.QLabel(Form)
        self.label_26.setObjectName("label_26")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_26)
        self.export_nucleotide_filter = QtWidgets.QComboBox(Form)
        self.export_nucleotide_filter.setObjectName("export_nucleotide_filter")
        self.export_nucleotide_filter.addItem("")
        self.export_nucleotide_filter.addItem("")
        self.export_nucleotide_filter.addItem("")
        self.export_nucleotide_filter.addItem("")
        self.export_nucleotide_filter.addItem("")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.export_nucleotide_filter)
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.export_mode = QtWidgets.QComboBox(Form)
        self.export_mode.setObjectName("export_mode")
        self.export_mode.addItem("")
        self.export_mode.addItem("")
        self.export_mode.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.export_mode)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.export_data_selection = QtWidgets.QComboBox(Form)
        self.export_data_selection.setObjectName("export_data_selection")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.export_data_selection.addItem("")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.export_data_selection)
        self.verticalLayout.addLayout(self.formLayout)
        self.export_split_datasets = QtWidgets.QCheckBox(Form)
        self.export_split_datasets.setObjectName("export_split_datasets")
        self.verticalLayout.addWidget(self.export_split_datasets)
        self.export_gapseq = QtWidgets.QPushButton(Form)
        self.export_gapseq.setObjectName("export_gapseq")
        self.verticalLayout.addWidget(self.export_gapseq)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_2.setText(_translate("Form", "Export Location"))
        self.export_location.setItemText(0, _translate("Form", "Select Directory"))
        self.export_location.setItemText(1, _translate("Form", "Import Directory"))
        self.label_17.setText(_translate("Form", "Localisation Filter"))
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
        self.label_26.setText(_translate("Form", "Nucleotide Filter"))
        self.export_nucleotide_filter.setItemText(0, _translate("Form", "None"))
        self.export_nucleotide_filter.setItemText(1, _translate("Form", "A"))
        self.export_nucleotide_filter.setItemText(2, _translate("Form", "T"))
        self.export_nucleotide_filter.setItemText(3, _translate("Form", "C"))
        self.export_nucleotide_filter.setItemText(4, _translate("Form", "G"))
        self.label.setText(_translate("Form", "Export Mode"))
        self.export_mode.setItemText(0, _translate("Form", "GapSeq (.json)"))
        self.export_mode.setItemText(1, _translate("Form", "Excel (.xlsx)"))
        self.export_mode.setItemText(2, _translate("Form", "Dat (.dat)"))
        self.label_3.setText(_translate("Form", "Export Data"))
        self.export_data_selection.setItemText(0, _translate("Form", "Donor"))
        self.export_data_selection.setItemText(1, _translate("Form", "Acceptor"))
        self.export_data_selection.setItemText(2, _translate("Form", "FRET Data"))
        self.export_data_selection.setItemText(3, _translate("Form", "FRET Efficiency"))
        self.export_data_selection.setItemText(4, _translate("Form", "FRET Data + FRET Efficiency"))
        self.export_data_selection.setItemText(5, _translate("Form", "ALEX Data"))
        self.export_data_selection.setItemText(6, _translate("Form", "ALEX Efficiency"))
        self.export_data_selection.setItemText(7, _translate("Form", "ALEX Data + ALEX Efficiency"))
        self.export_split_datasets.setText(_translate("Form", "Split Datasets into seperate files"))
        self.export_gapseq.setText(_translate("Form", "Export GapSeq"))
