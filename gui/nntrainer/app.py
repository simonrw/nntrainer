import sys
from .gui import nntrainer_ui as ui
from PyQt5 import QtWidgets, QtGui
import tensorflow as tf
import os

ARCHITECTURES = {"ResNet50": tf.keras.applications.ResNet50}


class NNTrainerApplication(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # Variables
        self.training_dir = None
        self.validation_dir = None

        # Perform initial setup
        self.setup_architecture_choices()

        # Create connections
        self.chooseTrainingDataButton.clicked.connect(self.select_training_dir)
        self.chooseValidationDataButton.clicked.connect(self.select_validation_dir)

    def setup_architecture_choices(self):
        self.architecture_choice.addItems(list(ARCHITECTURES.keys()))

    def select_training_dir(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ShowDirsOnly
        dirname = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Training data", os.path.expanduser("~"), options
        )
        self.set_training_dir(dirname)

    def select_validation_dir(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ShowDirsOnly
        dirname = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Validation data", os.path.expanduser("~"), options
        )
        self.set_validation_dir(dirname)

    def set_training_dir(self, dirname):
        self.training_dir = dirname
        self.chooseTrainingDataLabel.setText(dirname)

    def set_validation_dir(self, dirname):
        self.validation_dir = dirname
        self.chooseValidationDataLabel.setText(dirname)


def main():
    app = QtWidgets.QApplication(sys.argv)
    trainer = NNTrainerApplication()
    trainer.show()
    app.exec_()
