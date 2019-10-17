import sys
from .gui import nntrainer_ui as ui
from PyQt5 import QtWidgets, QtGui
import tensorflow as tf

ARCHITECTURES = {"ResNet50": tf.keras.applications.ResNet50}


class NNTrainerApplication(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # Perform initial setup
        self.setup_architecture_choices()

    def setup_architecture_choices(self):
        self.architecture_choice.addItems(list(ARCHITECTURES.keys()))


def main():
    app = QtWidgets.QApplication(sys.argv)
    trainer = NNTrainerApplication()
    trainer.show()
    app.exec_()
