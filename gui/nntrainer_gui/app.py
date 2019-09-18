from . import nntrainer_ui as ui
from PyQt5 import QtWidgets, QtGui


class NNTrainerApplication(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
