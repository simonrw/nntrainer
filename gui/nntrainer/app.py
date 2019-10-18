import sys
from .gui import nntrainer_ui as ui
from PyQt5 import QtWidgets, QtGui
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import os

ARCHITECTURES = {"ResNet50": tf.keras.applications.ResNet50}


def show_error_dialog(txt, icon=QtWidgets.QMessageBox.Critical):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Critical)
    msg.setText(txt)
    msg.setWindowTitle("Error")
    msg.exec_()


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
        self.trainButton.clicked.connect(self.run_training)
        self.quitButton.clicked.connect(self.quit)

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

    def quit(self):
        QtWidgets.QApplication.quit()

    def run_training(self):
        try:
            if self.training_dir is None:
                show_error_dialog("Training dir is not set")
                return

            model_cls = ARCHITECTURES[str(self.architecture_choice.currentText())]

            trainer = ModelTrainer(
                training_dir=self.training_dir,
                validation_dir=self.validation_dir,
                model_cls=model_cls,
            )

            trainer.run()

        except Exception as e:
            # General error handler, show the message to the user
            show_error_dialog(str(e))


class ModelTrainer(object):
    def __init__(
        self,
        *,
        training_dir,
        validation_dir,
        model_cls,
        n_fc,
        fc_dim,
        input_shape,
        classes
    ):
        self.training_dir = training_dir
        self.validation_dir = validation_dir
        self.model_cls = model_cls
        self.n_fc = n_fc
        self.fc_dim = fc_dim
        self.input_shape = input_shape
        self.classes = classes

    def run(self):
        model = self.build_model()

        print("Finished training")

    def build_model(self):
        # TODO: optionally get the input dimensions of the model
        input_tensor = tf.keras.layers.Input(shape=self.input_shape)
        base_model = self.model_cls(
            weights="imagenet",
            include_top=False,
            input_tensor=input_tensor,
            classes=self.classes,
        )

        model = Sequential()
        model.add(base_model)

        model.add(Flatten())

        # TODO: include fully connected layers
        for layer_idx in range(self.n_fc):
            model.add(Dense(self.fc_dim, activation="relu"))

        model.add(Dense(self.classes, activation="softmax"))

        # TODO: optimizer should be input
        # TODO: loss function should be input
        model.compile(
            optimizer="adam", metrics=["accuracy"], loss="categorical_crossentropy"
        )

        return model


def main():
    app = QtWidgets.QApplication(sys.argv)
    trainer = NNTrainerApplication()
    trainer.show()
    app.exec_()
