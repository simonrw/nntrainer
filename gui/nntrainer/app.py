import sys
from PyQt5 import QtWidgets, QtGui
import os
import threading
import argparse
import queue

from .gui import nntrainer_ui, nntrainer_augmentation
from .modeltrainer import ModelTrainer
from .trainingoptions import TrainingOptions, EarlyStoppingOptions
from .architectures import ARCHITECTURES

LOSS_FUNCTIONS = ["categorical_crossentropy"]
OPTIMISERS = ["Adam", "SGD"]


def show_error_dialog(txt, icon=QtWidgets.QMessageBox.Critical):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Critical)
    msg.setText(txt)
    msg.setWindowTitle("Error")
    msg.exec_()


class NNTrainerApplication(QtWidgets.QMainWindow, nntrainer_ui.Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # Thread handle for training process. There should only be one as each
        # training process takes up the whole computer for training.
        self.trainer_thread = None

        # Internal state
        self.training_dir = None
        self.validation_dir = None
        self.output_dir = None

        # Perform initial setup
        self.setup_architecture_choices()
        self.setup_loss_choices()
        self.setup_optimiser_choices()

        # Set up connections
        self.quitButton.clicked.connect(lambda self: QtWidgets.QApplication.quit())
        self.earlyStoppingEnable.toggled.connect(self.enable_disable_early_stopping)
        self.trainButton.clicked.connect(self.run_training)
        self.trainingDirButton.clicked.connect(self.select_training_dir)
        self.validationDirButton.clicked.connect(self.select_validation_dir)
        self.outputDirBrowse.clicked.connect(self.select_output_dir)

    def enable_disable_early_stopping(self, checked):
        self.patienceValue.setEnabled(checked)
        self.minDeltaValue.setEnabled(checked)

    def setup_architecture_choices(self):
        self.architectureSelector.addItems(list(ARCHITECTURES.keys()))

    def setup_loss_choices(self):
        self.lossSelector.addItems(LOSS_FUNCTIONS)

    def setup_optimiser_choices(self):
        self.optimiserSelector.addItems(OPTIMISERS)

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

    def select_output_dir(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ShowDirsOnly
        dirname = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Output directory", os.path.expanduser("~"), options
        )
        self.set_output_dir(dirname)

    def set_training_dir(self, dirname):
        dirname = os.path.realpath(dirname)

        self.training_dir = dirname
        self.trainingDirValue.setText(dirname)

    def set_validation_dir(self, dirname):
        dirname = os.path.realpath(dirname)

        self.validation_dir = dirname
        self.validationDirValue.setText(dirname)

    def set_output_dir(self, dirname):
        dirname = os.path.realpath(dirname)

        self.output_dir = dirname
        self.outputDirValue.setText(dirname)

    def set_name(self, name):
        self.outputStub.setText(name)

    def run_training(self):
        if self.trainer_thread is not None:
            show_error_dialog("cannot start multiple training threads")
            return

        validation_errors = []

        opts = TrainingOptions()

        # Get the data into the training options object, after doing some validation
        # Model page
        opts.architecture = self.architectureSelector.currentText()
        opts.output_classes = self.numOutputClassesSelector.value()
        opts.num_fc_layers = self.numFCLayers.value()
        opts.fc_neurones = self.numFCNeurones.value()

        # Training page
        if self.training_dir is None:
            validation_errors.append("Training directory has not been set")

        if self.training_dir and not os.path.isdir(self.training_dir):
            validation_errors.append("Training directory does not exist")

        opts.training_dir = self.training_dir

        if self.validation_dir and not os.path.isdir(self.validation_dir):
            validation_errors.append("Validation directory does not exist")

        opts.image_shape = (self.imageDimX.value(), self.imageDimY.value())
        opts.batch_size = self.batchSizeSelector.value()

        opts.training_epochs = self.epochsValue.value()
        opts.optimiser = self.optimiserSelector.currentText()
        opts.loss_function = self.lossSelector.currentText()

        if self.earlyStoppingEnable.isChecked():
            early_stopping_options = EarlyStoppingOptions(
                patience=self.patienceValue.value(),
                minimum_delta=self.minDeltaValue.value(),
            )
            opts.early_stopping = early_stopping_options

        # Augmentation page
        opts.horizontal_flip = self.horizontalFlipCheck.isChecked()
        opts.vertical_flip = self.verticalFlipCheck.isChecked()
        opts.rotation_angle = self.rotationAngleValue.value()
        opts.width_shift_range = self.widthShiftValue.value()
        opts.height_shift_range = self.heightShiftValue.value()
        opts.brightness_shift_range = self.brightnessShiftValue.value()

        # Output page
        if not self.outputStub.text():
            validation_errors.append("Output name must not be empty")
        opts.output_name = self.outputStub.text()

        if self.output_dir is None:
            validation_errors.append("Output directory has not been set")

        opts.output_directory = self.output_dir

        # Finally show the validation problems
        if validation_errors:
            show_error_dialog(
                "Validation errors:\n{}".format("\n".join(validation_errors))
            )
            return

        self.trainer_thread = threading.Thread(
            target=lambda: ModelTrainer(opts).run(
                update_fn=self.model_update_ui, finish_callback=self.thread_finished
            )
        )
        self.trainer_thread.start()

    def model_update_ui(self, msg):
        self.lossHistoryPlot.plot(msg)

    def thread_finished(self):
        show_error_dialog("thread finished")
        self.trainer_thread = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--training-dir", required=False, help="Training data directory"
    )
    parser.add_argument(
        "-V", "--validation-dir", required=False, help="Validation data directory"
    )
    parser.add_argument(
        "-n", "--name", required=False, help="Name of the training session"
    )
    parser.add_argument("-o", "--output-dir", required=False, help="Output directory")
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    trainer = NNTrainerApplication()

    # Configure CLI options
    if args.training_dir:
        trainer.set_training_dir(args.training_dir)

    if args.validation_dir:
        trainer.set_validation_dir(args.validation_dir)

    if args.output_dir:
        trainer.set_output_dir(args.output_dir)

    if args.name:
        trainer.set_name(args.name)

    trainer.show()
    app.exec_()
