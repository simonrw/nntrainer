import sys
from .gui import nntrainer_ui, nntrainer_augmentation
from PyQt5 import QtWidgets, QtGui
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import os
from typing import NamedTuple, Any, Union, Tuple, List, Optional
from dataclasses import dataclass

ARCHITECTURES = {
    "ResNet50": tf.keras.applications.ResNet50,
    "VGG16": tf.keras.applications.VGG16,
}
LOSS_FUNCTIONS = ["categorical_crossentropy"]
OPTIMISERS = ["Adam", "SGD"]


def show_error_dialog(txt, icon=QtWidgets.QMessageBox.Critical):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Critical)
    msg.setText(txt)
    msg.setWindowTitle("Error")
    msg.exec_()


@dataclass
class EarlyStoppingOptions:
    patience: int
    minimum_delta: float


@dataclass
class TrainingOptions:
    # Model
    architecture: Optional[str] = None
    output_classes: Optional[int] = None
    num_fc_layers: Optional[int] = None
    fc_neurones: Optional[int] = None

    # Training
    training_dir: Optional[str] = None
    validation_dir: Optional[str] = None
    image_shape: Optional[Tuple[int, int]] = None
    training_epochs: Optional[int] = None
    optimiser: Optional[str] = None
    loss_function: Optional[str] = None
    early_stopping: Optional[EarlyStoppingOptions] = None

    # Augmentation
    horizontal_flip: bool = False
    vertical_flip: bool = False
    rotation_angle: Optional[bool] = None
    width_shift_range: Optional[int] = None
    height_shift_range: Optional[int] = None
    brightness_shift_range: Optional[int] = None

    # Output
    output_name: Optional[str] = None
    output_directory: Optional[str] = None


class NNTrainerApplication(QtWidgets.QMainWindow, nntrainer_ui.Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

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
        self.training_dir = dirname
        self.trainingDirValue.setText(dirname)

    def set_validation_dir(self, dirname):
        self.validation_dir = dirname
        self.validationDirValue.setText(dirname)

    def set_output_dir(self, dirname):
        self.output_dir = dirname
        self.outputDirValue.setText(dirname)

    def configure_augmentation(self):
        augmentation_config = NNTrainerAugmentation.get_params(parent=self)
        if augmentation_config is not None:
            self.augmentation_config = augmentation_config

    def run_training(self):
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

        trainer = ModelTrainer(opts).run()
        # TODO: update the GUI with trainer outputs


class ModelTrainer(object):
    def __init__(self, opts: TrainingOptions):
        self.opts = opts

    def run(self):
        model = self.build_model()
        training_datagen = self.build_datagen(self.opts.training_dir)

        print("Finished training")

    def build_model(self):
        # Create the custom first layer, which has the dimensions of the data
        input_shape = (self.opts.image_shape[0], self.opts.image_shape[1], 3)
        input_tensor = tf.keras.layers.Input(shape=input_shape)

        # This is the base model class from the architecture that the user specified. We
        # constrain the solution to pre-trained models using imagenet weights.
        base_model = self.model_cls(
            weights="imagenet",
            include_top=False,
            input_tensor=input_tensor,
            classes=self.opts.output_classes,
        )

        # The final model
        model = Sequential()

        # Add the base model, i.e. the model that has already been trained using
        # ImageNet, minus the first few layers (as we asked for `include_top=False`).
        model.add(base_model)

        # Flatten the output of the feature extractor before adding the fully connected
        # layers.
        model.add(Flatten())

        # Add the fully connected layers
        for layer_idx in range(self.opts.num_fc_layers):
            model.add(Dense(self.opts.fc_neurones, activation="relu"))

        # Add the final classification layer
        model.add(Dense(self.opts.output_classes, activation="softmax"))

        # The model is complete, so compile it using the optimiser and loss functions
        # specified
        model.compile(
            optimizer=self.opts.optimiser,
            metrics=["accuracy"],
            loss=self.opts.loss_function,
        )

        return model

    def build_datagen(self, training_dir):
        gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocess_input_fn()
        )

    def preprocess_input_fn(self):
        """Returns the correct function for preprocessing the data, based on the
        architecture that has been selected
        """
        fns = {
            tf.keras.applications.ResNet50: tf.keras.applications.resnet50.preprocess_input,
            tf.keras.applications.VGG16: tf.keras.applications.vgg16.preprocess_input,
        }

        return fns[self.model_cls]

    @property
    def model_cls(self):
        return ARCHITECTURES[self.opts.architecture]


def main():
    app = QtWidgets.QApplication(sys.argv)
    trainer = NNTrainerApplication()
    trainer.show()
    app.exec_()
