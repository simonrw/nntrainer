import sys
from .gui import nntrainer_ui, nntrainer_augmentation
from PyQt5 import QtWidgets, QtGui
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import os
from typing import NamedTuple, Any, Union, Tuple, List, Optional
from dataclasses import dataclass

ARCHITECTURES = {"ResNet50": tf.keras.applications.ResNet50}
LOSS_FUNCTIONS = ["categorical_crossentropy"]
OPTIMISERS = ["Adam", "SGD"]


def show_error_dialog(txt, icon=QtWidgets.QMessageBox.Critical):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Critical)
    msg.setText(txt)
    msg.setWindowTitle("Error")
    msg.exec_()


@dataclass
class AugmentationConfig(object):
    rotation_range: int
    width_shift_range: Any
    height_shift_range: Any
    brightness_range: List[float]
    zoom_range: Union[float, Tuple[float]]
    horizontal_flip: bool
    vertical_flip: bool

    @classmethod
    def blank(cls):
        return cls(
            rotation_range=None,
            width_shift_range=None,
            height_shift_range=None,
            brightness_range=None,
            zoom_range=None,
            horizontal_flip=None,
            vertical_flip=None,
        )


class NNTrainerApplication(QtWidgets.QMainWindow, nntrainer_ui.Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # Perform initial setup
        self.setup_architecture_choices()
        self.setup_loss_choices()
        self.setup_optimiser_choices()

        # Set up connections
        self.quitButton.clicked.connect(lambda self: QtWidgets.QApplication.quit())
        self.earlyStoppingEnable.toggled.connect(self.enable_disable_early_stopping)

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

    def set_training_dir(self, dirname):
        self.training_dir = dirname
        self.chooseTrainingDataLabel.setText(dirname)

    def set_validation_dir(self, dirname):
        self.validation_dir = dirname
        self.chooseValidationDataLabel.setText(dirname)

    def configure_augmentation(self):
        augmentation_config = NNTrainerAugmentation.get_params(parent=self)
        if augmentation_config is not None:
            self.augmentation_config = augmentation_config

    def run_training(self):
        pass


class NNTrainerAugmentation(QtWidgets.QDialog, nntrainer_augmentation.Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

    @classmethod
    def get_params(cls, parent=None) -> AugmentationConfig:
        self = cls(parent)
        result = self.exec_()

        if result == QtWidgets.QDialog.Accepted:
            return AugmentationConfig(
                rotation_range=self.rotationSelector.value(),
                width_shift_range=self.widthSelector.value(),
                height_shift_range=self.heightSelector.value(),
                brightness_range=(
                    self.brightnessMinSelector.value(),
                    self.brightnessMaxSelector.value(),
                ),
                zoom_range=(self.zoomMinSelector.value(), self.zoomMaxSelector.value()),
                horizontal_flip=self.horizontalFlipCheck.checkState() == 1,
                vertical_flip=self.verticalFlipCheck.checkState() == 1,
            )


class ModelTrainer(object):
    def __init__(
        self,
        *,
        training_dir,
        validation_dir,
        model_cls,
        n_fc,
        fc_dim,
        optimiser,
        loss_function,
        input_shape,
        classes
    ):
        self.training_dir = training_dir
        self.validation_dir = validation_dir
        self.model_cls = model_cls
        self.n_fc = n_fc
        self.fc_dim = fc_dim
        self.optimiser = optimiser
        self.loss_function = loss_function
        self.input_shape = input_shape
        self.classes = classes

    def run(self):
        model = self.build_model()
        training_datagen = self.build_datagen(self.training_dir)

        print("Finished training")

    def build_model(self):
        # Create the custom first layer, which has the dimensions of the data
        input_tensor = tf.keras.layers.Input(shape=self.input_shape)

        # This is the base model class from the architecture that the user specified. We
        # constrain the solution to pre-trained models using imagenet weights.
        base_model = self.model_cls(
            weights="imagenet",
            include_top=False,
            input_tensor=input_tensor,
            classes=self.classes,
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
        for layer_idx in range(self.n_fc):
            model.add(Dense(self.fc_dim, activation="relu"))

        # Add the final classification layer
        model.add(Dense(self.classes, activation="softmax"))

        # The model is complete, so compile it using the optimiser and loss functions
        # specified
        model.compile(
            optimizer=self.optimiser, metrics=["accuracy"], loss=self.loss_function
        )

        return model

    def build_datagen(self, training_dir):
        gen = tf.keras.preprocessing.image.InputDataGenerator(
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


def main():
    app = QtWidgets.QApplication(sys.argv)
    trainer = NNTrainerApplication()
    trainer.show()
    app.exec_()
