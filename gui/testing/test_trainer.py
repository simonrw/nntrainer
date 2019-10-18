import pytest
from unittest import mock
from nntrainer.app import ModelTrainer
import tensorflow as tf


CLASSES = 15
INPUT_SHAPE = (560, 224, 3)
N_FC = 3
FC_DIM = 256
OPTIMISER = "Adam"
LOSS_FUNCTION = "categorical_crossentropy"
BATCH_SIZE = 32


@pytest.fixture(scope="session")
def training_dir():
    return "/"


@pytest.fixture(scope="session")
def validation_dir():
    return "/"


@pytest.fixture(scope="session", params=["ResNet50", "VGG16"])
def model_cls(request):
    return getattr(tf.keras.applications, request.param)


@pytest.fixture(scope="session")
def trainer(training_dir, validation_dir, model_cls):
    return ModelTrainer(
        training_dir=training_dir,
        validation_dir=validation_dir,
        model_cls=model_cls,
        n_fc=N_FC,
        fc_dim=FC_DIM,
        optimiser=OPTIMISER,
        loss_function=LOSS_FUNCTION,
        input_shape=INPUT_SHAPE,
        classes=CLASSES,
    )


@pytest.fixture(scope="session")
def model(trainer):
    return trainer.build_model()


@pytest.fixture(scope="session")
def datagen(trainer):
    return trainer.build_datagen(training_dir)


def test_build_model(model):
    assert model is not None


def test_first_layer_dimensions(model):
    first_layer = model.layers[0].layers[0]
    assert first_layer.input_shape == [
        (None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    ]


def test_last_layer_dimensions(model):
    last_layer = model.layers[-1]
    assert last_layer.output_shape == (None, CLASSES)


def test_preprocess_function(trainer):
    fns = {
        tf.keras.applications.ResNet50: tf.keras.applications.resnet50.preprocess_input,
        tf.keras.applications.VGG16: tf.keras.applications.vgg16.preprocess_input,
    }
    fn = fns[trainer.model_cls]
    assert trainer.preprocess_input_fn() is fn


@pytest.mark.skip
def test_building_datagen(datagen):
    batch, labels = next(datagen)
    assert batch.shape[0] == labels.shape[0] == BATCH_SIZE
    assert batch[0].shape == INPUT_SHAPE
