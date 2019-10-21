import pytest
from unittest import mock
from nntrainer.app import TrainingOptions
from nntrainer.modeltrainer import ModelTrainer
import tensorflow as tf


CLASSES = 15
INPUT_SHAPE = (560, 224)
N_FC = 3
FC_DIM = 256
OPTIMISER = "Adam"
LOSS_FUNCTION = "categorical_crossentropy"
BATCH_SIZE = 32


@pytest.fixture(scope="session", params=["ResNet50", "VGG16"])
def opts(request):
    opts = TrainingOptions()
    opts.architecture = request.param
    opts.image_shape = INPUT_SHAPE
    opts.num_fc_layers = N_FC
    opts.fc_neurones = FC_DIM
    opts.optimiser = OPTIMISER
    opts.loss_function = LOSS_FUNCTION
    opts.output_classes = CLASSES
    return opts


@pytest.fixture(scope="session")
def trainer(opts):
    return ModelTrainer(opts)


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
        (None, INPUT_SHAPE[0], INPUT_SHAPE[1], 3)
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
