import pytest
from unittest import mock
from nntrainer.app import ModelTrainer
import tensorflow as tf


@pytest.fixture(scope="session")
def input_shape():
    return (560, 224, 3)


@pytest.fixture(scope="session")
def classes():
    return 15


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
def n_fc():
    return 3


@pytest.fixture(scope="session")
def fc_dim():
    return 256


@pytest.fixture(scope="session")
def trainer(
    training_dir, validation_dir, model_cls, n_fc, fc_dim, input_shape, classes
):
    return ModelTrainer(
        training_dir=training_dir,
        validation_dir=validation_dir,
        model_cls=model_cls,
        n_fc=n_fc,
        fc_dim=fc_dim,
        input_shape=input_shape,
        classes=classes,
    )


@pytest.fixture(scope="session")
def model(trainer):
    return trainer.build_model()


def test_build_model(model):
    assert model is not None


def test_first_layer_dimensions(model, input_shape):
    first_layer = model.layers[0].layers[0]
    assert first_layer.input_shape == [
        (None, input_shape[0], input_shape[1], input_shape[2])
    ]


def test_last_layer_dimensions(model, classes):
    last_layer = model.layers[-1]
    assert last_layer.output_shape == (None, classes)
