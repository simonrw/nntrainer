import pytest
from unittest import mock
from nntrainer.app import ModelTrainer
from tensorflow.keras.applications import ResNet50, VGG16


@pytest.fixture
def training_dir():
    return "/"


@pytest.fixture
def validation_dir():
    return "/"


@pytest.fixture(params=[ResNet50, VGG16])
def model_cls(request):
    return request.param


@pytest.fixture
def trainer(training_dir, validation_dir, model_cls):
    return ModelTrainer(
        training_dir=training_dir, validation_dir=validation_dir, model_cls=model_cls
    )


def test_build_model(trainer):
    model = trainer.build_model()
    assert model is not None
