import pytest
import numpy as np
from unittest import mock
from nntrainer.app import TrainingOptions
from nntrainer.modeltrainer import ModelTrainer
import tensorflow as tf
import tempfile
import os
import shutil
from PIL import Image


CLASSES = 15
INPUT_SHAPE = (560, 224)
N_FC = 3
FC_DIM = 256
OPTIMISER = "Adam"
LOSS_FUNCTION = "categorical_crossentropy"
BATCH_SIZE = 32


def create_fake_image(path):
    data = np.random.randint(0, 255, size=INPUT_SHAPE, dtype=np.uint8)
    im = Image.fromarray(data)
    im.save(path)


def create_dir_of_fake_images(root_path, n):
    for i in range(n):
        fname = os.path.join(root_path, f"image{i}.png")
        create_fake_image(fname)


@pytest.fixture(scope="session")
def training_dir():
    tdir = tempfile.mkdtemp()

    classes = ["a", "b"]
    try:
        for cls in classes:
            out_dir = os.path.join(tdir, cls)
            os.makedirs(out_dir)
            create_dir_of_fake_images(out_dir, 64)

        yield tdir
    finally:
        shutil.rmtree(tdir)


@pytest.fixture(scope="session", params=["ResNet50", "VGG16"])
def opts(request, training_dir, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("opts")

    opts = TrainingOptions()
    opts.architecture = request.param
    opts.image_shape = INPUT_SHAPE
    opts.num_fc_layers = N_FC
    opts.fc_neurones = FC_DIM
    opts.optimiser = OPTIMISER
    opts.loss_function = LOSS_FUNCTION
    opts.output_classes = CLASSES
    opts.horizontal_flip = True
    opts.vertical_flip = True
    opts.rotation_angle = 20
    opts.training_dir = training_dir
    opts.batch_size = BATCH_SIZE
    opts.output_directory = output_dir
    opts.output_name = "modeltesting"
    return opts


@pytest.fixture(scope="session")
def trainer(opts):
    return ModelTrainer(opts)


@pytest.fixture(scope="session")
def model(trainer):
    return trainer.build_model()


@pytest.fixture(scope="session")
def training_datagen(trainer):
    return trainer.build_datagen(validation=False)


@pytest.fixture(scope="session")
def validation_datagen(trainer):
    return trainer.build_datagen(validation=True)


@pytest.fixture(scope="session")
def training_dataflow(training_datagen, trainer):
    return trainer.build_dataflow(training_datagen, validation=False)


def test_build_model(model):
    assert model is not None


def test_first_layer_dimensions(model):
    first_layer = model.layers[0].layers[0]
    assert first_layer.input_shape == [(None, INPUT_SHAPE[0], INPUT_SHAPE[1], 3)]


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


def test_building_datagen(training_datagen):
    assert training_datagen.horizontal_flip is True
    assert training_datagen.vertical_flip is True
    assert training_datagen.rotation_range == 20


def test_building_dataflow(training_dataflow):
    batch, labels = next(training_dataflow)
    assert batch.shape[0] == labels.shape[0] == BATCH_SIZE
    assert batch[0].shape == (INPUT_SHAPE[0], INPUT_SHAPE[1], 3)


def test_run(trainer):
    with mock.patch.object(trainer, "build_model") as build_model:
        trainer.run()

    build_model.return_value.fit_generator.assert_called_once()
