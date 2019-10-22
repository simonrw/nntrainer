#!/usr/bin/env python

"""
This script uses the `ModelTrainer` class directly and trains a model
"""

import warnings

warnings.simplefilter("ignore")
import os
from nntrainer.modeltrainer import ModelTrainer
from nntrainer.trainingoptions import TrainingOptions, EarlyStoppingOptions
import tempfile
from contextlib import contextmanager
import shutil
import numpy as np
from PIL import Image


def create_fake_image(path):
    data = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    im = Image.fromarray(data)
    im.save(path)


def create_dir_of_fake_images(root_path, n):
    for i in range(n):
        fname = os.path.join(root_path, f"image{i}.png")
        create_fake_image(fname)


@contextmanager
def tempdir():
    tdir = tempfile.mkdtemp()
    try:
        yield tdir
    finally:
        shutil.rmtree(tdir)


@contextmanager
def images_dir(classes=["a", "b"]):
    with tempdir() as tdir:
        for cls in classes:
            out_dir = os.path.join(tdir, cls)
            os.makedirs(out_dir)
            create_dir_of_fake_images(out_dir, 64)

        yield tdir


if __name__ == "__main__":
    with images_dir() as training_dir:
        with images_dir() as validation_dir:
            with tempdir() as output_dir:
                opts = TrainingOptions(
                    architecture="ResNet50",
                    output_classes=2,
                    num_fc_layers=3,
                    fc_neurones=1024,
                    batch_size=32,
                    training_dir=training_dir,
                    validation_dir=validation_dir,
                    image_shape=(224, 224),
                    training_epochs=32,
                    optimiser="Adam",
                    loss_function="categorical_crossentropy",
                    early_stopping=EarlyStoppingOptions(patience=5, minimum_delta=2),
                    horizontal_flip=True,
                    vertical_flip=True,
                    output_name="test",
                    output_directory=output_dir,
                )

                trainer = ModelTrainer(opts)
                history = trainer.run()
