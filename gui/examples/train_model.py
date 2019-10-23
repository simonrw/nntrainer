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


if __name__ == "__main__":
    training_dir = "exampledata/train"
    validation_dir = "exampledata/val"
    output_dir = "exampledata/out"

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
    history = trainer.run(update_fn=None, finish_callback=lambda: None)
