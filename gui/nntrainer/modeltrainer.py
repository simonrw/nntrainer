import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import Callback

from .trainingoptions import TrainingOptions
from .architectures import ARCHITECTURES


class TrainerError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class UiUpdateCallback(Callback):
    def __init__(self, update_fn):
        super().__init__()
        self.update_fn = update_fn

        # Training values
        self.loss_history = []

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        loss = logs["loss"]
        self.loss_history.append(loss)
        self.update_fn(self.loss_history)

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class ModelTrainer(object):
    def __init__(self, opts: TrainingOptions):
        self.opts = opts

    def run(self, update_fn=None, finish_callback=None):
        try:
            """Train a model on the datagenerators built"""
            model = self.build_model()

            # Set up the data generators
            training_datagen = self.build_datagen(validation=False)
            training_dataflow = self.build_dataflow(training_datagen, validation=False)

            if self.opts.validation_dir:
                validation_datagen = self.build_datagen(validation=True)
                validation_dataflow = self.build_dataflow(
                    validation_datagen, validation=True
                )
            else:
                validation_dataflow = None

            callbacks = self.build_callbacks(
                include_validation=validation_dataflow is not None,
                update_fn=update_fn,
            )

            args = dict(
                generator=training_dataflow,
                epochs=self.opts.training_epochs,
                verbose=0,
                callbacks=callbacks,
            )
            if validation_dataflow is not None:
                args["validation_data"] = validation_dataflow

            _history = model.fit_generator(**args)
        finally:
            if finish_callback is not None:
                finish_callback()

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

    def build_datagen(self, validation=False):
        if validation:
            gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_input_fn()
            )
        else:
            gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_input_fn(),
                horizontal_flip=self.opts.horizontal_flip,
                vertical_flip=self.opts.vertical_flip,
                rotation_range=self.opts.rotation_angle,
            )

        return gen

    def build_dataflow(self, datagen, validation=False):
        if validation:
            if self.opts.validation_dir is None:
                raise TrainerError("validation directory not set")
            return datagen.flow_from_directory(
                self.opts.validation_dir,
                target_size=self.opts.image_shape,
                batch_size=self.opts.batch_size,
            )
        else:
            if self.opts.training_dir is None:
                raise TrainerError("training directory not set")
            return datagen.flow_from_directory(
                self.opts.training_dir,
                target_size=self.opts.image_shape,
                batch_size=self.opts.batch_size,
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

    def build_callbacks(self, include_validation, update_fn):

        callbacks = [
            self.build_model_checkpoint_callback(include_validation),
            self.build_tensorboard_callback(),
        ]

        if update_fn is not None:
            callbacks.append(UiUpdateCallback(update_fn))

        if self.opts.early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    patience=self.opts.early_stopping.patience,
                    min_delta=self.opts.early_stopping.minimum_delta,
                )
            )

        return callbacks

    def build_model_checkpoint_callback(self, include_validation):
        checkpoint_filename = os.path.join(
            self.opts.output_directory, f"{self.opts.output_name}_checkpoints.h5"
        )
        if include_validation:
            checkpoint_metric = "val_loss"
        else:
            checkpoint_metric = "loss"

        return tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filename, checkpoint_metric, verbose=0, save_best_only=True
        )

    def build_tensorboard_callback(self):
        tensorboard_dir = os.path.join(
            self.opts.output_directory, f"{self.opts.output_name}_tb"
        )
        return tf.keras.callbacks.TensorBoard(
            tensorboard_dir
        )

    @property
    def model_cls(self):
        return ARCHITECTURES[self.opts.architecture]
