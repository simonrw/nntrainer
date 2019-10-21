from dataclasses import dataclass
from typing import NamedTuple, Any, Union, Tuple, List, Optional


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
