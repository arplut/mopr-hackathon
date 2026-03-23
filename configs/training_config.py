"""
Training configuration for MoPR Geospatial Semantic Segmentation.
All hyperparameters defined here as dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = "segformer"
    backbone: str = "nvidia/mit-b2"
    num_classes: int = 9
    in_channels: int = 4  # RGB + DSM
    image_size: int = 512
    pretrained_weights: str = "imagenet"  # or None for random init
    decode_head_hidden_size: int = 256


@dataclass
class DataConfig:
    """Data loading and paths configuration."""
    tile_dir: str = "/content/drive/MyDrive/MoPR/data/tiles"
    mask_dir: str = "/content/drive/MyDrive/MoPR/data/masks"
    train_list: str = "/content/drive/MyDrive/MoPR/data/train.txt"
    val_list: str = "/content/drive/MyDrive/MoPR/data/val.txt"
    num_workers: int = 2
    pin_memory: bool = True
    class_names: List[str] = field(default_factory=lambda: [
        "background", "RCC_roof", "tile_roof", "tin_roof", "thatched_roof",
        "road_pucca", "road_kaccha", "water_body", "vegetation"
    ])
    class_weights: List[float] = field(default_factory=lambda: [
        1.0, 3.0, 3.0, 4.0, 5.0, 2.0, 3.0, 1.5, 1.0
    ])
    ignore_index: int = -1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 100
    batch_size: int = 8
    lr: float = 6e-5
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"  # "cosine" or "linear"
    warmup_epochs: int = 5
    warmup_lr_init: float = 1e-6
    fp16: bool = True  # Mixed precision
    grad_clip: Optional[float] = 1.0
    grad_accumulation_steps: int = 1
    save_dir: str = "/content/drive/MyDrive/MoPR/checkpoints"
    save_every_n_epochs: int = 5
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1
    seed: int = 42

    # Loss function weights
    dice_weight: float = 0.5
    focal_weight: float = 0.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    flip_p: float = 0.5
    rotate90_p: float = 0.3
    rotate_p: float = 0.2
    rotate_limit: int = 20
    brightness_contrast_p: float = 0.3
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    scale_range: Tuple[float, float] = field(default_factory=lambda: (0.8, 1.2))
    shift_p: float = 0.2
    shift_limit: float = 0.1
    elastic_p: float = 0.1
    normalize_mean: Tuple[float, float, float] = field(default_factory=lambda: (0.485, 0.456, 0.406))
    normalize_std: Tuple[float, float, float] = field(default_factory=lambda: (0.229, 0.224, 0.225))
    dsm_normalize_mean: float = 0.0
    dsm_normalize_std: float = 1.0


@dataclass
class InferenceConfig:
    """Inference-time configuration."""
    tta_enabled: bool = False  # Test-Time Augmentation
    tta_transforms: int = 4  # Number of TTA augmentations
    overlap_fraction: float = 0.125  # 12.5% overlap when tiling large images
    batch_size: int = 16
    device: str = "cuda"
    confidence_threshold: Optional[float] = None  # None = no threshold


def get_default_config():
    """
    Returns all default configurations as a dictionary.
    Use this to initialize the full config at runtime.
    """
    return {
        "model": ModelConfig(),
        "data": DataConfig(),
        "training": TrainingConfig(),
        "augmentation": AugmentationConfig(),
        "inference": InferenceConfig(),
    }


# Convenience access
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_AUGMENTATION_CONFIG = AugmentationConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()
