import os
import torch
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for dataset parameters."""
    DATASET_NAME: str = "bentrevett/multi30k"
    DE_TOKENIZER: str = "bert-base-german-dbmdz-cased"
    EN_TOKENIZER: str = "bert-base-uncased"
    # DE_TOKENIZER: str = "de_core_news_sm"
    # EN_TOKENIZER: str = "en_core_web_sm"
    BATCH_SIZE: int = 16
    MAX_SEQ_LEN: int = 256
    CACHE_DIR: str = os.path.join(".", "data", "multi30k")


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    NUM_LAYERS: int = 6
    D_MODEL: int = 512
    NUM_HEADS: int = 8
    D_FF: int = 2048
    DROP_PROB: float = 0.2


@dataclass
class TrainConfig:
    """Configuration for training parameters."""
    INIT_LR: float = 1e-4
    FACTOR: float = 0.9
    ADAM_EPS: float = 5e-9
    PATIENCE: int = 10
    WARMUP: int = 20
    CLIP: float = 1.0
    WEIGHT_DECAY: float = 5e-4
    EPOCHS_NUM: int = 100

