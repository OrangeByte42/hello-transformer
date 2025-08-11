import os
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
    DATASET_CACHE_DIR: str = os.path.join(".", "data", "datasets", "multi30k")
    TOKENIZER_CACHE_DIR: str = os.path.join(".", "data", "tokenizers")

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    NUM_LAYERS: int = 6
    D_MODEL: int = 512
    NUM_HEADS: int = 8
    D_FF: int = 2048
    DROP_PROB: float = 0.3


@dataclass
class TrainConfig:
    """Configuration for training parameters."""
    WEIGHT_DECAY: float = 1e-4
    EPOCHS_NUM: int = 100
    WARMUP: int = 8
    INIT_LR: float = 5e-5
    ADAM_EPS: float = 1e-8
    PATIENCE: int = 8
    FACTOR: float = 0.8
    CLIP: float = 0.7


@dataclass
class SaveConfig:
    """Configuration for saving."""
    CHECKPOINT_DIR: str = os.path.join(".", "outs", "checkpoints")
    TRAIN_TRACE_DIR: str = os.path.join(".", "outs", "train_trace")
    SAVE_SAMPLE_BATCH_NUM: int = 1
    SAMPLE_TRACE_DIR: str = os.path.join(".", "outs", "sample_trace")

