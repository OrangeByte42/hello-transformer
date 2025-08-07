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
    DROP_PROB: float = 0.1


@dataclass
class TrainConfig:
    """Configuration for training parameters."""
    INIT_LR: float = 1e-5
    FACTOR: float = 0.9
    ADAM_EPS: float = 5e-9
    PATIENCE: int = 10
    CLIP: float = 1.0
    WEIGHT_DECAY: float = 5e-4
    WARMUP: int = 20
    EPOCHS_NUM: int = 100


@dataclass
class SaveConfig:
    """Configuration for saving."""
    CHECKPOINT_DIR: str = os.path.join(".", "outs", "checkpoints")
    TRAIN_TRACE_DIR: str = os.path.join(".", "outs", "train_traces")
    SAVE_SAMPLE_BATCH_NUM: int = 1
    SAMPLE_TRACE_DIR: str = os.path.join(".", "outs", "sample_traces")

