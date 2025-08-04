import torch

# Dataset configuration
DATASET_NAME: str = "bentrevett/multi30k"
# EN_TOKENIZER: str = "en_core_web_sm" # "bert-base-uncased" # "en_core_web_sm"
# DE_TOKENIZER: str = "de_core_news_sm" # "bert-base-german-dbmdz-cased" # "de_core_news_sm"
EN_TOKENIZER: str = "bert-base-uncased" # "en_core_web_sm"
DE_TOKENIZER: str = "bert-base-german-dbmdz-cased" # "de_core_news_sm"
BATCH_SIZE: int = 16
MAX_SEQ_LEN: int = 256


# Model architecture configuration
NUM_LAYERS: int = 6
D_MODEL: int = 512
NUM_HEADS: int = 8
D_FF: int = 2048

DROP_PROB: float = 0.2  # Reference uses 0.1, not 0.3


# Training configuration - Match reference implementation exactly
INIT_LR: float = 5e-4  # Reference uses 1e-4, not 1e-5
FACTOR: float = 0.9  # Match reference exactly
ADAM_EPS: float = 5e-9
PATIENCE: int = 5  # Match reference exactly
WARMUP: int = 20  # Reference uses 100 epochs warmup
EPOCHS_NUM: int = 100  # Reference uses more epochs
CLIP: float = 1.0
WEIGHT_DECAY: float = 5e-4  # Reference uses 5e-4
INF: float = float("inf")







