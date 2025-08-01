import torch


# Device configuration - Remove hardcoded device, let DDP handle device assignment
# DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset configuration
DATASET_NAME: str = "bentrevett/multi30k"
TOKENIZER_EN: str = "google-bert/bert-base-uncased"
TOKENIZER_DE: str = "google-bert/bert-base-german-dbmdz-cased"
MAX_SEQ_LEN: int = 256

# Model configuration
BATCH_SIZE: int = 16
NUM_LAYERS: int = 6
D_MODEL: int = 512
NUM_HEADS: int = 8
D_FF: int = 2048

# Training configuration - Match reference implementation closely
DROP_PROB: float = 0.1  # Back to reference value
INIT_LR: float = 1e-5  # Match reference exactly
FACTOR: float = 0.9  # Match reference exactly
ADAM_EPS: float = 5e-9
PATIENCE: int = 10  # Match reference exactly
WARMUP: int = 100  # Match reference exactly (much longer warmup)
EPOCHS_NUM: int = 1_000
CLIP: float = 1.0
WEIGHT_DECAY: float = 5e-4  # Match reference exactly
INF: float = float("inf")



