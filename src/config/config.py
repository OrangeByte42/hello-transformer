import torch


# Device configuration - Remove hardcoded device, let DDP handle device assignment
# DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset configuration
DATASET_NAME: str = "bentrevett/multi30k"
TOKENIZER_EN: str = "bert-base-uncased"
TOKENIZER_DE: str = "bert-base-german-dbmdz-cased"
MAX_SEQ_LEN: int = 128

# Model configuration
BATCH_SIZE: int = 32
D_MODEL: int = 512
NUM_HEADS: int = 8
D_FF: int = 2048
NUM_LAYERS: int = 6
DROP_PROB: float = 0.1

# Training configuration
INIT_LR: float = 1e-5
FACTOR: float = 0.9
ADAM_EPS: float = 5e-9
PATIENCE: int = 10
WARMUP: int = 100
EPOCHS_NUM: int = 1_000
CLIP: float = 1.0
WEIGHT_DECAY: float = 5e-4
INF: float = float("inf")



