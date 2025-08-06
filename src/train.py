import os
import argparse
import warnings

from src.configs import DatasetConfig, ModelConfig, TrainConfig, SaveConfig
from src.trainer.trainer import Trainer


if __name__ == '__main__':
    """Training script for the Transformer model."""
    # Remove warnings
    warnings.filterwarnings("ignore")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    # Dataset configurations
    parser.add_argument("--de_tokenizer", type=str, default="bert-base-german-dbmdz-cased", help="German tokenizer model name.")
    parser.add_argument("--en_tokenizer", type=str, default="bert-base-uncased", help="English tokenizer model name.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length for input data.")
    # Save configurations
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join(".", "outs", "checkpoints"), help="Directory to save checkpoints.")
    parser.add_argument("--train_trace_dir", type=str, default=os.path.join(".", "outs", "train_traces"), help="Directory to save training traces.")
    parser.add_argument("--save_sample_batch_num", type=int, default=0, help="Batch number to save sample translations from.")
    parser.add_argument("--sample_trace_dir", type=str, default=os.path.join(".", "outs", "sample_traces"), help="Directory to save sample traces.")
    # Training configurations
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup epochs.")
    parser.add_argument("--epochs_num", type=int, default=100, help="Total number of training epochs.")
    parser.add_argument("--ddp", action="store_true", help="Use Distributed Data Parallel (DDP) for training.")

    # Parse arguments once
    args = parser.parse_args()

    # Create configuration instances
    # Dataset configurations
    dataset_config: DatasetConfig = DatasetConfig(
        DE_TOKENIZER=args.de_tokenizer,
        EN_TOKENIZER=args.en_tokenizer,
        BATCH_SIZE=args.batch_size,
        MAX_SEQ_LEN=args.max_seq_len,
    )
    # Model configurations
    model_config: ModelConfig = ModelConfig()
    # Training configurations
    train_config: TrainConfig = TrainConfig(
        WARMUP=args.warmup,
        EPOCHS_NUM=args.epochs_num,
    )
    # Save configurations
    save_config: SaveConfig = SaveConfig(
        CHECKPOINT_DIR=args.checkpoint_dir,
        TRAIN_TRACE_DIR=args.train_trace_dir,
        SAVE_SAMPLE_BATCH_NUM=args.save_sample_batch_num,
        SAMPLE_TRACE_DIR=args.sample_trace_dir,
    )

    # Initialize the Trainer
    ddp: bool = args.ddp
    trainer: Trainer = Trainer(
        dataset_config=dataset_config,
        model_config=model_config,
        train_config=train_config,
        save_config=save_config,
        ddp=ddp,
    )

    # Start training
    trainer.train()

