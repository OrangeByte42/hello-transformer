import os
import math
import time
import torch

from typing import Any, List, Tuple
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config.config import *
from src.utils.loader import DataLoader4Multi30k
from src.model.transformer import Transformer
from src.utils.bleu import bleu, idx2word
from src.utils.timer import epoch_time
from src.utils.utils import count_parameters, cleanup


# Training Loop
def train_epoch(model: nn.Module, train_iter: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                clip: float, rank: int) -> float:
    """Train the model for one epoch"""
    model.train()
    epoch_loss: float = 0.0
    total_batches: int = 0

    for idx, batch in enumerate(train_iter):
        # Get batch data
        src_X, trg_X = batch
        src_X, trg_X = src_X.to(torch.device(rank)), trg_X.to(torch.device(rank))
        # Infer model output
        optimizer.zero_grad()
        output: torch.Tensor = model(src_X, trg_X[:, :-1])
        output_reshape: torch.Tensor = output.contiguous().view(-1, output.shape[-1])
        trg_X_reshape: torch.Tensor = trg_X[:, 1:].contiguous().view(-1)
        # Calculate loss
        loss: torch.Tensor = criterion(output_reshape, trg_X_reshape)
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        # Update epoch loss
        epoch_loss += loss.item()
        total_batches += 1
        # Echo training progress
        # print(f'step: {round((idx + 1) / len(train_iter) * 100, 2)}%, loss: {loss.item():.4f}')

    # Aggregate training loss across all processes
    local_loss = torch.tensor(epoch_loss, device=torch.device(rank))
    local_batch_count = torch.tensor(total_batches, device=torch.device(rank))

    # All-reduce across processes
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_batch_count, op=dist.ReduceOp.SUM)

    # Calculate global average loss
    global_avg_loss = local_loss.item() / local_batch_count.item()

    return global_avg_loss

# Evaluation Loop
def evaluate(model: nn.Module, valid_iter: torch.utils.data.DataLoader,
            criterion: nn.Module, trg_tokenizer: Any, rank: int) -> Tuple[float, float]:
    """Evaluate the model on the validation set"""
    model.eval()
    epoch_loss: float = 0.0
    total_bleu: List[float] = []
    total_samples: int = 0

    with torch.no_grad():
        for idx, batch in enumerate(valid_iter):
            # Get batch data
            src_X, trg_X = batch
            src_X, trg_X = src_X.to(torch.device(rank)), trg_X.to(torch.device(rank))
            # Infer model output
            output: torch.Tensor = model(src_X, trg_X[:, :-1])
            output_reshape: torch.Tensor = output.contiguous().view(-1, output.shape[-1])
            trg_X_reshape: torch.Tensor = trg_X[:, 1:].contiguous().view(-1)
            # Calculate loss
            loss: torch.Tensor = criterion(output_reshape, trg_X_reshape)
            epoch_loss += loss.item()
            # Calculate BLEU score for each sample in the batch
            for i in range(src_X.shape[0]):
                # Get reference sentence (exclude padding tokens)
                trg_tokens = trg_X[i].cpu().numpy().tolist()
                trg_words = idx2word(trg_tokens, trg_tokenizer)
                # Get predicted sentence
                output_tokens = output[i].max(dim=1)[1].cpu().numpy().tolist()
                output_words = idx2word(output_tokens, trg_tokenizer)
                # Calculate BLEU score if both sentences are non-empty
                if trg_words.strip() and output_words.strip():
                    bleu_score = bleu([output_words.split()], [trg_words.split()])
                    total_bleu.append(bleu_score)
                total_samples += 1

    # Aggregate metrics across all processes
    # Convert to tensors for all_reduce
    local_loss = torch.tensor(epoch_loss, device=torch.device(rank))
    local_bleu_sum = torch.tensor(sum(total_bleu) if total_bleu else 0.0, device=torch.device(rank))
    local_bleu_count = torch.tensor(len(total_bleu), device=torch.device(rank))
    local_batch_count = torch.tensor(len(valid_iter), device=torch.device(rank))

    # All-reduce across processes
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_bleu_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_bleu_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_batch_count, op=dist.ReduceOp.SUM)

    # Calculate global averages
    global_avg_loss = local_loss.item() / local_batch_count.item()
    global_avg_bleu = local_bleu_sum.item() / local_bleu_count.item() if local_bleu_count.item() > 0 else 0.0

    return global_avg_loss, global_avg_bleu

# Train
def train(model: Any, train_loader: DataLoader, val_loader: DataLoader,
            criterion: nn.Module, optimizer: Any, scheduler: Any,
            rank: int, trg_tokenizer: Any) -> None:
    """Train the transformer model"""
    best_loss: float = INF
    train_losses, test_losses, bleu_scores = [], [], []
    for epoch in range(EPOCHS_NUM):
        start_time: float = time.time()
        train_loader.sampler.set_epoch(epoch)
        train_loss: float = train_epoch(model, train_loader, optimizer, criterion, CLIP, rank)
        valid_loss, bleu_score = evaluate(model, val_loader, criterion, trg_tokenizer, rank)
        end_time: float = time.time()

        if epoch > WARMUP:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleu_scores.append(bleu_score)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Only save model and write files on rank 0 to avoid conflicts
        if rank == 0:
            if valid_loss < best_loss:
                best_loss = valid_loss
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join("checkpoints", f"model-loss-{valid_loss:.4f}-bleu-{bleu_score:.4f}.pt"))

            os.makedirs("out", exist_ok=True)
            with open(os.path.join("out", "train_losses.txt"), "w") as f:
                f.write(str(train_losses))

            with open(os.path.join("out", "test_losses.txt"), "w") as f:
                f.write(str(test_losses))

            with open(os.path.join("out", "bleu_scores.txt"), "w") as f:
                f.write(str(bleu_scores))

            print(f"Epoch: {epoch + 1:0>{len(str(EPOCHS_NUM))}}/{EPOCHS_NUM} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
            print(f"\tVal Loss: {valid_loss:.3f} | Val PPL: {math.exp(valid_loss):7.3f}")
            print(f"\tVal BLEU: {bleu_score:.3f}")

        # Synchronize all processes after each epoch
        dist.barrier()


if __name__ == "__main__":
    # Setup Devices and Distributed Environment
    assert torch.cuda.is_available(), "This code requires a GPU with CUDA support."
    assert torch.cuda.device_count() > 0, "This code requires at least two GPUs for distributed training."

    print(f"torch cuda available: {torch.cuda.is_available()}")
    print(f"torch cuda device count: {torch.cuda.device_count()}")

    # Initialize distributed training
    world_size: int = int(os.environ["WORLD_SIZE"])     # Total number of processes (GPUs)
    rank: int = int(os.environ["RANK"])     # Rank of the current process

    torch.cuda.set_device(rank)     # Set device for current process
    torch.backends.cudnn.benchmark = True   # Enable benchmark mode for faster training
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank, init_method='env://')
    dist.barrier()      # Synchronize all processes before starting training

    # Load Data
    data_loader: DataLoader4Multi30k = DataLoader4Multi30k(
        dataset_name=DATASET_NAME,
        tokenizer_en=TOKENIZER_EN,
        tokenizer_de=TOKENIZER_DE,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )
    train_loader, val_loader, test_loader = data_loader.load()

    print(f"Rank {rank} loaded data with {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples, and {len(test_loader.dataset)} test samples.")

    dist.barrier()      # Ensure all processes have loaded the data before proceeding

    # Instantiate Transformer Model and initialize parameters with correct device
    transformer: Transformer = Transformer(
        encoder_vocab_size=data_loader.tokenizer_de.vocab_size,
        decoder_vocab_size=data_loader.tokenizer_en.vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        drop_prob=DROP_PROB,
        src_pad_idx=data_loader.tokenizer_de.pad_token_id,
        trg_pad_idx=data_loader.tokenizer_en.pad_token_id,
        device=torch.device(rank),  # Use correct device for this rank
    )

    total_params, trainable_params = count_parameters(transformer)
    print(f"The transformer model has {total_params:,} ({float(total_params) / float(1_000_000_000):.2f}B) parameters.")
    print(f"The transformer model has {trainable_params:,} ({float(trainable_params) / float(1_000_000_000):.2f}B) trainable parameters.")

    # Initialize Weights
    def init_weights(m: nn.Module) -> None:
        """Initialize weights of the model"""
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_uniform_(m.weight.data)

    transformer.apply(init_weights)

    # Move model to GPU and wrap in DDP - remove redundant .to() call
    model: Any = DDP(transformer, device_ids=[rank], output_device=rank)

    # Config training
    optimizer: torch.optim.Adam = torch.optim.Adam(params=model.parameters(),
                                                    lr=INIT_LR,
                                                    weight_decay=WEIGHT_DECAY,
                                                    eps=ADAM_EPS)

    scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer=optimizer,
                                                    factor=FACTOR,
                                                    patience=PATIENCE)

    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=data_loader.tokenizer_en.pad_token_id).to(torch.device(rank))

    # Start training
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        trg_tokenizer=data_loader.tokenizer_en,
        rank=rank,
    )

    # Cleanup
    dist.barrier()      # Ensure all processes finish training before cleanup
    cleanup()


