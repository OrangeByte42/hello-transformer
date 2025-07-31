import os
import math
import time
import torch

from typing import Any, List, Tuple
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config.config import *
from src.utils.loader import DataLoader4Multi30k
from src.model.transformer import Transformer
from src.utils.bleu import bleu, idx2word
from src.utils.timer import epoch_time
from src.utils.utils import count_parameters


# Training Loop
def train_epoch(model: nn.Module, train_iter: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                clip: float) -> float:
    """Train the model for one epoch"""
    model.train()
    epoch_loss: float = 0.0
    for idx, batch in enumerate(train_iter):
        # Get batch data
        src_X, trg_X = batch
        src_X, trg_X = src_X.to(DEVICE), trg_X.to(DEVICE)
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
        # Echo training progress
        # print(f'step: {round((idx + 1) / len(train_iter) * 100, 2)}%, loss: {loss.item():.4f}')
    return epoch_loss / len(train_iter)

# Evaluation Loop
def evaluate(model: nn.Module, valid_iter: torch.utils.data.DataLoader,
            criterion: nn.Module, trg_tokenizer: Any) -> Tuple[float, float]:
    """Evaluate the model on the validation set"""
    model.eval()
    epoch_loss: float = 0.0
    total_bleu: List[float] = []
    with torch.no_grad():
        for idx, batch in enumerate(valid_iter):
            # Get batch data
            src_X, trg_X = batch
            src_X, trg_X = src_X.to(DEVICE), trg_X.to(DEVICE)
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
    # Calculate average BLEU score
    avg_bleu = sum(total_bleu) / len(total_bleu) if total_bleu else 0.0
    return epoch_loss / len(valid_iter), avg_bleu

# Train
def train(model: Any, train_loader: DataLoader, val_loader: DataLoader,
            criterion: nn.Module, optimizer: Any, scheduler: Any) -> None:
    """Train the transformer model"""
    best_loss: float = INF
    train_losses, test_losses, bleu_scores = [], [], []
    for epoch in range(EPOCHS_NUM):
        start_time: float = time.time()
        train_loss: float = train_epoch(model, train_loader, optimizer, criterion, CLIP)
        valid_loss, bleu_score = evaluate(model, val_loader, criterion, data_loader.tokenizer_en)
        end_time: float = time.time()

        if epoch > WARMUP:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleu_scores.append(bleu_score)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("checkpoints", f"model-loss-{valid_loss:.4f}-bleu-{bleu_score:.4f}.pt"))

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


if __name__ == "__main__":
    # Load Data
    data_loader: DataLoader4Multi30k = DataLoader4Multi30k(
        dataset_name=DATASET_NAME,
        tokenizer_en=TOKENIZER_EN,
        tokenizer_de=TOKENIZER_DE,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )
    train_loader, val_loader, test_loader = data_loader.load()

    # Build Model
    model: Transformer = Transformer(
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
        device=DEVICE
    )

    total_params, trainable_params = count_parameters(model)
    print(f"The transformer model has {total_params:,} ({float(total_params) / float(1_000_000_000):.2f}B) parameters.")
    print(f"The transformer model has {trainable_params:,} ({float(trainable_params) / float(1_000_000_000):.2f}B) trainable parameters.")

    # Initialize Weights
    def init_weights(m: nn.Module) -> None:
        """Initialize weights of the model"""
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_uniform_(m.weight.data)
    model.apply(init_weights).to(DEVICE)

    # Config training
    optimizer: torch.optim.Adam = torch.optim.Adam(params=model.parameters(),
                                                    lr=INIT_LR,
                                                    weight_decay=WEIGHT_DECAY,
                                                    eps=ADAM_EPS)
    scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer=optimizer,
                                                    factor=FACTOR,
                                                    patience=PATIENCE)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=data_loader.tokenizer_en.pad_token_id)

    # Start training
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )






