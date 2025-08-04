import os
import math
import time
import datetime
import torch

from typing import Any, List, Tuple
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config.config import *
from src.utils.dataloader.multi30k_loader import Multi30kDataLoader
from src.model.transformer import Transformer
from src.utils.evaluator.bleu import BLEUSccoreEvaluator
from src.utils.utils import count_parameters, cleanup, epoch_time


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
        output: torch.Tensor = model(src_X, trg_X[:, :-1])
        output_reshape: torch.Tensor = output.contiguous().view(-1, output.shape[-1])
        trg_X_reshape: torch.Tensor = trg_X[:, 1:].contiguous().view(-1)
        # Calculate loss
        loss: torch.Tensor = criterion(output_reshape, trg_X_reshape)
        optimizer.zero_grad()
        # scaler.scale(loss).backward()   # Backpropagation with mixed precision
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()     # Step optimizer
        # Update epoch loss
        epoch_loss += loss.item()
        total_batches += 1

    # Aggregate training loss across all processes
    local_loss = torch.tensor(epoch_loss, device=torch.device(rank))
    local_batch_count = torch.tensor(total_batches, device=torch.device(rank))

    # All-reduce across processes
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_batch_count, op=dist.ReduceOp.SUM)

    # Calculate global average loss
    global_avg_loss = local_loss.item() / local_batch_count.item()

    return global_avg_loss

# # Autoregressive generation function for proper BLEU evaluation
# def generate_autoregressive(model: nn.Module, src_X: torch.Tensor, max_len: int,
#                             sos_token_id: int, eos_token_id: int, device: torch.device) -> torch.Tensor:
#     """Generate sequences autoregressively for proper BLEU evaluation"""
#     batch_size = src_X.shape[0]
#     # Start with SOS token for all sequences in batch
#     generated = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)
#     # Handle both DDP and non-DDP models
#     actual_model = model.module if hasattr(model, 'module') else model
#     # Create source mask
#     src_mask = (src_X != actual_model.src_pad_id).unsqueeze(1).unsqueeze(1)
#     # Encode source sequence once
#     encoder_output = actual_model.encoder(src_X, src_mask)
#     for _ in range(max_len - 1):  # -1 because we already have SOS token
#         # Create target mask for current sequence
#         trg_mask = actual_model._make_trg_mask(generated)
#         # Decode current sequence
#         decoder_output = actual_model.decoder(generated, encoder_output, src_mask, trg_mask)
#         # Get next token probabilities for the last position
#         next_token_logits = decoder_output[:, -1, :]  # (batch_size, vocab_size)
#         next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch_size, 1)
#         # Append next tokens to generated sequence
#         generated = torch.cat([generated, next_tokens], dim=1)
#         # Check if all sequences have generated EOS token (optional optimization)
#         if torch.all(next_tokens.squeeze() == eos_token_id):
#             break
#     return generated

# # Evaluation Loop
# def evaluate(model: nn.Module, valid_iter: torch.utils.data.DataLoader,
#             criterion: nn.Module, trg_tokenizer: Any, rank: int) -> Tuple[float, float]:
#     """Evaluate the model on the validation set"""
#     model.eval()
#     epoch_loss: float = 0.0
#     total_bleu: List[float] = []
#     total_samples: int = 0

#     with torch.no_grad():
#         for idx, batch in enumerate(valid_iter):
#             # Get batch data
#             src_X, trg_X = batch
#             src_X, trg_X = src_X.to(torch.device(rank)), trg_X.to(torch.device(rank))
#             # Infer model output
#             output: torch.Tensor = model(src_X, trg_X[:, :-1])
#             output_reshape: torch.Tensor = output.contiguous().view(-1, output.shape[-1])
#             trg_X_reshape: torch.Tensor = trg_X[:, 1:].contiguous().view(-1)
#             loss: torch.Tensor = criterion(output_reshape, trg_X_reshape)
#             epoch_loss += loss.item()
#             # Generate sequences autoregressively for BLEU evaluation
#             generated_sequences = generate_autoregressive(
#                 model=model,
#                 src_X=src_X,
#                 max_len=trg_X.shape[1],  # Use same max length as target
#                 sos_token_id=trg_tokenizer.sos_token_id,  # Use SOS token from SpacyTokenizer
#                 eos_token_id=trg_tokenizer.eos_token_id,  # Use EOS token from SpacyTokenizer
#                 device=torch.device(rank)
#             )
#             # Calculate BLEU score for each sample in the batch (limit to avoid slowdown)
#             # For efficiency, only calculate BLEU for a subset of validation samples
#             max_bleu_samples = min(4, src_X.shape[0])  # Limit BLEU calculation to 4 samples per batch
#             for i in range(max_bleu_samples):
#                 # Get reference sentence (exclude SOS token)
#                 trg_tokens = trg_X[i, :].cpu().numpy().tolist()  # Skip SOS token
#                 trg_words = trg_tokenizer.decode(trg_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#                 # Get predicted sentence (exclude SOS token)
#                 pred_tokens = generated_sequences[i, :].cpu().numpy().tolist()  # Skip SOS token
#                 pred_words = trg_tokenizer.decode(pred_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#                 if rank == 0 and i == 0:
#                     print(f"Reference: {trg_words}")
#                     print(f"Prediction: {pred_words}")
#                 # Calculate BLEU score if both sentences are non-empty
#                 if trg_words.strip() and pred_words.strip():
#                     bleu_score = bleu([pred_words.split()], [trg_words.split()])
#                     total_bleu.append(bleu_score)
#                 total_samples += 1

#     # Aggregate metrics across all processes
#     # Convert to tensors for all_reduce
#     local_loss = torch.tensor(epoch_loss, device=torch.device(rank))
#     local_bleu_sum = torch.tensor(sum(total_bleu) if total_bleu else 0.0, device=torch.device(rank))
#     local_bleu_count = torch.tensor(len(total_bleu), device=torch.device(rank))
#     local_batch_count = torch.tensor(len(valid_iter), device=torch.device(rank))

#     # All-reduce across processes
#     dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
#     dist.all_reduce(local_bleu_sum, op=dist.ReduceOp.SUM)
#     dist.all_reduce(local_bleu_count, op=dist.ReduceOp.SUM)
#     dist.all_reduce(local_batch_count, op=dist.ReduceOp.SUM)

#     # Calculate global averages
#     global_avg_loss = local_loss.item() / local_batch_count.item()
#     global_avg_bleu = local_bleu_sum.item() / local_bleu_count.item() if local_bleu_count.item() > 0 else 0.0

#     return global_avg_loss, global_avg_bleu

# Autoregressive generation function for proper BLEU evaluation
def generate_autoregressive(model: nn.Module, src_X: torch.Tensor, max_len: int,
                            sos_token_id: int, eos_token_id: int, device: torch.device) -> torch.Tensor:
    """
    Generate sequences autoregressively.

    Args:
        model (nn.Module): The Transformer model (can be DDP wrapped).
        src_X (torch.Tensor): Source input tensor.
        max_len (int): Maximum length for generated sequences.
        sos_token_id (int): Start-of-sequence token ID.
        eos_token_id (int): End-of-sequence token ID.
        device (torch.device): The device to perform generation on.

    Returns:
        torch.Tensor: Generated sequences (token IDs).
    """
    batch_size = src_X.shape[0]
    # Start all sequences with SOS token
    generated = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)
    
    # Access the base model if wrapped by DDP
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Create source mask once
    src_mask = (src_X != actual_model.src_pad_id).unsqueeze(1).unsqueeze(1)
    
    # Encode source sequence once
    encoder_output = actual_model.encoder(src_X, src_mask)
    
    # Track which sequences have finished generation (generated EOS)
    finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):  # -1 because we already have SOS token
        # If all sequences have finished, break early
        if torch.all(finished_sequences):
            break

        # Create target mask for the currently generated sequence
        trg_mask = actual_model._make_trg_mask(generated)
        
        # Decode current sequence to get next token probabilities
        decoder_output = actual_model.decoder(generated, encoder_output, src_mask, trg_mask)
        
        # Get next token with highest probability for the last position
        next_token_logits = decoder_output[:, -1, :]  # (batch_size, vocab_size)
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch_size, 1)
        
        # Append next tokens to generated sequence
        # Ensure that tokens are only appended if the sequence hasn't finished
        # This is implicitly handled by `torch.cat` if we only update `generated`
        # and rely on the `finished_sequences` check to break the loop.
        generated = torch.cat([generated, next_tokens], dim=1)
        
        # Update finished status for sequences that just generated EOS
        finished_sequences = finished_sequences | (next_tokens.squeeze(-1) == eos_token_id)
        
    return generated

# ---
## Evaluation Function
# ---
def evaluate(model: nn.Module, valid_iter: torch.utils.data.DataLoader,
             criterion: nn.Module, trg_tokenizer: Any, rank: int) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set, calculating corpus-level BLEU.

    Args:
        model (nn.Module): The Transformer model (DDP wrapped).
        valid_iter (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        trg_tokenizer (Any): The tokenizer for target language (used for decoding).
        rank (int): The current process rank in DDP.

    Returns:
        Tuple[float, float]: A tuple containing (global average loss, global BLEU score).
    """
    model.eval()
    epoch_loss: float = 0.0
    # To collect all decoded sentences for corpus-level BLEU calculation
    all_references_corpus: List[List[List[str]]] = []  # Expected format for your bleu function
    all_hypothesis_corpus: List[List[str]] = []       # Expected format for your bleu function
    total_batches: int = 0

    with torch.no_grad():
        for idx, batch in enumerate(valid_iter):
            # Get batch data
            src_X, trg_X = batch
            src_X, trg_X = src_X.to(torch.device(rank)), trg_X.to(torch.device(rank))
            
            # 1. Calculate Loss
            # Infer model output for loss calculation (teacher forcing)
            output: torch.Tensor = model(src_X, trg_X[:, :-1])
            output_reshape: torch.Tensor = output.contiguous().view(-1, output.shape[-1])
            trg_X_reshape: torch.Tensor = trg_X[:, 1:].contiguous().view(-1)
            loss: torch.Tensor = criterion(output_reshape, trg_X_reshape)
            epoch_loss += loss.item()
            total_batches += 1

            # 2. Generate Sequences for BLEU Evaluation (Autoregressive)
            generated_sequences = generate_autoregressive(
                model=model,
                src_X=src_X,
                # Set max_len slightly longer than target to allow for full generation
                # Your `bleu` function expects token lists, so `decode` should handle special tokens
                max_len=trg_X.shape[1], # Added buffer for generation length
                sos_token_id=trg_tokenizer.sos_token_id,
                eos_token_id=trg_tokenizer.eos_token_id,
                device=torch.device(rank)
            )
            
            # 3. Decode Tokens to Words and Collect for BLEU Calculation
            for i in range(src_X.shape[0]): # Iterate over samples in the current batch
                # Get raw token IDs
                trg_tokens_raw = trg_X[i, :].cpu().numpy().tolist()
                pred_tokens_raw = generated_sequences[i, :].cpu().numpy().tolist()
                
                # Decode token IDs to word strings. Crucially, `skip_special_tokens=True`
                # ensures that SOS, EOS, PAD tokens are removed from the decoded string.
                # `clean_up_tokenization_spaces=True` helps with readability.
                trg_decoded_str = trg_tokenizer.decode(
                    trg_tokens_raw, skip_special_tokens=True
                )
                pred_decoded_str = trg_tokenizer.decode(
                    pred_tokens_raw, skip_special_tokens=True
                )
                
                # Append to global lists as tokenized words
                # Your `bleu` function expects `List[List[List[str]]]` for references
                # and `List[List[str]]` for hypotheses.
                # Since we have one reference per source, it's `[[['word1', 'word2']]]` for each entry.
                all_references_corpus.append([trg_decoded_str.split()])
                all_hypothesis_corpus.append(pred_decoded_str.split())

                # Print example predictions from rank 0 (first few samples of first batch)
                if rank == 0 and idx == 0 and i < 2: # Limit printing to avoid flood of output
                    print(f"Reference (Sample {i}, Batch {idx}): {trg_decoded_str}")
                    print(f"Prediction (Sample {i}, Batch {idx}): {pred_decoded_str}")

    # ---
    ## Distributed Aggregation of Metrics
    # ---

    # 1. Aggregate Loss Across Processes
    local_loss = torch.tensor(epoch_loss, device=torch.device(rank))
    local_batch_count = torch.tensor(total_batches, device=torch.device(rank))
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_batch_count, op=dist.ReduceOp.SUM)
    global_avg_loss = local_loss.item() / local_batch_count.item() if local_batch_count.item() > 0 else 0.0

    # 2. Gather All Predictions and References to Rank 0 for BLEU Calculation
    gathered_hypothesis = [None for _ in range(dist.get_world_size())]
    gathered_references = [None for _ in range(dist.get_world_size())]

    # `all_gather_object` is used for non-tensor objects like lists of strings
    dist.all_gather_object(gathered_hypothesis, all_hypothesis_corpus)
    dist.all_gather_object(gathered_references, all_references_corpus)

    global_bleu_score = 0.0
    if rank == 0:
        # Flatten the list of lists collected from all ranks
        # `gathered_hypothesis` will be `[[hyp_rank0_sample0, hyp_rank0_sample1], [hyp_rank1_sample0, hyp_rank1_sample1], ...]`
        # We need to flatten it to `[hyp_rank0_sample0, hyp_rank0_sample1, hyp_rank1_sample0, hyp_rank1_sample1, ...]`
        flat_hypothesis_corpus = [item for sublist in gathered_hypothesis for item in sublist]
        flat_references_corpus = [item for sublist in gathered_references for item in sublist]

        # Call your custom corpus-level BLEU function
        # Using smoothing=True is generally recommended, especially in early training,
        # to avoid zero precision for rare N-grams, which would result in 0 BLEU.
        bleu_evaluator: BLEUSccoreEvaluator = BLEUSccoreEvaluator(max_n_gram=4, smoothing=True)
        global_bleu_score = bleu_evaluator.calculate_bleu(
            hypothesis_corpus=flat_hypothesis_corpus,
            references_corpus=flat_references_corpus
        )
        print(f"Rank {rank}: Global BLEU (Custom Corpus-level) for current epoch: {global_bleu_score:.3f}")

    # 3. Broadcast the calculated BLEU score from Rank 0 to all other ranks
    bleu_tensor = torch.tensor(global_bleu_score, device=torch.device(rank))
    dist.broadcast(bleu_tensor, src=0)
    global_bleu_score = bleu_tensor.item() # All ranks now have the correct global BLEU score

    return global_avg_loss, global_bleu_score


# Train
def train(model: Any, train_loader: DataLoader, val_loader: DataLoader,
            criterion: nn.Module, optimizer: Any, scheduler: Any,
            rank: int, trg_tokenizer: Any) -> None:
    """Train the transformer model"""
    best_loss: float = INF
    best_bleu: float = 0.0
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
            if bleu_score > best_bleu:
                best_bleu = bleu_score
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join("checkpoints", f"model-loss-{valid_loss:.4f}-bleu-{bleu_score:.4f}.pt"))

            os.makedirs("out", exist_ok=True)
            with open(os.path.join("out", "train_losses.txt"), "w") as f:
                f.write(str(train_losses))

            with open(os.path.join("out", "test_losses.txt"), "w") as f:
                f.write(str(test_losses))

            with open(os.path.join("out", "bleu_scores.txt"), "w") as f:
                f.write(str(bleu_scores))

            print(f"Epoch: {epoch + 1:0>{len(str(EPOCHS_NUM))}}/{EPOCHS_NUM} | Time: {epoch_mins:0>2}m {epoch_secs:0>2}s :: ", end="  ")
            print(f"Train Loss: {train_loss:<6.3f} | Train PPL: {math.exp(train_loss):<8.3f}", end="  ")
            print(f"Val Loss: {valid_loss:<6.3f} | Val PPL: {math.exp(valid_loss):<8.3f}", end="  ")
            print(f"Val BLEU: {bleu_score:<6.3f}")

        # Synchronize all processes after each epoch
        dist.barrier()


if __name__ == "__main__":
    # Setup Devices and Distributed Environment
    assert torch.cuda.is_available(), "This code requires a GPU with CUDA support."
    assert torch.cuda.device_count() > 0, "This code requires at least two GPUs for distributed training."

    # Initialize distributed training
    world_size: int = int(os.environ["WORLD_SIZE"])     # Total number of processes (GPUs)
    rank: int = int(os.environ["RANK"])     # Rank of the current process
    local_rank: int = int(os.environ["LOCAL_RANK"])     # Local rank of the current process

    if rank == 0:
        print(f"torch cuda available: {torch.cuda.is_available()}")
        print(f"torch cuda device count: {torch.cuda.device_count()}")

    torch.backends.cudnn.benchmark = True   # Enable benchmark mode for faster training
    torch.cuda.set_device(local_rank)     # Set device for current process
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank, init_method='env://',
                            timeout=datetime.timedelta(seconds=300))    # Avoid timeout issues in distributed training
    dist.barrier()      # Synchronize all processes before starting training

    # Load Data
    data_loader: Multi30kDataLoader = Multi30kDataLoader(
        dataset_name=DATASET_NAME,
        de_tokenizer=DE_TOKENIZER,
        en_tokenizer=EN_TOKENIZER,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        ddp=True,  # Enable DDP for DataLoader
        cache_dir=os.path.join(".", "data", "multi30k")
    )
    train_loader, val_loader, test_loader = data_loader.load()

    print(f"Rank {rank} loaded data with {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples, and {len(test_loader.dataset)} test samples.")

    dist.barrier()      # Ensure all processes have loaded the data before proceeding

    # Instantiate Transformer Model and initialize parameters with correct device
    transformer: Transformer = Transformer(
        encoder_vocab_size=data_loader.de_tokenizer.vocab_size,
        decoder_vocab_size=data_loader.en_tokenizer.vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        drop_prob=DROP_PROB,
        src_pad_id=data_loader.de_tokenizer.pad_token_id,
        trg_pad_id=data_loader.en_tokenizer.pad_token_id,
        device=torch.device(rank),  # Use correct device for this rank
    )

    if rank == 0:
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

    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=data_loader.en_tokenizer.pad_token_id).to(torch.device(rank))

    # Start training
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        trg_tokenizer=data_loader.en_tokenizer,
        rank=rank,
    )

    # Cleanup
    dist.barrier()      # Ensure all processes finish training before cleanup
    cleanup()


