import os
import time
import datetime
import math
import torch
from torch import nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, Optional, Tuple, List, Dict, Union

from src.configs import DatasetConfig, ModelConfig, TrainConfig, SaveConfig
from src.data.dataloader.multi30k_loader import Multi30kDataLoader
from src.data.tokenizer.tokenizer import Tokenizer
from src.models.transformer import Transformer
from src.generator.autoregressive_transformer import AutoregressiveTransformer
from src.data.evaluator.bleu import BLEUScoreEvaluator
from src.utils.utils import save_obj_by_pickle, cleanup, count_parameters, epoch_time


class Trainer:
    """Trainer class for managing training, validation and evaluation of a model."""

    def __init__(self: Any,
                    dataset_config: DatasetConfig,
                    model_config: ModelConfig,
                    train_config: TrainConfig,
                    save_config: SaveConfig,
                    ddp: bool) -> None:
        """Initialize the Trainer with configurations.
        @param dataset_config: Configuration for dataset parameters.
        @param model_config: Configuration for model architecture.
        @param train_config: Configuration for training parameters.
        @param save_config: Configuration for saving checkpoints and logs.
        @param ddp: Flag to indicate if Distributed Data Parallel (DDP) training is enabled.
        """
        # Configurations
        self.dataset_config: DatasetConfig = dataset_config
        self.model_config: ModelConfig = model_config
        self.train_config: TrainConfig = train_config
        self.save_config: SaveConfig = save_config

        # DataLoader & Model
        self.data_loader: Optional[Multi30kDataLoader] = None
        self.model: Optional[Transformer] = None

        # Training components
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[ReduceLROnPlateau] = None
        self.criterion: Optional[nn.CrossEntropyLoss] = None

        self.ddp: bool = ddp

        # Training trace record
        self.best_loss: float = float('inf')
        self.best_bleu: float = 0.0
        self.train_losses: List[float] = list()
        self.valid_losses: List[float] = list()
        self.valid_bleu_scores: List[float] = list()

        self.sample_batch_num: int = self.save_config.SAVE_SAMPLE_BATCH_NUM
        # List of {"ref": str, "hyp": List[str]}, x[i]["hyp"][j] is the i-th sample's j-th hypothesis after j-th training epoch
        self.sample_trace: List[Dict[str, Union[str, List[str]]]] = list()

    def _init_dataloader(self: Any) -> Multi30kDataLoader:
        """Initialize the data loader for the Multi30k dataset.
        @return: An instance of Multi30kDataLoader.
        """
        # Instantiate the Multi30kDataLoader with the dataset configuration
        data_loader: Multi30kDataLoader = Multi30kDataLoader(
            dataset_name=self.dataset_config.DATASET_NAME,
            de_tokenizer=self.dataset_config.DE_TOKENIZER,
            en_tokenizer=self.dataset_config.EN_TOKENIZER,
            max_seq_len=self.dataset_config.MAX_SEQ_LEN,
            batch_size=self.dataset_config.BATCH_SIZE,
            ddp=self.ddp,
            dataset_cache_dir=self.dataset_config.DATASET_CACHE_DIR,
            tokenizer_cache_dir=self.dataset_config.TOKENIZER_CACHE_DIR,
        )

        # Set the data loader to the instance variable
        self.data_loader = data_loader

        # Echo the dataset information
        if not self.ddp or (self.ddp and dist.is_initialized() and dist.get_rank() == 0):
            print(f"[Initializing DataLoader]:")
            print(f"\t- Initialized DataLoader for {self.dataset_config.DATASET_NAME} dataset.")
            print(f"\t- Using {self.dataset_config.DE_TOKENIZER} for German and {self.dataset_config.EN_TOKENIZER} for English.")
            print(f"\t- Batch size: {self.dataset_config.BATCH_SIZE}, Max sequence length: {self.dataset_config.MAX_SEQ_LEN}")
            print(f"\t- En tokenizer vocabulary size: {data_loader.en_tokenizer.vocab_size}")
            print(f"\t- De tokenizer vocabulary size: {data_loader.de_tokenizer.vocab_size}")
            print(f"\tDataLoader initialized successfully.", end="\n\n")

        # Return the initialized data loader
        return data_loader

    def _init_model(self: Any, device: torch.device) -> Transformer:
        """Initialize the Transformer model with the given configurations.
        @param device: The device (CPU or GPU) to run the model on.
        @return: An instance of Transformer model.
        """
        # Ensure the data loader is initialized
        assert self.data_loader is not None, "Data loader must be initialized before model."

        # Instantiate the Transformer model with the model configuration
        transformer: Transformer = Transformer(
            encoder_vocab_size=self.data_loader.de_tokenizer.vocab_size,
            decoder_vocab_size=self.data_loader.en_tokenizer.vocab_size,
            max_seq_len=self.dataset_config.MAX_SEQ_LEN,
            src_pad_id=self.data_loader.de_tokenizer.pad_token_id,
            trg_pad_id=self.data_loader.en_tokenizer.pad_token_id,
            num_layers=self.model_config.NUM_LAYERS,
            d_model=self.model_config.D_MODEL,
            num_heads=self.model_config.NUM_HEADS,
            d_ff=self.model_config.D_FF,
            drop_prob= self.model_config.DROP_PROB,
            device=device,
        )

        # Intialize the model weights
        transformer.init_weights()

        # Set the model to the instance variable
        self.model = transformer

        # Echo the model parameters
        if not self.ddp or (self.ddp and dist.is_initialized() and dist.get_rank() == 0):
            total_params, trainable_params = count_parameters(transformer, self.ddp)
            print(f"[Instantiating Model]:")
            print(f"\t- Model Parameters:     {total_params:,} ({total_params / 1e9:.4f} B)")
            print(f"\t- Trainable Parameters: {trainable_params:,} ({trainable_params / 1e9:.4f} B)")
            print(f"\tInstantiated Transformer model successfully.", end="\n\n")

        # Return the initialized model
        return transformer

    def _init_training_components(self: Any, device: torch.device) -> None:
        """Intialize optimizer, scheduler and criterion for training.
        @param device: The device (CPU or GPU) to run the training components on.
        """
        # Ensure the model is initialized
        assert self.model is not None, "Model must be initialized before training components."

        # Create the optimizer
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.train_config.INIT_LR,
            weight_decay=self.train_config.WEIGHT_DECAY,
            eps=self.train_config.ADAM_EPS,
        )

        # Create the learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=self.train_config.FACTOR,
            patience=self.train_config.PATIENCE,
        )

        # Create the loss criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.data_loader.en_tokenizer.pad_token_id).to(device)

    def _epoch_train(self: Any, model: Transformer, train_iter: torch.utils.data.DataLoader,
                        optimizer: torch.optim.Optimizer, criterion: nn.CrossEntropyLoss,
                        clip: float, device: torch.device, ddp: bool) -> float:
        """Train the model for one epoch.
        @param model: The Transformer model to train.
        @param train_iter: DataLoader for the training data.
        @param optimizer: Optimizer for updating model parameters.
        @param criterion: Loss function for training.
        @param clip: Gradient clipping value.
        @param device: The device (CPU or GPU) to run the training on.
        @param ddp: Flag to indicate if Distributed Data Parallel (DDP) training is enabled.
        @return: Average training loss for the epoch.
        """
        # Set the model to training mode and create necessary variables
        model.train()

        epoch_loss: float = 0.0
        total_batches: int = 0

        # Iterate over the training data loader
        for batch in train_iter:
            # Get batch data and move to device
            src_X, trg_X = batch
            src_X, trg_X = src_X.to(device), trg_X.to(device)

            # Inference the model output
            # Teacher forcing: use trg_X[:, :-1] as input and trg_X[:, 1:] as target
            output: torch.Tensor = model(src_X, trg_X[:, :-1])
            reshaped_output: torch.Tensor = output.contiguous().view(-1, output.size(-1))
            reshaped_trg_X: torch.Tensor = trg_X[:, 1:].contiguous().view(-1)

            # Calculate the loss
            loss: torch.Tensor = criterion(reshaped_output, reshaped_trg_X)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # Update epoch loss and total batches
            epoch_loss += loss.item()
            total_batches += 1

        # Convert training loss and batch count to tensor
        local_loss: torch.Tensor = torch.tensor(epoch_loss, device=device)
        local_batch_count: torch.Tensor = torch.tensor(total_batches, device=device)

        # If using DDP, reduce the loss and batch count across all processes
        if ddp == True:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_batch_count, op=dist.ReduceOp.SUM)

        # Calculate the average loss across all processes
        global_avg_loss: float = local_loss.item() / local_batch_count.item()

        # Return the average loss for the epoch
        return global_avg_loss

    def _epoch_validate(self: Any, model: Transformer, generator: AutoregressiveTransformer,
                        valid_iter: torch.utils.data.DataLoader, criterion: nn.CrossEntropyLoss,
                        max_seq_len: int, trg_tokenizer: Tokenizer, evaluator: BLEUScoreEvaluator,
                        device: torch.device, ddp: bool) -> Tuple[float, float]:
        """Validate the model for one epoch.
        @param model: The Transformer model to validate.
        @param generator: Autoregressive generator for generating hypotheses.
        @param valid_iter: DataLoader for the validation data.
        @param criterion: Loss function for validation.
        @param max_seq_len: Maximum sequence length for generation.
        @param trg_tokenizer: Tokenizer for target language.
        @param evaluator: BLEU score evaluator for validation.
        @param device: The device (CPU or GPU) to run the validation on.
        @param ddp: Flag to indicate if Distributed Data Parallel (DDP) validation is enabled.
        @return: Tuple of (average validation loss, BLEU score).
        """
        # Set the model to evaluation mode and create necessary variables
        model.eval()

        epoch_loss: float = 0.0
        total_batches: int = 0

        all_ref_corpus: List[List[List[str]]] = list()  # shape: [batch_size, num_refs, seq_len]
        all_hyp_corpus: List[List[str]] = list()  # shape: [batch_size, seq_len]

        saved_sample_count: int = 0

        # Iterate over the validation data loader
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_iter):
                # Part 01. Get Loss
                # Get batch data and move to device
                src_X, trg_X = batch
                src_X, trg_X = src_X.to(device), trg_X.to(device)

                # Inference the model output (teacher forcing)
                output: torch.Tensor = model(src_X, trg_X[:, :-1])
                reshaped_output: torch.Tensor = output.contiguous().view(-1, output.shape[-1])
                reshaped_trg_X: torch.Tensor = trg_X[:, 1:].contiguous().view(-1)

                # Calculate the loss and update epoch loss and total batches
                loss: torch.Tensor = criterion(reshaped_output, reshaped_trg_X)
                epoch_loss += loss.item()
                total_batches += 1

                # Part 02. Generate Hypotheses for BLEU calculation
                # Generate hypotheses for BLEU calculation (Auto regressive)
                generated_seqs: torch.Tensor = generator.generate(src_X, max_seq_len)

                # Decode tokens to words and collect for BLEU calculation
                batch_size: int = src_X.shape[0]
                for sample_idx in range(batch_size):   # Iterate over samples in the current batch
                    # Get the current reference and hypothesis sequences
                    trg_token_ids: List[int] = trg_X[sample_idx, :].cpu().numpy().tolist()
                    pred_token_ids: List[int] = generated_seqs[sample_idx, :].cpu().numpy().tolist()

                    # Decode the reference and hypothesis sequences to words
                    all_ref_corpus.append([trg_tokenizer.convert_ids_to_tokens(trg_token_ids, skip_special_tokens=True)])
                    all_hyp_corpus.append(trg_tokenizer.convert_ids_to_tokens(pred_token_ids, skip_special_tokens=True))

                    # Record sample traces for analysis
                    if batch_idx < self.sample_batch_num:
                        # Decode the reference and hypothesis sequences to text
                        trg_text: str = trg_tokenizer.decode(trg_token_ids, skip_special_tokens=True)
                        pred_text: str = trg_tokenizer.decode(pred_token_ids, skip_special_tokens=True)

                        # Save the sample trace
                        if len(self.sample_trace) <= saved_sample_count:
                            self.sample_trace.append({
                                "ref": trg_text,
                                "hyp": [pred_text],
                            })
                        else:
                            assert self.sample_trace[saved_sample_count]["ref"] == trg_text, "Sample trace reference mismatch"
                            self.sample_trace[saved_sample_count]["hyp"].append(pred_text)
                        saved_sample_count += 1

        # Part 01. Calculate the average validation loss across all processes
        # Convert validation loss and batch count to tensor
        local_loss: torch.Tensor = torch.tensor(epoch_loss, device=device)
        local_batch_count: torch.Tensor = torch.tensor(total_batches, device=device)

        # If using DDP, reduce the loss and batch count across all processes
        if ddp == True:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_batch_count, op=dist.ReduceOp.SUM)

        # Calculate the average loss across all processes
        global_avg_loss: float = local_loss.item() / local_batch_count.item() if local_batch_count.item() > 0 else 0.0

        # Part 02. Calculate the BLEU score across all processes
        # If using DDP, gather all reference and hypothesis corpora across all processes
        if ddp == True:
            gathered_ref_corpus: List[List[List[List[str]]]] = [None] * dist.get_world_size()
            gathered_hyp_corpus: List[List[List[str]]] = [None] * dist.get_world_size()

            dist.all_gather_object(gathered_ref_corpus, all_ref_corpus)
            dist.all_gather_object(gathered_hyp_corpus, all_hyp_corpus)
        else:   # If not using DDP, just use the local corpora
            gathered_ref_corpus: List[List[List[List[str]]]] = [all_ref_corpus]
            gathered_hyp_corpus: List[List[List[str]]] = [all_hyp_corpus]

        # Calculate the global BLEU score
        global_bleu_score: float = 0.0
        if not ddp or (ddp and dist.is_initialized() and dist.get_rank() == 0):
            # Flatten the gathered corpora
            flat_ref_corpus: List[List[List[str]]] = [item for sublist in gathered_ref_corpus for item in sublist]
            flat_hyp_corpus: List[List[str]] = [item for sublist in gathered_hyp_corpus for item in sublist]

            # Calculate the BLEU score using the flattened corpora
            global_bleu_score = evaluator.evaluate(
                hypothesis_corpus=flat_hyp_corpus,
                references_corpus=flat_ref_corpus,
            )

        # If using DDP, broadcast the BLEU score to all processes
        if ddp == True:
            bleu_tensor: torch.Tensor = torch.tensor(global_bleu_score, device=device)
            dist.broadcast(bleu_tensor, src=0)
            global_bleu_score = bleu_tensor.item()

        # Return the average validation loss and BLEU score
        return global_avg_loss, global_bleu_score

    def _train(self: Any, model: Transformer, trg_tokenizer: Tokenizer,
                train_iter: torch.utils.data.DataLoader, valid_iter: torch.utils.data.DataLoader,
                optimizer: Any, scheduler: Any, criterion: nn.CrossEntropyLoss,
                device: torch.device, ddp: bool) -> None:
        """Train the model for multiple epochs with validation.
        @param model: The Transformer model to train.
        @param trg_tokenizer: Tokenizer for target language.
        @param train_iter: DataLoader for the training data.
        @param valid_iter: DataLoader for the validation data.
        @param optimizer: Optimizer for updating model parameters.
        @param scheduler: Learning rate scheduler for adjusting learning rate.
        @param criterion: Loss function for training.
        @param device: The device (CPU or GPU) to run the training on.
        @param ddp: Flag to indicate if Distributed Data Parallel (DDP) training is enabled.
        """
        # Necessary training configurations
        max_seq_len: int = self.dataset_config.MAX_SEQ_LEN
        epochs_num: int = self.train_config.EPOCHS_NUM
        clip: float = self.train_config.CLIP
        warmup: int = self.train_config.WARMUP

        # Get actual model (unwrap DDP if necessary)
        actual_model: Optional[Transformer] = None
        if ddp == True and isinstance(model, DDP):
            actual_model = model.module
        else:
            actual_model = model

        # Initialize the autoregressive generator for validation
        generator: AutoregressiveTransformer = AutoregressiveTransformer(actual_model, trg_tokenizer, device)

        # Initialize the BLEU evaluator
        evaluator: BLEUScoreEvaluator = BLEUScoreEvaluator(max_n_gram=4, smoothing=True)

        # Training loop
        for epoch in range(epochs_num):
            # Train for one epoch
            start_time: float = time.time()
            if ddp == True: train_iter.sampler.set_epoch(epoch)
            train_loss: float = self._epoch_train(model, train_iter, optimizer, criterion, clip, device, ddp)
            valid_loss, valid_bleu = self._epoch_validate(model, generator, valid_iter, criterion, max_seq_len,
                                                            trg_tokenizer, evaluator, device, ddp)
            end_time: float = time.time()

            # Step the scheduler
            if epoch > warmup: scheduler.step(valid_loss)

            # Echo the training and validation results
            if ddp == False or (ddp == True and dist.get_rank() == 0):
                elapsed_mins, elapsed_secs = epoch_time(start_time, end_time)
                print(f"[Training Trace] >>> "
                        f"Epoch:: {epoch + 1:0>{len(str(epochs_num))}}/{epochs_num}, {elapsed_mins:0>2}m {elapsed_secs:0>2}s :: "
                        f"Train Loss: {train_loss:<8.4f}, Train PPL: {math.exp(train_loss):<8.4f} | ",
                        f"Valid Loss: {valid_loss:<8.4f}, Valid PPL: {math.exp(valid_loss):<8.4f} | "
                        f"Valid BLEU: {valid_bleu:<8.4f} |")

            # Record the losses and BLEU scores
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.valid_bleu_scores.append(valid_bleu)

            # Save the best model based on validation loss and BLEU score (only on rank 0 if DDP)
            if (ddp == False) or (ddp == True and dist.get_rank() == 0):
                # Save the model if it has the best BLEU score
                if valid_bleu > self.best_bleu:
                    self.best_bleu = valid_bleu

                    save_dir: str = self.save_config.CHECKPOINT_DIR
                    os.makedirs(save_dir, exist_ok=True)

                    torch.save(actual_model.state_dict(), os.path.join(save_dir, f"epoch-{epoch:0>{len(str(epochs_num))}}-valid_loss-{valid_loss:0>7.4f}-bleu-{valid_bleu:0>7.4f}.pt"))

            if ddp == True: dist.barrier()

    def _train_with_ddp(self: Any) -> None:
        """Train the model using Distributed Data Parallel (DDP)."""
        assert self.ddp == True, "DDP flag must be set to True for DDP training."
        assert torch.cuda.is_available(), "This code requires a GPU with CUDA support."
        assert torch.cuda.device_count() > 1, "This code requires at least two GPUs for DDP."

        # Setup devices and distributed environment
        world_size: int = int(os.environ["WORLD_SIZE"])     # Total number of processes (GPUs)
        rank: int = int(os.environ["RANK"])     # Rank of the current process
        local_rank: int = int(os.environ["LOCAL_RANK"])     # Local rank of the current process on the node

        torch.backends.cudnn.benchmark = True   # Enable cuDNN benchmark for performance
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size,
                                rank=rank, timeout=datetime.timedelta(seconds=3000))   # Avoid timeout issues
        torch.cuda.set_device(rank)   # Set device for the current process (by global unique rank)
        dist.barrier()

        # Load Data
        data_loader: Multi30kDataLoader = self._init_dataloader()
        train_iter, valid_iter, test_iter = data_loader.load()
        dist.barrier()

        # Initialize Model
        transformer: Optional[Transformer] = self._init_model(device=torch.device(local_rank))
        transformer = DDP(transformer, device_ids=[local_rank], output_device=torch.device(local_rank))

        # Initialize Training Components
        self._init_training_components(device=torch.device(local_rank))

        # Start Training
        self._train(
            model=transformer,
            trg_tokenizer=data_loader.en_tokenizer,
            train_iter=train_iter,
            valid_iter=valid_iter,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion,
            device=torch.device(local_rank),
            ddp=self.ddp,
        )

        # Cleanup distributed environment
        dist.barrier()      # Ensure all processes reach this point before cleanup
        cleanup()

    def _train_without_ddp(self: Any) -> None:
        """Train the model without Distributed Data Parallel (DDP)."""
        assert self.ddp == False, "DDP flag must be set to False for non-DDP training."
        assert torch.cuda.is_available(), "This code requires a GPU with CUDA support."

        # Setup device
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Data
        data_loader: Multi30kDataLoader = self._init_dataloader()
        train_iter, valid_iter, test_iter = data_loader.load()

        # Initialize Model
        transformer: Transformer = self._init_model(device=device)

        # Initialize Training Components
        self._init_training_components(device=device)

        # Start Training
        self._train(
            model=transformer,
            trg_tokenizer=data_loader.en_tokenizer,
            train_iter=train_iter,
            valid_iter=valid_iter,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion,
            device=device,
            ddp=self.ddp,
        )

    def _save_traces_data(self: Any) -> None:
        """Save training trace and sample trace data."""
        # Save training trace data
        save_dir: str = self.save_config.TRAIN_TRACE_DIR
        os.makedirs(save_dir, exist_ok=True)

        train_trace: Dict[Optional[List[float]]] = {
            "train_losses": self.train_losses,
            "valid_losses": self.valid_losses,
            "valid_bleu_scores": self.valid_bleu_scores,
            "best_loss": self.best_loss,
            "best_bleu": self.best_bleu,
        }
        save_obj_by_pickle(save_dir, "train_trace.pkl", train_trace)

        # Save sample trace data
        save_dir: str = self.save_config.SAMPLE_TRACE_DIR
        os.makedirs(save_dir, exist_ok=True)

        save_obj_by_pickle(save_dir, "sample_trace.pkl", self.sample_trace)

    def train(self: Any) -> None:
        """Train the model based on the DDP flag."""
        # Train the model
        if self.ddp == True:
            self._train_with_ddp()
        else:
            self._train_without_ddp()

        # Save training trace data
        if (self.ddp == False) or (self.ddp == True):
            self._save_traces_data()


if __name__ == '__main__':
    """Example usage of the Trainer class."""
    # Remove warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Initialize configurations
    dataset_config: DatasetConfig = DatasetConfig()
    model_config: ModelConfig = ModelConfig()
    train_config: TrainConfig = TrainConfig()
    save_config: SaveConfig = SaveConfig()

    # Create a Trainer instance
    trainer: Trainer = Trainer(
        dataset_config=dataset_config,
        model_config=model_config,
        train_config=train_config,
        save_config=save_config,
        ddp=True,  # Set to True for DDP training
    )

    # Start training
    trainer.train()

