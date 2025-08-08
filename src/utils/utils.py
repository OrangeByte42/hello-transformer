import os
import pickle
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Tuple, Optional, Any

from src.models.transformer import Transformer

def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    """Calculate elapsed time in minutes and seconds
    @param start_time: Start time in seconds
    @param end_time: End time in seconds
    @return: Tuple of elapsed time in minutes and seconds
    """
    elapsed_time: float = end_time - start_time
    elapsed_mins: int = int(elapsed_time // 60)
    elapsed_secs: int = int(elapsed_time % 60)
    return elapsed_mins, elapsed_secs

def count_parameters(model: Optional[nn.Module], ddp: bool) -> Tuple[int, int]:
    """Count the number of trainable parameters in a PyTorch model.
    @param model: PyTorch model
    @return: (total number of parameters, total number of trainable parameters)
    """
    # Get the actual model if using DDP
    actual_model: Optional[Transformer] = None
    if ddp == True and isinstance(model, DDP):
        actual_model = model.module
    else:
        actual_model = model

    # Count total and trainable parameters
    total_params: int = sum(p.numel() for p in actual_model.parameters())
    trainable_params: int = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    return total_params, trainable_params

def cleanup() -> None:
    """Cleanup distributed training environment."""
    dist.destroy_process_group()    # Destroy the process group to clean up resources
    torch.cuda.empty_cache()    # Clear the CUDA memory cache

def save_obj_by_pickle(save_dir: str, file_name: str, obj: Any) -> None:
    """Save an object to a pickle file.
    @param save_dir: Directory to save the file
    @param file_name: Name of the file
    @param obj: Object to save
    """
    save_path: str = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

