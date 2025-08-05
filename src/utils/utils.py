import os
import torch
from torch import nn
import torch.distributed as dist
from typing import Tuple


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

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count the number of trainable parameters in a PyTorch model.
    @param model: PyTorch model
    @return: (total number of parameters, total number of trainable parameters)
    """
    total_params: int = sum(p.numel() for p in model.parameters())
    trainable_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def cleanup() -> None:
    """Cleanup distributed training environment."""
    dist.destroy_process_group()    # Destroy the process group to clean up resources
    torch.cuda.empty_cache()    # Clear the CUDA memory cache

def save_list(save_dir: str, file_name: str, list_data: list) -> None:
    """Save a list to a text file, one item per line.
    @param save_dir: Directory to save the file
    @param file_name: Name of the file
    @param list_data: List of items to save
    """
    save_path: str = os.path.join(save_dir, file_name)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(str(list_data))

