import torch
import torch.distributed as dist
from torch import nn
from typing import Tuple


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
    print("Distributed environment cleaned up.")
    torch.cuda.empty_cache()    # Clear the CUDA memory cache
    print("CUDA memory cache cleared.")




