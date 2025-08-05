import torch
from torch import nn
from typing import Any


class LayerNorm(nn.Module):
    """Layer Normalization"""
    def __init__(self: Any, d_model: int, eps: float = 1e-12) -> None:
        """constructor
        @param d_model: dimension of the model
        @param eps: small value to avoid division by zero
        @return: None
        """
        super(LayerNorm, self).__init__()
        self.gamma: nn.Parameter = nn.Parameter(torch.ones(d_model))
        self.beta: nn.Parameter = nn.Parameter(torch.zeros(d_model))
        self.eps: float = eps

    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        """apply layer normalization
        @param X: input tensor of shape (batch_size, seq_len, d_model)
        @return: normalized tensor of shape (batch_size, seq_len, d_model)
        """
        mean: torch.Tensor = X.mean(dim=-1, keepdim=True)
        var: torch.Tensor = X.var(dim=-1, keepdim=True, unbiased=False)
        output: torch.Tensor = (X - mean) / torch.sqrt(var + self.eps)
        output = self.gamma * output + self.beta
        return output

