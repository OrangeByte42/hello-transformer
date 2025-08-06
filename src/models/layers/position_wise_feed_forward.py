import torch
from torch import nn
from typing import Any


class PositionWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    def __init__(self: Any, d_model: int, d_ff: int, drop_prob: float) -> None:
        """constructor
        @param d_model: dimension of the model
        @param d_ff: dimension of the feed-forward network
        @param drop_prob: dropout probability
        @return: None
        """
        super(PositionWiseFeedForward, self).__init__()

        self.fc1: nn.Linear = nn.Linear(d_model, d_ff)
        self.relu: nn.ReLU = nn.ReLU()
        self.dropout: nn.Dropout = nn.Dropout(p=drop_prob)
        self.fc2: nn.Linear = nn.Linear(d_ff, d_model)

    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        """apply position-wise feed-forward network
        @param X: input tensor of shape (batch_size, seq_len, d_model)
        @return: output tensor of shape (batch_size, seq_len, d_model)
        """
        X = self.fc1(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.fc2(X)
        return X

