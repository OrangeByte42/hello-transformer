import torch
from torch import nn
from typing import Any, Union

from src.model.layers.layernorm import LayerNorm
from src.model.layers.multihead_attention import MultiHeadAttention
from src.model.layers.position_wise_feed_forward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    def __init__(self: Any, d_model: int, num_heads: int, d_ff: int, drop_prob: float) -> None:
        """constructor
        @param d_model: dimension of the model
        @param num_heads: number of attention heads
        @param d_ff: dimension of the feed-forward network
        @param drop_prob: dropout probability
        @return: None
        """

        super(EncoderLayer, self).__init__()

        # Sublayer-01
        self.mha: MultiHeadAttention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ln1: LayerNorm = LayerNorm(d_model=d_model)
        self.dropout1: nn.Dropout = nn.Dropout(p=drop_prob)

        # Sublayer-02
        self.ffn: PositionWiseFeedForward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, drop_prob=drop_prob)
        self.ln2: LayerNorm = LayerNorm(d_model=d_model)
        self.dropout2: nn.Dropout = nn.Dropout(p=drop_prob)

    def forward(self: Any, X: torch.Tensor, mask: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """apply encoder layer
        @param X: input tensor of shape (batch_size, seq_len, d_model)
        @param mask: optional mask tensor of shape (batch_size, 1, 1, seq_len)
        @return: output tensor of shape (batch_size, seq_len, d_model)
        """

        # Compute self-attention
        residual_X: torch.Tensor = X
        X = self.mha(Q=X, K=X, V=X, mask=mask)
        # Compute add & norm
        X = self.dropout1(X)
        X = self.ln1(X + residual_X)

        # Compute position-wise feed-forward network
        residual_X = X
        X = self.ffn(X)
        # Compute add & norm
        X = self.dropout2(X)
        X = self.ln2(X + residual_X)

        # Return the output
        return X



