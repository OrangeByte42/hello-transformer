import torch
from torch import nn
from typing import Any

from src.model.layers.layernorm import LayerNorm
from src.model.layers.multihead_attention import MultiHeadAttention
from src.model.layers.position_wise_feed_forward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    """Decoder Layer"""
    def __init__(self: Any, d_model: int, num_heads: int, d_ff: int, drop_prob: float) -> None:
        """constructor
        @param d_model: dimension of the model
        @param num_heads: number of attention heads
        @param d_ff: dimension of the feed-forward network
        @param drop_prob: dropout probability
        @return: None
        """
        super(DecoderLayer, self).__init__()
        self.mha1: MultiHeadAttention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ln1: LayerNorm = LayerNorm(d_model=d_model)
        self.dropout1: nn.Dropout = nn.Dropout(p=drop_prob)

        self.mha2: MultiHeadAttention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ln2: LayerNorm = LayerNorm(d_model=d_model)
        self.dropout2: nn.Dropout = nn.Dropout(p=drop_prob)

        self.ffn: PositionWiseFeedForward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, drop_prob=drop_prob)
        self.ln3: LayerNorm = LayerNorm(d_model=d_model)
        self.dropout3: nn.Dropout = nn.Dropout(p=drop_prob)

    def forward(self: Any, decoder_input_X: torch.Tensor, encoder_output_X: torch.Tensor,
                src_mask: torch.Tensor, cross_mask: torch.Tensor) -> torch.Tensor:
        """apply decoder layer
        @param decoder_input_X: input tensor of shape (batch_size, seq_len, d_model)
        @param encoder_output_X: output tensor from the encoder of shape (batch_size, seq_len, d_model)
        @param src_mask: mask tensor for the source sequence of shape (batch_size, 1, seq_len, seq_len)
        @param cross_mask: mask tensor for the cross-attention of shape (batch_size, 1, seq_len, seq_len)
        @return: output tensor of shape (batch_size, seq_len, d_model)
        """
        # Compute self-attention
        residual_X: torch.Tensor = decoder_input_X
        X: torch.Tensor = self.mha1(Q=decoder_input_X, K=decoder_input_X, V=decoder_input_X, mask=cross_mask)
        # Compute add & norm
        X = self.dropout1(X)
        X = self.ln1(X + residual_X)

        # Compute cross-attention
        residual_X = X
        X = self.mha2(Q=X, K=encoder_output_X, V=encoder_output_X, mask=src_mask)
        # Compute add & norm
        X = self.dropout2(X)
        X = self.ln2(X + residual_X)

        # Compute position-wise feed-forward network
        residual_X = X
        X = self.ffn(X)
        # Compute add & norm
        X = self.dropout3(X)
        X = self.ln3(X + residual_X)

        # Return the output
        return X


