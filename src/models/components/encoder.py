import torch
from torch import nn
from typing import Any

from src.models.blocks.encoder_layer import EncoderLayer
from src.models.layers.embedding import TransformerEmbedding


class Encoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self: Any, encoder_vocab_size: int, max_seq_len: int, padding_idx: int,
                num_layers: int, d_model: int, num_heads: int, d_ff: int,
                drop_prob: float, device: torch.device) -> None:
        """constructor
        @param encoder_vocab_size: size of the encoder vocabulary
        @param max_seq_len: maximum sequence length
        @param padding_idx: index of the padding token which decided by tokenizer
        @param num_layers: number of encoder layers
        @param d_model: dimension of the model
        @param num_heads: number of attention heads
        @param d_ff: dimension of the feed-forward network
        @param drop_prob: dropout probability
        @param device: device to use for the encoder
        """
        super(Encoder, self).__init__()

        self.embedding: TransformerEmbedding = TransformerEmbedding(
            vocab_size=encoder_vocab_size,
            max_seq_len=max_seq_len,
            padding_idx=padding_idx,
            d_model=d_model,
            drop_prob=drop_prob,
            device=device
        )

        self.layers: nn.ModuleList = nn.ModuleList([EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            drop_prob=drop_prob
        ) for _ in range(num_layers)])

        self.to(device)     # Move entire encoder to device

    def forward(self: Any, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """apply encoder
        @param X: input tensor of shape (batch_size, seq_len)
        @param mask: mask tensor of shape (batch_size, 1, 1, seq_len)
        @return: output tensor of shape (batch_size, seq_len, d_model)
        """
        # Embedding and Positional Encoding
        X = self.embedding(X)
        # Pass through each encoder layer
        for layer in self.layers:
            X = layer(X, mask)
        # Return the output
        return X

