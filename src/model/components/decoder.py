import torch
from torch import nn
from typing import Any

from src.model.blocks.decoder_layer import DecoderLayer
from src.model.layers.embedding import TransformerEmbedding


class Decoder(nn.Module):
    """Transformer Decoder"""
    def __init__(self: Any, decoder_vocab_size: int, max_seq_len: int, d_model: int, num_heads: int,
                d_ff: int, num_layers: int, drop_prob: float, device: torch.device) -> None:
        """constructor
        @param decoder_vocab_size: size of the decoder vocabulary
        @param max_seq_len: maximum sequence length
        @param d_model: dimension of the model
        @param num_heads: number of attention heads
        @param d_ff: dimension of the feed-forward network
        @param num_layers: number of decoder layers
        @param drop_prob: dropout probability
        @param device: device to use for the decoder
        """
        super(Decoder, self).__init__()

        self.embedding: TransformerEmbedding = TransformerEmbedding(
            vocab_size=decoder_vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            drop_prob=drop_prob,
            device=device
        )
        self.layers: nn.ModuleList = nn.ModuleList([DecoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            drop_prob=drop_prob
        ) for _ in range(num_layers)])
        self.fc: nn.Linear = nn.Linear(d_model, decoder_vocab_size)

    def forward(self: Any, decoder_input_X: torch.Tensor, encoder_output_X: torch.Tensor,
                src_mask: torch.Tensor, cross_mask: torch.Tensor) -> torch.Tensor:
        """apply decoder
        @param decoder_input_X: input tensor of shape (batch_size, seq_len)
        @param encoder_output_X: output tensor from the encoder of shape (batch_size, seq_len, d_model)
        @param src_mask: mask tensor for the source sequence of shape (batch_size, 1, seq_len, seq_len)
        @param cross_mask: mask tensor for the cross-attention of shape (batch_size, 1, seq_len, seq_len)
        @return: output tensor of shape (batch_size, seq_len, d_model)
        """
        X = self.embedding(decoder_input_X)
        for layer in self.layers:
            X = layer(X, encoder_output_X, src_mask, cross_mask)
        output: torch.Tensor = self.fc(X)
        return output







