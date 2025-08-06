import torch
from torch import nn
from typing import Any

from src.models.blocks.decoder_layer import DecoderLayer
from src.models.layers.embedding import TransformerEmbedding


class Decoder(nn.Module):
    """Transformer Decoder"""
    def __init__(self: Any, decoder_vocab_size: int, max_seq_len: int, padding_idx: int,
                num_layers: int, d_model: int, num_heads: int, d_ff: int,
                drop_prob: float, device: torch.device) -> None:
        """constructor
        @param decoder_vocab_size: size of the decoder vocabulary
        @param max_seq_len: maximum sequence length
        @param padding_idx: index of the padding token which decided by tokenizer
        @param num_layers: number of decoder layers
        @param d_model: dimension of the model
        @param num_heads: number of attention heads
        @param d_ff: dimension of the feed-forward network
        @param drop_prob: dropout probability
        @param device: device to use for the decoder
        """
        super(Decoder, self).__init__()

        self.embedding: TransformerEmbedding = TransformerEmbedding(
            vocab_size=decoder_vocab_size,
            max_seq_len=max_seq_len,
            padding_idx=padding_idx,
            d_model=d_model,
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

        self.to(device)     # Move entire decoder to device

    def forward(self: Any, decoder_input_X: torch.Tensor, encoder_output_X: torch.Tensor,
                src_mask: torch.Tensor, trg_mask: torch.Tensor) -> torch.Tensor:
        """apply decoder
        @param decoder_input_X: input tensor of shape (batch_size, seq_len)
        @param encoder_output_X: output tensor from the encoder of shape (batch_size, seq_len, d_model)
        @param src_mask: mask tensor for the source sequence of shape (batch_size, 1, 1, seq_len)
        @param trg_mask: mask tensor for the target sequence of shape (batch_size, 1, seq_len, seq_len)
        @return: output tensor of shape (batch_size, seq_len, decoder_vocab_size)
        """
        # Embedding and Positional Encoding
        X = self.embedding(decoder_input_X)
        # Pass through each decoder layer
        for layer in self.layers:
            X = layer(X, encoder_output_X, src_mask, trg_mask)
        output: torch.Tensor = self.fc(X)
        # Return logits
        return output

