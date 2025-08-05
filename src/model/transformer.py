import torch
from torch import nn
from typing import Any

from src.model.components.encoder import Encoder
from src.model.components.decoder import Decoder


class Transformer(nn.Module):
    """Transformer Model"""
    def __init__(self: Any, encoder_vocab_size: int, decoder_vocab_size: int,
                    max_seq_len: int, src_pad_id: int, trg_pad_id: int,
                    num_layers: int, d_model: int, num_heads: int, d_ff: int,
                    drop_prob: float, device: torch.device) -> None:
        """constructor
        @param encoder_vocab_size: size of the encoder vocabulary
        @param decoder_vocab_size: size of the decoder vocabulary
        @param max_seq_len: maximum sequence length
        @param src_pad_id: padding token id for the source sequence
        @param trg_pad_id: padding token id for the target sequence
        @param padding_idx: index of the padding token which decided by tokenizer
        @param num_layers: number of layers in the encoder and decoder
        @param d_model: dimension of the model
        @param num_heads: number of attention heads
        @param d_ff: dimension of the feed-forward network
        @param drop_prob: dropout probability
        @param device: device to use for the transformer
        """
        super(Transformer, self).__init__()

        self.src_pad_id: int = src_pad_id
        self.trg_pad_id: int = trg_pad_id

        self.device: torch.device = device

        self.encoder: Encoder = Encoder(
            encoder_vocab_size=encoder_vocab_size,
            max_seq_len=max_seq_len,
            padding_idx=src_pad_id,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            drop_prob=drop_prob,
            device=device
        )

        self.decoder: Decoder = Decoder(
            decoder_vocab_size=decoder_vocab_size,
            max_seq_len=max_seq_len,
            padding_idx=trg_pad_id,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            drop_prob=drop_prob,
            device=device
        )

        self.to(device)     # Move entire transformer to device

    def _make_src_mask(self: Any, src_X: torch.Tensor) -> torch.Tensor:
        """create source mask
        @param src_X: source input tensor of shape (batch_size, seq_len)
        @return: mask tensor of shape (batch_size, 1, 1, seq_len), shape[1] == 1 for mha_heads broadcast
        """
        # Only padding mask which aims to ignore the padding tokens in the source sequence
        return (src_X != self.src_pad_id).unsqueeze(1).unsqueeze(1)

    def _make_trg_mask(self: Any, trg_X: torch.Tensor) -> torch.Tensor:
        """create target mask (combines padding mask and causal mask)
        @param trg_X: target input tensor of shape (batch_size, seq_len)
        @return: mask tensor of shape (batch_size, 1, seq_len, seq_len), shape[1] == 1 for mha_heads broadcast
        """
        # Create padding mask
        trg_pad_mask: torch.Tensor = (trg_X != self.trg_pad_id).unsqueeze(1).unsqueeze(1)

        # Create subsequent mask (causal mask)
        trg_len: int = trg_X.shape[1]
        trg_sub_mask: torch.Tensor = torch.tril(torch.ones(trg_len, trg_len, device=trg_X.device, dtype=torch.bool))

        # Combine masks
        trg_mask: torch.Tensor = trg_pad_mask & trg_sub_mask

        # Return the mask
        return trg_mask

    def forward(self: Any, src_X: torch.Tensor, trg_X: torch.Tensor) -> torch.Tensor:
        """apply transformer
        @param src_X: source input tensor of shape (batch_size, src_seq_len)
        @param trg_X: target input tensor of shape (batch_size, trg_seq_len)
        @return: output tensor of shape (batch_size, trg_seq_len, decoder_vocab_size)
        """
        # Create masks
        src_mask: torch.Tensor = self._make_src_mask(src_X)
        trg_mask: torch.Tensor = self._make_trg_mask(trg_X)

        # Pass through encoder and decoder
        encoder_output_X: torch.Tensor = self.encoder(src_X, src_mask)
        decoder_output_X: torch.Tensor = self.decoder(trg_X, encoder_output_X, src_mask, trg_mask)

        # Return the output
        return decoder_output_X

