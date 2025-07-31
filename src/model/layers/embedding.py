import math
import torch
from torch import nn
from typing import Any


class TokenEmbedding(nn.Embedding):
    """token embedding layer"""
    def __init__(self: Any, vocab_size: int, d_model: int) -> None:
        """constructor
        @param vocab_size: size of the vocabulary
        @param d_model: dimension of the model
        @return: None
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class PositionalEncoding(nn.Module):
    """sine and cosine fixed positional encoding"""
    def __init__(self: Any, d_model: int, max_seq_len: int, device: torch.device) -> None:
        """constructor
        @param d_model: dimension of the model
        @param max_seq_len: maximum sequence length
        @param device: device to use for the positional encoding
        @return: None
        """
        super(PositionalEncoding, self).__init__()

        # Initialize the positional encoding matrix which same size with input sequence
        self.encoding: torch.Tensor = torch.zeros(max_seq_len, d_model, device=device)
        self.encoding.requires_grad = False

        # Compute the positional encodings
        pos: torch.Tensor = torch.arange(0, max_seq_len, device=device)
        pos = pos.float().unsqueeze(1)

        _2i: torch.Tensor = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10_000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10_000 ** (_2i / d_model)))

    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        """add positional encoding to the input sequence
        @param X: input sequence of shape (batch_size, seq_len, d_model)
        @return: positional encoding matrix of shape (seq_len, d_model)
        """
        return self.encoding[:X.shape[1], :]


class TransformerEmbedding(nn.Module):
    """token embedding & positional encoding layer"""
    def __init__(self: Any, vocab_size: int, d_model: int, max_seq_len: int, drop_prob: float, device: torch.device) -> None:
        """constructor
        @param vocab_size: size of the vocabulary
        @param d_model: dimension of the model
        @param max_seq_len: maximum sequence length
        @param drop_prob: dropout probability
        @param device: device to use for the embedding
        @return: None
        """
        super(TransformerEmbedding, self).__init__()
        self.token_embedding: TokenEmbedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.positional_encoding: PositionalEncoding = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len, device=device)
        self.dropout: nn.Dropout = nn.Dropout(p=drop_prob)

    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        """add token embedding and positional encoding to the input sequence
        @param X: input sequence of shape (batch_size, seq_len)
        @return: embedded sequence of shape (batch_size, seq_len, d_model)
        """
        token_embedded: torch.Tensor = self.token_embedding(X) * math.sqrt(self.token_embedding.embedding_dim)
        pos_encoded: torch.Tensor = self.positional_encoding(token_embedded)
        return self.dropout(token_embedded + pos_encoded)





