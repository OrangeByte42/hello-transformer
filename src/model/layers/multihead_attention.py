import math
import torch
from torch import nn
from typing import Any, Union, Tuple


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self: Any) -> None:
        """constructor
        @return: None
        """
        super(ScaledDotProductAttention, self).__init__()
        self.softmax: nn.Softmax = nn.Softmax(dim=-1)

    def forward(self: Any, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """compute scaled dot-product attention
        @param Q: query tensor of shape (batch_size, num_heads, seq_len_q, d_k)
        @param K: key tensor of shape (batch_size, num_heads, seq_len_k, d_k)
        @param V: value tensor of shape (batch_size, num_heads, seq_len_v, d_v) usually seq_len_v == seq_len_k
        @param mask: optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
        @return: output tensor of shape (batch_size, num_heads, seq_len_q, d_v) and attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # Compute the attention scores
        d_k: int = Q.shape[-1]
        attention_scores: torch.Tensor = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        # Apply the mask if provided
        if mask is not None:
            # attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        # Compute the attention weights
        attention_weights: torch.Tensor = self.softmax(attention_scores)
        # Compute the output
        output: torch.Tensor = attention_weights @ V
        # Return the output and attention weights
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self: Any, d_model: int, num_heads: int) -> None:
        """constructor
        @param d_model: dimension of the model
        @param num_heads: number of attention heads
        @return: None
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads: int = num_heads
        self.attention: ScaledDotProductAttention = ScaledDotProductAttention()
        self.W_q: nn.Linear = nn.Linear(d_model, d_model)
        self.W_k: nn.Linear = nn.Linear(d_model, d_model)
        self.W_v: nn.Linear = nn.Linear(d_model, d_model)
        self.W_o: nn.Linear = nn.Linear(d_model, d_model)

    def _split_heads(self: Any, X: torch.Tensor) -> torch.Tensor:
        """split tensor by number of heads
        @param X: input tensor of shape (batch_size, seq_len, d_model)
        @return: tensor of shape (batch_size, num_heads, seq_len, d_tensor)
        """
        batch_size, seq_len, d_model = X.shape
        d_tensor: int = d_model // self.num_heads
        output: torch.Tensor = X.view(batch_size, seq_len, self.num_heads, d_tensor).transpose(1, 2)
        return output

    def _concat_heads(self: Any, X: torch.Tensor) -> torch.Tensor:
        """concat tensor by number of heads
        @param X: input tensor of shape (batch_size, num_heads, seq_len, d_tensor)
        @return: tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_tensor = X.shape
        d_model: int = num_heads * d_tensor
        output: torch.Tensor = X.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return output

    def forward(self: Any, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """compute multi-head attention
        @param Q: query tensor of shape (batch_size, seq_len_q, d_model)
        @param K: key tensor of shape (batch_size, seq_len_k, d_model)
        @param V: value tensor of shape (batch_size, seq_len_v, d_model) usually seq_len_v == seq_len_k
        @param mask: optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
        @return: output tensor of shape (batch_size, seq_len_q, d_model)
        """
        # Dot-product with weight matrices
        Q, K, V = self.W_q(Q), self.W_k(K), self.W_v(V)
        # Split tensors into multiple heads
        Q, K, V = self._split_heads(Q), self._split_heads(K), self._split_heads(V)
        # Compute scaled dot-product attention
        out, _ = self.attention(Q, K, V, mask=mask)
        # Concatenate heads
        output: torch.Tensor = self._concat_heads(out)
        output = self.W_o(output)
        # Return the output tensor
        return output




