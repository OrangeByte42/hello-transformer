import torch
from typing import Any

from src.utils.tokenizer.tokenizer import Tokenizer
from src.model.transformer import Transformer


class AutoregressiveTransformer:
    """A class to package transformer model for autoregressive generation."""

    def __init__(self: Any, model: Transformer, trg_tokenizer: Tokenizer,
                    device: torch.device) -> None:
        """Initialize the autoregressive transformer model.
        @param model: Transformer model instance.
        @param tokenizer: Tokenizer instance for decoding.
        @param device: Device to run the model on (CPU or GPU).
        @return: None
        """
        self.model: Transformer = model
        self.trg_tokenizer: Tokenizer = trg_tokenizer
        self.device: torch.device = device

    def generate(self: Any, src_X: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        """Generate sequences autoregressively.
        @param src_X: Source input tensor of shape (batch_size, src_seq_len).
        @param max_seq_len: Maximum length of the generated sequence.
        @return: Generated sequences tensor of shape (batch_size, max_seq_len).
        """
        # Necessary variables
        sos_token_id: int = self.trg_tokenizer.sos_token_id
        eos_token_id: int = self.trg_tokenizer.eos_token_id
        pad_token_id: int = self.trg_tokenizer.pad_token_id

        batch_size: int = src_X.shape[0]
        generated_seqs: torch.Tensor = torch.full((batch_size, 1), sos_token_id, device=self.device)

        # Encode source sequences once
        src_mask: torch.Tensor = self.model.make_src_mask(src_X)
        encoder_output: torch.Tensor = self.model.encoder(src_X, src_mask)

        # Track which sequences have finished generation (generated EOS)
        finished: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Autoregressive generation loop
        for _ in range(max_seq_len - 1):    # -1 because we already have SOS token
            # If all sequences are finished, break early
            if torch.all(finished): break

            # Create target mask for current generated sequences
            trg_mask: torch.Tensor = self.model.make_trg_mask(generated_seqs)

            # Decode current sequence to get next token probabilities
            decoder_output: torch.Tensor = self.model.decoder(generated_seqs, encoder_output, src_mask, trg_mask)

            # Greedy decoding: select the token with the highest probability
            next_token_logits: torch.Tensor = decoder_output[:, -1, :]      # (batch_size, vocab_size)
            next_tokens: torch.Tensor = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch_size, 1)

            # Append the predicted next tokens to the generated sequences
            generated_seqs = torch.cat((generated_seqs, next_tokens), dim=1)

            # Update finished sequences for sequences that generated EOS
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)

        # Pad sequences to max_seq_len if necessary
        padded_seqs: torch.Tensor = torch.full((batch_size, max_seq_len), pad_token_id, device=self.device)
        padded_seqs[:, :generated_seqs.shape[1]] = generated_seqs
        generated_seqs = padded_seqs

        # Return the generated sequences
        return generated_seqs

