import os
from transformers import AutoTokenizer
from collections import Counter
from typing import Any, List, Dict, Set

from src.utils.tokenizer.tokenizer import Tokenizer

class BertTokenizer(Tokenizer):
    """BERT-based tokenizer for text preprocessing"""

    def __init__(self: Any, model_name: str, max_seq_len: int,
                    cache_dir: str = os.path.join(".", "data")) -> None:
        """Initialize BERT tokenizer
        @param model_name: BERT model name (e.g., 'bert-base-uncased', 'bert-base-german-dbmdz-cased')
        @param max_seq_len: Maximum sequence length
        """
        super(BertTokenizer, self).__init__(model_name, max_seq_len)  # Call parent constructor

        # Store model name and max sequence length
        self.model_name: str = model_name
        self.max_seq_len: int = max_seq_len

        # Load BERT tokenizer
        os.makedirs(cache_dir, exist_ok=True)   # Ensure data directory exists
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)    # Load BERT tokenizer

        # Special tokens
        self._pad_token: str = self.tokenizer.pad_token
        self._unk_token: str = self.tokenizer.unk_token
        self._sos_token: str = self.tokenizer.cls_token  # Start of sequence
        self._eos_token: str = self.tokenizer.sep_token  # End of sequence

        self.special_tokens: Set[str] = {
            self._pad_token,
            self._unk_token,
            self._sos_token,
            self._eos_token,
        }

        # Special token IDs
        self._pad_token_id: int = self.tokenizer.pad_token_id
        self._unk_token_id: int = self.tokenizer.unk_token_id
        self._sos_token_id: int = self.tokenizer.cls_token_id
        self._eos_token_id: int = self.tokenizer.sep_token_id

        # Initialize vocabulary
        self.vocab: Dict[str, int] = self.tokenizer.get_vocab()

    def _tokenize(self: Any, text: str) -> List[str]:
        """Tokenize text using BERT tokenizer
        @param text: Input text string
        @return: List of tokens
        """
        tokens: List[str] = self.tokenizer.tokenize(text)
        return tokens

    def build_vocab(self: Any, texts: List[str], min_freq: int = 0) -> None:
        """Build vocabulary from list of texts
        @param texts: List of text strings
        @param min_freq: Minimum frequency for token to be included
        """
        # Count token frequencies
        token_counts: Counter = Counter()
        for text in texts:
            tokens: List[str] = self._tokenize(text)
            token_counts.update(tokens)

        # Add tokens with sufficient frequency to the vocabulary
        for token, count in token_counts.items():
            if count < min_freq or token in self.vocab:
                continue
            self.tokenizer.add_tokens([token])

    def convert_ids_to_tokens(self: Any, ids: List[int], skip_special_tokens: bool) -> List[str]:
        """Convert a list of token IDs to tokens
        @param ids: List of token IDs
        @param skip_special_tokens: Whether to skip special tokens
        @return: List of tokens
        """
        tokens: List[str] = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
        return tokens

    def convert_tokens_to_ids(self: Any, tokens: List[str], skip_special_tokens: bool) -> List[int]:
        """Convert a list of tokens to token IDs
        @param tokens: List of tokens
        @param skip_special_tokens: Whether to skip special tokens
        @return: List of token IDs
        """
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        token_ids: List[int] = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    def encode(self: Any, text: str, add_special_tokens: bool = True, padding: bool = True,
                truncation: bool = True) -> List[int]:
        """Encode a text string into a list of token IDs
        @param text: Input text string
        @param add_special_tokens: Whether to add special tokens (sos, eos)
        @param padding: Whether to pad the sequence
        @param truncation: Whether to truncate the sequence to max_seq_len
        @return: List of token IDs
        """
        # Tokenize the text
        tokens: List[str] = self._tokenize(text)
        token_ids: List[int] = self.tokenizer.convert_tokens_to_ids(tokens)

        # Deal with special tokens
        if add_special_tokens == True:
            token_ids = [self._sos_token_id] + token_ids + [self._eos_token_id]

        # Deal with truncation
        if truncation and len(token_ids) > self.max_seq_len:
            if add_special_tokens == True:
                token_ids = token_ids[:self.max_seq_len - 1] + [self._eos_token_id]
            else:
                token_ids = token_ids[:self.max_seq_len]

        # Deal with padding
        if padding == True:
            token_ids += [self._pad_token_id] * (self.max_seq_len - len(token_ids))

        return token_ids

    def decode(self: Any, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs into a text string
        @param token_ids: List of token IDs to decode
        @param skip_special_tokens: Whether to skip special tokens in the output
        @return: Decoded text string
        """
        # Convert token IDs to tokens
        tokens: List[str] = self.tokenizer.convert_ids_to_tokens(token_ids)

        # Skip special tokens if required
        if skip_special_tokens == True:
            tokens = [token for token in tokens if token not in self.special_tokens]

        # Join tokens into a single string
        text: str = self.tokenizer.convert_tokens_to_string(tokens).strip()

        return text

