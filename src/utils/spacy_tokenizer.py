import spacy
from typing import Any, List, Dict, Set, Union, Optional
from collections import Counter


class SpacyTokenizer:
    """SpaCy-based tokenizer for text preprocessing"""

    def __init__(self: Any, model_name: str, max_seq_len: int = 256):
        """
        Initialize SpaCy tokenizer
        @param model_name: SpaCy model name (e.g., 'en_core_web_sm', 'de_core_news_sm')
        @param max_seq_len: Maximum sequence length
        """
        # Store model name and max sequence length
        self.model_name: str = model_name
        self.max_seq_len: str = max_seq_len
        self.nlp: Any = spacy.load(model_name)  # Load SpaCy model
        # Special tokens - matching reference implementation
        self.pad_token: str = "<pad>"
        self.unk_token: str = "<unk>"
        self.sos_token: str = "<sos>"  # Changed to match reference
        self.eos_token: str = "<eos>"
        self.special_tokens: Set[str] = {
            self.pad_token,
            self.unk_token,
            self.sos_token,
            self.eos_token,
        }
        # Special token IDs - using padding_idx=1 for token embedding
        self.pad_token_id: int = 0  # Changed to match reference (padding_idx=1)
        self.unk_token_id: int = 1  # UNK token gets id 0
        self.sos_token_id: int = 2  # SOS token
        self.eos_token_id: int = 3  # EOS token
        # Initialize vocabulary with special tokens
        self.vocab: Dict[str, int] = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.sos_token: self.sos_token_id,
            self.eos_token: self.eos_token_id,
        }
        self.id2token: Dict[int, str] = {
            self.pad_token_id: self.pad_token,
            self.unk_token_id: self.unk_token,
            self.sos_token_id: self.sos_token,
            self.eos_token_id: self.eos_token,
        }

    @property
    def vocab_size(self: Any) -> int:
        """Return the size of the vocabulary"""
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using SpaCy
        @param text: Input text string
        @return: List of tokens
        """
        tokens: List[str] = [tok.text for tok in self.nlp.tokenizer(text.lower().strip())]
        return tokens

    def build_vocabulary(self, texts: List[str], min_freq: int = 0):
        """
        Build vocabulary from list of texts
        @param texts: List of text strings
        @param min_freq: Minimum frequency for token to be included
        """
        print(f"Building vocabulary for tokenizer: {self.model_name}...")
        # Count token frequencies
        token_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            token_counts.update(tokens)
        # Add tokens with sufficient frequency
        for token, count in token_counts.items():
            if count >= min_freq and token not in self.vocab:
                assert len(self.vocab) == len(self.id2token), "Vocabulary and ID mapping must be consistent"
                token_id: int = self.vocab_size
                self.vocab[token] = token_id
                self.id2token[token_id] = token
        print(f"Vocabulary size: {self.vocab_size}")

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert token IDs to tokens"""
        return [self.id2token.get(id, self.unk_token) for id in ids]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to token IDs"""
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def encode(self, text: str, add_special_tokens: bool = False, padding: str = True,
                truncation: bool = False) -> List[int]:
        """
        Encode text to token IDs
        @param text: Input text
        @param add_special_tokens: Whether to add BOS/EOS tokens
        @param padding: Padding strategy ('max_length' or None)
        @param truncation: Whether to truncate long sequences
        @return: Dictionary with 'input_ids' key
        """
        # Tokenize the text
        tokens: List[str] = self._tokenize(text)
        token_ids: List[int] = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        # Deal with special tokens
        if add_special_tokens:
            token_ids = [self.sos_token_id] + token_ids + [self.eos_token_id]
        # Deal with truncation
        if truncation == True and len(token_ids) > self.max_seq_len:
            if add_special_tokens: token_ids = token_ids[:self.max_seq_len - 1] + [self.eos_token_id]
            else: token_ids = token_ids[:self.max_seq_len]
        # Deal with padding
        if padding == True:
            token_ids += [self.pad_token_id] * (self.max_seq_len - len(token_ids))
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True,
                clean_up_tokenization_spaces: bool = True) -> str:
        """
        Decode token IDs back to text
        @param token_ids: List of token IDs
        @param skip_special_tokens: Whether to skip special tokens
        @param clean_up_tokenization_spaces: Whether to clean up spaces
        @return: Decoded text string
        """
        tokens: List[str] = []
        for token_id in token_ids:
            # convert token ID to token
            if token_id in self.id2token: token = self.id2token[token_id]
            else: token = self.unk_token  # Use UNK token for unknown IDs
            # Deal with special tokens
            if skip_special_tokens == True and token in self.special_tokens:
                continue
            tokens.append(token)
        # Join tokens with spaces
        text: str = " ".join(tokens)
        # Clean up tokenization spaces if requested
        if clean_up_tokenization_spaces:
            text = text.strip()
        # Return the cleaned-up text
        return text






