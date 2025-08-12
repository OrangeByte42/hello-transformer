from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional


class Tokenizer(ABC):
    """Abstract base class for tokenizers, provides common interface"""

    @abstractmethod
    def __init__(self: Any, model_name: str, max_seq_len: int,
                    cache_dir: Optional[str] = None) -> None:
        """Intialize tokenizer with model name and max sequence length
        @param model_name: Name of the tokenizer model (e.g., 'en_core_web_sm', 'bert-base-uncased')
        @param max_seq_len: Maximum sequence length for padding/truncation
        @param cache_dir: Optional directory to cache the tokenizer model
        """
        # Special tokens
        self._pad_token: str = None
        self._unk_token: str = None
        self._sos_token: str = None
        self._eos_token: str = None

        # Special token IDs
        self._pad_token_id: int = None
        self._unk_token_id: int = None
        self._sos_token_id: int = None
        self._eos_token_id: int = None

        # Vocabulary
        self.vocab: Dict[str, int] = None

    # properties
    @property
    def vocab_size(self: Any) -> int:
        """Return the size of the vocabulary"""
        return len(self.vocab)

    @property
    def pad_token(self: Any) -> str:
        """Return the padding token"""
        return self._pad_token

    @property
    def unk_token(self: Any) -> str:
        """Return the unknown token"""
        return self._unk_token

    @property
    def sos_token(self: Any) -> str:
        """Return the start-of-sequence token"""
        return self._sos_token

    @property
    def eos_token(self: Any) -> str:
        """Return the end-of-sequence token"""
        return self._eos_token

    @property
    def pad_token_id(self: Any) -> int:
        """Return the ID of the padding token"""
        return self._pad_token_id

    @property
    def unk_token_id(self: Any) -> int:
        """Return the ID of the unknown token"""
        return self._unk_token_id

    @property
    def sos_token_id(self: Any) -> int:
        """Return the ID of the start-of-sequence token"""
        return self._sos_token_id

    @property
    def eos_token_id(self: Any) -> int:
        """Return the ID of the end-of-sequence token"""
        return self._eos_token_id

    # methods
    @abstractmethod
    def build_vocab(self: Any, texts: List[str], min_freq: int = 0) -> None:
        """Build vocabulary from a list of texts
        @param texts: List of text strings to build vocabulary from
        @param min_freq: Minimum frequency for a token to be included in the vocabulary
        """
        ...


    @abstractmethod
    def convert_ids_to_tokens(self: Any, ids: List[int], skip_special_tokens: bool) -> List[str]:
        """Convert a list of token IDs to tokens
        @param ids: List of token IDs to convert
        @param skip_special_tokens: Whether to skip special tokens in the output
        @return: List of tokens corresponding to the input IDs
        """
        ...

    @abstractmethod
    def convert_tokens_to_ids(self: Any, tokens: List[str], skip_special_tokens: bool) -> List[int]:
        """Convert a list of tokens to token IDs
        @param tokens: List of tokens to convert
        @param skip_special_tokens: Whether to skip special tokens in the input
        @return: List of token IDs corresponding to the input tokens
        """
        ...

    @abstractmethod
    def encode(self: Any, text: str, add_special_tokens: bool = True, padding: bool = True,
                truncation: bool = True) -> List[int]:
        """Encode a text string into a list of token IDs
        @param text: Input text string
        @param add_special_tokens: Whether to add special tokens (BOS/EOS)
        @param padding: Whether to pad the sequence to max_seq_len
        @param truncation: Whether to truncate the sequence to max_seq_len
        @return: List of token IDs
        """
        ...

    @abstractmethod
    def decode(self: Any, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back into a text string
        @param token_ids: List of token IDs to decode
        @param skip_special_tokens: Whether to skip special tokens in the output
        @return: Decoded text string
        """
        ...

    @staticmethod
    def load(model_name: str, max_seq_len: int,
                cache_dir: Optional[str] = None) -> 'Tokenizer':
        """Factory method to load the appropriate tokenizer based on model name
        @param model_name: Name of the tokenizer model to load
        @param max_seq_len: Maximum sequence length for the tokenizer
        @param cache_dir: Optional directory to cache the tokenizer model
        @return: An instance of a Tokenizer subclass
        """
        if model_name in ["en_core_web_sm", "de_core_news_sm"]:
            from src.data.tokenizer.spacy_tokenizer import SpacyTokenizer
            return SpacyTokenizer(model_name, max_seq_len, cache_dir)
        elif model_name in ["bert-base-uncased", "bert-base-german-dbmdz-cased"]:
            from src.data.tokenizer.bert_tokenizer import BertTokenizer
            return BertTokenizer(model_name, max_seq_len, cache_dir)
        else:
            raise ValueError(f"Unsupported tokenizer model: {model_name}")

