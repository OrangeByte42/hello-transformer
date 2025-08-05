from abc import ABC, abstractmethod
from typing import Any, List, Dict


class Tokenizer(ABC):
    """Abstract base class for tokenizers, provides common interface"""

    @abstractmethod
    def __init__(self: Any, model_name: str, max_seq_len: int) -> None:
        """Intialize tokenizer with model name and max sequence length"""
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
        """Build vocabulary from a list of texts"""
        ...


    @abstractmethod
    def convert_ids_to_tokens(self: Any, ids: List[int], skip_special_tokens: bool) -> List[str]:
        """Convert a list of token IDs to tokens"""
        ...

    @abstractmethod
    def convert_tokens_to_ids(self: Any, tokens: List[str], skip_special_tokens: bool) -> List[int]:
        """Convert a list of tokens to token IDs"""
        ...

    @abstractmethod
    def encode(self: Any, text: str, add_special_tokens: bool = True, padding: bool = True,
                truncation: bool = True) -> List[int]:
        """Encode a text string into a list of token IDs"""
        ...

    @abstractmethod
    def decode(self: Any, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back into a text string"""
        ...

    @staticmethod
    def load(model_name: str, max_seq_len: int) -> 'Tokenizer':
        """Factory method to load the appropriate tokenizer based on model name"""
        if model_name in ["en_core_web_sm", "de_core_news_sm"]:
            from src.utils.tokenizer.spacy_tokenizer import SpacyTokenizer
            return SpacyTokenizer(model_name, max_seq_len)
        elif model_name in ["bert-base-uncased", "bert-base-german-dbmdz-cased"]:
            from src.utils.tokenizer.bert_tokenizer import BertTokenizer
            return BertTokenizer(model_name, max_seq_len)
        else:
            raise ValueError(f"Unsupported tokenizer model: {model_name}")







