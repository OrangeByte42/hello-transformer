import spacy
from collections import Counter
from typing import Any, List, Dict, Set

from src.utils.tokenizer.tokenizer import Tokenizer

class SpacyTokenizer(Tokenizer):
    """SpaCy-based tokenizer for text preprocessing"""

    def __init__(self: Any, model_name: str, max_seq_len: int) -> None:
        """Initialize SpaCy tokenizer
        @param model_name: SpaCy model name (e.g., 'en_core_web_sm', 'de_core_news_sm')
        @param max_seq_len: Maximum sequence length for padding/truncation
        """
        super(SpacyTokenizer, self).__init__(model_name, max_seq_len)  # Call parent constructor

        # Store model name and max sequence length
        self.model_name: str = model_name
        self.max_seq_len: int = max_seq_len
        self.tokenizer: Any = spacy.load(model_name)    # Load SpaCy model

        # Special tokens
        self._pad_token: str = "<pad>"
        self._unk_token: str = "<unk>"
        self._sos_token: str = "<sos>"  # Start of sequence token
        self._eos_token: str = "<eos>"  # End of sequence token

        self.special_tokens: Set[str] = {
            self._pad_token,
            self._unk_token,
            self._sos_token,
            self._eos_token,
        }

        # Special token IDs
        self._pad_token_id: int = 0  # Padding token ID
        self._unk_token_id: int = 1  # Unknown token ID
        self._sos_token_id: int = 2  # Start of sequence token ID
        self._eos_token_id: int = 3  # End of sequence token ID

        # Initialize vocabulary with special tokens
        self.vocab: Dict[str, int] = {
            self._pad_token: self._pad_token_id,
            self._unk_token: self._unk_token_id,
            self._sos_token: self._sos_token_id,
            self._eos_token: self._eos_token_id,
        }

        self.id2token: Dict[int, str] = {v:k for k, v in self.vocab.items()}

    def _tokenize(self: Any, text: str) -> List[str]:
        """Tokenize text using SpaCy
        @param text: Input text string
        @return: List of tokens
        """
        tokens: List[str] = [tok.text for tok in self.tokenizer(text.lower().strip())]
        return tokens

    def build_vocab(self: Any, texts: List[str], min_freq: int = 0) -> None:
        """Build vocabulary from a list of texts
        @param texts: List of text strings to build vocabulary from
        @param min_freq: Minimum frequency for a token to be included in the vocabulary
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
            token_id: int = self.vocab_size
            self.vocab[token] = token_id
            self.id2token[token_id] = token

    def convert_ids_to_tokens(self: Any, ids: List[int], skip_special_tokens: bool) -> List[str]:
        """Convert a list of token IDs to tokens
        @param ids: List of token IDs to convert
        @param skip_special_tokens: Whether to skip special tokens in the output
        @return: List of tokens corresponding to the input IDs
        """
        # Initialize list to hold tokens
        tokens: List[str] = list()

        # Convert each ID to its corresponding token
        for token_id in ids:
            token: str = self.id2token.get(token_id, self._unk_token)
            if skip_special_tokens == True and token in self.special_tokens:
                continue
            tokens.append(token)

        # Return list of tokens
        return tokens

    def convert_tokens_to_ids(self: Any, tokens: List[str], skip_speicial_tokens: bool) -> List[int]:
        """Convert a list of tokens to token IDs
        @param tokens: List of tokens to convert
        @param skip_special_tokens: Whether to skip special tokens in the input
        @return: List of token IDs corresponding to the input tokens
        """
        # Initialize list to hold token IDs
        token_ids: List[int] = list()

        # Convert each token to its corresponding ID
        for token in tokens:
            if skip_speicial_tokens == True and token in self.special_tokens:
                continue
            token_id: int = self.vocab.get(token, self._unk_token_id)
            token_ids.append(token_id)

        # Return list of token IDs
        return token_ids

    def encode(self: Any, text: str, add_special_tokens: bool = True, padding: bool = True,
                truncation: bool = True) -> List[int]:
        """Encode a text string into a list of token IDs
        @param text: Input text string
        @param add_special_tokens: Whether to add special tokens (sos, eos)
        @param padding: Whether to pad the sequence to max_seq_len
        @param truncation: Whether to truncate the sequence to max_seq_len
        """
        # Tokenize the text
        tokens: List[str] = self._tokenize(text)
        token_ids: List[int] = [self.vocab.get(token, self._unk_token_id) for token in tokens]

        # Deal with special tokens
        if add_special_tokens == True:
            token_ids = [self._sos_token_id] + token_ids + [self._eos_token_id]

        # Deal with truncation
        if truncation == True and len(token_ids) > self.max_seq_len:
            if add_special_tokens:
                token_ids = token_ids[:self.max_seq_len - 1] + [self._eos_token_id]
            else:
                token_ids = token_ids[:self.max_seq_len]

        # Deal with padding
        if padding == True:
            token_ids += [self._pad_token_id] * (self.max_seq_len - len(token_ids))

        return token_ids

    def decode(self: Any, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back into a text string
        @param token_ids: List of token IDs to decode
        @param skip_special_tokens: Whether to skip special tokens in the output
        @return: Decoded text string
        """
        # Convert token IDs to tokens
        tokens: List[str] = list()
        for token_id in token_ids:
            token: str = self.id2token.get(token_id, self._unk_token)
            tokens.append(token)

        if skip_special_tokens == True:
            tokens = [token for token in tokens if token not in self.special_tokens]

        # Join tokens with spaces
        text: str = " ".join(tokens).strip()

        return text

