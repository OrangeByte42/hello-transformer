import os
import torch
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, List, Tuple


class DataLoader4Multi30k:
    """Loader for Multi30k dataset"""
    def __init__(self: Any, dataset_name: str, tokenizer_en: str, tokenizer_de: str,
                    max_seq_len: int, batch_size: int, padding: str = "max_length",
                    dataset_cache_dir: str = os.path.join(".", "data", "multi30k"),
                    tokenizer_en_cache_dir: str = os.path.join(".", "data", "tokenizer_en"),
                    tokenizer_de_cache_dir: str = os.path.join(".", "data", "tokenizer_de")) -> None:
        """constructor
        @param dataset_name: name of the dataset
        @param tokenizer_en: English tokenizer
        @param tokenizer_de: German tokenizer
        @param max_seq_len: maximum sequence length, which is used to pad the sequences
        @param batch_size: batch size for DataLoader
        @param device: device to use for the DataLoader
        """
        self.dataset: datasets.DatasetDict = load_dataset(dataset_name, cache_dir=dataset_cache_dir)
        self.tokenizer_de: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_de, cache_dir=tokenizer_de_cache_dir)
        self.tokenizer_en: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_en, cache_dir=tokenizer_en_cache_dir)
        self.max_seq_len: int = max_seq_len
        self.batch_size: int = batch_size
        self.padding: str = padding

    def _tokenize_de(self: Any, text: str) -> Any:
        """tokenize German text
        @param text: German text
        @return: tokenized text
        """
        return self.tokenizer_de(text, padding=self.padding, truncation=True, max_length=self.max_seq_len)

    def _tokenize_en(self: Any, text: str) -> Any:
        """tokenize English text
        @param text: English text
        @return: tokenized text
        """
        return self.tokenizer_en(text, padding=self.padding, truncation=True, max_length=self.max_seq_len)

    def _create_tensor_dataset(self: Any, src_data: Any, trg_data: Any) -> TensorDataset:
        """Create a TensorDataset from source and target data"""
        # Convert to tensors
        src_inputs_ids: List[Any] = [item["input_ids"] for item in src_data]
        trg_inputs_ids: List[Any] = [item["input_ids"] for item in trg_data]
        src_tensor: torch.Tensor = torch.tensor(src_inputs_ids)
        trg_tensor: torch.Tensor = torch.tensor(trg_inputs_ids)
        # Return TensorDataset
        return TensorDataset(src_tensor, trg_tensor)

    def load(self: Any) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load the Multi30k dataset and return DataLoaders for training, validation, and test sets
        @return: DataLoaders for training, validation, and test sets
        """
        # Tokenize the datasets
        train_de_tokenized = [self._tokenize_de(text) for text in self.dataset["train"]["de"]]
        train_en_tokenized = [self._tokenize_en(text) for text in self.dataset["train"]["en"]]

        val_de_tokenized = [self._tokenize_de(text) for text in self.dataset["validation"]["de"]]
        val_en_tokenized = [self._tokenize_en(text) for text in self.dataset["validation"]["en"]]

        test_de_tokenized = [self._tokenize_de(text) for text in self.dataset["test"]["de"]]
        test_en_tokenized = [self._tokenize_en(text) for text in self.dataset["test"]["en"]]
        # Create TensorDatasets
        train_data: TensorDataset = self._create_tensor_dataset(train_de_tokenized, train_en_tokenized)
        val_data: TensorDataset = self._create_tensor_dataset(val_de_tokenized, val_en_tokenized)
        test_data: TensorDataset = self._create_tensor_dataset(test_de_tokenized, test_en_tokenized)
        # Create DataLoaders
        train_loader: DataLoader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader: DataLoader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        test_loader: DataLoader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        # Return DataLoaders
        return train_loader, val_loader, test_loader









