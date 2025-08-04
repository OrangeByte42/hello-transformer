import os
import datasets
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from typing import Any, List, Tuple
from src.utils.spacy_tokenizer import SpacyTokenizer


class DataLoader4Multi30k:
    """Loader for Multi30k dataset"""
    def __init__(self: Any, dataset_name: str, tokenizer_en: str, tokenizer_de: str,
                    max_seq_len: int, batch_size: int, padding: str = "max_length",
                    dataset_cache_dir: str = os.path.join(".", "data", "multi30k")) -> None:
        """constructor
        @param dataset_name: name of the dataset
        @param tokenizer_en: English tokenizer model name
        @param tokenizer_de: German tokenizer model name
        @param max_seq_len: maximum sequence length, which is used to pad the sequences
        @param batch_size: batch size for DataLoader
        @param device: device to use for the DataLoader
        """
        self.dataset: datasets.DatasetDict = load_dataset(dataset_name, cache_dir=dataset_cache_dir)
        self.max_seq_len: int = max_seq_len
        self.batch_size: int = batch_size
        self.padding: str = padding

        self.tokenizer_de: SpacyTokenizer = SpacyTokenizer(tokenizer_de, max_seq_len)
        self.tokenizer_en: SpacyTokenizer = SpacyTokenizer(tokenizer_en, max_seq_len)

        # Build vocabularies from the dataset
        de_texts = list(self.dataset["train"]["de"]) + list(self.dataset["validation"]["de"])
        en_texts = list(self.dataset["train"]["en"]) + list(self.dataset["validation"]["en"])
        self.tokenizer_de.build_vocabulary(de_texts)
        self.tokenizer_en.build_vocabulary(en_texts)

    def _tokenize_de(self: Any, text: str) -> Any:
        """tokenize German text
        @param text: German text
        @return: tokenized text
        """
        return self.tokenizer_de.encode(text, padding=True, truncation=True, add_special_tokens=True)

    def _tokenize_en(self: Any, text: str) -> Any:
        """tokenize English text
        @param text: English text
        @return: tokenized text
        """
        return self.tokenizer_en.encode(text, padding=True, truncation=True, add_special_tokens=True)

    def _create_tensor_dataset(self: Any, src_data: Any, trg_data: Any) -> TensorDataset:
        """Create a TensorDataset from source and target data"""
        # Convert to tensors
        # src_inputs_ids: List[Any] = [item["input_ids"] for item in src_data]
        # trg_inputs_ids: List[Any] = [item["input_ids"] for item in trg_data]
        src_tensor: torch.Tensor = torch.tensor(src_data)
        trg_tensor: torch.Tensor = torch.tensor(trg_data)
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

        # Create DistributedSampler
        train_sampler: DistributedSampler = DistributedSampler(train_data, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        val_sampler: DistributedSampler = DistributedSampler(val_data, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
        test_sampler: DistributedSampler = DistributedSampler(test_data, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)

        # Create DataLoaders
        train_loader: DataLoader = DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler)
        val_loader: DataLoader = DataLoader(val_data, batch_size=self.batch_size, sampler=val_sampler)
        test_loader: DataLoader = DataLoader(test_data, batch_size=self.batch_size, sampler=test_sampler)

        # Return DataLoaders
        return train_loader, val_loader, test_loader









