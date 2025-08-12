import os
import datasets
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from typing import Any, Tuple, List, Optional

from src.data.tokenizer.tokenizer import Tokenizer


class Multi30kDataLoader:
    """DataLoader for Multi30k dataset"""
    def __init__(self: Any, dataset_name: str, de_tokenizer: str, en_tokenizer: str,
                    max_seq_len: int, batch_size: int, ddp: bool,
                    dataset_cache_dir: Optional[str] = None,
                    tokenizer_cache_dir: Optional[str] = None) -> None:
        """Constructor
        @param dataset_name: name of the dataset
        @param tokenizer_de: German tokenizer model name
        @param tokenizer_en: English tokenizer model name
        @param max_seq_len: maximum sequence length for padding
        @param batch_size: batch size for DataLoader
        @param ddp: whether to use Distributed Data Parallel (DDP)
        @param dataset_cache_dir: directory to cache the dataset
        @param tokenizer_cache_dir: directory to cache the tokenizers
        """
        # Load the dataset
        self.dataset: datasets.DatasetDict = load_dataset(dataset_name, cache_dir=dataset_cache_dir)
        self.max_seq_len: int = max_seq_len
        self.batch_size: int = batch_size
        self.ddp: bool = ddp

        # Load tokenizers
        self._de_tokenizer: Tokenizer = Tokenizer.load(model_name=de_tokenizer, max_seq_len=max_seq_len,
                                                        cache_dir=tokenizer_cache_dir)
        self._en_tokenizer: Tokenizer = Tokenizer.load(model_name=en_tokenizer, max_seq_len=max_seq_len,
                                                        cache_dir=tokenizer_cache_dir)

        # Build vocabularies from the dataset
        de_texts = list(self.dataset["train"]["de"]) + list(self.dataset["validation"]["de"]) + list(self.dataset["test"]["de"])
        en_texts = list(self.dataset["train"]["en"]) + list(self.dataset["validation"]["en"]) + list(self.dataset["test"]["en"])

        self._de_tokenizer.build_vocab(texts=de_texts, min_freq=0)
        self._en_tokenizer.build_vocab(texts=en_texts, min_freq=0)

    @property
    def de_tokenizer(self: Any) -> Tokenizer:
        """Get the German tokenizer"""
        return self._de_tokenizer

    @property
    def en_tokenizer(self: Any) -> Tokenizer:
        """Get the English tokenizer"""
        return self._en_tokenizer

    def _de_tokenize(self: Any, text: str) -> List[int]:
        """Tokenize German text
        @param text: German text
        @return: tokenized text
        """
        return self._de_tokenizer.encode(text, add_special_tokens=True, padding=True, truncation=True)

    def _en_tokenize(self: Any, text: str) -> List[int]:
        """Tokenize English text
        @param text: English text
        @return: tokenized text
        """
        return self._en_tokenizer.encode(text, add_special_tokens=True, padding=True, truncation=True)

    def _create_tensor_dataset(self: Any, src_data: List[List[int]], trg_data: List[List[int]]) -> TensorDataset:
        """Create a TensorDataset from source and target data"""
        # Convert to tensors
        src_tensor: torch.Tensor = torch.tensor(src_data)
        trg_tensor: torch.Tensor = torch.tensor(trg_data)
        # Return TensorDataset
        return TensorDataset(src_tensor, trg_tensor)

    def load(self: Any) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load the Multi30k dataset and return DataLoaders for train, validation and test sets
        @return: Tuple of DataLoaders for train, validation and test sets
        """
        # Tokenize the datasets
        train_de_tokenized: List[List[int]] = [self._de_tokenize(text) for text in self.dataset["train"]["de"]]
        train_en_tokenized: List[List[int]] = [self._en_tokenize(text) for text in self.dataset["train"]["en"]]

        validation_de_tokenized: List[List[int]] = [self._de_tokenize(text) for text in self.dataset["validation"]["de"]]
        validation_en_tokenized: List[List[int]] = [self._en_tokenize(text) for text in self.dataset["validation"]["en"]]

        test_de_tokenized: List[List[int]] = [self._de_tokenize(text) for text in self.dataset["test"]["de"]]
        test_en_tokenized: List[List[int]] = [self._en_tokenize(text) for text in self.dataset["test"]["en"]]

        # Create TensorDatasets
        train_dataset: TensorDataset = self._create_tensor_dataset(train_de_tokenized, train_en_tokenized)
        validation_dataset: TensorDataset = self._create_tensor_dataset(validation_de_tokenized, validation_en_tokenized)
        test_dataset: TensorDataset = self._create_tensor_dataset(test_de_tokenized, test_en_tokenized)

        # Create DataLoaders
        if self.ddp:
            # Create DistributedSampler for DDP
            train_sampler: DistributedSampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
            validation_sampler: DistributedSampler = DistributedSampler(validation_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
            test_sampler: DistributedSampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)

            # Create DataLoaders with DistributedSampler
            train_loader: DataLoader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler)
            validation_loader: DataLoader = DataLoader(validation_dataset, batch_size=self.batch_size, sampler=validation_sampler)
            test_loader: DataLoader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=test_sampler)
        else:
            # Create DataLoaders without DistributedSampler
            train_loader: DataLoader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            validation_loader: DataLoader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader: DataLoader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Return DataLoaders
        return train_loader, validation_loader, test_loader

