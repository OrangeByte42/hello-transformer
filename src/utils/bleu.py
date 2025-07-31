import math
import numpy as np
from typing import Any, List
from collections import Counter


def _bleu_stats(hypothesis_seq: List[str], reference_seq: List[str], max_n_gram: int = 4) -> List[int]:
    """Calculate BLEU statistics for n-grams up to max_n_gram.
    @param hypothesis_seq: List of tokens in the hypothesis sequence.
    @param reference_seq: List of tokens in the reference sequence.
    @param max_n_gram: Calculate BLEU statistics for [1, max_n_gram] n-grams.
    @return: List of statistics: [hypothesis_length, reference_length, clipped_ngram_count, total_ngram_count]
    """
    stats: List[int] = [len(hypothesis_seq), len(reference_seq)]
    for n in range(1, max_n_gram + 1):
        hypothesis_ngrams: Counter = Counter([tuple(hypothesis_seq[i:i + n]) for i in range(len(hypothesis_seq) + 1 - n)])
        reference_ngrams: Counter = Counter([tuple(reference_seq[i:i + n]) for i in range(len(reference_seq) + 1 - n)])
        stats.append(max([sum((hypothesis_ngrams & reference_ngrams).values()), 0]))
        stats.append(max([len(hypothesis_seq) + 1 - n, 0]))
    return stats

def _bleu_precision(stats: List[int], max_n_gram: int = 4) -> float:
    """Calculate BLEU precision from statistics.
    @param stats: List of statistics: [hypothesis_length, reference_length, clipped_ngram_count, total_ngram_count]
    @return: BLEU precision score
    """
    if any(stat == 0 for stat in stats): return 0.0
    hypothesis_length, reference_length = stats[:2]
    log_bleu_precision: float = sum([math.log(float(x) / float(y)) for x, y in zip(stats[2::2], stats[3::2])]) / max_n_gram
    bleu_precision: float = math.exp(min([0, 1 - float(hypothesis_length) / float(reference_length)]) + log_bleu_precision)
    return bleu_precision

def bleu(hypothesis_seqs: List[List[str]], reference_seqs: List[List[str]], max_n_gram: int = 4) -> float:
    """Calculate BLEU score for a list of hypothesis and reference sequences.
    @param hypothesis_seqs: List of hypothesis sequences, each sequence is a list of tokens.
    @param reference_seqs: List of reference sequences, each sequence is a list of tokens.
    @param max_n_gram: Calculate BLEU score for [1, max_n_gram] n-grams.
    @return: BLEU score
    """
    stats: List[int] = np.array([0.0] * (2 + 2 * max_n_gram))
    for hypothesis_seq, reference_seq in zip(hypothesis_seqs, reference_seqs):
        stats += np.array(_bleu_stats(hypothesis_seq, reference_seq, max_n_gram))
    return 100 * _bleu_precision(stats, max_n_gram)

def idx2word(idxs: List[int], tokenizer: Any) -> str:
    """Convert a list of indices to a string of words.
    @param idxs: List of indices.
    @param tokenizer: Tokenizer object with convert_ids_to_tokens method.
    @return: String of words corresponding to the indices.
    """
    words: List[str] = list()
    tokens = tokenizer.convert_ids_to_tokens(idxs)
    for token in tokens:
        # Skip special tokens like [CLS], [SEP], [PAD], etc.
        if not (token.startswith('[') and token.endswith(']')) and token != '<unk>':
            words.append(token)
    words: str = ' '.join(words)
    return words





