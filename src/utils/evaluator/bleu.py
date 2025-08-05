import math
import numpy as np
from collections import Counter
from typing import Any, List, Tuple, Dict


class BLEUSccoreEvaluator:
    """Evaluartor for BLEU score calculation"""

    def __init__(self: Any, max_n_gram: int = 4, smoothing: bool = False) -> None:
        """Initialize BLEU evaluator
        @param max_n_gram: Maximum n-gram size to consider for BLEU score
        @param smoothing: Whether to apply smoothing to the BLEU score calculation
        """
        self.max_n_gram: int = max_n_gram
        self.smoothing: bool = smoothing

    def _get_ngrams_stats(self: Any, segments: List[str]) -> Dict[int, Counter]:
        """Extract all n-grams from a tokenized segment from 1 to max_n_gram
        @param segments: Tokenized segment
        @return: Dictionary of n-grams by their size
        """
        ngrams_stats: Dict[int, Counter] = {}
        for n in range(1, self.max_n_gram + 1):
            ngrams: List[Tuple[str]] = [tuple(segments[i:i+n]) for i in range(len(segments) - n + 1)]
            ngrams_stats[n] = Counter(ngrams)
        return ngrams_stats

    def calculate_bleu(self: Any, hypothesis_corpus: List[List[str]], references_corpus: List[List[List[str]]]) -> float:
        """Calculate BLEU score for a corpus of hypotheses against multiple references
        @param hypothesis_corpus: List of hypotheses, each hypothesis is a list of tokens
        @param references_corpus: List of references, each reference is a list of lists of tokens (multiple references per hypothesis)
        @return: BLEU score whose range is [0, 100]
        """
        assert len(hypothesis_corpus) == len(references_corpus), "Hypotheses and references must have the same length."

        # Initialize statistics variables
        clipped_counts: np.ndarray = np.zeros(self.max_n_gram)
        total_counts: np.ndarray = np.zeros(self.max_n_gram)
        total_hyp_len: int = 0
        total_ref_len: int = 0

        # Iterate over each hypothesis and its references in total corpus
        for hypothesis, references in zip(hypothesis_corpus, references_corpus):
            # Update total hypothesis length
            hyp_len: int = len(hypothesis)
            total_hyp_len += hyp_len

            # Update total reference length
            # Calculate effective reference length, choosing the closest reference length
            ref_lengths: List[int] = [len(ref) for ref in references]
            closet_ref_len: int = min(ref_lengths, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))
            total_ref_len += closet_ref_len

            # Calculate n-gram statistics for hypothesis [count]
            hyp_ngrams_stats: Dict[int, Counter] = self._get_ngrams_stats(hypothesis)

            # Calculate n-gram statistics for references [max-ref-count]
            max_ref_counts: Dict[int, Counter] = {n:Counter() for n in range(1, self.max_n_gram + 1)}
            for reference in references:
                ref_ngrams_stats: Dict[int, Counter] = self._get_ngrams_stats(reference)
                for n in range(1, self.max_n_gram + 1):
                    for ngram, count in ref_ngrams_stats[n].items():
                        max_ref_counts[n][ngram] = max(max_ref_counts[n][ngram], count)

            # Calculate clipped counts & total counts
            for n in range(1, self.max_n_gram + 1):
                # calculate clipped counts
                for ngram, count in hyp_ngrams_stats[n].items():
                    clipped_counts[n - 1] += min(count, max_ref_counts[n].get(ngram, 0))
                # calculate total counts
                total_counts[n - 1] += len(hypothesis) - n + 1 if hyp_len >= n else 0

        # If total counts are zero, return 0.0
        if total_hyp_len == 0: return 0.0

        # Calculate precision for each n-gram
        precisions: np.ndarray = np.zeros(self.max_n_gram)
        for n in range(self.max_n_gram):
            if self.smoothing:
                precisions[n] = (clipped_counts[n] + 1) / (total_counts[n] + 1)
            else:
                precisions[n] = (clipped_counts[n] / total_counts[n]) if total_counts[n] > 0 else 0.0

        # Calculate geometric mean of precisions
        log_precisions_sum: float = 0.0
        weights: List[float] = [1.0 / self.max_n_gram] * self.max_n_gram    # use average weights

        non_zero_precisions: bool = False
        for idx, precision in enumerate(precisions):
            if precision > 0:
                log_precisions_sum += weights[idx] * math.log(precision)
                non_zero_precisions = True

        if non_zero_precisions == False: return 0.0
        geometric_mean: float = math.exp(log_precisions_sum)    # for log-sum, avoid underflow

        # Calculate brevity penalty
        brevity_penalty: float = 1.0
        if total_hyp_len < total_ref_len:
            brevity_penalty = math.exp(1 - total_ref_len / total_hyp_len)

        # Calculate final BLEU score
        bleu_score: float = brevity_penalty * geometric_mean

        # Return BLEU score as percentage
        return bleu_score * 100.0






















