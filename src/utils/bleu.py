# -*- coding: utf-8 -*-
import math
import numpy as np
from typing import List, Dict
from collections import Counter

def _get_ngrams(segment: List[str], max_n_gram: int) -> Dict[int, Counter]:
    """从一个句子中提取所有 n-grams (从 1 到 max_n_gram)"""
    ngrams_by_n = {}
    for n in range(1, max_n_gram + 1):
        # 如果句子长度小于 n，则无法提取 n-gram
        if len(segment) >= n:
            ngrams = [tuple(segment[i:i+n]) for i in range(len(segment) - n + 1)]
            ngrams_by_n[n] = Counter(ngrams)
        else:
            ngrams_by_n[n] = Counter()
    return ngrams_by_n

def bleu(
    hypothesis_corpus: List[List[str]],
    references_corpus: List[List[List[str]]],
    max_n_gram: int = 4,
    smoothing: bool = False
) -> float:
    """
    计算语料库级别的 BLEU 分数。

    @param hypothesis_corpus: 假设句子列表，每个句子是一个 token 列表。
    @param references_corpus: 参考句子列表，每个元素是对应假设的多个参考列表。
                               e.g., [[['ref1a', '...'], ['ref1b', ...]], [['ref2a', ...]]]
    @param max_n_gram: 计算 BLEU 分数的最大 n-gram。
    @param smoothing: 是否使用加1平滑 (Add-1 smoothing)。
    @return: BLEU 分数 (0-100)
    """
    # 检查输入是否匹配
    if len(hypothesis_corpus) != len(references_corpus):
        raise ValueError("The number of hypotheses and reference sets must be equal.")

    # 初始化累加统计量
    clipped_counts = np.zeros(max_n_gram)
    total_counts = np.zeros(max_n_gram)
    total_hyp_len = 0
    total_ref_len = 0

    # 遍历语料库中的每个句子
    for hypothesis, references in zip(hypothesis_corpus, references_corpus):
        hyp_len = len(hypothesis)
        total_hyp_len += hyp_len

        # 1. 计算有效参考长度 (Effective Reference Length)
        # 选择与假设长度最接近的参考长度
        ref_lengths = [len(ref) for ref in references]
        closest_ref_len = min(
            ref_lengths,
            key=lambda ref_len: (abs(ref_len - hyp_len), ref_len)
        )
        total_ref_len += closest_ref_len

        # 2. 计算 n-gram 统计
        hyp_ngrams_by_n = _get_ngrams(hypothesis, max_n_gram)
        
        # 为了计算 clipped count，我们需要所有参考中 n-gram 的最大出现次数
        max_ref_counts_by_n = {}
        for n in range(1, max_n_gram + 1):
            max_ref_counts_by_n[n] = Counter()
        
        for reference in references:
            ref_ngrams_by_n = _get_ngrams(reference, max_n_gram)
            for n in range(1, max_n_gram + 1):
                for ngram, count in ref_ngrams_by_n[n].items():
                    max_ref_counts_by_n[n][ngram] = max(max_ref_counts_by_n[n][ngram], count)

        # 3. 累加 clipped counts 和 total counts
        for n in range(1, max_n_gram + 1):
            # 累加 clipped counts
            for ngram, count in hyp_ngrams_by_n[n].items():
                clipped_counts[n-1] += min(count, max_ref_counts_by_n[n][ngram])
            
            # 累加 total counts
            # 正确的计算方式是句子中的 n-gram 总数，而不是唯一数量
            if len(hypothesis) >= n:
                total_counts[n-1] += len(hypothesis) - n + 1

    # 如果假设语料库为空，BLEU 为 0
    if total_hyp_len == 0:
        return 0.0

    # 4. 计算精度 (Precisions)
    precisions = np.zeros(max_n_gram)
    for n in range(max_n_gram):
        if smoothing:
            # Add-1 Smoothing
            precisions[n] = (clipped_counts[n] + 1) / (total_counts[n] + 1)
        else:
            if total_counts[n] > 0:
                precisions[n] = clipped_counts[n] / total_counts[n]
            else:
                # 如果分母为0，精度也为0
                precisions[n] = 0.0

    # 5. 计算几何平均值 (Geometric Mean of Precisions)
    log_precisions_sum = 0
    # 权重通常是均匀的
    weights = [1.0 / max_n_gram] * max_n_gram
    
    # 为了避免 log(0)，我们只对 > 0 的精度求和
    # 如果任何一个 pn=0，未平滑的 BLEU 将为 0
    non_zero_precisions = False
    for i, p in enumerate(precisions):
        if p > 0:
            log_precisions_sum += weights[i] * math.log(p)
            non_zero_precisions = True

    if not non_zero_precisions:
        return 0.0

    geometric_mean = math.exp(log_precisions_sum)
    
    # 6. 计算简短惩罚 (Brevity Penalty)
    bp = 1.0
    if total_hyp_len < total_ref_len:
        bp = math.exp(1 - total_ref_len / total_hyp_len)

    # 7. 计算最终 BLEU 分数
    bleu_score = bp * geometric_mean

    return bleu_score * 100