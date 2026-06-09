"""
Theoretical Capacity of Hyperdimensional Binary Vectors
========================================================
Implements the exact capacity formulas from the paper
(Section "Theoretical limits on capacity of HBVs").

The paper derives the probability of a bit being 1 after consensus-sum
retrieval with n stored records, given a bias p from a matching entry:

    P(bit=1 | n, p) = (1−p) · Σ_{k=n/2}^{n−1} C(n−1,k) / 2^{n−1}
                     +  p  · Σ_{k=n/2−1}^{n−1} C(n−1,k) / 2^{n−1}

From this we compute:
    - Expected Hamming distance H_n(n, p) = P(bit=1 | n, p)
    - Statistical significance: z = (0.5 − H_n) / σ
    - Capacity limit: max n such that z ≥ 3 (3-sigma confidence)
    - Practical bound: ~700 records at D=10,000 (matches paper Fig 7)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


def binomial_prob(n_records: int, p_match: float) -> float:
    """Compute P(bit=1) after consensus-sum retrieval with n records.

    From the paper (Section "Theoretical limits on capacity of HBVs"):
        The memory m = Σ x_i ⊗ a_i is a consensus sum of n records.
        When testing for match d = a_j, one term produces 0 (the match)
        and the n−1 other terms produce random vectors.

        The result bit is 1 if the majority of these n terms is 1.
        For even n, ties are broken by random choice (0 or 1).

        P(bit=1) = P(clear majority for 1) + 0.5·P(tie)

    For odd n: no ties. Need at least ceil(n/2) random ones to
    outweigh the deterministic 0 (when p_match=0).

    For even n: ties at exactly n/2 ones → coin flip.
        P(bit=1) = tail(n_rand, n/2+1) + 0.5 · pmf_exact(n_rand, n/2)

    When p_match=0.5 (no match): there is no deterministic term
    pulling results toward 0; all n terms are random. By symmetry
    of the consensus sum, P(bit=1) = 0.5 exactly.

    Args:
        n_records: Number of records in the consensus sum
        p_match: Deterministic-term bias (0 = exact match, 0.5 = random)

    Returns:
        P(bit=1) ∈ [0, 1]; 0.5 = random, <0.5 = match signal
    """
    n = n_records
    if n < 1:
        return 0.5
    if n == 1:
        return p_match

    n_rand = n - 1

    if n % 2 == 1:
        # Odd n: no ties. Need at least ceil(n/2) = (n+1)//2 random 1s
        # to outweigh the deterministic 0 (or (n-1)//2 random 1s if det=1)
        need = (n + 1) // 2
        tail_det0 = _binomial_tail(n_rand, need)
        tail_det1 = _binomial_tail(n_rand, need - 1)
        return (1.0 - p_match) * tail_det0 + p_match * tail_det1
    else:
        # Even n: ties at exactly n/2 ones get broken randomly.
        # Clear majority: at least n/2+1 random ones.
        need_clear = n // 2 + 1
        need_tie = n // 2

        # P(clear majority) + 0.5 * P(exact tie)
        clear_prob = _binomial_tail(n_rand, need_clear)
        tie_prob = _binomial_exact(n_rand, need_tie)
        prob_det0 = clear_prob + 0.5 * tie_prob

        # When det=1: need at least n/2 random ones (tie + clear)
        # Clear majority now starts at n/2, and ties at n/2-1
        clear_prob_det1 = _binomial_tail(n_rand, need_tie)
        tie_prob_det1 = _binomial_exact(n_rand, need_tie - 1)
        prob_det1 = clear_prob_det1 + 0.5 * tie_prob_det1

        return (1.0 - p_match) * prob_det0 + p_match * prob_det1


def _binomial_exact(n: int, k: int) -> float:
    """P(X = k) for X ~ Binomial(n, 0.5)."""
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) / (2.0 ** n)


def _binomial_tail(n: int, a_input: int) -> float:
    """P(X ≥ a) for X ~ Binomial(n, 0.5). Non-recursive.

    Uses symmetry P(≥a) = 1 − P(≥n−a+1) when a > n/2,
    then computes the smaller tail directly.

    Args:
        n: Number of trials
        a_input: Minimum count for "success"

    Returns:
        Probability in [0, 1]
    """
    a = a_input
    if a <= 0:
        return 1.0
    if a > n:
        return 0.0

    # Symmetry: compute the complementary tail if a > n/2
    complement = False
    if a > n // 2:
        a = n - a + 1
        complement = True

    # Now a ≤ n/2, compute P(≥a) iteratively
    binom = math.comb(n, a)
    total = binom
    for k in range(a, n):
        binom = binom * (n - k) / (k + 1)
        total += binom

    prob = total / (2.0 ** n)
    return 1.0 - prob if complement else prob


def compute_hamming_statistics(
    n_records: int,
    p_match: float = 0.0,
    dim: int = 10_000,
) -> Dict[str, float]:
    """Compute expected Hamming distance and statistical significance.

    When a record exists in memory (p_match = 0.0), the expected
    Hamming distance deviates from the baseline 0.5 by:

        Δ = 0.5 − H_n(n, 0)
        σ = sqrt(0.5 * 0.5 / D) = sqrt(0.25 / D)
        z = Δ / σ

    A z-score ≥ 3 indicates a statistically significant match with
    >99.7% confidence.

    Args:
        n_records: Number of records stored in the memory
        p_match: Bias probability for a matching entry (0 = exact match)
        dim: Vector dimensionality

    Returns:
        Dict with H_n, delta, sigma, z_score, is_significant, and
        equivalent_sigma
    """
    p1 = binomial_prob(n_records, p_match)
    h_n = p1
    delta = 0.5 - h_n  # deviation below 0.5 (match pulls H_n downward)
    sigma = 0.5 / math.sqrt(dim)  # per-bit std dev: sqrt(p(1−p)/dim) with p=0.5
    z = delta / sigma if sigma > 0 else 0.0

    return {
        "H_n": h_n,
        "expected_distance": h_n,
        "deviation_from_random": delta,
        "sigma": sigma,
        "z_score": z,
        "is_significant_3sigma": z >= 3.0,
        "is_significant_5sigma": z >= 5.0,
        "n_records": n_records,
        "dim": dim,
    }


def capacity_curve(
    max_records: int = 1000,
    dim: int = 10_000,
    z_threshold: float = 3.0,
) -> List[Dict[str, float]]:
    """Compute capacity curve: how H_n deviates as n increases.

    This reproduces the curve from paper Figure 7 — as more records
    are packed into memory, the Hamming distance to the correct
    match approaches 0.5 (random), reducing statistical significance.

    Args:
        max_records: Maximum number of records to evaluate
        dim: Vector dimensionality
        z_threshold: Z-score threshold for "significant"

    Returns:
        List of statistics per n
    """
    curve = []
    for n in range(1, max_records + 1):
        stats = compute_hamming_statistics(n, p_match=0.0, dim=dim)
        curve.append(stats)
    return curve


def find_capacity_limit(
    dim: int = 10_000,
    z_threshold: float = 3.0,
    max_search: int = 10000,
) -> int:
    """Find the maximum number of records before statistical breakdown.

    Returns the largest n such that z ≥ z_threshold.
    For D=10,000 and 3-sigma, this should be approximately 700-800,
    matching the paper's finding of "~700 records at D=10,000."

    Args:
        dim: Vector dimensionality
        z_threshold: Required z-score level
        max_search: Upper bound for search

    Returns:
        Maximum safe capacity
    """
    capacity = 0
    for n in range(1, max_search + 1):
        stats = compute_hamming_statistics(n, p_match=0.0, dim=dim)
        if stats["z_score"] >= z_threshold:
            capacity = n
        else:
            break
    return capacity