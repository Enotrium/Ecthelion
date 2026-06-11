"""Tests for hap.capacity — Theoretical Capacity (paper Section: Limits on capacity)."""

from hap.capacity import (
    binomial_prob,
    capacity_curve,
    compute_hamming_statistics,
    find_capacity_limit,
)


class TestBinomialProb:
    def test_exact_zero_for_match(self):
        # With p=0 (exact match), H_n should be < 0.5
        p = binomial_prob(1, 0.0)
        assert p == 0.0

    def test_random_for_no_match(self):
        # With p=0.5 (random), H_n should be exactly 0.5
        p = binomial_prob(1, 0.5)
        assert p == 0.5

    def test_approaches_0_5_as_n_grows(self):
        # Even with exact match, as n → ∞, H_n → 0.5
        p_small = binomial_prob(5, 0.0)
        p_large = binomial_prob(50, 0.0)
        assert 0.0 < p_small < p_large < 0.5

    def test_monotonic_in_n(self):
        # More records → H_n closer to 0.5 (match signal weaker)
        prev = binomial_prob(1, 0.0)
        for n in [2, 3, 5, 10, 20]:
            curr = binomial_prob(n, 0.0)
            assert curr >= prev
            prev = curr


class TestComputeHammingStatistics:
    def test_returns_keys(self):
        stats = compute_hamming_statistics(10, 0.0, 10000)
        for key in ["H_n", "deviation_from_random", "z_score", "is_significant_3sigma"]:
            assert key in stats

    def test_significance_at_low_n(self):
        stats = compute_hamming_statistics(10, 0.0, 10000)
        assert stats["is_significant_3sigma"]

    def test_not_significant_at_high_n(self):
        stats = compute_hamming_statistics(1000, 0.0, 10000)
        assert not stats["is_significant_3sigma"]

    def test_different_dimensions(self):
        stats_low = compute_hamming_statistics(100, 0.0, 1000)
        stats_high = compute_hamming_statistics(100, 0.0, 10000)
        # Higher dimension → smaller sigma → larger z
        assert stats_high["z_score"] > stats_low["z_score"]


class TestCapacityCurve:
    def test_returns_list(self):
        curve = capacity_curve(100, dim=10000)
        assert len(curve) == 100
        assert curve[0]["n_records"] == 1
        assert curve[-1]["n_records"] == 100

    def test_monotonic_decay(self):
        curve = capacity_curve(200, dim=10000)
        # H_n should increase monotonically (closer to 0.5)
        for i in range(1, len(curve)):
            assert curve[i]["H_n"] >= curve[i - 1]["H_n"] - 1e-12


class TestFindCapacityLimit:
    def test_returns_positive(self):
        cap = find_capacity_limit(dim=10000, z_threshold=3.0, max_search=2000)
        assert cap > 0

    def test_approximately_700_at_d10000(self):
        cap = find_capacity_limit(dim=10000, z_threshold=3.0, max_search=2000)
        # Paper says ~700 at D=10,000
        assert 600 <= cap <= 900

    def test_higher_dim_more_capacity(self):
        cap_low = find_capacity_limit(dim=1000, z_threshold=3.0, max_search=1000)
        cap_high = find_capacity_limit(dim=10000, z_threshold=3.0, max_search=5000)
        assert cap_high > cap_low
