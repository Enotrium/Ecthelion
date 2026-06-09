"""
Tests for HDC Core Primitives
==============================
Verifies all fundamental operations: XOR, bind, bundle, permute,
consensus sum, Hamming distance, orthogonality properties, and
production-grade edge cases.
"""

import torch
import pytest

from hap.hdc_core import (
    gen_hvs, hv_xor, hv_popcount, hv_hamming_sim,
    hv_bundle, hv_bind, hv_permute, hv_consensus_sum,
    hv_majority, hv_batch_sim, estimate_energy_hdv,
    ENERGY_XOR_PJ, ENERGY_POPCOUNT_PJ,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HV Generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenHVS:
    def test_shape_and_range_binary(self):
        hvs = gen_hvs(10, 100, "binary")
        assert hvs.shape == (10, 100)
        assert hvs.min() >= 0 and hvs.max() <= 1
        assert hvs.dtype == torch.float32

    def test_shape_and_range_bipolar(self):
        hvs = gen_hvs(5, 100, "bipolar")
        assert hvs.shape == (5, 100)
        assert (hvs == -1).any() and (hvs == 1).any()

    def test_seed_reproducibility(self):
        a = gen_hvs(1, 1000, seed=42)
        b = gen_hvs(1, 1000, seed=42)
        assert torch.equal(a, b)

    def test_seed_different(self):
        a = gen_hvs(1, 1000, seed=42)
        b = gen_hvs(1, 1000, seed=43)
        assert not torch.equal(a, b)

    def test_single_hv(self):
        hv = gen_hvs(1, 100, "binary").squeeze(0)
        assert hv.shape == (100,)

    def test_zero_hvs(self):
        hvs = gen_hvs(0, 100)
        assert hvs.shape == (0, 100)

    def test_device(self):
        if torch.cuda.is_available():
            hvs = gen_hvs(3, 100, device="cuda")
            assert hvs.device.type == "cuda"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            gen_hvs(1, 10, mode="float32")


# ═══════════════════════════════════════════════════════════════════════════════
# XOR
# ═══════════════════════════════════════════════════════════════════════════════

class TestXOR:
    """Paper Section II.A: XOR is involution, associative, commutative."""

    def test_involution(self):
        """a XOR a = 0 (self-inverse)"""
        a = gen_hvs(1, 1000, seed=42).squeeze(0)
        result = hv_xor(a, a)
        assert result.sum() == 0, "a XOR a should be all zeros"

    def test_associative(self):
        """(a XOR b) XOR c = a XOR (b XOR c)"""
        a, b, c = [gen_hvs(1, 1000, seed=s).squeeze(0) for s in [1, 2, 3]]
        lhs = hv_xor(hv_xor(a, b), c)
        rhs = hv_xor(a, hv_xor(b, c))
        assert torch.equal(lhs, rhs)

    def test_commutative(self):
        """a XOR b = b XOR a"""
        a = gen_hvs(1, 1000, seed=42).squeeze(0)
        b = gen_hvs(1, 1000, seed=43).squeeze(0)
        assert torch.equal(hv_xor(a, b), hv_xor(b, a))

    def test_unbinding(self):
        """(a XOR b) XOR a = b"""
        a = gen_hvs(1, 1000, seed=42).squeeze(0)
        b = gen_hvs(1, 1000, seed=43).squeeze(0)
        bound = hv_xor(a, b)
        recovered = hv_xor(bound, a)
        assert torch.equal(recovered, b)

    def test_unbinding_symmetric(self):
        """(a XOR b) XOR b = a"""
        a = gen_hvs(1, 500, seed=1).squeeze(0)
        b = gen_hvs(1, 500, seed=2).squeeze(0)
        bound = hv_xor(a, b)
        assert torch.equal(hv_xor(bound, b), a)

    def test_binary_output(self):
        a = gen_hvs(1, 100).squeeze(0)
        b = gen_hvs(1, 100).squeeze(0)
        r = hv_xor(a, b)
        assert torch.all((r == 0) | (r == 1))


# ═══════════════════════════════════════════════════════════════════════════════
# Hamming Distance
# ═══════════════════════════════════════════════════════════════════════════════

class TestHammingDistance:
    """Paper: 'the probability of it being associated' = 1 - H_n."""

    def test_identical(self):
        a = gen_hvs(1, 1000).squeeze(0)
        assert hv_hamming_sim(a, a) == 1.0

    def test_random_orthogonality(self):
        """For D=10,000, random HVs have H_n ≈ 0.5 with σ ≈ 0.005."""
        hvs = gen_hvs(100, 10_000, seed=42)
        sims = []
        for i in range(len(hvs) - 1):
            sims.append(hv_hamming_sim(hvs[i], hvs[i + 1]).item())
        mean_sim = sum(sims) / len(sims)
        assert 0.49 < mean_sim < 0.51, f"Random HVs should have H_n≈0.5, got {mean_sim}"

    def test_self_dissimilarity(self):
        """H(a, a) = 0 → popcount(XOR(a, a)) = 0."""
        a = gen_hvs(1, 1000).squeeze(0)
        assert hv_popcount(hv_xor(a, a)) == 0

    def test_triangle_like(self):
        """H(a, c) ≤ H(a, b) + H(b, c) - 2*overlap (not strict triangle)."""
        a = gen_hvs(1, 1000).squeeze(0)
        b = gen_hvs(1, 1000).squeeze(0)
        c = gen_hvs(1, 1000).squeeze(0)
        d_ab = (1.0 - hv_hamming_sim(a, b)) * 1000
        d_ac = (1.0 - hv_hamming_sim(a, c)) * 1000
        d_bc = (1.0 - hv_hamming_sim(b, c)) * 1000
        # XOR distance satisfies relaxed triangle
        assert d_ac <= d_ab + d_bc

    def test_range(self):
        a = gen_hvs(1, 100).squeeze(0)
        b = gen_hvs(1, 100).squeeze(0)
        sim = hv_hamming_sim(a, b)
        assert 0.0 <= sim <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Permute
# ═══════════════════════════════════════════════════════════════════════════════

class TestPermute:
    """Paper Section II.A: P is a permutation of index locations."""

    def test_distance_preserving(self):
        """H(P(a), P(b)) = H(a, b)"""
        a = gen_hvs(1, 1000).squeeze(0)
        b = gen_hvs(1, 1000).squeeze(0)
        sim_before = hv_hamming_sim(a, b)
        sim_after = hv_hamming_sim(hv_permute(a, 3), hv_permute(b, 3))
        assert abs(sim_before - sim_after) < 1e-6

    def test_invertibility(self):
        """P^(-k)(P^k(x)) = x"""
        x = gen_hvs(1, 1000).squeeze(0)
        permuted = hv_permute(x, 7)
        recovered = hv_permute(permuted, -7)
        assert torch.equal(x, recovered)

    def test_composition(self):
        """P^i(P^j(x)) = P^(i+j)(x)"""
        x = gen_hvs(1, 1000).squeeze(0)
        p1 = hv_permute(x, 3)
        p2 = hv_permute(p1, 5)
        direct = hv_permute(x, 8)
        assert torch.equal(p2, direct)

    def test_zero_shift(self):
        x = gen_hvs(1, 500).squeeze(0)
        assert torch.equal(hv_permute(x, 0), x)

    def test_full_cycle(self):
        """P^D(x) = x (cyclic shift wraps around)"""
        x = gen_hvs(1, 100).squeeze(0)
        assert torch.equal(hv_permute(x, 100), x)

    def test_large_shift(self):
        x = gen_hvs(1, 100, seed=1).squeeze(0)
        p = hv_permute(x, 250)
        p2 = hv_permute(x, -250)
        assert torch.equal(hv_permute(p, 150), hv_permute(p2, -150))


# ═══════════════════════════════════════════════════════════════════════════════
# Bundle / Majority / Consensus Sum
# ═══════════════════════════════════════════════════════════════════════════════

class TestBundle:
    """Paper: consensus sum = component-wise majority vote."""

    def test_bundle_shape(self):
        hvs = gen_hvs(5, 100, seed=42)
        bundle = hv_bundle(hvs)
        assert bundle.shape == (100,)
        assert torch.all((bundle == 0) | (bundle == 1))

    def test_bundle_single(self):
        hv = gen_hvs(1, 100, seed=42).squeeze(0)
        result = hv_bundle(hv)
        assert torch.equal(result, hv)

    def test_bundle_bipolar(self):
        hvs = gen_hvs(5, 100, "bipolar", seed=42)
        result = hv_bundle(hvs, mode="bipolar")
        assert result.shape == (100,)
        assert torch.all((result == 1) | (result == -1) | (result == 0))

    def test_majority_binary(self):
        hv = torch.tensor([0.6, 0.3, 0.8, 0.2])
        m = hv_majority(hv, mode="binary")
        assert torch.equal(m, torch.tensor([1., 0., 1., 0.]))

    def test_majority_binary_ties(self):
        """Exactly 0.5 should be NOT > 0.5, so result is 0."""
        hv = torch.tensor([0.5, 0.5, 0.5])
        m = hv_majority(hv, mode="binary")
        assert torch.equal(m, torch.tensor([0., 0., 0.]))

    def test_majority_bipolar_positive(self):
        hv = torch.tensor([0.5, -0.5, 0.0])
        m = hv_majority(hv, mode="bipolar")
        assert torch.equal(m, torch.tensor([1., -1., 1.]))  # 0.0 >= 0 → 1

    def test_majority_bipolar_negative(self):
        hv = torch.tensor([-0.1])
        m = hv_majority(hv, mode="bipolar")
        assert m[0] == -1.0

    def test_consensus_sum_majority(self):
        hvs = torch.tensor([[1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        cs = hv_consensus_sum(hvs)
        assert cs[0] == 1.0
        assert cs[1] == 1.0
        assert cs[2] == 0.0

    def test_consensus_sum_ties_binary(self):
        # Even number of identical vectors → ties everywhere
        hvs = torch.ones(2, 100)
        cs = hv_consensus_sum(hvs)
        assert cs.shape == (100,)
        assert torch.all((cs == 0) | (cs == 1))


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Similarity
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchSim:
    def test_batch_similarity(self):
        query = gen_hvs(1, 100).squeeze(0)
        memory = gen_hvs(10, 100)
        sims = hv_batch_sim(query, memory)
        assert sims.shape == (10,)
        assert torch.all((sims >= 0) & (sims <= 1))

    def test_exact_match(self):
        query = gen_hvs(1, 100, seed=42).squeeze(0)
        memory = gen_hvs(10, 100, seed=42)  # first row matches query
        sims = hv_batch_sim(query, memory)
        assert sims[0] == 1.0, "First memory item should match exactly"

    def test_single_memory(self):
        query = gen_hvs(1, 100, seed=1).squeeze(0)
        memory = gen_hvs(1, 100, seed=2)
        sims = hv_batch_sim(query, memory)
        assert sims.shape == (1,)

    def test_self_always_one(self):
        query = gen_hvs(1, 500).squeeze(0)
        memory = torch.stack([query, query, query])
        sims = hv_batch_sim(query, memory)
        assert torch.all(sims == 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Bind
# ═══════════════════════════════════════════════════════════════════════════════

class TestBind:
    def test_bind_unbind_roundtrip_binary(self):
        a = gen_hvs(1, 500, seed=10).squeeze(0)
        b = gen_hvs(1, 500, seed=20).squeeze(0)
        bound = hv_bind(a, b, "binary")
        recovered = hv_bind(bound, a, "binary")
        assert torch.equal(recovered, b)

    def test_bind_unbind_roundtrip_bipolar(self):
        a = gen_hvs(1, 500, seed=10, mode="bipolar").squeeze(0)
        b = gen_hvs(1, 500, seed=20, mode="bipolar").squeeze(0)
        bound = hv_bind(a, b, "bipolar")
        recovered = hv_bind(bound, a, "bipolar")
        assert torch.equal(recovered, b)

    def test_bind_invalid_mode(self):
        a = gen_hvs(1, 10).squeeze(0)
        b = gen_hvs(1, 10).squeeze(0)
        with pytest.raises(ValueError):
            hv_bind(a, b, "invalid")


# ═══════════════════════════════════════════════════════════════════════════════
# Energy Model
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnergy:
    def test_energy_returns_dict(self):
        energy = estimate_energy_hdv(dim=1000, n_xor=10)
        assert isinstance(energy, dict)
        assert "total_hdc_energy_pj" in energy

    def test_energy_ratio_positive(self):
        energy = estimate_energy_hdv(dim=1000, n_xor=100, n_popcount=50)
        assert energy["ratio_mac_to_hdc"] > 0

    def test_zero_ops(self):
        energy = estimate_energy_hdv(dim=1000)
        assert energy["total_hdc_energy_pj"] == 0.0

    def test_energy_constants(self):
        assert ENERGY_XOR_PJ > 0
        assert ENERGY_POPCOUNT_PJ > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Popcount
# ═══════════════════════════════════════════════════════════════════════════════

class TestPopcount:
    def test_all_ones(self):
        hv = torch.ones(100)
        assert hv_popcount(hv) == 100

    def test_all_zeros(self):
        hv = torch.zeros(100)
        assert hv_popcount(hv) == 0

    def test_batch(self):
        hvs = torch.ones(3, 100)
        counts = hv_popcount(hvs)
        assert torch.equal(counts, torch.tensor([100., 100., 100.]))


# ═══════════════════════════════════════════════════════════════════════════════
# HDCConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestHDCConfig:
    def test_defaults(self):
        from hap.hdc_core import HDCConfig
        cfg = HDCConfig()
        assert cfg.dim == 10_000
        assert cfg.mode == "binary"
        assert cfg.device == "cpu"

    def test_custom(self):
        from hap.hdc_core import HDCConfig
        cfg = HDCConfig(dim=5000, mode="bipolar", device="cuda", seed=123)
        assert cfg.dim == 5000
        assert cfg.mode == "bipolar"
        assert cfg.seed == 123