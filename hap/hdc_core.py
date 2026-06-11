"""
HDC Core — Pure Binary Hyperdimensional Computing Primitives
=============================================================
"Push the job onto encoding. Purely hardware. Then push encoding
 as far away from actual learning that you can learn very rapidly."

All operations are XOR + popcount only.
No multiplication, no floating-point similarity, no backpropagation.

From the paper (Mitrokhin, Sutor et al. 2019):
    - XOR: Involution, associative, commutative. Self-inverse.
    - Permutation: P(x) shuffles components. Preserves Hamming distance.
    - Consensus sum: Component-wise majority vote over a set of vectors.
    - Hamming distance: H(a,b) = popcount(XOR(a,b)). Normalized to [0,1].

The learning IS the consensus sum. "Save counts as superposition."
When you bind a perception HV to an action HV and add it to memory,
you're literally counting co-occurrences. Inference "sees either more
ones or more zeros" — the bias in the Hamming distance tells you
which class/action is the nearest neighbor.

Hardware properties (45nm CMOS, Horowitz ISSCC 2014):
    XOR:       0.1 pJ/bit
    Popcount:  0.2 pJ/op
    MAC:       4.6 pJ/op  (46× more expensive)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

from hap.exceptions import HAPDimensionError, HAPModeError

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HDCConfig:
    """Global HDC configuration.

    Attributes:
        dim: Hypervector dimensionality (Kanerva uses 10,000; paper uses 8,000)
        mode: 'binary' (0/1) or 'bipolar' (+1/-1). Binary uses XOR for binding.
        device: torch device ('cpu', 'cuda')
        seed: Random seed for reproducibility
    """
    dim: int = 10_000
    mode: str = "binary"
    device: str = "cpu"
    seed: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════════════
# HV Generation
# ═══════════════════════════════════════════════════════════════════════════════

def gen_hvs(
    n: int,
    dim: int,
    mode: str = "binary",
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate n random hypervectors of dimension dim.

    Random HVs are nearly orthogonal with high probability:
        E[H_n] = 0.5, σ = 0.005 (for dim=10,000)
    So H = 0.475 is 5σ from mean — essentially guaranteed non-random.

    Args:
        n: Number of hypervectors to generate
        dim: Dimensionality of each HV
        mode: 'binary' (0/1) or 'bipolar' (+1/-1)
        device: 'cpu' or 'cuda'
        seed: For reproducibility

    Returns:
        (n, dim) tensor of random hypervectors
    """
    dev = device or "cpu"
    g = torch.Generator(device=dev)
    if seed is not None:
        g.manual_seed(seed)

    if mode == "binary":
        return torch.randint(0, 2, (n, dim), generator=g, device=dev).float()
    elif mode == "bipolar":
        return (torch.randint(0, 2, (n, dim), generator=g, device=dev) * 2 - 1).float()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'binary' or 'bipolar'.")


# ═══════════════════════════════════════════════════════════════════════════════
# Core Operations — Every Algorithm in This Codebase Reduces to These
# ═══════════════════════════════════════════════════════════════════════════════

def hv_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bitwise XOR — the ONLY binding operation needed.

    Properties (from paper Section II.A):
        - Involution: a XOR a = 0 (self-inverse)
        - Associative: (a XOR b) XOR c = a XOR (b XOR c)
        - Commutative: a XOR b = b XOR a
        - Unbinding: (a XOR b) XOR a = b

    The XOR is the "multiply" of binary HDC.
    Binding = XOR. Unbinding = XOR (same op).

    Args:
        a, b: (..., D) binary hypervectors

    Returns:
        (..., D) XOR result
    """
    return (a != b).float()


def hv_popcount(hv: torch.Tensor) -> torch.Tensor:
    """Popcount — the ONLY similarity operation needed.

    Counts the number of 1-bits. For distance, use with XOR:
        H(a, b) = popcount(XOR(a, b))

    No cosine similarity. No dot product. No normalization.
    Just count bits. This is the hardware-friendly operation.

    Args:
        hv: (..., D) binary hypervector

    Returns:
        (...) popcount (integer count of 1s)
    """
    return hv.sum(dim=-1)


def hv_hamming_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Normalized Hamming similarity: 1 - H(a,b)/D.

    H_n(a,b) = popcount(XOR(a,b)) / D
        - 0.0 = identical (0 bits differ)
        - 0.5 = random (half differ)
        - 1.0 = complementary (all differ)

    For random HVs, H_n ≈ 0.5 with σ = 0.005 (dim=10,000).
    A value of 0.475 is 5σ from mean — statistically significant.

    "See either more ones or more zeros and the original integers
     determine summation." — The Hamming distance bias IS the decision.

    Args:
        a, b: (..., D) binary hypervectors

    Returns:
        (...) similarity in [0, 1]
    """
    return 1.0 - hv_popcount(hv_xor(a, b)) / a.shape[-1]


def hv_bundle(hvs: torch.Tensor, mode: str = "binary") -> torch.Tensor:
    """Bundle a set of HVs via component-wise sum (consensus sum).

    For binary: sum across set, then threshold at > n/2 (majority vote).
    For bipolar: sum across set, then sign threshold.

    "Bundle" = "superposition" = "consensus sum" = "addition."
    This is how we store multiple patterns in one HV.

    From the paper (Section II.A, operation 3):
        "The consensus sum, c+(A), over the set of vectors A:
         counts 1s and 0s component-wise across each element of A
         and sets the component to the corresponding value with the
         bigger count. Ties broken by randomly choosing 0 or 1."

    Args:
        hvs: (N, D) or (D,) — set of hypervectors to bundle
        mode: 'binary' or 'bipolar'

    Returns:
        (D,) bundled hypervector
    """
    if hvs.dim() == 1:
        return hvs
    return hv_majority(hvs.mean(dim=0), mode=mode)


def hv_bind(a: torch.Tensor, b: torch.Tensor, mode: str = "binary") -> torch.Tensor:
    """Bind two hypervectors.

    Binary mode: XOR (element-wise modulo-2 addition)
    Bipolar mode: element-wise multiplication (same as XOR in bipolar)

    Args:
        a, b: (..., D) hypervectors
        mode: 'binary' or 'bipolar'

    Returns:
        (..., D) bound hypervector
    """
    if mode == "binary":
        return hv_xor(a, b)
    elif mode == "bipolar":
        return a * b
    else:
        raise ValueError(f"Unknown mode: {mode}")


def hv_permute(hv: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Permute HV components by cyclic shift.

    From the paper (Section II.A, operation 2):
        "The permutation P: permutes a vector x's components into a new order.
         We can represent P as a permutation of index locations 1 to n."

    Permutations are key for encoding position and sequence order.
    P^i(x) encodes "i positions later in a sequence."

    Properties:
        - Distance preserving: H(P(a), P(b)) = H(a, b)
        - Repeated: P^i(P^j(x)) = P^(i+j)(x)
        - Invertible: P^(-i)(P^i(x)) = x

    Args:
        hv: (..., D) hypervector
        k: Number of positions to shift (positive = forward, negative = backward)

    Returns:
        (..., D) permuted hypervector
    """
    return torch.roll(hv, shifts=k, dims=-1)


def hv_consensus_sum(hvs: torch.Tensor) -> torch.Tensor:
    """Explicit consensus sum with tie-breaking.

    Implements the paper's c+(A) operation exactly:
        "Counts 1s and 0s component wise across each element of A
         and sets the component to the corresponding value with the
         bigger count. Ties, only possible in a sum of an even number
         of elements, can be broken by randomly choosing 0 or 1."

    Args:
        hvs: (N, D) set of HVs

    Returns:
        (D,) consensus sum HV
    """
    n, dim = hvs.shape
    ones = hvs.sum(dim=0)  # Count of 1s per component
    zeros = n - ones        # Count of 0s per component

    # Majority vote with tie-breaking
    result = (ones > zeros).float()

    # Random tie-breaking for components where ones == zeros
    tie_mask = (ones == zeros)
    if tie_mask.any():
        result[tie_mask] = torch.randint(0, 2, (tie_mask.sum().item(),),
                                          device=hvs.device).float()

    return result


def hv_majority(hv: torch.Tensor, mode: str = "binary") -> torch.Tensor:
    """Majority-vote thresholding for accumulated float → binary/bipolar conversion.

    For binary (0/1) vectors after accumulation:
        result[i] = 1 if hv[i] > 0.5 else 0
    For bipolar (-1/+1) vectors after accumulation:
        result[i] = 1 if hv[i] >= 0 else -1

    Args:
        hv: (D,) or (N, D) accumulated hypervector (floats)
        mode: 'binary' or 'bipolar' — determines threshold logic

    Returns:
        Thresholded hypervector(s) of same shape as input, matching dtype
    """
    if mode == "binary":
        return (hv > 0.5).float()
    elif mode == "bipolar":
        return torch.where(hv >= 0, torch.ones_like(hv), -torch.ones_like(hv)).float()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'binary' or 'bipolar'.")


def hv_batch_sim(query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    """Batch Hamming similarity between query and all memory items.

    Pure XOR + popcount, vectorized across the memory bank.
    This is O(D) per query — no quadratic attention.

    Args:
        query: (D,) query hypervector
        memory: (N, D) memory bank of N hypervectors

    Returns:
        (N,) similarity scores in [0, 1]
    """
    xor_results = (query.unsqueeze(0) != memory).float()
    popcounts = xor_results.sum(dim=1)
    return 1.0 - popcounts / query.shape[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: Legacy-compatible aliases
# ═══════════════════════════════════════════════════════════════════════════════

def sim(a: torch.Tensor, b: torch.Tensor, mode: str = "binary") -> torch.Tensor:
    """Legacy compatibility wrapper for hv_hamming_sim."""
    return hv_hamming_sim(a, b)


def bind(a: torch.Tensor, b: torch.Tensor, mode: str = "binary") -> torch.Tensor:
    """Legacy compatibility wrapper for hv_bind."""
    return hv_bind(a, b, mode)


def bundle(hvs: torch.Tensor) -> torch.Tensor:
    """Legacy compatibility wrapper for hv_bundle."""
    return hv_bundle(hvs)


def thresh(hv: torch.Tensor) -> torch.Tensor:
    """Legacy compatibility wrapper for hv_majority (bipolar)."""
    return torch.where(hv >= 0, torch.ones_like(hv), -torch.ones_like(hv))


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware Energy Model (Horowitz ISSCC 2014, 45nm CMOS)
# ═══════════════════════════════════════════════════════════════════════════════

ENERGY_XOR_PJ = 0.1       # pJ per bit — XOR binding
ENERGY_POPCOUNT_PJ = 0.2  # pJ per operation — Hamming distance
ENERGY_BIT_ADD_PJ = 0.05  # pJ per bit — bundling (integer accumulate)
ENERGY_PERMUTE_PJ = 0.01  # pJ per bit — cyclic shift / routing

ENERGY_INT8_MAC_PJ = 4.6  # pJ per multiply-accumulate (comparison)


def estimate_energy_hdv(dim: int,
                         n_xor: int = 0,
                         n_popcount: int = 0,
                         n_bundles: int = 0,
                         n_permutes: int = 0) -> Dict[str, float]:
    """Estimate energy for a sequence of HDC operations.

    Args:
        dim: Hypervector dimension
        n_xor: Number of XOR operations
        n_popcount: Number of popcount operations
        n_bundles: Number of component-wise additions (bundle steps)
        n_permutes: Number of permutation operations

    Returns:
        Dict with energy breakdown and comparison to MAC-based approach
    """
    xor_energy = n_xor * dim * ENERGY_XOR_PJ
    popcount_energy = n_popcount * ENERGY_POPCOUNT_PJ
    bundle_energy = n_bundles * dim * ENERGY_BIT_ADD_PJ
    permute_energy = n_permutes * dim * ENERGY_PERMUTE_PJ

    total_hdc = xor_energy + popcount_energy + bundle_energy + permute_energy

    # Equivalent MAC energy (what a transformer/neural net would need)
    # Typical: query × N prototypes = dim × N MACs
    equiv_mac_energy = (n_xor + n_popcount) * dim * ENERGY_INT8_MAC_PJ

    return {
        "xor_energy_pj": xor_energy,
        "popcount_energy_pj": popcount_energy,
        "bundle_energy_pj": bundle_energy,
        "permute_energy_pj": permute_energy,
        "total_hdc_energy_pj": total_hdc,
        "total_hdc_energy_nj": total_hdc / 1000.0,
        "equiv_mac_energy_pj": equiv_mac_energy,
        "equiv_mac_energy_nj": equiv_mac_energy / 1000.0,
        "ratio_mac_to_hdc": equiv_mac_energy / max(total_hdc, 1e-12),
    }
