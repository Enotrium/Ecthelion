"""
Sparse Binary Hyperdimensional Computing
========================================
Extends the HDC framework with sparse binary hypervectors (controlled density).
Classification and Recall with Binary Hyperdimensional Computing:
     Trade-offs in Choice of Density and Mapping Characteristics

Sparse HVs contain a small fraction of 1-bits (e.g., ρ = 2-10% density)
instead of the standard 50% dense representations. This trades some
information capacity for:
    - Context-Dependent Thinning (CDT): bundling without density explosion
    - Constant sparsity through progressive bit-flip encoding
    - Different noise resilience and capacity characteristics

Key operations:
    - gen_sparse_hvs:  Generate HVs with density ρ = M/D (M ones per D bits)
    - cdt_bundle:      Context-Dependent Thinning for sparse superposition
    - sparse_bundle:   OR-sum + CDT pipeline (the sparse equivalent of consensus sum)
    - sparse_majority: Fixed-density binarization via MajorityCDT (Kleyko)
    - sparse_bind:     Sparse binding (same as XOR, but works with sparse vectors)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

from hap.hdc_core import (
    hv_xor,
    hv_permute,
    hv_popcount,
    hv_hamming_sim,
    hv_batch_sim,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Sparse HV Generation
# ═══════════════════════════════════════════════════════════════════════════════

def gen_sparse_hvs(
    n: int,
    dim: int,
    density: float = 0.05,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate n sparse binary hypervectors with controlled density.

    Unlike dense binary HVs (ρ = 0.5), sparse HVs have a small fraction
    of 1-bits. This creates a fundamentally different regime:
        - Fewer active bits → lower capacity but faster operations
        - CDT bundling prevents density explosion
        - Different noise tolerance characteristics

    From Kleyko et al. (2021), Section V-A:
        "Sparse distributed representations are obtained by setting
         density ρ = M/D, where typically ρ ∈ [0.02, 0.1]."

    Args:
        n: Number of hypervectors to generate
        dim: Dimensionality of each HV
        density: Fraction of 1-bits (ρ). Range: (0, 0.5].
                 0.05 = 5% ones (common sparse setting)
                 0.5  = dense binary (standard)
        device: 'cpu' or 'cuda'
        seed: Random seed

    Returns:
        (n, dim) tensor of sparse binary HVs (0/1)
    """
    if not 0 < density <= 0.5:
        raise ValueError(f"Density must be in (0, 0.5], got {density}")

    dev = device or "cpu"
    g = torch.Generator(device=dev)
    if seed is not None:
        g.manual_seed(seed)

    return (torch.rand(n, dim, generator=g, device=dev) < density).float()


# ═══════════════════════════════════════════════════════════════════════════════
# Context-Dependent Thinning (CDT)
# ═══════════════════════════════════════════════════════════════════════════════

def cdt(
    hv: torch.Tensor,
    n_thinning: int = 2,
    shift_coef: float = 0.8,
) -> torch.Tensor:
    """Context-Dependent Thinning for a single sparse binary HV.

    When multiple sparse HVs are OR-summed, the density grows.
    CDT thins the result back by requiring that 1-bits survive
    only if they also appear in a shifted version of the vector.

    From Kleyko et al. (2021), Eq. in CDT.m:
        "Implementation of fixed number of permutations for thinning:
         For K iterations:
             shifted = circshift(superposition, [0, coef_shift + K])
             thinned = AND(superposition, shifted)
             result = OR(result, thinned)"

    Args:
        hv: (D,) binary vector to thin
        n_thinning: Number of permutation-thinning iterations (default: 2)
        shift_coef: Fraction of D for the base shift amount (default: 0.8)

    Returns:
        (D,) thinned binary vector
    """
    if hv.dim() > 1:
        raise ValueError(f"cdt expects 1D input, got shape {hv.shape}")

    dim = hv.shape[-1]
    base_shift = int(round(shift_coef * dim))
    result = torch.zeros(dim, dtype=hv.dtype, device=hv.device)

    superposition = hv.clone()

    for k in range(n_thinning):
        shift_amount = base_shift + k + 1
        shifted = hv_permute(superposition, shift_amount)
        thinned = torch.logical_and(superposition, shifted).float()
        result = torch.logical_or(result, thinned).float()

    return result


def sparse_bundle(
    hvs: torch.Tensor,
    n_thinning: int = 2,
    shift_coef: float = 0.8,
) -> torch.Tensor:
    """Bundle sparse HVs via OR-sum + CDT.

    The sparse equivalent of the consensus sum (hv_bundle).
    Standard bundling via majority vote doesn't work with sparse vectors
    because the OR-sum causes density to explode.

    Pipeline:
        1. OR-sum across all vectors (disjunctive superposition)
        2. Binarize: > 0 → 1
        3. CDT thinning: recover sparse density

    From Kleyko et al. (2021), Section V-B:
        "CDT bundles sparse vectors while maintaining approximately
         constant density through context-dependent thinning."

    Args:
        hvs: (N, D) sparse binary HVs to bundle
        n_thinning: CDT iterations (more = sparser result)
        shift_coef: Base shift fraction for CDT (0.8 is standard)

    Returns:
        (D,) bundled sparse binary HV
    """
    if hvs.dim() == 1:
        return cdt(hvs, n_thinning, shift_coef)

    # Disjunctive superposition (OR-sum)
    superposition = (hvs.sum(dim=0) > 0).float()

    return cdt(superposition, n_thinning, shift_coef)


def sparse_majority(
    superposition: torch.Tensor,
    target_ones: int,
) -> torch.Tensor:
    """MajorityCDT: binarize accumulated superposition to fixed density.

    From Kleyko et al. (2021), MajorityCDT.m:
        "After accumulating sparse vectors, we select the M positions
         with the highest counts as 1s, where M is the target number
         of ones (density × D)."

    This is different from dense thresholding (> 0.5). In the sparse
    regime, we maintain constant density by keeping only the top-K
    most-activated positions.

    Args:
        superposition: (D,) accumulated superposition (integer/float counts)
        target_ones: Target number of 1-bits in the result (M = ρ·D)

    Returns:
        (D,) sparse binary HV with exactly target_ones 1-bits
    """
    dim = superposition.shape[-1]
    target_ones = min(target_ones, dim)

    if target_ones <= 0:
        return torch.zeros(dim, dtype=superposition.dtype, device=superposition.device)

    # Select top-K positions by activation count
    _, indices = torch.topk(superposition, target_ones)
    result = torch.zeros(dim, dtype=superposition.dtype, device=superposition.device)
    result[indices] = 1.0

    return result


def sparse_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bind two sparse binary HVs.

    Sparse binding is the same as XOR for binary vectors.
    The binding of two sparse vectors is generally NOT sparse
    (it has ~2ρ·D ones for independent vectors), so follow-up
    thinning may be needed for cascaded bind-bundle chains.

    Args:
        a, b: (..., D) sparse binary HVs

    Returns:
        (..., D) XOR result
    """
    return hv_xor(a, b)


# ═══════════════════════════════════════════════════════════════════════════════
# Sparse Item Memory Generation
# ═══════════════════════════════════════════════════════════════════════════════

def gen_sparse_basis(
    n: int,
    dim: int,
    density: float = 0.05,
    proportional_spacing: bool = True,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Generate proportionally-spaced sparse basis vectors.

    For encoding continuous values with sparse HVs.
    Uses progressive bit-flipping to maintain constant density
    while varying the Hamming distance proportionally.

    From Kleyko et al. (2021), initItemMemoriesSparse.m:
        "SP = floor((density_act - D*((density_act/D)^2)) / MAXL)
         For each step, flip SP ones to zeros and SP zeros to ones,
         maintaining constant total density."

    Args:
        n: Number of basis vectors
        dim: Vector dimensionality
        density: Target ρ (fraction of 1-bits)
        proportional_spacing: If True, use progressive interpolation
        seed: Random seed
        device: torch device

    Returns:
        (n, dim) basis vectors (sparse binary)
    """
    dev = device or "cpu"
    s = seed or 42

    base = gen_sparse_hvs(n, dim, density, dev, s)
    M = int(round(density * dim))

    if not proportional_spacing:
        return base

    # Progressive bit-flip encoding maintaining constant density
    flip_per_step = max(1, M // max(n, 1))

    basis = [base[0]]
    for i in range(1, n):
        # Copy previous and flip SP bits (ones→zeros, zeros→ones)
        current = basis[-1].clone()
        pos1 = (current == 1).nonzero(as_tuple=True)[0]
        pos0 = (current == 0).nonzero(as_tuple=True)[0]

        if len(pos1) >= flip_per_step and len(pos0) >= flip_per_step:
            g = torch.Generator(device=dev)
            if seed is not None:
                g.manual_seed(seed + i * 1000)

            flip1_idx = pos1[torch.randperm(len(pos1), generator=g, device=dev)[:flip_per_step]]
            flip0_idx = pos0[torch.randperm(len(pos0), generator=g, device=dev)[:flip_per_step]]

            current[flip1_idx] = 0.0
            current[flip0_idx] = 1.0

        basis.append(current)

    return torch.stack(basis)


# ═══════════════════════════════════════════════════════════════════════════════
# Sparse Similarity
# ═══════════════════════════════════════════════════════════════════════════════

def sparse_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Jaccard-like similarity for sparse binary HVs.

    For sparse vectors, the Hamming distance has a different baseline
    than 0.5 (it depends on ρ). The overlap-based similarity is often
    more informative:

        sim(a, b) = |a ∧ b| / |a ∨ b|  (Jaccard)
                  = sum(a AND b) / sum(a OR b)

    Args:
        a, b: (..., D) sparse binary HVs

    Returns:
        (...) similarity in [0, 1]
    """
    and_sum = (a * b).sum(dim=-1)
    or_sum = ((a + b) > 0).float().sum(dim=-1)
    or_sum = or_sum.clamp(min=1)
    return and_sum / or_sum


def sparse_overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Raw overlap count: number of 1-bits in both a and b.

    overlap(a, b) = sum(a AND b)

    Args:
        a, b: (..., D) sparse binary HVs

    Returns:
        (...) overlap count (integer float)
    """
    return (a * b).sum(dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Energy Model Extension for Sparse Ops
# ═══════════════════════════════════════════════════════════════════════════════

# Sparse operations are potentially even cheaper because:
#  - OR-sum: 0.05 pJ/bit (only need to check if non-zero)
#  - CDT shift + AND: 0.01 + 0.1 = 0.11 pJ/bit per iteration
#  - Sparse XOR: still 0.1 pJ/bit (same as dense)
ENERGY_CDT_OR_PJ = 0.05    # pJ per bit — disjunctive superposition
ENERGY_CDT_AND_PJ = 0.08   # pJ per bit — AND for thinning
ENERGY_CDT_SHIFT_PJ = 0.01  # pJ per bit — circular shift


def estimate_energy_sparse(
    dim: int,
    density: float,
    n_or: int = 0,
    n_cdt: int = 0,
    n_xor: int = 0,
) -> Dict[str, float]:
    """Estimate energy for sparse HDC operations.

    Args:
        dim: Hypervector dimension
        density: ρ (fraction of 1-bits per vector)
        n_or: Number of OR-sum operations
        n_cdt: Number of CDT thinning operations
        n_xor: Number of XOR operations

    Returns:
        Dict with sparse energy breakdown
    """
    or_energy = n_or * dim * ENERGY_CDT_OR_PJ
    cdt_energy = n_cdt * dim * (ENERGY_CDT_AND_PJ + ENERGY_CDT_SHIFT_PJ)
    xor_energy = n_xor * dim * 0.1  # same as dense

    total = or_energy + cdt_energy + xor_energy

    return {
        "or_energy_pj": or_energy,
        "cdt_energy_pj": cdt_energy,
        "xor_energy_pj": xor_energy,
        "total_sparse_energy_pj": total,
        "total_sparse_energy_nj": total / 1000.0,
        "density": density,
    }