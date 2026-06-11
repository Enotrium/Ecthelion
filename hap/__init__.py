"""Hyperdimensional Active Perception (HAP) Framework
===================================================
Production-grade implementation of:
Learning Sensorimotor Control with Neuromorphic Sensors:
   Toward Hyperdimensional Active Perception"
  Science Robotics, vol. 4, no. 30, eaaw6736

Core thesis:
    "Push the job onto encoding — purely hardware.
     Push encoding as far away from actual learning
     that you can learn very rapidly."

    — Peter Sutor, paraphrasing Kanerva's HDC philosophy

What this means:
    The encoding pipeline is the heavy part (hardware-accelerated).
    The learning is trivial — just count co-occurrences via consensus sum.
    The encoding extracts meaningful hypervectors from raw sensor data.
    The learning is a single pass of bundling bound (perception, action) pairs.
    Inference is O(D) — one XOR + one popcount per class.

Architecture:
    Sensor Stream (DVS events, images, velocities)
        │
        ▼
    ┌─────────────────────────────────────────────────┐
    │  HDC Encoder (hardware push: position keys,     │
    │   intensity keys, permutation sequences,         │
    │   fractional power, grid cell encoding)          │
    └─────────────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────────┐
    │  Associative Memory (learning = consensus sum   │
    │   of bound perception-action pairs)              │
    │   "Save counts as superposition"                 │
    └─────────────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────────┐
    │  Inference (unbind query → nearest neighbor     │
    │   via Hamming distance)                          │
    │   "See either more ones or more zeros"           │
    └─────────────────────────────────────────────────┘
        │
        ▼
    Prediction (velocity, class, action)

Benchmark performance (from paper):
    - Ego-motion: 572 inferences/sec (500 velocity classes)
    - Training: 0.5s for 500 frames, 2s for 1500 frames
    - Memory: 8000-bit hypervectors → O(D) constant
    - Single CPU pass: no GPU needed
    - CNN-comparable accuracy at <1% the computational cost
"""

from hap.capacity import (
    binomial_prob,
    capacity_curve,
    compute_hamming_statistics,
    find_capacity_limit,
)
from hap.data_structures import (
    FrequencyEncoder,
    FSAEncoder,
    GraphEncoder,
    NGramEncoder,
    StackEncoder,
    TreeEncoder,
)
from hap.encoding import (
    DataRecordEncoder,
    DVSEncoder,
    PositionalIntensityEncoder,
    SequenceEncoder,
    TimeSliceEncoder,
    VelocityEncoder,
)
from hap.hap import (
    EgoMotionEstimator,
    HyperdimensionalActivePerception,
)
from hap.hdc_core import (
    HDCConfig,
    gen_hvs,
    hv_batch_sim,
    hv_bind,
    hv_bundle,
    hv_consensus_sum,
    hv_hamming_sim,
    hv_majority,
    hv_permute,
    hv_popcount,
    hv_xor,
)
from hap.memory import (
    ActionPerceptionMemory,
    AssociativeMemory,
    DataRecordMemory,
    HDCClassifier,
    RefineHDLearner,
)
from hap.sparse_hdc import (
    cdt,
    estimate_energy_sparse,
    gen_sparse_basis,
    gen_sparse_hvs,
    sparse_bind,
    sparse_bundle,
    sparse_majority,
    sparse_overlap,
    sparse_similarity,
)
from hap.tension import (
    build_cooc_graph,
    build_masses,
    learn_distributional_hvs,
    minimize_tension,
    tension_energy,
)

__all__ = [
    "ActionPerceptionMemory",
    # Memory
    "AssociativeMemory",
    "DVSEncoder",
    "DataRecordEncoder",
    "DataRecordMemory",
    "EgoMotionEstimator",
    "FSAEncoder",
    "FrequencyEncoder",
    # Data Structures
    "GraphEncoder",
    "HDCClassifier",
    "HDCConfig",
    # HAP
    "HyperdimensionalActivePerception",
    "NGramEncoder",
    "PositionalIntensityEncoder",
    "RefineHDLearner",
    "SequenceEncoder",
    "StackEncoder",
    # Encoding
    "TimeSliceEncoder",
    "TreeEncoder",
    "VelocityEncoder",
    # Capacity Analysis
    "binomial_prob",
    "build_cooc_graph",
    "build_masses",
    "capacity_curve",
    "cdt",
    "compute_hamming_statistics",
    "estimate_energy_sparse",
    "find_capacity_limit",
    # Core
    "gen_hvs",
    "gen_sparse_basis",
    # Sparse HDC
    "gen_sparse_hvs",
    "hv_batch_sim",
    "hv_bind",
    "hv_bundle",
    "hv_consensus_sum",
    "hv_hamming_sim",
    "hv_majority",
    "hv_permute",
    "hv_popcount",
    "hv_xor",
    "learn_distributional_hvs",
    "minimize_tension",
    "sparse_bind",
    "sparse_bundle",
    "sparse_majority",
    "sparse_overlap",
    "sparse_similarity",
    # Tension Minimization
    "tension_energy",
]
