"""
Hyperdimensional Active Perception (HAP) Framework
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

from hap.hdc_core import (
    gen_hvs,
    hv_xor,
    hv_popcount,
    hv_hamming_sim,
    hv_bundle,
    hv_bind,
    hv_permute,
    hv_consensus_sum,
    hv_majority,
    hv_batch_sim,
    HDCConfig,
)
from hap.encoding import (
    TimeSliceEncoder,
    VelocityEncoder,
    SequenceEncoder,
    DVSEncoder,
    PositionalIntensityEncoder,
    DataRecordEncoder,
)
from hap.memory import (
    AssociativeMemory,
    ActionPerceptionMemory,
    DataRecordMemory,
    HDCClassifier,
    RefineHDLearner,
)
from hap.hap import (
    HyperdimensionalActivePerception,
    EgoMotionEstimator,
)
from hap.sparse_hdc import (
    gen_sparse_hvs,
    cdt,
    sparse_bundle,
    sparse_majority,
    sparse_bind,
    gen_sparse_basis,
    sparse_similarity,
    sparse_overlap,
    estimate_energy_sparse,
)
from hap.data_structures import (
    GraphEncoder,
    TreeEncoder,
    FSAEncoder,
    NGramEncoder,
    FrequencyEncoder,
    StackEncoder,
)
from hap.tension import (
    tension_energy,
    minimize_tension,
    build_cooc_graph,
    build_masses,
    learn_distributional_hvs,
)
from hap.capacity import (
    binomial_prob,
    compute_hamming_statistics,
    capacity_curve,
    find_capacity_limit,
)

__all__ = [
    # Core
    "gen_hvs", "hv_xor", "hv_popcount", "hv_hamming_sim",
    "hv_bundle", "hv_bind", "hv_permute", "hv_consensus_sum",
    "hv_majority", "hv_batch_sim", "HDCConfig",
    # Encoding
    "TimeSliceEncoder", "VelocityEncoder", "SequenceEncoder",
    "DVSEncoder", "PositionalIntensityEncoder", "DataRecordEncoder",
    # Memory
    "AssociativeMemory", "ActionPerceptionMemory", "DataRecordMemory",
    "HDCClassifier", "RefineHDLearner",
    # HAP
    "HyperdimensionalActivePerception", "EgoMotionEstimator",
    # Sparse HDC
    "gen_sparse_hvs", "cdt", "sparse_bundle", "sparse_majority",
    "sparse_bind", "gen_sparse_basis", "sparse_similarity",
    "sparse_overlap", "estimate_energy_sparse",
    # Data Structures
    "GraphEncoder", "TreeEncoder", "FSAEncoder",
    "NGramEncoder", "FrequencyEncoder", "StackEncoder",
    # Tension Minimization
    "tension_energy", "minimize_tension", "build_cooc_graph",
    "build_masses", "learn_distributional_hvs",
    # Capacity Analysis
    "binomial_prob", "compute_hamming_statistics", "capacity_curve",
    "find_capacity_limit",
]
