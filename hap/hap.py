"""Hyperdimensional Active Perception (HAP) — Top-Level Framework
===============================================================
"Push the job onto encoding. Purely hardware. Then push encoding
 as far away from actual learning that you can learn very rapidly."

This module ties all components together into a cohesive framework:
    1. HyperdimensionalActivePerception — general HAP system
    2. EgoMotionEstimator — MVSEC-style ego-motion estimation pipeline

The pipeline:
    DVS Events → Time Images → Encode → Bind → Memory → Inference
        │                                          │
        └──── All the heavy work happens here ─────┘
                (purely hardware encodings)
        │                                          │
        └──── Learning is trivial ─────────────────┘
                (just counting co-occurrences)
"""

from __future__ import annotations

import logging
import time

import torch

from hap.encoding import (
    TimeSliceEncoder,
    VelocityEncoder,
)
from hap.hdc_core import (
    estimate_energy_hdv,
    hv_bind,
    hv_hamming_sim,
)
from hap.memory import (
    ActionPerceptionMemory,
    AssociativeMemory,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# HyperdimensionalActivePerception — General Framework
# ═══════════════════════════════════════════════════════════════════════════════


class HyperdimensionalActivePerception:
    """General HAP framework: perception → encoding → memory → action.

    This class orchestrates the full pipeline:
        Sensor data → Encoder(HVs) → Memory(bind) → Inference

    The "active" loop:
        1. Observe: sensor stream → HV encoding
        2. Memorize: bind(percept, action) → consensus sum
        3. Infer: unbind(memory, percept) → nearest action
        4. Act: use inferred action to guide next observation

    Args:
        dim: HV dimensionality (default: 10,000)
        mode: 'binary' or 'bipolar'
        encoder: Optional pre-configured encoder module
        memory: Optional pre-configured memory module
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dim: int = 10_000,
        mode: str = "binary",
        encoder: object | None = None,
        memory: object | None = None,
        seed: int | None = None,
    ):
        if dim < 2:
            raise ValueError(f"HV dimension must be at least 2, got {dim}")
        if mode not in ("binary", "bipolar"):
            raise ValueError(f"Mode must be 'binary' or 'bipolar', got {mode}")

        self.dim = dim
        self.mode = mode
        self.seed = seed or 42

        self.encoder = encoder
        self.memory = memory or AssociativeMemory(dim=dim, mode=mode)

        # Metrics
        self._train_time = 0.0
        self._infer_time = 0.0
        self._n_train = 0
        self._n_infer = 0

    def observe(self, raw_data: torch.Tensor) -> torch.Tensor:
        """Convert raw sensor data to HV using the encoder.

        Args:
            raw_data: Raw sensor data (depends on encoder type)

        Returns:
            (D,) encoded hypervector

        Raises:
            ValueError: if no encoder is configured
        """
        if self.encoder is None:
            raise ValueError(
                "No encoder configured. Set encoder before calling observe()."
                " Example: hap.encoder = TimeSliceEncoder(height=..., width=...)"
            )
        return self.encoder.encode(raw_data)

    def memorize(self, percept: torch.Tensor, action: torch.Tensor) -> None:
        """Store (percept, action) pair in memory.

        Args:
            percept: (D,) observation hypervector
            action: (D,) action hypervector
        """
        t0 = time.perf_counter()
        self.memory.train(percept, action)
        self._train_time += time.perf_counter() - t0
        self._n_train += 1

    def decide(
        self, percept: torch.Tensor, action_candidates: torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        """Infer best action for a percept.

        Args:
            percept: (D,) observation hypervector
            action_candidates: (K, D) candidate action HVs

        Returns:
            (best_action_idx, similarities)
        """
        t0 = time.perf_counter()
        idx, sims = self.memory.infer(percept, action_candidates)
        self._infer_time += time.perf_counter() - t0
        self._n_infer += 1
        return idx, sims

    @property
    def stats(self) -> dict:
        """Get performance statistics.

        Returns:
            Dict with timing, throughput, and energy information
        """
        n_train = max(self._n_train, 1)
        n_infer = max(self._n_infer, 1)

        energy = estimate_energy_hdv(
            dim=self.dim,
            n_xor=self._n_train + self._n_infer,
            n_popcount=self._n_infer * 10,  # ~10 candidates
            n_bundles=self._n_train,
        )

        return {
            "n_train": self._n_train,
            "n_infer": self._n_infer,
            "train_time_total_s": self._train_time,
            "train_time_per_sample_us": self._train_time / n_train * 1e6,
            "infer_time_total_s": self._infer_time,
            "infer_time_per_sample_us": self._infer_time / n_infer * 1e6,
            "train_throughput_hz": n_train / max(self._train_time, 1e-12),
            "infer_throughput_hz": n_infer / max(self._infer_time, 1e-12),
            "energy_nj": energy["total_hdc_energy_nj"],
            "energy_vs_mac_ratio": energy["ratio_mac_to_hdc"],
        }

    def reset(self) -> None:
        """Reset all metrics and memory."""
        self._train_time = 0.0
        self._infer_time = 0.0
        self._n_train = 0
        self._n_infer = 0
        if hasattr(self.memory, "clear"):
            self.memory.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# EgoMotionEstimator — MVSEC-Style Ego-Motion from DVS
# ═══════════════════════════════════════════════════════════════════════════════


class EgoMotionEstimator:
    """Ego-motion estimation from DVS event data using HBVs.

    From the paper (Section "Experiment 2: Ego-motion perception"):
        "We formulate the problem of estimating ego-motion of the
         camera as finding velocities in three degrees of freedom
         (rotational + translational: X and Z) from the DVS stream."

    Pipeline (matches paper Figure 4):
        1. Accumulate DVS events into time images (time slices)
        2. Encode each time image as an HV (TimeSliceEncoder)
        3. Encode ground truth velocities as HVs (VelocityEncoder)
        4. Bind (image_HV, velocity_HV) → memory via consensus sum
        5. For inference, for each velocity class v_i:
           unbound = bind(memory, v_i)  →  recover associated image pattern
           sim = 1 - H_n(unbound, query_image)  →  match to query
           pick argmax_i sim

    This matches the paper's Equation 4:
        m = Σ P^k(v_k) ⊗ i_k
        p(v_i) = P(bind(m, v_i), d) = 1 - H_n(bind(m, v_i), d)

    Args:
        width: DVS sensor width
        height: DVS sensor height
        dim: HV dimensionality
        n_angular_bins: Number of angular velocity classes (paper: 500)
        n_linear_x_bins: Number of X velocity classes (paper: ~47)
        n_linear_z_bins: Number of Z velocity classes (paper: ~119)
        velocity_step: Velocity discretization step (paper: 0.001)
        seed: Random seed
    """

    def __init__(
        self,
        width: int = 346,
        height: int = 260,
        dim: int = 8_000,
        n_angular_bins: int = 500,
        n_linear_x_bins: int = 47,
        n_linear_z_bins: int = 119,
        velocity_step: float = 0.001,
        seed: int | None = None,
    ):
        if dim < 10:
            raise ValueError(f"HV dimension must be at least 10, got {dim}")
        if n_angular_bins < 1 or n_linear_x_bins < 1 or n_linear_z_bins < 1:
            raise ValueError("Number of velocity bins must be positive")

        self.width = width
        self.height = height
        self.dim = dim
        self.velocity_step = velocity_step
        s = seed or 42

        # ── Encoders ─────────────────────────────────────────────────────
        self.time_encoder = TimeSliceEncoder(
            height=height,
            width=width,
            dim=dim,
            seed=s,
        )

        self.angular_encoder = VelocityEncoder(
            min_val=0.0,
            max_val=n_angular_bins * velocity_step,
            step=velocity_step,
            dim=dim,
            seed=s + 1,
        )
        self.linear_x_encoder = VelocityEncoder(
            min_val=0.0,
            max_val=n_linear_x_bins * velocity_step,
            step=velocity_step,
            dim=dim,
            seed=s + 2,
        )
        self.linear_z_encoder = VelocityEncoder(
            min_val=0.0,
            max_val=n_linear_z_bins * velocity_step,
            step=velocity_step,
            dim=dim,
            seed=s + 3,
        )

        # ── Per-class memories: one for each velocity DOF ────────────────
        self.angular_mem = ActionPerceptionMemory(
            n_classes=n_angular_bins,
            dim=dim,
        )
        self.linear_x_mem = ActionPerceptionMemory(
            n_classes=n_linear_x_bins,
            dim=dim,
        )
        self.linear_z_mem = ActionPerceptionMemory(
            n_classes=n_linear_z_bins,
            dim=dim,
        )

        # ── Pre-compute all velocity class HVs for fast inference ────────
        # These are the "v_i" vectors in the paper's p(v_i) = P(bind(m, v_i), d)
        self._angular_class_hvs = torch.stack(
            [self.angular_encoder.encode(i * velocity_step) for i in range(n_angular_bins)]
        )
        self._linear_x_class_hvs = torch.stack(
            [self.linear_x_encoder.encode(i * velocity_step) for i in range(n_linear_x_bins)]
        )
        self._linear_z_class_hvs = torch.stack(
            [self.linear_z_encoder.encode(i * velocity_step) for i in range(n_linear_z_bins)]
        )

        # ── Metrics ─────────────────────────────────────────────────────
        self._train_time = 0.0
        self._infer_count = 0
        self._total_train_samples = 0

    def encode_time_image(self, time_image: torch.Tensor) -> torch.Tensor:
        """Encode a DVS time image into an HV.

        Args:
            time_image: (H, W) tensor of timestamps

        Returns:
            (D,) encoded HV
        """
        return self.time_encoder.encode_time_slice(time_image)

    def encode_velocity(
        self, angular: float, linear_x: float, linear_z: float
    ) -> tuple[torch.Tensor, ...]:
        """Encode 3-DOF velocity into three HVs.

        Args:
            angular: Angular velocity (rad/s)
            linear_x: Linear velocity X (m/s)
            linear_z: Linear velocity Z (m/s)

        Returns:
            (angular_hv, linear_x_hv, linear_z_hv) as tensor tuple
        """
        return (
            self.angular_encoder.encode(angular),
            self.linear_x_encoder.encode(linear_x),
            self.linear_z_encoder.encode(linear_z),
        )

    def train(
        self, time_image: torch.Tensor, angular: float, linear_x: float, linear_z: float
    ) -> None:
        """Train on one (time_image, velocity) sample.

        The paper's training rule:
            m += bind(image_HV, velocity_HV)   [per DOF]

        For each velocity component, the image HV is bound to the
        velocity class HV and accumulated in that component's per-class
        memory. This creates a consensus sum per velocity class.

        Args:
            time_image: (H, W) time image
            angular: Angular velocity (rad/s)
            linear_x: Linear velocity X (m/s)
            linear_z: Linear velocity Z (m/s)
        """
        t0 = time.perf_counter()

        # Encode perception
        img_hv = self.encode_time_image(time_image)
        vel_hvs = self.encode_velocity(angular, linear_x, linear_z)

        # Discretize to class indices
        ang_idx = self._velocity_to_idx(angular, self.angular_mem.n_classes)
        lx_idx = self._velocity_to_idx(linear_x, self.linear_x_mem.n_classes)
        lz_idx = self._velocity_to_idx(linear_z, self.linear_z_mem.n_classes)

        # Bind (image, velocity) for each DOF and store in per-class memories
        # m[i] += bind(image, velocity_i) for class i
        bound_ang = hv_bind(img_hv, vel_hvs[0])
        bound_lx = hv_bind(img_hv, vel_hvs[1])
        bound_lz = hv_bind(img_hv, vel_hvs[2])

        self.angular_mem.train(bound_ang, ang_idx)
        self.linear_x_mem.train(bound_lx, lx_idx)
        self.linear_z_mem.train(bound_lz, lz_idx)

        self._train_time += time.perf_counter() - t0
        self._total_train_samples += 1

    def infer(self, time_image: torch.Tensor) -> dict[str, float]:
        """Infer velocity from a time image.

        From the paper (Equation 4):
            p(v_i) = 1 - H_n(bind(memory, v_i), d)

        For each velocity class v_i:
            unbound = bind(thresholded_memory, v_i)
            similarity = 1 - H_n(unbound, query_image)

        This unbinds the velocity class key from the memory, recovering
        the expected image pattern for that velocity. We then compare
        this to the actual query image.

        Args:
            time_image: (H, W) time image

        Returns:
            Dict with keys 'angular', 'linear_x', 'linear_z' velocities
            and confidence probabilities
        """
        img_hv = self.encode_time_image(time_image)

        # Paper's inference: p(v_i) = 1 - H_n(bind(m, v_i), d)
        # Threshold memories to binary for clean unbinding
        ang_mem_thresh = self.angular_mem._threshold_memory()
        lx_mem_thresh = self.linear_x_mem._threshold_memory()
        lz_mem_thresh = self.linear_z_mem._threshold_memory()

        # Angular: find best velocity class
        # _threshold_memory() returns per-class memories as (n_classes, D).
        # For each class i, unbind the class-specific memory with the velocity
        # key v_i to recover the expected image pattern, then compare to query.
        ang_sims = torch.zeros(self.angular_mem.n_classes, device=img_hv.device)
        for i in range(self.angular_mem.n_classes):
            if self.angular_mem._class_counts[i] > 0:
                unbound = hv_bind(ang_mem_thresh[i], self._angular_class_hvs[i])
                ang_sims[i] = hv_hamming_sim(unbound, img_hv)
        ang_best = ang_sims.argmax().item()

        # Linear X
        lx_sims = torch.zeros(self.linear_x_mem.n_classes, device=img_hv.device)
        for i in range(self.linear_x_mem.n_classes):
            if self.linear_x_mem._class_counts[i] > 0:
                unbound = hv_bind(lx_mem_thresh[i], self._linear_x_class_hvs[i])
                lx_sims[i] = hv_hamming_sim(unbound, img_hv)
        lx_best = lx_sims.argmax().item()

        # Linear Z
        lz_sims = torch.zeros(self.linear_z_mem.n_classes, device=img_hv.device)
        for i in range(self.linear_z_mem.n_classes):
            if self.linear_z_mem._class_counts[i] > 0:
                unbound = hv_bind(lz_mem_thresh[i], self._linear_z_class_hvs[i])
                lz_sims[i] = hv_hamming_sim(unbound, img_hv)
        lz_best = lz_sims.argmax().item()

        self._infer_count += 1

        return {
            "angular": ang_best * self.velocity_step,
            "linear_x": lx_best * self.velocity_step,
            "linear_z": lz_best * self.velocity_step,
            "angular_prob": ang_sims[ang_best].item(),
            "linear_x_prob": lx_sims[lx_best].item(),
            "linear_z_prob": lz_sims[lz_best].item(),
        }

    def infer_batch(self, time_images: torch.Tensor) -> list[dict]:
        """Infer velocities for a batch of time images.

        Args:
            time_images: (B, H, W) batch of time images

        Returns:
            List of B velocity dicts
        """
        return [self.infer(img) for img in time_images]

    def _velocity_to_idx(self, velocity: float, n_classes: int) -> int:
        """Convert velocity to class index with bounds checking."""
        idx = round(velocity / self.velocity_step)
        return max(0, min(idx, n_classes - 1))

    @property
    def stats(self) -> dict:
        """Get detailed performance statistics."""
        train_samples = max(self._total_train_samples, 1)
        return {
            "dim": self.dim,
            "n_angular": self.angular_mem.n_classes,
            "n_linear_x": self.linear_x_mem.n_classes,
            "n_linear_z": self.linear_z_mem.n_classes,
            "train_samples": self._total_train_samples,
            "train_time_s": self._train_time,
            "train_time_per_sample_us": self._train_time / train_samples * 1e6,
            "infer_count": self._infer_count,
            "train_throughput_hz": train_samples / max(self._train_time, 1e-12),
        }

    def save(self, path: str) -> None:
        """Save estimator state to disk."""
        torch.save(
            {
                "angular_mem": {
                    "memories": self.angular_mem._class_memories,
                    "counts": self.angular_mem._class_counts,
                },
                "linear_x_mem": {
                    "memories": self.linear_x_mem._class_memories,
                    "counts": self.linear_x_mem._class_counts,
                },
                "linear_z_mem": {
                    "memories": self.linear_z_mem._class_memories,
                    "counts": self.linear_z_mem._class_counts,
                },
                "config": {
                    "width": self.width,
                    "height": self.height,
                    "dim": self.dim,
                    "velocity_step": self.velocity_step,
                },
                "total_train_samples": self._total_train_samples,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load estimator state from disk."""
        data = torch.load(path, map_location="cpu")
        self.angular_mem._class_memories = data["angular_mem"]["memories"]
        self.angular_mem._class_counts = data["angular_mem"]["counts"]
        self.linear_x_mem._class_memories = data["linear_x_mem"]["memories"]
        self.linear_x_mem._class_counts = data["linear_x_mem"]["counts"]
        self.linear_z_mem._class_memories = data["linear_z_mem"]["memories"]
        self.linear_z_mem._class_counts = data["linear_z_mem"]["counts"]
        self._total_train_samples = data.get("total_train_samples", 0)
        logger.info(f"Loaded estimator from {path} ({self._total_train_samples} samples)")
