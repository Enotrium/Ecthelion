#!/usr/bin/env python3
"""
Demo: Synthetic Ego-Motion Estimation (MVSEC-Style)
====================================================
"Push the job onto encoding. Purely hardware. Then push encoding
 as far away from actual learning that you can learn very rapidly."

This demonstrates the paper's Experiment 2: Ego-motion perception
from DVS event data, using synthetic time images.

The pipeline mirrors the paper's Figure 4 and Equation 4:
    Events → Time Image → TimeSliceEncoder(HV) → Bind(Velocity) → Memory
    
    Memory = Σ P^k(v_k) ⊗ i_k
    p(v_i) = 1 - H_n(bind(m, v_i), d)

Training: 200 samples, single pass, < 2 seconds (small dim)
Inference: paper's Eq. 4 via unbind-and-match

Run:  python demo_ego_motion.py
"""

import time
import torch
import sys

sys.path.insert(0, ".")

from hap.hap import EgoMotionEstimator
from hap.hdc_core import estimate_energy_hdv


def make_synthetic_time_image(height: int = 260, width: int = 346,
                               velocity: float = 0.0,
                               x_velocity: float = 0.0,
                               z_velocity: float = 0.0,
                               seed: int = 0) -> torch.Tensor:
    """Create a synthetic DVS time image with motion artifacts.

    Simulates what a DVS time image looks like for a given velocity.
    The motion creates timestamp gradients in the direction of motion.

    Args:
        height, width: Image dimensions
        velocity: Simulated angular velocity (affects vertical gradient)
        x_velocity: Simulated X velocity (affects horizontal gradient)
        z_velocity: Simulated Z velocity (affects scale gradient)
        seed: Random seed for noise

    Returns:
        (H, W) tensor of timestamps in [0, 1]
    """
    torch.manual_seed(seed)
    img = torch.zeros(height, width)

    # Angular velocity → vertical gradient
    v_dir_y = max(-1.0, min(1.0, velocity * 5))
    # X velocity → horizontal gradient
    v_dir_x = max(-1.0, min(1.0, x_velocity * 10))
    # Z velocity → radial/scale gradient (motion toward/away from camera)
    v_z = max(-1.0, min(1.0, z_velocity * 10))

    num_events = int(500 + (abs(v_dir_y) + abs(v_dir_x) + abs(v_z)) * 800)

    for _ in range(num_events):
        # Bias positions toward velocity direction
        x = int((torch.rand(1).item() * 0.6 + 0.2 + v_dir_x * 0.2) * width)
        y_offset = int(v_dir_y * 25 * (torch.rand(1).item() - 0.5))
        y = int((height / 2) + y_offset)
        y = max(0, min(height - 1, y))
        x = max(0, min(width - 1, x))

        # Timestamp: linear gradient based on position
        t = (x / width * (0.5 + v_dir_x * 0.5) +
             y / height * (0.5 + v_dir_y * 0.5))
        img[y, x] = t

    # Add small noise
    noise = torch.randn(height, width) * 0.005
    img = img + noise
    img = img.clamp(0, 1)

    return img


def main():
    print("=" * 72)
    print("SYNTHETIC EGO-MOTION ESTIMATION (MVSEC-STYLE)")
    print("=" * 72)
    print()
    print("From Mitrokhin, Sutor, Fermüller, Aloimonos (2019):")
    print("  'Learning Sensorimotor Control with Neuromorphic")
    print("   Sensors: Toward Hyperdimensional Active Perception'")
    print("  Science Robotics, vol. 4, no. 30, eaaw6736")
    print()
    print("  Memory = Σ P^k(v_k) ⊗ i_k")
    print("  p(v_i) = 1 - H_n(bind(m, v_i), d)")
    print()

    # ── Config ────────────────────────────────────────────────────────────
    # Use moderate dimensions for demo speed
    WIDTH = 40
    HEIGHT = 30
    DIM = 2_000
    N_TRAIN = 200
    N_TEST = 30
    VELOCITY_STEP = 0.005  # rad/s or m/s steps
    N_ANGULAR = 40
    N_LINEAR_X = 15
    N_LINEAR_Z = 15

    print(f"Sensor size:      {WIDTH}x{HEIGHT}")
    print(f"HV dimension:     {DIM}")
    print(f"Training samples: {N_TRAIN}")
    print(f"Test samples:     {N_TEST}")
    print(f"Angular bins:     {N_ANGULAR}")
    print(f"LinearX bins:     {N_LINEAR_X}")
    print(f"LinearZ bins:     {N_LINEAR_Z}")
    print()

    # ── Initialize ────────────────────────────────────────────────────────
    print("Initializing estimators...")
    estimator = EgoMotionEstimator(
        width=WIDTH,
        height=HEIGHT,
        dim=DIM,
        n_angular_bins=N_ANGULAR,
        n_linear_x_bins=N_LINEAR_X,
        n_linear_z_bins=N_LINEAR_Z,
        velocity_step=VELOCITY_STEP,
        seed=42,
    )
    print("  Done.")
    print()

    # ── Training ──────────────────────────────────────────────────────────
    print("=" * 72)
    print("TRAINING: Single-pass online learning (Equation 4)")
    print("=" * 72)
    print()

    t0 = time.perf_counter()
    for i in range(N_TRAIN):
        # Generate synthetic velocity
        ang = (i % N_ANGULAR) * VELOCITY_STEP
        lx = (i % N_LINEAR_X) * VELOCITY_STEP
        lz = (i % N_LINEAR_Z) * VELOCITY_STEP

        # Generate corresponding time image
        img = make_synthetic_time_image(
            HEIGHT, WIDTH,
            velocity=ang,
            x_velocity=lx,
            z_velocity=lz,
            seed=i * 100,
        )

        estimator.train(img, ang, lx, lz)

    train_time = time.perf_counter() - t0

    print(f"Training time:    {train_time*1000:.1f} ms")
    print(f"Per sample:       {train_time / max(N_TRAIN, 1) * 1e6:.1f} µs")
    print(f"Memory stores:    {N_TRAIN} bound (image, velocity) pairs per DOF")
    print()

    # ── Inference ────────────────────────────────────────────────────────
    print("=" * 72)
    print("INFERENCE: p(v_i) = 1 - H_n(bind(m, v_i), d)")
    print("=" * 72)
    print()

    errors = {"angular": 0.0, "linear_x": 0.0, "linear_z": 0.0}
    correct = {"angular": 0, "linear_x": 0, "linear_z": 0}
    count = 0

    t0 = time.perf_counter()
    for i in range(N_TEST):
        # Generate test velocity (different seed)
        ang = ((i + 42) % N_ANGULAR) * VELOCITY_STEP
        lx = ((i + 42) % N_LINEAR_X) * VELOCITY_STEP
        lz = ((i + 42) % N_LINEAR_Z) * VELOCITY_STEP

        img = make_synthetic_time_image(
            HEIGHT, WIDTH,
            velocity=ang,
            x_velocity=lx,
            z_velocity=lz,
            seed=i * 100 + 9999,
        )

        result = estimator.infer(img)

        errors["angular"] += abs(result["angular"] - ang)
        errors["linear_x"] += abs(result["linear_x"] - lx)
        errors["linear_z"] += abs(result["linear_z"] - lz)
        if abs(result["angular"] - ang) < VELOCITY_STEP * 0.5:
            correct["angular"] += 1
        if abs(result["linear_x"] - lx) < VELOCITY_STEP * 0.5:
            correct["linear_x"] += 1
        if abs(result["linear_z"] - lz) < VELOCITY_STEP * 0.5:
            correct["linear_z"] += 1
        count += 1

    infer_time = time.perf_counter() - t0

    print(f"Inference count:  {count}")
    print(f"Inference time:   {infer_time*1000:.1f} ms total")
    print(f"Per inference:    {infer_time / max(count, 1) * 1e6:.1f} µs")
    print(f"Inferences/sec:   {count / max(infer_time, 1e-6):.0f}")
    print()

    print("Error analysis:")
    for comp in ["angular", "linear_x", "linear_z"]:
        mae = errors[comp] / max(count, 1)
        acc = correct[comp] / max(count, 1) * 100
        print(f"  {comp:>12}: MAE={mae:.4f}  Acc={acc:.1f}%  (step={VELOCITY_STEP})")

    # Show example predictions
    print()
    print("Example predictions (first 5):")
    for i in range(min(5, N_TEST)):
        ang = ((i + 42) % N_ANGULAR) * VELOCITY_STEP
        lx = ((i + 42) % N_LINEAR_X) * VELOCITY_STEP
        lz = ((i + 42) % N_LINEAR_Z) * VELOCITY_STEP

        img = make_synthetic_time_image(
            HEIGHT, WIDTH, velocity=ang,
            x_velocity=lx, z_velocity=lz,
            seed=i * 100 + 9999,
        )
        r = estimator.infer(img)
        status_ang = "✓" if abs(r["angular"] - ang) < VELOCITY_STEP * 0.5 else "✗"
        status_lx = "✓" if abs(r["linear_x"] - lx) < VELOCITY_STEP * 0.5 else "✗"
        status_lz = "✓" if abs(r["linear_z"] - lz) < VELOCITY_STEP * 0.5 else "✗"
        print(f"  #{i}: true=[{ang:.3f},{lx:.3f},{lz:.3f}] "
              f"pred=[{r['angular']:.3f},{r['linear_x']:.3f},{r['linear_z']:.3f}] "
              f"({status_ang}{status_lx}{status_lz}) "
              f"p=[{r['angular_prob']:.3f},{r['linear_x_prob']:.3f},{r['linear_z_prob']:.3f}]")

    print()

    # ── Energy Analysis ──────────────────────────────────────────────────
    print("=" * 72)
    print("HARDWARE ENERGY ANALYSIS (45nm CMOS, Horowitz ISSCC 2014)")
    print("=" * 72)
    print()

    energy = estimate_energy_hdv(
        dim=DIM,
        n_xor=N_TRAIN + count * 3,
        n_popcount=count * (N_ANGULAR + N_LINEAR_X + N_LINEAR_Z),
        n_bundles=N_TRAIN,
    )
    print(f"HDC total energy: {energy['total_hdc_energy_nj']:.3f} nJ")
    print(f"Equivalent MAC:   {energy['equiv_mac_energy_nj']:.1f} nJ")
    print(f"Energy ratio:     {energy['ratio_mac_to_hdc']:.0f}x more for NN")
    print()
    print("Per-sample breakdown:")
    print(f"  XOR binding:     {energy['xor_energy_pj']:.1f} pJ")
    print(f"  Popcount:        {energy['popcount_energy_pj']:.1f} pJ")
    print(f"  Bundle (accum):  {energy['bundle_energy_pj']:.1f} pJ")
    print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print("  'Push the job onto encoding. Purely hardware.")
    print("   Push encoding as far away from actual learning")
    print("   that you can learn very rapidly.'")
    print()
    print(f"  Training: {N_TRAIN} samples in {train_time*1000:.1f}ms (single pass)")
    print(f"  Inference: {count / max(infer_time, 1e-6):.0f} samples/sec")
    print(f"  Energy: {energy['ratio_mac_to_hdc']:.0f}x more efficient than NN equivalent")
    print()
    angular_acc = correct["angular"] / max(count, 1) * 100
    lx_acc = correct["linear_x"] / max(count, 1) * 100
    lz_acc = correct["linear_z"] / max(count, 1) * 100
    print(f"  Accuracy (within {VELOCITY_STEP/2}):")
    print(f"    Angular: {angular_acc:.0f}%")
    print(f"    LinearX: {lx_acc:.0f}%")
    print(f"    LinearZ: {lz_acc:.0f}%")
    print()

    print("Done.")


if __name__ == "__main__":
    main()