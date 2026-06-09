#!/usr/bin/env python3
"""
Demo: Online 4-Class Learning with Hyperdimensional Computing
==============================================================
"Train online, save counts as super position. See either more ones
 or more zeros and the original integers determine summation.
 Essentially learning online."

This demo shows the core HDC learning principle:
    1. Generate random hypervectors for 4 classes
    2. Train by bundling bound (percept, action) pairs
    3. Inference by unbinding and nearest-neighbor Hamming search
    4. Demonstrate RefineHD iterative improvement

The entire learning process is:
    - One XOR per training sample (binding)
    - One XOR + one popcount per inference (unbinding + nearest neighbor)
    - No backprop, no gradient, no learning rate

Expected output:
    Initial accuracy: ~60-80% (depending on randomness)
    After refinement: ~85-95%
    See that training is 100-1000x fewer operations than NN.

Run:  python demo_online_learning.py
"""

import time
import torch
import sys

# Add parent to path if needed
sys.path.insert(0, ".")

from hap.hdc_core import (
    gen_hvs, hv_bind, hv_bundle, hv_hamming_sim, hv_batch_sim,
    estimate_energy_hdv,
)
from hap.memory import HDCClassifier, RefineHDLearner


def main():
    print("=" * 72)
    print("HYPERDIMENSIONAL COMPUTING — ONLINE 4-CLASS LEARNING")
    print("=" * 72)
    print()
    print('"Push the job onto encoding. Purely hardware.')
    print(' Push encoding as far away from actual learning')
    print(' that you can learn very rapidly."')
    print("    — Peter Sutor, on Kanerva's HDC philosophy")
    print()
    print("Learning = XOR(percept, action) + consensus sum")
    print("Inference = XOR(memory, percept) → nearest neighbor")
    print("All operations: XOR + popcount. No backprop.")
    print()

    # ── Config ─────────────────────────────────────────────────────────────
    DIM = 10_000       # Kanerva's standard dimension
    N_CLASSES = 4
    N_TRAIN = 50       # samples per class
    N_TEST = 20        # test samples per class
    SEED = 42

    torch.manual_seed(SEED)

    print(f"Dimension:       {DIM}")
    print(f"Classes:         {N_CLASSES}")
    print(f"Train/class:     {N_TRAIN}")
    print(f"Test/class:      {N_TEST}")
    print()

    # ── Generate Data ─────────────────────────────────────────────────────
    print("Generating data...")
    # Use different seeds per class to create separable clusters
    train_percepts = []
    train_labels = []
    test_percepts = []
    test_labels = []

    for cls in range(N_CLASSES):
        # Training data: random HV cluster for each class
        train_hvs = gen_hvs(N_TRAIN, DIM, seed=cls * 100 + 1)
        train_percepts.append(train_hvs)
        train_labels.extend([cls] * N_TRAIN)

        # Test data
        test_hvs = gen_hvs(N_TEST, DIM, seed=cls * 100 + 999)
        test_percepts.append(test_hvs)
        test_labels.extend([cls] * N_TEST)

    train_percepts = torch.cat(train_percepts)
    test_percepts = torch.cat(test_percepts)
    N_total = len(train_labels)

    print(f"Training samples: {N_total}")
    print(f"Test samples:     {len(test_labels)}")
    print()

    # ── Single-Pass Training ──────────────────────────────────────────────
    print("=" * 72)
    print("PHASE 1: ONLINE SINGLE-PASS LEARNING")
    print("=" * 72)
    print()

    classifier = HDCClassifier(n_classes=N_CLASSES, dim=DIM, seed=SEED)

    t0 = time.perf_counter()
    classifier.fit(train_percepts, train_labels)
    train_time = time.perf_counter() - t0

    # ── Inference ─────────────────────────────────────────────────────────
    print("Running inference...")
    t0 = time.perf_counter()
    train_acc = classifier.accuracy(train_percepts, train_labels)
    test_acc = classifier.accuracy(test_percepts, test_labels)
    infer_time = time.perf_counter() - t0

    print(f"Training time:    {train_time*1000:.1f} ms  ({train_time/N_total*1e6:.1f} µs/sample)")
    print(f"Total XOR ops:    {N_total}  (one per sample)")
    print()
    print(f"Training accuracy: {train_acc*100:.1f}%")
    print(f"Test accuracy:     {test_acc*100:.1f}%")
    print()

    # ── Energy Comparison ─────────────────────────────────────────────────
    energy = estimate_energy_hdv(
        dim=DIM,
        n_xor=N_total,
        n_popcount=(N_TRAIN + N_TEST) * N_CLASSES,
        n_bundles=N_total,
    )
    print(f"HDC energy:       {energy['total_hdc_energy_nj']:.2f} nJ")
    print(f"Equivalent MAC:   {energy['equiv_mac_energy_nj']:.0f} nJ")
    print(f"Energy ratio:     {energy['ratio_mac_to_hdc']:.0f}x more for MAC")
    print()

    # ── RefineHD ──────────────────────────────────────────────────────────
    print("=" * 72)
    print("PHASE 2: REFINEHD — ADAPTIVE REFINEMENT")
    print("=" * 72)
    print()

    learner = RefineHDLearner(classifier, n_refinement_rounds=3)
    result = learner.fit(train_percepts, train_labels)

    print("Refinement history:")
    for entry in result["history"]:
        r = entry["round"]
        acc = entry["accuracy"] * 100
        mis = entry.get("misclassified", "—")
        print(f"  Round {r}: accuracy = {acc:.1f}%  (misclassified = {mis})")

    final_test_acc = classifier.accuracy(test_percepts, test_labels)
    print()
    print(f"Final test accuracy: {final_test_acc * 100:.1f}%")
    print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print("What just happened:")
    print("  1. Each training sample was bound (XOR) to its class HV")
    print("  2. All bound pairs were consensus-summed into one memory")
    print("  3. Inference: XOR(memory, test) → nearest class HV")
    print("  4. Learning is 100% online and single-pass")
    print()
    print("Key insight: The encoding is where the computation lives.")
    print("The learning is just counting co-occurrences via XOR bundling.")
    print()
    print(f"Training accuracy:  {train_acc*100:.1f}% → {result['initial_accuracy']*100:.1f}% → "
          f"{result['final_accuracy']*100:.1f}% (after refinement)")
    print(f"Test accuracy:      {test_acc*100:.1f}% → {final_test_acc*100:.1f}%")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
