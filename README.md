# Hyperdimensional Active Perception (HAP)

Production-grade implementation of:

> **Mitrokhin, Sutor, Fermüller, Aloimonos (2019)**  
> *"Learning Sensorimotor Control with Neuromorphic Sensors: Toward Hyperdimensional Active Perception"*  
> Science Robotics, vol. 4, no. 30, eaaw6736  
> [DOI: 10.1126/scirobotics.aaw6736](https://doi.org/10.1126/scirobotics.aaw6736)

---

## Core Philosophy

> *"Push the job onto encoding. Purely hardware. Then push encoding as far away from actual learning that you can learn very rapidly."*  
> — **Peter Sutor**, on Kanerva's HDC philosophy

Hyperdimensional Computing (HDC) flips the traditional AI pipeline:

| Traditional NN/Deep Learning | HDC (this framework) |
|---|---|
| Complex architecture search | Fixed encoding pipeline |
| Backpropagation through time | Single-pass consensus sum |
| Billions of multiply-accumulate (MAC) ops | Simple XOR + popcount |
| Hours/days of GPU training | Milliseconds of CPU training |
| High energy consumption | **46× less energy per op** |
| Gradient descent | Literally just counting |

**The insight:** Make encoding the expensive part (push it to hardware). Then learning is trivial — just bundle bound pairs of (perception, action) into a consensus sum. Inference is XOR + popcount — "see either more ones or more zeros."

## Production Readiness

| Feature | Status |
|---|---|
| Correct hv_majority thresholding (binary: > 0.5; bipolar: sign) | ✅ |
| Mode propagation through bundle/bind/inference pipeline | ✅ |
| Vectorized PositionalIntensityEncoder (gather-based, no pixel loops) | ✅ |
| Paper-aligned inference: p(v_i) = 1 - H_n(bind(m, v_i), d) | ✅ |
| Input validation with descriptive errors | ✅ |
| Deterministic seed chains for reproducibility | ✅ |
| Save/load serialization for all memory types | ✅ |
| Hardware energy model (45nm CMOS, Horowitz 2014) | ✅ |
| Comprehensive test suite (hdc_core, encoding, memory, integration) | ✅ |
| RefineHD adaptive refinement (Verges Boncompte 2025) | ✅ |
| 3-DOF ego-motion estimation (paper Experiment 2) | ✅ |

## Architecture

```
Sensor Stream (DVS events, images, velocities)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  HDC Encoder                                     │
│  (Position keys, intensity keys, permutations,  │
│   fractional power, velocity basis vectors,      │
│   temporal sequences via permute-and-XOR)        │
│                                                   │
│  "Push the job onto encoding"                     │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Associative Memory                               │
│  memory = Σ bind(percept, action)                │
│  "Save counts as super position"                  │
│  "Train online"                                    │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Inference (Equation 4)                           │
│  p(v_i) = 1 - H_n(bind(m, v_i), d)              │
│  For Classifier:                                  │
│  unbound = XOR(memory, percept)                  │
│  class = argmin Hamming(unbound, candidates)     │
│  "See either more ones or more zeros"             │
└─────────────────────────────────────────────────┘
    │
    ▼
Prediction (velocity, class, action)
```

## Quick Start

```bash
# Install
pip install -e .

# Run the 4-class online learning demo (60 seconds)
python demo_online_learning.py

# Run the ego-motion estimation demo (paper Experiment 2)
python demo_ego_motion.py

# Run tests
pytest tests/ -v
```

## Module Reference

| Module | Description |
|---|---|
| `hap.hdc_core` | Pure binary HDC primitives: XOR, bind, bundle, permute, Hamming distance, energy model |
| `hap.encoding` | All encoding schemes from the paper (positional, temporal, velocity, DVS) — vectorized |
| `hap.memory` | Associative memory, classifier, RefineHD adaptive learning, save/load |
| `hap.hap` | Top-level framework: HAP system and paper-aligned ego-motion estimator |

### `hap.hdc_core` — Core Primitives

```python
from hap.hdc_core import gen_hvs, hv_xor, hv_bundle, hv_permute, hv_hamming_sim

# Generate random hypervectors (nearly orthogonal)
hvs = gen_hvs(n=10, dim=10_000, seed=42)

# Bind (XOR) two HVs — the fundamental operation
bound = hv_xor(hvs[0], hvs[1])

# Bundle a set — consensus sum / majority vote
bundled = hv_bundle(hvs)

# Permute — temporal sequence encoding
shifted = hv_permute(hvs[0], k=5)

# Hamming similarity — 0.5 = random, 1.0 = identical
sim = hv_hamming_sim(hvs[0], hvs[1])
```

### `hap.encoding` — Encoding (The Heavy Part)

```python
from hap.encoding import (
    PositionalIntensityEncoder,  # 2D images → HVs
    TimeSliceEncoder,            # DVS time slices → HVs
    VelocityEncoder,             # Continuous velocities → basis HVs
    SequenceEncoder,             # Temporal sequences → HVs
    DVSEncoder,                  # Raw DVS events → HVs
    DataRecordEncoder,           # Multi-field records → HVs
)
```

### `hap.memory` — Learning (The Trivial Part)

```python
from hap.memory import HDCClassifier, RefineHDLearner

clf = HDCClassifier(n_classes=4, dim=10_000)
clf.fit(percepts, labels)  # Single pass, no backprop
pred = clf.predict(new_percept)  # XOR + popcount
acc = clf.accuracy(test_percepts, test_labels)

# RefineHD: adaptive refinement for misclassified samples
learner = RefineHDLearner(clf, n_refinement_rounds=3)
result = learner.fit(percepts, labels)
```

### `hap.hap` — Ego-Motion Estimation (Paper Experiment 2)

```python
from hap.hap import EgoMotionEstimator

est = EgoMotionEstimator(
    width=346, height=260, dim=8_000,
    n_angular_bins=500, n_linear_x_bins=47, n_linear_z_bins=119,
    velocity_step=0.001,
)

# Train: single pass
for time_image, angular_v, linear_x_v, linear_z_v in dataset:
    est.train(time_image, angular_v, linear_x_v, linear_z_v)

# Infer: p(v_i) = 1 - H_n(bind(m, v_i), d)
result = est.infer(new_time_image)
# {'angular': 0.123, 'linear_x': 0.045, 'linear_z': 0.089, ...}
```

## Key Results (from paper)

- **Ego-motion estimation**: 572 inferences/second (500 velocity classes)
- **Training time**: 0.5s for 500 frames, 2s for 1500 frames
- **Memory capacity**: ~700 records at D=10,000 before statistical breakdown
- **Hardware efficiency**: XOR = 0.1 pJ/bit vs MAC = 4.6 pJ/op (~46× cheaper)
- **Accuracy**: Comparable to CNNs on MVSEC dataset across all 5 subsets

## Demos

### `demo_online_learning.py`
Demonstrates the core HDC learning principle:
- Random hypervectors for 4 classes
- Single-pass training via XOR bundling
- Inference via Hamming distance nearest-neighbor
- RefineHD adaptive refinement

### `demo_ego_motion.py`
Demonstrates the paper's Experiment 2:
- Synthetic DVS time images with 3-DOF motion artifacts
- Angular + linear X + linear Z velocity estimation
- Paper-aligned inference: p(v_i) = 1 - H_n(bind(m, v_i), d)
- Hardware energy analysis

## Hardware Properties

| Operation | Energy (45nm CMOS) | Compared to MAC |
|---|---|---|
| XOR | 0.1 pJ/bit | 46× cheaper |
| Popcount | 0.2 pJ/op | 23× cheaper |
| Integer add | 0.05 pJ/bit | 92× cheaper |
| Permute | 0.01 pJ/bit | 460× cheaper |
| **MAC** | **4.6 pJ/op** | **baseline** |

## References

1. Mitrokhin, Sutor, Fermüller, Aloimonos (2019). *Learning Sensorimotor Control with Neuromorphic Sensors: Toward Hyperdimensional Active Perception.* Science Robotics, vol. 4, no. 30, eaaw6736.
2. Kanerva, P. (2009). *Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.* Cognitive Computation.
3. Verges Boncompte, P. (2025). *Classification with Hyperdimensional Computing.* PhD Thesis, UPC Barcelona.

## License

MIT