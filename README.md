# Hyperdimensional Active Perception (HAP)

Production-grade implementation of:

> **Mitrokhin, Sutor, Fermüller, Aloimonos (2019)**  
> *"Learning Sensorimotor Control with Neuromorphic Sensors: Toward Hyperdimensional Active Perception"*  
> Science Robotics, vol. 4, no. 30, eaaw6736  
> [DOI: 10.1126/scirobotics.aaw6736](https://doi.org/10.1126/scirobotics.aaw6736)

---

## Paper Figures & Benchmarks

### Figure 8: MVSEC Outdoor Day 1 Results (Core Result)

The HAP framework estimates 3-DOF ego-motion (angular + linear X + linear Z velocity) from DVS time images. Training is single-pass using only 500 frames from outdoor day 1; inference generalizes across all 5 MVSEC subsets (day + night).

![MVSEC Ego-Motion Results](docs/figures/page08_img0.png)

### Figure 7: Information Capacity — Hamming Distance Decay

As more time images are packed into a single 8,000-bit data record memory, the nearest-neighbor match signal decays. Even at 700 frames, matches remain 3–4 standard deviations above random — a single HBV can encode vast history.

![Hamming Distance Decay](docs/figures/page07_img0.png)

### Figure 6: DVS Data Visualization

Left: Time image (green = avg timestamp, red/blue = positive/negative event counts). Right: Corresponding grayscale frame. Note the motion blur on the classical frame — the DVS time image captures motion information that RGB cameras miss.

![DVS vs Classical](docs/figures/page07_img1.jpeg)

### Table 1: Quantitative Results on MVSEC (All 5 Subsets)

| Metric | Outdoor Day 1 | Outdoor Day 2 | Outdoor Night 1 | Outdoor Night 2 | Outdoor Night 3 |
|---|---|---|---|---|---|
| **Frames** | 5,134 | 12,196 | 5,133 | 5,497 | 5,429 |
| **Length (s)** | 128.3 | 304.9 | 128.3 | 137.4 | 135.7 |
| **Ang. bin (rad/s)** | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| **Lin. bin (m/s)** | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 |
| **Rotation (clusters)** | 104 | 101 | 40 | 74 | 87 |
| **X (clusters)** | 47 | 44 | 24 | 40 | 33 |
| **Z (clusters)** | 119 | 311 | 244 | 251 | 228 |
| **AEE** | 0.810 | 1.030 | 0.933 | 1.160 | 0.940 |
| **ARPE** | 0.122 | 0.225 | 0.243 | 0.095 | 0.083 |
| **ARRE** | 0.099 | 0.108 | 0.063 | 0.116 | 0.121 |

**Key result:** Training uses only 500 frames from outdoor day 1 (single pass, 0.5s). Testing generalizes across all 5 subsets including night sequences. Performance is comparable to CNNs trained for 30–50 epochs on 12,196 frames.

### Figure 1–5: Encoding Foundations

| Figure | Description | Page |
|---|---|---|
| **Fig 1** | Tension minimization: energy decay during distributional semantics learning | [Page 4](docs/figures/page_04.png) |
| **Fig 2** | Intensity minimization: 4 intensities form a proportional-distance line in HV space | [Page 5](docs/figures/page_05.png) |
| **Fig 3** | Hamming distance between intensity values 0–25 (distances increase away from diagonal) | [Page 5](docs/figures/page_05.png) |
| **Fig 4** | Moving a pixel spatially via row/column permutations | [Page 6](docs/figures/page_06.png) |
| **Fig 5** | Composing image encodings: arbitrary image assembly via permutation + XOR | [Page 6](docs/figures/page_06.png) |

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

---

## Paper-to-Code Mapping

Every section of the paper is mapped to a specific module in this implementation:

| Paper Section | Content | Implementation | Status |
|---|---|---|---|
| **§ Properties of HBVs** | XOR, Permutation, Consensus Sum (3 operations) | `hap.hdc_core` — `hv_xor`, `hv_permute`, `hv_consensus_sum`, `hv_bundle` | ✅ Full |
| **§ Sets & Sequences** | Set = XOR of elements; Sequence = permute-and-XOR | `hap.encoding.SequenceEncoder` | ✅ Full |
| **§ Ordered Pairs** | `c = P(a) * b` with random P encoding data type | `hap.hdc_core.hv_bind` | ✅ Full |
| **§ Data Records** | `R*V = Σ r_i * v_i` — role-filler binding | `hap.encoding.DataRecordEncoder` | ✅ Full |
| **§ Numerical Values** | Basis vectors with proportional Hamming distances | `hap.encoding.VelocityEncoder` (progressive interpolation) | ✅ Full |
| **§ Categorical + Numerical in Same Space** | Tension minimization: `arg min T(X+ΔX)` with `F_conn` + `F_prox` forces | Simplified: progressive interpolation (`0.7 * prev + 0.3 * random`) | ⚠️ Approximate |
| **§ Encoding Images as HBVs** | Row/col permutations: `R^i(C^j(intensity_HV[i,j]))` | `hap.encoding.PositionalIntensityEncoder` (vectorized) | ✅ Full |
| **§ Creating Memories** | Bundle bound (percept, action) pairs | `hap.memory.AssociativeMemory` | ✅ Full |
| **§ DVS / Neuromorphic Vision** | Time images: average timestamps in (x,y,t) slices | `hap.encoding.TimeSliceEncoder`, `hap.encoding.DVSEncoder` | ✅ Full |
| **§ CNN vs HBV Learning** | PilotNet CNN vs 6-layer NN on HBV encodings (7 vs 9 cm/s) | Not replicated (comparative experiment) | ⬜ Future |
| **§ Perception-Action Binding** | Data record: bind(time_image, velocity) per DOF | `hap.memory.ActionPerceptionMemory`, `hap.hap.EgoMotionEstimator` | ✅ Full |
| **§ Information Capacity** | Seq. length vs Hamming decay (Fig 7) — ~200 frames safe, 700 possible | `hap.memory.DataRecordMemory` (sliding window, capacity configurable) | ✅ Full |
| **§ Theoretical Capacity Limits** | `p(bit=1) = (1-p)·binomial + p·binomial` for n vectors | Embedded in docstrings; not explicitly computed | ⚠️ Docs |
| **§ MVSEC Ego-Motion (Exp 2)** | 3-DOF estimation, p(v_i)=1-H_n(bind(m,v_i),d), Table 1 | `hap.hap.EgoMotionEstimator` (full pipeline) | ✅ Full |
| **§ Training Speed** | 0.5s for 500 frames, 2s for 1500 frames, 572 inf/s for 500 classes | `EgoMotionEstimator.stats` property | ✅ Reported |
| **§ pyhdc Library** | Open-source Python lib for accelerated HBV operations | This entire package | ✅ Full |
| **§ RefineHD (Extended)** | Adaptive refinement for misclassified samples (Verges Boncompte 2025) | `hap.memory.RefineHDLearner` | ✅ Full |

---

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

## Paper Experiment 2: Ego-Motion Estimation

### Algorithm (from the paper)

```
Memory construction (per DOF: angular, linear X, linear Z):
    m = Σ v_i ⊗ (a_i ⊕ b_i ⊕ ...)   for images sharing velocity class v_i

Inference (Equation 4):
    p(v_i) = 1 - H_n(bind(m, v_i), d)

Where:
    - v_i are basis velocity vectors (step = 0.001)
    - d is the query time image encoding
    - bind = XOR (self-inverse: unbind = bind)
    - H_n = normalized Hamming distance
```

### Production Features

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

## Key Results (from paper)

- **Ego-motion estimation**: 572 inferences/second (500 velocity classes)
- **Training time**: 0.5s for 500 frames, 2s for 1500 frames
- **Memory capacity**: ~700 records at D=10,000 before statistical breakdown
- **Hardware efficiency**: XOR = 0.1 pJ/bit vs MAC = 4.6 pJ/op (~46× cheaper)
- **Accuracy**: Comparable to CNNs on MVSEC dataset across all 5 subsets (see Table 1 above)
- **Cross-condition generalization**: Trained on outdoor day 1 (500 frames), tested on day 2 and all 3 night subsets

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

## Paper Coverage Summary

| Category | Paper Content | Covered |
|---|---|---|
| **Core HDC operations** | XOR, permute, consensus sum | ✅ `hdc_core.py` |
| **Data structures** | Sets, ordered pairs, sequences, data records | ✅ `encoding.py` |
| **Image encoding** | Position-intensity bind via row/col permutations | ✅ `PositionalIntensityEncoder` |
| **Velocity encoding** | Basis vectors with proportional Hamming distances | ✅ `VelocityEncoder` |
| **DVS encoding** | Time images, event-level encoding | ✅ `TimeSliceEncoder`, `DVSEncoder` |
| **Associative memory** | Bind(percept, action) → consensus sum | ✅ `AssociativeMemory` |
| **Ego-motion (MVSEC)** | 3-DOF, p(v_i) = 1-H_n(bind(m,v_i),d) | ✅ `EgoMotionEstimator` |
| **Information capacity** | Sequence length vs Hamming decay (Fig 7) | ✅ `DataRecordMemory` |
| **Training speed** | 0.5s/500 frames, 572 inf/s | ✅ `stats` property |
| **Energy model** | 45nm CMOS (Horowitz 2014) | ✅ `estimate_energy_hdv()` |
| **RefineHD** | Adaptive refinement (Verges Boncompte 2025) | ✅ `RefineHDLearner` |
| **Tension minimization** | F_conn + F_prox energy minimization | ⚠️ Approx (progressive interpolation) |
| **CNN comparison** | PilotNet vs 6-layer NN on HBV encodings | ⬜ Not replicated |
| **MVSEC dataset loader** | DAVIS 240/346 event stream → time images | ⬜ Requires MVSEC data |

---

## References

1. Mitrokhin, Sutor, Fermüller, Aloimonos (2019). *Learning Sensorimotor Control with Neuromorphic Sensors: Toward Hyperdimensional Active Perception.* Science Robotics, vol. 4, no. 30, eaaw6736.
2. Kanerva, P. (2009). *Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.* Cognitive Computation.
3. Verges Boncompte, P. (2025). *Classification with Hyperdimensional Computing.* PhD Thesis, UPC Barcelona.
4. Horowitz, M. (2014). *Computing's energy problem (and what we can do about it).* ISSCC.

## License

MIT