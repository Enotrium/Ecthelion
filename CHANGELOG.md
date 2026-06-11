# Changelog

All notable changes to the Hyperdimensional Active Perception (HAP) framework.

## [1.0.0] — 2024-06-11

### Added
- Initial production-grade release
- Core HDC primitives (`hdc_core.py`): XOR, bind, bundle, permute, Hamming distance, consensus sum, batch similarity, hardware energy model
- Encoding pipeline (`encoding.py`): PositionalIntensityEncoder, TimeSliceEncoder, SequenceEncoder, VelocityEncoder, DVSEncoder, DataRecordEncoder
- Associative memory (`memory.py`): AssociativeMemory, ActionPerceptionMemory, DataRecordMemory, HDCClassifier, RefineHDLearner
- Top-level HAP framework (`hap.py`): HyperdimensionalActivePerception, EgoMotionEstimator
- Sparse HDC extension (`sparse_hdc.py`): gen_sparse_hvs, CDT, sparse_bundle, sparse_majority, sparse_bind, sparse similarity
- Data structures (`data_structures.py`): GraphEncoder, TreeEncoder, FSAEncoder, NGramEncoder, FrequencyEncoder, StackEncoder
- Tension minimization (`tension.py`): tension_energy, minimize_tension, learn_distributional_hvs
- Capacity analysis (`capacity.py`): binomial_prob, compute_hamming_statistics, capacity_curve, find_capacity_limit
- Demo scripts: `demo_online_learning.py`, `demo_ego_motion.py`
- 210 tests across 6 test modules

### Production Infrastructure
- Docker multi-stage build (`Dockerfile`)
- Docker Compose for test/demo services (`docker-compose.yml`)
- GitHub Actions CI/CD pipeline (lint, test matrix, build, Docker)
- Custom exception hierarchy (`hap/exceptions.py`)
- Structured logging across all modules
- `.gitignore`, `.dockerignore`, `MANIFEST.in`, `LICENSE`
- `ruf` linting config, `mypy` type checking config, pre-commit hooks