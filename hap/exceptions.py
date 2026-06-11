"""
HAP Exception Hierarchy — Domain-Specific Error Types
=======================================================
All exceptions raised by the HAP library inherit from `HAPError`,
making it easy for callers to catch all HAP-specific errors with
a single except clause.
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════════
# Base Exception
# ═══════════════════════════════════════════════════════════════════════════════


class HAPError(Exception):
    """Base exception for all HAP-internal errors."""


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Errors
# ═══════════════════════════════════════════════════════════════════════════════


class HAPConfigError(HAPError):
    """Invalid or missing configuration."""


class HAPDimensionError(HAPConfigError):
    """Dimensionality mismatch between hypervectors or encoders.

    Raised when:
        - Two HVs of different D are XOR'd/bound/compared
        - An encoder receives input with unexpected shape
        - A memory is probed with a vector of wrong dimension
    """


class HAPModeError(HAPConfigError):
    """Unsupported or mismatched HV mode (binary vs bipolar)."""


class HAPSeedError(HAPConfigError):
    """Seed-related configuration issue (e.g., conflicting seeds)."""


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime / Operation Errors
# ═══════════════════════════════════════════════════════════════════════════════


class HAPRuntimeError(HAPError):
    """Error during HAP computation or inference."""


class HAPEncodingError(HAPRuntimeError):
    """Error encoding a sensor sample into a hypervector.

    Raised when:
        - Input data is malformed or out-of-range for the encoder
        - A required basis vector has not been generated
        - An encoder receives unexpected dtype
    """


class HAPMemoryError(HAPRuntimeError):
    """Error related to associative memory operations.

    Raised when:
        - Inference is attempted on an empty (untrained) memory
        - Save/load encounters corrupted or incompatible data
        - Memory capacity is exceeded
    """


class HAPInferenceError(HAPRuntimeError):
    """Error during the inference step (unbind + nearest-neighbour).

    Raised when:
        - Candidate set is empty or malformed
        - Query vector shape is incompatible with stored memory
    """


class HAPCapacityError(HAPRuntimeError):
    """Statistical confidence in inference result is below threshold.

    This is a *non-fatal* error — it signals that the model should
    fall back to a safe default action or request more data rather
    than trusting a low-confidence prediction.
    """

    def __init__(self, message: str = "", z_score: float = 0.0):
        super().__init__(message)
        self.z_score = z_score


# ═══════════════════════════════════════════════════════════════════════════════
# I/O Errors
# ═══════════════════════════════════════════════════════════════════════════════


class HAPIOError(HAPError):
    """Error reading or writing model / memory data."""


class HAPSerializationError(HAPIOError):
    """Serialization or deserialization failure.

    Raised when:
        - A checkpoint file is missing, truncated, or has wrong schema version
        - The data format is incompatible with the current library version
    """


class HAPDatasetError(HAPIOError):
    """Error loading or parsing a dataset (MVSEC, DVS events, etc.)."""