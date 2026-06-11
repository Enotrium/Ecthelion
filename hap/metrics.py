"""
HAP Observability & Metrics — Prometheus / OpenTelemetry Integration
======================================================================
Optional instrumentation for production monitoring of HDC pipelines.

Usage:
    from hap.metrics import HAPMetrics
    metrics = HAPMetrics(dim=8_000, enable_prometheus=True)

    # In your training/inference loop:
    with metrics.track_encode():
        hv = encoder.encode(sensor_data)

    with metrics.track_train():
        memory.train(percept, action)

    with metrics.track_infer():
        result = memory.infer(query, candidates)

    # Export Prometheus metrics on :8000
    metrics.start_prometheus_server(port=8000)

Design:
    - Optional: if prometheus-client is not installed, metrics become no-ops
    - Low overhead: only tracks counts and durations (no per-sample distribution)
    - Thread-safe: counters and histograms from prometheus_client are atomic
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

# ── Try importing prometheus_client (optional dependency) ─────
try:
    from prometheus_client import Counter, Histogram, start_http_server, Gauge

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus-client not installed; metrics will be no-ops")


class _NoOpCounter:
    def inc(self, amount: float = 1):
        pass


class _NoOpHistogram:
    def observe(self, amount: float):
        pass


class _NoOpGauge:
    def set(self, value: float):
        pass

    def inc(self, amount: float = 1):
        pass

    def dec(self, amount: float = 1):
        pass


class HAPMetrics:
    """Production metrics for HDC pipelines.

    Attributes:
        dim: HV dimensionality (used for energy estimation context)
        enable_prometheus: If True, register real Prometheus collectors
    """

    def __init__(
        self,
        dim: int = 10_000,
        enable_prometheus: bool = True,
        metric_prefix: str = "hap",
    ):
        self.dim = dim
        self._enabled = enable_prometheus and _PROMETHEUS_AVAILABLE
        self._prefix = metric_prefix

        if self._enabled:
            self._encode_counter = Counter(
                f"{metric_prefix}_encode_total",
                "Total number of encoding operations",
                ["encoder_type"],
            )
            self._encode_latency = Histogram(
                f"{metric_prefix}_encode_latency_seconds",
                "Encoding operation duration",
                ["encoder_type"],
                buckets=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
            )
            self._train_counter = Counter(
                f"{metric_prefix}_train_total",
                "Total number of training samples",
                ["memory_type"],
            )
            self._train_latency = Histogram(
                f"{metric_prefix}_train_latency_seconds",
                "Training sample duration",
                ["memory_type"],
                buckets=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
            )
            self._infer_counter = Counter(
                f"{metric_prefix}_infer_total",
                "Total number of inference calls",
                ["memory_type"],
            )
            self._infer_latency = Histogram(
                f"{metric_prefix}_infer_latency_seconds",
                "Inference call duration",
                ["memory_type"],
                buckets=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
            )
            self._energy_counter = Counter(
                f"{metric_prefix}_energy_pj_total",
                "Cumulative energy consumed in picojoules",
                ["operation"],
            )
            self._model_size = Gauge(
                f"{metric_prefix}_model_size_bytes",
                "Approximate model/memory size in bytes",
                ["memory_type"],
            )
            self._confidence_hist = Histogram(
                f"{metric_prefix}_infer_confidence",
                "Inference confidence score (z-score or Hamming similarity)",
                ["memory_type"],
                buckets=(0.45, 0.475, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
            )
        else:
            self._encode_counter = _NoOpCounter()
            self._encode_latency = _NoOpHistogram()
            self._train_counter = _NoOpCounter()
            self._train_latency = _NoOpHistogram()
            self._infer_counter = _NoOpCounter()
            self._infer_latency = _NoOpHistogram()
            self._energy_counter = _NoOpCounter()
            self._model_size = _NoOpGauge()
            self._confidence_hist = _NoOpHistogram()

    # ── Context managers for tracking ─────────────────────────

    @contextmanager
    def track_encode(self, encoder_type: str = "generic"):
        """Track an encoding operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._encode_counter.labels(encoder_type).inc()
            self._encode_latency.labels(encoder_type).observe(elapsed)

    @contextmanager
    def track_train(self, memory_type: str = "generic"):
        """Track a training step."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._train_counter.labels(memory_type).inc()
            self._train_latency.labels(memory_type).observe(elapsed)

    @contextmanager
    def track_infer(self, memory_type: str = "generic"):
        """Track an inference operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._infer_counter.labels(memory_type).inc()
            self._infer_latency.labels(memory_type).observe(elapsed)

    # ── Direct record methods ─────────────────────────────────

    def record_train(self, memory_type: str = "generic", n_samples: int = 1) -> None:
        """Record training progress without explicit timing."""
        self._train_counter.labels(memory_type).inc(n_samples)

    def record_infer(self, memory_type: str = "generic") -> None:
        """Record inference call count."""
        self._infer_counter.labels(memory_type).inc()

    def record_energy(self, operation: str, energy_pj: float) -> None:
        """Record energy consumption for an operation."""
        self._energy_counter.labels(operation).inc(energy_pj)

    def set_model_size(self, memory_type: str, size_bytes: int) -> None:
        """Set the current model/memory size gauge."""
        self._model_size.labels(memory_type).set(size_bytes)

    def record_confidence(self, memory_type: str, confidence: float) -> None:
        """Record inference confidence score."""
        self._confidence_hist.labels(memory_type).observe(confidence)

    # ── Server ────────────────────────────────────────────────

    def start_prometheus_server(self, port: int = 8000, addr: str = "0.0.0.0") -> None:
        """Start an HTTP server exposing Prometheus metrics.

        Args:
            port: HTTP port to listen on
            addr: Bind address

        Raises:
            RuntimeError: if prometheus-client is not installed
        """
        if not _PROMETHEUS_AVAILABLE:
            raise RuntimeError(
                "prometheus-client not installed. Run: pip install prometheus-client"
            )
        logger.info("Starting Prometheus metrics server on %s:%d", addr, port)
        start_http_server(port, addr)


# ── Module-level convenience ──────────────────────────────────

_default_metrics: Optional[HAPMetrics] = None


def get_metrics(dim: int = 10_000) -> HAPMetrics:
    """Get or create the default module-level metrics instance."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = HAPMetrics(dim=dim)
    return _default_metrics