# ──────────────────────────────────────────────────────────────
# Hyperdimensional Active Perception (HAP) — Production Image
# ──────────────────────────────────────────────────────────────
# Multi-stage build: builder compiles extensions, runtime is slim.
# ──────────────────────────────────────────────────────────────

# ── Builder Stage ─────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# System deps for optional C extensions / OpenBLAS
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY hap/ ./hap/
COPY tests/ ./tests/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .[all]

# ── Runtime Stage ─────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="hyperdimensional_active_perception"
LABEL org.opencontainers.image.description="Production-grade HDC/VSA drone autonomy stack"
LABEL org.opencontainers.image.source="https://github.com/Enotrium/Ecthelion"

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Application user (non-root)
RUN useradd --create-home --shell /bin/bash hap && \
    mkdir -p /workspace/models /workspace/logs /workspace/data && \
    chown -R hap:hap /workspace

WORKDIR /workspace
USER hap

# Copy source for direct execution & demos
COPY --chown=hap:hap hap/ ./hap/
COPY --chown=hap:hap tests/ ./tests/
COPY --chown=hap:hap demo_ego_motion.py demo_online_learning.py ./

# Default command: run test suite to validate the build
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "from hap import gen_hvs, hv_xor, hv_bundle; print('OK')" || exit 1

CMD ["python3", "-m", "pytest", "tests/", "-q", "--tb=short"]