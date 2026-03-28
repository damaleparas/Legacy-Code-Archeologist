# ────────────────────────────────────────────────────────────────────────────
#  Dockerfile — LegacyCodeArcheologist OpenEnv
#  Build:  docker build -t lca-env .
#  Run:    docker run -p 5000:5000 lca-env
#          docker run -p 5000:5000 lca-env --task task_2_auth_header
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bookworm AS base

# System deps (patch utility for unified-diff support)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        patch \
        curl \
        procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies layer (cached unless requirements change) ───────────────────
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ───────────────────────────────────────────────────────
COPY models.py          ./
COPY env.py             ./
COPY task.py            ./
COPY grader.py          ./
COPY server.py          ./
COPY legacy_templates/  ./legacy_templates/
COPY tests/             ./tests/
COPY scripts/           ./scripts/

# ── Generate templates (idempotent) ─────────────────────────────────────────
RUN python scripts/generate_templates.py

# ── Smoke test ───────────────────────────────────────────────────────────────
RUN pytest tests/ -x -q --timeout=30 \
 && echo "✓ All tests passed"

# ── Runtime ──────────────────────────────────────────────────────────────────
EXPOSE 5000

# Health check: the OpenEnv server's /health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

ENTRYPOINT ["python", "server.py", "--host", "0.0.0.0", "--port", "5000"]
CMD []


# ── Dev / test-only target ────────────────────────────────────────────────────
FROM base AS test
CMD ["pytest", "tests/", "-v", "--timeout=60"]
