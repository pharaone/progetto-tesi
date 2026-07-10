"""Prometheus instrumentation shared by all microservices.

- setup_metrics(app, service) instruments a FastAPI app with standard HTTP
  metrics (request count, latency histograms, payload sizes) and exposes
  them at GET /metrics for Prometheus to scrape.
- Custom metrics cover the LLM layer (call count/latency per agent), the
  evaluation output (cards per verdict) and the analysis pipeline.

Grafana dashboards are provisioned from monitoring/grafana/dashboards.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# ---------------------------------------------------------------------------
# Custom metrics (registered once per process)
# ---------------------------------------------------------------------------

LLM_CALLS = Counter(
    "iso42001_llm_calls_total",
    "Total LLM invocations",
    ["service", "status"],  # status: success | error
)

LLM_LATENCY = Histogram(
    "iso42001_llm_call_duration_seconds",
    "Duration of a single LLM invocation",
    ["service"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)

REQUIREMENTS_EVALUATED = Counter(
    "iso42001_requirements_evaluated_total",
    "Evaluation cards produced, by verdict",
    ["service", "verdict"],
)

ANALYSES = Counter(
    "iso42001_analyses_total",
    "Gap analysis pipeline runs",
    ["status"],  # success | error
)

ANALYSIS_DURATION = Histogram(
    "iso42001_analysis_duration_seconds",
    "End-to-end duration of the gap analysis pipeline",
    buckets=(60, 300, 600, 1200, 1800, 3600, 7200),
)


def setup_metrics(app, service: str) -> None:
    """Instrument a FastAPI app and expose /metrics."""
    Instrumentator(
        excluded_handlers=["/metrics", "/health"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    # Stash the service name for consumers that want it
    app.state.metrics_service = service


@contextmanager
def track_llm_call(service: str):
    """Record duration and outcome of one LLM invocation.

    Usage:
        with track_llm_call("AS-1"):
            response = llm.invoke(messages)
    """
    start = time.perf_counter()
    try:
        yield
    except Exception:
        LLM_CALLS.labels(service=service, status="error").inc()
        LLM_LATENCY.labels(service=service).observe(time.perf_counter() - start)
        raise
    else:
        LLM_CALLS.labels(service=service, status="success").inc()
        LLM_LATENCY.labels(service=service).observe(time.perf_counter() - start)
