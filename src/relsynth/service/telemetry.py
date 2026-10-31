# SPDX-License-Identifier: Apache-2.0
"""Logs, metrics and traces.

* Structured logs: one JSON object per line on stdout (``log_json``), with the
  request id, so a log shipper (Fluent Bit, Vector) needs no parsing rules.
* Metrics: Prometheus text format at ``/metrics`` (counters and a request-latency
  histogram).  Labels never carry tenant names or user ids.
* Traces: OpenTelemetry, when ``otel`` is on and ``relsynth[service-otel]`` is
  installed; configure the exporter with the standard ``OTEL_*`` variables
  (``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_SERVICE_NAME`` ...).
"""
from __future__ import annotations

import contextlib
import json
import logging
import sys
import threading
import time
from collections import defaultdict

_tracer = None


class JsonFormatter(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        d = {"ts": round(r.created, 3), "level": r.levelname.lower(), "logger": r.name, "msg": r.getMessage()}
        for k in ("request_id", "tenant", "actor", "action", "target", "outcome", "status", "path", "ms"):
            v = getattr(r, k, None)
            if v is not None:
                d[k] = v
        if r.exc_info:
            d["exc"] = self.formatException(r.exc_info)
        return json.dumps(d, default=str)


def setup_logging(json_logs: bool = True, level: int = logging.INFO) -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger("relsynth.service")
    root.handlers[:] = [h]
    root.setLevel(level)
    root.propagate = False


class Metrics:
    BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

    def __init__(self):
        self._lock = threading.Lock()
        self.counters: dict[tuple, float] = defaultdict(float)
        self.hist: dict[tuple, list[float]] = {}
        self.gauges: dict[tuple, float] = {}

    @staticmethod
    def _key(name, labels):
        return (name, tuple(sorted(labels.items())))

    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        with self._lock:
            self.counters[self._key(name, labels)] += value

    def set(self, name: str, value: float, **labels) -> None:
        with self._lock:
            self.gauges[self._key(name, labels)] = value

    def observe(self, name: str, value: float, **labels) -> None:
        with self._lock:
            k = self._key(name, labels)
            h = self.hist.setdefault(k, [0.0] * (len(self.BUCKETS) + 2))
            for i, b in enumerate(self.BUCKETS):
                if value <= b:
                    h[i] += 1
            h[-2] += 1
            h[-1] += value

    def render(self) -> str:
        def lab(ls, extra=()):
            items = list(ls) + list(extra)
            return "{" + ",".join(f'{k}="{v}"' for k, v in items) + "}" if items else ""
        out = []
        with self._lock:
            for (n, ls), v in sorted(self.counters.items()):
                out.append(f"{n}{lab(ls)} {v}")
            for (n, ls), v in sorted(self.gauges.items()):
                out.append(f"{n}{lab(ls)} {v}")
            for (n, ls), h in sorted(self.hist.items()):
                for i, b in enumerate(self.BUCKETS):
                    out.append(f"{n}_bucket{lab(ls, [('le', b)])} {h[i]}")
                out.append(f"{n}_bucket{lab(ls, [('le', '+Inf')])} {h[-2]}")
                out.append(f"{n}_count{lab(ls)} {h[-2]}")
                out.append(f"{n}_sum{lab(ls)} {h[-1]}")
        return "\n".join(out) + "\n"


METRICS = Metrics()


def setup_tracing(enabled: bool, service_name: str = "relsynth-service") -> bool:
    """Install an OTLP tracer provider (when the SDK is installed)."""
    global _tracer
    if not enabled:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logging.getLogger("relsynth.service").warning("otel=true but opentelemetry-sdk is not installed")
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("relsynth.service")
    return True


@contextlib.contextmanager
def span(name: str, attributes: dict | None = None):
    t0 = time.perf_counter()
    if _tracer is None:
        yield None
    else:
        with _tracer.start_as_current_span(name, attributes=attributes or {}) as s:
            yield s
    METRICS.observe("relsynth_span_seconds", time.perf_counter() - t0, span=name)
