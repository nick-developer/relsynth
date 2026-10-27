# SPDX-License-Identifier: Apache-2.0
"""Evaluation framework 2.0: calibrated fidelity, utility and integrity checks.

    from relsynth.quality import assess, METRICS
    report = assess(spec, real_tables, synthetic_tables)

See docs/metrics.md for every metric's definition and docs/quality-report.md
for the report schema and the pass rules.
"""
from .registry import AREAS, METRICS
from .report import DEFAULTS, SCHEMA_VERSION, assess, quality_vector

__all__ = ["AREAS", "DEFAULTS", "METRICS", "SCHEMA_VERSION", "assess", "quality_vector"]
