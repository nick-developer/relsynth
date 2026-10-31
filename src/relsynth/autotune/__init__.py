# SPDX-License-Identifier: Apache-2.0
"""Self-tuning of the engine settings toward the user's objectives
(docs/autotune.md; research and design: docs/research/self-tuning.md).

    import relsynth.autotune as at
    res = at.autotune("spec.yaml", "objectives.yaml")
    res.met, res.confirmed, res.delta, res.report_markdown
"""
from __future__ import annotations

from .loop import AutotuneResult, autotune
from .objectives import Objectives, ObjectivesError, load as load_objectives, parse as parse_objectives

__all__ = ["autotune", "AutotuneResult", "Objectives", "ObjectivesError", "load_objectives", "parse_objectives"]
