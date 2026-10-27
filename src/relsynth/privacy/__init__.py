# SPDX-License-Identifier: Apache-2.0
"""Privacy risk measurement: attacks per person and per release context.

    import relsynth as rs
    rep = rs.privacy.assess("spec.yaml", "privacy.yaml")
    rs.privacy.write_report(rep, "privacy_report")      # JSON, Markdown, HTML

See docs/privacy.md.
"""
from .assess import assess, gate_verdict, relsynth_generator, sign_off
from .config import CONTEXTS, DEFAULTS, load as load_config
from .report import reference, to_html, to_json, to_markdown, write_report

__all__ = ["assess", "gate_verdict", "relsynth_generator", "sign_off", "CONTEXTS", "DEFAULTS", "load_config",
           "reference", "to_html", "to_json",
           "to_markdown", "write_report"]
