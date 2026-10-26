"""Privacy risk measurement: attacks per person and per release context.

    import relsynth as rs
    rep = rs.privacy.assess("spec.yaml", "privacy.yaml")
    rs.privacy.write_report(rep, "privacy_report")      # JSON, Markdown, HTML

See docs/privacy.md.
"""
from .assess import assess, relsynth_generator
from .config import CONTEXTS, DEFAULTS, load as load_config
from .report import to_html, to_json, to_markdown, write_report

__all__ = ["assess", "relsynth_generator", "CONTEXTS", "DEFAULTS", "load_config", "to_html", "to_json", "to_markdown",
           "write_report"]
