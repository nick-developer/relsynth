"""Pushdown: fit large fact tables from statistics computed where the data lives.

The relational model (latent segments, link and block rates, count laws,
participation, inclusion) reads a fact table only through exact aggregates of
its key columns (`stats.TableStats`), computed with GROUP BY and window
functions inside the database (or column-pruned reads of files).  The
row-level parts of the fact's model (attribute marginals and copulas, time
profiles, extra-dimension laws, text, rules) are fitted from a seeded uniform
sample of `synthesis.pushdown.sample_rows` rows.  See docs/theory.md, section
"Pushdown", for the propositions and their checks.
"""
from .loader import FitData, load_for_fit  # noqa: F401
from .planner import PushdownPlan, ineligible_reason, plan  # noqa: F401
from .stats import FrameStats, TableStats, normal_scores  # noqa: F401
