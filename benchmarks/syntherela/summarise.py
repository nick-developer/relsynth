# SPDX-License-Identifier: Apache-2.0
"""Markdown comparison table from run_syntherela.py results and the published
numbers in published.json:

    python benchmarks/syntherela/summarise.py syntherela_results.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = [("c2st_agg_max", "C2ST-Agg accuracy (lower is better)", False),
           ("cardinality_mean", "Cardinality similarity", True),
           ("hop1", "1-hop trends", True),
           ("hop2", "2-hop trends", True)]


def _ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return 100 * st.mean(xs), (100 * st.stdev(xs) if len(xs) > 1 else 0.0)


def table(results, published):
    lines = []
    for db, pub in published["databases"].items():
        rows = [r for r in results if r["database"] == db]
        if not rows:
            continue
        lines += [f"### {db} ({len(rows)} seeds, {rows[0]['rows_real']:,} rows)", "",
                  "| Metric | relsynth | " + " | ".join(published["methods"]) + " | relsynth rank |",
                  "|---|---|" + "---|" * len(published["methods"]) + "---|"]
        for key, label, higher in METRICS:
            m, s = _ms([r[key] for r in rows])
            if m is None or key not in pub:
                continue
            vals = pub[key]
            known = [v for v in vals if isinstance(v, (int, float))]
            better = sum(1 for v in known if (v > m if higher else v < m))
            cells = [("–" if v is None else (str(v) if isinstance(v, str) else f"{v:.2f}")) for v in vals]
            lines.append(f"| {label} | **{m:.2f} ± {s:.2f}** | " + " | ".join(cells)
                         + f" | {better + 1} of {len(known) + 1} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    res = json.loads(Path(sys.argv[1]).read_text())
    pub = json.loads((HERE / "published.json").read_text())
    print(table(res, pub))
