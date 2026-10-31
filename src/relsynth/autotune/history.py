# SPDX-License-Identifier: Apache-2.0
"""The run history of the self-tuning loop: one JSON line per run (a candidate
on one seed), aggregates only, the same rule as diagnostics bundles.

    <spec>.tune/history.jsonl   {"fingerprint", "key", "delta", "patches", "origin", "seed",
                                 "checks", "seconds", "error", "doctor", "time"}
    <spec>.tune/state.json      {"fingerprint", "incumbent": {key, delta, patches}}

Runs of another fingerprint (the spec or the objectives changed) stay in the
file but are not reused.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class History:
    def __init__(self, folder: Path, fingerprint: str):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "history.jsonl"
        self.state_path = self.dir / "state.json"
        self.fp = fingerprint
        self.records: list[dict] = []
        self._by: dict = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue                    # a line cut by an interrupted run
                if r.get("fingerprint") == fingerprint:
                    self._index(r)

    def _index(self, r):
        self.records.append(r)
        self._by[(r["key"], int(r["seed"]))] = r

    def get(self, key: str, seed: int) -> dict | None:
        return self._by.get((key, int(seed)))

    def keys(self) -> set:
        return {k for k, _ in self._by}

    def add(self, key, delta, patches, trial: dict, origin: str = "") -> dict:
        r = {"fingerprint": self.fp, "key": key, "delta": delta, "patches": patches or {}, "origin": origin,
             **trial, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        lead = ""
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as f:
                f.seek(-1, 2)
                lead = "" if f.read(1) == b"\n" else "\n"      # after a line cut by an interrupted run
        with self.path.open("a", encoding="utf-8") as f:
            f.write(lead + json.dumps(r, default=str) + "\n")
        self._index(r)
        return r

    def by_key(self) -> dict:
        out: dict = {}
        for r in self.records:
            out.setdefault(r["key"], []).append(r)
        return out

    def scores(self, obj) -> list:
        """[{key, delta, patches, seeds, score}] of every candidate (score: mean
        excess failures plus 10 per hard violation; lower is better)."""
        from .analysis import judge
        out = []
        for key, rs in self.by_key().items():
            rs = [r for r in rs if r["seed"] in obj.seeds_search]
            if not rs:
                continue
            J = judge(rs, obj)
            out.append({"key": key, "delta": rs[0]["delta"], "patches": rs[0]["patches"], "seeds": J.seeds,
                        "score": J.excess + 10.0 * J.hard_violations / max(J.n, 1)})
        return out

    def incumbent(self) -> dict | None:
        if not self.state_path.exists():
            return None
        try:
            st = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return st.get("incumbent") if st.get("fingerprint") == self.fp else None

    def set_incumbent(self, key, delta, patches):
        self.state_path.write_text(json.dumps({"fingerprint": self.fp, "incumbent": {
            "key": key, "delta": delta, "patches": patches or {}}}, default=str, indent=1), encoding="utf-8")

    def clear(self):
        """Forget this fingerprint's runs (other fingerprints' lines are kept)."""
        if self.path.exists():
            keep = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip() and _fp(ln) != self.fp]
            self.path.write_text("".join(x + "\n" for x in keep), encoding="utf-8")
        if self.state_path.exists():
            self.state_path.unlink()
        self.records, self._by = [], {}


def _fp(line: str):
    try:
        return json.loads(line).get("fingerprint")
    except json.JSONDecodeError:
        return None                         # a cut line: dropped with the cleared runs
