# SPDX-License-Identifier: Apache-2.0
"""Checks C1600-C1604: the agent propositions of docs/proofs (section T16,
Props. 1600-1604), each tried against random inputs that look for a break.
About a minute.

    pytest -m science tests/science/test_agent_checks.py
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import string

import pytest

import relsynth as rs
from relsynth import agents
from relsynth.agents.evidence import Evidence
from relsynth.agents.explain import explain
from relsynth.agents.guard import KIND_VERDICTS, OPS, check_answer, ungrounded_numbers
from relsynth.agents.items import parse

pytestmark = [pytest.mark.science]

RND = random.Random(1600)


def _rand_json(r: random.Random, depth=0):
    kinds = ["str", "num", "none", "bool", "list", "dict"] if depth < 3 else ["str", "num", "none", "bool"]
    k = r.choice(kinds)
    if k == "str":
        pool = ["confirm", "reject", "answer", "unsure", "explain", "approve", "E1", "E2", "E9", "parent",
                "https://x.example/a", "![i](http://e/x)", "<b>x</b>", "a@b.example", "0.37", "1000", "attacker",
                "employee", "region", "customer"]
        return r.choice(pool) if r.random() < 0.7 else "".join(r.choice(string.printable) for _ in range(r.randint(0, 40)))
    if k == "num":
        return r.choice([0, 1, -1, 0.5, 1e9, float("nan"), float("inf"), r.random() * 100])
    if k == "none":
        return None
    if k == "bool":
        return r.random() < 0.5
    if k == "list":
        return [_rand_json(r, depth + 1) for _ in range(r.randint(0, 4))]
    keys = ["verdict", "parent", "confidence", "evidence", "explanation", "change", "action", "patch"]
    return {r.choice(keys): _rand_json(r, depth + 1) for _ in range(r.randint(0, 7))}


def _db(tmp_path, seed):
    """A random small database with declared keys and links, a person table whose
    values are unique (so no scope may show them), and some planted violations."""
    r = random.Random(seed)
    p = tmp_path / f"r{seed}.db"
    db = sqlite3.connect(p)
    np_, nc = r.randint(3, 12), r.randint(50, 300)
    db.executescript("CREATE TABLE parent (parent_id INTEGER PRIMARY KEY, label TEXT);"
                     "CREATE TABLE person (person_id INTEGER PRIMARY KEY, secret TEXT, parent_id INTEGER "
                     "REFERENCES parent(parent_id), score REAL, flag TEXT);")
    for i in range(1, np_ + 1):
        db.execute("INSERT INTO parent VALUES (?,?)", (i, f"L{i}"))
    for i in range(1, nc + 1):
        db.execute("INSERT INTO person VALUES (?,?,?,?,?)",
                   (i, f"SECRET{seed}x{i}x{r.randint(0, 10**9)}", r.randint(1, np_), r.random(),
                    r.choice(["y"] * 20 + ["rare"])))
    db.commit()
    db.close()
    return p


def test_C1600_no_rare_value_reaches_an_agent(tmp_path):
    """Prop. 1600: across random databases and every scope, no value that occurs
    fewer than k times appears in evidence, answers, flags or model prompts."""
    for seed in range(6):
        p = _db(tmp_path, seed)
        draft = rs.introspect(f"sqlite:{p}", declared=False).raw
        draft["review"] = list(draft["review"]) + ["person.flag: constant (a single value)",
                                                   "person.secret: constant (a single value)"]
        con = sqlite3.connect(p)
        rare = {str(v) for (v,) in con.execute("SELECT secret FROM person")}
        rare |= {v for (v, n) in con.execute("SELECT flag, count(*) FROM person GROUP BY flag") if n < 10}
        con.close()
        for scope in ("schema", "aggregates", "values"):
            seen = []
            b = agents.ScriptedBackend(lambda s, u, sc, seen=seen: seen.append(u) or {})
            res = agents.review(draft, source=f"sqlite:{p}", backend=b, scope=scope)
            blob = json.dumps(res.to_dict(), default=str) + "".join(seen)
            assert not any(v in blob for v in rare), (seed, scope)


def test_C1601_guard_contains_any_model_output():
    """Prop. 1601: 20,000 random outputs over every item kind satisfy the guard's invariants."""
    spec = {"tables": {"employee": {}, "region": {}, "customer": {}}, "rules": []}
    lines = ["customer: primary key customer_id inferred (x)",
             "customer.region_id -> region: foreign key inferred (x)",
             "ASK customer.rep: looks like a foreign key but the data fits employee (a), region (b) equally well",
             "customer.email: pii (name matches email)", "NOTE: n", "customer.status: constant (a single value)"]
    items = parse(lines, spec)
    for n in range(20000):
        it = items[n % len(items)]
        evs = {f"E{i}": Evidence(f"E{i}", "inclusion", {"parent": RND.choice(["employee", "region"])},
                                 {"available": True, "share_rows_in_parent": RND.choice([1.0, 0.5]),
                                  "unique": True, "unique_not_null": True, "distinct": 1, "holds": True})
               for i in range(1, RND.randint(1, 3) + 1)}
        a, _ = check_answer(_rand_json(RND), it, evs, spec)
        assert a["verdict"] in KIND_VERDICTS.get(it.kind, ("explain",))
        assert a["parent"] in (None, *it.candidates, it.table) and (a["parent"] is None or a["parent"] in spec["tables"])
        assert set(a["evidence"]) <= set(evs)
        assert a["change"] is None or a["change"]["op"] in OPS
        assert not re.search(r"https?://|<[a-z/]|!\[|@\w+\.example", a["explanation"])
        if it.kind not in KIND_VERDICTS:
            assert a["verdict"] == "explain" and a["change"] is None


def test_C1602_no_confirmation_against_the_evidence(tmp_path):
    """Prop. 1602: planted duplicates, orphans and rule violations are never confirmed,
    even by a backend that confirms everything."""
    yes = agents.ScriptedBackend(lambda s, u, sc: {"verdict": "confirm", "parent": None, "confidence": 1,
                                                   "evidence": ["E1"], "explanation": ""})
    for seed in range(5):
        p = _db(tmp_path, 100 + seed)
        con = sqlite3.connect(p)
        con.execute("UPDATE person SET parent_id = 999 WHERE person_id = 1")     # an orphan
        con.commit()
        con.close()
        draft = rs.introspect(f"sqlite:{p}", declared=False).raw
        draft["review"] = ["person: primary key parent_id inferred (forced)",
                           "person.parent_id -> parent: foreign key inferred (forced)",
                           "person.score: constant (a single value)"]
        draft.setdefault("relationships", []).append({"child": "person", "child_key": "parent_id", "parent": "parent"})
        res = agents.review(draft, source=f"sqlite:{p}", backend=yes)
        assert all(a["verdict"] != "confirm" for a in res.answers), [a["verdict"] for a in res.answers]


def test_C1603_numbers_in_shown_text_are_grounded():
    """Prop. 1603: model text with any number outside the facts is never shown."""
    rep = {"vector": {"fidelity": {"checks": 12, "passed": 11, "failed": 1, "allowed_failures": 1, "pass": True},
                      "verdict": {"pass": True}},
           "checks": [{"id": "w1:t:c", "area": "fidelity", "metric": "w1", "value": 0.3125, "limit": 0.25,
                       "pass": False}]}
    for _ in range(3000):
        n = RND.choice([11, 12, 1, 0.3125, 0.31, 31.25, 0.25, 25, RND.randint(13, 10**6), round(RND.random(), 3)])
        txt = f"Fidelity passed {n} checks."
        b = agents.ScriptedBackend(lambda s, u, sc, t=txt: {"summary": t})
        e = explain(rep, backend=b)
        if e["source"] == "model":
            assert not ungrounded_numbers(e["text"], e["facts"])


def test_C1604_no_authority_without_a_person(tmp_path):
    """Prop. 1604: whatever a compromised model says, the spec is unchanged until a
    person decides, and only the accepted typed changes are made."""
    p = _db(tmp_path, 7)
    draft = rs.introspect(f"sqlite:{p}", declared=False).raw
    for _ in range(50):
        b = agents.ScriptedBackend(lambda s, u, sc: _rand_json(RND))
        res = agents.review(draft, source=f"sqlite:{p}", backend=b)
        assert res.spec == draft
        for who in ("agent:x", "service:y", ""):
            with pytest.raises(PermissionError):
                agents.apply_decisions(draft, res.answers, {a["item"]: "accept" for a in res.answers}, decided_by=who)
        new, log = agents.apply_decisions(draft, res.answers, {a["item"]: "accept" for a in res.answers},
                                          decided_by="user:p")
        for k in set(draft) | set(new):
            if k not in ("relationships", "rules", "review"):
                assert new.get(k) == draft.get(k), k
