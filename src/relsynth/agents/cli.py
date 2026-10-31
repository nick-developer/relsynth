# SPDX-License-Identifier: Apache-2.0
"""`relsynth agent ...`: agents that draft and explain (docs/agents.md).

    relsynth agent review draft.yaml --out answers.json --md answers.md
    relsynth agent review draft.yaml --backend openai --model qwen2.5:7b-instruct      # a local model (Ollama)
    relsynth agent apply draft.yaml answers.json --accept all-confirmed --by alice --out spec.yaml
    relsynth agent explain quality.json
    relsynth agent doctor bundle.json --out proposals.json
    relsynth agent draft contract spec.yaml --out contract.yaml
    relsynth agent draft policy spec.yaml --organisation "Acme" --out policy.draft.yaml

Agents propose; people decide.  `apply` is the person's decision and records
who made it.  With --registry and --key (a trusted key with role ``agent``)
each action is signed into the governance registry.
"""
from __future__ import annotations

import getpass
import json
from pathlib import Path

import yaml

from ..errors import EXIT_GATE, EXIT_OK


def _backend(a):
    from .backends import make_backend
    if a.backend == "rules":
        return make_backend("rules")
    cfg = {"kind": a.backend}
    for k in ("base_url", "model", "api_key_env"):
        if getattr(a, k, None):
            cfg[k] = getattr(a, k)
    if a.allow_remote:
        cfg["allow_remote"] = True
    return make_backend(cfg)


def _record(a, task, inputs, output, proposals, flags, backend, scope):
    if not getattr(a, "registry", None):
        return None
    from ..governance.registry import LocalRegistry
    from ..governance.sign import Signer
    from .record import action_statement, record_action
    st = action_statement(task=task, agent=f"agent:{a.agent_name}", on_behalf_of=f"user:{a.by or getpass.getuser()}",
                          backend={"name": getattr(backend, "name", "?"), "model": getattr(backend, "model", "?")},
                          scope=scope, inputs=inputs, output=output, proposals=proposals, flags=flags)
    d, seq = record_action(LocalRegistry(a.registry), st, Signer.load(a.key))
    return {"statement": d, "seq": seq}


def cmd_review(a, out):
    from ..api import Spec
    from .review import review
    from .scope import Scope
    spec = Spec.load(a.spec)
    raw = dict(spec.raw)
    src = a.source or raw.get("source")
    if isinstance(src, dict):
        src = dict(src)
        if src.get("path") and not Path(src["path"]).is_absolute():
            src["path"] = str((Path(a.spec).parent / src["path"]).resolve())
    b = _backend(a)
    res = review(raw, source=src, backend=b, scope=Scope(a.scope, a.k), only=a.only.split(",") if a.only else None)
    doc = res.to_dict()
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    if a.md:
        Path(a.md).write_text(res.to_markdown(), encoding="utf-8")
    rec = _record(a, "review", {"spec": raw}, doc["answers"], len(res.answers), len(res.flags), b, res.scope)
    s = res.summary()
    out.put(summary=s, record=rec)
    out.line(f"{s['items']} review items: " + ", ".join(f"{v} {k}" for k, v in sorted(s["verdicts"].items()))
             + f"; {s['attention']} need attention; {s['changes']} proposed changes; nothing applied")
    if not a.out and not a.md:
        out.line(res.to_markdown())
    return EXIT_OK


def cmd_apply(a, out):
    from .review import apply_decisions
    raw = yaml.safe_load(Path(a.spec).read_text(encoding="utf-8"))
    doc = json.loads(Path(a.answers).read_text(encoding="utf-8"))
    answers = doc["answers"] if isinstance(doc, dict) else doc
    dec: dict = {}
    for spec_, d in ((a.accept, "accept"), (a.reject, "reject")):
        for x in (spec_ or "").split(","):
            x = x.strip()
            if not x:
                continue
            if x == "all-confirmed":
                dec.update({ans["item"]: d for ans in answers if ans["verdict"] == "confirm"})
            elif x == "all":
                dec.update({ans["item"]: d for ans in answers})
            else:
                dec[x] = d
    by = f"user:{a.by or getpass.getuser()}"
    new, log = apply_decisions(raw, answers, dec, decided_by=by)
    Path(a.out).write_text(yaml.safe_dump(new, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if a.log:
        Path(a.log).write_text(json.dumps(log, indent=1), encoding="utf-8")
    changed = sum(1 for e in log if (e.get("change") or {}).get("applied"))
    out.put(decided=len(log), changed=changed, by=by)
    out.line(f"{len(log)} answers decided by {by}; {changed} changes made; spec written to {a.out}")
    return EXIT_OK


def cmd_explain(a, out):
    from .explain import explain
    rep = json.loads(Path(a.report).read_text(encoding="utf-8"))
    b = _backend(a)
    e = explain(rep, a.kind, backend=b)
    out.put(explanation=e)
    out.line(e["text"])
    for p in e["problems"]:
        out.line(f"(model text not used: {p})")
    return EXIT_OK


def cmd_doctor(a, out):
    from .drafts import doctor_proposals
    props = doctor_proposals(a.bundle)
    if a.out:
        Path(a.out).write_text(json.dumps(props, indent=1, default=str), encoding="utf-8")
    out.put(proposals=props)
    for p in props:
        out.line(f"{p['id']} [{p['severity']}] {p['where']}: {p['message'][:200]}")
        for x in p["patches"]:
            out.line(f"    proposes {x['path']} = {x.get('value')!r}")
        for x in p["advice_for_privacy_officer"]:
            out.line(f"    for the privacy officer: {x['path']} = {x.get('value')!r} (not proposed)")
    return EXIT_OK


def cmd_draft(a, out):
    from ..api import Spec
    from .drafts import draft_contract, draft_policy
    raw = Spec.load(a.spec).raw
    if a.what == "contract":
        txt = yaml.safe_dump(draft_contract(raw, owner=a.owner), sort_keys=False, allow_unicode=True)
    else:
        txt = draft_policy(raw, organisation=a.organisation, owner=a.owner or "Data Protection Officer",
                           source_id=a.source_id)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        out.line(f"{a.what} draft written to {a.out}; a person reviews and completes it")
    else:
        out.line(txt)
    out.put(draft=a.what, out=a.out)
    return EXIT_OK


def cmd_eval(a, out):
    from .evaluation import evaluate, truth_from_spec
    from .review import review
    from .scope import Scope
    raw = yaml.safe_load(Path(a.draft).read_text(encoding="utf-8"))
    truth = truth_from_spec(yaml.safe_load(Path(a.truth).read_text(encoding="utf-8")))
    res = review(raw, source=a.source, backend=_backend(a), scope=Scope(a.scope))
    ev = evaluate(res, truth)
    if a.out:
        Path(a.out).write_text(json.dumps(ev, indent=1, default=str), encoding="utf-8")
    out.put(evaluation=ev)
    out.line(f"{ev['correct']} correct, {ev['wrong']} wrong, {ev['abstained']} left to people of "
             f"{ev['judged_items']} items with ground truth; accuracy {ev['accuracy']:.3f}")
    return EXIT_GATE if a.min_accuracy is not None and not ev["accuracy"] >= a.min_accuracy else EXIT_OK


def add_parser(sub, common):
    g = sub.add_parser("agent", help="agents that draft and explain; people decide (docs/agents.md)")
    gs = g.add_subparsers(dest="agent_cmd", required=True)

    def backend_args(s):
        s.add_argument("--backend", default="rules", choices=["rules", "openai"],
                       help="rules (no model, default) or openai (an OpenAI-compatible server such as Ollama)")
        s.add_argument("--base-url", help="model server, default http://localhost:11434/v1")
        s.add_argument("--model", help="model name on that server")
        s.add_argument("--api-key-env", help="environment variable holding the server's API key")
        s.add_argument("--allow-remote", action="store_true",
                       help="allow a model server that is not on this machine (it gets aggregates at most)")

    def record_args(s):
        s.add_argument("--registry", help="sign the action into this governance registry")
        s.add_argument("--key", help="the agent's private key (a trusted key with role agent)")
        s.add_argument("--agent-name", default="relsynth-review")
        s.add_argument("--by", help="the person the agent acts for (default: the login name)")

    s = gs.add_parser("review", help="answer a draft's review: items with evidence (proposals only)")
    s.add_argument("spec")
    s.add_argument("--source", help="source location, if not the draft's source section")
    s.add_argument("--scope", default="aggregates", choices=["schema", "aggregates", "values"])
    s.add_argument("--k", type=int, default=10, help="values scope: a value is shown only if it occurs k times")
    s.add_argument("--only", help="comma-separated item kinds")
    s.add_argument("--out", help="answers as JSON")
    s.add_argument("--md", help="answers as a Markdown table")
    backend_args(s)
    record_args(s)
    s.set_defaults(fn=cmd_review)

    s = gs.add_parser("apply", help="a person's decisions on review answers: write the changed spec")
    s.add_argument("spec")
    s.add_argument("answers")
    s.add_argument("--accept", help="item ids, 'all-confirmed' or 'all'")
    s.add_argument("--reject", help="item ids or 'all'")
    s.add_argument("--by", help="who decides (default: the login name)")
    s.add_argument("--out", required=True)
    s.add_argument("--log", help="decision log as JSON")
    s.set_defaults(fn=cmd_apply)

    s = gs.add_parser("explain", help="a quality, privacy or scorecard report in plain language")
    s.add_argument("report")
    s.add_argument("--kind", choices=["quality", "privacy", "scorecard"])
    backend_args(s)
    s.set_defaults(fn=cmd_explain)

    s = gs.add_parser("doctor", help="doctor findings as proposals (privacy fixes go to the privacy officer)")
    s.add_argument("bundle")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_doctor)

    s = gs.add_parser("draft", help="draft a data contract or an organisation policy")
    s.add_argument("what", choices=["contract", "policy"])
    s.add_argument("spec")
    s.add_argument("--organisation", default="Your organisation")
    s.add_argument("--owner")
    s.add_argument("--source-id")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_draft)

    s = gs.add_parser("eval", help="accuracy of review answers against ground truth (a catalog draft or curated spec)")
    s.add_argument("draft", help="a draft made with introspect --no-declared (benchmark mode)")
    s.add_argument("truth", help="the ground truth: a draft with the declared catalog, or a curated spec")
    s.add_argument("--source")
    s.add_argument("--scope", default="aggregates", choices=["schema", "aggregates", "values"])
    s.add_argument("--min-accuracy", type=float, help="exit 1 below this accuracy")
    s.add_argument("--out")
    backend_args(s)
    s.set_defaults(fn=cmd_eval)

    for sp in gs.choices.values():
        common(sp)
    return g


__all__ = ["add_parser"]
