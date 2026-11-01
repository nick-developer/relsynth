# SPDX-License-Identifier: Apache-2.0
"""Check the claim register: every claim in the documentation traces to evidence.

    python tools/claims_check.py                 # exit 1 and list the problems
    python tools/claims_check.py --suggest FILE  # print register stubs for FILE's unregistered claims
    python tools/claims_check.py --json out.json # also write the resolved register (for the assurance case)

The register is ``docs/assurance/claims.yaml``.  Each entry binds a sentence of
a document (``quote``, which must still appear in ``doc``) to the evidence that
supports it.  Evidence is one of:

``prop``    a proposition label in ``docs/proofs/*.tex`` (``prop:dgauss``);
``check``   a numbered check (``C902``): a ``@check("C902", ...)`` in
            ``legacy/checks/run_checks.py`` or a ``def test_c902_...`` in ``tests/``;
``test``    a pytest node, ``tests/unit/test_dp.py::test_name`` (or ``::Class::test``);
``report``  a file in the repository that records a measurement;
``code``    ``src/relsynth/x.py::symbol``, for claims that are enforced by code
            (a refusal, a mask) and also have a ``test``;
``source``  an external text (a regulation, a standard): only for claims that
            say what a source says, never for a claim about relsynth.

The checker fails when:

1. a quote is no longer in its document (the claim changed: update the register);
2. a piece of evidence does not resolve (a renamed test, a removed proposition);
3. a claim about relsynth has no evidence other than ``source`` or ``code``;
4. a sentence in a document under ``scope`` reads as a claim (it matches the
   trigger words) and is neither covered by a register quote nor listed under
   ``not_claims`` with a reason;
5. a ``Prop. N`` reference in any Markdown document names a proposition number
   that the proofs do not define.

Proposition numbers are computed the way LaTeX numbers them: sequentially,
following every ``\\setcounter`` (including the save-and-restore pattern of
the numbered sections), over the files ``relsynth-proofs.tex`` inputs.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTER = Path("docs/assurance/claims.yaml")
PROOFS = Path("docs/proofs")
EVIDENCE_KINDS = ("prop", "check", "test", "report", "code", "source")
STRONG_KINDS = ("prop", "check", "test", "report")
STATUSES = ("supported", "partial", "limitation", "scope")
TRIGGER = re.compile(
    r"\b(guarantee[sd]?|prove[sn]?|proofs?|exact(?:ly)?|never|always|every|identical|byte[- ]for[- ]byte|"
    r"cannot|refuse[sd]?|tamper\w*|signed|bound(?:ed|s)?|differential privacy|calibrated|masked|enforce[sd]?|"
    r"no row|value for value|resolves?|holds?)\b",
    re.I,
)
PROP_REF = re.compile(r"\bProp(?:osition)?s?\.?[ ~]+(\d+(?:\s*(?:[–-]|to|and|,)\s*\d+)*)")


# ---------------------------------------------------------------- text helpers

def norm(text: str) -> str:
    """Whitespace collapsed, Markdown emphasis and links removed, for quote matching."""
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = text.replace("*", "").replace("`", "")
    return " ".join(text.split())


def sentences(text: str) -> list[str]:
    """Prose sentences and table rows of a Markdown document (code blocks, headings and comments dropped)."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        lines = [ln for ln in para.split("\n") if ln.strip() and not ln.lstrip().startswith("#")]
        if not lines:
            continue
        if all(ln.lstrip().startswith("|") for ln in lines):
            out += [norm(ln) for ln in lines if not re.fullmatch(r"\s*\|[\s|:-]*\|?\s*", ln)]
            continue
        items: list[str] = []
        for ln in lines:                      # list items are separate units
            if re.match(r"\s*(?:[-*]|\d+\.)\s", ln) or not items:
                items.append(ln)
            else:
                items[-1] += " " + ln
        for item in items:
            out += [s for s in re.split(r"(?<=[.!?:])\s+(?=[A-Z(*`])", norm(item)) if s]
    return out


# ---------------------------------------------------------------- evidence indices

def _strip_comments(line: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", line)


def _expand(path: Path) -> str:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comments(line)
        m = re.search(r"\\input\{([^}]+)\}", line)
        if m:
            name = m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex"
            if (path.parent / name).exists():
                out.append(_expand(path.parent / name))
            continue
        out.append(line)
    return "\n".join(out)


def proposition_numbers(root: Path = ROOT, aliases: dict[str, int] | None = None) -> dict[str, int]:
    """Label -> printed number of every proposition in the proofs document.

    A section may also name a fixed number in the title, ``[Worker independence;
    Prop.~950]``, while LaTeX prints the running number; such numbers are
    put in ``aliases`` (label -> number) when it is given.
    """
    text = _expand(root / PROOFS / "relsynth-proofs.tex")
    token = re.compile(
        r"\\begin\{proposition\}(\[[^\]]*\])?|\\setcounter\{(\w+)\}\{(?:\\value\{(\w+)\}|(-?\d+))\}|\\label\{(prop:[^}]+)\}"
    )
    counter, saved, labels, pending = 0, {}, {}, False
    for m in token.finditer(text):
        if m.group(0).startswith("\\begin"):
            counter += 1
            pending = True
            title = re.search(r"Prop\.~?\s*(\d{3,5})\]", m.group(1) or "")
        elif m.group(2):
            value = counter if m.group(3) == "proposition" else saved.get(m.group(3), 0) if m.group(3) else int(m.group(4))
            if m.group(2) == "proposition":
                counter = value
            else:
                saved[m.group(2)] = value
        elif pending:
            labels[m.group(5)] = counter
            if title and aliases is not None:
                aliases[m.group(5)] = int(title.group(1))
            pending = False
    return labels


def check_ids(root: Path = ROOT) -> set[str]:
    ids = set(re.findall(r'@check\("(C\d+)"', (root / "legacy/checks/run_checks.py").read_text(encoding="utf-8")))
    for path in (root / "tests").rglob("test_*.py"):
        ids |= {"C" + n for n in re.findall(r"^\s*def test_[cC](\d+)_", path.read_text(encoding="utf-8"), re.M)}
    return ids


def _symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(prefix + child.name)
                walk(child, prefix + child.name + "::")
            elif isinstance(child, (ast.Assign, ast.AnnAssign)) and not prefix:
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                names.update(t.id for t in targets if isinstance(t, ast.Name))

    walk(tree, "")
    return names


# ---------------------------------------------------------------- the check

@dataclass
class Result:
    problems: list[str] = field(default_factory=list)
    unclaimed: dict[str, list[str]] = field(default_factory=dict)
    claims: list[dict] = field(default_factory=list)
    props: dict[str, int] = field(default_factory=dict)


def _resolve(kind: str, value: str, root: Path, props: dict[str, int], checks: set[str], cache: dict) -> str | None:
    """None when the evidence resolves, else the reason it does not."""
    if kind == "prop":
        return None if value in props else f"no proposition labelled {value}"
    if kind == "check":
        return None if value in checks else f"no check {value}"
    if kind in ("test", "code"):
        file, _, symbol = value.partition("::")
        path = root / file
        if not path.is_file():
            return f"no file {file}"
        if not symbol:
            return None if kind == "code" else f"{value}: name the test (file::test)"
        if path not in cache:
            cache[path] = _symbols(path)
        return None if symbol in cache[path] else f"{file} has no {symbol}"
    if kind == "report":
        return None if (root / value.split("#")[0]).exists() else f"no file {value}"
    if kind == "source":
        return None if value.strip() else "empty source"
    return f"unknown evidence kind {kind}"


def run(root: Path = ROOT) -> Result:
    res = Result()
    reg = yaml.safe_load((root / REGISTER).read_text(encoding="utf-8"))
    aliases: dict[str, int] = {}
    res.props = props = proposition_numbers(root, aliases)
    numbers = set(props.values()) | set(aliases.values())
    checks = check_ids(root)
    sym_cache: dict = {}
    docs: dict[str, str] = {}

    def doc_text(name: str) -> str | None:
        if name not in docs:
            p = root / name
            docs[name] = norm(p.read_text(encoding="utf-8")) if p.is_file() else None
        return docs[name]

    # A claim that repeats another (the README summing up a guide) cites it as {claim: ID}
    # and is as strong as the claim it cites.
    direct = {
        c.get("id"): {k for ev in c.get("evidence") or [] if isinstance(ev, dict) for k in ev}
        for c in reg.get("claims", [])
    }
    seen: set[str] = set()
    quotes: dict[str, list[str]] = {}
    for c in reg.get("claims", []):
        cid = c.get("id", "?")
        where = f"{cid}"
        if cid in seen:
            res.problems.append(f"{where}: duplicate id")
        seen.add(cid)
        for key in ("doc", "quote", "status", "evidence"):
            if key not in c:
                res.problems.append(f"{where}: missing '{key}'")
        if c.get("status") not in STATUSES:
            res.problems.append(f"{where}: status must be one of {', '.join(STATUSES)}")
        doc, quote = c.get("doc", ""), c.get("quote", "")
        qs = quote if isinstance(quote, list) else [quote]
        text = doc_text(doc)
        if text is None:
            res.problems.append(f"{where}: no document {doc}")
        else:
            for q in qs:
                if norm(q) not in text:
                    res.problems.append(f"{where}: quote not found in {doc}: \"{q[:80]}\"")
        quotes.setdefault(doc, []).extend(norm(q) for q in qs)
        kinds = []
        resolved = []
        for ev in c.get("evidence") or []:
            if not isinstance(ev, dict) or len(ev) != 1:
                res.problems.append(f"{where}: evidence items are one-key maps, e.g. {{prop: prop:dgauss}}")
                continue
            (kind, value), = ev.items()
            value = str(value)
            if kind == "claim":
                if value not in direct:
                    why = f"cites claim {value}, which the register does not have"
                elif not direct[value] & set(STRONG_KINDS):
                    why = f"cites claim {value}, which has no prop, check, test or report of its own"
                else:
                    why = None
                    kinds.extend(direct[value])
            else:
                kinds.append(kind)
                why = _resolve(kind, value, root, props, checks, sym_cache)
            if why:
                res.problems.append(f"{where}: {why}")
            item = {"kind": kind, "ref": value}
            if kind == "prop" and value in props:
                item["number"] = props[value]
                if value in aliases:
                    item["named"] = aliases[value]
            resolved.append(item)
        if c.get("status") in ("supported", "partial", "limitation") and not set(kinds) & set(STRONG_KINDS):
            res.problems.append(f"{where}: a claim about relsynth needs a prop, check, test or report")
        res.claims.append({**{k: v for k, v in c.items() if k != "evidence"}, "evidence": resolved})

    exempt: dict[str, list[str]] = {}
    for item in reg.get("not_claims", []):
        if not item.get("reason"):
            res.problems.append(f"not_claims in {item.get('doc')}: give a reason for \"{str(item.get('quote'))[:60]}\"")
        q = norm(str(item.get("quote", "")))
        text = doc_text(item.get("doc", ""))
        if text is None or q not in text:
            res.problems.append(f"not_claims: quote not found in {item.get('doc')}: \"{q[:80]}\"")
        exempt.setdefault(item.get("doc", ""), []).append(q)

    for doc in reg.get("scope", []):
        path = root / doc
        if not path.is_file():
            res.problems.append(f"scope: no document {doc}")
            continue
        covered = quotes.get(doc, []) + exempt.get(doc, [])
        for s in sentences(path.read_text(encoding="utf-8")):
            if TRIGGER.search(s) and not any(q and (q in s or s in q) for q in covered):
                res.unclaimed.setdefault(doc, []).append(s)
        for s in res.unclaimed.get(doc, []):
            res.problems.append(f"{doc}: unregistered claim: \"{s[:110]}\"")

    ids = {c.get("id") for c in reg.get("claims", [])}
    for path in sorted((root / REGISTER.parent).glob("*.md")):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        for ref in sorted(set(re.findall(r"\b(?:PRV|DP|SEC|GOV|RM)-\d+\b", text)) - ids):
            res.problems.append(f"{rel}: cites {ref}, which the register does not have")
        for label in sorted(set(re.findall(r"`(prop:[\w-]+)`", text)) - set(props)):
            res.problems.append(f"{rel}: cites {label}, which the proofs do not define")
        for cid in sorted(set(re.findall(r"\bC\d{2,4}\b", text)) - checks):
            res.problems.append(f"{rel}: cites check {cid}, which does not exist")

    for path in sorted(root.glob("*.md")) + sorted((root / "docs").rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(("docs/research/", "CHANGELOG")):
            continue
        for m in PROP_REF.finditer(path.read_text(encoding="utf-8")):
            for a, b in re.findall(r"(\d+)(?:\s*[–-]\s*(\d+))?", m.group(1)):
                lo, hi = int(a), int(b or a)
                if hi < lo:
                    res.problems.append(f"{rel}: '{m.group(0).strip()}' is a range that runs backwards")
                for n in sorted({lo, hi}):
                    if n not in numbers:
                        ref = m.group(0).strip()
                        res.problems.append(f"{rel}: '{ref}' names Proposition {n}, which the proofs do not define")
    return res


def suggest(res: Result, doc: str) -> str:
    stubs = [
        {"id": "CL-XXX", "doc": doc, "quote": s, "status": "supported", "area": "", "evidence": [{"test": ""}]}
        for s in res.unclaimed.get(doc, [])
    ]
    return yaml.safe_dump(stubs, sort_keys=False, allow_unicode=True, width=120)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suggest", metavar="DOC", help="print register stubs for the unregistered claims of DOC")
    ap.add_argument("--json", metavar="FILE", help="write the resolved register as JSON")
    args = ap.parse_args(argv)
    res = run()
    if args.suggest:
        print(suggest(res, args.suggest))
        return 0
    if args.json:
        Path(args.json).write_text(json.dumps({"claims": res.claims}, indent=2, ensure_ascii=False), encoding="utf-8")
    for p in res.problems:
        print(p)
    n = len(res.claims)
    if res.problems:
        print(f"{len(res.problems)} problem(s) in {n} claims")
        return 1
    reg = yaml.safe_load((ROOT / REGISTER).read_text(encoding="utf-8"))
    print(f"ok: {n} claims traced, {len(reg.get('scope', []))} documents in scope, {len(res.props)} propositions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
