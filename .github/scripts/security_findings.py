# SPDX-License-Identifier: Apache-2.0
"""Gather the open security findings of the repository into one small JSON file.

Used by the ``security-patch`` workflow: its ``collect`` job downloads the raw
reports and this script turns them into the work list the ``patch`` job hands to
Claude.  Only the fields needed to make a fix are kept, every text is cut to a
short length and stripped of control characters, so the agent never sees long
free text written by third parties.

Sources (each optional; a missing or unreadable file is reported, not fatal):

* ``--dependabot``: ``GET /repos/{repo}/dependabot/alerts?state=open`` (needs a
  token with the Dependabot alerts permission; the workflow token has none);
* ``--code-scanning``: ``GET /repos/{repo}/code-scanning/alerts?state=open``;
  only alerts above warning are kept (rule severity ``error`` or security
  severity ``high``/``critical``: project rule 1 in CLAUDE.md);
  findings in paths the agent may not change (workflows, tests, tools, scanner
  configuration) go to ``manual`` and are not counted;
* ``--pip-audit LABEL=FILE``: ``pip-audit -f json`` reports, for example of the
  lowest versions the ``pyproject.toml`` ranges allow and of the newest ones;
  only vulnerabilities with a fixed version are kept;
* ``--open-prs``: ``GET /repos/{repo}/pulls?state=open``, to list the open
  Dependabot pull requests (so the agent does not duplicate them) and to tell
  whether an earlier ``security-patch/`` pull request is still open.

Writes the work list to ``--out`` and a Markdown summary (counts per source only,
since run logs of a public repository are public) to stdout; with ``--github-output`` it also
appends ``count=<n>`` and ``pending_pr=<url or empty>`` to that file.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

BRANCH_PREFIX = "security-patch/"
MAX_ITEMS = 200            # per source; keeps the job output far below GitHub's 1 MB limit
MAX_TEXT = 300             # characters kept from any free-text field
_SECURITY_LEVELS = {"high", "critical"}
# paths the patch job may not change (security-patch.yml refuses such a patch); findings
# there are listed for the maintainer instead of being handed to the agent
# (the same list as .github/scripts/patch_gate.py; hash-locked .github/requirements/
# files are recompiled by hand)
PROTECTED = (".github/", "tests/", "tools/", ".semgrep/", ".claude/")
PROTECTED_FILES = {".gitleaks.toml", ".coderabbit.yaml", "CLAUDE.md", "CONTRIBUTING.md"}


def protected(path: str | None) -> bool:
    """True when ``path`` lies where the agent may not make changes."""
    p = (path or "").lstrip("/")
    return p.startswith(PROTECTED) or p in PROTECTED_FILES


def clip(value: Any, limit: int = MAX_TEXT) -> str:
    """Return ``value`` as one printable line of at most ``limit`` characters."""
    text = "" if value is None else str(value)
    text = "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def load(path: str | None, problems: list[str]) -> Any:
    """Read a JSON list (an API listing); on any failure note it in ``problems`` and return ``None``.

    ``pip-audit`` reports are objects; an API error body (``{"message": ...}``, which
    ``gh api`` writes before it fails) is reported as a problem, not read as "no alerts".
    """
    if not path:
        return None
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        problems.append(f"{path}: {type(exc).__name__}")
        return None
    if isinstance(doc, dict) and "message" in doc and "dependencies" not in doc:
        problems.append(f"{path}: API error: {clip(doc.get('message'), 120)}")
        return None
    return doc


def _pages(doc: Any) -> list[dict]:
    """Flatten ``gh api --paginate --slurp`` output (a list of pages) or a single page."""
    if not isinstance(doc, list):
        return []
    if doc and all(isinstance(p, list) for p in doc):
        return [item for page in doc for item in page if isinstance(item, dict)]
    return [item for item in doc if isinstance(item, dict)]


def dependabot(doc: Any) -> list[dict]:
    """Open Dependabot alerts, one entry per alert."""
    out = []
    for a in _pages(doc):
        if a.get("state", "open") != "open":
            continue
        dep = a.get("dependency") or {}
        pkg = dep.get("package") or {}
        adv = a.get("security_advisory") or {}
        vul = a.get("security_vulnerability") or {}
        patched = (vul.get("first_patched_version") or {}).get("identifier")
        out.append({
            "source": "dependabot",
            "alert": a.get("number"),
            "ecosystem": clip(pkg.get("ecosystem"), 40),
            "package": clip(pkg.get("name"), 120),
            "manifest": clip(dep.get("manifest_path"), 200),
            "scope": clip(dep.get("scope"), 20),
            "advisory": clip(adv.get("ghsa_id"), 40),
            "cve": clip(adv.get("cve_id"), 40),
            "severity": clip(adv.get("severity") or vul.get("severity"), 20),
            "vulnerable": clip(vul.get("vulnerable_version_range"), 120),
            "fixed_in": clip(patched, 60) if patched else None,
            "summary": clip(adv.get("summary"), 160),
        })
    return out[:MAX_ITEMS]


def above_warning(alert: dict) -> bool:
    """Project rule 1: rule severity ``error`` or security severity high/critical."""
    rule = alert.get("rule") or {}
    return rule.get("severity") == "error" or (rule.get("security_severity_level") or "") in _SECURITY_LEVELS


def code_scanning(doc: Any) -> list[dict]:
    """Open code-scanning alerts above warning, one entry per alert."""
    out = []
    for a in _pages(doc):
        if a.get("state", "open") != "open" or not above_warning(a):
            continue
        rule = a.get("rule") or {}
        inst = a.get("most_recent_instance") or {}
        loc = inst.get("location") or {}
        out.append({
            "source": "code-scanning",
            "alert": a.get("number"),
            "tool": clip((a.get("tool") or {}).get("name"), 40),
            "rule": clip(rule.get("id"), 120),
            "severity": clip(rule.get("security_severity_level") or rule.get("severity"), 20),
            "path": clip(loc.get("path"), 200),
            "line": loc.get("start_line"),
            "message": clip((inst.get("message") or {}).get("text"), 200),
        })
    return out[:MAX_ITEMS]


def pip_audit(label: str, doc: Any) -> list[dict]:
    """Packages with a fixable vulnerability in one ``pip-audit -f json`` report, one entry per package."""
    out = []
    deps = doc.get("dependencies", []) if isinstance(doc, dict) else []
    for dep in deps:
        advisories: dict[str, list[str]] = {}
        for v in dep.get("vulns") or []:
            fixes = [clip(f, 40) for f in v.get("fix_versions") or []][:5]
            if fixes:
                advisories.setdefault(clip(v.get("id"), 40), fixes)
        if advisories:
            out.append({
                "source": f"pip-audit:{clip(label, 40)}",
                "package": clip(dep.get("name"), 120),
                "version": clip(dep.get("version"), 60),
                "advisories": advisories,
            })
    return out[:MAX_ITEMS]


def open_prs(doc: Any, repo: str = "") -> tuple[list[dict], str]:
    """Open Dependabot pull requests, and the URL of an open security-patch pull request.

    Only a ``security-patch/`` branch of ``repo`` itself counts as pending: anyone can
    open a pull request from a fork branch with that name, and it must not stop the job.
    """
    dependabot_prs, pending = [], ""
    for pr in _pages(doc):
        head = pr.get("head") or {}
        ref = head.get("ref") or ""
        same_repo = not repo or ((head.get("repo") or {}).get("full_name") == repo)
        login = (pr.get("user") or {}).get("login") or ""
        if ref.startswith(BRANCH_PREFIX) and same_repo:
            pending = pending or clip(pr.get("html_url"), 200)
        elif login == "dependabot[bot]" and same_repo:
            dependabot_prs.append({"number": pr.get("number"), "title": clip(pr.get("title"), 160), "branch": clip(ref, 160)})
    return dependabot_prs[:MAX_ITEMS], pending


def split_manual(found: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate the findings the agent may fix from those in protected paths.

    Workflow alerts (zizmor, Dependabot ``github-actions``), and alerts in tests, tools or
    scanner configuration, need the maintainer: the patch job refuses changes there.
    """
    fixable, manual = [], []
    for f in found:
        where = f.get("path") or f.get("manifest")
        needs_owner = f.get("ecosystem") == "actions" or f.get("tool") == "zizmor" or protected(where)
        (manual if needs_owner else fixable).append(f)
    return fixable, manual


def summary(work: dict) -> str:
    """Counts per source only.

    The job summary and log of a public repository are public, while code-scanning and
    Dependabot alerts are visible only to maintainers, so no alert detail is printed.
    """
    counts: dict[str, int] = {}
    for f in work["findings"]:
        counts[f["source"]] = counts.get(f["source"], 0) + 1
    lines = ["## Security findings", ""]
    if counts:
        lines += ["| source | findings |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    else:
        lines.append("No open finding above warning with a fix available.")
    if work.get("manual"):
        lines += ["", f"Findings in protected paths, left to the maintainer: {len(work['manual'])}"]
    if work["dependabot_prs"]:
        lines += ["", f"Open Dependabot pull requests: {len(work['dependabot_prs'])}"]
    if work["pending_pr"]:
        lines += ["", f"An earlier security-patch pull request is still open: {work['pending_pr']}"]
    for p in work["problems"]:
        lines.append(f"- source not read: {p}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dependabot")
    ap.add_argument("--code-scanning")
    ap.add_argument("--pip-audit", action="append", default=[], metavar="LABEL=FILE")
    ap.add_argument("--open-prs")
    ap.add_argument("--repo", default="", help="owner/name; only its own branches count as pending pull requests")
    ap.add_argument("--out", required=True)
    ap.add_argument("--github-output")
    args = ap.parse_args(argv)

    problems: list[str] = []
    found = dependabot(load(args.dependabot, problems)) + code_scanning(load(args.code_scanning, problems))
    for item in args.pip_audit:
        label, _, path = item.partition("=")
        if not path:
            ap.error(f"--pip-audit expects LABEL=FILE, got {item!r}")
        found += pip_audit(label, load(path, problems))
    findings, manual = split_manual(found)
    prs_doc = load(args.open_prs, problems)
    dependabot_prs, pending = open_prs(prs_doc, args.repo)
    if args.open_prs and prs_doc is None:
        # unknown is not "none": an earlier security-patch pull request may be open
        pending = "(open pull requests could not be read)"

    work = {"findings": findings, "manual": manual, "dependabot_prs": dependabot_prs, "pending_pr": pending,
            "problems": problems}
    Path(args.out).write_text(json.dumps(work, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stdout.write(summary(work))
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as fh:
            fh.write(f"count={len(findings)}\npending_pr={pending}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
