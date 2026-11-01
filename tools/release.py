# SPDX-License-Identifier: Apache-2.0
"""Prepare a release and check that the repository says the same version everywhere.

    python tools/release.py check                 # exit 1 and list every stale version
    python tools/release.py bump 2.1.0            # write the version everywhere, date the changelog
    python tools/release.py bump 2.1.0 --date 2026-11-02
    python tools/release.py notes 2.1.0           # print the changelog section (release notes)

Every release publishes a new package to PyPI and updates README.md and the
documentation (docs/development/release-checklist.md).  The places that name
the version are listed in PINS below: the package version, the install and
image commands in the README and the CI/CD guide, the example pipelines, the
two Dockerfiles, the Helm chart and the service's OpenAPI document.
``check`` runs in CI on every pull request and in the release workflow (where
it also requires a changelog section for the version), so a release cannot go
out with a README that still installs the previous version.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "src/relsynth/_version.py"
CHANGELOG = ROOT / "CHANGELOG.md"
SEMVER = r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?"

# file -> patterns; group 1 is the text before the version, the version follows it
PINS: dict[str, list[str]] = {
    "README.md": [r"(relsynth(?:\[[\w,-]+\])?==)", r"(relsynth-)(?=" + SEMVER + r"-py3)", r"(relsynth:)"],
    "docs/cicd.md": [r"(relsynth(?:\[[\w,-]+\])?==)", r"(relsynth:)"],
    "examples/ci/github-actions.yml": [r"(relsynth(?:\[[\w,-]+\])?==)"],
    "examples/ci/gitlab-ci.yml": [r"(relsynth(?:\[[\w,-]+\])?==)"],
    "examples/ci/azure-pipelines.yml": [r"(relsynth(?:\[[\w,-]+\])?==)"],
    "examples/ci/Jenkinsfile": [r"(relsynth:)"],
    "Dockerfile": [r"(relsynth:)"],
    "deploy/docker/Dockerfile.service": [r"(relsynth-service:)"],
    "deploy/helm/relsynth/Chart.yaml": [r'(appVersion: ")', r"(^version: )"],
    "docs/service/deployment.md": [r'(relsynth-service, tag: ")'],
}
OPENAPI = ROOT / "docs/service/openapi.json"


def current_version() -> str:
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)', VERSION_FILE.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit("no __version__ in src/relsynth/_version.py")
    return m.group(1)


def _found(text: str, prefix: str) -> list[str]:
    return [m.group(m.lastindex or 0) for m in re.finditer(prefix + "(" + SEMVER + ")", text, flags=re.M)]


def changelog_section(version: str) -> str | None:
    text = CHANGELOG.read_text(encoding="utf-8")
    m = re.search(rf"^## \[?{re.escape(version)}\]?(?: \(\d{{4}}-\d{{2}}-\d{{2}}\))?\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else None


def check(require_changelog: bool) -> list[str]:
    version = current_version()
    problems = []
    for name, patterns in PINS.items():
        path = ROOT / name
        if not path.exists():
            problems.append(f"{name}: missing (listed in tools/release.py PINS)")
            continue
        text = path.read_text(encoding="utf-8")
        hits = [v for p in patterns for v in _found(text, p)]
        if not hits:
            problems.append(f"{name}: no version found (pattern changed?)")
        for v in hits:
            if v != version:
                problems.append(f"{name}: names {v}, package is {version}")
    if OPENAPI.exists():
        v = json.loads(OPENAPI.read_text(encoding="utf-8"))["info"]["version"]
        if v != version:
            problems.append(f"docs/service/openapi.json: info.version {v}, package is {version}")
    text = CHANGELOG.read_text(encoding="utf-8")
    if not re.search(r"^## Unreleased\s*$", text, re.M):
        problems.append("CHANGELOG.md: no '## Unreleased' section")
    section = changelog_section(version)
    if section is None or not section:
        msg = f"CHANGELOG.md: no section for {version}"
        if require_changelog:
            problems.append(msg)
    return problems


def bump(version: str, date: str) -> list[str]:
    if not re.fullmatch(SEMVER, version):
        raise SystemExit(f"not a version: {version}")
    changed = []
    text = VERSION_FILE.read_text(encoding="utf-8")
    text = re.sub(r'(__version__\s*=\s*["\'])[^"\']+', rf"\g<1>{version}", text)
    VERSION_FILE.write_text(text, encoding="utf-8")
    changed.append(str(VERSION_FILE.relative_to(ROOT)))
    for name, patterns in PINS.items():
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        new = text
        for p in patterns:
            new = re.sub(p + SEMVER, lambda m: m.group(1) + version, new, flags=re.M)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed.append(name)
    if OPENAPI.exists():
        doc = json.loads(OPENAPI.read_text(encoding="utf-8"))
        if doc["info"]["version"] != version:
            raw = OPENAPI.read_text(encoding="utf-8")
            raw = raw.replace(f'"version": "{doc["info"]["version"]}"', f'"version": "{version}"', 1)
            OPENAPI.write_text(raw, encoding="utf-8")
            changed.append(str(OPENAPI.relative_to(ROOT)))
    text = CHANGELOG.read_text(encoding="utf-8")
    if changelog_section(version) is None:
        if not re.search(r"^## Unreleased\s*$", text, re.M):
            raise SystemExit("CHANGELOG.md has no '## Unreleased' section to release")
        text = re.sub(r"^## Unreleased\s*$", f"## Unreleased\n\n## {version} ({date})", text, count=1, flags=re.M)
        CHANGELOG.write_text(text, encoding="utf-8")
        changed.append("CHANGELOG.md")
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--release", action="store_true", help="also require a changelog section for the version")
    b = sub.add_parser("bump")
    b.add_argument("version")
    b.add_argument("--date", default=dt.date.today().isoformat())
    n = sub.add_parser("notes")
    n.add_argument("version", nargs="?")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        problems = check(require_changelog=a.release)
        for p in problems:
            print(p)
        if not problems:
            print(f"ok: version {current_version()} everywhere")
        return 1 if problems else 0
    if a.cmd == "bump":
        for name in bump(a.version, a.date):
            print("updated", name)
        return 0
    section = changelog_section(a.version or current_version())
    if section is None:
        print(f"no changelog section for {a.version or current_version()}", file=sys.stderr)
        return 1
    print(section)
    return 0


if __name__ == "__main__":
    sys.exit(main())
