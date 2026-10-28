# SPDX-License-Identifier: Apache-2.0
"""Check the licences of relsynth's installed runtime dependencies.

    python tools/check_licenses.py                  # extras: all, speed
    python tools/check_licenses.py --report licenses.md

Walks the dependency closure of the installed ``relsynth`` distribution
(``Requires-Dist``, with the extras given by ``--extras``) and classifies each
package's licence from its ``License-Expression`` (PEP 639), its ``License``
field or its trove classifiers.  Exit code 1 when a package has a licence that
is not allowed, or one that cannot be determined and has no entry in OVERRIDES.

Policy (CONTRIBUTING.md, "Dependencies"): permissive licences are allowed
everywhere; weak copyleft (LGPL, MPL) only in optional extras, where the user
installs the package; strong copyleft (GPL, AGPL) and unknown licences are not
allowed.
"""
from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata as md

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PERMISSIVE = {"MIT", "MIT-0", "BSD-2-Clause", "BSD-3-Clause", "0BSD", "ISC", "Apache-2.0", "PSF-2.0", "Python-2.0",
              "Zlib", "CC0-1.0", "UPL-1.0", "Unlicense", "HPND"}
WEAK_COPYLEFT = {"LGPL-2.1-only", "LGPL-2.1-or-later", "LGPL-3.0-only", "LGPL-3.0-or-later", "MPL-2.0"}

# Packages whose metadata carries no SPDX expression and no unambiguous classifier.
# Each entry was read from the package's licence file; re-check on a major upgrade.
OVERRIDES = {
    "scipy": "BSD-3-Clause",
    "python-dateutil": "Apache-2.0 OR BSD-3-Clause",
    "pyodbc": "MIT",
    "numba": "BSD-2-Clause",
    "llvmlite": "BSD-2-Clause AND Apache-2.0",
    "greenlet": "MIT AND PSF-2.0",
    "protobuf": "BSD-3-Clause",
    "pyasn1": "BSD-2-Clause",
}

CLASSIFIERS = {
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: BSD License": "BSD-3-Clause",
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0-only",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
}
SHORT = {"MIT": "MIT", "MIT LICENSE": "MIT", "BSD": "BSD-3-Clause", "BSD-3-CLAUSE": "BSD-3-Clause",
         "APACHE 2.0": "Apache-2.0", "APACHE-2.0": "Apache-2.0", "APACHE LICENSE 2.0": "Apache-2.0"}


def closure(root: str, extras: list[str]) -> dict[str, tuple[md.Distribution, bool]]:
    """{name: (distribution, optional)}; optional = reachable only through an extra of the root."""
    found: dict[str, tuple[md.Distribution, bool]] = {}

    def visit(name: str, wanted: tuple[str, ...], optional: bool) -> None:
        key = canonicalize_name(name)
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            return
        if key in found and (not found[key][1] or optional) and not wanted:
            return
        found[key] = (dist, optional and found.get(key, (None, True))[1])
        for line in dist.requires or []:
            req = Requirement(line)
            if req.marker is not None:
                if "extra" in str(req.marker):
                    if not any(req.marker.evaluate({"extra": e}) for e in wanted):
                        continue
                elif not req.marker.evaluate():
                    continue
            via_extra = optional or (key == canonicalize_name(root) and req.marker is not None and "extra" in str(req.marker))
            visit(req.name, tuple(req.extras), via_extra)

    visit(root, tuple(extras), False)
    found.pop(canonicalize_name(root), None)
    return found


def expression(name: str, dist: md.Distribution) -> str | None:
    if name in OVERRIDES:
        return OVERRIDES[name]
    meta = dist.metadata
    if meta.get("License-Expression"):
        return meta["License-Expression"]
    text = (meta.get("License") or "").strip()
    if text.upper() in SHORT:
        return SHORT[text.upper()]
    ids = sorted({CLASSIFIERS[c] for c in meta.get_all("Classifier") or [] if c in CLASSIFIERS})
    return " OR ".join(ids) if ids else None


def verdict(expr: str | None, optional: bool) -> str:
    """'allowed', 'optional-only' or 'denied' for an SPDX expression (AND binds tighter than OR)."""
    if not expr:
        return "denied"

    def term(t: str) -> int:              # 2 permissive, 1 weak copyleft, 0 other
        t = t.strip().strip("()").strip()
        return 2 if t in PERMISSIVE else 1 if t in WEAK_COPYLEFT else 0

    best = max(min(term(t) for t in re.split(r"\s+AND\s+", alt)) for alt in re.split(r"\s+OR\s+", expr))
    if best == 2:
        return "allowed"
    if best == 1:
        return "allowed (optional extra)" if optional else "denied"
    return "denied"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extras", default="all,speed")
    ap.add_argument("--report", help="write a Markdown table of every dependency and its licence")
    a = ap.parse_args(argv)
    rows = []
    for name, (dist, optional) in sorted(closure("relsynth", [e for e in a.extras.split(",") if e]).items()):
        expr = expression(name, dist)
        rows.append((name, dist.version, expr or "unknown", "optional" if optional else "required", verdict(expr, optional)))
    lines = ["| Package | Version | Licence | Dependency | Verdict |", "|---|---|---|---|---|"]
    lines += [f"| {n} | {v} | {e} | {o} | {d} |" for n, v, e, o, d in rows]
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write("# Dependency licences\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    bad = [r for r in rows if r[4] == "denied"]
    for n, v, e, o, _ in bad:
        print(f"::error::{n} {v}: licence '{e}' is not allowed for a {o} dependency", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
