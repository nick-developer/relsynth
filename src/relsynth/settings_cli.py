# SPDX-License-Identifier: Apache-2.0
"""`relsynth settings ...`: see, explain, check and write engine settings.

    relsynth settings show spec.yaml [--all]       effective values and where each comes from
    relsynth settings explain copula.sample_rows   type, default, range, tuning direction
    relsynth settings validate my-settings.yaml    check a settings file
    relsynth settings template --out settings.yaml every key at its default, commented
    relsynth settings reference                    the Markdown reference (docs/settings-reference.md)
"""
from __future__ import annotations

from pathlib import Path

from .errors import EXIT_OK


def _emit(text, dest, out, what):
    if dest:
        Path(dest).write_text(text, encoding="utf-8")
        out.line(f"wrote {dest} ({what})")
        out.put(written=str(dest))
    else:
        out.line(text.rstrip("\n"))


def cmd_settings(a, out):
    from . import settings as S
    if a.action == "show":
        if not a.target:
            raise S.SettingsError("settings show needs a spec")
        from .spec import from_ir, read_yaml
        raw = from_ir(read_yaml(a.target))
        prov = S.provenance(raw, Path(a.target).parent)
        rows = {k: {"value": v, "source": src} for k, (v, src) in prov.items() if a.all or src != "default"}
        out.put(settings=rows)
        if not rows:
            out.line("every setting is at its default")
        for k, r in rows.items():
            out.line(f"{k:<40} {r['value']!s:<24} {r['source']}")
        return EXIT_OK
    if a.action == "explain":
        if not a.target:
            raise S.SettingsError("settings explain needs a key, e.g. copula.sample_rows")
        p = S.params().get(a.target)
        if p is None:
            raise S.SettingsError(S._unknown(a.target))
        d = {"key": p.key, "type": p.type, "default": p.default, "low": p.low, "high": p.high,
             "choices": list(p.choices), "tunable": p.tunable, "scale": p.scale, "affects": list(p.affects),
             "direction": p.direction, "doc": p.doc}
        out.put(param=d)
        for k, v in d.items():
            if v not in (None, [], {}, ""):
                out.line(f"{k:<10} {v}")
        return EXIT_OK
    if a.action == "validate":
        if not a.target:
            raise S.SettingsError("settings validate needs a settings file")
        syn = S.read_file(a.target)               # raises with every error
        out.put(valid=True, keys=sorted(S.diff(S.Settings(syn).merged())))
        out.line(f"{a.target}: valid ({len(S.diff(S.Settings(syn).merged()))} keys differ from the defaults)")
        return EXIT_OK
    if a.action == "template":
        _emit(S.template(), a.out, out, "settings template")
        return EXIT_OK
    _emit(S.reference(), a.out, out, "settings reference")
    return EXIT_OK


def add_parser(sub, common):
    s = sub.add_parser("settings", help="engine settings: show, explain, validate, template, reference")
    s.add_argument("action", choices=["show", "explain", "validate", "template", "reference"])
    s.add_argument("target", nargs="?", help="spec (show), key (explain) or settings file (validate)")
    s.add_argument("--all", action="store_true", help="show: every key, not only the ones set")
    s.add_argument("--out", help="template / reference: write to this file")
    s.set_defaults(fn=cmd_settings)
