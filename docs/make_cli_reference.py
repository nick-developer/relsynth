"""Write docs/cli.md from the command-line parser, so the reference always
matches the code:

    python docs/make_cli_reference.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from relsynth.cli import build_parser

INTRO = """# Command-line reference

Generated from the parser by `python docs/make_cli_reference.py`; every option below is what `relsynth COMMAND --help` prints.

```
relsynth COMMAND [options]
```

Locations (`--source`, `--target`, `--synthetic`) are written as `csv:DIR`, `parquet:DIR`, `sqlite:FILE` or a database URL (`postgresql://`, `mssql://`, `oracle://`). A plain folder path means CSV. Instead of a location, `--source-url-env NAME` or `--target-url-env NAME` names an environment variable that holds the whole URL; `--source-password-env NAME` supplies only the password. Passwords are never accepted on the command line, and they are masked in all output.

Every command also takes:

| Option | Meaning |
|---|---|
| `--json` | Print exactly one JSON document on stdout (`command`, `version`, `exit_code`, and the command's results; `error` and `error_type` on failure). Progress goes to stderr. |
| `--log-level debug\\|info\\|warning\\|error` | Progress detail on stderr (default `info`; `RELSYNTH_QUIET=1` sets `warning`). |

Commands that read a spec also take `--set KEY=VALUE` (repeatable), which overrides any spec value by its dotted path, for example `--set synthesis.seed=7` or `--set 'synthesis.scale.tables={customer: 10}'` (the value is read as YAML).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Quality gate failed (`--fail-under`), or `verify` found a failing check |
| 2 | Invalid spec or arguments (including a spec that is not valid YAML) |
| 3 | Connection, permission or file error (also a target that exists with `if_exists: fail`) |
| 4 | Engine failure while fitting or sampling |
| 130 | Interrupted |
"""

COMMON = {"--json", "--log-level", "-h"}


def _opt_row(a: argparse.Action) -> str | None:
    if isinstance(a, argparse._HelpAction) or a.help == argparse.SUPPRESS:
        return None
    names = ", ".join(f"`{s}`" for s in a.option_strings) if a.option_strings else f"`{a.dest}`"
    if any(s in COMMON for s in a.option_strings):
        return None
    if a.option_strings and a.nargs != 0:
        meta = a.metavar or (a.dest.upper() if not a.choices else "|".join(map(str, a.choices)))
        names += f" {meta}".replace("|", "\\|") + (" ..." if a.nargs in ("*", "+") else "")
    default = "" if a.default in (None, False, argparse.SUPPRESS) or a.nargs == 0 else f"`{a.default}`"
    req = "yes" if (a.required or not a.option_strings) else ""
    hlp = (a.help or "").replace("|", "\\|").replace("%%", "%")
    return f"| {names} | {req} | {default} | {hlp} |"


def render() -> str:
    os.environ["COLUMNS"] = "100"          # usage lines wrap the same on every machine
    p = build_parser()
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    helps = {c.dest: c.help for c in sub._choices_actions}
    out = [INTRO, "## Commands", "", "| Command | What it does |", "|---|---|"]
    out += [f"| [`{n}`](#{n}) | {helps.get(n, '')} |" for n in sub.choices]
    for name, sp in sub.choices.items():
        h = helps.get(name, "")
        out += ["", f"## {name}", "", h[:1].upper() + h[1:] + ".", "", "```", sp.format_usage().strip(), "```", ""]
        rows = [r for r in (_opt_row(a) for a in sp._actions) if r]
        if rows:
            out += ["| Option | Required | Default | Meaning |", "|---|---|---|---|", *rows]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    dest = Path(__file__).with_name("cli.md")
    dest.write_text(render(), encoding="utf-8")
    print(f"wrote {dest}")
