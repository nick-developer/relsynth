# SPDX-License-Identifier: Apache-2.0
"""One job, in its own process: ``python -m relsynth.service.runner <job.json>``.

The worker starts this process per job, so a job's memory, CPU and wall-clock
limits apply to it alone, cancellation is a signal to its process group, and
the source secret lives only in this process's environment.  Real rows exist
only in this process's memory: the outputs are the synthetic files, the spec
draft, and the reports, which the worker encrypts and then deletes.

Exit codes are relsynth's (0 ok, 1 quality gate, 2 spec, 3 I/O, 4 engine).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

SECRET_ENV = "RELSYNTH_SVC_SOURCE_SECRET"  # noqa: S105  # nosec B105 (the variable's name)
FILE_KINDS = ("csv", "parquet", "files", "sqlite")


def source_section(src: dict, has_secret: bool, secret_is_url: bool) -> dict:
    s = dict(src)
    for k in ("password", "password_env", "url_env"):     # only the service injects credentials
        s.pop(k, None)
    if has_secret:
        s["url_env" if secret_is_url else "password_env"] = SECRET_ENV
    return s


def main(argv: list[str]) -> int:
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    src = source_section(job["source"], job.get("has_secret", False), job.get("secret_is_url", False))
    result: dict = {"kind": job["kind"], "started": time.time()}
    code = 0
    try:
        import yaml

        import relsynth as rs
        if job["kind"] == "introspect":
            p = job.get("params", {})
            spec = rs.introspect(src, name=p.get("name", "draft"), sample_rows=int(p.get("sample_rows", 20000)),
                                 include=p.get("include"), exclude=p.get("exclude"))
            raw = dict(spec.raw)
            raw.pop("source", None)                         # the connection supplies it at run time
            raw.pop("target", None)                         # the service writes to its own storage
            (out / "spec.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            result.update(tables=len(spec.tables), review=list(spec.review))
        else:
            p = job.get("params", {})
            raw = yaml.safe_load(job["spec"]) or {}
            raw["source"] = src
            fmt = p.get("format", "csv")
            raw["target"] = {"kind": fmt, "path": str(out / "data")}
            spec = rs.Spec.from_dict(raw, base_dir=out)
            r = rs.run(spec, scale=p.get("scale"), seed=p.get("seed"), datasets=p.get("datasets"),
                       privacy=True, write=True, raise_errors=False)
            if r.report is not None:
                (out / "report.json").write_text(r.report.to_json(), encoding="utf-8")
                result.update(score=r.report.score, passed=r.report.passed, total=r.report.total)
            if r.data is not None:
                result["rows"] = {t: int(n) for t, n in r.data.rows.items()}
                if p.get("quality", True):
                    try:
                        q = rs.assess_quality(spec, r.data)
                        (out / "quality.json").write_text(q.to_json(), encoding="utf-8")
                        result["quality_ok"] = bool(q.ok)
                    except Exception as e:  # noqa: BLE001  the 2.0 report is optional; the run stands
                        result["quality_error"] = str(e)[:500]
            if r.errors:
                result["errors"] = [str(e)[:1000] for e in r.errors]
                from relsynth.errors import exit_code_for
                code = max((exit_code_for(e) for _, e in r.exceptions), default=4)
            gate = p.get("fail_under")
            if not code and gate is not None and r.report is not None and r.report.score < float(gate):
                code = 1
                result["errors"] = [f"quality gate: score {r.report.score:.4f} < {gate}"]
    except Exception as e:  # noqa: BLE001
        from relsynth.errors import exit_code_for
        code = exit_code_for(e)
        # the message only: a traceback can quote source values from the failing line
        result["errors"] = [f"{type(e).__name__}: {str(e)[:1000]}"]
        if job.get("debug"):
            result["traceback"] = traceback.format_exc()[-4000:]
    result["finished"] = time.time()
    result["exit_code"] = code
    (out / "result.json").write_text(json.dumps(result, default=str), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
