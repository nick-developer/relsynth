# SPDX-License-Identifier: Apache-2.0
"""Job workers: lease a job, run it in a child process, store its outputs.

Durability.  A job is a row in the metadata store.  A worker leases it for
``lease_s`` seconds and renews the lease every ``heartbeat_s`` while the job
runs.  If the worker or its node dies, the lease runs out and any other worker
claims the job again (the attempt counter goes up), so runs survive restarts
and rolling upgrades.  Completion is recorded only by the lease holder, so a
worker that lost its lease cannot overwrite the outcome of the new attempt.

Retries.  I/O failures (exit code 3: a database restart, a network blip) are
retried with exponential backoff up to ``max_attempts``; spec errors (2), engine
errors (4) and quality-gate failures (1) are final.

Limits.  Each attempt runs in its own process group with optional address-space
and CPU-time limits (``job_memory_mb``, ``job_cpu_s``) and a wall-clock timeout
(``job_timeout_s``); in Kubernetes the pod's cgroup limits apply as well.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess  # nosec B404 (one fixed command: the job runner)
import sys
import threading
import time
import uuid
from pathlib import Path

import yaml

from . import secrets as sec
from .artifacts import Artifacts
from .config import Settings
from .store import Store, runs
from .telemetry import METRICS, span

log = logging.getLogger("relsynth.service.worker")

RETRYABLE = {3}
# the environment a job process inherits: nothing else (no worker credentials)
ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "PYTHONPATH", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "OPENBLAS_CORETYPE", "TMPDIR", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
            "ODBCSYSINI", "ORACLE_HOME", "TNS_ADMIN", "LD_LIBRARY_PATH")
DATA_EXT = (".csv", ".parquet")


def _limits(mem_mb: int, cpu_s: int):
    def apply():  # runs in the child between fork and exec
        import resource
        os.setsid()
        with contextlib.suppress(Exception):     # die with the worker (Linux): no orphaned jobs
            import ctypes
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
        if mem_mb:
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        if cpu_s:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
    return apply


class Worker:
    def __init__(self, settings: Settings, store: Store, artifacts: Artifacts, name: str | None = None):
        self.s, self.store, self.art = settings, store, artifacts
        self.name = name or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.stopping = threading.Event()
        self.tenants = settings.csv("worker_tenants") or None
        self.exclude = settings.csv("worker_exclude_tenants") or None

    # main loop ---------------------------------------------------------------
    def serve(self, poll_s: float = 2.0, once: bool = False) -> int:
        done = 0
        while not self.stopping.is_set():
            job = self.store.claim(self.name, self.s.lease_s, self.tenants, self.exclude)
            if job is None:
                if once:
                    return done
                self.stopping.wait(poll_s)
                continue
            self.process(job)
            done += 1
        return done

    def process(self, job: dict) -> str:
        tid, rid = job["tenant_id"], job["id"]
        if job["attempts"] > job["max_attempts"]:
            self.store.finish(tid, rid, self.name, state="failed", error="no attempts left (worker lost repeatedly)")
            self._audit(job, "run.failed", "error", {"reason": "attempts exhausted"})
            return "failed"
        if job["cancel_requested"]:
            self.store.finish(tid, rid, self.name, state="cancelled")
            self._audit(job, "run.cancelled", "ok", {})
            return "cancelled"
        self._audit(job, "run.started", "ok", {"attempt": job["attempts"], "worker": self.name})
        with span("relsynth.job", {"relsynth.tenant": tid, "relsynth.run": rid, "relsynth.kind": job["kind"]}):
            state = self._execute(job)
        METRICS.inc("relsynth_jobs_total", state=state, kind=job["kind"])
        return state

    # one attempt -------------------------------------------------------------
    def _execute(self, job: dict) -> str:
        tid, rid = job["tenant_id"], job["id"]
        work = Path(self.s.data_dir).resolve() / "work" / tid / f"{rid}-{job['attempts']}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, mode=0o700)
        try:
            env = {k: v for k, v in os.environ.items() if k in ENV_KEEP}
            try:
                payload = self._payload(job, work, env)
            except Exception as e:  # noqa: BLE001  bad connection or secret: final, with a reason
                return self._final(job, "failed", 2, f"cannot prepare the job: {e}")
            (work / "job.json").write_text(json.dumps(payload), encoding="utf-8")
            os.chmod(work / "job.json", 0o600)
            code, cancelled, timed_out, lost = self._spawn(job, work, env)
            if lost:
                log.warning("lost the lease on %s; another worker owns it now", rid)
                return "lost"
            result = {}
            with contextlib.suppress(Exception):
                result = json.loads((work / "out" / "result.json").read_text(encoding="utf-8"))
            if cancelled:
                return self._final(job, "cancelled", None, "cancelled by request", result)
            if timed_out:
                return self._final(job, "failed", 4, f"timed out after {self.s.job_timeout_s} s", result)
            if code in RETRYABLE and job["attempts"] < job["max_attempts"]:
                delay = min(30 * 2 ** (job["attempts"] - 1), 900)
                ok = self.store.finish(tid, rid, self.name, state="queued", lease_owner=None,
                                       next_attempt_at=time.time() + delay, exit_code=code,
                                       error="; ".join(result.get("errors", []))[:4000])
                self._audit(job, "run.retry", "error", {"exit_code": code, "delay_s": delay})
                return "queued" if ok else "lost"
            if code not in (0, 1) or not (work / "out").exists():
                return self._final(job, "failed", code, "; ".join(result.get("errors", [])) or f"exit code {code}",
                                   result)
            self._store_outputs(job, work / "out", result)
            return self._final(job, "succeeded", code, "; ".join(result.get("errors", [])) or None, result)
        finally:
            shutil.rmtree(work, ignore_errors=True)       # plaintext outputs never outlive the attempt

    def _payload(self, job: dict, work: Path, env: dict) -> dict:
        tid = job["tenant_id"]
        con = self.store.connection(tid, job["connection_id"]) if job["connection_id"] else None
        spec = self.store.spec(tid, job["spec_id"]) if job["spec_id"] else None
        if con is None:
            raise ValueError("the run's connection no longer exists")
        from .policy import check_source
        check_source(con["source"], tid, self.s)
        if con.get("secret_ref"):
            sec.check_ref(con["secret_ref"], tid)
            env["RELSYNTH_SVC_SOURCE_SECRET"] = sec.resolve(con["secret_ref"])
        return {"kind": job["kind"], "source": con["source"], "has_secret": bool(con.get("secret_ref")),
                "secret_is_url": bool((con.get("labels") or {}).get("secret_is_url")),
                "spec": spec["body"] if spec else None, "params": job.get("params") or {},
                "out_dir": str(work / "out")}

    def _spawn(self, job: dict, work: Path, env: dict) -> tuple[int, bool, bool, bool]:
        tid, rid = job["tenant_id"], job["id"]
        env["TMPDIR"] = str(work)
        p = subprocess.Popen([sys.executable, "-m", "relsynth.service.runner", str(work / "job.json")],  # noqa: S603  # nosec B603
                             cwd=work, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             preexec_fn=_limits(self.s.job_memory_mb, self.s.job_cpu_s))
        deadline = time.time() + self.s.job_timeout_s
        cancelled = timed_out = lost = False
        next_beat = time.time() + self.s.heartbeat_s
        while p.poll() is None:
            time.sleep(0.2)
            if self.stopping.is_set() and os.environ.get("RELSYNTH_SVC_DRAIN") == "kill":
                lost = True          # give the job back: its lease runs out and another worker resumes it
            if time.time() >= next_beat:
                next_beat = time.time() + self.s.heartbeat_s
                r = self.store.heartbeat(tid, rid, self.name, self.s.lease_s)
                if r is None:
                    lost = True
                elif r["cancel_requested"]:
                    cancelled = True
            if time.time() > deadline:
                timed_out = True
            if cancelled or timed_out or lost:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(p.pid, signal.SIGKILL)
                break
        return (p.wait(), cancelled, timed_out, lost)

    def _store_outputs(self, job: dict, out: Path, result: dict) -> None:
        tid, rid = job["tenant_id"], job["id"]
        if job["kind"] == "introspect":
            body = (out / "spec.yaml").read_text(encoding="utf-8")
            raw = yaml.safe_load(body) or {}
            p = job.get("params") or {}
            spec = self.store.add_spec(tid, job["project_id"], p.get("name") or "draft", body,
                                       list(raw.get("review") or []), job["connection_id"], job["created_by"])
            result["spec_id"] = spec["id"]
            self._audit(job, "spec.drafted", "ok", {"spec": spec["id"], "version": spec["version"]})
            return
        for name, kind in (("report.json", "report"), ("quality.json", "quality")):
            if (out / name).exists():
                self.art.put(tid, rid, kind, name, out / name)
        data = out / "data"
        if data.exists():
            for f in sorted(data.rglob("*")):
                if f.is_file() and f.suffix in DATA_EXT:
                    self.art.put(tid, rid, "data", str(f.relative_to(data)), f)

    def _final(self, job: dict, state: str, code, error, result: dict | None = None) -> str:
        tid, rid = job["tenant_id"], job["id"]
        summary = {k: v for k, v in (result or {}).items() if k not in ("traceback",)}
        values = dict(state=state, exit_code=code, error=(error or None) and str(error)[:4000], summary=summary)
        if state == "succeeded" and job["kind"] == "run":
            values["release_state"] = "pending_approval" if self.s.approval_internal else "approved"
        ok = self.store.finish(tid, rid, self.name, **values)
        if not ok:
            return "lost"
        self._audit(job, f"run.{state}", "ok" if state == "succeeded" else "error",
                    {"exit_code": code, **({"error": str(error)[:300]} if error else {})})
        return state

    def _audit(self, job, action, outcome, detail):
        self.store.append_audit(job["tenant_id"], f"service:{self.name}", action, job["id"], outcome, detail)


def drain_handler(worker: Worker):
    def h(_sig, _frm):
        worker.stopping.set()
    return h


__all__ = ["Worker", "drain_handler", "runs"]
