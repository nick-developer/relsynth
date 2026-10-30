# SPDX-License-Identifier: Apache-2.0
"""Compute presets for generator plug-ins (the ``compute.yaml`` file).

relsynth's default engine is CPU-only.  A generator plug-in declares its GPU
need (``Capabilities.gpu``: never, optional, required); the spec chooses a
preset (``synthesis.compute.preset``: cpu, gpu-auto, gpu-single, gpu-multi or
one from ``synthesis.compute.file``) with expert ``overrides``.  resolve()
turns the two into a ComputePlan for one generator, or fails with a reason:

* a CPU-only generator always gets the CPU (a GPU preset is noted, not an error);
* a GPU preset gets the GPUs it asks for when they are visible; when they are
  not, the preset's ``fallback`` decides: ``cpu`` (only for a generator whose
  GPU need is optional) or ``error``;
* the cpu preset never uses a GPU, and a generator that requires one fails.

GPUs are found without importing a deep-learning framework: the device lists
of ``nvidia-smi -L`` and ``rocm-smi --showid``, restricted by
``CUDA_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES``.  ``RELSYNTH_GPUS``
(for example ``cuda:2`` or ``none``) replaces the probe (tests, containers).
"""
from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess  # nosec B404 - fixed device-listing commands, no shell, no user input
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

FIELDS: dict[str, tuple | None] = {"device": ("cpu", "gpu"), "gpus": None, "devices": None, "backend": ("auto", "cuda", "rocm"),
          "precision": ("fp32", "tf32", "bf16", "fp16"), "batch_size": None, "memory_limit": None,
          "threads": None, "fallback": ("cpu", "error")}
PRESET_FILE = Path(__file__).with_name("data") / "compute.yaml"


class ComputeError(ValueError):
    exit_code = 2


@dataclass
class ComputePlan:
    """What a generator may use.  device: cpu, cuda or rocm."""
    device: str = "cpu"
    device_ids: list = field(default_factory=list)
    precision: str = "fp32"
    batch_size: int | None = None
    memory_limit_bytes: int | None = None
    threads: int | None = None
    preset: str = "cpu"
    notes: list = field(default_factory=list)

    @property
    def gpu(self):
        return self.device != "cpu"

    def to_dict(self):
        return asdict(self)


def presets(file=None, base_dir=None) -> dict:
    """The built-in presets, updated by the presets of `file` (if given)."""
    out = yaml.safe_load(PRESET_FILE.read_text())["presets"]
    if file:
        p = Path(file)
        if not p.is_absolute() and base_dir:
            p = Path(base_dir) / p
        if not p.exists():
            raise ComputeError(f"synthesis.compute.file: {p} not found")
        extra = yaml.safe_load(p.read_text()) or {}
        if not isinstance(extra.get("presets"), dict):
            raise ComputeError(f"{p}: a compute file needs a `presets` mapping")
        for name, pr in extra["presets"].items():
            out[name] = {**out.get(name, out["cpu"]), **(pr or {})}
    return out


def _size(v):
    if v is None:
        return None
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]i?B|B)?\s*", str(v), flags=re.I)
    if not m:
        raise ComputeError(f"memory_limit {v!r}: use a size such as 8GiB or 512MiB")
    unit = (m.group(2) or "B").upper()
    mult = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
            "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}[unit]
    return int(float(m.group(1)) * mult)


def validate(settings: dict, base_dir=None) -> list[str]:
    """Errors of a synthesis.compute block."""
    errs = []
    try:
        pr = presets((settings or {}).get("file"), base_dir)
    except (ComputeError, OSError, yaml.YAMLError) as e:
        return [str(e)]
    name = (settings or {}).get("preset", "cpu")
    if name not in pr:
        errs.append(f"synthesis.compute.preset {name!r} is not a preset (have: {', '.join(pr)})")
        return errs
    merged = {**pr[name], **((settings or {}).get("overrides") or {})}
    for k, v in merged.items():
        if k not in FIELDS:
            errs.append(f"synthesis.compute: unknown field {k!r} (allowed: {', '.join(FIELDS)})")
        elif (allowed := FIELDS[k]) and v not in allowed:
            errs.append(f"synthesis.compute.{k} must be one of {', '.join(allowed)}, not {v!r}")
    if merged.get("gpus") not in ("all", None) and not (isinstance(merged.get("gpus"), int) and merged["gpus"] >= 0):
        errs.append(f"synthesis.compute.gpus must be a number >= 0 or all, not {merged.get('gpus')!r}")
    for k in ("batch_size", "threads"):
        v = merged.get(k, "auto")
        if v != "auto" and not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
            errs.append(f"synthesis.compute.{k} must be auto or a positive integer, not {v!r}")
    d = merged.get("devices")
    if d is not None and not (isinstance(d, list) and all(isinstance(i, int) and i >= 0 for i in d)):
        errs.append("synthesis.compute.devices must be null or a list of device ids")
    try:
        _size(merged.get("memory_limit"))
    except ComputeError as e:
        errs.append(f"synthesis.compute.{e}")
    return errs


def _visible(env, ids):
    v = os.environ.get(env)
    if v is None or v.strip() == "":
        return ids
    keep = [int(x) for x in v.split(",") if x.strip().isdigit()]
    return [i for i in ids if i in keep]


def _run(cmd):
    exe = shutil.which(cmd[0])
    if not exe:
        return ""
    try:
        return subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=10, check=False).stdout  # noqa: S603 # nosec B603
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_gpus() -> dict:
    """{backend: [device ids]} of the GPUs this process can see."""
    fake = os.environ.get("RELSYNTH_GPUS")
    if fake is not None:
        out = {}
        for part in fake.split(","):
            m = re.fullmatch(r"\s*(cuda|rocm):(\d+)\s*", part)
            if m:
                out[m.group(1)] = list(range(int(m.group(2))))
        return out
    out = {}
    cuda = [int(m) for m in re.findall(r"^GPU (\d+):", _run(["nvidia-smi", "-L"]), flags=re.M)]
    if cuda:
        out["cuda"] = _visible("CUDA_VISIBLE_DEVICES", cuda)
    rocm = sorted({int(m) for m in re.findall(r"GPU\[(\d+)\]", _run(["rocm-smi", "--showid"]))})
    if rocm:
        out["rocm"] = _visible("HIP_VISIBLE_DEVICES", rocm)
    return {k: v for k, v in out.items() if v}


def resolve(settings: dict | None, gpu_need: str = "never", who: str = "", base_dir=None,
            gpus: dict | None = None) -> ComputePlan:
    """The compute plan of one generator.  settings: the spec's synthesis.compute."""
    settings = settings or {}
    errs = validate(settings, base_dir)
    if errs:
        raise ComputeError("; ".join(errs))
    name = settings.get("preset", "cpu")
    cfg = copy.deepcopy({**presets(settings.get("file"), base_dir)[name], **(settings.get("overrides") or {})})
    thr = None if cfg.get("threads", "auto") == "auto" else int(cfg["threads"])
    bs = None if cfg.get("batch_size", "auto") == "auto" else int(cfg["batch_size"])
    plan = ComputePlan(precision=cfg.get("precision", "fp32"), batch_size=bs, threads=thr, preset=name,
                       memory_limit_bytes=_size(cfg.get("memory_limit")))
    who = who or "generator"
    if gpu_need == "never":
        if cfg["device"] == "gpu":
            plan.notes.append(f"{who} runs on the CPU only; the GPU preset {name!r} does not apply to it")
        return plan
    if cfg["device"] == "cpu":
        if gpu_need == "required":
            raise ComputeError(f"{who} needs a GPU but the compute preset is {name!r}; choose gpu-auto, gpu-single "
                               "or gpu-multi in synthesis.compute.preset")
        return plan
    found = detect_gpus() if gpus is None else gpus
    backends = [cfg["backend"]] if cfg.get("backend", "auto") != "auto" else ["cuda", "rocm"]
    avail = next(((b, found[b]) for b in backends if found.get(b)), None)
    want = cfg.get("devices")
    if avail is not None and want is not None:
        ids = [i for i in want if i in avail[1]]
        if len(ids) < len(want):
            avail = None if not ids else (avail[0], ids)
        else:
            avail = (avail[0], ids)
    n = cfg.get("gpus", 1)
    if avail is not None and n != "all" and len(avail[1]) < int(n):
        avail = None
    if avail is None:
        what = f"{n} {'/'.join(backends)} GPU(s)" + (f" (devices {want})" if want is not None else "")
        if cfg.get("fallback") == "cpu" and gpu_need == "optional":
            plan.notes.append(f"{who}: {what} not found; running on the CPU (preset {name!r} has fallback: cpu)")
            return plan
        reason = "the generator cannot run on a CPU" if gpu_need == "required" else f"preset {name!r} has fallback: error"
        raise ComputeError(f"{who}: {what} not found and {reason}; found: {found or 'no GPU'}")
    backend, ids = avail
    plan.device = backend
    plan.device_ids = ids if n == "all" else ids[: int(n)]
    return plan
