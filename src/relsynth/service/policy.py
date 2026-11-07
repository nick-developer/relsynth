# SPDX-License-Identifier: Apache-2.0
"""What a tenant's connection may point at.

* ``kind`` must be in ``allowed_kinds``.
* File sources (csv, parquet, files, sqlite) must lie below ``<file root>/<tenant>``
  after resolving symlinks and ``..``, so one tenant cannot read another's
  uploads or the service's own files.
* Database URLs may name only hosts matching ``allowed_hosts`` (when set), which
  limits server-side request forgery from the workers' network position.
* Credentials never go in the source section: they come from ``secret_ref``.

What a tenant's spec may refer to (``spec_refusals``): nothing on the service's
own file system.  Internal keys (``_base_dir``) and file references
(``synthesis.settings_file`` other than a built-in ``preset:``, and
``synthesis.compute.file``) would let a spec read files outside the tenant's
area, so the service refuses them.
"""
from __future__ import annotations

import fnmatch
import os
from urllib.parse import urlsplit

from .auth import TENANT_RE

FILE_KINDS = ("csv", "parquet", "files", "sqlite")
CREDENTIAL_KEYS = ("password", "password_env", "url_env", "token", "private_key")


class SourceRefused(ValueError):
    pass


def check_source(src: dict, tenant: str, settings) -> dict:
    if not isinstance(src, dict) or "kind" not in src:
        raise SourceRefused("a connection needs a source section with a kind")
    kind = str(src["kind"])
    if kind not in settings.csv("allowed_kinds"):
        raise SourceRefused(f"source kind {kind!r} is not allowed on this service")
    for k in CREDENTIAL_KEYS:
        if k in src:
            raise SourceRefused(f"{k!r} is not allowed in a connection; store the credential in a secret "
                                "and set secret_ref")
    if kind in FILE_KINDS:
        if not TENANT_RE.match(tenant):
            raise SourceRefused("invalid tenant id")
        roots = settings.csv("file_roots")
        if not roots:
            raise SourceRefused("file sources are disabled (no RELSYNTH_SVC_FILE_ROOTS)")
        p = src.get("path") or (src.get("url", "").split("///", 1)[-1] if kind == "sqlite" else "")
        if not p:
            raise SourceRefused(f"a {kind} source needs a path")
        # realpath + prefix test: symlinks and ".." are resolved before the check
        real = os.path.realpath(p)
        allowed = [os.path.realpath(os.path.join(r, tenant)) for r in roots]
        if not any(real == a or real.startswith(a + os.sep) for a in allowed):
            raise SourceRefused(f"path must lie below {' or '.join(str(a) for a in allowed)}")
    elif src.get("url"):
        u = src["url"]
        if "@" in u.split("//", 1)[-1].split("/", 1)[0] and ":" in u.split("//", 1)[-1].split("@", 1)[0]:
            raise SourceRefused("the URL holds a password; store it in a secret and set secret_ref")
        hosts = settings.csv("allowed_hosts")
        if hosts:
            host = (urlsplit(u).hostname or "").lower()
            if not any(fnmatch.fnmatch(host, h.lower()) for h in hosts):
                raise SourceRefused(f"host {host!r} is not in the allowed hosts")
    return src


def _settings_presets() -> tuple:
    """Names of the built-in settings presets (relsynth.settings is imported only when needed)."""
    from ..settings import PRESETS
    return tuple(PRESETS)


def spec_refusals(raw: dict) -> list[str]:
    """Why a tenant's spec cannot run on the service (empty when it can).

    A spec reaches the service as text from a tenant, so it may not name files on
    the service's machine: keys starting with ``_`` are internal (``_base_dir``
    moves the folder relative paths resolve against), a settings file other than a
    built-in ``preset:NAME`` and a compute file are paths the workers would read."""
    out = [f"key {k!r} is internal and cannot be set in a spec" for k in raw if str(k).startswith("_")]
    syn = raw.get("synthesis")
    if syn is not None and not isinstance(syn, dict):
        out.append("synthesis must be a mapping")
    if not isinstance(syn, dict):
        syn = {}
    ref = syn.get("settings_file")
    if ref is not None and not (isinstance(ref, str) and ref.startswith("preset:")
                                and ref.split(":", 1)[1] in _settings_presets()):
        out.append("synthesis.settings_file: the service accepts only a built-in preset:NAME, "
                   "not a file on the service")
    comp = syn.get("compute")
    if not isinstance(comp, dict):
        comp = {}
    if comp.get("file") is not None:
        out.append("synthesis.compute.file: the service does not read compute files; "
                   "use a built-in preset with overrides")
    return out
