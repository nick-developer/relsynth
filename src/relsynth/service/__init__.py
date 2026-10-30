# SPDX-License-Identifier: Apache-2.0
"""relsynth as a service: REST API, job workers, identity, tenancy and audit.

Install with ``pip install "relsynth[service]"`` and start with
``relsynth-service serve`` (API and UI) and ``relsynth-service worker`` (jobs).
See docs/service/README.md.
"""
from __future__ import annotations

__all__ = ["create_app", "Settings", "Store", "Worker"]


def __getattr__(name: str):
    if name == "create_app":
        from .app import create_app
        return create_app
    if name == "Settings":
        from .config import Settings
        return Settings
    if name == "Store":
        from .store import Store
        return Store
    if name == "Worker":
        from .worker import Worker
        return Worker
    raise AttributeError(name)
