# SPDX-License-Identifier: Apache-2.0
"""The conditional generator contract (generator plug-in API, contract 1.0).

relsynth's engine owns the relational skeleton of a dataset: which rows exist,
their keys, the links between them, the latent segment of every row, the hard
rules, uniqueness, scaling and the random streams.  A *generator* owns one
thing only: the joint law of a table's own attribute columns given that
skeleton.  It is fitted once per table and then asked, for every synthetic row,
for attribute values given the row's conditioning context:

* the row's latent **group** (an entity row's segment, a fact row's segment
  block, see Prop. 700), with the coarser group the engine falls back to;
* optionally the **parent attributes** joined onto the row (the synthetic
  parent's values, columns named in ``TableData.aux``).

Everything a generator returns is post-processed by the engine (computed
columns, derive rules, rejection on check rules, unique attribute keys), so a
generator never sees or breaks keys and links.  What it is responsible for is
stated in docs/generators.md and Props. 700-705.

A generator is a class with the methods of :class:`TableGenerator` and a
:class:`Capabilities` record.  It is discovered through the ``relsynth.generators``
entry-point group (see :mod:`relsynth.generators.registry`) and chosen per
table in the spec (``tables.<t>.generator``).
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import pickle  # nosec B403 - see TableGenerator.load: only trusted model files are loaded
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

CONTRACT_VERSION = "1.0"
# column kinds a generator can declare (the modelled column types of the spec)
COLUMN_KINDS = ("integer", "float", "categorical", "boolean", "date", "datetime")
CONDITIONING = ("group", "parent")
DP_SUPPORT = ("none", "input", "output")
GPU_NEED = ("never", "optional", "required")


class GeneratorError(ValueError):
    """A generator cannot be found, is incompatible, or failed (CLI exit code 2)."""
    exit_code = 2


@dataclass(frozen=True)
class Capabilities:
    """What a generator can do.  The registry checks a table against it before
    fitting (registry.check), so an incompatible choice fails early with a reason.

    modalities: entity shapes of the IR the generator models (``record`` for a
        table row; ``document``, ``node``, ``event`` for later releases).
    column_kinds: modelled column types it accepts.
    conditioning: ``group`` (latent segment of the row; required) and
        ``parent`` (values of the row's synthetic parent attributes).
    dp: ``none``; ``input`` (it can fit from privatised sufficient statistics);
        ``output`` (it spends a privacy budget itself and reports it in
        privacy_cost).  T07 owns the semantics.
    gpu: ``never``, ``optional`` or ``required`` (resolved by relsynth.compute).
    pushdown: fits from in-database sufficient statistics plus a bounded
        sample (the engine's pushdown hands it the sample rows).
    posterior: draws its parameters from a posterior (posterior_draw changes
        them), so ``synthesis.posterior.datasets`` gives posterior predictive
        datasets (Prop. 703).  False: every dataset uses the same fitted model.
    log_prob: implements log_prob.
    ir_versions: IR versions it accepts, as a version specifier.
    """
    modalities: tuple = ("record",)
    column_kinds: tuple = COLUMN_KINDS
    conditioning: tuple = ("group",)
    dp: str = "none"
    gpu: str = "never"
    pushdown: bool = True
    posterior: bool = True
    log_prob: bool = False
    ir_versions: str = ">=1.0,<3"

    def __post_init__(self):
        bad = [k for k in self.column_kinds if k not in COLUMN_KINDS]
        if bad:
            raise GeneratorError(f"capabilities: unknown column kinds {bad} (allowed: {', '.join(COLUMN_KINDS)})")
        if "group" not in self.conditioning or any(c not in CONDITIONING for c in self.conditioning):
            raise GeneratorError(f"capabilities: conditioning must include 'group' and only use {CONDITIONING}")
        if self.dp not in DP_SUPPORT:
            raise GeneratorError(f"capabilities: dp must be one of {DP_SUPPORT}")
        if self.gpu not in GPU_NEED:
            raise GeneratorError(f"capabilities: gpu must be one of {GPU_NEED}")

    def to_dict(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(self).items()}


@dataclass
class FitContext:
    """Everything except the table's own data that a fit may use.

    groups: int label per row (-1: the row is not used, e.g. a fact row with an
        orphan key); None for a single group.
    parent: {group: coarser group} - the engine's back-off for groups with few
        rows (a fact block's dimension segment).  Groups the sampler meets that
        were not fitted are given with their coarse label.
    per_group: fit a dependence structure per group (entities) or one pooled
        structure (facts).
    rng: the engine's seeded generator.  A generator must draw every random
        number from it (or from generators derived from it), never from global
        state, so that a seed gives the same dataset (Prop. 701).
    config: the spec's synthesis settings (read-only).
    options: the table's ``generator.options`` from the spec.
    compute: the resolved compute plan (relsynth.compute.ComputePlan).
    privacy_budget: a budget handed down by the DP engine (T07); None today.
    """
    groups: np.ndarray | None
    parent: dict
    per_group: bool
    rng: np.random.Generator
    config: dict
    options: dict = field(default_factory=dict)
    compute: Any = None
    privacy_budget: Any = None
    table: str = ""


@dataclass
class Conditioning:
    """The context of the rows to sample: one group label per row and, when the
    generator declares ``parent`` conditioning and the table has parent
    attributes, a frame of those values (columns TableData.aux, one row per row)."""
    groups: np.ndarray
    given: pd.DataFrame | None = None

    def __len__(self):
        return len(self.groups)


@dataclass(frozen=True)
class PrivacyCost:
    """Privacy spent by a fit.  ``dp=False`` means the generator gives no formal
    guarantee (its output is protected by the engine's measures and measured by
    the attack suite only); with ``dp=True`` epsilon and delta are what it spent
    of the budget it was given.  Accounting across tables belongs to T07."""
    dp: bool = False
    epsilon: float | None = None
    delta: float | None = None
    mechanism: str | None = None
    note: str = ""


class TableGenerator:
    """Base class of a conditional generator of one table's attribute columns.

    A subclass sets ``name``, ``version`` (the version of its model format and
    algorithm, used for pinning and for refusing to load an older model) and
    ``capabilities``, and implements fit, sample and, if it has a posterior,
    posterior_draw.  Instances are created per table by the registry.

    ``data`` in fit is the engine's encoding of the table (relsynth.table_model
    .TableModel): ``codecs`` (one per modelled column, with encode/decode),
    ``X`` (encoded columns: int codes with -1 for null, or floats with NaN),
    ``cols``/``out_cols``/``aux`` (all, own and parent-attribute columns),
    ``n_rows`` and ``decode(values, rng)`` (encoded matrix -> output frame).
    Generators that need the raw values decode X with the codecs.
    """
    name: str = ""
    version: str = "0"
    capabilities: Capabilities = Capabilities()
    description: str = ""
    fitted: bool = False

    def __init__(self, options: dict | None = None):
        self.options = dict(options or {})

    # ---- required
    def fit(self, data, context: FitContext, privacy_budget=None) -> TableGenerator:
        raise NotImplementedError

    def sample(self, n: int, conditioning: Conditioning, rng: np.random.Generator) -> pd.DataFrame:
        """n rows of the table's own columns (data.out_cols, in that order) for
        the rows described by conditioning (len(conditioning) == n)."""
        raise NotImplementedError

    # ---- optional
    def posterior_draw(self, rng: np.random.Generator, mode: str = "draw") -> TableGenerator:
        """Replace the current parameters by one posterior draw (mode "draw") or
        by posterior means ("mean").  The engine calls it once per dataset, so
        datasets 1..m are m posterior predictive draws.  Generators without a
        posterior keep their parameters (capabilities.posterior = False)."""
        return self

    def posterior_draws(self, m: int, rng: np.random.Generator):
        """Iterator over m posterior draws (the generator itself after each)."""
        for _ in range(m):
            yield self.posterior_draw(rng)

    def log_prob(self, frame: pd.DataFrame, conditioning: Conditioning) -> np.ndarray:
        raise GeneratorError(f"generator {self.name} does not implement log_prob")

    def uncertainty(self) -> dict:
        """Summary of the posterior uncertainty and convergence of the fit
        (shown in the model summary)."""
        return {}

    def privacy_cost(self) -> PrivacyCost:
        return PrivacyCost(dp=False, note=f"{self.name}: no formal privacy guarantee")

    def fitted_groups(self) -> set:
        """Groups with their own parameters (the others back off to their coarse group)."""
        return set()

    def summary(self) -> dict:
        return {"generator": self.name, "version": self.version}

    # ---- persistence
    def state(self) -> dict:
        """The fitted state to persist; the default is the instance dictionary."""
        return dict(self.__dict__)

    def set_state(self, state: dict):
        self.__dict__.update(state)

    def serialize(self) -> bytes:
        """A model file: a JSON header (generator, model version, contract
        version, SHA-256 of the payload) and the pickled state."""
        payload = pickle.dumps(self.state(), protocol=pickle.HIGHEST_PROTOCOL)
        head = {"format": "relsynth-generator", "contract": CONTRACT_VERSION, "generator": self.name,
                "version": self.version, "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "sha256": hashlib.sha256(payload).hexdigest()}
        return json.dumps(head).encode() + b"\n" + base64.b64encode(payload)

    @classmethod
    def load(cls, blob: bytes, *, trusted: bool = False) -> TableGenerator:
        """Load a model file written by serialize.  The payload is a pickle, which
        can run code when it is read: only files this installation wrote (or you
        otherwise trust) may be loaded, and ``trusted=True`` says so.  The header
        must name this generator and its model version."""
        if not trusted:
            raise GeneratorError("generator model files are pickles: load only files you trust, with trusted=True")
        head_line, _, body = blob.partition(b"\n")
        head = json.loads(head_line)
        if head.get("format") != "relsynth-generator":
            raise GeneratorError("not a relsynth generator model file")
        if head.get("generator") != cls.name or head.get("version") != cls.version:
            raise GeneratorError(f"model file is {head.get('generator')} {head.get('version')}, "
                                 f"not {cls.name} {cls.version}")
        if head.get("contract", "").split(".")[0] != CONTRACT_VERSION.split(".")[0]:
            raise GeneratorError(f"model file uses generator contract {head.get('contract')}, "
                                 f"this relsynth has {CONTRACT_VERSION}")
        payload = base64.b64decode(body)
        if hashlib.sha256(payload).hexdigest() != head.get("sha256"):
            raise GeneratorError("model file is corrupt (SHA-256 mismatch)")
        obj = cls.__new__(cls)
        obj.set_state(pickle.loads(payload))  # noqa: S301 # nosec B301 - trusted=True required above
        return obj
