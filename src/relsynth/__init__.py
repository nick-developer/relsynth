"""relsynth: Bayesian synthetic data for relational databases.

    import relsynth as rs
    spec = rs.introspect("sqlite:shop.db")      # draft a specification
    model = rs.fit(spec)                         # fit the Bayesian model
    data = model.sample(scale=2.0)               # a synthetic database
    model.evaluate(data).require(0.98)           # quality gate
    data.write("csv:synthetic/")

See README.md and docs/ for the user guide, the CLI, the connectors and the
theory (propositions and proofs).
"""
from ._version import __version__

# Submodules named like API functions are loaded first: importing a submodule for
# the first time binds its name on the package, which would hide the function.
from . import doctor as _doctor_module  # noqa: F401
from . import evaluate as _evaluate_module  # noqa: F401
from . import introspect as _introspect_module  # noqa: F401
from . import verify as _verify_module  # noqa: F401
from .api import (Diagnosis, EvaluationReport, FittedModel, RunResult, Spec, SyntheticData, clone_schema, connect,
                  doctor, evaluate, export_model, fit, introspect, ping, profile, run, sample, tune, validate_spec,
                  verify)
from .engine import Synthesizer
from .errors import EngineError, QualityGateError, RelsynthError, SourceError
from .spec import SpecError, dump_spec, load_spec
from .verify import VerifyReport

__all__ = ["__version__", "Spec", "FittedModel", "SyntheticData", "EvaluationReport", "RunResult", "Diagnosis",
           "introspect", "validate_spec", "fit", "sample", "evaluate", "run", "doctor", "tune", "export_model", "verify",
           "clone_schema", "profile", "ping", "connect", "Synthesizer", "VerifyReport", "load_spec", "dump_spec",
           "RelsynthError", "SpecError", "SourceError", "EngineError", "QualityGateError"]
