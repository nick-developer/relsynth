# SPDX-License-Identifier: Apache-2.0
"""Exceptions raised by relsynth, and the exit code the CLI maps each one to.

    RelsynthError            base class of everything relsynth raises on purpose
      SpecError              the specification is invalid (also a ValueError)
      SourceError            a source or target cannot be reached, read or written
      EngineError            fitting or sampling failed
      QualityGateError       the synthetic data failed the quality gate
"""
from __future__ import annotations

# CLI exit codes (stable: CI pipelines branch on them)
EXIT_OK = 0
EXIT_GATE = 1        # ran, but the quality gate failed (scorecard below --fail-under)
EXIT_SPEC = 2        # invalid spec, arguments or usage
EXIT_IO = 3          # connection, read or write failure
EXIT_ENGINE = 4      # fitting, sampling or an unexpected internal error


class RelsynthError(Exception):
    """Base class of the errors relsynth raises."""
    exit_code = EXIT_ENGINE


class SourceError(RelsynthError, OSError):
    """A source or target cannot be reached, read or written."""
    exit_code = EXIT_IO


class EngineError(RelsynthError, RuntimeError):
    """Fitting or sampling failed."""
    exit_code = EXIT_ENGINE


class QualityGateError(RelsynthError):
    """The synthetic data did not pass the quality gate."""
    exit_code = EXIT_GATE

    def __init__(self, message: str, report=None):
        super().__init__(message)
        self.report = report


def exit_code_for(exc: BaseException) -> int:
    """The CLI exit code of an exception."""
    from .spec import SpecError
    if isinstance(exc, RelsynthError):
        return exc.exit_code
    if isinstance(exc, (SpecError, ValueError, KeyError, TypeError)) and _is_spec(exc):
        return EXIT_SPEC
    if isinstance(exc, (OSError, ConnectionError, TimeoutError)):
        return EXIT_IO
    try:
        import sqlalchemy.exc as sx
        if isinstance(exc, (sx.OperationalError, sx.InterfaceError, sx.ProgrammingError, sx.NoSuchTableError,
                            sx.ArgumentError, sx.NoSuchModuleError)):
            return EXIT_IO
    except ImportError:          # pragma: no cover
        pass
    return EXIT_ENGINE


def _is_spec(exc) -> bool:
    from .spec import SpecError
    return isinstance(exc, SpecError)
