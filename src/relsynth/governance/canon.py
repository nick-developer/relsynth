# SPDX-License-Identifier: Apache-2.0
"""Canonical JSON (RFC 8785, JCS) and the digests of governance artefacts.

Every signed or hashed object (manifests, ledger entries, checkpoints,
approvals) is serialised with `canonical()` first, so the same object always
gives the same bytes on every platform: object keys sorted by their UTF-16
code units, no insignificant white space, strings with the minimal JSON
escapes, numbers in the shortest form ECMAScript prints them (RFC 8785
sec. 3.2.2.3).  Governance artefacts write privacy figures as decimal
strings, so floats only appear in policy documents; they are still
canonicalised exactly.
"""
from __future__ import annotations

import hashlib
import math

__all__ = ["canonical", "sha256", "digest", "es_number"]


def es_number(x) -> str:
    """A number as ECMAScript's Number.prototype.toString prints it (RFC 8785)."""
    if isinstance(x, bool):
        raise TypeError("booleans are not numbers")
    if isinstance(x, int):
        if abs(x) > 2 ** 53:
            raise ValueError(f"integer {x} is outside the range JSON numbers hold exactly (I-JSON, RFC 7493)")
        return str(x)
    if not math.isfinite(x):
        raise ValueError("NaN and infinity are not JSON numbers")
    if x == 0:
        return "0"
    r = repr(float(x))                     # shortest round-trip digits, as ECMAScript requires
    sign = "-" if r.startswith("-") else ""
    r = r.lstrip("-")
    if "e" in r:
        mant, exp = r.split("e")
        e = int(exp)
    else:
        mant, e = r, 0
    if "." in mant:
        ip, fp = mant.split(".")
    else:
        ip, fp = mant, ""
    fp = fp.rstrip("0")
    digits = (ip + fp).lstrip("0")
    # position of the decimal point relative to the first significant digit
    lead_zeros = len(ip + fp) - len((ip + fp).lstrip("0"))
    n = len(ip) - lead_zeros + e           # value = 0.d1d2...dk * 10^n
    digits = digits.rstrip("0") or "0"
    k = len(digits)
    if k <= n <= 21:
        out = digits + "0" * (n - k)
    elif 0 < n <= 21:
        out = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        ex = n - 1
        es = ("+" if ex >= 0 else "-") + str(abs(ex))
        out = digits[0] + ("." + digits[1:] if k > 1 else "") + "e" + es
    return sign + out


_ESC = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _string(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch in _ESC:
            out.append(_ESC[ch])
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _utf16_key(k: str):
    return k.encode("utf-16-be")


def _emit(o, out: list) -> None:
    if o is None:
        out.append("null")
    elif o is True:
        out.append("true")
    elif o is False:
        out.append("false")
    elif isinstance(o, (int, float)):
        out.append(es_number(o))
    elif isinstance(o, str):
        out.append(_string(o))
    elif isinstance(o, (list, tuple)):
        out.append("[")
        for i, v in enumerate(o):
            if i:
                out.append(",")
            _emit(v, out)
        out.append("]")
    elif isinstance(o, dict):
        for k in o:
            if not isinstance(k, str):
                raise TypeError(f"object keys must be strings, got {k!r}")
        out.append("{")
        for i, k in enumerate(sorted(o, key=_utf16_key)):
            if i:
                out.append(",")
            out.append(_string(k))
            out.append(":")
            _emit(o[k], out)
        out.append("}")
    else:
        raise TypeError(f"{type(o).__name__} is not JSON")


def canonical(obj) -> bytes:
    """The RFC 8785 canonical UTF-8 bytes of a JSON value."""
    out: list = []
    _emit(obj, out)
    return "".join(out).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(obj) -> str:
    """SHA-256 of the canonical form of a JSON value (bytes are hashed as they are)."""
    return sha256(bytes(obj) if isinstance(obj, (bytes, bytearray)) else canonical(obj))
