# SPDX-License-Identifier: Apache-2.0
"""Synthetic content for binary columns (images, documents, files).

relsynth 2.1 wrote binary columns as NULL.  This module gives them content that
is safe by construction: the bytes never come from the source.  What is learned
from the source is the null share, the file type of each value (from its first
bytes) and its length:

* a value's cell is (MIME type, length class).  A length shared by at least
  ``k`` values of a type (``privacy.suppress_small_counts``) is its own class
  (a repeated "no image" placeholder of 1,673 bytes stays 1,673 bytes); every
  other length falls in its power-of-two bin [2^b, 2^(b+1)).  A MIME type seen
  fewer than ``k`` times is pooled into application/octet-stream.
* p(null) ~ Beta(nulls + 1/2, values + 1/2) and the cell shares ~ Dirichlet
  (counts + 1/2) over the observed cells, drawn once per dataset (posterior mode
  draw) or set to their means (mode mean); a bin's length is log-uniform in it.
* content (``generator``): ``placeholder`` writes a fixed, valid one-pixel image
  or one-page document of the cell's type, shipped with relsynth, padded with
  zero bytes after its end marker to the drawn length (random bytes for a type
  without a placeholder); ``random`` writes uniform random bytes of the drawn
  length; ``none`` keeps NULL (2.1).

Values are generated on a stream of their own (derived from the data stream's
state without drawing from it), so switching the generator changes the binary
columns only: every other column of the dataset is byte-identical (Prop. 1780).
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib

import numpy as np
import pandas as pd

GENERATORS = ("none", "placeholder", "random")
OCTET = "application/octet-stream"

# (magic prefix, MIME type), longest first where prefixes overlap
MAGIC = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"GIF87a", "image/gif"), (b"GIF89a", "image/gif"),
         (b"\xff\xd8\xff", "image/jpeg"), (b"%PDF-", "application/pdf"),
         (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "application/x-ole-storage"), (b"PK\x03\x04", "application/zip"),
         (b"II*\x00", "image/tiff"), (b"MM\x00*", "image/tiff"), (b"\x1f\x8b", "application/gzip"))


def sniff(b: bytes) -> str:
    """The MIME type of a value from its first bytes (octet-stream when unknown)."""
    for m, t in MAGIC:
        if b[:len(m)] == m:
            return t
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return OCTET


def _png() -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)          # 1x1, 8-bit greyscale
    idat = zlib.compress(b"\x00\xff", 9)                          # filter 0, one white pixel
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _pdf() -> bytes:
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] >>"]
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offs)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


# one white 1x1 GIF89a, and a 1x1 greyscale baseline JPEG (written once by an image library)
_GIF = bytes.fromhex("47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b")
_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300100b0c0e0c0a100e0d0e1211101318281a181616183123251d283a333d"
    "3c3933383740485c4e404457453738506d51575f626768673e4d71797064785c656763ffc0000b080001000101011100ffc4001f0000"
    "010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d010203000411"
    "05122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445"
    "464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9"
    "aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda00080101"
    "00003f00f40affd9")
#: the placeholder file of each MIME type (relsynth's own bytes, never the source's)
PLACEHOLDERS = {"image/png": _png(), "image/gif": _GIF, "image/jpeg": _JPEG, "application/pdf": _pdf()}


def byte_length(v) -> int | None:
    """Length in bytes of a stored value: bytes as they are, 0x-hex text (CSV exports) as
    the bytes it spells, other text as UTF-8; None for a null."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (bytes, bytearray, memoryview)):
        return len(v)
    s = str(v)
    if s[:2] in ("0x", "0X") and len(s) % 2 == 0:
        try:
            bytes.fromhex(s[2:])
            return (len(s) - 2) // 2
        except ValueError:
            pass
    return len(s.encode("utf-8"))


def _head(v) -> bytes:
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v[:16])
    s = str(v)
    if s[:2] in ("0x", "0X"):
        try:
            return bytes.fromhex(s[2:34])
        except ValueError:
            return b""
    return b""


class BinaryColumn:
    """Null share, MIME types and length law of one binary column (module docstring)."""

    def __init__(self, generator: str = "placeholder", mime: str | None = None, k: int = 5):
        if generator not in GENERATORS:
            raise ValueError(f"binary generator must be one of {', '.join(GENERATORS)}, not {generator!r}")
        self.generator, self.mime, self.k = generator, mime, max(int(k), 1)
        self.n_null = self.n = 0
        self.cells: list[tuple[str, str, int]] = []      # (mime, "len" | "bin", length or bin)
        self.counts = np.zeros(0)

    def fit(self, values) -> BinaryColumn:
        s = pd.Series(list(values), dtype=object)
        lens = [byte_length(v) for v in s]
        keep = [i for i, n in enumerate(lens) if n is not None]
        self.n, self.n_null = len(keep), len(s) - len(keep)
        if not keep:
            return self
        mimes = [self.mime or sniff(_head(s.iloc[i])) for i in keep]
        mc = pd.Series(mimes).value_counts()
        mimes = [m if (self.mime or mc[m] >= self.k) else OCTET for m in mimes]
        L = np.array([lens[i] for i in keep], dtype=np.int64)
        lc = pd.Series(list(zip(mimes, L))).value_counts()
        cells: dict = {}
        for m, n in zip(mimes, L):
            cell = (m, "len", int(n)) if lc[(m, n)] >= self.k else (m, "bin", int(n).bit_length())
            cells[cell] = cells.get(cell, 0) + 1
        order = sorted(cells, key=lambda c: (c[0], c[1], c[2]))
        self.cells = order
        self.counts = np.array([cells[c] for c in order], dtype=float)
        return self

    @property
    def info(self) -> str:
        if not self.n:
            return f"binary ({self.generator}): every value is null in the source"
        types = sorted({c[0] for c in self.cells})
        exact = sum(1 for c in self.cells if c[1] == "len")
        return (f"binary ({self.generator}): {', '.join(types)}; {exact} repeated length(s) kept, "
                f"{len(self.cells) - exact} power-of-two length bin(s); null share {self.n_null / (self.n + self.n_null):.3f}")

    def generate(self, n: int, rng: np.random.Generator, mode: str = "draw") -> list:
        """n values (bytes or None)."""
        if self.generator == "none" or n == 0:
            return [None] * n
        a0, b0 = self.n_null + 0.5, self.n + 0.5
        p_null = rng.beta(a0, b0) if mode == "draw" else a0 / (a0 + b0)
        if not self.n:
            return [None] * n
        alpha = self.counts + 0.5
        w = rng.dirichlet(alpha) if mode == "draw" else alpha / alpha.sum()
        is_null = rng.random(n) < p_null
        cell = rng.choice(len(self.cells), size=n, p=w)
        u = rng.random(n)
        out: list = [None] * n
        for i in np.flatnonzero(~is_null):
            m, kind, v = self.cells[cell[i]]
            if kind == "len":
                length = v
            elif v == 0:
                length = 0
            else:            # log-uniform in [2^(v-1), 2^v)
                lo = 1 << (v - 1)
                length = min(int(lo * 2.0 ** u[i]), (lo << 1) - 1)
            out[i] = self._content(m, length, rng)
        return out

    def _content(self, mime: str, length: int, rng: np.random.Generator) -> bytes:
        ph = PLACEHOLDERS.get(mime) if self.generator == "placeholder" else None
        if ph is None:
            return rng.bytes(length)
        return ph + bytes(max(length - len(ph), 0))

    def to_dict(self) -> dict:
        return {"generator": self.generator, "mime": self.mime, "k": self.k, "nulls": self.n_null, "values": self.n,
                "cells": [{"mime": m, "kind": k, "value": v, "count": int(c)}
                          for (m, k, v), c in zip(self.cells, self.counts)]}


def stream(rng: np.random.Generator, table: str, column: str) -> np.random.Generator:
    """A generator for one binary column, keyed by the data stream's current state
    (read, not advanced) and the column: the data stream draws exactly what it
    draws without binary content, and a partition or dataset gets its own values."""
    from .runtime.streams import stream as philox
    state = json.dumps(rng.bit_generator.state, sort_keys=True, default=lambda a: np.asarray(a).tolist())
    return philox("binary", table, column, hashlib.sha256(state.encode()).hexdigest())


def column_options(cs: dict, synthesis: dict) -> dict:
    """Effective {generator, mime} of a binary column: its own `binary` block over
    synthesis.binary."""
    base = dict((synthesis or {}).get("binary") or {})
    own = cs.get("binary") or {}
    return {"generator": own.get("generator", base.get("generator", "none")), "mime": own.get("mime")}
