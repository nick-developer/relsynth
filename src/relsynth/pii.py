# SPDX-License-Identifier: Apache-2.0
"""Format-preserving generators for direct identifiers. Nothing is learned from
the source values except formats (digit/letter shapes); values never repeat
source values except by chance."""
from __future__ import annotations

import re
import uuid as _uuid


FIRST = ("Alex Maria Nikos Elena John Sofia Omar Lena Ivan Chen Aisha Lucas Emma Noah Mia "
         "Leo Zoe Ravi Anna Tom Sara Ben Nina Paul Ella Max Ines Yuki Ali Kate").split()
LAST = ("Smith Papadopoulos Garcia Muller Rossi Kim Silva Novak Cohen Ahmed Dubois Jensen "
        "Nowak Tanaka Ivanova Brown Costa Murphy Weber Lopez Fischer Moreau Khan Ortiz").split()


def _shape(v):
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", v))


class PiiGenerator:
    def __init__(self, cfg, source_values, label="ID"):
        self.kind = cfg.get("generator")
        self.label = label
        self.pattern = cfg.get("pattern")
        vals = [str(v) for v in source_values if v is not None and str(v) != "nan"]
        self.null_rate = 1 - len(vals) / max(len(source_values), 1)
        shapes = {}
        for v in vals[:5000]:
            s = _shape(v)
            shapes[s] = shapes.get(s, 0) + 1
        # only formats shared by >= 5 rows are kept
        self.shapes = [s for s, c in shapes.items() if c >= 5]
        # names stored as initials ("A", "J.") are generated as initials
        short = [v for v in vals if len(v.strip()) <= 2]
        self.initials = len(vals) > 0 and len(short) >= 0.5 * len(vals)
        self.initial_dot = self.initials and sum(v.strip().endswith(".") for v in short) >= 0.5 * len(short)

    def generate(self, n, rng):
        out = []
        for _i in range(n):
            if rng.random() < self.null_rate:
                out.append(None)
                continue
            if self.kind == "email":
                f, l = rng.choice(FIRST).lower(), rng.choice(LAST).lower()
                out.append(f"{f}.{l}{rng.integers(1000)}@example.com")
            elif self.kind == "person_name":
                lab = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", self.label).lower()
                if self.initials:
                    out.append(chr(65 + int(rng.integers(26))) + ("." if self.initial_dot else ""))
                elif re.search(r"(first|given|fore|middle)_?name", lab):
                    out.append(str(rng.choice(FIRST)))
                elif re.search(r"(last|family|sur)_?name", lab):
                    out.append(str(rng.choice(LAST)))
                else:
                    out.append(f"{rng.choice(FIRST)} {rng.choice(LAST)}")
            elif self.kind == "uuid":
                out.append(str(_uuid.UUID(int=int(rng.integers(2**63)) << 64 | int(rng.integers(2**63)))))
            elif not self.pattern and not self.shapes:
                # no format is shared by 5 rows (free-form names): a surrogate label
                out.append(f"{self.label} {rng.integers(10**5):05d}")
            else:
                shape = self.pattern or self.shapes[rng.integers(len(self.shapes))]
                out.append("".join(str(rng.integers(10)) if ch in "9#" else
                                   chr(65 + rng.integers(26)) if ch == "A" else ch for ch in shape))
        return out
