# SPDX-License-Identifier: Apache-2.0
"""Builds spec-reference.md from spec-reference.src.md and the schema in relsynth/schema.py."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from relsynth.schema import reference_tables  # noqa: E402

src = (HERE / "spec-reference.src.md").read_text()
(HERE / "spec-reference.md").write_text(src.replace("<!-- KEYS -->", reference_tables()))
print("wrote spec-reference.md")
