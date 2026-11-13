# SPDX-License-Identifier: Apache-2.0
"""Binary columns (G4, relsynth.binary): length law, MIME placeholders, own stream.
The statistical check is C1780 in tests/science/test_binary_checks.py."""
from __future__ import annotations

import copy
import io
import sys
import zlib

import numpy as np
import pandas as pd
import pytest

import relsynth as rs
from relsynth.binary import OCTET, PLACEHOLDERS, BinaryColumn, byte_length, column_options, sniff, stream
from relsynth.data import write_tables
from relsynth.settings import registry
from relsynth.spec import DEFAULT_SYNTHESIS, SpecError, load_spec, validate

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from conftest import scenario  # noqa: E402


def _blobs(n=400, seed=0):
    """A photo column as product catalogues have it: one shared "no image" GIF,
    JPEGs of many lengths, a few nulls."""
    rng = np.random.default_rng(seed)
    noimg = b"GIF89a" + rng.bytes(1667)
    out = []
    for i in range(n):
        u = rng.random()
        if u < 0.08:
            out.append(None)
        elif u < 0.6:
            out.append(noimg)
        else:
            out.append(b"\xff\xd8\xff\xe0" + rng.bytes(int(rng.lognormal(9, 0.6))))
    return out


def test_settings_and_schema():
    assert DEFAULT_SYNTHESIS["binary"]["generator"] == "none"           # opt-in: 2.1 output by default
    assert registry()["binary.generator"].choices == ("none", "placeholder", "random")
    assert column_options({"binary": {"generator": "random"}}, {"binary": {"generator": "placeholder"}}) == \
        {"generator": "random", "mime": None}
    assert column_options({}, {"binary": {"generator": "placeholder"}})["generator"] == "placeholder"


def test_spec_validation_of_binary_options():
    t, s, _ = scenario("S2_star")
    s = copy.deepcopy(s)
    s["tables"]["product"]["columns"]["photo"] = {"type": "binary", "binary": {"generator": "random", "mime": "image/png"}}
    assert not validate(load_spec(copy.deepcopy(s)))
    s["tables"]["product"]["columns"]["photo"] = {"type": "binary", "binary": {"generator": "copy"}}
    with pytest.raises(SpecError, match="binary.generator"):
        load_spec(copy.deepcopy(s))
    s["tables"]["product"]["columns"]["photo"] = "binary"
    s["tables"]["product"]["columns"]["price"] = {"type": "float", "binary": {"generator": "random"}}
    with pytest.raises(SpecError, match="binary columns only"):
        load_spec(copy.deepcopy(s))


def test_sniff_and_lengths():
    for mime, ph in PLACEHOLDERS.items():
        assert sniff(ph) == mime
    assert sniff(b"\x00\x01") == OCTET
    assert byte_length(b"abc") == 3 and byte_length("0x0aff") == 2 and byte_length(None) is None
    assert byte_length(float("nan")) is None and byte_length("é") == 2


def test_placeholders_are_valid_files():
    png = PLACEHOLDERS["image/png"]
    assert png.endswith(b"IEND\xaeB`\x82") and zlib.decompress(png[png.index(b"IDAT") + 4:-16]) == b"\x00\xff"
    pdf = PLACEHOLDERS["application/pdf"]
    xref = int(pdf.rsplit(b"startxref\n", 1)[1].split(b"\n")[0])
    assert pdf[xref:].startswith(b"xref") and pdf.rstrip().endswith(b"%%EOF")
    pil = pytest.importorskip("PIL.Image")
    for mime in ("image/png", "image/gif", "image/jpeg"):
        padded = PLACEHOLDERS[mime] + bytes(500)            # trailing zero bytes still decode
        im = pil.open(io.BytesIO(padded))
        im.load()
        assert im.size == (1, 1)


def test_length_law_keeps_repeated_lengths_and_bins_the_rest():
    vals = _blobs()
    m = BinaryColumn("placeholder", k=5).fit(vals)
    exact = [c for c in m.cells if c[1] == "len"]
    assert exact == [("image/gif", "len", 1673)]
    assert all(c[0] == "image/jpeg" and c[1] == "bin" for c in m.cells if c not in exact)
    assert m.n_null == sum(v is None for v in vals)
    rng = np.random.default_rng(1)
    out = m.generate(20000, rng, "mean")
    got = pd.Series([len(v) if v is not None else -1 for v in out])
    src = pd.Series([len(v) if v is not None else -1 for v in vals])
    assert abs((got == -1).mean() - (src == -1).mean()) < 0.03
    assert abs((got == 1673).mean() - (src == 1673).mean()) < 0.04
    # the JPEG lengths follow the source's power-of-two bins
    gb = got[(got > 0) & (got != 1673)].map(int.bit_length).value_counts(normalize=True)
    sb = src[(src > 0) & (src != 1673)].map(int.bit_length).value_counts(normalize=True)
    assert 0.5 * (gb.reindex(sb.index.union(gb.index), fill_value=0)
                  - sb.reindex(sb.index.union(gb.index), fill_value=0)).abs().sum() < 0.05
    # content: placeholders of the value's type, never a source value
    srcset = {bytes(v) for v in vals if v is not None}
    assert not srcset & {v for v in out if v is not None}
    gifs = [v for v in out if v is not None and len(v) == 1673]
    assert gifs and all(v.startswith(PLACEHOLDERS["image/gif"]) for v in gifs)


def test_rare_lengths_and_types_are_not_released_exactly():
    vals = [b"%PDF-" + bytes(100 + i) for i in range(4)] + [b"\xff\xd8\xff" + bytes(5000)] * 6
    m = BinaryColumn("placeholder", k=5).fit(vals)
    assert ("application/octet-stream", "bin", 7) in m.cells          # 4 PDFs: pooled type, binned length
    assert ("image/jpeg", "len", 5003) in m.cells
    m1 = BinaryColumn("placeholder", k=1).fit(vals)
    assert {c[2] for c in m1.cells if c[1] == "len"} >= {105, 106, 5003}


def test_random_generator_and_declared_mime():
    vals = _blobs(200)
    out = BinaryColumn("random", k=5).fit(vals).generate(500, np.random.default_rng(0))
    assert not any(v.startswith(PLACEHOLDERS["image/gif"]) for v in out if v is not None)
    png = BinaryColumn("placeholder", mime="image/png", k=5).fit(vals).generate(300, np.random.default_rng(0))
    assert all(v.startswith(PLACEHOLDERS["image/png"]) for v in png if v is not None)
    assert BinaryColumn("none").fit(vals).generate(5, np.random.default_rng(0)) == [None] * 5
    assert BinaryColumn("random").fit([None, None]).generate(3, np.random.default_rng(0)) == [None] * 3
    with pytest.raises(ValueError):
        BinaryColumn("copy")


def test_stream_does_not_advance_the_data_stream():
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    s1, s2 = stream(a, "t", "c"), stream(a, "t", "c")
    assert s1.random() == s2.random() and a.random() == b.random()
    assert stream(a, "t", "c").random() != stream(a, "t", "d").random()


@pytest.fixture(scope="module")
def shop_db(tmp_path_factory):
    d = tmp_path_factory.mktemp("bin")
    t, s, _ = scenario("S2_star")
    t, s = copy.deepcopy(t), copy.deepcopy(s)
    t["product"]["photo"] = _blobs(len(t["product"]), seed=3)
    s["tables"]["product"]["columns"]["photo"] = "binary"
    s["synthesis"].update({"gibbs": {"iterations": 30, "final_iterations": 10}})
    db = d / "shop.db"
    write_tables(load_spec(copy.deepcopy(s)), t, {"kind": "sqlite", "path": str(db), "structure": "infer"})
    s["source"] = {"kind": "sqlite", "path": str(db)}
    return s, t


def _fit_sample(spec, gen):
    s = copy.deepcopy(spec)
    s["synthesis"]["binary"] = {"generator": gen}
    return dict(rs.fit(s, seed=4).sample())


def test_pipeline_fills_binary_columns_and_changes_nothing_else(shop_db):
    """Prop. 1780: the binary stream leaves every other column byte-identical."""
    spec, real = shop_db
    none = _fit_sample(spec, "none")
    ph = _fit_sample(spec, "placeholder")
    assert none["product"]["photo"].isna().all()
    vals = ph["product"]["photo"]
    assert vals.notna().mean() > 0.8
    for t in none:
        cols = [c for c in none[t].columns if c != "photo"]
        pd.testing.assert_frame_equal(none[t][cols], ph[t][cols])
    src = {bytes(v) for v in real["product"]["photo"] if v is not None}
    assert not src & {bytes(v) for v in vals.dropna()}
    assert (vals.dropna().map(len) == 1673).mean() > 0.35


def test_binary_round_trips_through_sqlite_and_csv_targets(shop_db, tmp_path):
    spec, _ = shop_db
    s = copy.deepcopy(spec)
    s["synthesis"]["binary"] = {"generator": "random"}
    m = rs.fit(s, seed=4)
    data = m.sample()
    data.write({"kind": "sqlite", "path": str(tmp_path / "o.db")})
    data.write({"kind": "csv", "path": str(tmp_path / "csv")})
    back = pd.read_csv(tmp_path / "csv" / "product.csv")["photo"].dropna()
    assert back.str.startswith("0x").all()
    import sqlite3
    with sqlite3.connect(tmp_path / "o.db") as con:
        n = con.execute("select count(*) from product where typeof(photo) = 'blob'").fetchone()[0]
    assert n == int(dict(data)["product"]["photo"].notna().sum())
