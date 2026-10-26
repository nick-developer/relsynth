# SPDX-License-Identifier: Apache-2.0
"""The golden regression databases: where they come from, how they are built,
and the spec and domain checks each one is run with.

Public datasets are downloaded once into the cache (RELSYNTH_BENCH_CACHE,
default bench/.cache) and checked against a pinned SHA-256; nothing is ever
committed (docs/datasets.md).  bi/COREP is the project owner's private data: it
is read from RELSYNTH_BENCH_BI_DIR (data/*.csv, spec.yaml and checks.py) and
skipped when that is not set.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("RELSYNTH_BENCH_CACHE", HERE / ".cache"))
CONTOSO_GEN = "https://raw.githubusercontent.com/sql-bi/Contoso-Data-Generator-V2/eaeb57a9eaa6ad0cdab4fb527552102685434ee0/scripts/sql/"

# name -> (url, sha256); checked 2026-10-26 against the copies used in layer 2
DOWNLOADS = {
    "Chinook_Sqlite.sqlite": ("https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite",
                              "bdf635be69850bd3be09c9a2dbeef7ddfb80036bd3ef3381383cd03b61e4a61a"),
    "AdventureWorksDW-data-warehouse-install-script.zip": (
        "https://github.com/microsoft/sql-server-samples/releases/download/adventureworks/"
        "AdventureWorksDW-data-warehouse-install-script.zip",
        "73c27309d17cd30bf5351665401106abf649d9f9a9ecb0c770682f6be965aba8"),
    "parquet-100k.7z": ("https://github.com/sql-bi/Contoso-Data-Generator-V2-Data/releases/download/ready-to-use-data/"
                        "parquet-100k.7z", "ded28dd7d3de15aee2bcaa1c0153d73929ddb7c1a25d0692f15283e171f63af5"),
    "CreateTablesCommon.sql": (CONTOSO_GEN + "CreateTablesCommon.sql",
                               "46e8d39530f3d984c0fb13a5f3c398cf45f94c90ead462ca32c45c0153b224f3"),
    "CreateTablesSales.sql": (CONTOSO_GEN + "CreateTablesSales.sql",
                              "dc344238e0dd5ec18a05f95b134b2102db6f1ac47a5ee63f16f67a2472032e52"),
    "CreateTablesOrders.sql": (CONTOSO_GEN + "CreateTablesOrders.sql",
                               "d82ef30a2b8faba85ee4510fc1cffb08b1b50c6f7d9f6e1cde5080bca4e425d8"),
}


def say(msg: str) -> None:
    print(f"[bench] {msg}", file=sys.stderr, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str) -> Path:
    """The cached file, downloaded and verified on first use."""
    url, digest = DOWNLOADS[name]
    path = CACHE / "downloads" / name
    if path.exists() and sha256(path) == digest:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    say(f"downloading {url}")
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:  # noqa: S310 (pinned https URLs)
        shutil.copyfileobj(r, f)
    got = sha256(tmp)
    if got != digest:
        tmp.unlink()
        raise RuntimeError(f"{name}: SHA-256 {got} does not match the pinned {digest}")
    tmp.replace(path)
    return path


def _builder(script: str, *args: Path) -> None:
    subprocess.run([sys.executable, str(HERE / "builders" / script), *map(str, args)], check=True,  # noqa: S603
                   stdout=subprocess.DEVNULL)


def build_chinook() -> Path:
    return download("Chinook_Sqlite.sqlite")


def build_adventureworks() -> Path:
    out = CACHE / "adventureworks" / "AdventureWorksDW.db"
    if out.exists():
        return out
    src = CACHE / "adventureworks" / "scripts"
    if not (src / "instawdbdw.sql").exists():
        with zipfile.ZipFile(download("AdventureWorksDW-data-warehouse-install-script.zip")) as z:
            z.extractall(src)
        inner = [p.parent for p in src.rglob("instawdbdw.sql")]
        if inner and inner[0] != src:
            for p in inner[0].iterdir():
                p.replace(src / p.name)
    say("building AdventureWorks DW as SQLite")
    tmp = out.with_suffix(".part")
    _builder("adventureworks_sqlite.py", src, tmp)
    tmp.replace(out)
    return out


def contoso_parquet() -> Path:
    out = CACHE / "contoso" / "parquet"
    if out.exists() and any(out.glob("*.parquet")):
        return out
    import py7zr                                      # bench extra
    tmp = CACHE / "contoso" / "unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with py7zr.SevenZipFile(download("parquet-100k.7z")) as z:
        z.extractall(tmp)
    files = sorted(tmp.rglob("*.parquet"))
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        f.replace(out / f.name)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def build_contoso_sqlite() -> Path:
    out = CACHE / "contoso" / "contoso.db"
    if out.exists():
        return out
    pq = contoso_parquet()
    ddl = CACHE / "contoso" / "ddl"
    ddl.mkdir(parents=True, exist_ok=True)
    for f in ("CreateTablesCommon.sql", "CreateTablesSales.sql", "CreateTablesOrders.sql"):
        shutil.copy(download(f), ddl / f)
    say("building Contoso as SQLite")
    tmp = out.with_suffix(".part")
    _builder("contoso_sqlite.py", ddl, pq, tmp)
    tmp.replace(out)
    return out


def sqlite_tables(path: Path) -> dict:
    import pandas as pd
    con = sqlite3.connect(path)
    try:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        return {n: pd.read_sql_query(f'SELECT * FROM "{n}"', con) for n in names}  # noqa: S608 (names from the catalog)
    finally:
        con.close()


@dataclass
class Dataset:
    name: str
    location: callable              # -> relsynth source location string
    spec: callable                  # -> spec path
    checks: callable | None = None  # (real_location, synthetic_dir) -> {check: (real, synthetic)}
    public: bool = True
    note: str = ""
    tags: list = field(default_factory=list)


def _spec(name):
    return lambda: HERE / "specs" / f"{name}.yaml"


def _aw_checks(location: str, syn_dir: Path) -> dict:
    from .checks import adventureworks as m
    real = sqlite_tables(Path(location.split(":", 1)[1]))
    rs, ss = m.checks({t: real[t] for t in m.TABLES if t in real}), m.checks(m.load_synth(syn_dir))
    return {k: (rs.get(k), ss.get(k)) for k in dict.fromkeys([*rs, *ss])}


def _decimals_to_float(df):
    """Parquet DECIMAL columns arrive as Python Decimal objects; the checks compute in float."""
    import decimal
    for c in df.columns:
        if df[c].dtype == object:
            v = df[c].dropna()
            if len(v) and isinstance(v.iloc[0], decimal.Decimal):
                df[c] = df[c].astype(float)
    return df


def _contoso_checks(location: str, syn_dir: Path) -> dict:
    from .checks import contoso as m
    kind, path = location.split(":", 1)
    if kind == "sqlite":
        real = {t.lower(): d for t, d in sqlite_tables(Path(path)).items()}
    else:
        import pandas as pd
        real = {p.stem.lower(): pd.read_parquet(p) for p in Path(path).glob("*.parquet")}
    real = {t: _decimals_to_float(d.rename(columns=str.lower)) for t, d in real.items() if t in m.TABLES}
    syn = m.load(syn_dir)
    if kind != "sqlite":
        # the Parquet release names two columns differently from the SQL scripts (the SQLite builder maps
        # them the same way); the synthetic tables keep the source's names
        for d in (real, syn):
            if "orderrows" in d:
                d["orderrows"] = d["orderrows"].rename(columns={"rownumber": "linenumber"})
            if "orders" in d:
                d["orders"] = d["orders"].rename(columns={"dt": "orderdate"})
    mr, ms = m.measures(real), m.measures(syn)
    cmp_ = m.compare(m.dists(real), m.dists(syn))
    out = {k: (mr.get(k), ms.get(k)) for k in dict.fromkeys([*mr, *ms])}
    out.update({k: (0.0, v) for k, v in cmp_.items()})
    return out


def _bi_dir() -> Path | None:
    d = os.environ.get("RELSYNTH_BENCH_BI_DIR")
    return Path(d) if d and (Path(d) / "spec.yaml").exists() else None


def _bi_checks(location: str, syn_dir: Path) -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("bi_checks", _bi_dir() / "checks.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    real, syn = m.load(Path(location.split(":", 1)[1])), m.load(syn_dir)
    rr, ss = m.checks(real, real), m.checks(syn, real)
    out = {k: (rr.get(k), ss.get(k)) for k in rr}
    out.update({k: (None, v) for k, v in m.per_cell(real, syn).items()})
    return out


DATASETS = {
    "chinook": Dataset("chinook", lambda: f"sqlite:{build_chinook()}", _spec("chinook"),
                       note="Chinook 1.4.5 (11 tables); spec = examples/chinook_model/spec.yaml"),
    "adventureworks": Dataset("adventureworks", lambda: f"sqlite:{build_adventureworks()}", _spec("adventureworks"),
                              _aw_checks, note="AdventureWorks DW 2022 as SQLite (27 tables, 1.05M rows); checks G1-G14"),
    "contoso_sqlite": Dataset("contoso_sqlite", lambda: f"sqlite:{build_contoso_sqlite()}", _spec("contoso_sqlite"),
                              _contoso_checks, note="Contoso V2 100k as SQLite with the generator's keys; checks T1-T17"),
    "contoso_parquet": Dataset("contoso_parquet", lambda: f"parquet:{contoso_parquet()}", _spec("contoso_parquet"),
                               _contoso_checks, note="Contoso V2 100k, raw Parquet files; checks T1-T17"),
    "bi_corep_lr": Dataset("bi_corep_lr", lambda: f"csv:{_bi_dir() / 'data'}", lambda: _bi_dir() / "spec.yaml",
                           _bi_checks, public=False,
                           note="private; RELSYNTH_BENCH_BI_DIR with data/, spec.yaml and checks.py (B1-B8)"),
}


def available(name: str) -> tuple[bool, str]:
    if name == "bi_corep_lr" and _bi_dir() is None:
        return False, "RELSYNTH_BENCH_BI_DIR is not set (private dataset, local only)"
    return True, ""
