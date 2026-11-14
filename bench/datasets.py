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
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from relsynth._http import urlopen

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("RELSYNTH_BENCH_CACHE", HERE / ".cache"))
SDV = "https://sdv-datasets-public.s3.amazonaws.com/"
CONTOSO_GEN = "https://raw.githubusercontent.com/sql-bi/Contoso-Data-Generator-V2/eaeb57a9eaa6ad0cdab4fb527552102685434ee0/scripts/sql/"

HF_RELBENCH = "https://huggingface.co/datasets/stanford-star/relbench-v1/resolve/7abf5d46a48534d918704ca2af278861b932f4dc/"
REL_TRIAL = {  # table -> SHA-256 (the LFS object id of the commit above, read 2026-11-13)
    "conditions": "c6feddc073a5808f22561941d2da67c35fbede34df5c4012161f13fe73bad311",
    "conditions_studies": "b9d4518962b64e2abf3699c404333d57d0ac7ba5d2dcf16fdd4769a47cfb5dc1",
    "designs": "8a7f4267afa815ecc9b1bfc37139beffdef7d10fede4caa5a00e5a0d9b93c427",
    "drop_withdrawals": "11c1546cd2f4c0d6dfaff93d998845a72d23d9fccdd4bcd7856392c863dcca4c",
    "eligibilities": "d599b207f37f2aa99e7dcdf7d07410c899e3fa4d4c805c361005464700e72514",
    "facilities": "4b214d93990481874b86d78938b3fdc84680db3cd6ee6e54d56c67a4787dceee",
    "facilities_studies": "9095867fc5b18b6114ac940fc966894b7df2dda376d4e9294f5639f474cc5b3b",
    "interventions": "77603a9692d68c22084d96e2a380eaf1faf9fa61e88e925fb0c0d2651e427aee",
    "interventions_studies": "541dc25a29c43a8e334b0d5996c545dfff939a5e4daf01fe2a6f1cc8a1da4533",
    "outcome_analyses": "12a4f8d57c3701668bd04c0f51d8a29799175927f2d49726914d6d1c98be9147",
    "outcomes": "d5b6149cecfdf28f65c7e8c2b9277d95f5af831a795e41c18bc252abeff0666c",
    "reported_event_totals": "84143d6ed8c142c4a32697eecaad5e5d988a8437f73ebaebf868fb326ca33cf1",
    "sponsors": "8490c115bdcabcc8f6e79cf26413e4db83bccd344158f1375e27e134de10307c",
    "sponsors_studies": "9d89cf918252a0bfd449aa6a8ea41fb95e139ca181b54d9361789b4b9290e020",
    "studies": "f3e43811026e7ca85a8ea35c05a80b5a1bec6463850705167cad3a44c03f1841",
}

# name -> (url, sha256); checked 2026-10-26 against the copies used in layer 2
NYC_TLC = {   # month of 2024 -> SHA-256 of the yellow-taxi trip file
    1: "c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510",
    2: "c76c43c18c6c6664080dd920baab4928988d5786a6b65980792ca7cd796f9f20",
    3: "2d4cdc8fb96726cdd3803b13b02d2e61e71d45720aff0ebc693a8bdd1f249823",
    4: "ce37f736c7cf9e92164c850ef2b502a8f7bb590fc1804b6edbcb80293da01b89",
    5: "1974230be2b5a3ea92e47b5eb77f32dbf1de896d81ad8df4ab6faba52749ad23",
    6: "677cf14c8347f745b583f012fbdba072334c6c4efa17bfd6a64369f2ba30329c",
    7: "af12d6b04daeb78550799ae99933d495b27d715b6dbb0fad9bb750a4537d3070",
    8: "7315643dd16bafc03e27d9903edbfd35404c2550a695d0da0a3e73852d940cfb",
    9: "2f42e4a383a6635de803001462848d0aee2b47f6ce899126805d2d05361fb843",
    10: "d19fe05aa0b259af24eb47051735293287f1e6dc263ba04030ec1ab4c6ec2650",
    11: "5ef321876de5007a7c147a347389133fc94fd7ac28f82eb1d57d8f2cf05cfd3b",
    12: "41ebf7db80bebde60c58e5143c14cdf38ad04a0f3e3ff44215b3e240d55f6c78",
}

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
    # checked 2026-10-30: the SDV public demo copy of the CTU "Financial" (Berka, PKDD'99) database that SyntheRela
    # starts from, and the SDNist 2.3.0 release of the NIST CRC ACS excerpts
    "berka-data.zip": ("https://sdv-datasets-public.s3.amazonaws.com/multi_table/financial/data.zip",
                       "377621d7d204e0e5d4737c97fce70a669bcde14f274b013fd8a0a3508ca78e97"),
    "berka-metadata.json": ("https://sdv-datasets-public.s3.amazonaws.com/multi_table/financial/metadata.json",
                            "ddbc2aa7cf49e01d298522964d1c15bcd97a15536d8255d1c3d0826879e99e87"),
    "diverse_communities_data_excerpts.zip": (
        "https://github.com/usnistgov/SDNist/releases/download/v2.3.0/diverse_communities_data_excerpts.zip",
        "2c244e4f7e61b1e15dffc3e6f6bea5ec755f34106253cf5c42bf422f6964d3c9"),
    # G7 matrix (2026-11-13). SDV's public demo copies of UCI Adult, CoverType and Census-Income (KDD), CC BY 4.0
    # (source.txt of each names the UCI page and licence), and of the Synthea "1K sample" CSV (Apache-2.0, MITRE)
    "adult-data.zip": (SDV + "single_table/adult/data.zip",
                       "2155eacdb9c913a1895d107e858bcf4069259cb7befa6528376a036f7914e2c7"),
    "adult-metadata.json": (SDV + "single_table/adult/metadata.json",
                            "4d9403aa3ef29d0bc2545f9f60293899123b3ec8b4dbc9f1f1ec3aefc8bb6f05"),
    "covtype-data.zip": (SDV + "single_table/covtype/data.zip",
                         "a06719e597ad89e2dcdedc543793290a1a95296939a34f926747f6d2bc398bc4"),
    "covtype-metadata.json": (SDV + "single_table/covtype/metadata.json",
                              "3c7ce0bc1ea47c84f9fd8ec5e183ab22aa72e6b207408ec94517f5eb679b6443"),
    "census-data.zip": (SDV + "single_table/census/data.zip",
                        "599445053327002f60ef5abbb25a16cc9b693594d9cc5fcd2ce48052bcefe90d"),
    "census-metadata.json": (SDV + "single_table/census/metadata.json",
                             "373e383802764efb0ebf824a55e1e9dba7ecf11b5dac22fed8de4c4545c8b06b"),
    "synthea-data.zip": (SDV + "multi_table/synthea/data.zip",
                         "b474615889ab8889471c4d6ea9d08066018bac60206ef6edffb3d759cb197fc2"),
    "synthea-metadata.json": (SDV + "multi_table/synthea/metadata.json",
                              "e5369471f4032a9c2e86ae712b34c8e86fa5f197403af3bc94439ca6edf36101"),
    # research only (docs/datasets.md, IMDb non-commercial terms): the CTU "imdb_MovieLens" database, SDV copy
    "imdb_movielens-data.zip": (SDV + "multi_table/imdb_MovieLens/data.zip",
                                "d2cb689a713eb51d18c0522ba28a9a9606b1705f4edd48976aaf521383211f5d"),
    "imdb_movielens-metadata.json": (SDV + "multi_table/imdb_MovieLens/metadata.json",
                                     "ee3f3601f1343149dcdb84fa7d1a5880cc29d1b1ef9b48007eb45e657a995c18"),
    "synthea-with-dependencies-v3.4.0.jar": (
        "https://github.com/synthetichealth/synthea/releases/download/v3.4.0/synthea-with-dependencies.jar",
        "38678aab1e667d26671163824aad60ee5e30fa366f3d59d5ea5765ebb5702432"),
    # pinned from the first download on a GitHub-hosted runner (hosts this repository's build environment cannot
    # reach); an empty digest makes download() fail and print the file's SHA-256 for pinning
    "mimic-iv-clinical-database-demo-2.2.zip": (
        "https://physionet.org/static/published-projects/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2.zip",
        "97301a03820e8f41af211cf3462ddc19aefe75bbed05f11753859affaafeb8ec"),
    "online_retail_ii.zip": ("https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
                             "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"),
    "coffee.stackexchange.com.7z": ("https://archive.org/download/stackexchange/coffee.stackexchange.com.7z",
                                    "ed92031e75185b1e5549ae257b5280423941d17751042b0369b1f2d3e8d90952"),
    "taxi_zone_lookup.csv": ("https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
                             "1a99e105092230f8620f301edcca7f80d3080642ff404d28ed957d3fa222c8ed"),
    **{f"yellow_tripdata_2024-{m:02d}.parquet": (
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-{m:02d}.parquet", NYC_TLC.get(m, ""))
       for m in range(1, 13)},
    # RelBench rel-trial (ClinicalTrials.gov via AACT; the relbench-v1 card says cc-by-sa-4.0) and the Open Food
    # Facts product database (ODbL), both at a pinned Hugging Face commit; SHA-256 = the LFS object id
    **{f"rel-trial-{t}.parquet": (f"{HF_RELBENCH}rel-trial/db/{t}.parquet", d) for t, d in REL_TRIAL.items()},
    "openfoodfacts-food.parquet": (
        "https://huggingface.co/datasets/openfoodfacts/product-database/resolve/"
        "8673ac5e7af06427071fb435c7c8f95d7e51c4a7/food.parquet",
        "e70c5043101cfcf2f7c7c451c80b327384fe073a2738e4ee1febfeb4a0bde474"),
    # the T12 mirror of UCI ElectricityLoadDiagrams20112014 (CC BY 4.0), as benchmarks/timeseries pins it
    "electricity.txt.gz": ("https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/"
                           "master/electricity/electricity.txt.gz",
                           "3c4c069588198c1fcc95cace7bb69c99922129edfd673b7286661dad20badefa"),
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
    if path.exists() and digest and sha256(path) == digest:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    say(f"downloading {url}")
    tmp = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    got = sha256(tmp)
    if got != digest:
        tmp.unlink()
        if not digest:
            raise RuntimeError(f"{name}: no SHA-256 pinned yet; this download's is {got} (check the file, then pin "
                               "it in bench/datasets.py DOWNLOADS)")
        raise RuntimeError(f"{name}: SHA-256 {got} does not match the pinned {digest}")
    tmp.replace(path)
    return path


def download_all(names: list[str]) -> list[Path]:
    """Every file, downloaded and verified; the unpinned ones are all fetched first and reported
    together, so that one runner pass prints every digest a set needs."""
    paths, unpinned = [], []
    for n in names:
        try:
            paths.append(download(n))
        except RuntimeError as e:
            if "no SHA-256 pinned yet" not in str(e):
                raise
            unpinned.append(str(e))
    if unpinned:
        raise RuntimeError("\n".join(unpinned))
    return paths


def _builder(script: str, *args) -> None:
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


def build_berka() -> Path:
    out = CACHE / "berka" / "berka.db"
    if out.exists():
        return out
    say("building Berka (financial) as SQLite")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    _builder("berka_sqlite.py", download("berka-data.zip"), download("berka-metadata.json"), tmp)
    tmp.replace(out)
    return out


def build_acs() -> Path:
    out = CACHE / "acs" / "acs.db"
    if out.exists():
        return out
    say("building the ACS excerpt as SQLite")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    _builder("acs_sqlite.py", download("diverse_communities_data_excerpts.zip"), tmp)
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
    quality: dict = field(default_factory=dict)     # 2.0 quality report settings (cost budget); None: not run
    # privacy attacks: {"shadow": {...}, "generators": [...]}; None: not run.  generators: the other
    # generator plug-ins attacked with the same settings (G5: every built-in generator has privacy rows)
    privacy: dict | None = None
    repro: bool = False                             # synthesise each seed twice and compare the files
    overrides: tuple = ()                           # spec overrides of this variant (e.g. a scale)
    # G7 matrix: the tier that runs it (pr: every pull request, < 15 min in all; weekly: 6 h budget per job;
    # monthly: SF10+ and the large sets; local: private or research-only data), its seeds there, the seed
    # groups of its CI jobs (one job when empty), where it fits ("github": a GitHub-hosted runner;
    # "maintainer": a bigger runner or the maintainer's machine), the schema-level checks X1-X6
    # (bench/checks/generic.py) and the features of bench/README.md it exercises
    tier: str = "weekly"
    seeds: tuple = (1, 2, 3, 4, 5)
    jobs: tuple = ()
    runner: str = "github"
    generic: bool = False
    coverage: tuple = ()
    kind: str = "relational"                        # "documents": benchmarks/documents (MongoDB samples)
    requires: callable | None = None                # -> (available, why not)
    minutes: float = 0.0                            # a seed's usual minutes on a runner: --jobs starts long runs first


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


def _sqlite_checks(module: str):
    """Domain checks of a SQLite database whose check module computes each value on
    any copy of the tables, given the real ones (checks(tables, ref))."""
    def run(location: str, syn_dir: Path) -> dict:
        import importlib
        m = importlib.import_module(f".checks.{module}", __package__)
        real = sqlite_tables(Path(location.split(":", 1)[1]))
        syn = m.load(syn_dir)
        rr, ss = m.checks(real, real), m.checks(syn, real)
        return {k: (rr.get(k), ss.get(k)) for k in rr}
    return run


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


# Cost budget of the 2.0 quality report on AdventureWorks: its default 20/10 null splits take about an hour
# per seed (T05); 5/3 splits keep every check calibrated with wider limits.
AW_QUALITY = {"null_splits": 5, "multi_splits": 3}
# The shadow-model membership attack where one release fit is cheap (one person table): a quarter of
# relsynth's default shadows; DOMIAS and the Anonymeter attacks run everywhere privacy runs.
# every built-in generator plug-in (relsynth.generators); the spec's own is attacked first
ALL_GENERATORS = ("copula", "cart", "independent")
SHADOW_SMALL = {"enabled": True, "n_train": 16, "n_test": 16, "shadow_size": 500, "targets_vulnerable": 5,
                "targets_random": 5, "n_jobs": os.cpu_count() or 1}

# ---------------------------------------------------------------- G7 builders
OUT = object()     # where a builder writes, in a _built argument list


def _built(rel: str, script: str, *args, dir_out: bool = False) -> Path:
    """CACHE/<rel>, built once by `bench/builders/<script> ARGS`: OUT in args is the output path, a callable
    gives a path (a download), anything else is passed as text."""
    out = CACHE / rel
    if (out.is_dir() and any(out.iterdir())) if dir_out else out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    if dir_out:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        tmp.unlink(missing_ok=True)
    argv = [tmp if a is OUT else (a() if callable(a) else a) for a in args]
    say(f"building {rel}")
    _builder(script, *argv)
    tmp.replace(out)
    return out


def _sdv(name: str) -> callable:
    return lambda: _built(f"{name}/{name}.db", "sdv_sqlite.py", lambda: download(f"{name}-data.zip"),
                          lambda: download(f"{name}-metadata.json"), OUT)


# The copula's rank-likelihood chain on the 500-column table of stress_m/l keeps extending (R-hat over 125k
# correlation entries, up to copula.max_iterations 3000 at about 15 s per sweep): over 45 minutes for that one
# table on 4 cores (G7 finding F4, roadmap/reports/G7.md). The bench caps the chain at its first 120 iterations
# so that the run fits a hosted runner's 6 hours; the spec is otherwise the unedited draft.
# The quality suite's default 10 calibration splits refit the utility models (one-vs-rest logistic
# regressions over the one-hot columns) 31 times per table, which on 120 tables took the job past 6 hours
# although the run itself takes 7 minutes; stress_m/l use AdventureWorks' 3 splits (finding F13).
STRESS_BUDGET = ("synthesis.copula.max_iterations=120",)
# A fact of tens of millions of rows drawn in one block holds its latent normals and their uniforms for
# every row at once (41M trips x 19 columns: two 6 GiB matrices on NYC TLC 12 months), past a hosted
# runner's 16 GB. The largest sets draw their facts with the partitioned runtime (docs/runtime.md),
# 1M rows at a time; the spec is otherwise the unedited draft.
PARTITIONED = ("synthesis.runtime.executor=local",)
# the stress schema's sizes: tables, snowflake depth, wide-table columns, fact rows (bench/builders/stress_schema.py)
STRESS = {"stress_s": (24, 4, 60, 20_000), "stress_m": (120, 8, 500, 200_000), "stress_l": (300, 8, 500, 5_000_000)}


def _sqlite(rel: str, script: str, download_name: str) -> callable:
    """The location of a SQLite database built by `script DOWNLOAD OUT` (one download)."""
    return lambda: f"sqlite:{_built(rel, script, lambda: download(download_name), OUT)}"


def build_stress(name: str) -> Path:
    tables, depth, wide, rows = STRESS[name]
    out = CACHE / "stress" / f"{name}.db"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    say(f"building {name} ({tables} tables, depth {depth}, {wide} columns, {rows:,} fact rows)")
    # argument list without a shell; the paths sit under the operator's own bench cache
    subprocess.run([sys.executable, str(HERE / "builders" / "stress_schema.py"), str(tmp), "--tables", str(tables),  # noqa: S603  # nosemgrep
                    "--depth", str(depth), "--wide", str(wide), "--rows", str(rows)], check=True,
                   stdout=subprocess.DEVNULL)
    tmp.replace(out)
    return out


def build_tpc(kind: str, sf: int) -> Path:
    return _built(f"{kind}_sf{sf}/parquet", "tpc_parquet.py", kind, str(sf), OUT, dir_out=True)


def build_rel_trial() -> Path:
    out = CACHE / "rel_trial" / "parquet"
    if out.is_dir() and len(list(out.glob("*.parquet"))) == len(REL_TRIAL):
        return out
    out.mkdir(parents=True, exist_ok=True)
    for t in REL_TRIAL:
        shutil.copy(download(f"rel-trial-{t}.parquet"), out / f"{t}.parquet")
    return out


def build_nyc_tlc(months: int) -> Path:
    out = CACHE / f"nyc_tlc_{months}m" / "parquet"
    if out.is_dir() and any(out.glob("*.parquet")):
        return out
    zones, *paths = download_all(["taxi_zone_lookup.csv",
                                  *(f"yellow_tripdata_2024-{m:02d}.parquet" for m in range(1, months + 1))])
    tmp = out.with_name("parquet.part")
    shutil.rmtree(tmp, ignore_errors=True)
    say(f"building NYC TLC trips, {months} month(s)")
    # argument list without a shell; the paths sit under the operator's own bench cache
    subprocess.run([sys.executable, str(HERE / "builders" / "nyc_tlc_parquet.py"), str(zones),  # noqa: S603  # nosemgrep
                    str(tmp), *map(str, paths)], check=True, stdout=subprocess.DEVNULL)
    tmp.replace(out)
    return out


def build_synthea(patients: int) -> Path:
    out = CACHE / f"synthea_{patients}" / "synthea.db"
    if out.exists():
        return out
    gen = CACHE / f"synthea_{patients}" / "generated"
    if not (gen / "csv" / "patients.csv").exists():
        say(f"generating a Synthea population of {patients:,} patients")
        _builder("synthea_generate.py", download("synthea-with-dependencies-v3.4.0.jar"), str(patients), gen)
    tmp = out.with_suffix(".part")
    _builder("sdv_sqlite.py", gen / "csv", download("synthea-metadata.json"), tmp)
    tmp.replace(out)
    shutil.rmtree(gen, ignore_errors=True)
    return out


def _java() -> tuple[bool, str]:
    return (shutil.which("java") is not None, "java (JDK 17+) is not installed: Synthea cannot generate patients")


def _research() -> tuple[bool, str]:
    return (os.environ.get("RELSYNTH_BENCH_RESEARCH") == "1",
            "research-only data (non-commercial terms): set RELSYNTH_BENCH_RESEARCH=1 on a local machine to run it, "
            "and publish aggregates only (docs/datasets.md)")


INTROSPECT_MAX_GIB = float(os.environ.get("BENCH_INTROSPECT_MAX_GIB", "12"))


def _rss_gib(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1 << 20)
    except OSError:
        pass                            # not Linux, or the process has ended
    return 0.0


def _watch_memory(proc, name: str) -> int:
    """Wait for introspect; print its resident memory every 5 minutes.  Above INTROSPECT_MAX_GIB (a 16 GB
    runner would kill the job without a word) it is stopped with SIGABRT, and Python's fault handler prints
    where every thread was, so the job log names the line that took the memory."""
    import signal
    import time
    last = peak = 0.0
    start = time.monotonic()
    while proc.poll() is None:
        time.sleep(2)
        rss = _rss_gib(proc.pid)
        peak = max(peak, rss)
        if rss > INTROSPECT_MAX_GIB:
            say(f"{name}: introspect uses {rss:.1f} GiB (limit {INTROSPECT_MAX_GIB:g}); stopped, traceback below")
            proc.send_signal(signal.SIGABRT)
            proc.wait()
            return 137
        if time.monotonic() - last >= 300:
            last = time.monotonic()
            say(f"{name}: introspect running {(last - start) / 60:.0f} min, memory {rss:.1f} GiB (peak {peak:.1f})")
    say(f"{name}: introspect peak memory {peak:.1f} GiB")
    return proc.returncode


def _drafted(name: str) -> callable:
    """The unedited `relsynth introspect` draft of the dataset (relsynth 1.0 format, which bench/assess.py
    reads), made once per commit and dataset into the cache: G7 datasets run with the draft of the code under
    test, so a change of what introspect proposes shows in their results (the frozen specs of the 2.0
    golden databases keep separating the two)."""
    def spec() -> Path:
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=HERE, capture_output=True,  # noqa: S603, S607
                             text=True, check=False).stdout.strip() or "nogit"
        out = CACHE / "drafts" / f"{name}-{sha}.yaml"
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        loc = DATASETS[name].location()
        tmp = out.with_name(out.name + ".part.yaml")
        say(f"{name}: drafting the spec with relsynth introspect")
        # argument list without a shell; the paths sit under the operator's own bench cache
        proc = subprocess.Popen([sys.executable, "-m", "relsynth", "introspect", "--source", loc, "--out", str(tmp),  # noqa: S603  # nosemgrep
                                 "--name", name, "--format", "v1", "--log-level", "debug"],
                                stdout=subprocess.DEVNULL, env={**os.environ, "PYTHONFAULTHANDLER": "1"})
        # its progress goes to stderr, so a job log shows the stage where a large set stalls or runs out of memory,
        # and at debug level a failure prints its traceback there
        code = _watch_memory(proc, name)
        if code or not tmp.exists():
            raise RuntimeError(f"{name}: introspect failed ({code}); its messages are above")
        tmp.replace(out)
        return out
    return spec


DATASETS = {
    # ---------------------------------------------------------------- the 2.0 golden databases (frozen specs)
    "chinook": Dataset("chinook", lambda: f"sqlite:{build_chinook()}", _spec("chinook"),
                       note="Chinook 1.4.5 (11 tables); spec = examples/chinook_model/spec.yaml", repro=True,
                       tier="pr", generic=True, minutes=2.5,
                       coverage=("size_mb", "self_reference", "many_to_many", "small_privacy_units", "events")),
    "adventureworks": Dataset("adventureworks", lambda: f"sqlite:{build_adventureworks()}", _spec("adventureworks"),
                              _aw_checks, note="AdventureWorks DW 2022 as SQLite (27 tables, 1.05M rows); checks G1-G14",
                              quality=AW_QUALITY, seeds=(1, 2, 3), jobs=((1, 2), (3,)), generic=True,
                              coverage=("tables_10_plus", "snowflake_depth_3", "self_reference", "scd", "events")),
    "contoso_sqlite": Dataset("contoso_sqlite", lambda: f"sqlite:{build_contoso_sqlite()}", _spec("contoso_sqlite"),
                              _contoso_checks, note="Contoso V2 100k as SQLite with the generator's keys; checks T1-T17",
                              privacy={"shadow": {"enabled": False}}, tier="monthly", seeds=(1,),
                              coverage=("events", "high_cardinality")),
    "contoso_parquet": Dataset("contoso_parquet", lambda: f"parquet:{contoso_parquet()}", _spec("contoso_parquet"),
                               _contoso_checks, note="Contoso V2 100k, raw Parquet files; checks T1-T17",
                               coverage=("events", "high_cardinality", "size_100mb")),
    "berka": Dataset("berka", lambda: f"sqlite:{build_berka()}", _spec("berka"), _sqlite_checks("berka"),
                     note="Berka / PKDD'99 Czech bank (SyntheRela 'financial', 8 tables, 1.08M rows); checks F1-F16",
                     privacy={"shadow": {"enabled": False}, "generators": ALL_GENERATORS}, generic=True,
                     coverage=("many_to_many", "events", "skew", "size_100mb")),
    "acs": Dataset("acs", lambda: f"sqlite:{build_acs()}", _spec("acs"), _sqlite_checks("acs"),
                   note="ACS PUMS 2019, NIST CRC national excerpt (27,253 people, 24 columns); checks A1-A14",
                   privacy={"shadow": SHADOW_SMALL, "generators": ALL_GENERATORS}, repro=True,
                   coverage=("single_table", "messy_values", "high_cardinality")),
    "bi_corep_lr": Dataset("bi_corep_lr", lambda: f"csv:{_bi_dir() / 'data'}", lambda: _bi_dir() / "spec.yaml",
                           _bi_checks, public=False, tier="local",
                           note="private; RELSYNTH_BENCH_BI_DIR with data/, spec.yaml and checks.py (B1-B8)",
                           coverage=("high_cardinality", "skew")),
    # ---------------------------------------------------------------- G7 matrix (unedited introspect drafts)
    "adult": Dataset("adult", lambda: f"sqlite:{_sdv('adult')()}", _drafted("adult"),
                     note="UCI Adult (32,561 people, 15 columns; SDV copy, CC BY 4.0)", tier="pr", seeds=(1,),
                     generic=True, privacy={"shadow": {"enabled": False}}, quality=AW_QUALITY, minutes=6,   # PR budget
                     coverage=("single_table", "size_mb", "messy_values", "skew")),
    "covtype": Dataset("covtype", lambda: f"sqlite:{_sdv('covtype')()}", _drafted("covtype"),
                       note="UCI CoverType (581,012 rows, 55 columns: 44 binary indicators; SDV copy, CC BY 4.0)",
                       seeds=(1, 2, 3), generic=True, coverage=("single_table", "wide_tables", "skew", "size_100mb")),
    "census_kdd": Dataset("census_kdd", lambda: f"sqlite:{_sdv('census')()}", _drafted("census_kdd"),
                          note="UCI Census-Income KDD (299,285 people, 42 columns; SDV copy, CC BY 4.0)",
                          seeds=(1, 2, 3), generic=True,
                          coverage=("single_table", "high_cardinality", "messy_values", "skew", "size_100mb")),
    "synthea_1k": Dataset("synthea_1k", lambda: f"sqlite:{_sdv('synthea')()}", _drafted("synthea_1k"),
                          note="Synthea 1K sample (18 tables, 1.84M rows, 1,163 patients; SDV copy, Apache-2.0)",
                          seeds=(1,), generic=True,
                          coverage=("tables_10_plus", "events", "high_cardinality", "size_gb")),
    "synthea_100k": Dataset("synthea_100k", lambda: f"sqlite:{build_synthea(100_000)}", _drafted("synthea_100k"),
                            note="Synthea 3.4.0, 100,000 generated patients (Massachusetts, seed 7)", tier="monthly",
                            seeds=(1,), runner="maintainer", generic=True, requires=_java,
                            coverage=("tables_10_plus", "events", "size_10gb")),
    "synthea_1m": Dataset("synthea_1m", lambda: f"sqlite:{build_synthea(1_000_000)}", _drafted("synthea_1m"),
                          note="Synthea 3.4.0, 1,000,000 generated patients (Massachusetts, seed 7)", tier="monthly",
                          overrides=PARTITIONED, seeds=(1,), runner="maintainer", generic=True, requires=_java,
                          coverage=("tables_10_plus", "events", "size_100gb")),
    "mimic_demo": Dataset("mimic_demo",
                          _sqlite("mimic_demo/mimic.db", "mimic_sqlite.py", "mimic-iv-clinical-database-demo-2.2.zip"),
                          _drafted("mimic_demo"),
                          note="MIMIC-IV Clinical Database Demo 2.2 (100 patients, 31 tables; ODbL 1.0)",
                          seeds=(1, 2, 3), generic=True,
                          coverage=("tables_10_plus", "small_privacy_units", "events", "composite_keys")),
    "online_retail_ii": Dataset("online_retail_ii",
                                _sqlite("online_retail_ii/retail.db", "online_retail_sqlite.py", "online_retail_ii.zip"),
                                _drafted("online_retail_ii"),
                                note="UCI Online Retail II (1.07M invoice lines as a 4-table star; CC BY 4.0)",
                                seeds=(1, 2, 3), generic=True,
                                coverage=("messy_values", "high_cardinality", "skew", "events")),
    "stackexchange": Dataset("stackexchange",
                             _sqlite("stackexchange/coffee.db", "stackexchange_sqlite.py", "coffee.stackexchange.com.7z"),
                             _drafted("stackexchange"),
                             note="coffee.stackexchange.com data dump (7 tables; CC BY-SA 4.0)", seeds=(1, 2, 3),
                             generic=True, coverage=("long_text", "self_reference", "skew", "messy_values")),
    "rel_trial": Dataset("rel_trial", lambda: f"parquet:{build_rel_trial()}", _drafted("rel_trial"),
                         note="RelBench rel-trial (15 tables, about 5.5M rows; ClinicalTrials.gov via AACT)",
                         seeds=(1,), generic=True,
                         coverage=("tables_10_plus", "many_to_many", "long_text", "events", "size_gb")),
    "electricity": Dataset("electricity",
                           _sqlite("electricity_bench/electricity.db", "electricity_sqlite.py", "electricity.txt.gz"),
                           _spec("electricity"),
                           note="UCI ElectricityLoadDiagrams20112014, last year hourly (321 meters, 2.8M rows); T12 spec",
                           seeds=(1,), generic=True, coverage=("regular_series", "size_100mb")),
    "nyc_tlc_1m": Dataset("nyc_tlc_1m", lambda: f"parquet:{build_nyc_tlc(1)}", _drafted("nyc_tlc_1m"),
                          note="NYC TLC yellow taxi trips 2024-01 (about 3M trips) and the zone lookup", seeds=(1,),
                          generic=True, coverage=("events", "messy_values", "skew", "size_100mb")),
    "nyc_tlc_12m": Dataset("nyc_tlc_12m", lambda: f"parquet:{build_nyc_tlc(12)}", _drafted("nyc_tlc_12m"),
                           note="NYC TLC yellow taxi trips 2024 (12 months, about 41M trips)", tier="monthly",
                           overrides=PARTITIONED,
                           seeds=(1,), generic=True, coverage=("events", "messy_values", "skew", "size_gb")),
    "open_food_facts": Dataset("open_food_facts",
                               lambda: "parquet:" + str(_built("open_food_facts/parquet", "openfoodfacts_extract.py",
                                                                lambda: download("openfoodfacts-food.parquet"), OUT,
                                                                dir_out=True)),
                               _drafted("open_food_facts"),
                               note="Open Food Facts, 200,000-product extract of the 8 GB export (ODbL)",
                               tier="monthly", seeds=(1,), generic=True,
                               coverage=("long_text", "nested_documents", "wide_tables", "messy_values",
                                         "high_cardinality")),
    "tpch_sf1": Dataset("tpch_sf1", lambda: f"parquet:{build_tpc('tpch', 1)}", _drafted("tpch_sf1"),
                        note="TPC-H data generator at SF1 (8 tables, 8.7M rows; not TPC-H results)", seeds=(1,),
                        generic=True, coverage=("composite_keys", "size_gb", "snowflake_depth_3")),
    "tpch_sf10": Dataset("tpch_sf10", lambda: f"parquet:{build_tpc('tpch', 10)}", _drafted("tpch_sf10"),
                         note="TPC-H data generator at SF10 (87M rows; not TPC-H results)", tier="monthly",
                         overrides=PARTITIONED, seeds=(1,), generic=True, coverage=("composite_keys", "size_10gb")),
    "tpch_sf100": Dataset("tpch_sf100", lambda: f"parquet:{build_tpc('tpch', 100)}", _drafted("tpch_sf100"),
                          note="TPC-H data generator at SF100 (870M rows; not TPC-H results)", tier="monthly",
                          overrides=PARTITIONED,
                          seeds=(1,), runner="maintainer", generic=True, coverage=("composite_keys", "size_100gb")),
    "tpcds_sf1": Dataset("tpcds_sf1", lambda: f"parquet:{build_tpc('tpcds', 1)}", _drafted("tpcds_sf1"),
                         note="TPC-DS data generator at SF1 (24 tables; not TPC-DS results)", seeds=(1,), generic=True,
                         coverage=("tables_10_plus", "scd", "composite_keys", "snowflake_depth_3", "size_gb")),
    "tpcds_sf10": Dataset("tpcds_sf10", lambda: f"parquet:{build_tpc('tpcds', 10)}", _drafted("tpcds_sf10"),
                          note="TPC-DS data generator at SF10 (24 tables; not TPC-DS results)", tier="monthly",
                          overrides=PARTITIONED,
                          seeds=(1,), generic=True, coverage=("tables_10_plus", "scd", "composite_keys", "size_10gb")),
    "stress_s": Dataset("stress_s", lambda: f"sqlite:{build_stress('stress_s')}", _drafted("stress_s"),
                        note="schema stress generator, small (27 tables, depth 4, 60-column table, 26k rows)",
                        tier="pr", seeds=(1,), generic=True, repro=True, quality=AW_QUALITY, minutes=8,
                        coverage=("tables_10_plus", "snowflake_depth_4", "self_reference", "many_to_many",
                                  "composite_keys", "cyclic_keys", "binary_columns", "regular_series", "skew",
                                  "messy_values", "size_mb")),
    "stress_m": Dataset("stress_m", lambda: f"sqlite:{build_stress('stress_m')}", _drafted("stress_m"),
                        overrides=STRESS_BUDGET,
                        quality=AW_QUALITY,
                        note="schema stress generator, medium (120 tables, depth 8, 500-column table, 340k rows)",
                        seeds=(1,), generic=True,
                        coverage=("tables_100_plus", "snowflake_depth_4", "self_reference", "many_to_many",
                                  "composite_keys", "cyclic_keys", "binary_columns", "wide_tables", "regular_series",
                                  "skew", "messy_values")),
    "stress_l": Dataset("stress_l", lambda: f"sqlite:{build_stress('stress_l')}", _drafted("stress_l"),
                        overrides=STRESS_BUDGET,
                        quality=AW_QUALITY,
                        note="schema stress generator, large (300 tables, depth 8, 500-column table, 5M fact rows)",
                        tier="monthly", seeds=(1,), generic=True,
                        coverage=("tables_100_plus", "snowflake_depth_4", "wide_tables", "cyclic_keys",
                                  "binary_columns", "skew", "size_gb")),
    "mongodb_samples": Dataset("mongodb_samples", lambda: "documents:mongodb", lambda: None, kind="documents",
                               note="MongoDB sample datasets (supplies, analytics, geospatial, training) through "
                                    "relsynth.documents", tier="pr", seeds=(1,), minutes=8,
                               quality={"splits": 3},     # calibration splits of the shape limits (PR budget)
                               coverage=("nested_documents", "size_mb", "high_cardinality")),
    "imdb_movielens": Dataset("imdb_movielens", lambda: f"sqlite:{_sdv('imdb_movielens')()}",
                              _drafted("imdb_movielens"), tier="local", seeds=(1,), generic=True,
                              requires=_research,
                              note="research only: IMDb x MovieLens (CTU copy via SDV; non-commercial), aggregates only",
                              coverage=("many_to_many", "skew")),
}


def private_store() -> Path | None:
    """Where results of private datasets are kept (never in the repository):
    <RELSYNTH_BENCH_BI_DIR>/bench/{history/, baseline.json}."""
    d = _bi_dir()
    return d / "bench" if d else None


def is_private(name: str) -> bool:
    return name in DATASETS and not DATASETS[name].public


def available(name: str) -> tuple[bool, str]:
    if name == "bi_corep_lr" and _bi_dir() is None:
        return False, "RELSYNTH_BENCH_BI_DIR is not set (private dataset, local only)"
    req = DATASETS[name].requires
    return req() if req is not None else (True, "")


# The features every dataset of the matrix declares in `coverage`; bench/README.md lists them with the datasets that
# exercise each, and every feature of REQUIRED_FEATURES must be exercised by at least two (tests/unit/test_bench_matrix.py)
FEATURES = {
    "single_table": "one table (microdata)",
    "tables_10_plus": "10 or more tables",
    "tables_100_plus": "100 or more tables",
    "size_mb": "KB to tens of MB",
    "size_100mb": "about 100 MB to 1 GB",
    "size_gb": "1 GB to 10 GB",
    "size_10gb": "10 GB to 100 GB",
    "size_100gb": "100 GB and more",
    "snowflake_depth_3": "snowflake chains of depth 3",
    "snowflake_depth_4": "snowflake chains of depth 4 or more",
    "self_reference": "self-references (trees)",
    "cyclic_keys": "a cycle of links between tables",
    "many_to_many": "many-to-many bridges",
    "composite_keys": "composite primary and foreign keys",
    "scd": "slowly changing dimensions (type 2)",
    "events": "dated events (facts with a time column)",
    "regular_series": "regular time series (one value per entity and step)",
    "long_text": "long free text",
    "nested_documents": "nested documents and lists",
    "wide_tables": "wide tables (50 or more columns)",
    "binary_columns": "binary columns (BLOB)",
    "high_cardinality": "high-cardinality categoricals",
    "skew": "skew and imbalance (fan-out, classes)",
    "messy_values": "messy values (codes for missing, negatives, outliers, stray text)",
    "small_privacy_units": "small privacy units (under 200 people)",
}
REQUIRED_FEATURES = tuple(f for f in FEATURES if f not in ("snowflake_depth_3", "size_100mb", "cyclic_keys",
                                                            "binary_columns"))
TIERS = ("pr", "weekly", "monthly", "local")


def tier_datasets(tier: str) -> list[str]:
    """The datasets a tier runs (a tier runs only its own: the PR tier is not repeated weekly)."""
    return [n for n, d in DATASETS.items() if d.tier == tier]
