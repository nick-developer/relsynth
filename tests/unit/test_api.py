"""The Python API as a user calls it: introspect -> fit -> sample -> evaluate ->
write, run with quality gates, doctor, export, clone, profile, ping."""
from __future__ import annotations

import json

import pandas as pd
import pytest

import relsynth as rs
from relsynth.errors import EngineError, QualityGateError, SourceError, exit_code_for
from relsynth.spec import SpecError


def test_public_names():
    for name in ("Spec", "connect", "introspect", "fit", "sample", "evaluate", "run", "doctor", "tune",
                 "export_model", "clone_schema", "profile", "ping", "validate_spec", "__version__"):
        assert hasattr(rs, name), name
    assert rs.__version__.count(".") == 2


def test_introspect_draft_is_valid_and_runnable(star_sqlite, tmp_path):
    _, db = star_sqlite
    spec = rs.introspect(f"sqlite:{db}")
    assert set(spec.tables) == {"customer", "product", "sales"}
    assert {(r["child"], r["parent"]) for r in spec.relationships} == {("sales", "customer"), ("sales", "product")}
    assert spec.errors() == [] and isinstance(spec.review, list)
    data = rs.sample(spec.with_target(f"csv:{tmp_path}"), scale=0.2)
    assert set(data) == set(spec.tables) and data.rows["customer"] == pytest.approx(300, abs=1)


def test_fit_sample_evaluate_write(star_sqlite, tmp_path):
    spec, _ = star_sqlite
    model = rs.fit(spec, seed=5)
    assert "FittedModel" in repr(model) and model.summary()
    d0, d0b, d1 = model.sample(dataset=0), model.sample(dataset=0), model.sample(dataset=1)
    assert d0["sales"].equals(d0b["sales"]) and not d0["sales"].equals(d1["sales"])
    big = model.sample(scale={"customer": 2.0}, dataset=0)
    assert len(big["customer"]) == 2 * len(spec_tables(spec)["customer"])
    assert len(big["product"]) == len(spec_tables(spec)["product"])
    assert len(model.sample(scale={"default": 0.5}, dataset=0)["product"]) == 150
    with pytest.raises(rs.SpecError):
        model.sample(scale={"nope": 2})
    rep = model.evaluate(d0)
    assert rep.ok and rep.score == 1.0 and rep.total > 20
    assert json.loads(rep.to_json(tmp_path / "r.json"))["scorecard"]["total"] == rep.total
    assert rep.to_markdown().startswith("#")
    # sales keys point at synthetic customers
    assert d0["sales"]["customer_id"].dropna().isin(d0["customer"]["customer_id"]).all()
    d0.to_csv(tmp_path / "csv")
    assert sorted(p.stem for p in (tmp_path / "csv").glob("*.csv")) == ["customer", "product", "sales"]
    pytest.importorskip("pyarrow")
    d0.to_parquet(tmp_path / "pq")
    assert len(pd.read_parquet(tmp_path / "pq" / "sales.parquet")) == len(d0["sales"])
    d0.write(f"sqlite:{tmp_path / 'syn.db'}")
    assert rs.connect(f"sqlite:{tmp_path / 'syn.db'}").count("sales") == len(d0["sales"])


def spec_tables(spec):
    return {t: pd.read_sql_table(t, f"sqlite:///{spec['source']['path']}") for t in spec["tables"]}


def test_evaluate_external_tables(star_sqlite):
    spec, _ = star_sqlite
    real = spec_tables(spec)
    rep = rs.evaluate(spec, real, real=real, privacy=False)
    assert rep.total > 0
    # an obviously broken synthetic database fails
    bad = {t: df.copy() for t, df in real.items()}
    bad["sales"] = bad["sales"].sample(frac=0.2, random_state=1)
    assert rs.evaluate(spec, bad, real=real, privacy=False).score < 1.0


def test_quality_report_api(star_sqlite, tmp_path):
    spec, _ = star_sqlite
    real = spec_tables(spec)
    syn = rs.fit(spec, seed=1).sample()
    q = rs.assess_quality(spec, syn, config={"null_splits": 4, "multi_splits": 3})
    assert set(q.vector) == {"fidelity", "utility", "integrity", "verdict"}
    assert q.scorecard.total > 0 and q.raw["scorecard_1"]["total"] == q.scorecard.total
    q.to_json(tmp_path / "q.json")
    assert '"schema_version": "2.0.0"' in (tmp_path / "q.json").read_text()
    bad = {t: df.copy() for t, df in real.items()}
    bad["sales"]["customer_id"] = bad["sales"]["customer_id"].sample(frac=1, random_state=3).to_numpy()
    with pytest.raises(QualityGateError):
        rs.assess_quality(spec, bad, real=real, privacy=False, config={"calibrate": False}).require(["fidelity"])
    with pytest.raises(rs.SpecError):
        rs.assess_quality(spec, real, real=real, config={"classifier": "svm"})


def test_run_and_quality_gate(star_sqlite, tmp_path):
    spec, _ = star_sqlite
    r = rs.run(spec, out=tmp_path / "o", bundle=tmp_path / "b.json", privacy=False, seed=1)
    assert r.ok and r.report.ok and (tmp_path / "o" / "sales.csv").exists() and (tmp_path / "b.json").exists()
    with pytest.raises(QualityGateError) as e:
        rs.run(spec, write=False, privacy=False, fail_under=1.01)
    assert exit_code_for(e.value) == 1 and e.value.report is not None
    r2 = rs.run(spec, datasets=2, out=tmp_path / "many", privacy=False)
    assert r2.ok and len(r2.written) >= 2


def test_run_without_raising_reports_errors(tmp_path):
    spec = {"source": {"kind": "sqlite", "path": str(tmp_path / "missing.db")},
            "tables": {"t": {"primary_key": "id", "columns": {"id": "id"}}}}
    r = rs.run(spec, write=False, raise_errors=False)
    assert not r.ok and r.errors
    with pytest.raises(SourceError):
        rs.run(spec, write=False)


def test_invalid_spec_raises_spec_error():
    with pytest.raises(SpecError) as e:
        rs.fit({"tables": {"t": {"primary_key": "id", "columns": {"id": "id"}}},
                "synthesis": {"pushdown": {"mode": "sometimes"}}})
    assert exit_code_for(e.value) == 2


def test_doctor_and_fixed_spec(star_sqlite, tmp_path):
    spec, _ = star_sqlite
    r = rs.run(spec, write=False, privacy=False, raise_errors=False)
    dx = rs.doctor(r)
    assert isinstance(dx.markdown, str) and "Diagnosis" in repr(dx)
    assert isinstance(dx.fixed_spec(spec), rs.Spec)
    r.save_bundle(tmp_path / "b.json")
    assert len(rs.doctor(tmp_path / "b.json").issues) == len(dx.issues)


def test_export_model(star_sqlite, tmp_path):
    spec, _ = star_sqlite
    out = rs.export_model(spec, tmp_path / "m", formats=("json", "md", "py"), draws=2, replica_check=False)
    assert (tmp_path / "m").is_dir() and out
    js = json.loads(next((tmp_path / "m").glob("*.json")).read_text())
    assert "tables" in json.dumps(js)[:100000]


@pytest.mark.slow
def test_exported_run_model_reproduces_run(star_sqlite, tmp_path):
    """run_model.py from an exported model writes the same files as rs.run, also
    when the spec was given as a dict (the bundle then carries its own spec.yaml)."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    spec, _ = star_sqlite
    rs.run(spec, out=f"csv:{tmp_path / 'a'}", privacy=False)
    rs.export_model(spec, tmp_path / "m", formats=("json", "py"), replica_check=False)
    assert "***" not in (tmp_path / "m" / "spec.yaml").read_text() or "url_env" in (tmp_path / "m" / "spec.yaml").read_text()
    src = str(Path(rs.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=src, RELSYNTH_QUIET="1")
    r = subprocess.run([sys.executable, str(tmp_path / "m" / "run_model.py"), "--out", "b"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    a = sorted((tmp_path / "a").glob("*.csv"))
    assert a and all(f.read_bytes() == (tmp_path / "b" / f.name).read_bytes() for f in a)


def test_clone_profile_ping(star_sqlite, tmp_path):
    _, db = star_sqlite
    made = rs.clone_schema(f"sqlite:{db}", f"sqlite:{tmp_path / 'e.db'}")
    assert made.index("customer") < made.index("sales")
    with pytest.raises(SourceError):
        rs.clone_schema(f"sqlite:{db}", f"csv:{tmp_path / 'f'}")
    prof = rs.profile(f"sqlite:{db}", tables=["customer"])
    assert prof["customer"]["rows"] == 1500 and "top" not in next(iter(prof["customer"]["columns"].values()))
    top = rs.profile(f"sqlite:{db}", tables=["customer"], top=3)
    assert all(len(c.get("top", [])) <= 3 for c in top["customer"]["columns"].values())
    assert rs.ping(f"sqlite:{db}")["ok"]
    bad = rs.ping("postgresql://u:secretpw@127.0.0.1:9/x?connect_timeout=2")
    assert not bad["ok"] and "secretpw" not in json.dumps(bad)


def test_engine_errors_are_typed(star_sqlite, monkeypatch):
    spec, _ = star_sqlite
    import relsynth.engine as E

    def boom(self, *a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(E.Synthesizer, "fit", boom)
    with pytest.raises(EngineError) as e:
        rs.fit(spec)
    assert exit_code_for(e.value) == 4


def test_child_entities_follow_a_scaled_parent(tmp_path):
    """Invoices (an entity child of customer) follow customer's factor unless they
    have their own; the scorecard expects the same sizes."""
    from conftest import write_scenario
    spec = write_scenario("S9_small_shop", {"kind": "sqlite", "path": str(tmp_path / "shop.db")})
    model = rs.fit(spec, seed=3)
    base = model.sample(dataset=0)
    big = model.sample(scale={"customer": 3}, dataset=0)
    assert len(big["customer"]) == 3 * len(base["customer"])
    assert len(big["invoice"]) == 3 * len(base["invoice"])
    assert abs(len(big["invoice_line"]) / len(base["invoice_line"]) - 3) < 0.3
    assert len(big["track"]) == len(base["track"])                       # not a child of customer
    rep = model.evaluate(big, privacy=False)
    assert not [f for f in rep.failed if f["area"].startswith("invoice.customer_id")]
    own = model.sample(scale={"customer": 3, "invoice": 1}, dataset=0)
    assert len(own["invoice"]) == len(base["invoice"])                   # its own factor wins
