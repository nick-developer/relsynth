"""The Spec object: loading, validation, overrides, saving (secrets masked)."""
from __future__ import annotations

import pytest
import yaml

import relsynth as rs
from relsynth.spec import SpecError, load_spec

BASE = {"tables": {"c": {"primary_key": "id", "columns": {"id": "id", "v": "integer"}}}}


def test_load_and_resolve_defaults(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(BASE))
    s = rs.Spec.load(p)
    assert s.tables == ["c"] and s.path == p and s.base_dir == tmp_path
    r = s.resolved
    assert r["synthesis"]["pushdown"] == {"mode": "auto", "min_rows": 1_000_000, "sample_rows": 500_000}
    assert r["_base_dir"] == str(tmp_path.resolve())


def test_missing_file_is_a_spec_error(tmp_path):
    with pytest.raises(SpecError):
        rs.Spec.load(tmp_path / "nope.yaml")


@pytest.mark.parametrize("bad, msg", [
    ({"synthesis": {"pushdown": {"mode": "sometimes"}}}, "pushdown.mode"),
    ({"synthesis": {"pushdown": {"sample_rows": 0}}}, "pushdown.sample_rows"),
    ({"synthesis": {"pushdown": {"min_rows": -5}}}, "pushdown.min_rows"),
    ({"synthesis": {"posterior": {"mode": "avg"}}}, "posterior.mode"),
    ({"target": {"kind": "excel", "path": "o"}}, "target.kind"),
    ({"target": {"kind": "sqlite", "path": "o.db", "structure": "copy"}}, "target.structure"),
    ({"source": {"kind": "postgresql"}}, "url"),
])
def test_invalid_settings_are_refused(bad, msg):
    s = rs.Spec.from_dict({**BASE, **bad})
    errs = s.errors()
    assert errs and any(msg in e for e in errs), errs
    with pytest.raises(SpecError):
        s.check()


@pytest.mark.parametrize("kind, extra", [("parquet", {"path": "o"}), ("csv", {"path": "o"}),
                                         ("sqlite", {"path": "o.db"}), ("postgresql", {"url": "postgresql://u@h/d"}),
                                         ("mssql", {"url": "mssql://u@h/d"}), ("oracle", {"url": "oracle://u@h/?service_name=x"})])
def test_every_target_kind_validates(kind, extra):
    assert rs.Spec.from_dict({**BASE, "target": {"kind": kind, **extra}}).errors() == []


def test_set_overrides_are_typed_and_copy_on_write():
    s = rs.Spec.from_dict(BASE)
    s2 = s.set("synthesis.scale.default", "2.5").set("synthesis.pushdown.mode", "on").set("target.kind", "parquet")
    assert s2.raw["synthesis"]["scale"]["default"] == 2.5
    assert s2.raw["synthesis"]["pushdown"]["mode"] == "on"
    assert "synthesis" not in s.raw                      # the original is unchanged
    assert s2.set("a.b", "[1, 2]").raw["a"]["b"] == [1, 2]
    with pytest.raises(SpecError):
        s2.set("synthesis.scale.default.x", 1)           # not a mapping


def test_unquoted_on_off_in_yaml_means_the_setting(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(BASE) + "synthesis:\n  pushdown:\n    mode: on\n")
    assert rs.Spec.load(p).resolved["synthesis"]["pushdown"]["mode"] == "on"
    p.write_text(yaml.safe_dump(BASE) + "synthesis:\n  pushdown:\n    mode: off\n")
    assert rs.Spec.load(p).resolved["synthesis"]["pushdown"]["mode"] == "off"


def test_with_source_parses_locations():
    s = rs.Spec.from_dict(BASE)
    assert s.with_source("sqlite:a.db").raw["source"] == {"kind": "sqlite", "path": "a.db"}
    assert s.with_source("parquet:lake").raw["source"] == {"kind": "parquet", "path": "lake"}
    assert s.with_source("csv:dir").raw["source"] == {"kind": "csv", "path": "dir"}
    assert s.with_source("postgresql+psycopg://u@h/d").raw["source"]["kind"] == "postgresql"
    assert s.with_target("mssql+pyodbc://u@h/d").raw["target"]["kind"] == "mssql"


def test_save_masks_passwords(tmp_path):
    s = rs.Spec.from_dict({**BASE, "source": {"kind": "postgresql", "url": "postgresql://scott:tiger@db/prod"}})
    p = s.save(tmp_path / "out.yaml")
    txt = p.read_text()
    assert "tiger" not in txt and "url_env: RELSYNTH_SOURCE_URL" in txt
    raw = s.save(tmp_path / "raw.yaml", mask_secrets=False).read_text()
    assert "tiger" in raw


def test_yaml_aliases_are_refused(tmp_path):
    bomb = "a: &a [1, 1]\nb: &b [*a, *a]\nc: [*b, *b]\n"
    p = tmp_path / "bomb.yaml"
    p.write_text(bomb)
    with pytest.raises(Exception):
        rs.Spec.load(p)


def test_yaml_python_tags_are_refused(tmp_path):
    p = tmp_path / "evil.yaml"
    p.write_text("tables: !!python/object/apply:os.system ['echo hacked']\n")
    with pytest.raises(Exception) as e:
        rs.Spec.load(p).resolved
    assert "hacked" not in str(e.value) or "python/object" in str(e.value)


def test_model_report_markdown(tmp_path):
    md = rs.Spec.from_dict(BASE).model_report(tmp_path / "m")
    assert "c" in md and (tmp_path / "m.md").exists() and (tmp_path / "m.json").exists()


def test_load_spec_accepts_spec_like_inputs(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(BASE))
    for x in (str(p), p, BASE):
        assert load_spec(x)["tables"]["c"]["primary_key"] == "id"
    assert rs.Spec.coerce(rs.Spec.from_dict(BASE)).tables == ["c"]
    with pytest.raises(TypeError):
        rs.Spec.coerce(42)
