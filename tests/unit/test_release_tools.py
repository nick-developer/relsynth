# SPDX-License-Identifier: Apache-2.0
"""The release tools: one version everywhere, a changelog section per release,
and a proofs document whose sections are all included and whose fixed
proposition numbers are unique (tools/release.py, tools/check_proofs.py)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_repository_names_one_version():
    release = _load("release")
    assert release.check(require_changelog=False) == []


def test_released_versions_have_a_changelog_section():
    release = _load("release")
    assert release.changelog_section("1.0.1")
    assert release.changelog_section("not-a-version") is None


def test_check_names_a_stale_pin(tmp_path, monkeypatch):
    release = _load("release")
    (tmp_path / "README.md").write_text('pip install "relsynth[postgres]==0.0.1"\n', encoding="utf-8")
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "PINS", {"README.md": release.PINS["README.md"]})
    monkeypatch.setattr(release, "OPENAPI", tmp_path / "none.json")
    monkeypatch.setattr(release, "CHANGELOG", tmp_path / "CHANGELOG.md")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## Unreleased\n", encoding="utf-8")
    problems = release.check(require_changelog=True)
    assert any("names 0.0.1" in p for p in problems)
    assert any("no section for" in p for p in problems)


def test_bump_rewrites_pins_and_dates_the_changelog(tmp_path, monkeypatch):
    release = _load("release")
    (tmp_path / "README.md").write_text('pip install "relsynth==1.0.0"\ndocker build -t relsynth:1.0.0 .\n', encoding="utf-8")
    (tmp_path / "_version.py").write_text('__version__ = "1.0.0"\n', encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## Unreleased\n\n### Added\n\n- x\n", encoding="utf-8")
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "VERSION_FILE", tmp_path / "_version.py")
    monkeypatch.setattr(release, "CHANGELOG", tmp_path / "CHANGELOG.md")
    monkeypatch.setattr(release, "PINS", {"README.md": release.PINS["README.md"]})
    monkeypatch.setattr(release, "OPENAPI", tmp_path / "none.json")
    release.bump("1.1.0", "2026-11-01")
    assert release.current_version() == "1.1.0"
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "relsynth==1.1.0" in readme and "relsynth:1.1.0" in readme
    log = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## Unreleased\n\n## 1.1.0 (2026-11-01)" in log
    assert release.changelog_section("1.1.0").startswith("### Added")
    assert release.check(require_changelog=True) == []


def test_proofs_structure_and_unique_numbers():
    proofs = _load("check_proofs")
    assert proofs.main() == 0
