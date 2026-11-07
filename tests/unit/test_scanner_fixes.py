# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the code scanning fixes: http(s)-only URL opening and
quoted DDL in the benchmark builders."""
from __future__ import annotations

import sqlite3
import urllib.request

import pytest

from bench.builders.sqlite_ddl import create_table, quote
from relsynth._http import urlopen


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.org/x", "data:text/plain,hi", "/etc/passwd"])
def test_urlopen_refuses_other_schemes(url):
    """file://, ftp://, data: and bare paths are refused before anything is opened."""
    with pytest.raises(ValueError, match="only http"):
        urlopen(url, timeout=1)


def test_urlopen_refuses_file_requests():
    """A Request object is checked the same way as a string."""
    with pytest.raises(ValueError, match="only http"):
        urlopen(urllib.request.Request("file:///etc/passwd"), timeout=1)


def test_opener_has_no_file_handler():
    """A redirect cannot reach file:// either: the opener has no handler for it."""
    from relsynth import _http
    assert not any(isinstance(h, (urllib.request.FileHandler, urllib.request.FTPHandler, urllib.request.DataHandler))
                   for h in _http._OPENER.handlers)


def test_create_table_quotes_hostile_names():
    """Names with quotes and SQL in them stay names; the other table is untouched."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE keep (x INTEGER)")
    bad = 'x"; DROP TABLE keep; --'
    create_table(con, bad, [(bad, "TEXT", True), ("id", "INTEGER", True)], primary_key=["id"],
                 foreign_keys=[(["id"], "keep", ["x"])])
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert names == {"keep", bad}
    assert [r[1] for r in con.execute(f"PRAGMA table_info({quote(bad)})")] == [bad, "id"]


def test_create_table_refuses_unknown_types():
    """Column types come from a fixed list, so a type cannot carry SQL."""
    con = sqlite3.connect(":memory:")
    with pytest.raises(ValueError, match="column type"):
        create_table(con, "t", [("a", "TEXT); DROP TABLE t; --", False)])
