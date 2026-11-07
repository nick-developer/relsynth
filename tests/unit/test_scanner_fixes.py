# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the code scanning fixes: http(s)-only URL opening and
quoted DDL in the benchmark builders."""
from __future__ import annotations

import sqlite3
import urllib.request

import pytest

from bench.builders.sqlite_ddl import create_table, insert_rows, quote
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


def test_insert_rows_quotes_metadata_names():
    """Table and column names with apostrophes and double quotes are quoted on insert and in keys."""
    con = sqlite3.connect(":memory:")
    t, c = 'o\'brien "t"', 'it\'s "col"'
    create_table(con, t, [(c, "VARCHAR(20)", False), ("k", "INTEGER", True)], unique=[[c]])
    insert_rows(con, t, [c, "k"], [("a'b\"c", 1), ("x", 2)])
    assert con.execute(f"SELECT {quote(c)} FROM {quote(t)} ORDER BY k").fetchall() == [("a'b\"c",), ("x",)]
    with pytest.raises(sqlite3.IntegrityError):
        insert_rows(con, t, [c, "k"], [("x", 3)])


def test_create_table_refuses_unknown_types():
    """Column types come from a fixed list, so a type cannot carry SQL."""
    con = sqlite3.connect(":memory:")
    with pytest.raises(ValueError, match="column type"):
        create_table(con, "t", [("a", "TEXT); DROP TABLE t; --", False)])


def test_opener_refuses_cross_origin_redirects():
    """A redirect to another host, port or scheme is refused, so credentials never follow it."""
    import urllib.error
    import urllib.request

    from relsynth import _http
    h = _http._SameOriginRedirect()
    req = urllib.request.Request("https://vault.example:8200/v1/x", headers={"X-Vault-Token": "t"})
    for target in ("http://vault.example:8200/v1/y", "https://evil.example/v1/y", "https://vault.example:9000/y"):
        with pytest.raises(urllib.error.HTTPError, match="another origin"):
            h.redirect_request(req, None, 302, "Found", {}, target)
    same = h.redirect_request(req, None, 302, "Found", {}, "https://VAULT.example:8200/v1/y")
    assert same.full_url.endswith("/v1/y")
    assert _http._origin("https://a.example/x") == _http._origin("https://a.example:443/y")


def test_opener_errors_never_carry_credentials():
    """Refusals name only scheme, host and port: no user, password, path or query."""
    import urllib.error
    import urllib.request

    from relsynth import _http
    with pytest.raises(ValueError) as e:
        _http.urlopen("ftp://user:hunter2@files.example/x?token=abc", timeout=1)
    assert "hunter2" not in str(e.value) and "token" not in str(e.value)
    req = urllib.request.Request("https://u:pw1@vault.example/v1/x?t=s3")
    with pytest.raises(urllib.error.HTTPError) as e:
        _http._SameOriginRedirect().redirect_request(req, None, 302, "Found", {},
                                                     "https://u:pw2@evil.example/y?token=s4")
    text = f"{e.value} {e.value.filename} {e.value.msg}"
    assert not any(s in text for s in ("pw1", "pw2", "s3", "s4", "/v1/x")), text
    assert "evil.example" in text
