# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the code scanning fixes: http(s)-only URL opening and
quoted DDL in the benchmark builders."""
from __future__ import annotations

import re

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
    """A request with credentials is not redirected to another host, port or scheme."""
    import urllib.error
    import urllib.request

    from relsynth import _http
    h = _http._SameOriginRedirect()
    req = urllib.request.Request("https://vault.example:8200/v1/x", headers={"X-Vault-Token": "t"})
    for target in ("http://vault.example:8200/v1/y", "https://evil.example/v1/y",
                   "https://vault.example:9000/y", "ftp://vault.example/v1/y"):
        with pytest.raises(urllib.error.HTTPError, match="refused"):
            h.redirect_request(req, None, 302, "Found", {}, target)
    same = h.redirect_request(req, None, 302, "Found", {}, "https://VAULT.example:8200/v1/y")
    assert same.full_url.endswith("/v1/y")
    assert _http._origin("https://a.example/x") == _http._origin("https://a.example:443/y")


def test_plain_downloads_follow_cdn_redirects_but_never_downgrade():
    """A request without credentials may move to another https host, never to http;
    an API key or a password in the URL counts as credentials."""
    import urllib.error
    import urllib.request

    from relsynth import _http
    h = _http._SameOriginRedirect()
    plain = urllib.request.Request("https://github.com/o/r/releases/download/v1/x.zip")
    moved = h.redirect_request(plain, None, 302, "Found", {}, "https://release-assets.githubusercontent.com/x")
    assert moved.full_url.startswith("https://release-assets.githubusercontent.com/")
    with pytest.raises(urllib.error.HTTPError, match="from https"):
        h.redirect_request(plain, None, 302, "Found", {}, "http://release-assets.githubusercontent.com/x")
    for req in (urllib.request.Request("https://api.example/v1", headers={"Authorization": "Bearer k"}),
                urllib.request.Request("https://api.example/v1", headers={"X-Api-Key": "k"}),
                urllib.request.Request("https://u:p@api.example/v1")):
        with pytest.raises(urllib.error.HTTPError, match="credentials"):
            h.redirect_request(req, None, 302, "Found", {}, "https://other.example/v1")


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


def test_refused_redirect_error_keeps_no_target_or_body():
    """A refused redirect's HTTPError drops Location, cookies and the body (which may repeat the target)."""
    import email.message
    import io
    import urllib.error
    import urllib.request

    from relsynth import _http
    headers = email.message.Message()
    headers["Location"] = "https://u:pw2@evil.example/y?token=s4"
    headers["Set-Cookie"] = "sid=s5"
    headers["Content-Type"] = "text/html"
    body = io.BytesIO(b"<a href='https://evil.example/y?token=s4'>moved</a>")
    req = urllib.request.Request("https://vault.example/v1/x", headers={"X-Vault-Token": "t0"})
    with pytest.raises(urllib.error.HTTPError) as e:
        _http._SameOriginRedirect().redirect_request(req, body, 302, "Found", headers, headers["Location"])
    err = e.value
    assert err.headers.get("Location") is None and err.headers.get("Set-Cookie") is None
    assert err.headers.get("Content-Type") == "text/html"
    assert err.read() == b"" and body.closed
    text = f"{err} {err.filename} {err.msg} {dict(err.headers.items())}"
    assert not any(s in text for s in ("pw2", "s4", "s5", "t0", "/v1/x")), text


def _e2e_flow():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "deploy" / "e2e" / "e2e_flow.py"
    spec = importlib.util.spec_from_file_location("e2e_flow_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_e2e_flow_refuses_every_redirect():
    """The e2e script's requests all carry a bearer token or password: no redirect is followed,
    neither to another host nor to http, and the error names no URL, token or body."""
    import io
    import urllib.error
    import urllib.request

    mod = _e2e_flow()
    h = mod._NoRedirect()
    req = urllib.request.Request("https://api.example/api/v1/t/a/runs?x=s3",
                                 headers={"Authorization": "Bearer tok1"})
    for target in ("https://evil.example/y?token=s4", "http://api.example/api/v1/t/a/runs",
                   "https://api.example/elsewhere"):
        body = io.BytesIO(b"moved to " + target.encode())
        with pytest.raises(urllib.error.HTTPError, match="redirect refused") as e:
            h.redirect_request(req, body, 302, "Found", {"Location": target}, target)
        err = e.value
        text = f"{err} {err.filename} {err.msg} {dict(err.headers.items())} {err.read()!r}"
        assert not any(s in text for s in ("tok1", "s3", "s4", "evil", "/api/v1")), text
        assert body.closed
    assert any(isinstance(x, mod._NoRedirect) for x in mod._OPENER.handlers)
    assert not any(type(x) is urllib.request.HTTPRedirectHandler for x in mod._OPENER.handlers)
    with pytest.raises(ValueError) as e:
        mod._open(urllib.request.Request("file:///etc/passwd"), timeout=1)
    assert "passwd" not in str(e.value)


def test_semgrep_summary_paths_cannot_issue_workflow_commands():
    """A file name with ::, ##[ or a line break prints as text in the Semgrep summary."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).parents[2] / ".github" / "scripts" / "sarif_clean.py"
    spec = importlib.util.spec_from_file_location("sarif_clean", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("src/##[error]review-required.py", "src/x::error file=a::b.py", "a\n::error::x.py", "a`b.py"):
        out = mod._summary_safe(name)
        assert "::" not in out and "##[" not in out and "\n" not in out and "`" not in out, out
    assert mod._summary_safe("src/relsynth/app.py") == "src/relsynth/app.py"
    # a pipe in a path cannot split the summary row into extra cells
    safe = mod._summary_safe("src/a|[review](https://evil.example)|b.py")
    row = f"| `rule` | `{safe}` | 7 |"
    assert len(re.findall(r"(?<!\\)\|", row)) == 4, row
    assert row.endswith("| 7 |")


def test_legacy_ident_check_quotes_database_metadata():
    """The v09 identification check quotes table and column names from the database:
    apostrophes and embedded double quotes neither break nor escape the SQL."""
    import importlib.util
    from pathlib import Path

    import pandas as pd
    path = Path(__file__).resolve().parents[2] / "legacy" / "checks" / "v09_ident.py"
    spec = importlib.util.spec_from_file_location("v09_ident_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    con = sqlite3.connect(":memory:")
    parent, child, key = "par'\" x", "kid'\" x", "k'\"y"
    con.execute(f"create table {mod._q(parent)} ({mod._q(key)} integer primary key)")
    con.execute(f"create table {mod._q(child)} ({mod._q(key)} integer references {mod._q(parent)}({mod._q(key)}))")
    con.executemany(f"insert into {mod._q(parent)} values (?)", [(1,), (2,)])
    con.executemany(f"insert into {mod._q(child)} values (?)", [(1,), (2,), (2,)])
    (t, ) = [r[0] for r in con.execute("select name from sqlite_master where name like 'kid%'")]
    row = list(con.execute("select * from pragma_foreign_key_list(?)", (t,)))[0]
    p, c, pc = row[2], row[3], row[4]
    assert (p, c, pc) == (parent, key, key)
    assert sorted(pd.read_sql(f"select {mod._q(c)} from {mod._q(t)}", con)[c]) == [1, 2, 2]
    assert mod._q('a"b') == '"a""b"'


@pytest.mark.parametrize("target", ["file://user:pw9@host/etc/passwd?token=s9", "data:text/plain,s9",
                                    "ftp://user:pw9@files.example/x?token=s9"])
def test_redirect_to_another_scheme_is_refused_before_urllib_echoes_it(target):
    """urllib's own check for a non-http(s) target raises an HTTPError holding the target URL,
    its headers and the body; both openers refuse it first, through http_error_302 itself."""
    import email.message
    import io
    import urllib.error
    import urllib.request

    from relsynth import _http
    for handler in (_http._SameOriginRedirect(), _e2e_flow()._NoRedirect()):
        for code in (301, 302, 303, 307):
            headers = email.message.Message()
            headers["Location"] = target
            body = io.BytesIO(b"see " + target.encode())
            req = urllib.request.Request("https://api.example/v1/x?t=s3", headers={"Authorization": "Bearer tok1"})
            with pytest.raises(urllib.error.HTTPError) as e:
                getattr(handler, f"http_error_{code}")(req, body, code, "Found", headers)
            err = e.value
            text = f"{err} {err.filename} {err.msg} {dict(err.headers.items())} {err.read()!r}"
            assert not any(s in text for s in ("pw9", "s9", "passwd", "tok1", "s3", "/v1/x")), text
            assert err.__context__ is None and err.__cause__ is None
            assert body.closed


def test_shared_opener_sanitises_urllib_refusals_after_the_origin_check():
    """A redirected POST (refused by urllib itself) and a redirect loop give errors without the
    target, the request's path or query, or the response body."""
    import email.message
    import io
    import urllib.error
    import urllib.request

    from relsynth import _http
    h = _http._SameOriginRedirect()
    headers = email.message.Message()
    headers["Location"] = "https://api.example/next?token=s9"
    post = urllib.request.Request("https://api.example/v1/x?t=s3", data=b"{}")
    with pytest.raises(urllib.error.HTTPError, match="POST") as e:
        h.http_error_307(post, io.BytesIO(b"s9"), 307, "Temporary Redirect", headers)   # 307 keeps POST: urllib refuses
    err = e.value
    text = f"{err} {err.filename} {err.msg} {dict(err.headers.items())} {err.read()!r}"
    assert not any(s in text for s in ("s9", "s3", "/v1/x")), text
    assert err.__context__ is None and err.__cause__ is None      # no chained urllib error with the target

    loop = urllib.request.Request("https://api.example/v1/x?t=s3")
    loop.redirect_dict = {"https://api.example/next?token=s9": h.max_repeats}
    with pytest.raises(urllib.error.HTTPError, match="loop") as e:
        h.http_error_302(loop, io.BytesIO(b"s9"), 302, "Found", headers)
    err = e.value
    text = f"{err} {err.filename} {err.msg} {dict(err.headers.items())} {err.read()!r}"
    assert not any(s in text for s in ("s9", "s3", "/v1/x")), text
    assert err.__context__ is None and err.__cause__ is None


def test_legacy_retail_measures_refuse_csv_links_outside_the_folder(tmp_path):
    """meas_retail.read_folder reads the folder's CSV files but refuses one that is a
    symbolic link to a file elsewhere."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "legacy" / "fakedata" / "meas_retail.py"
    spec = importlib.util.spec_from_file_location("meas_retail_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    folder = tmp_path / "out"
    folder.mkdir()
    (folder / "Store.csv").write_text("StoreKey\n1\n")
    assert list(mod.read_folder(folder)) == ["Store"]
    secret = tmp_path / "secret.csv"
    secret.write_text("password\nhunter2\n")
    try:
        (folder / "Orders.csv").symlink_to(secret)
    except OSError as e:
        pytest.skip(f"cannot create a symlink here: {e}")
    with pytest.raises(SystemExit, match="outside"):
        mod.read_folder(folder)
