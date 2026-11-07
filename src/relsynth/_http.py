# SPDX-License-Identifier: Apache-2.0
"""Open http:// and https:// URLs only.

``urllib.request.urlopen`` also opens ``file://``, ``ftp://`` and ``data:`` URLs,
so a URL that comes from a spec, a setting or the command line could read local
files.  ``urlopen`` here refuses any other scheme before connecting, and its
opener has no handler for them either (a redirect cannot reach one).  urllib
forwards every request header on a redirect, so a request that carries
credentials (``Authorization``, ``X-Vault-Token``, cookies, an API key) follows
redirects only within its scheme, host and port; a plain download may follow
one to another host (release files are served from a CDN).  No redirect may go
from https to http.  Proxies come from the environment as with
``urllib.request.urlopen``.
"""
from __future__ import annotations

import email.message
import io
import urllib.error
import urllib.request
from urllib.parse import urlsplit

SCHEMES = ("http", "https")


def _origin(url: str) -> tuple:
    """(scheme, host, port) of `url`, with the scheme's default port filled in."""
    u = urlsplit(url)
    scheme = u.scheme.lower()
    try:
        port = u.port
    except ValueError:                  # a malformed port matches no origin
        port = -1
    return scheme, (u.hostname or "").lower(), port or {"http": 80, "https": 443}.get(scheme)


def _where(url: str) -> str:
    """scheme://host[:port] of `url`, for messages: never its user, password, path or query."""
    scheme, host, port = _origin(url)
    return f"{scheme}://{host}:{port}" if host else (f"a {scheme} URL" if scheme else "a URL without a scheme")


# headers that carry credentials (compared in lower case)
CREDENTIAL_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie", "x-vault-token", "x-api-key",
                                "api-key"})


def _has_credentials(req: urllib.request.Request) -> bool:
    """True when the request sends a credential header (redirected or not)."""
    names = {k.lower() for k in (*req.headers, *req.unredirected_hdrs)}
    return bool(names & CREDENTIAL_HEADERS) or "@" in urlsplit(req.full_url).netloc


# response headers left out of a refused redirect's HTTPError: the target URL (which may carry
# credentials or tokens in its user, path or query) and cookies the response tried to set
_DROPPED_HEADERS = frozenset({"location", "uri", "content-location", "set-cookie", "set-cookie2"})


def _refused(req, code, msg, headers, fp) -> urllib.error.HTTPError:
    """An HTTPError for a refused redirect that names only the origin: its headers keep
    no target URL or cookie, and its body (which may repeat the target) is dropped."""
    kept = email.message.Message()
    for name, value in (headers.items() if headers is not None else ()):
        if name.lower() not in _DROPPED_HEADERS:
            kept[name] = value
    if fp is not None:
        fp.close()
    return urllib.error.HTTPError(_where(req.full_url), code, msg, kept, io.BytesIO(b""))


class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    """Never downgrade https to http; keep requests with credentials on their origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """The redirected request, or HTTPError when the redirect could leak the request's credentials."""
        old, new = _origin(req.full_url), _origin(newurl)
        if old[0] == "https" and new[0] != "https":
            raise _refused(req, code, f"redirect from https to {_where(newurl)} refused", headers, fp)
        if new != old and _has_credentials(req):
            raise _refused(req, code, f"redirect with credentials to another origin refused: {_where(newurl)}",
                           headers, fp)
        try:
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        except urllib.error.HTTPError:  # urllib refuses e.g. a POST: its error keeps the headers
            raise _refused(req, code, f"redirect of a {req.get_method()} request refused", headers, fp) from None

    def http_error_302(self, req, fp, code, msg, headers):
        """Follow a redirect as urllib does, but refuse a target that is not http(s) here, before
        urllib's own check (whose HTTPError repeats the target URL, headers and body), and
        strip urllib's redirect-loop error the same way."""
        target = headers.get("location") or headers.get("uri") if headers is not None else None
        if target is not None:
            scheme = urlsplit(target).scheme.lower()
            if scheme not in ("", *SCHEMES):
                raise _refused(req, code, f"redirect to {_where(target)} refused: only http(s)", headers, fp)
        try:
            return super().http_error_302(req, fp, code, msg, headers)
        except urllib.error.HTTPError as e:
            if e.filename == req.full_url and str(e.msg).startswith(self.inf_msg):
                raise _refused(req, code, "redirect loop refused", headers, fp) from None
            raise

    http_error_301 = http_error_303 = http_error_307 = http_error_302
    if hasattr(urllib.request.HTTPRedirectHandler, "http_error_308"):    # Python 3.11+
        http_error_308 = http_error_302


_OPENER = urllib.request.OpenerDirector()
for _handler in (urllib.request.ProxyHandler(), urllib.request.UnknownHandler(), urllib.request.HTTPHandler(),
                 urllib.request.HTTPSHandler(), urllib.request.HTTPDefaultErrorHandler(),
                 _SameOriginRedirect(), urllib.request.HTTPErrorProcessor()):
    _OPENER.add_handler(_handler)


def urlopen(url: str | urllib.request.Request, *, timeout: float):
    """Open `url` (a string or a Request) like ``urllib.request.urlopen``; any
    scheme other than http or https raises ValueError."""
    full = url.full_url if isinstance(url, urllib.request.Request) else str(url)
    scheme = urlsplit(full).scheme.lower()
    if scheme not in SCHEMES:
        raise ValueError(f"only http(s) URLs can be opened, not {scheme or 'a URL without a scheme'}")
    return _OPENER.open(url, timeout=timeout)
