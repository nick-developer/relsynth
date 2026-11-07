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


class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    """Never downgrade https to http; keep requests with credentials on their origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """The redirected request, or HTTPError when the redirect could leak the request's credentials."""
        old, new = _origin(req.full_url), _origin(newurl)
        if old[0] == "https" and new[0] != "https":
            raise urllib.error.HTTPError(_where(req.full_url), code,
                                         f"redirect from https to {_where(newurl)} refused", headers, fp)
        if new != old and _has_credentials(req):
            raise urllib.error.HTTPError(_where(req.full_url), code,
                                         f"redirect with credentials to another origin refused: {_where(newurl)}",
                                         headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


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
