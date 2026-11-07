# SPDX-License-Identifier: Apache-2.0
"""Open http:// and https:// URLs only.

``urllib.request.urlopen`` also opens ``file://``, ``ftp://`` and ``data:`` URLs,
so a URL that comes from a spec, a setting or the command line could read local
files.  ``urlopen`` here refuses any other scheme before connecting, and its
opener has no handler for them either (a redirect cannot reach one).  Redirects
are followed only to the same scheme, host and port: callers send credentials
(``Authorization``, ``X-Vault-Token``) that urllib would otherwise forward to
another origin or downgrade from https to http.  Proxies come from the
environment as with ``urllib.request.urlopen``.
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
    return scheme, (u.hostname or "").lower(), u.port or {"http": 80, "https": 443}.get(scheme)


class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only within the origin of the request."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """The redirected request, or HTTPError when `newurl` is another origin."""
        if _origin(newurl) != _origin(req.full_url):
            raise urllib.error.HTTPError(req.full_url, code, f"redirect to another origin refused: {newurl[:200]}",
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
        raise ValueError(f"only http(s) URLs can be opened, not {scheme or 'a URL without a scheme'}: "
                         f"{full[:200]}")
    return _OPENER.open(url, timeout=timeout)
