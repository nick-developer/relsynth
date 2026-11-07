# SPDX-License-Identifier: Apache-2.0
"""Open http:// and https:// URLs only.

``urllib.request.urlopen`` also opens ``file://``, ``ftp://`` and ``data:`` URLs,
so a URL that comes from a spec, a setting or the command line could read local
files.  ``urlopen`` here refuses any other scheme before connecting, and its
opener has no handler for them either (a redirect cannot reach one).  Proxies
come from the environment as with ``urllib.request.urlopen``.
"""
from __future__ import annotations

import urllib.request
from urllib.parse import urlsplit

SCHEMES = ("http", "https")

_OPENER = urllib.request.OpenerDirector()
for _handler in (urllib.request.ProxyHandler(), urllib.request.UnknownHandler(), urllib.request.HTTPHandler(),
                 urllib.request.HTTPSHandler(), urllib.request.HTTPDefaultErrorHandler(),
                 urllib.request.HTTPRedirectHandler(), urllib.request.HTTPErrorProcessor()):
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
