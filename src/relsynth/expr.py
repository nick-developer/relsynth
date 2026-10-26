# SPDX-License-Identifier: Apache-2.0
"""Safe, vectorised expression evaluator for business rules.

Rules are written in a small Python-like language and evaluated over the
columns of one table:

    total = quantity * unit_price - discount            (derive)
    discount <= quantity * unit_price                   (check)
    FilingStatus = where(ReportedCellCount > 0, 'Filed', 'Declared not filed')
    ModuleKey.ApplicableFromDateKey <= ReferenceDateKey  (check through a foreign key)

Supported: numbers, strings, True/False/None; + - * / // % **; comparisons
(also `in [..]`, `not in [..]`); and / or / not (also & | ~); dotted names
`fk.column` (and longer chains) that follow a foreign key to a parent column;
functions where, isnull, notnull, coalesce, abs, round, floor, ceil, sqrt,
log, exp, minimum, maximum, between, days (a date as a day number, whatever
its storage) and add_days (a date shifted by a number of days).

Nulls follow SQL semantics: arithmetic with a null gives null, a comparison
with a null is unknown, `and`/`or` use Kleene logic, and a check whose value is
unknown counts as satisfied.  round() and integer derives round half away from
zero, as SQL does (round(2.5) = 3, round(-2.5) = -3).  `&` and `|` bind tighter
than comparisons, so `x > 0 & y > 0` is refused: write (x > 0) & (y > 0) or use
and / or.  Text compared with a number (or a date) must be a number (or an ISO
date); the text 'true' / 'false' (any case) is a boolean where one is expected.
Nothing is passed to Python's eval: the expression is parsed with `ast` and
only the node types above are allowed; expressions are limited in length and
nesting, and regular expressions in length and in nested repetitions (which
take exponential time).
"""
from __future__ import annotations

import ast
import re
import warnings

import numpy as np
import pandas as pd


class ExprError(ValueError):
    pass


_NAME_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")
MAX_EXPR_LEN = 10000
MAX_DEPTH = 100
MAX_REGEX_LEN = 500
_REGEX_FUNCS = ("regex_match", "regex_replace")


def parse(expr: str) -> ast.Expression:
    """The syntax tree of an expression, after the checks that do not need data:
    length, nesting depth, '&' / '|' against comparisons, constant regexes."""
    expr = str(expr).strip()
    if len(expr) > MAX_EXPR_LEN:
        raise ExprError(f"expression longer than {MAX_EXPR_LEN} characters")
    try:
        with warnings.catch_warnings():     # '\.' in a regex literal of a rule is meant as written
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"cannot parse {expr[:200]!r}: {e.msg}") from None
    except (RecursionError, MemoryError):
        raise ExprError(f"expression nested too deeply (more than {MAX_DEPTH} levels)") from None
    stack = [(tree.body, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_DEPTH:
            raise ExprError(f"expression nested too deeply (more than {MAX_DEPTH} levels)")
        if isinstance(node, ast.Compare) and any(isinstance(x, ast.BinOp) and isinstance(x.op, (ast.BitAnd, ast.BitOr))
                                                 for x in [node.left, *node.comparators]):
            raise ExprError(f"'&' and '|' bind tighter than comparisons in {expr[:200]!r}: write "
                            "(a > 0) & (b > 0), or use and / or")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _REGEX_FUNCS \
                and len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
            check_regex(node.args[1].value)
        stack.extend((ch, depth + 1) for ch in ast.iter_child_nodes(node))
    return tree


def check_regex(pattern: str):
    """Refuse patterns that can take exponential time (catastrophic backtracking):
    a repetition inside a repetition, (a+)+ or (a*)*, or a repeated choice whose
    alternatives can start alike, (a|a)* or (a|ab)+; and patterns over MAX_REGEX_LEN."""
    pattern = str(pattern)
    if len(pattern) > MAX_REGEX_LEN:
        raise ExprError(f"regular expression longer than {MAX_REGEX_LEN} characters")
    try:
        import re._parser as sp
    except ImportError:                           # Python < 3.11
        import sre_parse as sp  # noqa: F401
    try:
        tree = sp.parse(pattern)
    except (re.error, RecursionError, OverflowError) as e:
        raise ExprError(f"invalid regular expression {pattern!r}: {e}") from None
    try:
        import re._constants as C
    except ImportError:                           # Python < 3.11
        import sre_constants as C
    REPEATS = (C.MAX_REPEAT, C.MIN_REPEAT) + ((C.POSSESSIVE_REPEAT,) if hasattr(C, "POSSESSIVE_REPEAT") else ())

    def items(sub):
        return list(sub.data if hasattr(sub, "data") else sub)

    def first(seq):
        """Characters an item sequence can start with: a set, or None for 'any'."""
        for op, av in items(seq):
            if op == C.LITERAL:
                return {chr(av)}
            if op == C.IN:
                out = set()
                for o2, a2 in av:
                    if o2 == C.LITERAL:
                        out.add(chr(a2))
                    elif o2 == C.RANGE and a2[1] - a2[0] < 512:
                        out |= {chr(c) for c in range(a2[0], a2[1] + 1)}
                    else:
                        return None
                return out
            if op == C.SUBPATTERN:
                return first(av[-1])
            if op in (C.AT,):
                continue
            return None
        return set()

    def branches(seq):
        """Alternatives under a repetition must start differently (the parser has
        already factored a common prefix: (a|ab) is a(|b), so an empty one counts)."""
        for op, av in items(seq):
            if op == C.SUBPATTERN:
                branches(av[-1])
            elif op == C.BRANCH:
                fs = [first(b) if items(b) else None for b in av[1]]
                for i in range(len(fs)):
                    for j in range(i + 1, len(fs)):
                        if fs[i] is None or fs[j] is None or fs[i] & fs[j]:
                            raise ExprError(f"regular expression {pattern!r} repeats alternatives that can match "
                                            "the same text (like (a|ab)*), which can take exponential time")

    def walk(seq, in_repeat):
        for op, av in items(seq):
            if op in REPEATS:
                lo, hi, sub = av
                unbounded = hi == C.MAXREPEAT or hi > 1
                if in_repeat and unbounded:
                    raise ExprError(f"regular expression {pattern!r} repeats a repetition (like (a+)+), which can "
                                    "take exponential time; write it without the nested repetition")
                if unbounded:
                    branches(sub)
                walk(sub, in_repeat or unbounded)
            elif op == C.SUBPATTERN:
                walk(av[-1], in_repeat)
            elif op == C.BRANCH:
                for b in av[1]:
                    walk(b, in_repeat)
            elif op in (C.ASSERT, C.ASSERT_NOT):
                walk(av[1], in_repeat)
    try:
        walk(tree, False)
    except RecursionError:
        raise ExprError(f"regular expression {pattern[:100]!r} nested too deeply") from None
    return pattern


def split_assignment(expr: str):
    """'col = rhs' -> ('col', 'rhs').  '==' is not an assignment."""
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)(.*)$", expr, flags=re.S)
    if not m:
        raise ExprError(f"derive rule needs 'column = expression': {expr!r}")
    return m.group(1), m.group(2).strip()


def names_in(expr: str) -> set[str]:
    """Column names and dotted paths (fk.column) an expression reads."""
    out = set()

    def visit(node):
        if isinstance(node, ast.Attribute):
            p = _path(node)
            if p:
                out.add(p)
                return
        if isinstance(node, ast.Call):
            for a in node.args:
                visit(a)
            return
        if isinstance(node, ast.Name):
            if node.id not in ("True", "False", "None"):
                out.add(node.id)
            return
        for ch in ast.iter_child_nodes(node):
            visit(ch)
    tree = parse(expr)
    try:
        visit(tree)
    except (RecursionError, MemoryError):
        raise ExprError("expression nested too deeply") from None
    return out


def _path(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


# ---------------------------------------------------------------- value model
# Every intermediate value is a numpy array of length n:
#   * float arrays (nan = null) for numbers, booleans and dates
#   * object arrays (None = null) for strings and anything else
def to_array(s) -> np.ndarray:
    if isinstance(s, np.ndarray) and s.dtype != object:
        return s.astype(float) if s.dtype.kind in "biu" else s
    s = pd.Series(s) if not isinstance(s, pd.Series) else s
    if pd.api.types.is_datetime64_any_dtype(s):
        v = s.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
        v[s.isna().to_numpy()] = np.nan
        return v / 86400e9                         # days since 1970-01-01
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float).to_numpy()
    obj = s.to_numpy(dtype=object)
    nulls = np.asarray(pd.isna(obj), dtype=bool)
    kind = pd.api.types.infer_dtype(obj[~nulls], skipna=False) if (~nulls).any() else "empty"
    if kind in ("boolean", "integer", "floating", "mixed-integer-float"):
        out = np.full(len(obj), np.nan)
        out[~nulls] = obj[~nulls].astype(float)
        return out
    out = obj.copy()
    out[nulls] = None
    return out


def _isnull(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    try:
        return bool(isinstance(v, float) and np.isnan(v))
    except TypeError:
        return False


def _null_mask(a):
    if a.dtype == object:
        return np.asarray(pd.isna(a), dtype=bool)
    return np.isnan(a)


def _as_float(a):
    if a.dtype != object:
        return a
    out = np.full(len(a), np.nan)
    for i, v in enumerate(a):
        if _isnull(v):
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            raise ExprError(f"cannot use {v!r} as a number") from None
    return out


def _bool3(a):
    """Three-valued truth: 1.0 / 0.0 / nan.  The text 'true' / 'false' (any case)
    is a boolean; other text has no truth value."""
    if a.dtype == object:
        return np.array([np.nan if _isnull(v) else _truth(v) for v in a])
    return np.where(np.isnan(a), np.nan, (a != 0).astype(float))


def _truth(v):
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "false"):
            return float(t == "true")
        raise ExprError(f"text {v!r} used as a condition: only 'true' and 'false' are booleans")
    return float(bool(v))


def _broadcast(v, n):
    if isinstance(v, np.ndarray):
        return v
    if v is None:
        return np.full(n, np.nan)
    if isinstance(v, (bool, int, float, np.number)):
        return np.full(n, float(v))
    return np.array([v] * n, dtype=object)


_OPS = {ast.Eq: np.equal, ast.NotEq: np.not_equal, ast.Lt: np.less, ast.LtE: np.less_equal,
        ast.Gt: np.greater, ast.GtE: np.greater_equal}


def _text_as_number(obj, what="compare"):
    """Text compared with a number or a date (day numbers here): numbers when every
    value is a number, else day numbers when every value is a date; else an error."""
    m = ~_null_mask(obj)
    vals = obj[m]
    if not len(vals):
        return np.full(len(obj), np.nan)
    tf = {"true": 1.0, "false": 0.0}
    vals = np.array([tf.get(v.strip().lower(), v) if isinstance(v, str) else v for v in vals], dtype=object)
    f = pd.to_numeric(pd.Series(vals, dtype=object), errors="coerce").to_numpy(dtype=float)
    if np.isnan(f).any():
        try:
            f = _days(vals)
        except ExprError:
            f = np.full(len(vals), np.nan)
        if np.isnan(f).any():
            bad = vals[np.isnan(f)][0]
            raise ExprError(f"cannot {what} the text {bad!r} with a number or a date: the text must be a number "
                            "or an ISO date ('2021-01-31')")
    out = np.full(len(obj), np.nan)
    out[m] = f
    return out


def _compare(op, a, b):
    if (a.dtype == object) != (b.dtype == object) and not np.isnan(a if b.dtype == object else b).all():
        # text against a number or a date: the text is read as one
        if a.dtype == object:
            a = _text_as_number(a)
        else:
            b = _text_as_number(b)
    na, nb = _null_mask(a), _null_mask(b)
    if a.dtype != object and b.dtype != object:
        with np.errstate(invalid="ignore"):
            r = {ast.Eq: np.equal, ast.NotEq: np.not_equal, ast.Lt: np.less, ast.LtE: np.less_equal,
                 ast.Gt: np.greater, ast.GtE: np.greater_equal}[type(op)](a, b).astype(float)
    elif _all_text(a, na) and _all_text(b, nb):
        # text with text: one vectorised comparison over the non-null rows
        r = np.zeros(len(a))
        ok = ~(na | nb)
        if ok.any():
            r[ok] = _OPS[type(op)](a[ok], b[ok]).astype(float)
    else:
        f = {ast.Eq: lambda x, y: x == y, ast.NotEq: lambda x, y: x != y, ast.Lt: lambda x, y: x < y,
             ast.LtE: lambda x, y: x <= y, ast.Gt: lambda x, y: x > y, ast.GtE: lambda x, y: x >= y}[type(op)]
        r = np.zeros(len(a))
        for i in range(len(a)):
            if na[i] or nb[i]:
                continue
            x, y = a[i], b[i]
            if isinstance(x, (bool, np.bool_)) or isinstance(y, (bool, np.bool_)):
                x, y = _maybe_bool(x), _maybe_bool(y)
            try:
                r[i] = float(f(x, y))
            except TypeError:
                r[i] = float(f(str(x), str(y)))
    r[na | nb] = np.nan
    return r


def _all_text(a, nulls):
    return a.dtype == object and pd.api.types.infer_dtype(a[~nulls], skipna=False) in ("string", "empty")


def _maybe_bool(v):
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return v


def _and(a, b):
    out = np.where((a == 0) | (b == 0), 0.0, np.where(np.isnan(a) | np.isnan(b), np.nan, 1.0))
    return out


def _or(a, b):
    return np.where((a == 1) | (b == 1), 1.0, np.where(np.isnan(a) | np.isnan(b), np.nan, 0.0))


def _where(c, a, b):
    c = _bool3(c)
    pick = c == 1                                   # unknown -> else branch (SQL CASE)
    if a.dtype != object and b.dtype != object:
        return np.where(pick, a, b)
    out = np.where(pick, a.astype(object), b.astype(object))
    out[np.asarray(pd.isna(out), dtype=bool)] = None
    return out


def _coalesce(*xs):
    out = xs[0].copy() if xs[0].dtype == object else xs[0].astype(object)
    for x in xs[1:]:
        m = _null_mask(out)
        out[m] = x[m] if x.dtype == object else x.astype(object)[m]
    return to_array(pd.Series(out))


def _const_arg(a, what):
    """A function argument that must be one constant (e.g. a regex pattern)."""
    vals = {v for v in (a.tolist() if a.dtype == object else a.tolist()) if not _isnull(v)}
    if len(vals) != 1:
        raise ExprError(f"{what} must be a single constant")
    return vals.pop()


def _str1(f):
    """Apply a string function elementwise; nulls stay null."""
    def g(x, *args):
        xs = x if x.dtype == object else np.array([None if np.isnan(v) else _fmt(v) for v in x], dtype=object)
        return np.array([None if _isnull(v) else f(str(v), *args) for v in xs], dtype=object)
    return g


def _fmt(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def _regex_replace(x, pattern, repl):
    rx = re.compile(check_regex(_const_arg(pattern, "regex_replace pattern")))
    r = str(_const_arg(repl, "regex_replace replacement")) if not _null_mask(repl).all() else ""
    return _str1(lambda s: rx.sub(r, s))(x)


def _regex_match(x, pattern):
    rx = re.compile(check_regex(_const_arg(pattern, "regex_match pattern")))
    nm = _null_mask(x)
    xs = _str1(lambda s: s)(x)
    return np.array([np.nan if nm[i] else float(rx.search(v) is not None) for i, v in enumerate(xs)])


def _substr(x, start, length=None):
    st = int(_const_arg(start, "substr start"))
    ln = None if length is None else int(_const_arg(length, "substr length"))
    return _str1(lambda s: s[st:] if ln is None else s[st:st + ln])(x)


def _concat(*xs):
    parts = [_str1(lambda s: s)(x) for x in xs]
    return np.array([None if any(_isnull(p[i]) for p in parts) else "".join(p[i] for p in parts)
                     for i in range(len(parts[0]))], dtype=object)


def round_half_away(x, d=0):
    """Round half away from zero (SQL ROUND): 2.5 -> 3, -2.5 -> -3, 0.125 -> 0.13 at
    d=2.  A value within float error of a half (2.675 is 2.67499.. in binary) counts
    as the half, as in decimal arithmetic."""
    x = np.asarray(x, dtype=float)
    m = 10.0 ** abs(d)
    y = np.abs(x) * m if d >= 0 else np.abs(x) / m
    with np.errstate(invalid="ignore"):
        fl = np.floor(y)
        tie = np.abs(y - fl - 0.5) <= 1e-9 * np.maximum(1.0, y)
        r = np.copysign(np.where(tie, fl + 1.0, np.round(y)), x)
        out = r / m if d >= 0 else r * m
    return np.where(np.isfinite(x), out, x)


def _num1(f):
    return lambda x: f(_as_float(x))


def _days(x):
    """Day numbers (days since 1970-01-01, with fraction) of dates given as
    datetimes (already day numbers here), ISO strings or Python dates; numbers
    pass through.  Lets rules compare and shift dates whatever their storage."""
    if x.dtype != object:
        return x
    s = pd.Series(x, dtype=object)
    d = pd.to_datetime(s, errors="coerce", format="mixed")
    if d.dt.tz is not None:
        d = d.dt.tz_convert("UTC").dt.tz_localize(None)
    out = ((d - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    bad = np.isnan(out) & ~_null_mask(x)
    if bad.any():
        out[bad] = _as_float(x[bad])
    return out


def _add_days(x, n):
    return _days(x) + _as_float(n)


def _ymd(x):
    d = _days(np.asarray(x, dtype=object) if np.asarray(x).dtype == object else np.asarray(x, dtype=float))
    d = np.broadcast_to(np.asarray(d, dtype=float), np.shape(d))
    ok = ~np.isnan(d)
    dt = pd.to_datetime(np.where(ok, d, 0.0), unit="D")
    return dt.year.to_numpy(), dt.month.to_numpy(), dt.day.to_numpy(), ok


def _years_between(start, end):
    """Whole years from start to end (an age on a reference date): the year
    difference, less one when end's month and day come before start's."""
    n = max(len(np.atleast_1d(start)), len(np.atleast_1d(end)))
    ys, ms, ds, oks = _ymd(np.broadcast_to(np.atleast_1d(start), (n,)))
    ye, me, de, oke = _ymd(np.broadcast_to(np.atleast_1d(end), (n,)))
    v = (ye - ys - ((me < ms) | ((me == ms) & (de < ds)))).astype(float)
    return np.where(oks & oke, v, np.nan)


def _months_between(start, end):
    """Whole months from start to end."""
    n = max(len(np.atleast_1d(start)), len(np.atleast_1d(end)))
    ys, ms, ds, oks = _ymd(np.broadcast_to(np.atleast_1d(start), (n,)))
    ye, me, de, oke = _ymd(np.broadcast_to(np.atleast_1d(end), (n,)))
    v = ((ye - ys) * 12 + me - ms - (de < ds)).astype(float)
    return np.where(oks & oke, v, np.nan)


_FUNCS = {
    "where": _where,
    "isnull": lambda x: _null_mask(x).astype(float),
    "notnull": lambda x: (~_null_mask(x)).astype(float),
    "coalesce": _coalesce,
    "abs": _num1(np.abs),
    "floor": _num1(np.floor),
    "ceil": _num1(np.ceil),
    "sqrt": _num1(np.sqrt),
    "log": _num1(np.log),
    "exp": _num1(np.exp),
    "round": lambda x, d=None: round_half_away(_as_float(x), int(np.nanmax(_as_float(d))) if d is not None else 0),
    "minimum": lambda a, b: np.where(np.isnan(_as_float(a)) | np.isnan(_as_float(b)), np.nan,
                                     np.minimum(_as_float(a), _as_float(b))),
    "maximum": lambda a, b: np.where(np.isnan(_as_float(a)) | np.isnan(_as_float(b)), np.nan,
                                     np.maximum(_as_float(a), _as_float(b))),
    "regex_replace": _regex_replace,
    "regex_match": _regex_match,
    "substr": _substr,
    "concat": _concat,
    "upper": _str1(str.upper),
    "lower": _str1(str.lower),
    "strip": _str1(str.strip),
    "length": lambda x: np.array([np.nan if _isnull(v) else float(len(v)) for v in _str1(lambda s: s)(x)]),
    "days": _days,
    "add_days": _add_days,
    "years_between": _years_between,
    "months_between": _months_between,
    "between": lambda x, lo, hi: _and(_compare(ast.GtE(), x, lo), _compare(ast.LtE(), x, hi)),
}


class Evaluator:
    """Evaluates expressions over one DataFrame.  `resolve(path)` returns the
    values of a dotted path (`fk.parent_column`) aligned to the frame's rows."""

    def __init__(self, df: pd.DataFrame, resolve=None):
        self.df, self.n, self.resolve = df, len(df), resolve
        self._cache = {}

    def column(self, name):
        if name not in self._cache:
            if name in self.df.columns:
                self._cache[name] = to_array(self.df[name])
            elif "." in name and self.resolve is not None:
                self._cache[name] = to_array(pd.Series(self.resolve(name)))
            else:
                raise ExprError(f"unknown column {name!r}")
        return self._cache[name]

    def eval(self, expr: str) -> np.ndarray:
        tree = parse(expr)
        try:
            return _broadcast(self._ev(tree.body), self.n)
        except (RecursionError, MemoryError) as e:
            raise ExprError(f"expression too large to evaluate ({type(e).__name__})") from None

    def _ev(self, node):
        n = self.n
        if isinstance(node, ast.Constant):
            v = node.value
            if v is None or isinstance(v, (bool, int, float, str)):
                return v
            raise ExprError(f"constant {v!r} not allowed")
        if isinstance(node, ast.Name):
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            if node.id == "None":
                return None
            return self.column(node.id)
        if isinstance(node, ast.Attribute):
            p = _path(node)
            if p is None:
                raise ExprError("only column paths like fk.column may use '.'")
            return self.column(p)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._const(e) for e in node.elts]
        if isinstance(node, ast.UnaryOp):
            v = _broadcast(self._ev(node.operand), n)
            if isinstance(node.op, (ast.Not, ast.Invert)):
                b = _bool3(v)
                return np.where(np.isnan(b), np.nan, 1 - b)
            if isinstance(node.op, ast.USub):
                return -_as_float(v)
            if isinstance(node.op, ast.UAdd):
                return _as_float(v)
        if isinstance(node, ast.BoolOp):
            vals = [_bool3(_broadcast(self._ev(v), n)) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = _and(out, v) if isinstance(node.op, ast.And) else _or(out, v)
            return out
        if isinstance(node, ast.BinOp):
            a, b = _broadcast(self._ev(node.left), n), _broadcast(self._ev(node.right), n)
            if isinstance(node.op, ast.BitAnd):
                return _and(_bool3(a), _bool3(b))
            if isinstance(node.op, ast.BitOr):
                return _or(_bool3(a), _bool3(b))
            if isinstance(node.op, ast.Add) and (a.dtype == object or b.dtype == object):
                if all(isinstance(v, str) or _isnull(v) for v in np.concatenate([a, b]) if v is not None):
                    return np.array([None if _isnull(x) or _isnull(y) else f"{x}{y}" for x, y in zip(a, b)],
                                    dtype=object)
            x, y = _as_float(a), _as_float(b)
            with np.errstate(divide="ignore", invalid="ignore"):
                ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
                       ast.FloorDiv: np.floor_divide, ast.Mod: np.mod, ast.Pow: np.power}
                if type(node.op) not in ops:
                    raise ExprError(f"operator {type(node.op).__name__} not allowed")
                r = ops[type(node.op)](x, y)
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                return np.where(np.isfinite(r) | np.isnan(r), r, np.nan)
            return r
        if isinstance(node, ast.Compare):
            left = _broadcast(self._ev(node.left), n)
            out = np.ones(n)
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    vals = self._ev(comp)
                    if not isinstance(vals, list):
                        raise ExprError("'in' needs a list of constants")
                    r = self._isin(left, vals)
                    if isinstance(op, ast.NotIn):
                        r = np.where(np.isnan(r), np.nan, 1 - r)
                    out = _and(out, r)
                    continue
                right = _broadcast(self._ev(comp), n)
                out = _and(out, _compare(op, left, right))
                left = right
            return out
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise ExprError(f"function {ast.unparse(node.func)!r} not allowed; use one of {sorted(_FUNCS)}")
            if node.keywords:
                raise ExprError("keyword arguments are not supported")
            args = [_broadcast(self._ev(a), n) for a in node.args]
            return _FUNCS[node.func.id](*args)
        if isinstance(node, ast.IfExp):
            return _where(_broadcast(self._ev(node.test), n), _broadcast(self._ev(node.body), n),
                          _broadcast(self._ev(node.orelse), n))
        raise ExprError(f"expression element {type(node).__name__} not allowed")

    def _const(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            return -node.operand.value
        raise ExprError("lists may only hold constants")

    def _isin(self, a, vals):
        nm = _null_mask(a)
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if a.dtype == object and vals and len(nums) == len(vals):
            a = _text_as_number(a, "match")       # text codes against numbers: read as numbers
        if a.dtype != object:
            fv = [float(v) for v in vals if isinstance(v, (int, float, bool)) and v is not None]
            txt = [v for v in vals if isinstance(v, str)]
            if txt:                               # numbers or dates against text constants
                fv += list(_text_as_number(np.array(txt, dtype=object), "match"))
            r = np.isin(a, fv).astype(float)
        else:
            try:
                r = pd.Series(a, dtype=object).isin(list(vals)).to_numpy(dtype=float)
            except TypeError:                     # unhashable values: one by one
                sv = set(vals)
                r = np.array([float(v in sv) if not nm[i] else 0.0 for i, v in enumerate(a)])
        r[nm] = np.nan
        return r


def check_mask(df, expr, resolve=None) -> np.ndarray:
    """True where the rule holds or is unknown (nulls)."""
    v = _bool3(Evaluator(df, resolve).eval(expr))
    return ~(v == 0)


def derive_values(df, expr, resolve=None, ctype=None):
    """Values of the right-hand side, cast to the declared column type."""
    lhs, rhs = split_assignment(expr)
    v = Evaluator(df, resolve).eval(rhs)
    return lhs, cast(v, ctype)


def cast(v, ctype):
    if ctype in ("integer", "id") and v.dtype != object:
        return pd.array(np.where(np.isnan(v), np.nan, round_half_away(v)), dtype="Float64").astype("Int64")
    if ctype == "boolean":
        b = _bool3(v)
        return pd.array([None if np.isnan(x) else bool(x) for x in b], dtype="boolean")
    if ctype == "float" and v.dtype != object:
        return v
    if ctype in ("date", "datetime") and v.dtype != object:
        return pd.to_datetime(v * 86400e9, errors="coerce")
    if v.dtype != object:
        return v
    return np.array([None if _isnull(x) else x for x in v], dtype=object)


def is_path(name: str) -> bool:
    return bool(_NAME_PATH.match(name))
