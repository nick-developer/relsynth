# SPDX-License-Identifier: Apache-2.0
"""Draft a spec from a live database (any SQLAlchemy URL, including
PostgreSQL, Oracle and MS SQL Server), a SQLite file, or a folder of CSV files.

Identifying the data model is dependency discovery (see profiling.py).  The
pipeline:

 1. catalog: types, primary keys, foreign keys and unique constraints exactly as
    declared, composite keys kept together;
 2. uniform random samples per table, complete groups where a dependency lives
    within groups (an order's lines, a product's history, a parent's children);
 3. keys: declared, else a unique non-null column with a key-like name;
    columns named after their own table and unique there are natural keys,
    never links;
 4. links: declared ones, plus inclusion dependencies with name evidence on
    every other key-like column (also in tables that declare some links);
 5. roles: facts, entities, calendars, code lists and party tables;
 6. column types (binary columns, dates stored as midnight datetimes, codes);
 7. lookups: a column equal to a parent's column (or a grandparent's) on every
    row, whatever their names, when that equality is unlikely by chance;
 8. order headers: a document number (degenerate dimension) that determines
    the customer, the dates and the currency of its rows, with line numbers;
 9. offset keys: two keys into the same calendar whose gap is never negative;
10. derive rules (amount = price x quantity, tax = 8% of sales) with the stored
    precision as tolerance, running balances, aggregates (count, sum, min, max);
11. grains: declared composite keys, else pairs of keys that never repeat
    where independence predicts at least 10 repeats (screened on the sample,
    confirmed on every row of the table).

Every inference is listed under `review:` in the draft, so it can be checked;
`ASK` lines are questions the data cannot answer.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .data import canonical_keys, is_stringlike
from .discovery import Discovery
from .profiling import (MIN_ROWS, MIN_SHARE, algebraic_rules, canon, choose_rules, cumulative_rule, decimals,
                        equality, fd_profile, is_dateish, sequence_share, significant)
from .source import Source

PII_NAMES = {
    "email": r"e_?mail", "person_name": r"(^|_)(first|last|full|given|family|sur|middle)_?name$|contact_name",
    "phone": r"phone|mobile|fax|tel(ephone)?$",
    "pattern": r"ssn|iban|passport|tax_?id|vat|national_?id|card_?(no|number)|account_?(no|number)|address|street|"
               r"zip|post_?code|postal_?code|(^|_)lei$|(^|_)bic$|(^|_)swift|(^|_)isin$|(^|_)cusip$|(^|_)duns$|company_?(reg|number)",
}
# the words above match whole words of the snake_case name: after its start or an
# underscore (camel case is split by snake), before its end, an underscore, a digit
# or a word run on in lower case (emailaddress, zipcode); never inside a word
# (Elevation, Fairfax, Intel, ZipperCount)
PII_TOKENS = {g: re.compile(rf"(?:^|_)(?:{p})(?=_|\d|$|address|number|no|nr|num|code|id)") for g, p in PII_NAMES.items()}
PERSONAL_GENERATORS = {"email", "person_name", "phone"}
# party / organisation names: identifying when they are (nearly) unique per row
NAME_COLUMN = r"(^|_)(entity|company|institution|organi[sz]ation|customer|client|party|legal|bank|firm|person|" \
              r"reseller|store|supplier|vendor|buyer)_?name$"
TABLE_PREFIX = r"^(dim|fact|fct|d|f|tbl|t|ref|lkp|lu)_"
# columns that mark a table as describing people (its bare 'name' is then a person's name)
PERSON_MARKERS = r"(^|_)(first|last|given|family|middle)_?name$|e_?mail|phone|mobile|birth|gender|(^|_)ssn$|national_?id"


# tables whose rows are parties (who acts, reports, buys, owns): their near-unique
# labels are names of those parties, i.e. identifiers, never reference labels
PARTY_TABLE = (r"(^|_)(reporter|reporting_agent|customer|client|supplier|vendor|employee|staff|member|user|"
               r"account_holder|entity|institution|bank|company|organi[sz]ation|party|counterpart(y|ies)|issuer|"
               r"obligor|borrower|lender|insurer|policy_?holder|patient|student|person|people|contact|broker|"
               r"dealer|merchant|tenant|owner|beneficiary|guarantor|reseller|retailer|distributor|buyer|"
               r"prospect(ive)?_?buyer|agent|partner|subscriber|applicant)s?$")


SELF_REF = r"(parent|manager|reports_?to|boss|supervisor)"


def is_party_table(table, columns) -> bool:
    return is_person_table(columns) or bool(re.search(PARTY_TABLE, re.sub(TABLE_PREFIX, "", snake(table))))


def is_person_table(columns) -> bool:
    return any(re.search(PERSON_MARKERS, snake(c)) for c in columns)


def snake(name: str) -> str:
    """PascalCase / camelCase / UPPER_CASE -> snake_case (EntityKey -> entity_key)."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", str(name))
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()


def table_stem(t: str) -> str:
    """Table name without a dimensional-model prefix: DIM_Entity -> entity."""
    return re.sub(TABLE_PREFIX, "", snake(t))


def _has_time(s):
    s = pd.to_datetime(s, errors="coerce", format="mixed").dropna()
    return bool(len(s) and ((s.dt.hour != 0) | (s.dt.minute != 0) | (s.dt.second != 0)).mean() > 0.0)


def _infer_type(name, s: pd.Series, sqltype="", person=False, rows=None, party=False):
    """person: the table describes people (see is_person_table); party: its rows are
    parties (is_party_table); rows: its row count."""
    from .source import is_binary_sqltype
    if is_binary_sqltype(sqltype) or s.dropna().head(50).map(lambda v: isinstance(v, (bytes, bytearray, memoryview))).any():
        return {"type": "binary"}, "binary data (images, files): not synthesised, left empty"
    n = s.notna().sum()
    if n == 0:
        return {"type": "constant"}, "all values are empty"
    lname = snake(name)
    if lname in ("name", "full_name", "display_name") and person:
        return {"type": "pii", "pii": {"generator": "person_name"}}, "name of a person (the table has personal columns)"
    for gen, pat in PII_TOKENS.items():
        if pat.search(lname):
            return {"type": "pii", "pii": {"generator": gen}}, f"name matches {gen}"
    if re.search(NAME_COLUMN, lname) and is_stringlike(s) and s.nunique() >= 0.9 * n:
        return {"type": "pii", "pii": {"generator": "pattern"}}, "a (nearly) unique party name: treated as an identifier"
    if s.dtype == bool or "bool" in sqltype or sqltype == "bit":
        return {"type": "boolean"}, ""
    if pd.api.types.is_datetime64_any_dtype(s) or "date" in sqltype or "time" in sqltype:
        if "time" in sqltype and sqltype.startswith("time"):
            return {"type": "categorical"}, "time of day"
        if _has_time(s):
            return {"type": "datetime"}, ""
        return {"type": "date"}, ("every value is at midnight: a calendar date stored as a date-time"
                                  if "time" in sqltype else "")
    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna()
        if set(v.unique()) <= {0, 1} and v.nunique() == 2:
            return {"type": "boolean"}, "only 0/1 values"
        if np.allclose(v, np.round(v)):
            return {"type": "integer"}, ""
        return {"type": "float"}, ""
    sv = s.dropna().astype(str)
    if sv.nunique() == 1:
        return {"type": "constant"}, "a single value"
    if sv.str.contains(r"^[\w.+-]+@[\w-]+\.[\w.]+$").mean() > 0.8:
        return {"type": "pii", "pii": {"generator": "email"}}, "values look like emails"
    if is_dateish(s):
        parsed = pd.to_datetime(sv, errors="coerce", format="mixed")
        return ({"type": "datetime"} if _has_time(parsed) else {"type": "date"}), "values parse as dates"
    num = pd.to_numeric(sv, errors="coerce")
    zeros = sv.str.strip().str.match(r"^[+-]?0\d")
    if num.notna().mean() > 0.98 and zeros.any():
        # '0101001' is a code: as a number it would lose its leading zero and become another code
        hint = f"numeric codes with leading zeros ({zeros.mean():.0%} of values); kept as text"
        return ({"type": "categorical"} if sv.nunique() <= 0.5 * len(sv) else {"type": "id"}), hint
    if num.notna().mean() > 0.98 and set(num.dropna().unique()) <= {0, 1} and num.nunique() == 2:
        return {"type": "boolean"}, "only 0/1 values"
    if num.notna().mean() > 0.98:
        whole = np.allclose(num.dropna(), np.round(num.dropna()))
        return ({"type": "integer"} if whole else {"type": "float"}), "numeric strings"
    if sv.str.lower().isin(["true", "false", "yes", "no", "y", "n", "t", "f"]).mean() > 0.98 and sv.nunique() == 2:
        return {"type": "boolean"}, ""
    toks = sv.str.split().str.len().mean()
    ratio = sv.nunique() / max(len(sv), 1)
    if toks >= 5:
        return {"type": "text"}, f"free text (mean {toks:.1f} words)"
    if ratio > 0.9 and sv.nunique() >= 10:
        if rows is not None and rows <= 100 and not (person or party):
            return {"type": "categorical"}, "labels of a small reference table"
        if person or party or _code_like(s) or toks < 1.5:
            return {"type": "pii", "pii": {"generator": "pattern"}}, "near-unique strings: treated as an identifier"
        return {"type": "text"}, (f"near-unique labels or titles (mean {toks:.1f} words): generated as short text; "
                                  "declare pii if they name people or organisations")
    vc = sv.value_counts()
    rare = float(vc[vc < 5].sum() / max(len(sv), 1))
    if len(vc) >= 50 and rare >= 0.2 and toks >= 1.5 and not _code_like(s):
        # a long tail of multi-word labels (composers, product descriptions): most
        # values are too rare to release as categories
        return {"type": "text"}, (f"long-tail labels ({rare:.0%} of rows hold values seen fewer than 5 times, "
                                  f"mean {toks:.1f} words): generated as short text; declare pii if they name people")
    return {"type": "categorical"}, ""


def _code_like(s):
    """Strings that are codes ('C_40.00 r0010 c0010', 'v1-F01-003', 'SO43697'): few
    tokens, each containing a digit, unlike names ('Reporter 001 plc')."""
    sv = s.dropna().astype(str)
    if not len(sv):
        return False
    toks = sv.str.split()
    return bool(toks.str.len().mean() <= 4 and toks.map(lambda ts: all(any(ch.isdigit() for ch in w) for w in ts)).mean() >= 0.95)


def _keyset(s):
    return set(canonical_keys(pd.Series(s)).dropna().astype(str))


def _related_name(c, pc):
    a, b = snake(c), snake(pc)
    return a == b or a.endswith("_" + b) or b.endswith("_" + a)


# ====================================================================== pipeline
class _Profiler(Discovery):
    def __init__(self, source, sample_rows, include, exclude, groups, evidence="names+data", declared=True):
        from .engine import say
        self.say = say
        self.src = Source(source, include, exclude)
        self.evidence = evidence
        if not declared:
            # benchmark mode: the catalog's keys and links are hidden, the model is
            # identified from the data alone
            for m in self.src.meta.values():
                m["pk"], m["fks"], m["uniques"] = [], [], []
        self.source = source
        self.n_sample = sample_rows
        self.n_groups = groups
        self.review = []
        self.rng = np.random.default_rng(7)

    def ask(self, msg):
        self.review.append(msg)

    # ---------------------------------------------------------------- load
    def load(self):
        src = self.src
        self.names = list(src.names)
        self.counts, self.df, self.whole = {}, {}, {}
        for t in self.names:
            self.counts[t] = src.count(t)
            self.df[t], self.whole[t] = src.sample(t, self.n_sample)
            self.say(f"introspect: read {t} ({self.counts[t]:,} rows"
                     + ("" if self.whole[t] else f", random sample of {len(self.df[t]):,}") + ")")
        if src.unseeded and not src.random_order:
            self.ask("NOTE: this database has no random function known to relsynth; samples of "
                     + ", ".join(src.unseeded) + " (no primary key) are the first rows")
        elif src.unseeded:
            self.ask("NOTE: samples of " + ", ".join(src.unseeded) + " (no declared primary or unique key, or "
                     "very large) use the database's random order, so a second introspect may draft slightly "
                     "different rules for them")

    def full_keys(self, t, c):
        if self.whole[t]:
            return _keyset(self.df[t][c])
        return _keyset(self.src.column(t, c))

    def parent_rows(self, p, key, keys):
        """Rows of parent p whose key column is one of keys (from the sample when it
        is the whole table, else fetched)."""
        if self.whole[p]:
            return self.df[p]
        return self.src.rows_in(p, key, keys)

    # ---------------------------------------------------------------- keys
    def keys(self):
        from .keyfind import KeyFinder
        self.kf = KeyFinder(self, self.evidence)
        self.pk_cols, self.pks, self.unique = {}, {}, {}
        for t in self.names:
            m = self.src.meta[t]
            df = self.df[t]
            pk = list(m["pk"])
            self.unique[t] = [list(u) for u in m["uniques"]]
            for u in m["uniques"]:
                self.ask(f"{t}: unique key {u} (declared)")
            if len(pk) > 1:
                self.unique[t].insert(0, pk)
                self.ask(f"{t}: composite primary key {pk} (declared); kept as a unique key")
            if not pk:
                c0, why = self.kf.primary_key(t)
                if c0 is not None:
                    pk = [c0]
                    self.ask(f"{t}: primary key {c0} inferred ({why})")
            self.pk_cols[t] = pk
            self.pks[t] = pk[0] if len(pk) == 1 else None
        # natural keys: unique in their table, named after it (ProductAlternateKey of
        # DimProduct): attributes of the table, never links
        self.natural = {}
        for t in self.names:
            stem = table_stem(t)
            df = self.df[t]
            # the stem must be a whole word (ReporterKey is not a key of Report) and the
            # column unique, unless its name says alternate/natural/business key
            self.natural[t] = {c for c in df.columns if c != self.pks[t] and
                               ((re.search(rf"(^|_){re.escape(stem)}(_|$)", snake(c)) and df[c].notna().any()
                                 and df[c].dropna().is_unique)
                                or re.search(r"alternate|natural|business_key", snake(c)))
                               and not re.search(SELF_REF, snake(c))}

    # ---------------------------------------------------------------- links
    def links(self):
        rels = []
        declared = {t: set() for t in self.names}
        for t in self.names:
            for fk in self.src.meta[t]["fks"]:
                p, cols, pcols = fk["parent"], fk["cols"], fk["pcols"]
                if p not in self.df:
                    self.ask(f"{t}.{cols}: references {p}, which is not read (excluded); link dropped")
                    continue
                declared[t] |= set(cols)
                if len(cols) == 1:
                    if pcols == [self.pks.get(p)]:
                        rels.append({"child": t, "child_key": cols[0], "parent": p, "parent_key": pcols[0]})
                    else:
                        self.ask(f"ASK {t}.{cols[0]}: declared link to {p}.{pcols[0]}, which is not {p}'s primary key; "
                                 "relsynth joins on primary keys, so the link is left out")
                else:
                    keys = [self.pk_cols.get(p)] + self.unique.get(p, [])
                    if sorted(pcols) in [sorted(k) for k in keys if k]:
                        rels.append({"child": t, "child_key": list(cols), "parent": p, "parent_key": list(pcols)})
                        if pcols != self.pk_cols.get(p) and pcols not in self.unique[p]:
                            self.unique[p].append(list(pcols))
                    else:
                        self.ask(f"ASK {t}.{cols}: declared composite link to {p}.{pcols}, which is not a key of {p}")
        from .keyfind import acyclic, describe
        parents = [(pt, self.pks[pt]) for pt in self.names if self.pks.get(pt) is not None]
        for t in self.names:
            df = self.df[t]
            for c in df.columns:
                if c in declared[t] or c == self.pks[t] or c in self.src.meta[t]["binary"] or c in self.natural[t]:
                    continue
                best, amb = self.kf.best_parent(t, c, parents)
                if best:
                    rels.append({"child": t, "child_key": c, "parent": best["parent"], "_score": best["score"],
                                 "_why": describe(best)})
                elif amb:
                    self.ask(f"ASK {t}.{c}: looks like a foreign key "
                             + (f"to {amb[0]['parent']} ({describe(amb[0])}), but the data does not establish it"
                                if len(amb) == 1 else
                                "but the data fits " + ", ".join(f"{x['parent']} ({describe(x)})" for x in amb[:4])
                                + " equally well")
                             + "; declare it under relationships, or it is synthesised as a plain column")
        rels = acyclic(rels)
        for r in rels:
            why = r.pop("_why", None)
            r.pop("_score", None)
            if why:
                self.ask(f"{r['child']}.{r['child_key']} -> {r['parent']}: foreign key inferred ({why})")
        for r in rels:
            if r["child"] == r["parent"]:
                r["kind"] = "self_reference"
        self.rels = rels

    def simple(self, t):
        return [r for r in self.rels if r["child"] == t and isinstance(r["child_key"], str) and r["parent"] != t]

    def key_cols(self, t):
        cols = set(self.pk_cols[t])
        for r in self.rels:
            if r["child"] == t:
                cols |= set(r["child_key"]) if isinstance(r["child_key"], list) else {r["child_key"]}
        return cols

    # ---------------------------------------------------------------- roles
    def calendars(self):
        """Entity tables with a date column that has one row per day over a range."""
        self.calendar = {}
        for t in self.names:
            if self.pks[t] is None or not self.whole[t] or len(self.df[t]) < 28:
                continue
            df = self.df[t]
            for c in df.columns:
                if c == self.pks[t] or not is_dateish(df[c]) or df[c].isna().any() or not df[c].is_unique:
                    continue
                d = pd.to_datetime(df[c], errors="coerce", format="mixed")
                span = (d.max() - d.min()).days + 1
                if span > 0 and len(df) / span >= 0.95 and (d.dt.normalize() == d).all():
                    self.calendar[t] = c
                    self.ask(f"{t}: calendar (one row per day, {d.min().date()} to {d.max().date()}, order column {c}); "
                             "copied as public reference data")
                    break

    def roles(self):
        self.role = {}
        referenced_single = {r["parent"] for r in self.rels if isinstance(r["child_key"], str) and r["parent"] != r["child"]}
        for t in self.names:
            fks = self.simple(t)
            selfref = any(r["child"] == t and r["parent"] == t for r in self.rels)
            nested = any(r["child"] == t and isinstance(r["child_key"], list) for r in self.rels)
            rows = self.counts[t]
            if t in referenced_single or selfref or t in self.calendar:
                self.role[t] = "entity"
            elif nested or (fks and (len(self.pk_cols[t]) > 1 or self.pks[t] is None)):
                self.role[t] = "fact"
            elif fks and (len(fks) >= 2 or rows >= 2 * self.counts.get(fks[0]["parent"], rows)
                          or any(r["parent"] in self.calendar for r in fks)):
                # rows that point at several parents, outnumber their parent, or are
                # dated by a calendar (one row per day, found from the data) are events
                self.role[t] = "fact"
            else:
                self.role[t] = "entity"
        drop = [t for t in self.names if self.role[t] == "entity" and self.pks[t] is None]
        self.reference = set()
        for t in list(drop):
            linked = any(r["child"] == t or r["parent"] == t for r in self.rels)
            key = self.unique[t][0] if self.unique[t] else (None if linked else self.composite_key(t))
            if not linked and key:
                # a reference table (rates by date and currency pair): kept, copied, and
                # offered to keyed lookups
                self.reference.add(t)
                if key not in self.unique[t]:
                    self.unique[t].insert(0, key)
                    self.ask(f"{t}: unique key {key} inferred (no repeat in the sample where independent columns "
                             "would repeat at least 10 times)")
                drop.remove(t)
        for t in drop:
            self.ask(f"{t}: no primary key and no link to another table; left out of the draft "
                     "(add it with a primary key to synthesise it)")
            self.names.remove(t)
            self.rels = [r for r in self.rels if r["child"] != t and r["parent"] != t]

    # ---------------------------------------------------------------- types
    def types(self):
        self.cols = {}
        for t in self.names:
            df = self.df[t]
            person = is_person_table(df.columns)
            party = is_party_table(t, df.columns)
            keys = self.key_cols(t)
            cols = {}
            for c in df.columns:
                if c in keys:
                    cols[c] = {"type": "id"}
                    continue
                ty, why = _infer_type(c, df[c], self.src.meta[t]["types"].get(c, ""), person=person,
                                      rows=self.counts[t], party=party)
                cols[c] = ty
                if why:
                    self.ask(f"{t}.{c}: {ty['type']} ({why})")
            self.cols[t] = cols

    def copies(self):
        """Reference data copied as-is: calendars; small (<= 200 rows) non-party
        tables without personal data; and code lists (<= 20,000 rows, referenced,
        made of codes and labels, no measures) such as currencies or geography.
        A copied table may only reference copied tables, so this is a fixpoint."""
        self.copy = set()
        changed = True
        while changed:
            changed = False
            for t in self.names:
                if t in self.copy or self.role[t] != "entity":
                    continue
                cols = self.cols[t]
                party = is_party_table(t, list(cols))
                parents = [r["parent"] for r in self.simple(t)]
                if not all(p in self.copy for p in parents):
                    continue
                personal = any(cs["type"] == "pii" and cs["pii"]["generator"] in PERSONAL_GENERATORS
                               for cs in cols.values()) or any(re.search(NAME_COLUMN, snake(c)) for c in cols)
                referenced = any(r["parent"] == t and r["child"] != t for r in self.rels)
                rows = self.counts[t]
                why = None
                if t in self.calendar:
                    why = "calendar"
                elif party or personal:
                    if t in self.reference:
                        self.reference.discard(t)
                        self.names.remove(t)
                        self.ask(f"{t}: no primary key and no link, and it holds personal data; left out of the "
                                 "draft (add it with a primary key to synthesise it)")
                        changed = True
                        break
                    continue
                elif t in self.reference:
                    why = (f"reference table ({rows:,} rows, key {self.unique[t][0]}) that no table links to; "
                           "read by keyed lookups")
                elif rows <= 200:
                    why = f"{rows} rows of reference data"
                elif referenced and rows <= 20000 and self.has_codes(t) and not any(
                        cs["type"] in ("float", "date", "datetime") for c, cs in cols.items()) and not any(
                        cs["type"] == "integer" and self.df[t][c].nunique() > 50 and c not in self.natural[t]
                        for c, cs in cols.items()):
                    why = f"code list ({rows} rows of codes and labels that other tables point to)"
                if why:
                    self.copy.add(t)
                    changed = True
                    for c, cs in cols.items():
                        if cs["type"] in ("pii", "text"):
                            cols[c] = {"type": "categorical"}
                            self.review[:] = [x for x in self.review if not x.startswith(f"{t}.{c}: ")]
                    if t not in self.calendar:
                        self.ask(f"{t}: {why}, copied as-is (set mode: synthesize if it holds confidential data)")

    def has_codes(self, t):
        """A code list holds codes next to its labels (USD, AR, 1110, SO-12): a column
        other than the surrogate key that is a natural key, is named as a code, or
        holds short one-word values in capitals or with digits.  A table of titles
        or names only (albums, artists) is content, not a code list."""
        df = self.df[t]
        for c in df.columns:
            if c in self.pk_cols[t] or c in {r["child_key"] for r in self.simple(t)}:
                continue
            if c in self.natural[t] or re.search(r"(^|_)(code|iso|abbr|abbreviation|symbol)(_|$)", snake(c)):
                return True
            v = df[c].dropna().astype(str)
            if len(v) and v.str.len().max() <= 12 and (~v.str.contains(r"\s")).mean() >= 0.99 and \
                    (v.str.fullmatch(r"[A-Z0-9_.\-/]+") | v.str.contains(r"\d")).mean() >= 0.95:
                return True
        return False

    # ---------------------------------------------------------------- lookups
    def lookups(self):
        """Columns equal to a column of a parent (one hop) or grandparent (two hops)
        on every sampled row, nulls included, beyond chance.  Paths form no cycle."""
        self.lk = {}
        for t in self.names:
            if t in self.copy:
                continue
            df = self.df[t]
            cols = self.cols[t]
            fks = self.simple(t)
            pkset = set(self.pk_cols[t])
            nested = {c for r in self.rels if r["child"] == t and isinstance(r["child_key"], list) for c in r["child_key"]}
            # personal data and labels copied from a parent (a billing address equal to
            # the customer's address) are lookups too: the copy then takes the
            # synthetic parent's generated value, which keeps both the identity and privacy
            targets = [c for c, cs in cols.items() if c not in pkset and c not in nested
                       and cs["type"] not in ("binary", "constant")]
            if not targets or not len(df):
                continue
            paths = []          # (path, values aligned to df rows, related-name flag, hops)
            for r in fks:
                p, pk = r["parent"], self.pks.get(r["parent"])
                if pk is None:
                    continue
                pr = self.parent_rows(p, pk, df[r["child_key"]].dropna().unique()).drop_duplicates(pk).reset_index(drop=True)
                ix = pd.Index(canonical_keys(pr[pk]).astype(str))
                pos = ix.get_indexer(canonical_keys(df[r["child_key"]]).astype(str))
                for pc in pr.columns:
                    if pc == pk or self.cols.get(p, {}).get(pc, {}).get("type") in ("binary", "constant"):
                        continue
                    vals = np.where(pos >= 0, pr[pc].to_numpy(dtype=object)[np.maximum(pos, 0)], None)
                    paths.append((f"{r['child_key']}.{pc}", vals, pos >= 0, 1))
                for r2 in self.simple(p):
                    p2, pk2 = r2["parent"], self.pks.get(r2["parent"])
                    if pk2 is None:
                        continue
                    k1 = np.where(pos >= 0, pr[r2["child_key"]].to_numpy(dtype=object)[np.maximum(pos, 0)], None)
                    prow2 = self.parent_rows(p2, pk2, pd.Series(k1).dropna().unique())
                    pr2 = prow2.drop_duplicates(pk2)
                    ix2 = pd.Index(canonical_keys(pr2[pk2]).astype(str))
                    pos2 = ix2.get_indexer(canonical_keys(pd.Series(k1)).astype(str))
                    for pc in pr2.columns:
                        if pc == pk2 or self.cols.get(p2, {}).get(pc, {}).get("type") in ("binary", "constant"):
                            continue
                        vals = np.where(pos2 >= 0, pr2[pc].to_numpy(dtype=object)[np.maximum(pos2, 0)], None)
                        paths.append((f"{r['child_key']}.{r2['child_key']}.{pc}", vals, pos2 >= 0, 2))
            found = {}
            small, full = {}, {}
            for c in targets:
                isd = is_dateish(df[c])
                a = canon(df[c], as_date=isd)
                best = None
                for path, vals, ok, hops in paths:
                    if path.split(".")[0] == c:
                        continue
                    if ok.sum() < min(MIN_ROWS, len(df)) or ok.mean() < 0.95:
                        continue
                    okidx = np.flatnonzero(ok)
                    # quick screen on 200 rows, then the full test
                    if (path, isd) not in small:
                        small[(path, isd)] = canon(vals[okidx[:200]], as_date=isd)
                    if np.mean(a[okidx[:200]] == small[(path, isd)]) < MIN_SHARE:
                        continue
                    if (path, isd) not in full:
                        full[(path, isd)] = canon(vals, as_date=isd)
                    b = full[(path, isd)]
                    share, chance = equality(a[ok], b[ok])
                    if share < MIN_SHARE or not significant(share, chance, int(ok.sum())):
                        continue
                    rank = (hops, not _related_name(c, path.split(".")[-1]), chance)
                    if best is None or rank < best[0]:
                        best = (rank, path, share, chance)
                if best:
                    found[c] = best
            # no cycles: a lookup's first hop must not depend on the column itself
            dep = {c: v[1].split(".")[0] for c, v in found.items()}

            def reaches(a, b, seen=()):
                x = dep.get(a)
                return x is not None and (x == b or (x not in seen and reaches(x, b, seen + (a,))))
            for c in sorted(found, key=lambda c: found[c][0]):
                if reaches(dep[c], c):
                    del dep[c]
                    continue
                _, path, share, chance = found[c]
                self.lk[(t, c)] = path
                cols[c] = {"type": "lookup", "lookup": path}
                self.review[:] = [x for x in self.review if not x.startswith(f"{t}.{c}: ")]
                self.ask(f"{t}.{c}: lookup {path} (equal on {share:.1%} of {len(df):,} sampled rows; "
                         f"{chance:.1%} expected by chance)")

    # ---------------------------------------------------------------- headers
    def headers(self):
        """Degenerate dimensions: a document number d in a fact with repeats, that
        determines several of the fact's links (and other columns) by functional
        dependencies holding beyond chance; its rows are then an order header and
        its lines.  Line numbers 1..n are tested on complete orders."""
        self.header = {}
        for t in self.names:
            if self.role[t] != "fact" or any(r["child"] == t and isinstance(r["child_key"], list) for r in self.rels):
                continue
            df = self.df[t]
            fks = {r["child_key"] for r in self.simple(t)}
            if any(self.cols[t][c]["type"] == "sequence" for c in df.columns):
                continue
            if len(fks) < 2 or len(df) < 100:
                continue
            best = None
            for d in df.columns:
                cs = self.cols[t][d]
                if d in fks or cs["type"] in ("float", "date", "datetime", "binary", "constant", "boolean", "lookup"):
                    continue
                s = df[d]
                nd = s.nunique()
                if s.isna().any() or nd < 20 or nd > 0.97 * len(df):
                    continue
                hc = []
                for c in df.columns:
                    if c == d or self.cols[t][c]["type"] in ("constant", "binary"):
                        continue
                    agree, chance, nm = fd_profile(s, df[c])
                    if nm >= MIN_ROWS and agree >= MIN_SHARE and significant(agree, chance, nm):
                        hc.append(c)
                hf = [c for c in hc if c in fks]
                if hf and len(fks - set(hf)) >= 1 and (best is None or len(hf) > len(best[1])):
                    best = (d, hf, hc)
            if not best:
                continue
            d, hf, hc = best
            # complete orders: every row of a random set of document numbers
            keys = self.rng.choice(df[d].dropna().unique(), size=min(self.n_groups, df[d].nunique()), replace=False)
            grp = df[df[d].isin(keys)] if self.whole[t] else self.src.rows_in(t, d, list(keys))
            line, line_start = None, 1
            for c in df.columns:
                # a line number is often part of the declared key, hence typed id
                if c in hc or c == d or c in fks or self.cols[t][c]["type"] not in ("integer", "categorical", "id"):
                    continue
                for start in (1, 0):
                    sh, ng = sequence_share(grp[d], grp[c], start)
                    if ng >= MIN_ROWS and sh >= 0.99:     # ng: orders with two lines or more
                        line, line_start = c, start
                        break
                if line:
                    break
            hset = set(hc)
            hdr = {"key": d, "columns": [c for c in hc if (c not in self.pk_cols[t] or c in fks) and not (
                self.cols[t][c]["type"] == "lookup" and (isinstance(self.cols[t][c]["lookup"], dict) or
                                                         self.cols[t][c]["lookup"].split(".")[0] not in hset))]}
            if line:
                hdr["line_number"] = line
                hdr["columns"] = [c for c in hdr["columns"] if c != line]
            self.header[t] = hdr
            self.cols[t][d] = {"type": "id"}
            if line:
                self.cols[t][line] = {"type": "sequence", "within": d}
                if line_start != 1:
                    self.cols[t][line]["start"] = line_start
            self.review[:] = [x for x in self.review if not x.startswith(f"{t}.{d}: ")]
            self.ask(f"{t}: order header {d} ({df[d].nunique():,} values in the sample) determines "
                     f"{', '.join(hdr['columns'])}" + (f"; {line} numbers the lines 1..n" if line else "")
                     + "; generated as one header per order with its lines")

    # ---------------------------------------------------------------- offsets
    def order_column(self, p):
        if p in self.calendar:
            return self.calendar[p]
        if not self.whole[p] or self.pks[p] is None:
            return None
        df = self.df[p]
        for c in df.columns:
            if c != self.pks[p] and is_dateish(df[c]) and df[c].notna().all() and df[c].is_unique:
                return c
        return None

    def offsets(self):
        """Two or more keys of a table into the same ordered parent (a calendar):
        the key that comes first on (almost) every row is the base, each other key
        an offset from it when the gap is never negative or takes few values."""
        self.offset = {}
        for t in self.names:
            if t in self.copy or self.role[t] == "view":
                continue
            by_parent = {}
            for r in self.simple(t):
                if self.cols[t].get(r["child_key"], {}).get("type") == "lookup":
                    continue
                by_parent.setdefault(r["parent"], []).append(r["child_key"])
            hdr = self.header.get(t)
            for p, cs in by_parent.items():
                if len(cs) < 2:
                    continue
                if hdr:
                    hc = set(hdr["columns"])
                    if len({c in hc for c in cs}) > 1:
                        cs = [c for c in cs if c in hc]
                        if len(cs) < 2:
                            continue
                o = self.order_column(p)
                if o is None:
                    continue
                pdf = self.df[p]
                omap = pd.Series(canon(pdf[o], as_date=True), index=canonical_keys(pdf[self.pks[p]]).astype(str))
                omap = omap[~omap.index.duplicated()]
                vals = {c: pd.to_numeric(omap.reindex(canonical_keys(self.df[t][c]).astype(str)).to_numpy(), errors="coerce")
                        for c in cs}
                base = max(cs, key=lambda c: np.nanmean([np.mean(vals[c] <= vals[x]) for x in cs if x != c]))
                for c in cs:
                    if c == base:
                        continue
                    off = vals[c] - vals[base]
                    m = np.isfinite(off)
                    if m.sum() < MIN_ROWS:
                        continue
                    nonneg = float(np.mean(off[m] >= 0))
                    nu = len(np.unique(off[m]))
                    if nonneg >= MIN_SHARE or nu <= 30:
                        self.offset[(t, c)] = (base, o)
                        u, k = np.unique(off[m], return_counts=True)
                        top = ", ".join(f"{x:g} ({n / m.sum():.0%})" for x, n in sorted(zip(u, k), key=lambda z: -z[1])[:3])
                        self.ask(f"{t}.{c}: offset from {base} in {p}.{o} (gap never negative on {nonneg:.1%} of rows; "
                                 f"most common gaps {top}); generated as {base} plus a modelled gap")

    # ---------------------------------------------------------------- rules
    def derive_rules(self):
        self.rules = []
        for t in self.names:
            if t in self.copy or self.role[t] == "view":
                continue
            df = self.df[t]
            cols = self.cols[t]
            keys = self.key_cols(t)
            num = [c for c, cs in cols.items() if c not in keys and (cs["type"] in ("integer", "float") or (
                cs["type"] == "lookup" and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
                and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5))]
            if len(num) < 2 or len(df) < MIN_ROWS:
                continue
            num = num[:25]
            dec = {c: decimals(df[c]) for c in num}
            cands = algebraic_rules(df, num, dec, self.rng)
            lookups = {c for c in num if cols[c]["type"] == "lookup"}
            cands = [x for x in cands if x[0] not in lookups]
            for lhs, expr, _ins, _cx, tol in choose_rules(cands, basic=lookups):
                rule = {"table": t, "type": "derive", "expr": f"{lhs} = {expr}"}
                if tol:
                    rule["tolerance"] = tol
                self.rules.append(rule)
                self.ask(f"{t}: derive rule {lhs} = {expr} (holds on the sampled rows"
                         + (f" to the stored precision {tol:g}" if tol else "") + ")")

    def cumulative_rules(self):
        """Running balances in facts: within the rows of one parent ordered by a date
        (or a key into a calendar), y_t = y_{t-1} + step_t.  Tested on complete groups."""
        for t in self.names:
            if self.role.get(t) != "fact":
                continue
            df = self.df[t]
            cols = self.cols[t]
            num = [c for c, cs in cols.items() if cs["type"] in ("integer", "float")]
            if len(num) < 2:
                continue
            orders = [c for c, cs in cols.items() if cs["type"] in ("date", "datetime")] + \
                     [r["child_key"] for r in self.simple(t) if r["parent"] in self.calendar]
            groups = [r["child_key"] for r in self.simple(t) if r["parent"] not in self.calendar]
            done = False
            for g in groups:
                if done:
                    break
                keys = self.rng.choice(df[g].dropna().unique(), size=min(60, df[g].nunique()), replace=False)
                grp = df[df[g].isin(keys)] if self.whole[t] else self.src.rows_in(t, g, list(keys))
                for o in orders:
                    if o == g:
                        continue
                    if grp.duplicated([g, o]).any():
                        continue
                    found = cumulative_rule(grp, g, o, num)
                    for y, step in found:
                        self.rules.append({"table": t, "type": "cumulative", "column": y, "by": g, "order": o,
                                           "step": step})
                        self.ask(f"{t}: running balance {y} = previous {y} + {step} within {g}, in order of {o} "
                                 "(holds on complete groups)")
                    if found:
                        done = True
                        break

    def aggregates(self):
        """Parent columns equal to an aggregate of their children (count, sum, sum of a
        product, min, max), tested on every child row of a random set of parents."""
        for r in self.rels:
            t, p = r["child"], r["parent"]
            if t == p or isinstance(r["child_key"], list) or self.pks.get(p) is None or p in self.copy \
                    or self.role[t] == "view":
                continue
            pk = self.pks[p]
            pdf = self.df[p]
            pcols = [c for c, cs in self.cols[p].items() if cs["type"] in ("integer", "float", "date", "datetime")
                     and pdf[c].nunique() >= 3]
            if not pcols:
                continue
            keys = pdf[pk].dropna().unique()
            keys = self.rng.choice(keys, size=min(len(keys), 400), replace=False)
            child = self.df[t][self.df[t][r["child_key"]].isin(keys)] if self.whole[t] else \
                self.src.rows_in(t, r["child_key"], list(keys))
            if len(child) < 10:
                continue
            ccols = self.cols[t]
            num = [c for c, cs in ccols.items() if cs["type"] in ("integer", "float", "lookup")
                   and pd.to_numeric(child[c], errors="coerce").notna().mean() > 0.5]
            dts = [c for c, cs in ccols.items() if cs["type"] in ("date", "datetime") or
                   (cs["type"] == "lookup" and is_dateish(child[c]))]
            key = canonical_keys(child[r["child_key"]]).astype(str).to_numpy(dtype=object)
            # child columns as numbers, computed once per relation (pure functions of the column)
            num_of, canon_of = {}, {}

            def as_num(col):
                if col not in num_of:
                    num_of[col] = pd.to_numeric(child[col], errors="coerce").to_numpy(float)
                return num_of[col]

            def as_canon(col, isdate):
                if (col, isdate) not in canon_of:
                    canon_of[(col, isdate)] = pd.to_numeric(pd.Series(canon(child[col], as_date=isdate)),
                                                            errors="coerce").to_numpy(float)
                return canon_of[(col, isdate)]
            prow = pdf[pdf[pk].isin(keys)].drop_duplicates(pk)
            pkeys = canonical_keys(prow[pk]).astype(str).to_numpy(dtype=object)
            # a child aggregate is the same for every parent column c: computed once
            agg_of = {}
            for c in pcols:
                isdate = self.cols[p][c]["type"] in ("date", "datetime")
                target = canon(prow[c], as_date=isdate)
                target = pd.to_numeric(pd.Series(target), errors="coerce").to_numpy(dtype=float)
                if isdate:
                    cands = [(agg, of) for of in dts for agg in ("min", "max")]
                else:
                    cands = [("count", None)] + [("sum", f"{a} * {b}") for i, a in enumerate(num) for b in num[i + 1:]] + \
                            [("sum", a) for a in num] + [(agg, a) for a in num for agg in ("min", "max")]
                for agg, of in cands:
                    # a child column that is a lookup of this very parent column (an order
                    # line's date copied from its order) makes min = max = the parent's value:
                    # that is the lookup seen from the other side, and declaring both would
                    # make each computed from the other (a cycle)
                    if of is not None and any(ccols.get(x, {}).get("type") == "lookup" and
                                              str(ccols[x]["lookup"]) == f"{r['child_key']}.{c}" for x in of.split(" * ")):
                        continue
                    ck = (agg, of, isdate and of is not None and " * " not in of)
                    if ck not in agg_of:
                        if agg == "count":
                            v = pd.Series(1.0, index=key).groupby(level=0).sum()
                        else:
                            if of is not None and " * " in of:
                                a, b = of.split(" * ")
                                x = as_num(a) * as_num(b)
                            else:
                                x = as_canon(of, isdate)
                            v = getattr(pd.Series(x, index=key).groupby(level=0), agg)()
                        v = v.reindex(pkeys).to_numpy(float)
                        if agg in ("count", "sum"):
                            v = np.nan_to_num(v)
                        agg_of[ck] = v
                    v = agg_of[ck]
                    m = np.isfinite(target) | np.isfinite(v)
                    if m.sum() < 10:
                        continue
                    eq = np.isclose(np.nan_to_num(v[m], nan=-1e300), np.nan_to_num(target[m], nan=-1e300),
                                    atol=0.005 + 1e-9, rtol=1e-9)
                    if eq.mean() >= MIN_SHARE:
                        rule = {"table": p, "type": "aggregate", "column": c, "agg": agg, "from": t}
                        if of:
                            rule["of"] = of
                        if sum(1 for x in self.rels if x["child"] == t and x["parent"] == p) > 1:
                            rule["via"] = f"{t}.{r['child_key']}->{p}"
                        self.rules.append(rule)
                        self.ask(f"{p}.{c} = {agg}({of or 'rows'}) of {t}: aggregate inferred "
                                 f"(holds for every one of {m.sum()} parents checked)")
                        break

    # ---------------------------------------------------------------- grains
    def grains(self):
        for t in self.names:
            if t in self.copy or self.role[t] == "view":
                continue
            df = self.df[t]
            if self.unique[t] or t in self.header:
                continue
            fks = [r["child_key"] for r in self.simple(t)]
            cats = [c for c, cs in self.cols[t].items() if cs["type"] in ("categorical", "boolean")
                    and c not in fks and 1 < df[c].nunique() <= 50]
            cand = [(a, b) for i, a in enumerate(fks) for b in fks[i + 1:]]
            if self.role[t] == "fact":
                cand += [(a, c) for a in fks for c in cats]
            for a, b in cand:
                sub = df[[a, b]].dropna()
                n = len(sub)
                if n < 100 or sub.duplicated().any():
                    continue
                N = self.counts[t]
                space = max(sub[a].nunique() * sub[b].nunique(), 1)
                expected = n * (n - 1) / (2 * space)      # repeated pairs expected if the columns were drawn independently
                if expected >= 10 and not self.whole[t]:
                    # the sample only screens: the pair must be unique on every row
                    if self.src.columns(t, [a, b]).dropna().duplicated().any():
                        self.ask(f"{t}: [{a}, {b}] has no repeat in the sample but repeats in the whole table; "
                                 "not a unique key")
                        continue
                if expected >= 10:
                    self.unique[t].append([a, b])
                    self.ask(f"{t}: unique key [{a}, {b}] inferred ({n} {'rows' if n == N else 'sampled rows'}, "
                             f"no repeated pair where about {expected:.0f} would be expected by chance); "
                             "it is kept in the synthetic data")

    # ---------------------------------------------------------------- spec
    def anchor(self, t):
        fks = self.simple(t)
        hdr = self.header.get(t)
        if hdr:
            fks = [r for r in fks if r["child_key"] in hdr["columns"]] or fks
        stem = table_stem(t)

        def covered(r):
            p, pk = r["parent"], self.pks.get(r["parent"])
            if pk is None:
                return False
            pv = self.full_keys(p, pk)
            cv = _keyset(self.df[t][r["child_key"]])
            return bool(pv) and len(pv & cv) >= 0.99 * len(pv)
        depth = {r["parent"]: sum(1 for x in self.simple(r["parent"])) for r in fks}
        # the 'one' side whose row counts are kept: the table the fact is named after;
        # not a calendar or a copied code list; a large parent whose every row has
        # rows here (a header always has lines, a product need not sell); one that
        # links to other tables itself; the larger one
        return max(fks, key=lambda r: (stem.startswith(table_stem(r["parent"])) and stem != table_stem(r["parent"]),
                                        r["parent"] not in self.calendar, r["parent"] not in self.copy,
                                        covered(r) and self.counts.get(r["parent"], 0) >= 100,
                                        depth[r["parent"]] > 0, self.counts.get(r["parent"], 0)))["parent"]

    def build(self, name):
        spec = {"version": 1, "name": name, "source": self.source, "target": {"kind": "csv", "path": "synthetic"},
                "tables": {}, "relationships": [], "rules": []}
        for t in self.names:
            ts = {"role": self.role[t], "columns": self.cols[t]}
            if self.role[t] == "view":
                ts["view_of"] = self.view[t]
                spec["tables"][t] = ts
                continue
            if self.pks[t]:
                ts["primary_key"] = self.pks[t]
            if t in self.copy:
                ts["mode"] = "copy"
            if self.unique[t]:
                ts["unique"] = self.unique[t]
            if t in getattr(self, "dims", {}):
                ts["dimensions"] = self.dims[t]
            if t in getattr(self, "bridge", {}):
                ts["bridge"] = self.bridge[t]
            if t in getattr(self, "unknown", {}):
                ts["unknown_members"] = self.unknown[t]
            if t in getattr(self, "scd_info", {}):
                ts["scd"] = self.scd_info[t]
            if getattr(self, "affinity", {}).get(t):
                ts["link_affinity"] = self.affinity[t]
            if getattr(self, "group_fx", {}).get(t):
                ts["group_effects"] = self.group_fx[t]
            if t in getattr(self, "key_order", {}):
                ts["key_order"] = self.key_order[t]
            if self.role[t] == "fact":
                f = {}
                if self.simple(t) and not any(r["child"] == t and isinstance(r["child_key"], list) for r in self.rels):
                    a = self.anchor(t)
                    f["anchor"] = a
                    u2 = next((u for u in self.unique[t] if len(u) == 2), None)
                    pr = None
                    if u2:
                        par = {r["child_key"]: r["parent"] for r in self.simple(t)}
                        if all(c in par for c in u2) and a in [par[c] for c in u2]:
                            pr = next(par[c] for c in u2 if par[c] != a) if len({par[c] for c in u2}) == 2 else None
                    if pr is None:
                        hc = set(self.header.get(t, {}).get("columns", []))
                        rest = [r for r in self.simple(t) if r["parent"] != a and r["child_key"] not in hc
                                and self.cols[t].get(r["child_key"], {}).get("type") != "lookup"
                                and (t, r["child_key"]) not in self.offset]
                        if rest:
                            pr = max(rest, key=lambda r: (r["parent"] not in self.calendar,
                                                          self.counts.get(r["parent"], 0)))["parent"]
                    if pr and pr != a:
                        f["partner"] = pr
                if t in self.header:
                    f["header"] = self.header[t]
                tc = next((c for c, v in self.cols[t].items() if v["type"] in ("date", "datetime")), None)
                if tc and t not in self.header:
                    f.update({"time_column": tc, "time_bucket": "month"})
                ts["fact"] = f
                part = f", partner {f['partner']}" if f.get("partner") else ""
                tcol = f", time column {tc}" if f.get("time_column") else ""
                self.ask(f"{t}: role fact" + (f" (anchor {f['anchor']}{part}{tcol})"
                                              if f.get("anchor") else " (nested in its parent fact)")
                         + "; check the anchor is the 'one' side whose row counts should be preserved")
            spec["tables"][t] = ts
        for r in self.rels:
            r = dict(r)
            if isinstance(r["child_key"], str) and (r["child"], r["child_key"]) in self.offset:
                base, o = self.offset[(r["child"], r["child_key"])]
                r["offset_from"] = base
                r["order_by"] = o
            spec["relationships"].append(r)
        spec["rules"] = self.rules
        spec["synthesis"] = {"seed": 7, "scale": {"default": 1.0}}
        spec["review"] = self.review
        return spec


def introspect(source: dict, name="draft", sample_rows=20000, include=None, exclude=None, groups=300,
               evidence="names+data", declared=True) -> dict:
    """Draft a spec.  include / exclude: lists of table-name patterns (regular
    expressions, case-insensitive, matched in full).  evidence: "names+data" (column
    names count as one weighted piece of evidence) or "data" (values only);
    declared=False hides the catalog's keys and links (to test identification)."""
    P = _Profiler(source, sample_rows, include, exclude, groups, evidence, declared)
    P.load()
    P.keys()
    P.links()
    P.calendars()
    P.roles()
    P.types()
    P.copies()
    P.say("introspect: lookups")
    P.lookups()
    P.keyed_lookups()
    P.mapped_lookups()
    P.line_keys()
    P.views()
    P.say("introspect: order headers, offsets and rules")
    P.headers()
    P.offsets()
    P.col_offsets()
    P.sequences()
    P.derive_rules()
    P.soft_rules()
    P.scd()
    P.unknown_members()
    P.bridges()
    P.dimensions()
    P.nested_labels()
    P.ratios()
    P.group_effects()
    P.cumulative_rules()
    P.aggregates()
    P.grains()
    P.say("introspect: cross-table rules")
    P.cross_rules()
    P.link_affinities()
    P.key_orders()
    return P.build(name)
