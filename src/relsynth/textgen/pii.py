# SPDX-License-Identifier: Apache-2.0
"""Personal data in free text: detection, typed placeholders and surrogates.

Defence in depth for every text method (docs/text.md, Prop. 1506): before a
text model sees a source text, the spans found by the recognizers are replaced
by typed placeholders (``<PERSON>``, ``<EMAIL>``, ...); a generated text gets
fresh, format-valid surrogates for its placeholders.  The placeholders are
tokens of the text models, so a model learns *where* a name goes, never the
name.

Recognizers:

* ``regex``  (built in, no download): e-mail, URL, IPv4/IPv6, IBAN (ISO 13616
  mod-97 check), payment card (Luhn check), phone, date, long identifiers,
  street addresses, and persons after honorifics and sign-off cues.
* ``presidio`` (Microsoft Presidio, MIT; needs a local spaCy model).
* ``spacy`` (spaCy NER, MIT; a local pipeline such as en_core_web_lg).
* ``gliner`` (GLiNER, Zaratiana et al. NAACL 2024; Apache-2.0 weights such as
  urchade/gliner_multi_pii-v1, read from a local folder).

They are combined by union: a span found by any recognizer is redacted.  The
recall of a configuration is measured with `evaluate` on a labelled set
(`labelled_set`, or the user's own) and reported with Wilson intervals; no
recognizer is assumed to be complete, which is why the leakage audit
(`relsynth.textgen.audit`) runs on the output as well.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

TYPES = ("PERSON", "EMAIL", "PHONE", "IBAN", "CARD", "IP", "URL", "DATE", "ID", "ADDRESS", "LOCATION", "ORG")
PLACEHOLDER = re.compile(r"<(" + "|".join(TYPES) + r"|NUM)>")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str
    source: str = "regex"
    score: float = 1.0


# ------------------------------------------------------------------ checksums
def luhn_ok(digits: str) -> bool:
    d = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    s = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x *= 2
            x -= 9 if x > 9 else 0
        s += x
    return s % 10 == 0


def iban_ok(s: str) -> bool:
    s = re.sub(r"\s", "", s).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", s):
        return False
    r = s[4:] + s[:4]
    return int("".join(str(int(ch, 36)) for ch in r)) % 97 == 1


def _iban_check(country: str, bban: str) -> str:
    r = bban + country + "00"
    return f"{98 - int(''.join(str(int(ch, 36)) for ch in r)) % 97:02d}"


# ------------------------------------------------------------------ regex recognizer
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_CAP = r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?"
_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("URL", re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+[^\s<>\"'.,;:!?)]")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    ("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("IP", re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|\b(?:[0-9a-fA-F]{1,4}:){1,6}:(?:[0-9a-fA-F]{1,4})?\b")),
    ("DATE", re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)? "
                        + _MONTH + r",? \d{4}\b|\b" + _MONTH + r" \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b")),
    ("PHONE", re.compile(r"(?<![\w+])(?:\+\d{1,3}[ .-]?)?(?:\(\d{1,4}\)[ .-]?)?\d{2,4}(?:[ .-]?\d{2,4}){1,3}(?!\w)")),
    ("ADDRESS", re.compile(r"\b\d{1,5}[a-z]? (?:" + _CAP + r" ){1,3}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|"
                           r"Boulevard|Blvd|Way|Court|Ct|Place|Pl|Square|Sq|Terrace)\b\.?|\b(?:[Ff]lat|[Aa]pt\.?|"
                           r"[Aa]partment|[Uu]nit|[Ss]uite) \d{1,4}[A-Za-z]?\b")),
    ("PERSON", re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Madam)\.? (" + _CAP + r"(?: " + _CAP + r")?)")),
    ("PERSON", re.compile(r"(?:[Rr]egards|[Tt]hanks|[Tt]hank you|[Cc]heers|[Ss]incerely|[Bb]est),?\s+(" + _CAP
                          + r"(?: " + _CAP + r")?)\b")),
    ("PERSON", re.compile(r"\b(?:[Mm]y name is|[Tt]his is|I am|I'm|[Cc]ontact|[Aa]sk for|[Ss]poke (?:to|with)|"
                          r"[Cc]alled|[Nn]amed|[Pp]atient|[Hh]older|[Cc]ustomer|[Cc]lient|[Ee]mployee|"
                          r"[Cc]olleague) (" + _CAP + r"(?: " + _CAP + r")?)\b")),
    ("ID", re.compile(r"\b[A-Z]{0,4}-?\d{5,}\b|\b[A-Z]{1,3}\d{2,}[A-Z]\d*\b")),
]
_NOT_PERSON = {"The", "This", "That", "Please", "Hello", "Hi", "Dear", "Thanks", "Thank", "Regards", "Monday", "Tuesday",
               "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Support", "Team", "Customer", "Service", "Sorry",
               "Not", "Still", "Very", "Really", "Also", "Just", "Here", "There", "Yes", "No", "Ok", "Okay", "Good",
               "Going", "Trying", "Looking", "Having", "Waiting", "Writing", "Unable", "Able", "Happy", "Sure", "Glad"}


def _regex_spans(text: str) -> list[Span]:
    out = []
    for typ, pat in _PATTERNS:
        for m in pat.finditer(text):
            s, e = (m.start(1), m.end(1)) if pat.groups else (m.start(), m.end())
            frag = text[s:e]
            if typ == "CARD" and not luhn_ok(frag):
                continue
            if typ == "IBAN" and not iban_ok(frag):
                continue
            if typ == "PHONE" and (sum(c.isdigit() for c in frag) < 7 or not re.search(r"[ .()+-]", frag)
                                   and sum(c.isdigit() for c in frag) < 9):
                continue
            if typ == "PERSON":
                words = frag.split()
                while words and words[-1] in _NOT_PERSON:
                    words.pop()
                if not words or words[0] in _NOT_PERSON:
                    continue
                e = s + len(" ".join(words))
            out.append(Span(s, e, typ, "regex"))
    return out


# ------------------------------------------------------------------ optional recognizers
_PRESIDIO = {"PERSON": "PERSON", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE", "IBAN_CODE": "IBAN",
             "CREDIT_CARD": "CARD", "IP_ADDRESS": "IP", "URL": "URL", "DATE_TIME": "DATE", "LOCATION": "LOCATION",
             "US_SSN": "ID", "US_PASSPORT": "ID", "US_DRIVER_LICENSE": "ID", "UK_NHS": "ID", "US_BANK_NUMBER": "ID",
             "MEDICAL_LICENSE": "ID", "NRP": None, "ORGANIZATION": "ORG"}
_SPACY = {"PERSON": "PERSON", "PER": "PERSON", "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "ADDRESS", "ORG": "ORG",
          "DATE": "DATE"}
GLINER_LABELS = {"person": "PERSON", "email": "EMAIL", "phone number": "PHONE", "address": "ADDRESS",
                 "credit card number": "CARD", "iban": "IBAN", "date of birth": "DATE", "date": "DATE",
                 "organization": "ORG", "location": "LOCATION", "id number": "ID", "ip address": "IP", "url": "URL"}


class Recognizer:
    name = "base"

    def spans(self, text: str) -> list[Span]:          # pragma: no cover - interface
        raise NotImplementedError


class RegexRecognizer(Recognizer):
    name = "regex"

    def spans(self, text):
        return _regex_spans(text)


class PresidioRecognizer(Recognizer):
    """Microsoft Presidio's AnalyzerEngine with a local spaCy model (no download)."""
    name = "presidio"

    def __init__(self, model: str = "en_core_web_lg", threshold: float = 0.35, language: str = "en"):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        nlp = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy",
                                                   "models": [{"lang_code": language, "model_name": model}]})
        self.engine = AnalyzerEngine(nlp_engine=nlp.create_engine(), supported_languages=[language])
        self.threshold, self.language = threshold, language

    def spans(self, text):
        res = self.engine.analyze(text=text, language=self.language, score_threshold=self.threshold)
        return [Span(r.start, r.end, _PRESIDIO[r.entity_type], "presidio", float(r.score)) for r in res
                if _PRESIDIO.get(r.entity_type)]


class SpacyRecognizer(Recognizer):
    name = "spacy"

    def __init__(self, model: str = "en_core_web_lg"):
        import spacy
        self.nlp = spacy.load(model)

    def spans(self, text):
        return [Span(e.start_char, e.end_char, _SPACY[e.label_], "spacy") for e in self.nlp(text).ents
                if e.label_ in _SPACY]


class GlinerRecognizer(Recognizer):
    """GLiNER zero-shot NER from a local folder (local_files_only: never downloads)."""
    name = "gliner"

    def __init__(self, model: str, threshold: float = 0.4, labels: dict | None = None):
        from gliner import GLiNER
        self.model = GLiNER.from_pretrained(model, local_files_only=True)
        self.threshold, self.labels = threshold, dict(labels or GLINER_LABELS)

    def spans(self, text):
        ents = self.model.predict_entities(text, list(self.labels), threshold=self.threshold)
        return [Span(e["start"], e["end"], self.labels[e["label"]], "gliner", float(e.get("score", 1.0))) for e in ents
                if e["label"] in self.labels]


def make_recognizer(cfg) -> Recognizer:
    """`regex`, or {name: presidio|spacy|gliner, model: <local name or folder>, threshold: ...}."""
    if isinstance(cfg, str):
        cfg = {"name": cfg}
    cfg = dict(cfg)
    name = cfg.pop("name")
    try:
        if name == "regex":
            return RegexRecognizer()
        if name == "presidio":
            return PresidioRecognizer(**cfg)
        if name == "spacy":
            return SpacyRecognizer(**cfg)
        if name == "gliner":
            return GlinerRecognizer(**cfg)
    except ImportError as e:
        from ..spec import SpecError
        raise SpecError(f"PII recognizer {name!r} needs its library: pip install 'relsynth[text-pii]' ({e})") from e
    from ..spec import SpecError
    raise SpecError(f"unknown PII recognizer {name!r}: regex, presidio, spacy or gliner")


# ------------------------------------------------------------------ the pipeline
def _resolve(spans: list[Span]) -> list[Span]:
    """Union of overlapping spans: overlapping spans merge into one, typed by the
    longest member (so a redaction never leaves part of an entity behind)."""
    out: list[Span] = []
    for s in sorted(spans, key=lambda x: (x.start, -(x.end - x.start))):
        if out and s.start < out[-1].end:
            last = out[-1]
            typ = last.type if (last.end - last.start) >= (s.end - s.start) else s.type
            out[-1] = Span(last.start, max(last.end, s.end), typ, last.source, max(last.score, s.score))
        else:
            out.append(s)
    return out


class PiiPipeline:
    """Union of recognizers; `redact` replaces every span by its typed placeholder."""

    def __init__(self, recognizers=("regex",)):
        self.recognizers = [r if isinstance(r, Recognizer) else make_recognizer(r) for r in recognizers]

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.recognizers]

    def spans(self, text: str) -> list[Span]:
        found = []
        for r in self.recognizers:
            found += r.spans(text)
        return _resolve(found)

    def redact(self, text):
        if text is None or (isinstance(text, float) and math.isnan(text)):
            return text, []
        text = str(text)
        sp = self.spans(text)
        parts, pos = [], 0
        for s in sp:
            parts += [text[pos:s.start], f"<{s.type}>"]
            pos = s.end
        parts.append(text[pos:])
        return "".join(parts), sp

    def redact_all(self, texts) -> tuple[list, dict]:
        out: list = []
        counts: dict[str, int] = {}
        for t in texts:
            r, sp = self.redact(t)
            out.append(r)
            for s in sp:
                counts[s.type] = counts.get(s.type, 0) + 1
        return out, counts


# ------------------------------------------------------------------ surrogates
_FIRST = ("Alex Maria Nikos Elena John Sofia Omar Lena Ivan Chen Aisha Lucas Emma Noah Mia Leo Zoe Ravi Anna Tom Sara "
          "Ben Nina Paul Ella Max Ines Yuki Ali Kate").split()
_LAST = ("Smith Papadopoulos Garcia Muller Rossi Kim Silva Novak Cohen Ahmed Dubois Jensen Nowak Tanaka Ivanova Brown "
         "Costa Murphy Weber Lopez Fischer Moreau Khan Ortiz").split()
_STREETS = "Example Sample Test Demo Placeholder Fictional".split()
_CITIES = "Springfield Riverton Lakeside Hillview Brookfield Fairview".split()


def surrogate(typ: str, rng: np.random.Generator) -> str:
    """A fresh, format-valid value of the type, from reserved or fictional ranges:
    example.com addresses (RFC 2606), 555-01xx numbers, documentation IPs
    (RFC 5737), Luhn-valid test cards, mod-97-valid IBANs of a test bank."""
    i = lambda n: int(rng.integers(n))  # noqa: E731
    if typ == "PERSON":
        return f"{_FIRST[i(len(_FIRST))]} {_LAST[i(len(_LAST))]}"
    if typ == "EMAIL":
        return f"{_FIRST[i(len(_FIRST))].lower()}.{_LAST[i(len(_LAST))].lower()}{i(1000)}@example.com"
    if typ == "PHONE":
        return f"+1 555 01{i(100):02d}"
    if typ == "IBAN":
        bban = "TEST" + "".join(str(i(10)) for _ in range(14))
        s = "GB" + _iban_check("GB", bban) + bban
        return " ".join(s[k:k + 4] for k in range(0, len(s), 4))
    if typ == "CARD":
        body = [4, 0, 0, 0] + [i(10) for _ in range(11)]
        for c in range(10):
            if luhn_ok("".join(map(str, body + [c]))):
                body.append(c)
                break
        s = "".join(map(str, body))
        return " ".join(s[k:k + 4] for k in range(0, 16, 4))
    if typ == "IP":
        return f"{('192.0.2', '198.51.100', '203.0.113')[i(3)]}.{1 + i(254)}"
    if typ == "URL":
        return f"https://www.example.com/{_STREETS[i(len(_STREETS))].lower()}/{i(1000)}"
    if typ == "DATE":
        return str(np.datetime64("2020-01-01") + np.timedelta64(i(1826), "D"))
    if typ in ("ID", "NUM"):
        return str(int(rng.integers(10 ** 5, 10 ** 8)))
    if typ == "ADDRESS":
        return f"{1 + i(200)} {_STREETS[i(len(_STREETS))]} Street"
    if typ == "LOCATION":
        return _CITIES[i(len(_CITIES))]
    if typ == "ORG":
        return f"{_STREETS[i(len(_STREETS))]} Ltd"
    return f"<{typ}>"


def fill(text: str, rng: np.random.Generator) -> str:
    """Replace every placeholder by a fresh surrogate of its type."""
    if not text:
        return text
    return PLACEHOLDER.sub(lambda m: surrogate(m.group(1), rng), text)


# ------------------------------------------------------------------ measured recall
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate(pipeline: PiiPipeline, labelled: list[dict]) -> dict:
    """Recall of a pipeline on labelled texts [{text, spans: [(start, end, type)]}].

    `redaction` recall counts a gold entity as caught when every character of it
    is inside a predicted span (what matters for redaction: nothing of it is left);
    `typed` recall also needs the predicted type to match.  Precision counts
    predicted spans that overlap a gold entity.  Each figure has a 95% Wilson
    interval."""
    per: dict[str, dict] = {}
    tp_pred = n_pred = 0
    for row in labelled:
        text = row["text"]
        pred = pipeline.spans(text)
        n_pred += len(pred)
        gold = [tuple(g) for g in row["spans"]]
        tp_pred += sum(any(p.start < ge and gs < p.end for gs, ge, _ in gold) for p in pred)
        for gs, ge, gt in gold:
            d = per.setdefault(gt, {"n": 0, "redacted": 0, "typed": 0})
            d["n"] += 1
            cover = np.zeros(ge - gs, bool)
            typed = False
            for p in pred:
                lo, hi = max(gs, p.start), min(ge, p.end)
                if lo < hi:
                    cover[lo - gs:hi - gs] = True
                    typed |= p.type == gt
            full = bool(cover[[not text[gs + k].isspace() for k in range(ge - gs)]].all())
            d["redacted"] += full
            d["typed"] += full and typed
    out: dict = {"recognizers": pipeline.names, "types": {}}
    N = R = T = 0
    for t, d in sorted(per.items()):
        out["types"][t] = {"n": d["n"], "recall": d["redacted"] / d["n"], "recall_ci": wilson(d["redacted"], d["n"]),
                           "typed_recall": d["typed"] / d["n"]}
        N, R, T = N + d["n"], R + d["redacted"], T + d["typed"]
    out["overall"] = {"n": N, "recall": R / max(N, 1), "recall_ci": wilson(R, N), "typed_recall": T / max(N, 1),
                      "precision": tp_pred / max(n_pred, 1), "precision_ci": wilson(tp_pred, n_pred)}
    return out
