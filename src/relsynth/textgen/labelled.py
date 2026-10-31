# SPDX-License-Identifier: Apache-2.0
"""A generated, labelled set of ticket-, note- and chat-style texts with planted
personal data and exact character offsets, for measuring the recall of a PII
configuration (`relsynth text pii-eval`).

It is generated, not collected: it contains no real person.  Names are drawn
from syllables and a held-out list that the built-in regex recognizer does not
know, so its recall on names reflects its context cues only.  It measures the
formats it plants; a deployment should add a labelled sample of its own texts
(same JSON-lines format: {"text": ..., "spans": [[start, end, type], ...]}).
"""
from __future__ import annotations

import numpy as np

from .pii import _iban_check

HELD_OUT_FIRST = ("Marta Jonas Priya Tobias Fatima Henrik Leila Mateo Ingrid Kwame Aiko Dmitri Olga Rafael Signe "
                  "Tariq Beatriz Emil Hana Viktor Noor Pablo Greta Kenji Amara Lars Selin Bruno Yara Oskar").split()
HELD_OUT_LAST = ("Lindqvist Okafor Brennan Varga Haddad Moreno Petrov Sato Nyberg Castillo Kowalski Duarte Abiodun "
                 "Fontaine Halvorsen Mendes Rahimi Szabo Ferreira Yilmaz Achterberg Quintero").split()
MONTHS = "January February March April May June July August September October November December".split()

TEMPLATES = [
    "Hi, my name is {PERSON} and I was charged twice. Please call me on {PHONE}.",
    "Please send the refund to {IBAN}, account holder {PERSON}.",
    "The card {CARD} was declined yesterday, can you check?",
    "I moved to {ADDRESS}, please update my delivery address.",
    "You can reach me at {EMAIL} or {PHONE}.",
    "Customer {PERSON} (born {DATE}) asked to close the account.",
    "Login attempts from {IP} were blocked on {DATE}.",
    "The link {URL} gives an error 500 since this morning.",
    "Spoke with {PERSON} about invoice {ID}, promised a call back.",
    "Regards, {PERSON}",
    "Thanks,\n{PERSON}\n{PHONE}",
    "Dear support, my order {ID} has not arrived. {PERSON}, {ADDRESS}",
    "Mr. {PERSON} reported that the app crashes on login.",
    "Dr {PERSON} needs a copy of the contract signed on {DATE}.",
    "Patient {PERSON}, MRN {ID}, seen on {DATE}; follow up by {EMAIL}.",
    "please change my email from {EMAIL} to {EMAIL}",
    "transfer to {IBAN} failed, ref {ID}",
    "my new number is {PHONE} thx",
    "{PERSON} called about a double charge on card ending in {CARD}.",
    "Delivery to {ADDRESS} on {DATE} was missed, driver could not find the flat.",
    "Hello, this is {PERSON} from accounts. Our IBAN is {IBAN}.",
    "Can {PERSON} be added as a second user? Her email is {EMAIL}.",
    "Server {IP} stopped answering after the update on {DATE}.",
    "I tried {URL} but it asks for a password I never set.",
    "Ask for {PERSON} at the front desk, room {ID}.",
    "The parcel for {PERSON} went to {ADDRESS} instead.",
    "Contact {PERSON} on {PHONE} or by mail at {EMAIL}.",
    "Card {CARD}, expiry 09/28, holder {PERSON}.",
]
NO_PII = [
    "The app crashes when I open the reports page on my tablet.",
    "Why did the price of the pro plan go up this month?",
    "Tracking shows delivered but I have not received the package.",
    "Please close my account, I no longer need it.",
    "The export button does nothing in the latest version.",
    "I forgot my password and the reset link does not arrive.",
    "Since the update the dashboard is very slow.",
    "Can I change the delivery date to next week?",
]


def _syll_name(rng) -> str:
    return "".join(rng.choice(list("bdfgklmnprstvz")) + rng.choice(list("aeiou")) for _ in range(int(rng.integers(2, 4)))
                   ).title()


def value(typ: str, rng: np.random.Generator) -> str:
    i = lambda n: int(rng.integers(n))  # noqa: E731
    if typ == "PERSON":
        first = HELD_OUT_FIRST[i(len(HELD_OUT_FIRST))] if rng.random() < 0.6 else _syll_name(rng)
        last = HELD_OUT_LAST[i(len(HELD_OUT_LAST))] if rng.random() < 0.6 else _syll_name(rng)
        return first if rng.random() < 0.2 else f"{first} {last}"
    if typ == "EMAIL":
        return f"{_syll_name(rng).lower()}{rng.choice(['.', '_', ''])}{_syll_name(rng).lower()}{i(100)}@" \
               f"{rng.choice(['mail.com', 'corp.example.org', 'post.example.net', 'web.example.de'])}"
    if typ == "PHONE":
        f = i(4)
        if f == 0:
            return f"+{i(90) + 10} {i(900) + 100} {i(9000) + 1000} {i(900) + 100}"
        if f == 1:
            return f"({i(900) + 100}) {i(900) + 100}-{i(9000) + 1000}"
        if f == 2:
            return f"0{i(9000) + 1000} {i(900000) + 100000}"
        return f"{i(900) + 100}.{i(900) + 100}.{i(9000) + 1000}"
    if typ == "IBAN":
        cc = ("DE", "FR", "GR", "NL", "ES")[i(5)]
        bban = "".join(str(i(10)) for _ in range(18 if cc != "NL" else 10))
        if cc == "NL":
            bban = "ABNA" + bban
        s = cc + _iban_check(cc, bban) + bban
        return " ".join(s[k:k + 4] for k in range(0, len(s), 4)) if rng.random() < 0.5 else s
    if typ == "CARD":
        from .pii import luhn_ok
        while True:
            body = [int(rng.choice([4, 5]))] + [i(10) for _ in range(15)]
            if luhn_ok("".join(map(str, body))):
                s = "".join(map(str, body))
                return " ".join(s[k:k + 4] for k in range(0, 16, 4)) if rng.random() < 0.6 else s
    if typ == "IP":
        return f"{i(223) + 1}.{i(256)}.{i(256)}.{i(254) + 1}"
    if typ == "URL":
        return f"https://{_syll_name(rng).lower()}.example.com/{_syll_name(rng).lower()}?id={i(10000)}"
    if typ == "DATE":
        d, m, y = i(28) + 1, i(12), 1950 + i(75)
        f = i(3)
        return (f"{y}-{m + 1:02d}-{d:02d}", f"{d}/{m + 1}/{y}", f"{d} {MONTHS[m]} {y}")[f]
    if typ == "ID":
        return rng.choice([f"{i(10 ** 7) + 10 ** 6}", f"INV-{i(10 ** 6) + 10 ** 5}", f"AB{i(10 ** 6) + 10 ** 5}"])
    if typ == "ADDRESS":
        return f"{i(300) + 1} {HELD_OUT_LAST[i(len(HELD_OUT_LAST))]} " \
               f"{rng.choice(['Street', 'Road', 'Avenue', 'Lane', 'Drive'])}"
    raise ValueError(typ)


def labelled_set(n: int = 1000, seed: int = 1506, no_pii_share: float = 0.15) -> list[dict]:
    """n labelled texts: {"text", "spans": [[start, end, type]]}."""
    import re
    rng = np.random.default_rng(seed)
    out = []
    slot = re.compile(r"\{([A-Z]+)\}")
    for _ in range(n):
        if rng.random() < no_pii_share:
            out.append({"text": str(rng.choice(NO_PII)), "spans": []})
            continue
        tpl = TEMPLATES[int(rng.integers(len(TEMPLATES)))]
        text, spans, pos = "", [], 0
        for m in slot.finditer(tpl):
            text += tpl[pos:m.start()]
            v = value(m.group(1), rng)
            spans.append([len(text), len(text) + len(v), m.group(1)])
            text += v
            pos = m.end()
        text += tpl[pos:]
        out.append({"text": text, "spans": spans})
    return out
