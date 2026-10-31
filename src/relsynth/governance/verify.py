# SPDX-License-Identifier: Apache-2.0
"""`relsynth gov verify`: the registry's chain, signatures and digests, or a
release package against its signed manifest (design sec. 7, Prop. 803).

A registry check walks the whole log: every entry's hash against its
predecessor (Prop. 803), every signed checkpoint against the entry it names,
every object an entry names against its digest, every envelope's signature
against the keys trusted when it was written, and the ledger's protocol
(settlements never above their reservations, no refund after a touch).  It
returns the problems in log order; the first one is the first place the
registry was changed.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..errors import GovernanceError
from .canon import canonical, sha256
from .registry import CHECKPOINT_TYPE, GENESIS, KEYS_TYPE, chain_hash
from .sign import PublicKey, payload, verify_envelope


def _roles_from(env) -> dict:
    return {k["id"]: list(k.get("roles", [])) for k in payload(env)["predicate"]["keys"]}


def _keys_from(env) -> dict:
    out = {}
    for k in payload(env)["predicate"]["keys"]:
        out[k["id"]] = PublicKey.from_text(k["public"])
    return out


def verify_registry(reg) -> list[str]:
    probs: list[str] = []
    try:
        log = reg._read_log()
    except Exception as e:  # noqa: BLE001
        return [str(e)]
    prev = GENESIS
    hashes = {}
    for i, rec in enumerate(log):
        if rec.get("seq") != i or rec.get("prev") != prev or chain_hash(prev, rec.get("entry", {})) != rec.get("hash"):
            probs.append(f"log entry {i}: the hash chain is broken here (an entry was edited, removed, inserted or "
                         "reordered)")
            return probs
        prev = rec["hash"]
        hashes[i] = rec["hash"]
    # checkpoints: the head they sign must be on the chain, so a truncation after one is caught
    root = {reg.config["root_key"]["id"]: reg.root_key()}
    keys_now: dict = {}
    roles_now: dict = {}
    all_keys: dict = dict(root)
    for rec in log:
        if rec["entry"]["type"] == "keys":
            try:
                env = reg.get_json(rec["entry"]["envelope"])
            except Exception as e:  # noqa: BLE001
                probs.append(f"log entry {rec['seq']}: {e}")
                continue
            if not verify_envelope(env, root):
                probs.append(f"log entry {rec['seq']}: trusted-keys list not signed by the root key")
            all_keys.update(_keys_from(env))
    for cp in reg.checkpoints():
        st = payload(cp)
        if st.get("predicateType") != CHECKPOINT_TYPE:
            probs.append("a checkpoint file is not a checkpoint statement")
            continue
        p = st["predicate"]
        if not verify_envelope(cp, all_keys):
            probs.append(f"checkpoint {p.get('seq')}: signature does not verify")
        elif hashes.get(p["seq"]) != p["hash"]:
            probs.append(f"checkpoint {p['seq']}: the log does not contain the signed head (the log was truncated or "
                         "rewritten)")
    ledger: dict = {}
    for rec in log:
        ent, seq = rec["entry"], rec["seq"]
        t = ent["type"]
        if t == "keys":
            kenv = reg.get_json(ent["envelope"])
            keys_now, roles_now = _keys_from(kenv), _roles_from(kenv)
            continue
        if "envelope" in ent:
            try:
                raw = reg.get_blob(ent["envelope"])
            except Exception as ex:  # noqa: BLE001
                probs.append(f"log entry {seq} ({t}): {ex}")
                continue
            if sha256(raw) != ent["envelope"]:
                probs.append(f"log entry {seq} ({t}): the stored object does not match its digest")
                continue
            env = json.loads(raw)
            signed = verify_envelope(env, keys_now)
            if not signed:
                probs.append(f"log entry {seq} ({t}): no valid signature by a key trusted at that time")
            st = payload(env)
            if t == "manifest":
                if sha256(canonical(st)) != ent["manifest"]:
                    probs.append(f"log entry {seq}: manifest digest does not match the signed statement")
                if st["predicate"]["run"]["operator"]["key"] not in signed:
                    probs.append(f"log entry {seq}: the manifest is not signed by its operator")
                for r in st["predicate"].get("reports", []):
                    d = r["digest"]["sha256"]
                    try:
                        if sha256(reg.get_blob(d)) != d:
                            probs.append(f"log entry {seq}: the {r['kind']} report does not match its digest")
                    except Exception:  # noqa: BLE001
                        probs.append(f"log entry {seq}: the {r['kind']} report {d[:12]}… is missing")
                for s in st.get("subject", []):
                    d = s["digest"]["sha256"]
                    try:
                        if sha256(reg.get_blob(d)) != d:
                            probs.append(f"log entry {seq}: output {s['name']} does not match its digest")
                    except GovernanceError:  # nosec B110 - outputs are kept only when stored (the organisation's retention)
                        pass
            elif t == "approval" and st["predicate"]["approver"]["key"] not in signed:
                probs.append(f"log entry {seq}: the approval is not signed by the approver it names")
            elif t == "approval" and "agent" in roles_now.get(st["predicate"]["approver"]["key"], ()):
                probs.append(f"log entry {seq}: the approval is signed by an agent key; approvals are made by people")
        if t.startswith("ledger."):
            k = (ent["domain"], ent["run"])
            s = ledger.setdefault(k, {})
            if t == "ledger.reserve":
                s["reserve"] = (float(ent["rho"]), float(ent["delta"]))
            elif t == "ledger.touch":
                s["touch"] = True
            elif t == "ledger.settle":
                r = s.get("reserve")
                if r is None or float(ent["rho"]) > r[0] + 1e-12 or float(ent["delta"]) > r[1] + 1e-12:
                    probs.append(f"log entry {seq}: run {ent['run']} settled above its reservation")
            elif t == "ledger.release" and s.get("touch"):
                probs.append(f"log entry {seq}: run {ent['run']} was refunded after it read data")
    return probs


def verify_package(folder) -> list[str]:
    """Check a release package: trusted keys signed by the root key it names,
    manifest, approvals and token signed by trusted keys, the token naming the
    manifest and the approvals, and every data file matching the manifest."""
    f = Path(folder)
    probs = []
    try:
        root = json.loads((f / "registry.json").read_text())["root_key"]
        rootk = {root["id"]: PublicKey.from_text(root["public"])}
        kenv = json.loads((f / "trusted-keys.dsse").read_text())
        if not verify_envelope(kenv, rootk):
            return ["trusted-keys.dsse is not signed by the registry's root key"]
        keys = _keys_from(kenv)
        menv = json.loads((f / "manifest.dsse").read_text())
        tenv = json.loads((f / "release-token.dsse").read_text())
    except (OSError, KeyError, ValueError) as e:
        return [f"incomplete release package: {e}"]
    st = payload(menv)
    if st["predicate"]["run"]["operator"]["key"] not in verify_envelope(menv, keys):
        probs.append("manifest.dsse: not signed by its operator's trusted key")
    md = sha256(canonical(st))
    tok = payload(tenv)["predicate"]
    if not verify_envelope(tenv, keys):
        probs.append("release-token.dsse: no valid signature")
    if tok["manifest"] != md:
        probs.append("the release token names another manifest")
    appr = {}
    for p in sorted((f / "approvals").glob("*.dsse")):
        env = json.loads(p.read_text())
        ap = payload(env)["predicate"]
        if ap["approver"]["key"] not in verify_envelope(env, keys):
            probs.append(f"{p.name}: not signed by the approver it names")
        if ap["manifest"] != md:
            probs.append(f"{p.name}: approves another manifest")
        appr[sha256(canonical(env))] = ap
    missing = set(tok["approvals"]) - set(appr)
    if missing:
        probs.append(f"the token cites {len(missing)} approval(s) that are not in the package")
    if not any(a["decision"] == "approve" and a["approver"]["key"] != st["predicate"]["run"]["operator"]["key"]
               for a in appr.values()):
        probs.append("no approval by someone other than the operator")
    want = {s["name"]: s["digest"]["sha256"] for s in st["subject"]}
    for name, d in want.items():
        p = f / "data" / f"{name}.csv"
        if not p.exists():
            probs.append(f"data/{name}.csv is missing")
        elif sha256(p.read_bytes()) != d:
            probs.append(f"data/{name}.csv does not match the signed manifest")
    extra = {p.stem for p in (f / "data").glob("*.csv")} - set(want)
    if extra:
        probs.append(f"files not in the manifest: {sorted(extra)}")
    cp = f / "checkpoint.dsse"
    if cp.exists() and not verify_envelope(json.loads(cp.read_text()), keys):
        probs.append("checkpoint.dsse: signature does not verify")
    return probs


__all__ = ["verify_registry", "verify_package", "KEYS_TYPE"]
