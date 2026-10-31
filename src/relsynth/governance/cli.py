# SPDX-License-Identifier: Apache-2.0
"""`relsynth gov ...`: registry, keys, policy, budget, approvals and releases.

    relsynth gov keys new dpo.key --name "Data Protection Officer"
    relsynth gov init registry/ --owner-key dpo.key --policy-out policy.yaml
    relsynth gov policy check policy.yaml
    relsynth gov policy sign policy.yaml --registry registry/ --key dpo.key
    relsynth gov keys trust analyst.pub.json --roles operator --registry registry/ --owner-key dpo.key
    relsynth run spec.yaml --registry registry/ --context internal --target dev-sandbox --source-id dwh --key analyst.key
    relsynth gov ls --registry registry/
    relsynth gov show RUN --registry registry/
    relsynth gov budget customers --registry registry/
    relsynth gov approve RUN --context internal --key dpo.key --registry registry/
    relsynth gov release RUN --context internal --target dev-sandbox --key analyst.key --out package/
    relsynth gov revoke RUN --reason "..." --key dpo.key
    relsynth gov verify registry/            (or a release package folder)

The registry folder may also come from the environment variable RELSYNTH_REGISTRY.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..errors import EXIT_GATE, EXIT_OK, EXIT_SPEC, GovernanceError


def _reg(a):
    from .registry import LocalRegistry
    root = getattr(a, "registry", None) or os.environ.get("RELSYNTH_REGISTRY")
    if not root:
        raise GovernanceError("give --registry or set RELSYNTH_REGISTRY")
    return LocalRegistry(root)


def _key(path):
    from .sign import Signer
    return Signer.load(path)


def cmd_keys(a, out):
    from .sign import PublicKey, Signer
    if a.action == "new":
        if not a.path:
            raise GovernanceError("give the path of the new private key")
        if Path(a.path).exists():
            raise GovernanceError(f"{a.path} exists; refusing to overwrite a key")
        s = Signer.generate(a.name or Path(a.path).stem)
        pp = os.environ.get("RELSYNTH_KEY_PASSPHRASE")
        s.save(a.path, pp.encode() if pp else None)
        out.put(key={"id": s.id, "name": s.name, "private": a.path, "public": str(Path(a.path).with_suffix(".pub.json"))})
        out.line(f"new key {s.id} ({s.name})\n  private: {a.path} (mode 0600; keep it secret)\n"
                 f"  public:  {Path(a.path).with_suffix('.pub.json')}")
        return EXIT_OK
    reg = _reg(a)
    if a.action == "list":
        tr = reg.trusted()
        rows = [{"id": k, "name": v["name"], "roles": v["roles"]} for k, v in tr.items()]
        out.put(keys=rows)
        for r in rows:
            out.line(f"{r['id']}  {r['name']}  roles: {', '.join(r['roles'])}")
        return EXIT_OK
    if a.action == "trust":
        if not (a.path and a.owner_key and a.roles):
            raise GovernanceError("keys trust needs the public key file, --owner-key and --roles")
        meta = json.loads(Path(a.path).read_text())
        pk = PublicKey.from_text(meta["public"])
        seq = reg.trust(_key(a.owner_key), pk, a.name or meta.get("name", pk.id), a.roles.split(","))
        out.put(trusted={"id": pk.id, "seq": seq})
        out.line(f"trusted {pk.id} as {a.roles} (log entry {seq})")
        return EXIT_OK
    raise GovernanceError(f"unknown action {a.action}")


def cmd_init(a, out):
    from .policy import skeleton
    from .registry import LocalRegistry
    from .sign import Signer
    owner = _key(a.owner_key)
    reg = LocalRegistry.create(a.path, owner, a.owner_role)
    out.put(registry={"path": a.path, "id": reg.id, "root_key": owner.id})
    out.line(f"registry {reg.id} at {a.path}; root key {owner.id} ({owner.name}, role {a.owner_role})")
    if a.policy_out:
        if Path(a.policy_out).exists():
            out.line(f"{a.policy_out} exists; no skeleton written")
        else:
            Path(a.policy_out).write_text(skeleton(owner=owner.name), encoding="utf-8")
            out.line(f"policy skeleton -> {a.policy_out}: fill in every TODO (relsynth sets no epsilon), then "
                     "`relsynth gov policy sign`")
    _ = Signer
    return EXIT_OK


def cmd_policy(a, out):
    from . import policy as P
    if a.action == "check":
        import yaml
        doc = yaml.safe_load(Path(a.path).read_text(encoding="utf-8"))
        probs = P.validate(doc)
        out.put(valid=not probs, problems=probs)
        if probs:
            out.line("the policy is invalid:")
            for p in probs:
                out.line(f"  - {p}")
            return EXIT_SPEC
        pol = P.load(doc)
        rows = {}
        for d in pol.domains:
            c = pol.caps(d)
            rows[d] = c.to_dict()
            out.line(f"domain {d}: unit {pol.domains[d]['unit']['entity']}, delta {c.delta:g}, lifetime epsilon "
                     f"{c.lifetime_epsilon:g} = rho {c.lifetime_rho:.6g}"
                     + (f", per release epsilon {c.per_release_epsilon:g} = rho {c.per_release_rho:.6g}"
                        if c.per_release_rho else "")
                     + (f", period {pol.domains[d]['budget']['period']['length']} epsilon {c.period_epsilon:g} "
                        "(a throttle, not a guarantee)" if c.period else ""))
        for n, c in pol.contexts.items():
            out.line(f"context {n}: {'external, DP required' if c['external'] else 'internal'}; approval on "
                     f"{c['approval']['on_verdicts']} by {c['approval']['approver_roles']}")
        out.put(caps=rows, digest=pol.digest)
        out.line(f"valid; digest {pol.digest}")
        return EXIT_OK
    if a.action == "sign":
        reg = _reg(a)
        pol = P.load(a.path)
        seq = P.sign(pol, _key(a.key), reg)
        reg.checkpoint(_key(a.key))
        out.put(policy={"digest": pol.digest, "version": pol.version, "seq": seq})
        out.line(f"signed policy version {pol.version} ({pol.digest[:12]}…), log entry {seq}")
        return EXIT_OK
    if a.action == "show":
        pol = P.current(_reg(a))
        out.put(policy=pol.doc, digest=pol.digest)
        out.line(json.dumps(pol.doc, indent=1))
        return EXIT_OK
    raise GovernanceError(f"unknown action {a.action}")


def cmd_ls(a, out):
    from . import runs
    rows = runs(_reg(a), a.domain, a.state)
    out.put(runs=rows)
    for r in rows:
        out.line(f"{r['run']}  {r['state']:<9} {r['mode']:<4} {r['context']:<10} {r['domain'] or '-':<12} {r['time']}")
    if not rows:
        out.line("no runs")
    return EXIT_OK


def cmd_show(a, out):
    from . import show
    s = show(_reg(a), a.run)
    out.put(**{k: v for k, v in s.items()})
    p = s["statement"]["predicate"]
    out.line(f"run {p['run']['id']} by {p['run']['operator']['name']} ({p['run']['started']} – {p['run']['finished']})")
    out.line(f"  manifest {s['manifest']}  log entry {s['seq']}{'  REVOKED' if s['revoked'] else ''}")
    out.line(f"  source {p['source']['id']} (domain {p['source']['domain'] or '-'}); release to "
             f"{p['release']['context']}/{p['release']['target']}")
    pr = p["privacy"]
    if pr["mode"] == "dp":
        out.line(f"  DP: unit {pr['unit']}, epsilon {pr['stated']['epsilon']} at delta {pr['stated']['delta']}; "
                 f"domain cumulative epsilon {pr['ledger']['domain_epsilon']}")
    else:
        out.line("  no differential privacy")
    out.line(f"  verdicts: {p.get('verdicts') or 'none (no privacy assessment)'}")
    for c, d in p["policy"]["decisions"].items():
        out.line(f"  policy at run time ({c}): {'allow' if d['allow'] else 'not yet'}"
                 + "".join(f"\n    - {r}" for r in d["reasons"]))
    for ap in s["approvals"]:
        out.line(f"  {ap['decision']} by {ap['approver']['name']} for {ap['context']} at {ap['time']}"
                 + (f": {ap['reason']}" if ap.get("reason") else ""))
    for r in s["releases"]:
        out.line(f"  released to {r['context']}/{r['target']} at {r['time']}")
    return EXIT_OK


def cmd_budget(a, out):
    from .ledger import Ledger
    from .policy import current
    reg = _reg(a)
    pol = current(reg)
    doms = [a.domain] if a.domain else list(pol.domains)
    res = {}
    for d in doms:
        if d not in pol.domains:
            raise GovernanceError(f"no domain {d!r} in the policy")
        s = Ledger(reg, pol).summary(d)
        res[d] = s
        out.line(f"domain {d}: spent rho {s['rho']:.6g} = epsilon {s['epsilon']:.4g} at delta {s['delta']:g} "
                 f"(lifetime cap epsilon {s['lifetime_epsilon_cap']:g}, rho {s['lifetime_rho_cap']:.6g}); "
                 f"remaining rho {s['remaining_rho']:.6g}")
        if "period_rho" in s:
            out.line(f"  this period: rho {s['period_rho']:.6g} of {s['period_rho_cap']:.6g} (a throttle)")
        for run, v in s["runs"].items():
            out.line(f"  {run}  {v['state']:<8} rho {v['rho']:.6g}  {v['time']}")
        if s["nondp_releases"]:
            out.line(f"  {len(s['nondp_releases'])} release(s) of these people were made without DP; every DP "
                     "statement of this domain says so")
    out.put(budget=res)
    return EXIT_OK


def cmd_approve(a, out):
    from .manifest import approve
    from .policy import current
    reg = _reg(a)
    seq = approve(reg, current(reg), a.run, a.context, _key(a.key), reject=a.reject, reason=a.reason or "")
    reg.checkpoint(_key(a.key))
    out.put(approval={"seq": seq, "decision": "reject" if a.reject else "approve"})
    out.line(f"{'rejected' if a.reject else 'approved'} {a.run} for {a.context} (log entry {seq})")
    return EXIT_OK


def cmd_release(a, out):
    from ..errors import ReleaseDenied
    from . import release
    try:
        res = release(_reg(a), a.run, a.context, a.target, _key(a.key), out=a.out)
    except ReleaseDenied as e:
        out.put(allow=False, reasons=e.reasons)
        out.line(f"refused: {a.run} to {a.context}/{a.target}")
        for r in e.reasons:
            out.line(f"  - {r}")
        return EXIT_GATE
    out.put(**res)
    out.line(f"released {a.run} to {a.context}/{a.target}; token {res['token'][:12]}…"
             + (f"; package -> {a.out}" if a.out else ""))
    return EXIT_OK


def cmd_revoke(a, out):
    from .manifest import revoke
    reg = _reg(a)
    seq = revoke(reg, a.run, _key(a.key), a.reason)
    reg.checkpoint(_key(a.key))
    out.put(revocation={"seq": seq})
    out.line(f"revoked {a.run} (log entry {seq}); no budget is refunded")
    return EXIT_OK


def cmd_verify(a, out):
    from .registry import LocalRegistry
    from .verify import verify_package, verify_registry
    p = Path(a.path)
    if (p / "config.json").exists():
        probs = verify_registry(LocalRegistry(p))
        what = "registry"
    else:
        probs = verify_package(p)
        what = "release package"
    out.put(valid=not probs, problems=probs, kind=what)
    if probs:
        out.line(f"{what} {p}: {len(probs)} problem(s); the first is where it was changed:")
        for x in probs:
            out.line(f"  - {x}")
        return EXIT_GATE
    out.line(f"{what} {p}: intact (chain, signatures, digests)")
    return EXIT_OK


def cmd_checkpoint(a, out):
    reg = _reg(a)
    env = reg.checkpoint(_key(a.key))
    from .sign import payload
    p = payload(env)["predicate"]
    out.put(checkpoint=p)
    out.line(f"checkpoint at log entry {p['seq']}: {p['hash']}")
    return EXIT_OK


def add_parser(sub, common):
    g = sub.add_parser("gov", help="governance: registry, keys, policy, privacy budget, approvals and releases")
    gs = g.add_subparsers(dest="gov_cmd", required=True, metavar="ACTION")

    def reg_arg(p):
        p.add_argument("--registry", help="registry folder (default: $RELSYNTH_REGISTRY)")

    s = gs.add_parser("keys", help="create, list or trust Ed25519 signing keys")
    s.add_argument("action", choices=["new", "list", "trust"])
    s.add_argument("path", nargs="?", help="new: private key path; trust: a .pub.json file")
    s.add_argument("--name", help="display name of the key's owner")
    s.add_argument("--roles", help="trust: comma-separated roles, e.g. dpo or operator")
    s.add_argument("--owner-key", help="trust: the privacy owner's private key (signs the trusted list)")
    reg_arg(s)
    s.set_defaults(fn=cmd_keys)

    s = gs.add_parser("init", help="create a registry whose root of trust is the privacy owner's key")
    s.add_argument("path")
    s.add_argument("--owner-key", required=True, help="the privacy owner's private key (gov keys new)")
    s.add_argument("--owner-role", default="dpo")
    s.add_argument("--policy-out", help="also write a policy skeleton (no budgets) here")
    s.set_defaults(fn=cmd_init)

    s = gs.add_parser("policy", help="check, sign or show the organisation policy")
    s.add_argument("action", choices=["check", "sign", "show"])
    s.add_argument("path", nargs="?", help="the policy file (check, sign)")
    s.add_argument("--key", help="sign: the privacy owner's private key")
    reg_arg(s)
    s.set_defaults(fn=cmd_policy)

    s = gs.add_parser("ls", help="runs and their states")
    s.add_argument("--domain")
    s.add_argument("--state", choices=["assessed", "approved", "rejected", "released", "revoked"])
    reg_arg(s)
    s.set_defaults(fn=cmd_ls)

    s = gs.add_parser("show", help="a run's manifest, DP figures, verdicts, approvals and decisions")
    s.add_argument("run", help="run id or manifest digest (a unique prefix is enough)")
    reg_arg(s)
    s.set_defaults(fn=cmd_show)

    s = gs.add_parser("budget", help="the privacy budget: spent, reserved, caps, non-DP releases")
    s.add_argument("domain", nargs="?")
    reg_arg(s)
    s.set_defaults(fn=cmd_budget)

    s = gs.add_parser("approve", help="sign an approval (or --reject) of a run for a context")
    s.add_argument("run")
    s.add_argument("--context", required=True)
    s.add_argument("--key", required=True, help="the approver's private key")
    s.add_argument("--reject", action="store_true")
    s.add_argument("--reason")
    reg_arg(s)
    s.set_defaults(fn=cmd_approve)

    s = gs.add_parser("release", help="evaluate the policy and, on allow, write a signed release package")
    s.add_argument("run")
    s.add_argument("--context", required=True)
    s.add_argument("--target", required=True, help="target id from the policy's allowed_targets")
    s.add_argument("--key", required=True, help="the releasing operator's private key")
    s.add_argument("--out", help="folder for the release package")
    reg_arg(s)
    s.set_defaults(fn=cmd_release)

    s = gs.add_parser("revoke", help="revoke a run (no budget is refunded)")
    s.add_argument("run")
    s.add_argument("--reason", required=True)
    s.add_argument("--key", required=True)
    reg_arg(s)
    s.set_defaults(fn=cmd_revoke)

    s = gs.add_parser("verify", help="check a registry or a release package for tampering")
    s.add_argument("path")
    s.set_defaults(fn=cmd_verify)

    s = gs.add_parser("checkpoint", help="sign the registry's current head")
    s.add_argument("--key", required=True)
    reg_arg(s)
    s.set_defaults(fn=cmd_checkpoint)

    for sp in gs.choices.values():
        common(sp)
    return g
