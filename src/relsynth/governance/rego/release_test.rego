# Tests of release.rego (`opa test src/relsynth/governance/rego/`).  The same
# cases, and more, run against the Python port through the fixtures in
# tests/data/governance/decisions.json (tests/unit/governance/test_policy_parity.py).
package relsynth.release_test

import rego.v1

import data.relsynth.release

policy := {
	"contexts": {
		"internal": {
			"external": false, "dp_required": false,
			"approval": {"on_verdicts": ["review"], "approver_roles": ["dpo"], "quorum": 1},
			"allowed_targets": ["dev"],
		},
		"partner": {
			"external": true, "dp_required": true,
			"approval": {"on_verdicts": ["pass", "review"], "approver_roles": ["dpo"], "quorum": 1},
			"allowed_targets": ["sftp"],
		},
	},
	"domains": {"customers": {"unit": {"entity": "Customer"}}},
	"quality": {"min_pass_rate": 0.95},
	"approvals": {"separation_of_duties": true},
}

dp_manifest := {
	"privacy": {"mode": "dp", "unit": "Customer"},
	"source": {"domain": "customers"},
	"spec": {"public_facts_only": true},
	"verdicts": {"partner": "pass", "internal": "pass"},
	"run": {"operator": {"key": "op"}},
	"reports": [],
}

ledger_ok := {
	"domain": "customers", "delta": 1e-9, "run_rho": 0.05, "domain_rho_after": 0.08,
	"lifetime_rho_cap": 0.435, "lifetime_epsilon_cap": 6, "per_release_rho_cap": 0.056,
	"per_release_epsilon_cap": 2,
}

approval := {"key": "dpo1", "approver": "Dana", "decision": "approve", "manifest": "m", "roles": ["dpo"], "expired": false}

base := {
	"policy": policy, "manifest": dp_manifest, "manifest_digest": "m", "ledger": ledger_ok,
	"privacy_checks": [], "quality": {"pass_rate": 0.99}, "approvals": [approval],
	"request": {"action": "release", "context": "partner", "target": "sftp", "package": []},
}

test_dp_release_with_approval_is_allowed if {
	d := release.decision with input as base
	d.allow
	d.approval_required
	d.reasons == []
}

test_partner_without_dp_is_denied if {
	m := object.union(dp_manifest, {"privacy": {"mode": "none"}})
	d := release.decision with input as object.union(base, {"manifest": m})
	not d.allow
	some r in d.reasons
	contains(r, "differential privacy is required, and this run has none")
}

test_lifetime_cap_is_enforced if {
	l := object.union(ledger_ok, {"domain_rho_after": 0.5})
	d := release.decision with input as object.union(base, {"ledger": l})
	not d.allow
	some r in d.reasons
	startswith(r, `domain "customers" would reach rho 0.5, above its lifetime cap 0.435`)
}

test_external_needs_an_approval_not_by_the_operator if {
	a := object.union(approval, {"key": "op"})
	d := release.decision with input as object.union(base, {"approvals": [a]})
	not d.allow
	some r in d.reasons
	contains(r, "needs 1 approval(s)")
}

test_failed_verdict_is_denied if {
	m := object.union(dp_manifest, {"verdicts": {"partner": "fail"}})
	d := release.decision with input as object.union(base, {"manifest": m})
	not d.allow
}

test_internal_pass_needs_no_approval if {
	m := object.union(dp_manifest, {"privacy": {"mode": "none"}})
	r := {"action": "release", "context": "internal", "target": "dev", "package": []}
	d := release.decision with input as object.union(base, {"manifest": m, "request": r, "approvals": [], "ledger": {}})
	d.allow
	not d.approval_required
}
