# Built-in release rules of relsynth (Rego v1, OPA 1.x).
#
# Query: data.relsynth.release.decision
# Input: see docs/governance/design.md, section 8.  The Python side converts
# the policy's epsilon caps into zCDP rho at the domain's delta (Prop. 800)
# and passes the ledger state as numbers, so this file only compares.
#
# These rules restate invariants that relsynth also enforces in code; an
# organisation's own modules (package relsynth.org) can add deny reasons
# but cannot remove these.  relsynth.governance.decide.builtin is a Python
# port of this file; CI runs `opa test` on release_test.rego and checks that
# both give the same decision on the fixtures in tests/data/governance/.
package relsynth.release

import rego.v1

ctx := input.policy.contexts[input.request.context]

m := input.manifest

default external := true

external := ctx.external

# --- reasons to refuse ------------------------------------------------------

deny contains msg if {
	not ctx
	msg := sprintf("context %q is not defined in the organisation policy", [input.request.context])
}

deny contains msg if {
	external
	m.privacy.mode != "dp"
	msg := sprintf("context %q is outside the organisation: differential privacy is required, and this run has none", [input.request.context])
}

deny contains msg if {
	m.privacy.mode == "dp"
	m.source.domain != input.ledger.domain
	msg := sprintf("the run's domain %q does not match the ledger's %q", [m.source.domain, input.ledger.domain])
}

deny contains msg if {
	m.privacy.mode == "dp"
	unit := input.policy.domains[m.source.domain].unit.entity
	m.privacy.unit != unit
	msg := sprintf("the run protects %q but the policy's privacy unit for domain %q is %q", [m.privacy.unit, m.source.domain, unit])
}

deny contains msg if {
	external
	not m.spec.public_facts_only
	msg := "an external DP release must use only public schema facts or charged statistics (spec.public_facts_only is not true)"
}

deny contains msg if {
	m.privacy.mode == "dp"
	input.ledger.run_rho > input.ledger.per_release_rho_cap
	msg := sprintf("this release spends rho %v, above the per-release cap %v (epsilon %v at delta %v)", [input.ledger.run_rho, input.ledger.per_release_rho_cap, input.ledger.per_release_epsilon_cap, input.ledger.delta])
}

deny contains msg if {
	m.privacy.mode == "dp"
	input.ledger.domain_rho_after > input.ledger.lifetime_rho_cap
	msg := sprintf("domain %q would reach rho %v, above its lifetime cap %v (epsilon %v at delta %v)", [input.ledger.domain, input.ledger.domain_rho_after, input.ledger.lifetime_rho_cap, input.ledger.lifetime_epsilon_cap, input.ledger.delta])
}

deny contains msg if {
	m.privacy.mode == "dp"
	input.ledger.period_rho_cap
	input.ledger.period_rho_after > input.ledger.period_rho_cap
	msg := sprintf("domain %q would spend rho %v this period, above the period cap %v", [input.ledger.domain, input.ledger.period_rho_after, input.ledger.period_rho_cap])
}

deny contains msg if {
	verdict := object.get(m, ["verdicts", input.request.context], "missing")
	verdict in {"fail", "blocked", "missing"}
	msg := sprintf("the privacy verdict for context %q is %q; it cannot be released or approved", [input.request.context, verdict])
}

deny contains msg if {
	some check in input.privacy_checks
	check.status == "fail"
	msg := sprintf("%s: %v is above the limit %v", [check.attack, check.value, check.limit])
}

deny contains msg if {
	min_rate := input.policy.quality.min_pass_rate
	input.quality.pass_rate < min_rate
	msg := sprintf("quality pass rate %v is below the organisation's minimum %v", [input.quality.pass_rate, min_rate])
}

deny contains msg if {
	not input.request.target in object.get(ctx, "allowed_targets", [])
	msg := sprintf("target %q is not allowed for context %q", [input.request.target, input.request.context])
}

deny contains msg if {
	external
	some r in m.reports
	r.digest.sha256 in input.request.package
	not r.external_safe
	msg := sprintf("the %s report was computed on real data without DP and cannot leave the organisation", [r.kind])
}

# --- approvals --------------------------------------------------------------

approval_required if {
	m.verdicts[input.request.context] in ctx.approval.on_verdicts
}

approval_required if external

valid_approvals contains a.key if {
	some a in input.approvals
	a.decision == "approve"
	a.manifest == input.manifest_digest
	some role in a.roles
	role in ctx.approval.approver_roles
	not separation_violated(a)
	not a.expired
}

separation_violated(a) if {
	object.get(input.policy, ["approvals", "separation_of_duties"], true)
	a.key == m.run.operator.key
}

deny contains msg if {
	approval_required
	need := object.get(ctx.approval, "quorum", 1)
	count(valid_approvals) < need
	msg := sprintf("context %q needs %d approval(s) from %v; %d valid", [input.request.context, need, ctx.approval.approver_roles, count(valid_approvals)])
}

deny contains msg if {
	some a in input.approvals
	a.decision == "reject"
	a.manifest == input.manifest_digest
	msg := sprintf("rejected by %s: %s", [a.approver, object.get(a, "reason", "no reason given")])
}

# --- organisation rules -----------------------------------------------------

deny contains msg if {
	some msg in data.relsynth.org.deny
}

decision := {
	"allow": count(deny) == 0,
	"approval_required": approval_required,
	"reasons": sort([r | some r in deny]),
}

default approval_required := false
