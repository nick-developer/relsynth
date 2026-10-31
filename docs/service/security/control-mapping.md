# Control mapping: SOC 2, ISO/IEC 27001:2022, NIST SP 800-53 rev. 5

How the relsynth service supports the controls an organisation operating it will be audited against. relsynth is software the organisation runs; it is not a certified service. The mapping says which product feature supports a control and what evidence the product produces. **Org** marks the part the operating organisation must still do (policy, people, process). The SOC 2 column uses the 2017 Trust Services Criteria (revised points of focus, 2022).

| Control area | SOC 2 TSC | ISO 27001:2022 Annex A | NIST 800-53 r5 | relsynth feature | Evidence the product produces | Org |
|---|---|---|---|---|---|---|
| Identity and SSO | CC6.1 | A.5.16, A.8.5 | IA-2, IA-5, IA-8 | OIDC token checks; SAML via broker; service accounts | config; `test_service_auth.py` results | IdP MFA policy, joiner/mover/leaver |
| Role-based access | CC6.1, CC6.3 | A.5.15, A.5.18, A.8.2, A.8.3 | AC-2, AC-3, AC-6 | per-tenant roles, permission table, ABAC | role table in identity.md; `test_roles_and_refusals` | access reviews (quarterly) |
| Separation of duties | CC6.3 | A.5.3 | AC-5 | approver ≠ run creator, quorum | approvals in audit log | assign roles to different people |
| Tenant isolation | CC6.1, C1.1 | A.8.3, A.8.22 | AC-4, SC-2, SC-4 | tenant-scoped store and API, per-tenant keys, pools, volumes | `test_tenant_isolation_on_every_route`; e2e report | decide tenancy model |
| Encryption at rest | CC6.1, C1.1 | A.8.24 | SC-12, SC-13, SC-28 | AES-256-GCM envelope encryption, KMS integration | `test_service_crypto_policy.py`; key config | key custody, HSM/KMS policy |
| Encryption in transit | CC6.7 | A.8.20, A.8.24 | SC-8 | HSTS; TLS at ingress; verify-full DB connections documented | headers test | certificates, mesh |
| Secrets management | CC6.1 | A.5.17, A.8.24 | IA-5(7), SC-28 | secret references only; Vault/cloud managers; per-tenant prefixes | `test_secret_refs` | secret rotation |
| Privileged access | CC6.2 | A.8.2 | AC-6(5) | platform_admin without data access; tenant_admin limited to service accounts | role table | MFA on admin roles |
| Logging and audit | CC7.2, CC7.3 | A.8.15, A.8.16 | AU-2, AU-3, AU-6, AU-9, AU-12 | JSON logs, hash-chained per-tenant audit log, verify endpoint | `/audit/verify` output; audit export | SIEM, review, retention, WORM copy of head |
| Monitoring | CC7.1, CC7.2 | A.8.16 | SI-4, CA-7 | metrics, traces, health checks, alerts guide | `/metrics`; operations.md | on-call |
| Data minimisation | P4.1, C1.1 | A.5.34, A.8.10, A.8.11 | PT-2, SI-12, SC-28 | no source rows persisted; reports only aggregates; consumers see no privacy items | `test_no_source_row_is_persisted` | DPIA, purpose limitation |
| Release approval | CC8.1, PI1.3 | A.5.12, A.5.14 | AC-21, CM-3 | approval before download; external context requires DP (T07/T08) | approval records, download refusals in audit | approval policy |
| Change management | CC8.1 | A.8.32 | CM-3, CM-4 | CI gates, signed releases, migration jobs, rollback | CI runs, CHANGELOG | CAB |
| Secure development | CC8.1 | A.8.25, A.8.26, A.8.28, A.8.29 | SA-11, SA-15, SR-3 | threat model, ASVS L2 checklist, tests, SAST (ruff S, bandit), pip-audit | threat-model.md, asvs-l2.md, CI logs | developer training |
| Vulnerability management | CC7.1 | A.8.8 | RA-5, SI-2 | pip-audit, Dependabot, SBOM | SBOM, audit reports | patch cadence |
| Penetration testing | CC4.1 | A.8.29 | CA-8 | test plan | pentest-plan.md | commission tests |
| Hardening | CC6.8 | A.8.9 | CM-6, CM-7 | non-root, read-only, no capabilities, restricted Pod Security, NetworkPolicy | rendered chart; dry run under `restricted` | cluster hardening (CIS) |
| Availability | A1.1, A1.2 | A.5.30, A.8.14 | CP-2, CP-10, SC-5 | HA API, resumable jobs, PDB, retries, limits | worker-kill test results | capacity planning |
| Backup and restore | A1.2, A1.3 | A.8.13 | CP-9, CP-10 | backup guide; encrypted artefacts safe off-site | operations.md | schedule, quarterly restore test |
| Supply chain | CC9.2 | A.5.19–A.5.22, A.8.30 | SR-3, SR-4, SR-11 | pinned dependencies, SBOM, Sigstore, SLSA provenance, offline bundle with checksums | release artefacts | supplier reviews |
| Incident response | CC7.3–CC7.5 | A.5.24–A.5.28 | IR-4, IR-5, IR-6 | audit trail, request ids, chain verification | runbook | IR plan |
| Data residency and deployment model | C1.1 | A.5.23 | SA-9 | on-premises or own cloud account, no telemetry, air-gapped bundle | deployment.md | choose region |

## Privacy regulation anchors

- GDPR Art. 25 (data protection by design and default) and Art. 32 (security of processing): encryption, access control, minimisation and audit above.
- GDPR Art. 30 / accountability: the audit log and, with T08, signed manifests per run.
- Releases outside the organisation need differential privacy (user decision, 2026-10-26); the service refuses external contexts until the DP engine and governance build are installed, and the privacy officer's approval is required for internal releases (decision D4).
- T17 (compliance evidence and assurance case) builds on this mapping.

## Sources

AICPA, *2017 Trust Services Criteria for Security, Availability, Processing Integrity, Confidentiality, and Privacy* (revised points of focus 2022). ISO/IEC 27001:2022 and ISO/IEC 27002:2022. NIST SP 800-53 rev. 5 (2020, updated 2023). OWASP ASVS 4.0.3. NIST SP 800-226 (2025) for DP statements.
