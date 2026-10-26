# Security policy

## Reporting a vulnerability

Please do **not** open a public issue, discussion or pull request for a security problem.

Report it privately through GitHub: **Security → Report a vulnerability** on the repository, or directly at <https://github.com/nick-developer/relsynth/security/advisories/new>. Include the version, what an attacker can do, and the steps or a proof of concept that shows it. If the problem involves data, describe it; do not send real personal data.

What to expect:

| Step | Target |
|---|---|
| Acknowledgement | within 5 working days |
| First assessment (accepted or not, severity) | within 15 working days |
| Fix or mitigation for a high or critical issue | within 90 days of the report |

We agree a disclosure date with you, publish a GitHub security advisory with a CVE when the issue is accepted, and credit you unless you prefer otherwise.

## Supported versions

| Version | Security fixes |
|---|---|
| Latest minor release | yes |
| The minor release before it | yes, for 6 months after the newer release |
| Older | no |

## In scope

relsynth reads sensitive databases, so these count as vulnerabilities:

- leaking source rows, credentials or connection strings (into synthetic output, logs, JSON output, diagnostics bundles or exported models);
- a privacy guarantee that does not hold as documented (for example a differential-privacy accounting error);
- code execution or file access outside the configured folders through a spec, an expression, a regular expression or a file name;
- SQL injection through identifiers or values;
- a tampered or unsigned release artefact.

Weaknesses of a statistical method that are documented (for example the disclosure risk of a release without differential privacy) are design questions; please open an ordinary issue for them.
