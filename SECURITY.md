# Security Policy

## Supported versions
This repository is a collaboration pipeline and tooling baseline. It does not provide versioned security support guarantees.

If you use this repository as a template, you should define supported versions in your derived project.

## Reporting a vulnerability
If you believe you have found a security vulnerability, please do not open a public issue.

Instead, report it privately:
- Contact: <SECURITY_CONTACT>
- Include: a clear description, reproduction steps, and affected files/versions

You will receive an acknowledgement within 7 days.

## Scope
In scope:
- Accidental inclusion of secrets in the repository
- Unsafe handling of file paths, command execution, or deserialization in tooling
- CI workflow security issues
- Supply-chain risks in dependencies used by tooling

Out of scope:
- Vulnerabilities in third-party dependencies not used by this repository
- Misconfigurations in downstream projects that copy this repository, unless caused by this repository’s defaults

## Sensitive data and secrets
- Do not commit secrets (API keys, tokens, private keys).
- Use environment variables or a secrets manager.
- If a secret is accidentally committed, rotate it immediately and remove it from Git history if necessary.

## Dependency update policy
- Keep tooling dependencies minimal.
- Prefer standard library where possible.
- Review dependency updates before merging.
- CI should run on all PRs before merge.

## Disclosure process
Upon receiving a report:
1. Triage and confirm the issue.
2. Assess impact and define remediation.
3. Patch and validate with CI.
4. Document the fix in the changelog and/or release notes, as appropriate.
