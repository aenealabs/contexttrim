# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |
| Older   | No        |

We support only the most recent released version. Please upgrade before reporting a vulnerability.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security issues privately through GitHub's [private vulnerability reporting](https://github.com/aenealabs/contexttrim/security/advisories/new): go to the repository's **Security** tab and click **Report a vulnerability**. This keeps the report confidential until a fix is released.

Include:
- A description of the vulnerability and its potential impact
- Steps to reproduce or a minimal proof-of-concept
- The contexttrim version and Python version affected

You can expect an acknowledgment within 48 hours and a resolution or mitigation plan within 14 days. We will credit you in the release notes unless you request otherwise.

## Scope

contexttrim is a pure text/token-accounting library with no network access and no execution of user-supplied code. The attack surface is limited to:

- Untrusted message content driving unexpected behavior in the strategies (`strategies.py`) or token counter (`_tokens.py`)
- Supply-chain attacks via the package itself (we maintain zero dependencies to minimize this risk)
