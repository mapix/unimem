# Security policy

unimem sits in front of all model traffic of your coding agents and stores long-term memory about you and your
projects. Security issues are taken seriously.

## Supported versions

Only the latest release (currently the 0.4.x line) gets security fixes.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report them privately through
[GitHub Security Advisories](https://github.com/mapix/unimem/security/advisories/new) ("Report a vulnerability").
Please include:

- the unimem version (`unimem --version`) and OS;
- the harness and its version, if relevant;
- steps to reproduce, or a proof of concept;
- the impact as you understand it.

You should get an acknowledgement within 7 days. Once a fix is available, the issue will be disclosed in coordination
with you.

## Threat model and known non-goals

These are documented limitations, not vulnerabilities, unless you find a way around their stated mitigations:

- **No authentication on localhost.** The gateway, `/mcp` and the web UI (`/ui`, `/ui/api/*`) have no auth. They are
  meant to be bound to `127.0.0.1` only (the default). Binding to a non-loopback address exposes your memory and lets
  anyone proxy model requests with *their* credentials through your machine.
- **Credentials are forwarded, never stored.** The `Authorization` / `x-api-key` / OAuth headers pass through to the
  configured upstream. They are not written to the database or the logs. A path that persists or logs them **is** a
  vulnerability.
- **Memory poisoning / prompt injection.** Memories are text that ends up in the model's context. The write-time
  injection screen is heuristic and weak out-of-sample, so a malicious note can be stored. Passive injection filters
  instruction-like notes, but explicit `unimem_recall` can still return them. A bypass of the *passive* filter, or a
  way for remote content to plant memories without the agent or user writing them, is in scope.
- **Secret screening is best effort.** Regexes catch common key formats before a memory is written, and episodes are
  redacted. A common credential format that slips through is worth reporting.
- **Upstream URL configuration.** Whoever controls `~/.unimem/config.toml` controls where your traffic goes. Protect
  that file the way you protect your shell rc files.

## Data at rest

`~/.unimem/unimem.db` is an unencrypted SQLite file. Rely on OS-level disk encryption, and keep the file's
permissions restricted to your user.
