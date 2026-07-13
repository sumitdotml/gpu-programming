# AGENT_MISTAKES

Persistent repository memory for recurring agent/model mistakes.

Initialized on 2026-02-17.

## Usage Rules

- Read this file before any repository edit task.
- Record every detected mistake occurrence.
- Deduplicate by normalized `pattern` + `scope_tags` + `prevention_rule`.
- For repeated patterns, update existing entry fields instead of creating duplicates.

## Required Entry Fields

Every entry must include:

- `id`
- `status` (`active` or `resolved`)
- `severity` (`low`, `medium`, or `high`)
- `scope_tags` (list)
- `pattern`
- `prevention_rule`
- `validation_check`
- `first_seen` (YYYY-MM-DD)
- `last_seen` (YYYY-MM-DD)
- `occurrence_count` (integer >= 1)
- `evidence` (one or more file:line and/or commit refs)

## Entry Template

Use this exact shape for new entries.

```md
### MISTAKE-YYYYMMDD-001
- id: MISTAKE-YYYYMMDD-001
- status: active
- severity: medium
- scope_tags: [code, docs, tests, config, infra, planning]
- pattern: <normalized mistake pattern>
- prevention_rule: <specific action that prevents recurrence>
- validation_check: <deterministic pass/fail check>
- first_seen: YYYY-MM-DD
- last_seen: YYYY-MM-DD
- occurrence_count: 1
- evidence:
  - file:relative/path:line
  - commit:<hash>
```

## Entries

### MISTAKE-20260707-001
- id: MISTAKE-20260707-001
- status: active
- severity: medium
- scope_tags: [docs, planning]
- pattern: public agent guidance included machine-local adjacent repository paths
- prevention_rule: keep local paths and private repo context in ignored CONTEXT.md or tutor_harness.md, and have AGENTS.md only point to those local files
- validation_check: `git status --ignored --short` shows CONTEXT.md and tutor_harness.md ignored, and `AGENTS.md` does not contain literal adjacent repo paths
- first_seen: 2026-07-07
- last_seen: 2026-07-07
- occurrence_count: 1
- evidence:
  - file:AGENTS.md:12
  - file:.gitignore:1

### MISTAKE-20260713-001
- id: MISTAKE-20260713-001
- status: active
- severity: medium
- scope_tags: [learning, tutoring]
- pattern: exercise hint substituted all coordinate values and reduced the remaining work to arithmetic
- prevention_rule: stop after the conceptual or layout hint and ask the learner to construct the indexed expression unless they explicitly request a stronger hint
- validation_check: before sending an exercise hint, verify that the response does not contain the complete substituted expression or leave only mechanical arithmetic
- first_seen: 2026-07-13
- last_seen: 2026-07-13
- occurrence_count: 1
- evidence:
  - file:tutor_harness.md:69
