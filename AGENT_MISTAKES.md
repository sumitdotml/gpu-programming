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

### MISTAKE-20260923-001
- id: MISTAKE-20260923-001
- status: active
- severity: medium
- scope_tags: [learning, tutoring]
- pattern: exercise question was built on an unverified premise about code structure (claimed one k-loop index expression changes between the row and column matmul kernels, when both keep the same flat form and only the source of i and j changes)
- prevention_rule: before posing a comparison question about how code changes between two exercise variants, privately derive both variants and confirm the premise holds
- validation_check: every "which part stays the same / which changes" question is checked against a private derivation of both versions before sending
- first_seen: 2026-09-23
- last_seen: 2026-09-23
- occurrence_count: 1
- evidence:
  - file:pmpp/matmul_thread_per_row.cu:85

### MISTAKE-20260923-002
- id: MISTAKE-20260923-002
- status: active
- severity: medium
- scope_tags: [tooling, docs]
- pattern: recommended a shell command for the user to run without testing it on their platform (macOS BSD sed with GNU-only \b word boundaries, which silently matches nothing)
- prevention_rule: before handing the user a file-editing shell command, run it against a scratch copy on the same machine and confirm the diff
- validation_check: any suggested in-place edit command has been executed on a scratch copy in this session and produced the expected change
- first_seen: 2026-09-23
- last_seen: 2026-09-23
- occurrence_count: 1
- evidence:
  - file:pmpp/matmul_thread_per_col.cu:11

### MISTAKE-20260926-001
- id: MISTAKE-20260926-001
- status: active
- severity: low
- scope_tags: [docs, learning]
- pattern: first draft of a hand-built interactive SVG learning artifact shipped layout and interaction defects (click target covered by later-drawn siblings; rotated kernel label overlapping block captions) that were only caught by rendering and driving the page
- prevention_rule: before reporting an interactive HTML artifact as done, render every step in a real browser at desktop and phone widths, in light and dark, and drive the controls with real input events
- validation_check: a scripted headless-browser run clicks each control, reports no console errors, no horizontal overflow at 390px, and screenshots of each state show no overlapping text
- first_seen: 2026-09-26
- last_seen: 2026-09-26
- occurrence_count: 1
- evidence:
  - file:artifacts/pmpp/ch03/matmul-host-device-flow.html:361
  - file:artifacts/pmpp/ch03/matmul-host-device-flow.html:414

### MISTAKE-20260926-002
- id: MISTAKE-20260926-002
- status: active
- severity: medium
- scope_tags: [docs, learning]
- pattern: teaching artifact used overbroad or ambiguous wording for pointer location, copy semantics, and matrix indexing
- prevention_rule: compare each teaching claim with the specific host function, kernel indexing, and CUDA API behavior before publishing; avoid universal claims and wording that implies elementwise matmul or movement instead of copying
- validation_check: artifact descriptions identify &M_d as host-side only in matmul(), say cudaMemcpy copies values, and describe M and N accesses without implying matching indices
- first_seen: 2026-09-26
- last_seen: 2026-09-26
- occurrence_count: 1
- evidence:
  - file:artifacts/pmpp/ch03/matmul-host-device-flow.html:221
  - file:artifacts/pmpp/ch03/matmul-host-device-flow.html:253
  - file:artifacts/pmpp/ch03/matmul-host-device-flow.html:259
