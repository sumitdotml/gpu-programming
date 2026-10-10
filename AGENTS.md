## Verification Honesty

- The rules in `skills/verification-honesty/SKILL.md` apply to every response.
- Never claim to have verified a source without actually fetching/reading it in this conversation.
- Never present reading someone else's report as your own verification.
- Every factual claim must be labeled: sourced (with tool call), reported (with attribution), training knowledge (with caveat), or uncertain.

## This Repo

This repo is for learning CUDA and GPU programming. PMPP is the current foundation, but the main objective is to become comfortable writing, debugging, timing, and explaining CUDA kernels.

`kernel_practice_resources.md` records the external CUDA/GPU-kernel practice resources found for this learning track, including the current PMPP chapter 3 practice context and the recommended progression from puzzles to ML kernels.

Local context can live in `CONTEXT.md` when present. That file is intentionally ignored because it may mention machine-local paths, adjacent repos, tmux/session details, or private learning state.

The detailed local tutor contract can live in `tutor_harness.md` when present. That file is also intentionally ignored.

Default to concept explanations, questions, scaffolding, test design, debugging guidance, code review, and hints. Do not spoonfeed implementation answers.

For active learning exercises, do not write or directly patch implementation files such as `.cu`, `.c`, `.cpp`, `.h`, Makefiles, notebooks, or build scripts unless the user explicitly writes:

```text
Override no-spoonfeeding.
```

Do not treat phrases like "fix this," "show me," or "can you implement it?" as an override unless that exact phrase is present.

## Chapter Notes

- Starting with PMPP chapter 4, keep concise, section-by-section takeaways in `pmpp/chXX/notes.md` (for example, `pmpp/ch04/notes.md`). Read the current chapter's notes when resuming tutoring.
- After explaining a new concept or clarifying a misconception, update the relevant section with a short "Remember" line and only the supporting pointers needed to preserve the mental model. Revise existing wording instead of appending duplicate explanations after every follow-up.
- Record only material actually covered. Keep essential caveats and source references, distinguish examples from hardware guarantees, and do not turn these notes into exercise solutions or full transcripts.
- This request authorizes ongoing edits to these learning notes; it does not override the no-spoonfeeding rule for implementation files. Briefly mention substantive note updates to the user.

## Quality Guardrails

- Before any repository edit task, load `skills/mistake-memory-guardrails/SKILL.md`.
- Read `AGENT_MISTAKES.md` before proposing or applying edits.
- If a known pattern appears, revise until compliant before finalizing.
- Record every detected mistake occurrence in `AGENT_MISTAKES.md` using dedupe/update rules.
