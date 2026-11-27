---
name: evidence-driven-coding
description: 'Use when implementing, debugging, or reviewing code changes that need disciplined local investigation, falsifiable hypotheses, minimal edits, focused validation, and evidence-based escalation.'
argument-hint: 'Describe the code path, failing behavior, or requested change.'
user-invocable: true
---

# Evidence-Driven Coding

## Purpose

Guide a coding task from a concrete anchor to a verified result while keeping investigation local, edits small, and conclusions testable.

## When to Use

- Implement a requested behavior in an existing codebase.
- Debug a failing test, command, diagnostic, or runtime behavior.
- Review a change for regressions, missing tests, or behavioral risk.
- Refactor code when the controlling abstraction is not immediately obvious.

## Procedure

1. **Find the concrete anchor.** Start from the named file, symbol, failing behavior, command, test, or nearby implementation. If none is named, perform one targeted search.
2. **Read the smallest useful context.** Inspect the owning abstraction and one nearby test, call site, or implementation. If the anchor only forwards or registers behavior, follow one hop to the code that computes, mutates, or controls it.
3. **State a local hypothesis.** Before editing, write down one falsifiable explanation for the behavior and one cheap check that could disconfirm it.
4. **Choose the smallest testable edit.** Prefer existing helpers, APIs, conventions, and public interfaces. Preserve unrelated user changes and avoid broad refactors. If confidence is incomplete, make a reversible probe that exposes the missing type, control-flow gap, or validation failure.
5. **Validate immediately.** After the first substantive edit, run the narrowest executable check available: the failing behavior check, a focused test, or a scoped compile/lint/typecheck. Do not broaden exploration before this check.
6. **Interpret the result.**
   - If it fails and supports the hypothesis, repair the same slice and rerun the same check.
   - If it falsifies the hypothesis, take one nearby hop toward the direct controller and revise the hypothesis.
   - If it is ambiguous, perform one nearby disambiguating read or inspect a neighboring test/call site, then choose local repair or the one-hop move.
   - If it succeeds but adjacent work remains, make the smallest follow-up edit and rerun focused validation.
7. **Complete the verification.** Run at least one post-edit executable check whenever the environment supports one. Use a diff-only review only when no focused command is available. Check relevant diagnostics, tests, and user-visible behavior.
8. **Report precisely.** Summarize the root cause or behavior, files changed, validation performed, and any remaining test gaps or residual risk. Link workspace files with clickable paths when presenting references.

## Decision Rules

- Stop searching once you can name the controlling path, hypothesis, discriminating check, and small edit. Additional broad exploration is drift.
- Prefer root-cause fixes over surface patches, but keep the change within the owning module and behavioral surface.
- Scale tests to blast radius: focused coverage for narrow changes; broader coverage for shared contracts or user-facing workflows.
- Do not fix unrelated failures. Mention them separately when they prevent complete validation.
- Do not revert changes you did not make, commit changes, or create branches unless explicitly requested.
- Ask a clarifying question only when the desired outcome, scope, or safe validation cannot be inferred from local evidence; otherwise proceed with a stated assumption.

## Review Mode

When the task is a review, lead with findings ordered by severity. Ground each finding in a workspace file and explain the behavioral risk. Then list open questions, test gaps, and a brief change summary.
