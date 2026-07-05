---
name: rephrase
description: Use when the user wants their request restated, clarified, or repeated back for discussion and confirmation before work begins. This skill is for alignment-first conversations where Codex should understand the task, rephrase it using accurate and professional software, mathematical, or engineering language, surface assumptions or ambiguities, and explicitly wait for the user's confirmation or instruction before implementation, editing, execution, or analysis continues.
---

# Rephrase

Use this skill when the user wants alignment before action.

## Purpose

Restate the user's request precisely, in professional technical language, so the user can confirm that the problem statement, scope, and priorities are understood correctly.

## Workflow

1. Extract the core objective, constraints, and success criteria from the user's message.
2. Rephrase the request in accurate software, mathematical, or engineering language as appropriate to the subject.
3. Preserve the user's intent without narrowing, extending, or reinterpreting the task beyond what was stated.
4. Call out material assumptions, uncertainties, or tradeoffs that affect execution.
5. Stop there and wait for explicit user confirmation or an explicit instruction to proceed.

## Behavioral rules

- Do not start implementation, file edits, command execution, or project work unless the user explicitly instructs you to proceed.
- Do not treat implicit approval as sufficient.
- Prefer precise terminology over casual paraphrase.
- Keep the rephrasing concise, but complete enough to confirm scope and intent.
- When the topic is technical, use disciplined language about algorithms, numerical methods, data flow, interfaces, constraints, and expected outputs.

## Output pattern

Use a structure close to this:

- Restated request
- Key constraints or priorities
- Open assumptions or points to confirm

End by waiting for confirmation rather than advancing into work.
