---
name: git-workflow
description: Use when the user asks to create a git branch, commit local changes, push to a remote, prepare a pull request branch, or repeat the branch-commit-push workflow used for local project changes. This skill guides Codex through inspecting git state, preserving existing work, creating a focused branch, writing a thorough commit message, pushing with upstream tracking, and reporting exact results.
---

# GitWorkflow

Use this skill for routine branch, commit, and push work in a shared local repository.

## Core Rules

- Inspect before acting: run `git status --short --branch`, `git branch --show-current`, and `git remote -v` as needed.
- Preserve user work. Do not discard, reset, checkout over, or amend changes unless the user explicitly asks.
- If the user asks to commit "all local changes", stage the full working tree with `git add -A` after showing or checking the status.
- If unrelated or generated files are present, mention them before committing when the user has not clearly asked for all local changes.
- Use a dedicated branch for substantive work unless the user has already chosen a branch.
- Prefer `git switch -c <branch-name>` for new branches.
- Use clear branch names: lowercase words separated by hyphens.
- Write commit messages that explain intent, scope, behavior impact, and verification.
- Push new branches with upstream tracking: `git push -u origin <branch-name>`.
- If push fails because of sandboxed network access, retry with escalated permission.
- Report the final branch, commit hash, push status, and any remote PR URL.

## Workflow

1. Check repository state.
2. If requested, create and switch to a branch.
3. Stage the intended files.
4. Review the staged summary with `git diff --cached --stat`.
5. Commit with a multi-paragraph message when the change is non-trivial.
6. Verify the working tree is clean with `git status --short`.
7. Push the branch with upstream tracking.
8. Report the result concisely.

## Commit Message Pattern

Use a short imperative subject:

```text
Refactor CFD motion runner into package
```

For non-trivial changes, add body paragraphs covering:

- What changed.
- Why it changed.
- What behavior should remain unchanged.
- How it was verified.
- Any intentional limitations, such as not running slow external integration tests.

Example command shape:

```bash
git commit \
  -m "Refactor CFD motion runner into package" \
  -m "Create a new package entry point while preserving the old script as a fallback." \
  -m "Keep this as a structure-only refactor with no intentional runtime behavior changes." \
  -m "Verified with compile checks, smoke tests, and static inventory comparisons."
```

## Push Handling

Before pushing:

- Confirm the current branch with `git branch --show-current`.
- Confirm the remote with `git remote -v`.
- Prefer `origin` unless the user specifies another remote.

Push command:

```bash
git push -u origin <branch-name>
```

If the remote reports a moved repository, include the new location in the final response. If the remote prints a pull request URL, include it.

## Final Response

Keep the final response short and factual:

- Branch name.
- Commit hash and subject.
- Whether the working tree is clean.
- Whether the push succeeded.
- Pull request URL if available.
