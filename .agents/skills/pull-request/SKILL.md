---
name: pull-request
description: Open or update a pull request - branch, title, body with Before/After blocks, measurement tables, backports and attribution. Use before writing any PR title or description in this repository.
---

# Pull requests

## Branch

- Stack follow-ups and say so: a PR based on another PR's branch carries that diff when
  squash-merged.
- PRs merge fast. Before pushing to an existing PR or editing its body, check
  `gh pr view N --json state`; if it is merged, start a new branch off `origin/main`.

## Title

PRs are squash-merged with the title as the subject. It is imperative, names the API
involved, and fits on one line of `git log --oneline`: one clause, not a list.

## Body

The body is the design record, kept short, with no section headers unless the change is
large:

1. The issue, in a sentence or two, with its number or "Part of #N".
2. A **Before** and an **After** block: a reproducer and its output on `main` and on the
   branch, or a measurement table.
3. One paragraph on the mechanism and the fix.
4. **Left out**: what was deliberately not done, and why.

A documentation-only PR needs just the source of the material.

Say when the PR:

- touches dispatch (adds, removes or reroutes a method);
- changes behaviour that was working, such as a result type or stored zeros, with the
  reproducer for each change;
- needs a docs update;
- unblocks Base CI, so that the stdlib bump is triggered (code reaches users only when
  JuliaLang/julia bumps its pinned version);
- needs a paired LinearAlgebra PR, in which case CI stays red until both merge;
- is a bug or regression fix that qualifies for a backport.

### What not to report

Do not report what CI checks: no sentence that the tests, either test mode, the
whitespace or GPL check, the ambiguity or Aqua check, the doctests or the docs build pass
or were run, and no Julia build, pass counts or suite selections. Do not say where the
new tests went or that they fail on `main`: the diff and the **Before** block show both.

Do report, under "Not checked by CI:", the checks CI does not run: a bitwise comparison
of results or of line coverage against `main`, a sweep against dense, a dispatch sweep, a
downstream package's test suite, a benchmark. Give the size of the sweep and what differed.

## Measurements

A performance PR includes a before/after table of time and allocations at realistic
sizes. State the size (`n`, `nnz`), the statistic (minimum of N runs) and the thread
count, and keep both columns in one table. Measure `main` and the branch in the same
session on the same machine, each in a fresh process. Bound the memory of large runs
(for example with `systemd-run --user --scope -p MemoryMax=...`), and keep the dense
dimension of a dense result small.

## Backports

Bug and regression fixes only, never new methods or behaviour changes. Never bump compat
on a release branch or push to one. Leave the `backport 1.x` label to the maintainers.

## Attribution

A PR written by an agent says which tool wrote it, with a session link when there is one.
