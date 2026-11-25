---
name: restructure-code
description: Plan and carry out a restructuring of package code - moving code within or between files, naming, operand normalization, merging duplicated kernels, routing rare operand forms through a copy - as a series of small, verified PRs, and decide whether a restructuring is worth its cost. Use when asked to refactor, reorganize, simplify or clean up package code.
---

# Restructuring code

The goal is the architecture in `AGENTS.md`. This is how to get there without changing
behaviour by accident, and how to decide whether to go at all.

## 0. Decide whether it is worth it

Each Julia release ships its own copy of this package, and fixes reach it as backports
cherry-picked from `main`. Code that has moved makes every later backport to an older
release conflict, and buries `git blame` under the move.

- Restructure where the code shows a concrete problem: duplicated loops, routes of one
  operation that disagree, a complexity bug, or dispatch nobody can follow. Not for taste.
- Do not move code that an open bug fix or a pending backport touches; land that first.
- Prefer one move of a region to several, and do the move when the region is quiet.
- Once a pure move or rename has merged, add its commit to a `.git-blame-ignore-revs`
  file so that `git blame` skips it.

## 1. Map before you move

Trace every route of the operation: the entry method, the result container, the operand
conversions, the kernel, and every copy along the way. Write the map down, pinned to a
commit. Then write the plan as a tracking issue with numbered items, one PR each,
behaviour-neutral items first, in merge order. Behaviour changes and performance gaps that
the trace turns up are listed separately and get their own PRs.

## 2. Behaviour-neutral steps

- **Layout and moves.** Move whole top-level definitions, each with the comments above
  it, into sections that follow the dispatch, within a file or into the file that is the
  function's home, and list each file's sections at the top. Change no line of code. Do
  it with a script that parses the files into top-level items, places each by a rule on
  its signature, fails on an unmatched item or an unused rule, and checks that every
  non-blank line appears as often as before across the files. Put the script in the PR
  body so it can be rerun when `main` moves, and ask reviewers to read the layout, not
  the diff. A move between files can change the order of definitions; check the method
  count and ambiguities as for a dispatch change.
- **Shared preambles.** Size checks, destination scaling and factor handling that each
  kernel spells out become one `@inline` helper. Check that the generated code is unchanged.
- **Merge mirrored kernels.** Kernels that differ in one term take the difference as a
  parameter (a `Val`, a function such as `identity`/`transpose`/`adjoint`), and a trivial
  coefficient as `nothing`, so the same specialized loops still compile.
- **Normalize operands once.** Replace one method per combination of wrappers with an
  operand-conversion function that has one method per kind of operand, and one or two
  entry methods on the union. Call the entry function again on the converted operands
  when a more specific method must still see them.
- **Rename.** Name kernels after their operands and result, state the scheme in a comment
  at the top of the file, and rename with a script that matches whole identifiers and
  asserts that no new name is already taken.
- **Make the dispatchers agree.** Every dispatcher of an operation normalizes its input
  the same way. Internal callers go to a private helper, never to a public-signature method.

Renames merge before the layout PR, which is regenerated last.

## 3. Behaviour changes, each in its own PR

- **Route the long tail through a copy.** Add the rare operand form to the operand union so it
  is copied once, in O(nnz), into an existing kernel's input. Measure the common forms; if
  one gets slower, state by how much or keep the special kernel. Record the routes you
  measured and rejected, so they are not measured again.
- **Unify semantics.** When the routes of one operation disagree with each other or with
  dense (zero coefficients, stored zeros, index type, signed zeros, errors), send them to one
  kernel and list each visible difference with a reproducer.
- **Fix the complexity** the trace found, with a before/after table.

## Proving a step neutral

- **Bitwise digest.** Keep one script outside the repository that runs a fixed grid of
  calls (eltypes, wrappers, NaN, Inf and signed zeros, special coefficients) and serializes
  the results. Every PR in the series compares its digest with `main` and reports the
  number of calls compared.
- **Dispatch sweep.** For a change to dispatch, sweep every pair of operand forms and
  compare the method's module, the result's type, value and stored structure, and
  `@allocated` with `main`.
- **Method count.** `count(m -> m.module === SparseArrays, methods(f))` before and after is
  the Before/After block of a dispatch consolidation.
- Consolidating methods onto unions can create ambiguities with Base or LinearAlgebra; the
  ambiguity and Aqua checks decide how far a step can go.

None of this tooling goes in the repository; scripts go in the PR body. The PR states
what was compared and how many cases differed.
