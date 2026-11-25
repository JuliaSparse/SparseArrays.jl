---
name: restructure-code
description: Reorganize a source file or a family of methods - layout, naming, operand normalization, merging duplicated kernels, routing rare wrappers through a copy - as a series of small, verified PRs. Use when asked to refactor, simplify, clean up or review the structure of package code.
---

# Restructuring code

The goal is the architecture in `AGENTS.md`: few entry methods, operands normalized in one
place, kernels named after what they take and return, rare cases routed through a copy,
and one set of semantics per operation. This is how to get there without changing
behaviour by accident.

## 1. Map before you move

Trace every route of the operation: the entry method, the result container, the operand
conversions, the kernel, and every copy along the way. Write the map down, pinned to a
commit. Then write the plan as a tracking issue with numbered items, one PR each,
behaviour-neutral items first, in merge order. Behaviour changes and performance gaps that
the trace turns up are listed separately and get their own PRs.

## 2. Behaviour-neutral steps

- **Layout.** Move whole top-level definitions, each with the comments above it, into
  sections that follow the dispatch, and list the sections at the top of the file. Change
  no line of code. Do it with a script that parses the file into top-level items, places
  each by a rule on its signature, fails on an unmatched item or an unused rule, and
  checks that every non-blank line appears as often as before. Put the script in the PR so
  it can be rerun when `main` moves, and ask reviewers to read the layout, not the diff.
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

- **Route the long tail through a copy.** Add the rare wrapper to the operand union so it
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

Commit none of this tooling. The PR states what was compared and how many cases differed.
