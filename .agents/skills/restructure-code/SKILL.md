---
name: restructure-code
description: Reorganize a SparseArrays.jl source file or a family of methods (layout, naming, operand normalization, merging duplicate kernels, routing rare wrappers through a copy) the way the src/matmul.jl series #1195-#1213 did. Use when asked to refactor, simplify, clean up or review the structure of src/ code.
---

# Restructuring code

The model is the `src/matmul.jl` series: the plan in #1195, then #1196-#1213. The file went
from 43 `*` methods spread over unrelated parts of the file to 28, in sections that follow
the dispatch, with every kernel named after its operands. Each step was its own PR and
was checked against `main`. The target architecture is in the Architecture section of
`AGENTS.md`; this is how to get there.

## 1. Map before you move

Trace every route of the operation: the entry method, the result container, the operand
conversions, the kernel, and every copy on the way. Write the map down (an issue or an
artifact), with line numbers at a named commit. Then write the plan as a tracking issue
with numbered items, each one PR, behaviour-neutral items first, in merge order.
Behaviour changes and performance gaps that the trace turns up are listed separately and
go in their own PRs.

## 2. The behaviour-neutral steps

Typical items, in the order #1195 used:

- **Layout** (#1198). Move whole top-level definitions, each with the comments above it,
  into sections in dispatch order, and list the sections at the top of the file. Change
  no line of code. Do it with a script that parses the file into top-level items, places
  each by a signature rule, fails on an unmatched item or an unused rule, and checks that
  every non-blank line appears as often as before. Put the script in the PR so it can be
  rerun when `main` moves, and ask reviewers to read the layout rather than the diff.
- **Shared preambles** (#1196). The size checks, `β` scaling and `α` factor that each
  kernel spells out become one `@inline` helper. Check that the LLVM IR of the kernels is
  unchanged apart from block order.
- **Merge mirrored kernels** (#1197). Two kernels that differ in one term take the
  difference as a parameter (`Val(left)`, a `tfun`), and a unit `α` or zero `β` as
  `nothing`, so the same specialized loops still compile.
- **Normalize operands once** (#1200, #1208). Replace one method per pair of wrappers
  with an `_<op>_operand` function that has one method per kind of operand, and one or two
  entry methods on the union. Call the entry function again on the converted operands
  where a later method must still see them.
- **Rename** (#1202). Name kernels `_<A>_<B>_<op>[_<result>][!]` after their operands and
  result, and state the scheme in a comment at the top of the file. Rename with a script
  that matches whole identifiers and asserts that no new name exists yet.
- **Make the dispatchers agree** (#1203). Every dispatcher of an operation normalizes its
  input the same way. Internal callers go to a private helper rather than a
  public-signature method.

Merge order matters: renames before the layout PR, and the layout PR is regenerated last.

## 3. Then the behaviour changes

Each goes in its own PR, stacked on the neutral ones:

- **Route the long tail through a copy** (#1205, #1210, #1212). Add the rare wrapper to the
  operand union so that it is copied once into a kernel's input. Measure the common forms:
  if one gets slower, give the size of the slowdown, or keep the special kernel. Record
  the candidates that you measured and rejected in the PR, so they are not re-measured.
- **Unify semantics** (#1199, #1209). When the routes of one operation disagree with each
  other or with dense (strong-zero `α`, stored zeros, index type, signed zeros), route them
  to one kernel and list each visible difference with a reproducer.
- **Fix the complexity** that the trace found (#1201, #1204), with a before/after table.

## Proving a step neutral

- **Bitwise digest.** Keep one script outside the repository that runs a fixed grid of
  products and `mul!` calls (eltypes, wrappers, NaN, Inf and signed zeros, `α` and `β`
  among `true`, `false`, `0.0` and others) and serializes the results. Every PR in the
  series compares its digest with `main` and reports the count, for example "5134 calls
  `isequal` to `main`".
- **Dispatch sweep.** For a change to dispatch, sweep every pair of operand forms and
  compare the method's module, the result type, value, stored structure and `@allocated`
  with `main`.
- **Method count.** `count(m -> m.module === SparseArrays, methods(f))` before and after
  is the Before/After block of a dispatch consolidation.
- Run the `ambiguous` and `aqua` suites: consolidating methods onto unions can create
  ambiguities with LinearAlgebra's methods, and they decide how far a step can go.

Commit none of this tooling. The PR states what was compared and how many cases differed,
under "Not checked by CI:".
