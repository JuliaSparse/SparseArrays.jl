# AGENTS.md

Guidance for agents and other contributors to SparseArrays.jl, distilled from the
project's merged pull requests. Follow it unless a maintainer says otherwise.

## What this repo is

A Julia stdlib providing `SparseMatrixCSC`, `SparseVector`, sparse broadcast and
linear algebra, and the SuiteSparse solver wrappers (CHOLMOD, UMFPACK, SPQR) under
`src/solvers/`. Three directories carry their own `AGENTS.md`, which applies on top of
this one; read it before working there:

- `src/solvers/AGENTS.md`: rules for the SuiteSparse solver layer.
- `test/AGENTS.md`: suite layout, the coverage each reduced test grid must keep, and
  how to measure test time.
- `gen/AGENTS.md`: regenerating `src/solvers/wrappers.jl`, and upgrading SuiteSparse
  and Clang.jl.

Being a stdlib shapes everything below:

- The `julia` compat in `Project.toml` tracks the next release, so develop and test on
  `julia +nightly`. Each Julia release ships a fixed copy of this package, so there are
  no `@static if VERSION` branches; to reach an older Julia, backport instead.
- Code reaches users only when JuliaLang/julia bumps its pinned SparseArrays version.
  Say in the PR when a fix unblocks Base CI so the bump gets triggered.
- LinearAlgebra is pinned in lockstep. Never write version-sniffing shims for its
  internals. Changes to the hooks shared with it need a paired PR there, and CI stays
  red until both sides merge.
- Base runs this package's tests in its own CI, serially and without ParallelTestRunner.
  Keep `test/runtests.jl`'s serial fallback, and keep test time down.
- No new dependencies. Every test dependency needs a `[compat]` entry.
- Loading the package is measured in sysimage invalidations. Do not define methods
  Base or LinearAlgebra already provide generically.
- Every source and test file starts with the Julia MIT license banner.

## Layout

Files in `src/` and `test/` are named by area. What the names do not tell you:

- `sparsevector.jl` holds everything for vectors except products (`matmul.jl`) and
  concatenation (`concatenation.jl`), which cover matrices and vectors together.
- `linalg.jl` holds `dot`, `kron`, solves, norms and the LinearAlgebra wrappers.
- The shared dispatch aliases live in `SparseArrays.jl`.
- Test files are listed explicitly in `test/runtests.jl`.

## Running tests

```sh
julia +nightly --project -e 'using Pkg; Pkg.test(test_args=["fixed"])'   # one file; omit test_args for all
julia +nightly --project -e 'using Test, LinearAlgebra, SparseArrays; include("test/fixed.jl")'
julia .ci/check-whitespace.jl
julia +nightly --project -e 'using Pkg; Pkg.test(test_args=["ambiguous"])'   # Aqua and ambiguity checks
julia +nightly --project=docs -e 'using Pkg; Pkg.develop(path="."); include("docs/make.jl")'  # doctests
```

The doctest command edits `docs/Project.toml`; discard that change before committing.

The Aqua and ambiguity checks in `test/ambiguous.jl` run only when selected by name, and
as a separate CI job.
Everything that needs SuiteSparse is tested under `test/solvers/`, which runs only when
`Base.USE_GPL_LIBS` is true; `test/runtests.jl` holds the only check. One CI job runs
`--check-bounds=yes` to catch bad `@inbounds`.

## Style

- Comment only genuinely tricky code (non-obvious workarounds, platform quirks, ABI
  hacks) and say why, not what. Comments describe the current state; change history
  belongs in the PR text.
- Style-only changes go in their own PR. Never let an editor reformat whitespace in a
  functional PR.

## Coding conventions

- **Sparse kernels are O(nnz).** Any elementwise `AbstractArray` fallback reached by a
  sparse type is a bug. When you specialize a function, its relatives (equality,
  hashing, adjoint and transpose wrappers) need the same treatment.
- **Validate before mutating.** In-place kernels check shapes, aliasing and pattern
  compatibility up front and throw with the destination untouched. Cheap direct check
  first, expensive dry run only if that fails.
- **Unalias.** Use `Base.mightalias` / `Base.unalias` when a destination may share
  storage with an input. Structurally empty arrays report as aliased on recent Julia,
  so compare only the buffers you write.
- **Never infer structure from `nnz`.** Stored zeros are the recurring correctness
  trap; walk the column.
- **Write through the storage accessors.** Use `nonzeros`, `rowvals`, `getcolptr`,
  `nzrange` and `parent`, never fields, and never indexed `setindex!` on a result whose
  pattern you already know. Do not materialize with `findnz`.
- **Follow dense semantics** when sparse behaviour is in doubt: shape rules,
  promotion, unaliasing. Sweep sparse against dense locally; commit only the
  reproducer.
- **Extend, don't shadow.** An unqualified definition that shares a Base or
  LinearAlgebra name silently creates a dead local function. Qualify it.
- **Dispatch narrowly.** Cross-file `Union` aliases live in one documented section of
  `SparseArrays.jl`; reuse LinearAlgebra's aliases and do not add one when an
  existing one fits. Define against `Matrix`/`Vector` rather than `AbstractMatrix`
  where a wider signature would capture structured LinearAlgebra types. Views of
  sparse arrays are first-class: define methods on the view aliases too.
- **Products and solves follow the dense factor.** Sparse times dense returns dense,
  sparse times a banded structured type stays sparse, solves return dense.
- **Prefer explicit helpers over `invoke`** for fallbacks; tooling cannot model
  `invoke` chains.
- **Errors name the type and the reason**, and the working alternative when there is
  one. `ArgumentError` for bad values, `DimensionMismatch` for shapes.
- **Fixed-pattern arrays stay experimental and unexported.** Their pattern is
  read-only, `copy` shares it, and only shape-taking `similar` returns a writable
  array.
- **`@inbounds` only after the bounds are provably guarded.** Performance PRs include a
  before/after table of time and allocations at realistic sizes.

## Tests

- Regression tests go next to the feature they exercise, in an existing testset when
  one fits. After an expected throw, assert the destination is unchanged.
- Cover real and complex eltypes, vector and matrix, with representative rather than
  exhaustive grids. Pure-Julia kernels are generic over `Ti`; one index type is enough.
- A method that exists only for speed needs a test proving it is dispatched to, not
  just a correctness check against dense, which passes on the fallback too.
- No wall-clock assertions. Allocation bounds prove constancy, not zero. Match
  `@test_throws` messages loosely, because Base rewords errors.
- Seed randomized tests of heuristics. Check a re-enabled test on nightly before
  removing its skip.
- Ambiguity failures on nightly are often Base's. Check before blaming a change.

## Pull requests

- Do each change in its own git worktree on a branch off `main`, never on `main`
  itself. Do not commit anything under `.claude/`.
- One logical change per PR. Stack follow-ups and say so. A PR based on another PR's
  branch carries that diff when squash-merged.
- PRs are squash-merged with the title as the subject. Titles are imperative and name
  the API involved.
- The body is the design record, kept short: the issue, a reproducer in a code block,
  one paragraph on mechanism and fix, one line on what was tested on which build, and
  anything deliberately left out. No section headers unless the change is large. A
  documentation-only PR needs just the source of the material and how it was checked.
- Say when a PR touches dispatch, needs a docs update, or should be backported
  (label `backport 1.x`). Backports are bug and regression fixes only, never new
  methods or behaviour changes. Never bump compat on a release branch or push to one.
- Internal helpers are not API. Public surface changes are explicit, with docs.
- Agent-authored PRs say which tool wrote them, with a session link when there is one.
