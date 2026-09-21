# AGENTS.md

Guidance for agents and other contributors to SparseArrays.jl, distilled from the
project's merged pull requests. Follow it unless a maintainer says otherwise.

## What this repo is

A Julia stdlib providing `SparseMatrixCSC`, `SparseVector`, sparse broadcast and
linear algebra, and the SuiteSparse solver wrappers (CHOLMOD, UMFPACK, SPQR) under
`src/solvers/`. Being a stdlib shapes everything below:

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
- `src/solvers/wrappers.jl` is generated from the SuiteSparse headers by `gen/`. Never
  edit it by hand; add unwanted macros to the generator's ignore list instead. A
  SuiteSparse bump cannot be tested here until nightly bundles the new jll, so its CI
  is expected to be red; do not delete the merged bump PR's branch.
- Every source and test file starts with the Julia MIT license banner.

## Layout

| Path | Contents |
|---|---|
| `src/abstractsparse.jl` | Abstract types, `issparse`, the shared dispatch aliases. |
| `src/sparsematrix.jl`, `src/sparsevector.jl` | Storage, constructors, indexing, in-place ops. |
| `src/higherorderfns.jl` | Broadcast and `map!` kernels. |
| `src/matmul.jl` | `*`, `mul!`, `lmul!`, `rmul!` and their kernels, for matrices and vectors. |
| `src/linalg.jl`, `src/sparseconvert.jl` | `dot`, `kron`, solves, norms, LinearAlgebra wrappers, conversions. |
| `src/readonly.jl` | Read-only vector behind the experimental fixed-pattern arrays. |
| `src/solvers/` | Hand-written solver layers, library loading, generated bindings. |
| `gen/` | Wrapper generator; see its README. |
| `test/` | One file per area, listed explicitly in `runtests.jl`. |
| `docs/src/` | Documenter sources; public solver API is documented in `solvers.md`. |

## Running tests

```sh
julia +nightly --project -e 'using Pkg; Pkg.test()'                       # full suite
julia +nightly --project -e 'using Pkg; Pkg.test(test_args=["fixed"])'   # one file
julia +nightly --project -e 'using Test, LinearAlgebra, SparseArrays; include("test/fixed.jl")'
julia .ci/check-whitespace.jl
julia +nightly --project=docs -e 'using Pkg; Pkg.develop(path="."); include("docs/make.jl")'  # doctests
```

The doctest command edits `docs/Project.toml`; discard that change before committing.

The Aqua and ambiguity checks run as a separate CI job from `test/ambiguous.jl`.
Solver tests run only when `Base.USE_GPL_LIBS` is true; guard every reference to a
solver module, not just the test bodies. Only one CI job uses `--check-bounds=yes`,
because it invalidates precompiled code; it exists to catch bad `@inbounds`.

## Style

- Prefer writing less. Every line in a patch, script or recipe should earn its place.
- Keep comments minimal. Comment only genuinely tricky code (non-obvious workarounds,
  platform quirks, ABI hacks) and say why, not what.
- Comments describe the current state, never the change history. That belongs in the
  PR text.
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
  `abstractsparse.jl`; reuse LinearAlgebra's aliases and do not add one when an
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

## SuiteSparse solver layer

- Never throw a Julia exception from inside a C callback. Record the error, return, and
  check the status after the call.
- Own every C pointer exactly once. Anything that can throw between allocation and
  wrapping frees on the way out; multi-output calls initialize every output to null
  and free the siblings if wrapping one fails.
- `free!` nulls the wrapper's pointer so finalizers and repeated calls are no-ops.
  Keep wrappers rooted while reading through raw pointers.
- Guard the invariants the C side assumes (contiguous outputs, matching types, sorted
  and packed flags on CHOLMOD sparse structs). Drop stale state on failure and free
  eagerly on refactorization.
- The solvers work in double precision. Convert inputs explicitly; `float` is not a
  Float64 cast for generic eltypes. Convert results back and keep `\` type-stable.
- Accept strided right-hand sides and their adjoint and transpose wrappers.
- Initialization is lazy and process-once, done before the first C call rather than in
  `__init__`. Library handles belong to `LibSuiteSparse`; every submodule imports each
  symbol it uses explicitly, and never references a library by bare symbol.
- Workspaces live inside the factorization behind an internal lock. Factorizations
  expose no public lock interface; use separate workspaces for parallel solves and
  document threading in the docs.
- Do not change ordering or tolerance defaults without an opt-in keyword.

## Tests

- Regression tests go next to the feature they exercise, in an existing testset when
  one fits. After an expected throw, assert the destination is unchanged.
- Cover real and complex eltypes, vector and matrix. Test both index types only where
  the code dispatches to SuiteSparse, which selects the C entry point by index type;
  pure-Julia kernels are generic over `Ti` and one index type is enough.
- A method that exists only for speed needs a test proving it is dispatched to, not
  just a correctness check against dense, which passes on the fallback too.
- No wall-clock assertions. Allocation bounds prove constancy, not zero. Match
  `@test_throws` messages loosely, because Base rewords errors.
- Keep eltype grids representative rather than exhaustive. Explicit `GC.gc()` calls
  belong only in the isolated solver lifetime tests, where they exercise finalization
  and rooting. Check a re-enabled test on nightly before removing its skip.
- Seed randomized tests of heuristics. Single-precision solver tests need
  well-conditioned inputs.
- Ambiguity failures on nightly are often Base's, and Windows hangs have historically
  been CHOLMOD threading under GitHub Actions. Check before blaming a change.

## Pull requests

- Do each change in its own git worktree on a branch off `main`, never on `main`
  itself.
- One logical change per PR. Stack follow-ups and say so. A PR based on another PR's
  branch carries that diff when squash-merged.
- Squash-merged with the PR title as the subject. Titles are imperative and name the
  API involved. Only backport PRs use merge commits, and GitHub remembers the last
  choice.
- The body is the design record: the issue, a minimal reproducer, the mechanism, the
  fix, what was tested on which build, and anything deliberately left out. Keep it
  short: a reproducer in a code block, one paragraph on mechanism and fix, one line on
  tests. No section headers unless the change is large. A documentation-only PR needs
  just the source of the material and how it was checked.
- Say when a PR touches dispatch, needs a docs update, or should be backported.
- Backports: label `backport 1.x`; a maintainer batches cherry-picks onto a backport
  branch. Bug and regression fixes only, never new methods or behaviour changes. Never
  bump compat on a release branch or push to one directly.
- Internal helpers are not API. Public surface changes are explicit, with docs.
- Agent-authored PRs say which tool wrote them, with a session link when there is one.
- Do not commit anything under `.claude/`; it is untracked local scratch.
