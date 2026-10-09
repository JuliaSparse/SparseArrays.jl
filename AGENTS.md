# AGENTS.md

Guidance for agents and other contributors to SparseArrays.jl, distilled from the
project's merged pull requests. Follow it unless a maintainer says otherwise.

## What this repo is

A Julia stdlib providing `SparseMatrixCSC`, `SparseVector`, sparse broadcast and
linear algebra, and the SuiteSparse solver wrappers (CHOLMOD, UMFPACK, SPQR) under
`src/solvers/`. Three directories carry their own `AGENTS.md`, which applies on top of
this one; read it before working there:

- `src/solvers/AGENTS.md`: rules for the SuiteSparse solver layer.
- `test/AGENTS.md`: suite layout, standard and comprehensive mode, where a new test
  goes, and how to measure test time.
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
- Base runs this package's tests in its own CI, serially, without ParallelTestRunner and
  with `--depwarn=error`. Keep `test/runtests.jl`'s serial fallback, and keep test time
  down. A test that passes only sometimes fails the bump: see the rule on random inputs
  under Tests.
- Base's `juliac --trim=safe` test loads this package, so the code must stay trimmable;
  the `trim` CI job builds the app in `test/trim` to check it.
- No new dependencies.
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
- Everything that depends on the GPL SuiteSparse libraries lives in `src/solvers/` and
  `test/solvers/`, and nowhere else. `Base.USE_GPL_LIBS` is checked once per tree, with
  `@static`: where `SparseArrays.jl` includes the solvers and where `test/runtests.jl`
  lists their suites. Code outside `src/solvers/` reaches a solver only through the
  generic LinearAlgebra functions (`lu`, `qr`, `cholesky`, `\`), never by naming a
  solver module. A test that factorizes or solves with a sparse matrix goes in
  `test/solvers/`, whatever feature it is about; the other suites must pass on a build
  without GPL libraries. `.ci/check-gpl-usage.jl`, run by the `code-checks` CI job, fails
  on solver names outside those directories.

## Running tests

```sh
julia +nightly --project -e 'using Pkg; Pkg.test(test_args=["fixed"])'   # one file; omit test_args for all
julia +nightly --project -e 'using Pkg; Pkg.test(test_args=["--comprehensive"])'   # comprehensive mode
julia +nightly --project -e 'using Test, LinearAlgebra, SparseArrays; include("test/fixed.jl")'
julia .ci/check-whitespace.jl
julia +nightly --project -e 'using Pkg; Pkg.test(test_args=["ambiguous", "aqua"])'   # ambiguity and Aqua checks
julia +nightly --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'  # doctests
julia .ci/check-gpl-usage.jl   # no solver names outside src/solvers/ and test/solvers/
```

The `ambiguous` and `aqua` suites run only when selected by name. The commands for the
trimmed app are in the `trim` job of `.github/workflows/ci.yml`.
The tests run in two modes, described in `test/AGENTS.md`. Standard mode is what
`Pkg.test`, Julia's own CI and every CI job but the coverage job run. Comprehensive mode
also runs the tests guarded with `@static if COMPREHENSIVE` and `test/issues.jl`; only the
coverage job runs it, so run it locally before a PR that touches a kernel.

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
- **Write through the storage accessors.** Use `getnzval`, `getrowval`, `getcolptr`,
  `nzrange` and `parent` on a matrix (`nonzeros` and `nonzeroinds` on a vector), never
  fields, and never indexed `setindex!` on a result whose
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
- **A result that would store every entry is dense.** That holds for sparse `+` and `-`
  with a dense array or a scalar, in operator and broadcast form, and for a broadcast
  whose function is not zero where its arguments are. Decide it from the function and
  the argument types so that the result type stays inferable, and look at the shape of
  the result before densifying an argument.
- **Elements are opaque.** The eltype need not be a machine number: `zero(Tv)` may be of
  another type than `Tv` (a JuMP variable) or not exist, so take the implicit zero from
  `_densezero` and do not convert it to `Tv`. Do not `copy` an element. Do not assume
  that a reducer commutes; walk the positions in order unless `op` is known to. A change
  to the result type of an arithmetic operation is checked against JuMP's and
  MutableArithmetics' test suites.
- **Keep kernels trimmable.** Julia does not specialize a method on a function or a
  splatted argument that it only passes on, which `--trim` rejects and which allocates.
  Write `f::F` and `Vararg{T,N}` there.
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

`test/AGENTS.md` has the detail: the two modes, the fixtures and helpers, and what test
time costs. In short:

- A new test is a comprehensive test: put it next to the feature it exercises, in an
  existing testset when one fits, inside `@static if COMPREHENSIVE ... end`. An
  unguarded test is only for new code, and one representative case of it.
- Test time is compilation, so it grows with the number of type combinations a test
  compiles. Cover real and complex eltypes, vector and matrix, with `Int` indices, and
  take any other type or helper from `test/SparseTestHelpers.jl`.
- Inputs are fixed and built inside the testset, from the fixtures or a literal that
  guarantees the property the test relies on. Random input is only for a test whose
  point is randomness, and then it is seeded.
- Compare a sparse result with `mismatch(S, D) === nothing`, which checks its structure
  and types as well as its values. After an expected throw, assert the destination is
  unchanged.
- Do not add an assertion that an existing loop already makes for the same kernel path.
- A method that exists only for speed needs a test proving it is dispatched to, not
  just a correctness check against dense, which passes on the fallback too.
- No wall-clock assertions. Allocation bounds prove constancy, not zero. Match
  `@test_throws` messages loosely, because Base rewords errors.
- Ambiguity failures on nightly are often Base's. Check before blaming a change.

## Pull requests

- Do each change in its own git worktree on a branch off `main`, never on `main`
  itself.
- One logical change per PR. Stack follow-ups and say so. A PR based on another PR's
  branch carries that diff when squash-merged.
- PRs are squash-merged with the title as the subject. Titles are imperative, name the
  API involved, and fit on one line of `git log --oneline`: one clause, not a list of
  everything the PR does.
- The body is the design record, kept short: the issue, a **Before** and an **After**
  block (a reproducer with its output on `main` and on the branch, or a measurement
  table), one paragraph on mechanism and fix, and anything deliberately left out. No
  section headers unless the change is large. A documentation-only PR needs just the
  source of the material.
- **Do not report what CI checks.** No sentence says that the tests, either test mode,
  the whitespace or GPL usage check, the ambiguity or Aqua check, the doctests or the
  docs build pass, were run, or are clean, and none gives the Julia build they ran on,
  the pass counts, or which suites were selected. The same goes for where the new tests
  were put and that they fail on `main`: the diff and the **Before** block show both.
  Mention a check only when CI does not run it: a downstream package's test suite, a
  bitwise or line-coverage comparison against `main`, a sweep against dense, a
  benchmark.
- Say when a PR touches dispatch, needs a docs update, or should be backported
  (label `backport 1.x`). Backports are bug and regression fixes only, never new
  methods or behaviour changes. Never bump compat on a release branch or push to one.
- Internal helpers are not API. Public surface changes are explicit, with docs.
- Agent-authored PRs say which tool wrote them, with a session link when there is one.
