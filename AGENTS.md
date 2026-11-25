# AGENTS.md

Guidance for agents and other contributors to SparseArrays.jl, distilled from the
project's merged pull requests. Follow it unless a maintainer says otherwise.

SparseArrays is a Julia stdlib: `SparseMatrixCSC`, `SparseVector`, sparse broadcast and
linear algebra, and the SuiteSparse solver wrappers (CHOLMOD, UMFPACK, SPQR) in
`src/solvers/`. `src/solvers/AGENTS.md` and `test/AGENTS.md` add rules for their
directories; read them before working there.

## Skills

Procedures for particular tasks live in `.agents/skills/<name>/SKILL.md`. Read the one
that fits before you start:

- `pull-request`: branch, title and body of a pull request, and the measurements it shows.
- `restructure-code`: reorganizing a source file or a family of methods.
- `manual-page`: `docs/src/index.md` and the docstrings it lists.
- `fixed-pattern`: `FixedSparseCSC` and `FixedSparseVector`.
- `test-time`: measuring test time and cutting a test grid.
- `suitesparse-wrappers`: regenerating `src/solvers/wrappers.jl`, and upgrading
  SuiteSparse or Clang.jl.

## Being a stdlib

- Develop and test on `julia +nightly`; the `julia` compat tracks the next release. No
  `@static if VERSION` branches: an older Julia gets a backport.
- LinearAlgebra is pinned in lockstep. No version-sniffing shims for its internals; a
  change to a hook shared with it needs a paired PR there.
- Base runs these tests serially, without ParallelTestRunner, with `--depwarn=error`, and
  a flaky test fails the version bump. Keep `test/runtests.jl`'s serial fallback, and
  keep test time down.
- The code must stay trimmable (`juliac --trim=safe`, checked by the `trim` CI job).
- No new dependencies. Package load is measured in sysimage invalidations: do not define
  methods that Base or LinearAlgebra already provide generically.
- Every source and test file starts with the Julia MIT license banner.

## The GPL boundary

Everything that depends on the GPL SuiteSparse libraries lives in `src/solvers/` and
`test/solvers/`, and nowhere else. `Base.USE_GPL_LIBS` is checked once per tree, with
`@static`: where `SparseArrays.jl` includes the solvers and where `test/runtests.jl` lists
their suites. Other code reaches a solver only through the generic LinearAlgebra
functions (`lu`, `qr`, `cholesky`, `\`), never by naming a solver module, and a test that
factorizes or solves goes in `test/solvers/`. `.ci/check-gpl-usage.jl` enforces this.

## Layout

Files in `src/` and `test/` are named by area. `sparsevector.jl` holds everything for
vectors except products (`matmul.jl`) and concatenation (`concatenation.jl`); `linalg.jl`
holds `dot`, `kron`, solves, norms and the LinearAlgebra wrappers; the shared dispatch
aliases live in `SparseArrays.jl`; test files are listed explicitly in
`test/runtests.jl`.

## Commands

```sh
julia +nightly --project -e 'using Pkg; Pkg.test(test_args=["fixed"])'   # one suite; omit test_args for all
julia +nightly --project -e 'using Pkg; Pkg.test(test_args=["--comprehensive"])'
julia +nightly --project -e 'using Test, LinearAlgebra, SparseArrays; include("test/fixed.jl")'
julia +nightly --project -e 'using Pkg; Pkg.test(test_args=["ambiguous", "aqua"])'   # run only when named
julia +nightly --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
julia .ci/check-whitespace.jl
julia .ci/check-gpl-usage.jl
```

The trimmed app is built by the `trim` job in `.github/workflows/ci.yml`. Comprehensive
mode (see `test/AGENTS.md`) runs only in the coverage CI job; run it locally before a PR
that touches a kernel or a constructor.

## Architecture

Every operation is layered the same way.

- **Entry methods are few.** They dispatch on unions of operand kinds, normalize each
  operand in one place, and call a private function that picks the kernel. A method
  exists to dispatch or to resolve an ambiguity, not once per pair of wrappers. Internal
  calls go to private helpers, never to a public-signature method.
- **Kernels are named after their operands and result**, and a file states its naming
  scheme and its sections, in dispatch order, at the top.
- **The long tail goes through a copy.** A rare wrapper (a non-range view, an adjoint, a
  triangle of a view) is copied once, in O(nnz), into the form an existing kernel takes.
  A dedicated kernel stays only where a measurement shows it beats the copy on the
  common cases.
- **One operation, one set of semantics.** Every route of an operation gives the same
  stored zeros, index type, strong-zero `α` and errors. When two routes disagree, send
  both to one kernel rather than patching each.
- **Variants are parameters, not copies.** Pass the side as a `Val`, the wrapper as a
  function (`identity`, `transpose`, `adjoint`), or a trivial coefficient as `nothing`, so
  one loop body compiles to each specialized loop. Shared preambles (size checks, scaling
  the destination) are one `@inline` helper.
- **Dispatchers normalize their input once**, and every dispatcher of an operation reads
  it the same way.

## Coding conventions

- **Sparse kernels are O(nnz).** An elementwise `AbstractArray` fallback reached by a
  sparse type is a bug. Specializing a function means specializing its relatives
  (equality, hashing, adjoint and transpose wrappers, views) too.
- **Validate before mutating.** In-place kernels check shapes, aliasing and pattern
  compatibility up front and throw with the destination untouched: a cheap direct check
  first, an expensive dry run only if that fails.
- **Unalias** with `Base.mightalias` / `Base.unalias`, comparing only the buffers you write:
  structurally empty arrays report as aliased.
- **Never infer structure from `nnz`.** Stored zeros are the recurring trap; walk the column.
- **Use the storage accessors** (`getnzval`, `getrowval`, `getcolptr`, `nzrange`, `parent`;
  `nonzeros`, `nonzeroinds`), never fields, never indexed `setindex!` into a pattern you
  already know, never `findnz` to materialize.
- **Follow dense semantics** where sparse behaviour is in doubt: shapes, promotion,
  aliasing, NaN and signed zeros. Products and solves follow the dense factor: sparse
  times dense is dense, sparse times a banded structured type stays sparse, solves return
  dense. A result that would store every entry is dense (sparse `+` or `-` with a dense
  array or a scalar, a broadcast of a function that is not zero at zero). Decide it from
  the function and argument types so it stays inferable, and look at the shape of the
  result before densifying an argument.
- **Extend, don't shadow.** Qualify any definition that shares a Base or LinearAlgebra name.
- **Dispatch narrowly.** Cross-file `Union` aliases live in one section of
  `SparseArrays.jl`; reuse LinearAlgebra's. Prefer `Matrix`/`Vector` to `AbstractMatrix`
  where a wider signature would capture structured types. Views are first-class.
- **Elements are opaque.** `zero(Tv)` may be another type or not exist: take the implicit
  zero from `_densezero` and do not convert it. Do not `copy` an element, and do not
  assume a reducer commutes. Check a change to an arithmetic result type against JuMP's
  and MutableArithmetics' test suites.
- **Keep kernels trimmable**: `f::F` and `Vararg{T,N}` for arguments that are only passed on.
- **No `invoke`** for fallbacks; write an explicit helper.
- **Errors name the type, the reason and the alternative.** `ArgumentError` for values,
  `DimensionMismatch` for shapes.
- **`@inbounds` only where the bounds are provably checked.**
- **Comments say why**, for genuinely tricky code, and describe the current state.
- **Internal helpers are not API.** Public surface changes are explicit and documented.

## Tests

`test/AGENTS.md` has the rules. In short: a new test is comprehensive (behind
`@static if COMPREHENSIVE`, or in `test/issues.jl`) unless it is the one representative
case of new code; inputs are fixed fixtures or literals; and a sparse result is compared
with `mismatch(S, D) === nothing`.

## Changes

- Each change in its own git worktree and branch off `main`, one logical change per PR.
- Behaviour-neutral restructuring, behaviour changes and style-only changes go in
  separate PRs. Never let an editor reformat a functional PR.
- Sweep sparse against dense locally; commit only the reproducer.
