# AGENTS.md

Guidance for AI agents and other contributors working in SparseArrays.jl. Follow the
conventions below unless a maintainer says otherwise.

## What this repo is

SparseArrays.jl is a Julia stdlib. It ships `SparseMatrixCSC`, `SparseVector`,
their fixed-pattern variants (`FixedSparseCSC`, `FixedSparseVector`, unexported and
experimental), sparse broadcast/`map!`, sparse linear algebra, and the SuiteSparse
solver wrappers (CHOLMOD, UMFPACK, SPQR) under `src/solvers/`.

- `src/solvers/wrappers.jl` is **generated**; never hand-edit it (see "SuiteSparse
  wrappers" below).
- Every source and test file starts with the Julia license banner:
  `# This file is a part of Julia. License is MIT: https://julialang.org/license`.

## Layout

| Path | Contents |
|---|---|
| `src/abstractsparse.jl` | Abstract types, `issparse`, and the shared dispatch `Union` aliases (one section, documented naming scheme). |
| `src/sparsematrix.jl`, `src/sparsevector.jl` | CSC/vector storage, constructors, indexing, `setindex!`, `copyto!`, `sort!`, transpose. |
| `src/higherorderfns.jl` | Sparse broadcast and `map!` kernels, storage growth helpers. |
| `src/linalg.jl`, `src/sparseconvert.jl` | Products, solves, wrappers for LinearAlgebra types, conversions. |
| `src/readonly.jl` | `ReadOnly` vector used by the fixed-pattern arrays. |
| `src/solvers/{cholmod,umfpack,spqr}.jl` | Hand-written solver layers over the C API. |
| `src/solvers/LibSuiteSparse.jl` | `LazyLibrary` handles and the `set_libdir!` / `JULIA_SUITESPARSE_LIBDIR` override. |
| `src/solvers/wrappers.jl` | Clang.jl-generated `@ccall` bindings. Do not edit. |
| `gen/` | Generator for `wrappers.jl` (`Makefile`, `generator.toml`, `prologue.jl`). |
| `test/` | One file per area, listed explicitly in `test/runtests.jl`. |
| `docs/src/{index,solvers}.md` | Documenter sources; add new public API docstrings to `solvers.md` when relevant. |

## Running tests

Tests run under ParallelTestRunner, one subprocess per file, `--jobs=$(Sys.CPU_THREADS)`.

```sh
# full suite
julia +nightly --project -e 'using Pkg; Pkg.test()'

# one or more files (names without .jl, matched by ParallelTestRunner)
julia +nightly --project -e 'using Pkg; Pkg.test(test_args=["fixed", "higherorderfns"])'

# fastest inner loop: include a single file directly
julia +nightly --project -e 'using Test, LinearAlgebra, SparseArrays; include("test/fixed.jl")'

# Aqua + method-ambiguity check (separate CI job)
julia +nightly --project -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path=pwd()); Pkg.add(name="Aqua", version="0.8"); include("test/ambiguous.jl")'

# whitespace check (CI fails on trailing whitespace / tabs)
julia .ci/check-whitespace.jl
```

Notes:

- The SuiteSparse-dependent files (`cholmod*`, `umfpack`, `spqr`, `linalg*`,
  `threads_suite`) only run when `Base.USE_GPL_LIBS` is true; guard new solver tests
  the same way the existing files do.
- `test/ambiguous.jl` runs `Aqua.test_all` with `unbound_args` and `piracies` marked
  broken, plus `detect_ambiguities`. New methods must not introduce ambiguities. When a
  PR touches dispatch, state in the PR that the unbound-parameter and piracy counts are
  unchanged.
- Add a new test file to the `testfiles` list in `test/runtests.jl`; the runner does
  not glob.
- Do not add version gates (`@static if VERSION ≥ ...`) for Julia versions below the
  `julia` compat; #801 removed the stale ones.
- CI runs on nightly on Linux x64/x86, Windows x64/x86 and macOS aarch64. Only one job
  uses `--check-bounds=yes` and coverage because both invalidate precompiled code.

## Style

- Prefer writing less. Every line in a patch, script or recipe should earn its place.
- Keep comments minimal. Comment only code that is genuinely tricky (non-obvious
  workarounds, platform quirks, ABI hacks) and say why, not what.
- Do not comment obvious steps or restate what the code already says.
- Comments describe the current state, never the change history. "Previously this
  did X" or "changed in #NNN" belongs in the PR text, not the source.

## Coding conventions seen in recent PRs

**Validate before mutating.** Several bugs (#786, #819, #820, #821, #822) were
paths that partially wrote a destination and then threw, leaving it corrupted. Any
in-place kernel must check shapes, aliasing and pattern compatibility up front and throw
with the destination untouched. Prefer a cheap direct check first and a dry run of the
expensive merge only when that fails (#820).

**Aliasing.** Use `Base.mightalias` / `Base.unalias` when a destination may share
storage with an input (#813, #821). Be aware that on Julia 1.11+ all empty `Vector{T}`
share one `Memory`, so structurally empty arrays report as aliased; compare only the
buffers you actually write in that case.

**Error messages.** Replace internal errors such as
`can't resize SparseArrays.ReadOnly{...}` with an `ArgumentError` that names the type
and the reason, e.g. "cannot store a nonzero f(0) into a FixedSparseCSC, its sparsity
pattern is read-only". Use `DimensionMismatch` for shape errors.

**Fixed-pattern arrays.** `FixedSparseCSC` / `FixedSparseVector` keep their pattern
read-only. Structure-preserving `similar(F)` returns a fixed array; shape-taking
`similar(F, dims)` returns a plain `SparseMatrixCSC` / `SparseVector` (#822). `copy` of
a fixed array shares its read-only structure, so operations that need a writable copy
must build one explicitly (#786).

**Follow dense semantics.** When sparse behaviour is in doubt, match what `Array`
does: `setindex!` shape rules via `Base.setindex_shape_check` (#809), `promote_rule`
via `Base.el_same` (#807), `map!` unaliasing like dense `map!` (#813). Sweep a grid of
sparse-vs-dense cases locally, but commit only the issue's reproducer as a test.

**Dispatch aliases.** Cross-file `Union` aliases live in one section of
`src/abstractsparse.jl` with a documented naming scheme (`Sparse<Family>OrView`,
`<X>MaybeAdjOrTrans`, ...). Reuse LinearAlgebra's aliases (`AdjOrTrans`, etc.) rather
than redefining them. Solver scalar-type aliases stay with their solver (#816). Do not
add a new alias when an existing one fits.

**Restrict new methods narrowly.** Define rules against `Matrix` / `Vector` rather
than `AbstractMatrix` when a broader signature could interfere with structured
LinearAlgebra types (#807). Use accessors (`parent`, `getcolptr`, `rowvals`,
`nonzeros`) rather than fields (#785).

**Broadcast.** `is_supported_sparse_broadcast` decides whether an argument list can be
sparsified; unsupported arguments must fall back to the generic path via `copy(bc)`
rather than recursing (#808). Result storage grows on demand through `_growstorage!`,
never preallocated at the nnz upper bound (#803). Include a before/after table of time
and allocations in a performance PR.

## SuiteSparse solver layer (`src/solvers/`)

The September audit (#790–#796, #800, #802, #806) established these rules:

- **Never throw a Julia exception from inside a C callback.** The CHOLMOD
  `error_handler` records the first message in task-local storage and returns;
  `check_status(common)` / the `@checked` macro inspect `Common.status` after the call
  and throw `CHOLMODException` from Julia (#796).
- **Own every C pointer exactly once.** Wrapper constructors free their pointer if they
  throw. Outer helpers that inspect `itype`/`xtype` before reaching the inner
  constructor must free on throw too (#795). When a C routine returns several outputs
  through `Ref{Ptr}`s, initialize them all to `C_NULL`, hand each to its wrapper inside
  `try`/`catch`, and free any siblings still held on failure (#790).
- **`free!` must null the wrapper's pointer** so that a finalizer running afterwards is
  a no-op, and calling `free!` twice is safe. Read the pointer with
  `getfield(x, :ptr)` so `free!` never throws on a null pointer (#791, #792, #793).
- **Keep wrappers rooted** with `GC.@preserve` while reading through raw pointers
  obtained from them (#792).
- **Guard invariants the C side assumes.** CHOLMOD `solve2` overwrites the output
  descriptor's leading dimension, so `ldiv!` requires a contiguous output (#794).
  UMFPACK numeric factorizations are dropped on failure and freed eagerly on
  refactorization (#793).
- **Accept strided inputs.** Right-hand sides should accept `StridedVecOrMat` and their
  adjoint/transpose wrappers, converted to the factor's eltype, not just `Vector` /
  `Matrix` (#800, #802).
- **Library handles** are `LazyLibrary`s owned by `LibSuiteSparse`, not imported from
  `SuiteSparse_jll`, so `set_libdir!` / `JULIA_SUITESPARSE_LIBDIR` can redirect the
  whole set before first use (#806). `@ccall libcholmod.…` sites in `wrappers.jl` bind
  to those constants.

### Regenerating the wrappers

To move to a new SuiteSparse release (#824):

1. Bump `SuiteSparse_jll` compat in `Project.toml`.
2. In `gen/Makefile` set `VER` and `JLL_BUILD` (the jll build number, e.g. `+1`).
3. `cd gen && make`. This downloads the x86_64-linux-gnu jll tarball for its headers
   and rewrites `src/solvers/wrappers.jl`.
4. Add newly emitted junk macros to the ignore list in `gen/generator.toml` rather than
   editing the output.
5. Diff `wrappers.jl` and say in the PR whether any signatures, structs or enums
   changed, or only version constants.
6. Run the solver test files against a nightly that bundles the new jll.

## Tests

- Put regression tests next to the feature they exercise, in an existing `@testset`
  when one fits. Name issue-specific sets `"Issue #NNN"` (see `test/issues.jl`).
- After an expected throw, assert the destination is unchanged (`nnz`, `getcolptr`,
  `rowvals`, `nonzeros`) — this is what caught the corruption bugs.
- Cover both `Int32` and `Int64` index types and both real and complex eltypes for
  solver changes, and both `SparseVector` and `SparseMatrixCSC` for kernel changes.
- Trim redundant loops rather than adding more (#787), and check every re-enabled test
  on nightly before removing a skip (#801).
- Threads coverage: `test/threads_suite.jl` spawns a second process with a different
  `JULIA_NUM_THREADS`; it runs on Windows too (#797). Do not reintroduce the GHA skip.

## Pull requests

- One logical change per PR. Stack follow-ups on the previous PR and say so
  ("Stacked on #820"). Base branches carefully: #810 accidentally carried #804's diff
  and needed a revert (#817).
- Squash-merged with the PR title as the commit subject. Titles are imperative,
  specific, and mention the API involved, e.g. "Throw when the destination of sparse
  `transpose!`/`adjoint!` aliases the source".
- The PR body is the design record. Include: `Fixes #NNN` / `Ref #NNN`; a minimal REPL
  reproducer of the bug; the mechanism (which function did what); the fix; what was
  tested and on which Julia build; anything deliberately left out ("`map!` on a fixed
  destination still errors, unrelated, left alone"). Tables for counts or benchmarks.
- Mention `test/ambiguous.jl` results when touching dispatch, and doc updates in
  `docs/src/solvers.md` when adding public solver API.
- A documentation-only PR needs just the source of the material and how it was
  checked; do not restate what the diff shows.
- Agent-authored PRs end with the tool attribution and session link (Claude Code) or a
  "PR made by Codex" line.

## Things to avoid

- Editing `src/solvers/wrappers.jl` by hand.
- Widening a `Union` alias or a method signature to `AbstractMatrix` without checking
  LinearAlgebra's structured types and `detect_ambiguities`.
- Adding `@static if VERSION` gates for versions below the compat bound.
- Skipping platforms in CI instead of fixing or investigating the failure.
- Exporting `FixedSparseCSC` / `FixedSparseVector`; they remain experimental.
- Committing anything under `.claude/`. It holds local agent worktrees and is not
  in `.gitignore`, so check `git status` before staging.
