# AGENTS.md for `test/`

How the test suite is organized, what its reduced grids must keep covering, and how to
measure test time, in addition to the top-level `AGENTS.md`.

## Layout

- `runtests.jl` owns the suite inventory, used by both ParallelTestRunner and the serial
  fallback for Julia Base CI. On both paths selectors match suite names by prefix and a
  `!prefix` selector excludes; the serial path takes selectors only, and errors when
  nothing matches. `--jobs=N` and `PTR_NUM_JOBS` are honoured; without either, the
  runner uses `Sys.CPU_THREADS` workers. Adding a feature file does not require adding a
  worker task.
- `testhelpers.jl` holds the helpers every suite shares: the `getproperty` guard that
  makes field access on the sparse types an error, `same_pattern` and `exact_equal`,
  the `OpCount` eltype with `mulcount`/`eqcount`/`opcount_sparse`, `CountedReads`,
  `hasunionlocal`, `quaternion_type` and `SimpleSMatrix`. It is a plain definition file:
  every suite is a module that `include`s it first (`../testhelpers.jl` from `solvers/`
  and `torture/`),
  so the guard applies whatever the worker order. `ambiguous.jl` alone does not include
  it. Put a helper that a second suite needs there rather than duplicating it.
- Test files are named after the source area they cover. `solvers/` mirrors
  `src/solvers/`: its suites are `solvers/cholmod`, `solvers/umfpack`, `solvers/spqr`,
  `solvers/solvers` and `solvers/threads`, so the selector `solvers` runs them all.
- The files in `solvers/` carry no `Base.USE_GPL_LIBS` guards of their own; the
  top-level `AGENTS.md` has the rule for what belongs there.
- `triangular.jl` holds the triangular product and solve tests as one scheduling unit,
  and the two grids share their fixtures. `concatenation.jl` likewise holds all
  concatenation tests, and `matmul.jl` every other product test, for vectors as well as
  matrices: scaling, the BLAS-2 grid and products with LinearAlgebra's Q types, while
  `sparsevector.jl` keeps the vector `axpy!` and `dot` tests. The `transpose`, `adjoint`
  and `permute` tests, including the in-place forms, live in `sparsematrix.jl`.
- Preserve issue references on regression tests.
- `torture/` is the second tier of the suite, selected by the `torture` prefix and
  otherwise skipped on both paths; the top-level `AGENTS.md` has the policy. Its files
  mirror the core files by area (`torture/indexing.jl` for `indexing.jl`, and so on;
  `torture/solvers/` for `solvers/`, listed under the same GPL guard), and each is a
  self-contained module that includes only `../testhelpers.jl`: it never includes a
  core suite file, and duplicates the fixtures it needs. A testset moved there keeps
  its issue number. The core suite must stay complete without it, so a torture test is
  a regression guard or an exhaustive grid, never the only test of a feature. When a
  core testset keeps one representative of a loop (an eltype, a size, a random draw),
  the removed cases go to the torture file with a comment on both sides saying so.
- `ambiguous.jl` is in the inventory but skipped unless a selector names it; CI gives it
  its own job. It restores the depot, load path, environment and active project in a
  `finally`, so an Aqua failure on Base CI leaves the worker usable.
- `solvers/threads.jl` owns tests requiring fresh process state. Its `testprocess.jl`
  helper preserves the active project and resolved load path, verifies the checkout
  loaded by the child, and explicitly selects default-pool thread counts. The file has
  two roles: the children it starts with one and four threads include it again with
  `SPARSEARRAYS_TEST_THREADS_CHILD` set, which selects the solver concurrency checks
  themselves. It also runs `cholmod_lifetime.jl`, which is not a suite of its own, and
  covers library-directory selection. Keep the rooting stress
  workload in the lifetime suite until a demonstrated reproducer supports a smaller
  replacement.

## Coverage dimensions

Factored grids separate independent dimensions instead of compiling their full
Cartesian product. Each reduction must identify the replacement for every removed
dimension; equal assertion counts or line coverage alone are insufficient.

| Family | Retained coverage |
| --- | --- |
| Sparse-vector triangular solves | Four triangular wrappers × identity/transpose/adjoint × dense/sparse backing with Float64 and ComplexF64. RHS patterns are empty, first-only, last-only, interior gaps, and stored-zero endpoints. All 19 existing promotion pairs remain: the Int64/Float64/ComplexF64 cross product, plus Int32/BigInt/Float32/BigFloat/ComplexF32 paired with Float64 in both directions. Promotion uses lower/unit-lower representatives for each backing and checks valid in-place cases. Dense-backed speed specializations have dispatch assertions. |
| Scalar/sparse broadcast | All four array forms, including the transposed column matrix, and one/two/more-than-two-array kernels. Scalar placement uses distinct values and an order-sensitive function; arity cases cover zero-preserving and non-zero-preserving functions. All seven eight-argument inference cases and their allocation bound remain. In-place references use dense destinations. |
| Sparse/triangular products | All four wrappers and both operands' three transforms, in both multiplication orders, with Float64 and ComplexF32. All nine Int/Float64/ComplexF32 promotion pairs use upper/lower representatives. Dense result types are checked. Triangular sparse times dense keeps Int, Float64 and ComplexF32; triangular times sparse vector keeps the matching Int/Float64/ComplexF64 pairs and one promoting pair. The other eltypes and pairs are in `torture/triangular.jl`. |
| Dense/sparse `mul!` | Both operand orders and every transform pair for Int, Float64, ComplexF64, and BigFloat. Boolean and numeric zero/one/general coefficient pairs cover sparse identity/transpose/adjoint kernels and plain/wrapped dense-left kernels separately. Existing noncommutative regressions remain. The matrix-vector (BLAS-2) products zip the three alpha/beta pairs and keep one wrapper per kind; the in-place sparse-sparse product keeps one size with zipped coefficients. The full grids are in `torture/sparsevector.jl` and `torture/matmul.jl`. |
| CHOLMOD operations | Single-input tests cover both precisions, real/complex values, and both supported C index types. Mixed-input operations retain all precision/type pairs; the real/complex mix of matrix and right-hand side in `\` (#27860) is in `torture/solvers/cholmod.jl`. Ownership, invalid-wrapper cleanup, repeated `free!`, buffer isolation, Common accounting, and rooting tests remain in the lifetime process. |
| UMFPACK and SPQR grids | The eltype and index-type grids are zipped to (Float64, Int) and (ComplexF64, Int32); the symbolic-reuse test keeps Float64, ComplexF64 and Float32. The full grids, the half-precision eltypes and the shared-factor thread grid are in `torture/solvers/`. |
| Triangular scale and structure | The broad correctness grid uses size 100. Explicit empty-column, stored-zero, missing-diagonal, conjugated-diagonal, vector/matrix, and view cases cover structure. Size 127 retains Int8 diagonal-capacity coverage; operation-count checks retain a size-1,000 specialized-path case and mark the known transformed-product generic fallback broken. |

Other inference, aliasing, fixed-pattern, shape, empty-input, validation, and
issue-specific tests remain independently useful. Retain targeted pure-Julia overflow
and conversion tests. Complexity checks use operation counts or allocation growth.

## Running and measuring

The top-level `AGENTS.md` has the everyday commands. From the repository root, to
exercise the serial fallback without picking up a globally installed ParallelTestRunner,
and to measure one suite:

```sh
JULIA_LOAD_PATH="@:@stdlib" julia +nightly --project --startup-file=no test/runtests.jl
JULIA_LOAD_PATH="@:@stdlib" julia +nightly --project --startup-file=no test/runtests.jl torture
julia +nightly --project --startup-file=no --threads=1 --check-bounds=auto .ci/measure-tests.jl higherorderfns
julia +nightly --project --startup-file=no --threads=1 --check-bounds=auto .ci/measure-tests.jl torture/indexing
julia +nightly --project --startup-file=no --threads=1 --check-bounds=yes .ci/measure-tests.jl higherorderfns --verbose
```

The measurement script starts from a seeded RNG, selects one BLAS thread, verifies
that SparseArrays comes from the checkout, and times the selected suite inside a
Test testset. Its `MEASURE` line records elapsed, compilation, and GC seconds,
allocated bytes, Julia version, and package source path. `--verbose` enables nested
testset timing through Test. Package loading before the timed include and work in
child processes are not included in the parent's compilation/allocation totals;
process-suite elapsed time includes waiting for its children.

- Use a fresh Julia process for every sample and at least three samples per revision,
  with the same Julia build, machine, bounds setting, thread count, and cache state.
  Compare medians and variation.
- Report cold dependency preparation separately from warmed preparation and test
  execution.
- Retain CI's verbose per-suite reporting; compare bounds-job elapsed time and summed
  test-job durations across the unchanged platform matrix, as well as compilation
  versus execution.
- After reducing work, compare groupings in a trial runner with the same worker count
  and inventory. Change the default groups only with a repeatable scheduling benefit.
