# AGENTS.md for `test/`

How the test suite is organized, what its reduced grids must keep covering, and how to
measure test time, in addition to the top-level `AGENTS.md`.

## Layout

- `runtests.jl` owns the suite inventory, used by both ParallelTestRunner and the serial
  fallback for Julia Base CI. Selectors match by prefix. Adding a feature file does not
  require adding a worker task.
- Test files are named after the source area they cover. `solvers/` mirrors
  `src/solvers/`: its suites are `solvers/cholmod`, `solvers/umfpack`, `solvers/spqr`
  and `solvers/solvers`, so the selector `solvers` runs them all. They run only when
  `Base.USE_GPL_LIBS` is true; pure-Julia kernel tests also run on builds without GPL
  libraries.
- `triangular.jl` holds the triangular product and solve tests as one scheduling unit,
  and the two grids share their fixtures. `concatenation.jl` likewise holds all
  concatenation tests.
- Preserve issue references on regression tests.
- `ambiguous.jl` is not part of the default run; CI gives it its own job.
- `threads.jl` owns tests requiring fresh process state. Its `testprocess.jl` helper
  preserves the active project and resolved load path, verifies the checkout loaded by
  the child, and explicitly selects default-pool thread counts. The scripts it runs in
  child processes live in `solvers/` and are not suites themselves: the solver
  concurrency checks in `solvers/threads.jl`, with one and four threads, and
  `solvers/cholmod_lifetime.jl`. It also covers library-directory selection, whose
  argument checks run without GPL libraries. Keep the rooting stress
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
| Sparse/triangular products | All four wrappers and both operands' three transforms, in both multiplication orders, with Float64 and ComplexF32. All nine Int/Float64/ComplexF32 promotion pairs use upper/lower representatives. Dense result types are checked. |
| Dense/sparse `mul!` | Both operand orders and every transform pair for Int, Float64, ComplexF64, and BigFloat. Boolean and numeric zero/one/general coefficient pairs cover sparse identity/transpose/adjoint kernels and plain/wrapped dense-left kernels separately. Existing noncommutative regressions remain. |
| CHOLMOD operations | Single-input tests cover both precisions, real/complex values, and both supported C index types. Mixed-input operations retain all precision/type pairs. Ownership, invalid-wrapper cleanup, repeated `free!`, buffer isolation, Common accounting, and rooting tests remain in the lifetime process. |
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
julia +nightly --project --startup-file=no --threads=1 --check-bounds=auto .ci/measure-tests.jl higherorderfns
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
