# SparseArrays tests

Develop and test on Julia nightly. `runtests.jl` owns the ordinary suite inventory,
used by both ParallelTestRunner and the serial fallback for Julia Base CI. Selectors
match by prefix. Test files are named after the source area they cover, without
`sparse`/`linalg` prefixes; the solver suites are `cholmod`, `umfpack`, `spqr` and `solvers`.

`triangular.jl` contains the triangular product and solve tests in one scheduling
unit. `concatenation.jl` likewise contains all concatenation tests. Keep
feature-specific issue regressions next to that feature and preserve issue
references. The product and solve grids share their fixtures in the triangular file.

Ordinary tests cover numerical results, sparse structure, validation, aliasing,
inference, and algorithmic complexity. `ambiguous.jl` runs Aqua and ambiguity checks
in its separate quality environment; Aqua and Pkg are not ordinary test-target
dependencies. GPL solver references are guarded, while pure-Julia kernel tests also
run on builds without GPL libraries.

`threads_suite.jl` owns tests requiring fresh process state. Its `testprocess.jl`
helper preserves the active project and resolved load path, verifies the checkout
loaded by the child, and explicitly selects default-pool thread counts. It runs
solver concurrency checks with one and four threads, library-directory selection,
and `cholmod_lifetime.jl`. Deliberate GC calls are permitted only in that isolated
lifetime suite: they exercise finalization, allocation accounting, and temporary
rooting. Keep the rooting stress workload until a demonstrated reproducer supports
a smaller replacement.

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
issue-specific tests remain independently useful. Preserve both index types when
they select C entry points, and retain targeted pure-Julia overflow and conversion
tests. Complexity checks use operation counts or allocation growth, never elapsed
time assertions.

## Running and measuring

Run commands from the repository root:

```sh
julia +nightly --project -e 'using Pkg; Pkg.test()'
julia +nightly --project -e 'using Pkg; Pkg.test(test_args=["higherorderfns"])'
JULIA_LOAD_PATH="@:@stdlib" julia +nightly --project --startup-file=no test/runtests.jl
julia +nightly --project --startup-file=no --threads=1 --check-bounds=auto .ci/measure-tests.jl higherorderfns
julia +nightly --project --startup-file=no --threads=1 --check-bounds=yes .ci/measure-tests.jl higherorderfns --verbose
```

The restricted load path exercises the serial fallback without picking up a
globally installed ParallelTestRunner. Run quality checks using the same temporary
environment as the quality CI job:

```sh
julia +nightly --startup-file=no -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path=pwd()); Pkg.add(name="Aqua", version="0.8"); include("test/ambiguous.jl")'
```

The measurement script starts from a seeded RNG, selects one BLAS thread, verifies
that SparseArrays comes from the checkout, and times the selected suite inside a
Test testset. Its `MEASURE` line records elapsed, compilation, and GC seconds,
allocated bytes, Julia version, and package source path. `--verbose` enables nested
testset timing through Test. Package loading before the timed include and work in
child processes are not included in the parent's compilation/allocation totals;
process-suite elapsed time includes waiting for its children.

Use a fresh Julia process for every sample and at least three samples per revision,
with the same Julia build, machine, bounds setting, thread count, and cache state.
Compare medians and variation. Report cold dependency preparation separately from
warmed preparation and test execution. Retain CI's verbose per-suite reporting;
compare bounds-job elapsed time and summed test-job durations across the unchanged
platform matrix as well as compilation versus execution. Source coverage and the
coverage table above complement those measurements. After reducing work, compare
groupings in a trial runner with the same worker count and inventory. Change the
default groups only with a repeatable scheduling benefit; adding a feature file
does not require adding a worker task.
