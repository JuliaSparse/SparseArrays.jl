# AGENTS.md for `test/`

How the test suite is organized, its standard and comprehensive modes, where a new
test goes, and how to measure test time, in addition to the top-level `AGENTS.md`.

## Layout

- `runtests.jl` owns the suite inventory, used by both ParallelTestRunner and the serial
  fallback for Julia Base CI. On both paths selectors match suite names by prefix and a
  `!prefix` selector excludes; the serial path takes selectors, `--list` and
  `--comprehensive` only, and errors when nothing matches. `--jobs=N` and `PTR_NUM_JOBS`
  are honoured; without either, the runner uses `Sys.CPU_THREADS` workers. Adding a
  feature file does not require adding a worker task.
- `SparseTestHelpers.jl` is a module holding everything the suites share: the
  `getproperty` guard that makes field access on the sparse types an error, the type
  sets (`STD_ELTYPES` and `core_itypes`, and `itypes`, which adds `Int32`, for guarded
  code), the grid helpers `eachvalue` and `pairwise`, `same_pattern` and `exact_equal`,
  and every type a test defines (`OpCount`
  with `mulcount`/`eqcount`/`opcount_sparse`, `CountedReads`, `WrappedSparseVector`,
  `NonCSCSparse`, `SimpleSMatrix`, `MockTropical`, `Meters` and the rest). Every suite is
  a module that `include`s `testhelpers.jl` first (`../testhelpers.jl` from a
  subdirectory), which loads that module into `Main` once per process and brings its
  exports in. A serial run therefore defines each test type once, and a kernel compiled
  for it in one suite is reused by the next. **Do not define a `struct` in a suite file**:
  add it to `SparseTestHelpers.jl` and export it, and reuse a type that is already there
  when it fits. A helper that a second suite needs goes there as well. `ambiguous.jl`
  alone does not include `testhelpers.jl`.
- Test files are named after the source area they cover. `solvers/` mirrors
  `src/solvers/`: its suites are `solvers/cholmod`, `solvers/umfpack`, `solvers/spqr`,
  `solvers/solvers` and `solvers/threads`, so the selector `solvers` runs them all.
- `trim/` is not a suite: it is a small app that the `trim` CI job builds with
  `juliac --trim=safe` and runs. It covers the concatenation hooks SparseArrays adds to
  Base for dense arrays, which Julia's own trim test reaches, and the main sparse
  operations: construction, indexing, broadcast and `map`, reductions, search, norms,
  products and `mul!`, triangular solves, structural functions, `SparseVector`, views,
  the LinearAlgebra wrappers, `FixedSparseCSC`, and the solvers through the generic
  LinearAlgebra functions, over Int, Bool, Float32, Float64 and complex eltypes and
  Int32 indices. `show` is left out, because Base's array printing does not trim. A
  trimmed binary cannot yet load a `LazyLibrary`, so it cannot call BLAS and writes
  expected values out, and it runs the solvers only when given the `solvers` argument;
  CI builds them, which verifies that they trim.
- The files in `solvers/` carry no `Base.USE_GPL_LIBS` guards of their own; the
  top-level `AGENTS.md` has the rule for what belongs there.
- `triangular.jl` holds the triangular product and solve tests as one scheduling unit,
  and the two grids share their fixtures. `concatenation.jl` likewise holds all
  concatenation tests, and `matmul.jl` every other product test, for vectors as well as
  matrices: scaling, the BLAS-2 grid and products with LinearAlgebra's Q types, while
  `sparsevector.jl` keeps the vector `axpy!` and `dot` tests. The `transpose`, `adjoint`
  and `permute` tests, including the in-place forms, live in `sparsematrix.jl`.
- Preserve issue references on regression tests.
- `ambiguous.jl` is in the inventory but skipped unless a selector names it; CI gives it
  its own job. It restores the depot, load path, environment and active project in a
  `finally`, so an Aqua failure on Base CI leaves the worker usable.
- `solvers/threads.jl` owns the tests requiring fresh process state. Its `testprocess.jl`
  helper preserves the active project and resolved load path, verifies the checkout
  loaded by the child, and explicitly selects default-pool thread counts. A child
  compiles everything it runs from scratch, and that, not its iteration count, is its
  cost. So standard mode starts one child, with four threads, for the concurrent
  factorizations and one shared `lu` and `cholesky` factor in `threads_child.jl`, which
  loads nothing it does not need. The other shared-factor cases at one and four threads,
  `cholmod_lifetime.jl` (not a suite of its own) and the library-directory override in a
  child are guarded, in the same file. Keep the rooting stress workload in the lifetime
  file until a demonstrated reproducer supports a smaller replacement.

## Standard and comprehensive mode

- **Standard** mode is what `Pkg.test`, Julia's own CI and every CI job but the coverage
  job, which is the Linux x64 one, run. It tests each feature and kernel once, for
  correctness, over `Float64` and `ComplexF64` with `Int` indices. Another type, wrapper
  or shape appears only where it is the point of the test, and `Int8`, `Int32` and
  `UInt8` not at all: a solver suite uses the build's `Int`, so the 32-bit C entry points
  get their standard coverage from the 32-bit CI jobs. Julia's CI runs every suite
  serially in one process, so standard mode must stay fast.
- **Comprehensive** mode runs, in the same files, the tests guarded with
  `@static if COMPREHENSIVE` as well, and the `issues` suite, which a standard run skips
  unless a selector names it. It is selected by `SPARSEARRAYS_TEST_COMPREHENSIVE=true` in
  the environment, which `runtests.jl` sets for the `--comprehensive` argument, and in CI
  by the coverage job only. The guarded tests are the issue regressions and the wider
  corner cases: more element and index types, wrappers, promotion pairs, sizes, and the
  allocation, inference and dispatch checks beyond one per kernel.

**A new test is guarded.** That covers a regression test for an issue and any additional
case for code the standard tests already exercise. An unguarded test is only for new
code, and is one representative case; its variations are guarded.

`COMPREHENSIVE` comes from `SparseTestHelpers.jl`. Guard with `@static`, which is
resolved when the macro expands, so that a standard run does not lower or compile the
guarded code:

```julia
@static if COMPREHENSIVE          # whole testsets; the body is not indented
@testset "issue #1234" begin ... end
end

@testset "solve, $T" for T in (STD_ELTYPES..., (@static COMPREHENSIVE ? (Float32, BigFloat) : ())...)
    ...                           # one body for both modes
    @static if COMPREHENSIVE
        @test_throws DimensionMismatch ...
    end
end
```

Write a test body once. When the two modes differ in the values a test runs over, the
guard goes on the values, not around a copy of the body.

Prefer a subset to a full Cartesian grid, which is worth its compile time only over two
short dimensions. Take `eachvalue(dims...)`, in which every value of every dimension
appears, or `pairwise(dims...)`, in which every two values of different dimensions meet
(for two dimensions that is the full grid), and add the corner cases by name: empty,
stored zeros, a missing diagonal, unsorted or aliased input, an unusual index type.

Test time is almost all compilation, so it is the number of distinct type combinations
that costs, not sizes or repetitions. Three things follow for the test code itself. A
whole top-level statement compiles as one block: keep top-level testsets short, and use
a `@testset for` rather than a top-level `for`, `let` or `begin` around testsets.
`testhelpers.jl` sets `@compiler_options optimize=0` in every suite module, because the
suites' own code runs once; the methods under test are compiled as usual. Interpreting
the suites instead (`compile=min`) saves no more and breaks the allocation tests. And a
helper that takes arrays or types is compiled again for each combination it is called
with: mark its arguments `@nospecialize` unless it measures allocations or inference.

## What a reduction must keep

Cutting a grid separates independent dimensions instead of compiling their Cartesian
product. Each reduction must identify the replacement for every removed dimension; equal
assertion counts or line coverage alone are insufficient. In standard mode:

- every kernel and every method that exists for speed is called at least once, with its
  dispatch or operation-count assertion;
- real and complex values, vector and matrix, and each wrapper family (adjoint and
  transpose, triangular, symmetric and Hermitian, views) appear for each operation that
  specializes on them, though not in every combination;
- the solver suites reach both precisions once per solver, with the build's `Int`;
- error paths keep their check that the destination is untouched.

The structural corner cases (empty columns, stored zeros, a missing diagonal), the
promotion pairs, the unusual element and index types, the allocation and inference
bounds beyond one per kernel, and the lifetime and thread-count variations are
comprehensive. Complexity checks use operation counts or allocation growth.

## Running and measuring

The top-level `AGENTS.md` has the everyday commands. From the repository root, to
exercise the serial fallback without picking up a globally installed ParallelTestRunner,
and to measure one suite:

```sh
JULIA_LOAD_PATH="@:@stdlib" julia +nightly --project --startup-file=no test/runtests.jl
julia +nightly --project --startup-file=no --threads=1 --check-bounds=auto .ci/measure-tests.jl higherorderfns
julia +nightly --project --startup-file=no --threads=1 --check-bounds=yes .ci/measure-tests.jl higherorderfns --verbose
SPARSEARRAYS_TEST_COMPREHENSIVE=true julia +nightly --project --startup-file=no --threads=1 .ci/measure-tests.jl higherorderfns
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
- Julia keeps a persistent cache of JIT-compiled code under `~/.julia/cache`, which makes
  a repeated run far faster than a fresh CI machine. Set `JULIA_OBJCACHE=0` when
  measuring.
- On a shared machine load moves a sample by a factor of two or more. Interleave the
  revisions suite by suite and compare minima; the allocated bytes are a load-independent
  check on the direction.
- The number of method instances compiled is the other load-independent measure, and the
  one that tracks test time most closely: run with `--trace-compile=FILE` and count its
  lines. The file also shows which types and methods a suite spends its time on.
- Julia's CI runs every suite in standard mode in one process, where a kernel compiled
  by one suite is reused by the next, so the sum of per-suite measurements overstates
  it. Measure the first command above, with `JULIA_OBJCACHE=0`, for the figure that
  matters.
- Report cold dependency preparation separately from warmed preparation and test
  execution.
- Retain CI's verbose per-suite reporting; compare bounds-job elapsed time and summed
  test-job durations across the unchanged platform matrix, as well as compilation
  versus execution.
- After reducing work, compare groupings in a trial runner with the same worker count
  and inventory. Change the default groups only with a repeatable scheduling benefit.
