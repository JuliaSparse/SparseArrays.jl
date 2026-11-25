---
name: test-time
description: Measure SparseArrays.jl test time and cut test grids without losing coverage. Use when a change adds or reorganizes tests, when asked to speed up the suite, or before claiming a test-time improvement.
---

# Test time

Test time is almost all compilation, so it is the number of distinct type combinations
that costs, not sizes or repetitions. `test/AGENTS.md` has the rules for writing tests;
this is how to measure them and how to cut a grid.

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

From the repository root, to exercise the serial fallback without picking up a globally installed ParallelTestRunner,
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
- Keep the `--verbose` per-suite reporting that `runtests.jl` turns on in CI.
