---
name: test-time
description: Measure test time and cut test grids without losing coverage. Use when a change adds or reorganizes tests, when asked to speed up the suite, or before claiming a test-time improvement.
---

# Test time

Test time is almost all compilation, so it is the number of distinct type combinations
that costs, not sizes or repetitions.

## What a reduction must keep

Cutting a grid separates independent dimensions instead of compiling their Cartesian
product. Each reduction names the replacement for every removed dimension; equal
assertion counts or line coverage alone do not show that. In standard mode:

- every kernel and every method that exists for speed is called at least once, with its
  dispatch or operation-count assertion;
- real and complex values, vector and matrix, and each wrapper family (adjoint and
  transpose, triangular, symmetric and Hermitian, views) appear for each operation that
  specializes on them, though not in every combination;
- each solver is reached in both precisions, with the build's `Int`;
- error paths keep their check that the destination is untouched.

The structural corner cases (empty columns, stored zeros, a missing diagonal), promotion
pairs, unusual element and index types, allocation and inference bounds beyond one per
kernel, and lifetime and thread-count variations are comprehensive. Complexity checks use
operation counts or allocation growth.

## Measuring

Use the repository's measurement script, which times one suite in a fresh process with a
seeded RNG and one BLAS thread, and reports elapsed, compilation and GC time and
allocated bytes.

- Take a fresh Julia process for every sample and at least three samples per revision,
  with the same Julia build, machine, bounds setting, thread count and cache state.
- Disable Julia's persistent cache of compiled code (`JULIA_OBJCACHE=0`): with it, a
  repeated run is far faster than a fresh CI machine.
- On a shared machine load moves a sample by a factor of two or more. Interleave the
  revisions suite by suite and compare minima; allocated bytes are a load-independent check
  on the direction.
- The number of method instances compiled is the load-independent measure that tracks
  test time most closely: run with `--trace-compile=FILE` and count its lines. The file
  also shows which types and methods a suite spends its time on.
- Julia's CI runs every suite serially in one process, where a kernel compiled by one
  suite is reused by the next, so the sum of per-suite figures overstates it. The figure
  that matters is the whole serial run.
- Child processes compile everything they run from scratch; their cost is in the parent's
  elapsed time but not in its compilation totals.
