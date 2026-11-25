---
name: write-tests
description: Write, place and organize tests - standard and comprehensive mode, fixed inputs, structural comparisons, keeping compile time down, and what an assertion must prove. Use whenever adding, changing or reorganizing a test.
---

# Writing tests

Julia's CI runs every suite of this stdlib serially in one process, with
`--depwarn=error`, and a test that fails for some inputs fails the version bump. Test time
is almost all compilation. Both facts shape every rule below.

## Two modes

- **Standard** mode is what `Pkg.test`, Julia's CI and almost every CI job run. It tests
  each feature and kernel once, for correctness, over one real and one complex eltype
  with `Int` indices. Another type, wrapper or shape appears only where it is the point
  of the test, and small or 32-bit index types not at all; the 32-bit CI jobs reach the
  32-bit entry points.
- **Comprehensive** mode also runs the guarded tests and the standalone issue suites. The
  guarded tests are the issue regressions and the wider corner cases: more element and
  index types, wrappers, promotion pairs, sizes, and allocation, inference and dispatch
  checks beyond one per kernel. One CI job runs it; run it locally before a PR that
  touches a kernel or a constructor.

**A new test is comprehensive.** A standalone issue reproducer goes in the issue suite
(the solver one if it uses a sparse solver); those suites are selected as a whole and need
no guards. Additional feature cases go in the existing feature testset, behind the
compile-time comprehensive guard. An unguarded test is only for new code, and is one
representative case; its variations are guarded. Keep existing standard regression
coverage when reorganizing, and keep issue references on regression tests.

Guard with `@static if`, which is resolved when the macro expands, so a standard run does
not lower or compile the guarded code. Write a test body once: when the modes differ in
the values a test runs over, the guard goes on the values, not around a copy of the body.

## Inputs

- **Check more than the values.** `S == D` against a dense reference passes when `S` is
  dense, has unsorted or repeated indices, or the wrong element or index type. Use the
  structural comparison helper, which checks all of these and prints what is wrong.
- **Use the shared fixtures**, not a square real random matrix. They have an empty row
  and column, a stored zero, both rectangular orientations and complex values that differ
  from their conjugates: what it takes for a swapped dimension, a stored zero read as
  structural, or a missing `conj` to change a result. They are of the types standard mode
  already compiles.
- **Inputs are fixed.** Use a fixture, a literal, or a matrix built so that the property
  the test relies on (positive definite, full rank, an off-diagonal entry) is guaranteed,
  and assert the property when the test depends on it. Randomness stays, seeded, only
  where it is the point: random constructors, sorting and permutation, a stress loop.
- **A testset owns its inputs.** Build them inside the testset. Arrays in module-level
  variables can be written by any testset, so what a later one sees depends on order,
  which also changes when a testset is guarded.
- Single-precision solver tests need well-conditioned inputs. Explicit `GC.gc()` calls
  belong only in isolated lifetime tests, where they exercise finalization and rooting.

## Compile time

- Prefer a subset to a full Cartesian grid, which is worth its compile time only over two
  short dimensions. Cover every value of every dimension, or every pair of values across
  dimensions, and add the corner cases by name: empty, stored zeros, a missing diagonal,
  unsorted or aliased input, an unusual index type.
- A whole top-level statement compiles as one block: keep top-level testsets short, and
  use `@testset for` rather than a top-level `for`, `let` or `begin` around testsets.
- Suites run with minimal optimization, because their own code runs once. Code inlined
  into a suite may then allocate where it otherwise would not, so a tight allocation bound
  measures through a shared helper function.
- A helper that takes arrays or types compiles again for each combination: mark its
  arguments `@nospecialize` unless it measures allocations or inference.
- A test type or helper that more than one suite needs lives in the shared helper module,
  never in a suite file, so a serial run defines it once and reuses what it compiled.
  Reuse an existing test type when one fits.
- A child process compiles everything it runs from scratch, and that, not its iteration
  count, is its cost. Standard mode starts as few children as possible.
- Load a file that defines a module or type with a top-level statement, not from a
  function a testset calls: the testset runs in the world from before the definition,
  which is an error under `--depwarn=error`.

## Assertions

- Do not add an assertion that an existing loop already makes for the same kernel path.
- A method that exists only for speed needs a test proving it is dispatched to, such as an
  operation count, not only a comparison with dense, which passes on the fallback too.
- After an expected throw, assert that the destination is unchanged.
- No wall-clock assertions. An allocation bound proves constancy, not zero.
- Match `@test_throws` messages loosely, because Base rewords errors.
- Ambiguity failures on nightly are often Base's. Check before blaming a change.
