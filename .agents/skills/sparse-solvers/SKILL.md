---
name: sparse-solvers
description: Rules for the SuiteSparse solver layer (CHOLMOD, UMFPACK, SPQR) - C memory ownership, callbacks, precision, workspaces, refactorization, and which calls lock a factorization so that solves can share one in parallel. Use before changing any solver code or factorization method.
---

# The sparse solver layer

The solvers wrap GPL C libraries, so they live apart from the rest of the package, which
reaches them only through LinearAlgebra's generic functions. The generated bindings are
never edited by hand; see the `suitesparse-wrappers` skill.

## C interop

- Never throw a Julia exception from inside a C callback. Record the error, return, and
  check the status after the call.
- Own every C pointer exactly once. Anything that can throw between allocation and
  wrapping frees on the way out; a call with several outputs initializes each to null and
  frees the siblings if wrapping one fails.
- Freeing nulls the wrapper's pointer, so finalizers and repeated frees are no-ops. Keep
  wrappers rooted while reading through raw pointers.
- Guard the invariants the C side assumes (contiguous outputs, matching types, sorted and
  packed flags). Drop stale state on failure, and free eagerly on refactorization.
- Initialization is lazy and once per process, before the first C call, not in
  `__init__`. Library handles belong to one module; every submodule imports each symbol it
  uses explicitly and never names a library by bare symbol.

## Numerics and interface

- The solvers work in double precision. Convert inputs explicitly (`float` is not a
  `Float64` cast for generic eltypes), convert results back, and keep `\` type-stable.
- Accept strided right-hand sides and their adjoint and transpose wrappers.
- A factorization holds no solve scratch. `ldiv!` allocates its own or takes a
  caller-owned workspace through a keyword, for `F`, `F'` and `transpose(F)` alike.
- When the output of a solve may alias the right-hand side, `copy` the right-hand side:
  `Base.unalias` of a view copies its whole parent.
- A refactorization checks size, eltype and symmetry before it writes to the factor, and
  a failed one leaves no stale factors behind.
- The solvers offer the same calls: `copy` and `deepcopy` of `F`, `F'` and `transpose(F)`;
  `ldiv!` with two or three arguments, a workspace, and a complex right-hand side for a
  real factorization; a workspace built from any of the three, and `similar` of one. A
  method added to one solver goes to all of them.
- Do not change ordering or tolerance defaults without an opt-in keyword.

## Threading

- A factorization that can change after construction has one plain `ReentrantLock`
  field. Every call that writes its mutable state (C factor objects, matrix arrays,
  status, control and info) holds that lock for the whole call, including argument checks
  that read the factor and any success check after a refactorization. So does `copy`, and
  everything that copies the factor internally.
- A call that only reads the factorization takes no lock, so tasks can share one factor in
  parallel: solves, determinants, the diagonal and the factor properties. The caller
  ensures that a factor is not modified while another task reads it, and the docs say so.
- A lock-free call writes nothing to the factorization: solve statistics go to the
  workspace, and a C option that would write to the factor is not passed. Before making
  another C call lock-free, check in the library's documentation or source that it does
  not write to the factor object.
- A factorization that is immutable after construction has no lock, and must stay that
  way: one in-place method would need a lock on every solve again.
- `copy(F)` is fully independent: it shares nothing that any call modifies, and it holds
  the source's lock while it reads that state. A task that refactorizes works on its own
  copy.
- Never hold the locks of two factorizations at once: lock the source, copy, release,
  then work on the copy. Where a locked method calls another, rely on reentrancy or split
  out an unlocked kernel, whichever keeps hot paths to one acquisition.
- Finalizers never lock; an explicit free locks, and the finalizer calls an unlocked
  helper.
- Document the threading behaviour in the manual.
