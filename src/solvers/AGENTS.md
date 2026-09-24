# AGENTS.md for `src/solvers/`

Rules for the SuiteSparse solver layer (CHOLMOD, UMFPACK, SPQR), in addition to the
top-level `AGENTS.md`. This directory and `test/solvers/` hold everything that depends
on the GPL libraries; see the Layout section there.

- `wrappers.jl` is generated from the SuiteSparse headers by `gen/`. Never edit it by
  hand; `gen/AGENTS.md` covers regenerating it and upgrading SuiteSparse.
- Never throw a Julia exception from inside a C callback. Record the error, return, and
  check the status after the call.
- Own every C pointer exactly once. Anything that can throw between allocation and
  wrapping frees on the way out; multi-output calls initialize every output to null
  and free the siblings if wrapping one fails.
- `free!` nulls the wrapper's pointer so finalizers and repeated calls are no-ops.
  Keep wrappers rooted while reading through raw pointers.
- Guard the invariants the C side assumes (contiguous outputs, matching types, sorted
  and packed flags on CHOLMOD sparse structs). Drop stale state on failure and free
  eagerly on refactorization, except what copies still share: `UmfpackLU` copies are
  copy-on-write, so never write or free anything while its share count exceeds one.
- The solvers work in double precision. Convert inputs explicitly; `float` is not a
  Float64 cast for generic eltypes. Convert results back and keep `\` type-stable.
- Accept strided right-hand sides and their adjoint and transpose wrappers.
- Initialization is lazy and process-once, done before the first C call rather than in
  `__init__`. Library handles belong to `LibSuiteSparse`; every submodule imports each
  symbol it uses explicitly, and never references a library by bare symbol.
- Every call that reads or changes a factorization's mutable state (its factors,
  workspaces, or arrays shared with copies) holds that factorization's internal lock
  for the whole call, including argument checks that read the factor. Use
  `@_readlock F expr` and `@_writelock F expr` from `factorlock.jl`, with a
  `_factorlock(F)` method returning `getfield(F, :_lock)`; the lock is a `FactorLock`
  (readers-writer) or a `ReentrantLock` (both modes exclusive), and call sites say
  which mode they need either way. The lock is never public: hide `_lock` from
  `propertynames` and `getproperty`, and read it internally with `getfield`.
- Public methods are thin locking wrappers around unlocked kernels that assume the
  caller holds the lock, so internal paths take it once. A read cannot be upgraded to a
  write (it throws); take the write lock up front. Never hold the locks of two
  factorizations at once: copy-then-mutate code reads the source, releases it, then
  locks the new object. Finalizers never lock.
- Use separate copies for parallel solves, and document threading in the docs.
- Do not change ordering or tolerance defaults without an opt-in keyword.
- Test both index types: SuiteSparse selects the C entry point by index type.
  Single-precision tests need well-conditioned inputs. Explicit `GC.gc()` calls belong
  only in the isolated lifetime tests, where they exercise finalization and rooting.
  Windows hangs have historically been CHOLMOD threading under GitHub Actions.
