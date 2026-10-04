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
  eagerly on refactorization.
- The solvers work in double precision. Convert inputs explicitly; `float` is not a
  Float64 cast for generic eltypes. Convert results back and keep `\` type-stable.
- Accept strided right-hand sides and their adjoint and transpose wrappers.
- Initialization is lazy and process-once, done before the first C call rather than in
  `__init__`. Library handles belong to `LibSuiteSparse`; every submodule imports each
  symbol it uses explicitly, and never references a library by bare symbol.
- Each factorization has one plain `_lock::ReentrantLock` field. Every call that touches
  its mutable state (C factor objects, matrix arrays, workspaces, status, control and
  info) holds `@lock F._lock` for the whole call, including argument checks that read
  the factor and any `issuccess` check after refactorization. Read-only calls serialize
  too. Where a locked method calls another, rely on reentrancy or split out an unlocked
  kernel, whichever keeps hot paths to one acquisition.
- `copy(F)` is fully independent: it shares nothing that any call modifies, and it
  holds the source's lock while it reads any of that state. Parallel work uses one copy
  per task; document threading in the docs.
- Never hold the locks of two factorizations at once: lock the source, make the private
  copy, release, then work on the copy. Finalizers never lock; an explicit `free!`
  locks and the finalizer calls an unlocked helper.
- Do not change ordering or tolerance defaults without an opt-in keyword.
- Test both index types: SuiteSparse selects the C entry point by index type.
  Single-precision tests need well-conditioned inputs. Explicit `GC.gc()` calls belong
  only in the isolated lifetime tests, where they exercise finalization and rooting.
  Windows hangs have historically been CHOLMOD threading under GitHub Actions.
