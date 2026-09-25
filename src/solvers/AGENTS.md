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
- Factorizations have no locks. Solves and other reads must not write the factorization:
  keep solve scratch in a workspace passed to `ldiv!` or allocated per call, and pass
  `C_NULL` for UMFPACK's `Info` where the call only reads. Changing a factorization while
  another task uses it is the caller's responsibility, as for dense factorizations.
- `copy(F)` and `deepcopy(F)` return an independent factorization that shares nothing
  with `F`.
- Do not change ordering or tolerance defaults without an opt-in keyword.
- Test both index types: SuiteSparse selects the C entry point by index type.
  Single-precision tests need well-conditioned inputs. Explicit `GC.gc()` calls belong
  only in the isolated lifetime tests, where they exercise finalization and rooting.
  Windows hangs have historically been CHOLMOD threading under GitHub Actions.
