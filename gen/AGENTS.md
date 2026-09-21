# AGENTS.md for `gen/`

How to regenerate the SuiteSparse wrappers and upgrade SuiteSparse, in addition to the
top-level `AGENTS.md` and `src/solvers/AGENTS.md`.

## Updating SuiteSparse

Moving SparseArrays.jl to a new SuiteSparse release takes four steps, in order:

1. Update SuiteSparse in Yggdrasil.
2. Update the version here and regenerate the wrappers (below).
3. Run BumpStdlibs to update the SparseArrays.jl version in Julia master.
4. Update the relevant stdlibs in Julia to pull in the new releases.

The bump cannot be tested here until nightly bundles the new jll, so its CI is expected
to be red; do not delete the merged bump PR's branch.

## Regenerating the wrappers

1. `cd` to this directory.
2. Update the SuiteSparse version (`VER`) and the `SuiteSparse_jll` build number
   (`JLL_BUILD`) in `Makefile`.
3. Run `make`. The updated wrappers are written to `src/solvers/wrappers.jl`; the banner
   at the top of that file comes from `prologue.jl`.

- `Makefile` downloads the `x86_64-linux-gnu` build of `SuiteSparse_jll` to obtain the
  headers. The headers are platform independent, so the generated wrappers are used on
  all platforms.
- Keep the version in `Makefile` in sync with the `SuiteSparse_jll` compat entry in the
  top-level `Project.toml`.
- To drop a macro Clang.jl cannot handle, add it to `output_ignorelist` in
  `generator.toml`.

## Upgrading Clang.jl

1. `cd` to this directory.
2. To change major version, change the compat bound in `Project.toml`. A breaking
   release may require adapting `generator.jl`.
3. Run `julia --project`, then `pkg> up` in the REPL.
