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
  all platforms. `generator.jl` takes that unpacked directory as its only argument and
  never reads the headers of the running Julia's `SuiteSparse_jll`, which is a fixed
  stdlib that Pkg cannot upgrade or pin.
- Keep the version in `Makefile` in sync with the `SuiteSparse_jll` compat entry in the
  top-level `Project.toml`.
- `generator.jl` lists the headers to wrap, and `library_names` in `generator.toml` maps
  each header to its library. The keys are matched as patterns against the header path,
  so `/amd.h` and `/colamd.h` carry the separator that keeps the first from matching the
  second. AMD and COLAMD are BSD-licensed and present on a build without GPL libraries,
  so their wrappers may be called from outside `src/solvers/`.
- To drop a macro Clang.jl cannot handle, add it to `output_ignorelist` in
  `generator.toml`.
- Never edit `src/solvers/wrappers.jl` by hand: change `prologue.jl`, `generator.toml` or
  `generator.jl` and regenerate.

## Upgrading Clang.jl or JuliaFormatter

1. `cd` to this directory.
2. To change major version, change the compat bound in `Project.toml`. A breaking
   Clang.jl release may require adapting `generator.jl`.
3. Run `make` and commit the regenerated `src/solvers/wrappers.jl` together with the
   compat change; a diff in the wrappers is the output change of the new release, and
   the PR text should say so.
