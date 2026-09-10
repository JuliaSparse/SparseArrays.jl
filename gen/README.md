# How to auto-generate the wrappers

1. `cd` to this directory
2. Update the SuiteSparse version in `Makefile`
3. run `make`. The updated wrappers are written to `src/solvers/wrappers.jl`; the banner
   at the top of that file comes from `prologue.jl` in this directory.

Note: `Makefile` downloads the `x86_64-linux-gnu` build of `SuiteSparse_jll` to obtain the
headers. The headers are platform independent, so the generated wrappers are used on all
platforms. Keep the version in `Makefile` in sync with the `SuiteSparse_jll` compat entry
in the top-level `Project.toml`.

# How to upgrade Clang.jl

1. `cd` to this directory
2. if you want to change major version, change the compat bound in `Project.toml`.
   Note: since you're going through a breaking release, you _may_ have to adapt the `generator.jl` script
3. run `julia --project` and then in the Julia REPL, run `pkg> up`
