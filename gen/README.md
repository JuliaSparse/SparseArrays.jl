# How to auto-generate the wrappers

1. `cd` to this directory
2. run `julia --project generator.jl`, then you could find the updated wrappers in the `lib` folder

## How to upgrade SuiteSparse_jll

1. update `SuiteSparse_jll` in Yggdrasil to the desired version
2. `cd` to this directory
3. run `julia --project` and then in the Julia REPL, run `pkg>add SuiteSparse#<COMMIT_HASH>`, where `<COMMIT_HASH>` is the commit hash corresponding to the desired version of the package

## How to upgrade Clang.jl

1. `cd` to this directory
2. if you want to change major version, change the compat bound in `Project.toml`.
   Note: since you're going through a breaking release, you _may_ have to adapt the `generator.jl` script
3. run `julia --project` and then in the Julia REPL, run `pkg> up`
