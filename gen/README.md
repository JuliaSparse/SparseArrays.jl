# How to auto-generate the wrappers

1. `cd` to this directory
2. Update the SuiteSparse version in `Makefile`
3. run `make`, then you could find the updated wrappers in the `lib` folder

# How to upgrade Clang.jl

1. `cd` to this directory
2. if you want to change major version, change the compat bound in `Project.toml`.
   Note: since you're going through a breaking release, you _may_ have to adapt the `generator.jl` script
3. run `julia --project` and then in the Julia REPL, run `pkg> up`
