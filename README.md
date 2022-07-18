# SparseArrays

| **Documentation**                                                 | **Build Status**                                                                                |
|:-----------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
|  [![][docs-img]][docs-url] | [![][ci-img]][ci-url] [![][codecov-img]][codecov-url] |

[docs-img]: https://img.shields.io/badge/docs-blue.svg
[docs-url]: https://sparsearrays.juliasparse.org/dev/

[docs-v1-img]: https://img.shields.io/badge/docs-v1-blue.svg
[docs-v1-url]: https://sparsearrays.juliasparse.org/v1/

[ci-img]: https://github.com/JuliaSparse/sparsearrays.jl/workflows/CI/badge.svg?branch=main
[ci-url]: https://github.com/JuliaSparse/sparsearrays.jl/actions?query=workflow%3A%22CI%22

[codecov-img]: https://codecov.io/gh/JuliaSparse/sparsearrays.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaSparse/sparsearrays.jl

This package ships as part of the Julia stdlib.

SparseArrays.jl provides functionality for working with sparse arrays in Julia.

## Using newer version of this package with julia master

You need to build Julia from scratch to use the git version (or other version) of this package. The build process is the same, but `DEPS_GIT` should be set when building i.e. `make DEPS_GIT="SparseArrays" ...`. The other option is to manually select the commit in `stdlib/SparseArrays.version`. 

It's also possible to load a development version of the package using [the trick used in `Pkg.jl`](https://github.com/JuliaLang/Pkg.jl) but the capabilities are limited as all other packages will depend on the stdlib version of the package and will not work with the modified package. 

The main environment may become inconsistent so you might need to run `Pkg.instantiate()` and/or `Pkg.resolve()` in the main or project environments if Julia complains about missing `Serialization` in this package's dependencies. 

For older (1.8 and before) `SuiteSparse.jl` needs to be bumped too.


