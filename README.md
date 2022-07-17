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

## Using newer version of this package

You need to build Julia from scratch to use the git version of this package. The build process is the same, but in Julia-1.8 or later, `DEPS_GIT` should be set like `make DEPS_GIT="SparseArrays SuiteSparse"`. It's important to build both packages from git as older SuiteSparse.jl overwrite the UmfpackLU structures in this repo (and possibly more) as it is loaded afterward. Another solution is changing the commit chosen in `stdlib/{SparseArrays/SuiteSparse}.version`. Again the SuiteSparse commit needs to be the latest if you choose a commit in SparseArrays after May 2020 (see commit `a15fe4b79307cd5bd55f00609297bbe37072033be). The main environment may become inconsistent so you might need to run `Pkg.instantiate()` and/or `Pkg.resolve` in case Julia complains about missing `Serialization` in this package's dependencies.


Of course it's possible to load a development version of the package using [the trick used in `Pkg.jl`](https://github.com/JuliaLang/Pkg.jl) but the capabilities are limited as all other packages will depend on the stdlib version of the package.
