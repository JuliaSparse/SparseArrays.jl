# SparseArrays

| **Documentation**                                                 | **Build Status**                                                                                |
|:-----------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
|  [![][docs-img]][docs-url] | [![][ci-img]][ci-url] [![][codecov-img]][codecov-url] |

[docs-img]: https://img.shields.io/badge/docs-blue.svg
[docs-url]: https://sparsearrays.juliasparse.org/dev/

[docs-v1-img]: https://img.shields.io/badge/docs-v1-blue.svg
[docs-v1-url]: https://sparsearrays.juliasparse.org/v1/

[ci-img]: https://github.com/JuliaSparse/SparseArrays.jl/actions/workflows/ci.yml/badge.svg?branch=main
[ci-url]: https://github.com/JuliaSparse/SparseArrays.jl/actions/workflows/ci.yml?query=branch%3Amain

[codecov-img]: https://codecov.io/gh/JuliaSparse/sparsearrays.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaSparse/sparsearrays.jl

This package ships as part of the Julia stdlib.

SparseArrays.jl provides functionality for working with sparse arrays in Julia.

## Updating SuiteSparse

In order to upgrade SparseArrays.jl to use a new release of SuiteSparse, the following steps are necessary:
1. Update SuiteSparse in Yggdrasil
2. Update the SuiteSparse wrappers in SparseArrays.jl/gen and generate the new wrappers
3. Run BumpStdlibs to update the SparseArrays.jl version in julia master
4. Update the relevant stdlibs in Julia to pull in the new releases
