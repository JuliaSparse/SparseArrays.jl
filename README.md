# SparseArrays

| **Documentation**                                                 | **Build Status**                                                                                |
|:-----------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
|  [![][docs-img]][docs-url] | [![][ci-img]][ci-url] [![][codecov-img]][codecov-url] |

[docs-img]: https://img.shields.io/badge/docs-blue.svg
[docs-url]: https://sparsearrays.juliasparse.org/dev/

[ci-img]: https://github.com/JuliaSparse/SparseArrays.jl/actions/workflows/ci.yml/badge.svg?branch=main
[ci-url]: https://github.com/JuliaSparse/SparseArrays.jl/actions/workflows/ci.yml?query=branch%3Amain

[codecov-img]: https://codecov.io/gh/JuliaSparse/sparsearrays.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaSparse/sparsearrays.jl

SparseArrays.jl provides functionality for working with sparse arrays in Julia.

This package ships as part of the Julia stdlib, so its version is tied to the Julia
version: each Julia release bundles a fixed copy, and it cannot be upgraded separately.

```julia
using SparseArrays, LinearAlgebra

A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [4.0, 5.0, 1.0, 6.0])  # rows, columns, values
A[3, 1]          # 1.0
nnz(A)           # 4 stored entries
x = A \ ones(3)  # sparse direct solve through SuiteSparse
```

## Contributing

See [AGENTS.md](AGENTS.md) for the conventions this repository follows. Development
happens against Julia nightly. To run the tests, and to build the docs and run the doctests:

```sh
julia +nightly --project -e 'using Pkg; Pkg.test()'
julia +nightly --project=docs -e 'using Pkg; Pkg.develop(path="."); include("docs/make.jl")'
```

The second command edits `docs/Project.toml`; discard that change before committing.

## Updating SuiteSparse

In order to upgrade SparseArrays.jl to use a new release of SuiteSparse, the following steps are necessary:
1. Update SuiteSparse in Yggdrasil
2. Update the SuiteSparse wrappers in SparseArrays.jl/gen and generate the new wrappers
3. Run BumpStdlibs to update the SparseArrays.jl version in julia master
4. Update the relevant stdlibs in Julia to pull in the new releases
