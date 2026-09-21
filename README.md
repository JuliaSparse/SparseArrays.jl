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

Development happens against Julia nightly. To run the tests:

```sh
julia +nightly --project -e 'using Pkg; Pkg.test()'
```

[AGENTS.md](AGENTS.md) has the conventions this repository follows, the commands for
running a single test file, the whitespace check and the doctests, and pointers to the
guides for the solvers, the test suite and upgrading SuiteSparse.
