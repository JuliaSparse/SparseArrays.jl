# This file is a part of Julia. License is MIT: https://julialang.org/license

# Pure-Julia supernodal sparse LU with threshold partial pivoting: the left-looking
# supernodal factorization of SuperLU (Demmel, Eisenstat, Gilbert, Li and Liu, SIMAX 20(3),
# 1999), processed in panels. A square matrix with a nearly symmetric pattern takes the
# symmetric strategy: the maximum-weight matching and scaling of PARDISO (Schenk and
# Gärtner, FGCS 20(3), 2004), an AMD ordering of A + Aᵀ, and panels from its supernodes. Any
# other matrix takes the unsymmetric one: a COLAMD ordering of A and panels from its column
# elimination tree.
#
# The matching, the symmetric pattern and the elimination tree, postorder and supernode
# amalgamation of the analysis are ported from LinearSolve.jl's `src/SupernodalLU` (MIT,
# Copyright (c) 2021 SciML and contributors). The ordering calls the BSD-licensed
# SuiteSparse AMD, which ships on builds without GPL libraries.

module Supernodal

using LinearAlgebra
using LinearAlgebra: AdjointFactorization, TransposeFactorization, require_one_based_indexing
using ..SparseArrays: SparseArrays, SparseMatrixCSC, AbstractSparseMatrixCSC, getcolptr,
    rowvals, nonzeros, nnz
using ..LibSuiteSparse: amd_order, amd_l_order, colamd, colamd_l, colamd_recommended,
    colamd_l_recommended, init_suitesparse, COLAMD_STATS, COLAMD_STATUS

include("ordering.jl")
include("symbolic.jl")
include("matching.jl")
include("numeric.jl")
include("solve.jl")
include("factorization.jl")

end # module
