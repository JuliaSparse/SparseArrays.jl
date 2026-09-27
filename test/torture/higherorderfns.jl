# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for sparse `map` and `broadcast`. Each testset names the
# issue it guards; they run only when the `torture` suite is selected.

module TortureHigherOrderFnsTests
using Test
using SparseArrays
using LinearAlgebra
include("../testhelpers.jl")

@testset "issue #5824" begin
    @test sprand(4,5,0.5).^0 == sparse(fill(1,4,5))
end

@testset "issue #12118: sparse matrices are closed under +, -, min, max" begin
    A12118 = sparse([1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5])
    B12118 = sparse([1,2,4,5],   [1,2,3,5],   [2,1,-1,-2])

    @test A12118 + B12118 == sparse([1,2,3,4,4,5], [1,2,3,3,4,5], [3,3,3,-1,4,3])
    @test typeof(A12118 + B12118) == SparseMatrixCSC{Int,Int}

    @test A12118 - B12118 == sparse([1,2,3,4,4,5], [1,2,3,3,4,5], [-1,1,3,1,4,7])
    @test typeof(A12118 - B12118) == SparseMatrixCSC{Int,Int}

    @test max.(A12118, B12118) == sparse([1,2,3,4,5], [1,2,3,4,5], [2,2,3,4,5])
    @test typeof(max.(A12118, B12118)) == SparseMatrixCSC{Int,Int}

    @test min.(A12118, B12118) == sparse([1,2,4,5], [1,2,3,5], [1,1,-1,-2])
    @test typeof(min.(A12118, B12118)) == SparseMatrixCSC{Int,Int}
end

@testset "issue #13024" begin
    A13024 = sparse([1,2,3,4,5], [1,2,3,4,5], fill(true,5))
    B13024 = sparse([1,2,4,5],   [1,2,3,5],   fill(true,4))

    @test broadcast(&, A13024, B13024) == sparse([1,2,5], [1,2,5], fill(true,3))
    @test typeof(broadcast(&, A13024, B13024)) == SparseMatrixCSC{Bool,Int}

    @test broadcast(|, A13024, B13024) == sparse([1,2,3,4,4,5], [1,2,3,3,4,5], fill(true,6))
    @test typeof(broadcast(|, A13024, B13024)) == SparseMatrixCSC{Bool,Int}

    @test broadcast(⊻, A13024, B13024) == sparse([3,4,4], [3,3,4], fill(true,3), 5, 5)
    @test typeof(broadcast(⊻, A13024, B13024)) == SparseMatrixCSC{Bool,Int}

    @test broadcast(max, A13024, B13024) == sparse([1,2,3,4,4,5], [1,2,3,3,4,5], fill(true,6))
    @test typeof(broadcast(max, A13024, B13024)) == SparseMatrixCSC{Bool,Int}

    @test broadcast(min, A13024, B13024) == sparse([1,2,5], [1,2,5], fill(true,3))
    @test typeof(broadcast(min, A13024, B13024)) == SparseMatrixCSC{Bool,Int}

    for op in (+, -)
        @test op(A13024, B13024) == op(Array(A13024), Array(B13024))
    end
    for op in (max, min, &, |, xor)
        @test op.(A13024, B13024) == op.(Array(A13024), Array(B13024))
    end
end

# Check that `broadcast` methods specialized for unary operations over `SparseMatrixCSC`s
# are called. (Issue #18705.) EDIT: #19239 unified broadcast over a single sparse matrix,
# eliminating the former operation classes.
@testset "issue #18705" begin
    S = sparse(Diagonal(1.0:5.0))
    @test isa(sin.(S), SparseMatrixCSC)
end

# Check that `broadcast` methods specialized for unary operations over
# `SparseMatrixCSC`s determine a reasonable return type.
@testset "issue #18974" begin
    S = sparse(Diagonal(Int64(1):Int64(4)))
    @test eltype(sin.(S)) == Float64
end

# The full grid behind the factored scalar/sparse broadcast tests of `higherorderfns.jl`:
# every array form leading the argument list, one to five arrays, every placement of
# one to three scalars among them, and both a zero-preserving and an order-sensitive
# function.
@testset "broadcast[!] over every combination of scalars and sparse vectors/matrices" begin
    N, M, p = 10, 12, 0.5
    elT = Float64
    s, t, u = Float32(2), Float32(3), Float32(5)
    V = sprand(elT, N, p)
    Vᵀ = transpose(sprand(elT, 1, N, p))
    A = sprand(elT, N, M, p)
    Aᵀ = transpose(sprand(elT, M, N, p))
    forms = (A, V, Aᵀ, Vᵀ)
    ordered(xs...) = foldl((x, y) -> 2x + y, xs)
    function check_scalar_broadcast(f, sparseargs, alloc_limit=1028)
        denseargs = map(x -> x isa AbstractArray ? Array(x) : x, sparseargs)
        fX = broadcast(f, denseargs...)
        X = @inferred broadcast(f, sparseargs...)
        @test X == sparse(fX)
        @test typeof(X) === typeof(sparse(fX))
        @test (@inferred broadcast!(f, X, sparseargs...)) === X
        @test X == sparse(broadcast!(f, fX, denseargs...))
        X = sparse(fX)
        # Transposed sparse inputs require materializing CSC copies.
        extra = sum(x -> x isa Transpose ? @allocated(SparseMatrixCSC(x)) + 128 : 0, sparseargs)
        @test (@allocated broadcast!(f, X, sparseargs...)) <= extra + alloc_limit
    end

    @testset "leading form $lead, $nargs arrays" for lead in 1:4, nargs in 1:5
        arrays = ntuple(i -> forms[mod1(lead + i - 1, 4)], nargs)
        l = arrays[1:cld(nargs, 2)]
        r = arrays[cld(nargs, 2)+1:end]
        for args in ((s, l..., r...), (l..., s, r...), (l..., r..., s),
                     (s, l..., t, r...), (s, l..., r..., t), (l..., s, r..., t),
                     (s, t, l..., r...), (l..., s, t, r...), (l..., r..., s, t),
                     (s, l..., t, r..., u), (s, l..., t, u, r...),
                     (l..., s, t, r..., u), (l..., s, t, u, r...))
            for f in (*, ordered)
                check_scalar_broadcast(f, args)
            end
        end
    end
end

end # module
