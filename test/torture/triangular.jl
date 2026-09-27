# This file is a part of Julia. License is MIT: https://julialang.org/license

# Exhaustive grids and long-tail regression tests for triangular sparse products and
# solves; they run only when the `torture` suite is selected.

module TortureTriangularTests
using Test
using SparseArrays
using LinearAlgebra
using Random
include("../testhelpers.jl")

# The eltypes of the triangular sparse times dense product that the core suite does not
# run.
@testset "multiplication of triangular sparse and dense matrices" begin
    n = 7
    B = rand(n, 3)
    _triangular_sparse_matrix(n, ULT, T) = T == Int ? ULT(sparse(rand(0:10, n, n))) : ULT(sprandn(T, n, n, 0.4))
    for T in (Float16, Float32, ComplexF16, ComplexF64)
        for AT in (adjoint, transpose)
            for TR in (UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular)
                TS = AT(_triangular_sparse_matrix(n, TR, T))
                @test isa(TS * B, DenseMatrix)
                @test TS * B ≈ Matrix(TS)*B
            end
        end
    end
end

# The full eltype product of the triangular times sparse vector grid; the core suite
# runs the diagonal pairs and one mixed pair.
@testset "Triangular and SparseVector multiplications, all eltype pairs" begin
    n = 10
    types = (Int, Float64, ComplexF64)
    tritypes = (LowerTriangular, UnitUpperTriangular)
    for ta in types
        for tri in tritypes
            if ta == Int
                T = tri(rand(1:9, n, n))
            else
                T = tri(randn(ta, n, n))
            end
            for tb in types
                if tb == Int
                    x = sparse(rand(0:4, n))
                else
                    x = sprandn(tb, n, 0.6)
                end
                @test T * x ≈ Array(T) * Array(x)
                @test T' * x ≈ Array(T)' * Array(x)
                @test transpose(T) * x ≈ transpose(Array(T)) * Array(x)
                @test x' * T ≈ Array(x)' * Array(T)
                @test x' * T' ≈ Array(x)' * Array(T)'
                @test x' * transpose(T) ≈ Array(x)' * transpose(Array(T))
            end
        end
    end

    # 0-dimensional case
    x = sparse(zeros(0))
    for tri in tritypes
        T = tri(zeros(0, 0))
        @test T*x == Array(T) * Array(x)
        @test T' * x == Array(T)' * Array(x)
        @test transpose(T) * x == transpose(Array(T)) * Array(x)
        @test x' * T == Array(x)' * Array(T)
        @test x' * T' == Array(x)' * Array(T)'
        @test x' * transpose(T) == Array(x)' * transpose(Array(T))
    end
end

@testset "issue #13792, use sparse triangular solvers for sparse triangular solves" begin
    local A, n, x
    n = 100
    A, b = sprandn(n, n, 0.5) + sqrt(n)*I, fill(1., n)
    @test LowerTriangular(A)\(LowerTriangular(A)*b) ≈ b
    @test UpperTriangular(A)\(UpperTriangular(A)*b) ≈ b
    A[2,2] = 0
    dropzeros!(A)
    @test_throws LinearAlgebra.SingularException LowerTriangular(A)\b
    @test_throws LinearAlgebra.SingularException UpperTriangular(A)\b
end

# Five random draws; the core suite runs one.
@testset "complex matrix-vector multiplication and triangular or diagonal left-division" begin
    for i = 1:5
        a = I + 0.1*sprandn(5, 5, 0.2)
        b = randn(5,3) + im*randn(5,3)
        c = randn(5) + im*randn(5)
        d = randn(5) + im*randn(5)
        α = rand(ComplexF64)
        β = rand(ComplexF64)
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(mul!(similar(b), a, b) - Array(a)*b)) < 100*eps()) # for compatibility with present matmul API. Should go away eventually.
        @test (maximum(abs.(mul!(similar(c), a, c) - Array(a)*c)) < 100*eps()) # for compatibility with present matmul API. Should go away eventually.
        @test (maximum(abs.(mul!(similar(b), transpose(a), b) - transpose(Array(a))*b)) < 100*eps()) # for compatibility with present matmul API. Should go away eventually.
        @test (maximum(abs.(mul!(similar(c), transpose(a), c) - transpose(Array(a))*c)) < 100*eps()) # for compatibility with present matmul API. Should go away eventually.
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.((a'*c + d) - (Array(a)'*c + d))) < 1000*eps())
        @test (maximum(abs.((α*transpose(a)*c + β*d) - (α*transpose(Array(a))*c + β*d))) < 1000*eps())
        @test (maximum(abs.((transpose(a)*c + d) - (transpose(Array(a))*c + d))) < 1000*eps())
        c = randn(6) + im*randn(6)
        @test_throws DimensionMismatch α*transpose(a)*c + β*c
        @test_throws DimensionMismatch α*transpose(a)*fill(1.,5) + β*c

        a = I + 0.1*sprandn(5, 5, 0.2) + 0.1*im*sprandn(5, 5, 0.2)
        b = randn(5,3)
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())

        a = I + tril(0.1*sprandn(5, 5, 0.2))
        b = randn(5,3) + im*randn(5,3)
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        a = I + tril(0.1*sprandn(5, 5, 0.2) + 0.1*im*sprandn(5, 5, 0.2))
        b = randn(5,3)
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        a = I + triu(0.1*sprandn(5, 5, 0.2))
        b = randn(5,3) + im*randn(5,3)
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        a = I + triu(0.1*sprandn(5, 5, 0.2) + 0.1*im*sprandn(5, 5, 0.2))
        b = randn(5,3)
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())
        # UpperTriangular/LowerTriangular solve
        a = UpperTriangular(I + triu(0.1*sprandn(5, 5, 0.2)))
        b = sprandn(5, 5, 0.2)
        @test (maximum(abs.(a\b - Array(a)\Array(b))) < 1000*eps())
        # test error throwing for bwdTrisolve
        @test_throws DimensionMismatch a\Matrix{Float64}(I, 6, 6)
        a = LowerTriangular(I + tril(0.1*sprandn(5, 5, 0.2)))
        b = sprandn(5, 5, 0.2)
        @test (maximum(abs.(a\b - Array(a)\Array(b))) < 1000*eps())
        # test error throwing for fwdTrisolve
        @test_throws DimensionMismatch a\Matrix{Float64}(I, 6, 6)

        a = sparse(Diagonal(randn(5) + im*randn(5)))
        b = randn(5,3)
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        b = randn(5,3) + im*randn(5,3)
        @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(a'b - Array(a)'b)) < 100*eps())
        @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())
    end
end

@testset "#16716" begin
    origmat = [-1.5 -0.7; 0.0 1.0]
    transmat = copy(origmat')
    utmat = UpperTriangular(origmat)
    ltmat = LowerTriangular(transmat)
    uutmat = LinearAlgebra.UnitUpperTriangular(origmat)
    ultmat = LinearAlgebra.UnitLowerTriangular(transmat)

    zerospvec = spzeros(Float64, 2)
    zerodvec = zeros(Float64, 2)

    for mat in (utmat, ltmat, uutmat, ultmat)
        @test isequal(\(mat, zerospvec), zerodvec)
        @test isequal(\(adjoint(mat), zerospvec), zerodvec)
        @test isequal(\(transpose(mat), zerospvec), zerodvec)
        @test isequal(ldiv!(mat, copy(zerospvec)), zerospvec)
        @test isequal(ldiv!(adjoint(mat), copy(zerospvec)), zerospvec)
        @test isequal(ldiv!(transpose(mat), copy(zerospvec)), zerospvec)
    end
end

end # module
