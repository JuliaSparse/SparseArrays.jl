# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseLinalgSolversTests
using Test

@static if !Base.USE_GPL_LIBS
    @info "This Julia build excludes the use of SuiteSparse GPL libraries. Skipping SparseLinalgSolvers Tests"
else

using SparseArrays
using Random
using LinearAlgebra

@testset "explicit zeros" begin
    a = SparseMatrixCSC(2, 2, [1, 3, 5], [1, 2, 1, 2], [1.0, 0.0, 0.0, 1.0])
    @test lu(a)\[2.0, 3.0] ≈ [2.0, 3.0]
    @test cholesky(a)\[2.0, 3.0] ≈ [2.0, 3.0]
end

@testset "complex matrix-vector multiplication and left-division" begin
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
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())
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
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

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

        a = I + triu(0.1*sprandn(5, 5, 0.2))
        b = randn(5,3) + im*randn(5,3)
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

@testset "sparse matrix cond" begin
    local A = sparse(reshape([1.0], 1, 1))
    Ac = sprandn(20, 20,.5) + im*sprandn(20, 20,.5)
    Ar = sprandn(20, 20,.5) + eps()*I
    @test cond(A, 1) == 1.0
    # For a discussion of the tolerance, see #14778
    @test 0.99 <= cond(Ar, 1) \ opnorm(Ar, 1) * opnorm(inv(Array(Ar)), 1) < 3
    @test 0.99 <= cond(Ac, 1) \ opnorm(Ac, 1) * opnorm(inv(Array(Ac)), 1) < 3
    @test 0.99 <= cond(Ar, Inf) \ opnorm(Ar, Inf) * opnorm(inv(Array(Ar)), Inf) < 3
    @test 0.99 <= cond(Ac, Inf) \ opnorm(Ac, Inf) * opnorm(inv(Array(Ac)), Inf) < 3
    @test_throws ArgumentError cond(A,2)
    @test_throws ArgumentError cond(A,3)
    Arect = spzeros(10, 6)
    @test_throws DimensionMismatch cond(Arect, 1)
    @test_throws ArgumentError cond(Arect,2)
    @test_throws DimensionMismatch cond(Arect, Inf)
end

@testset "sparse matrix opnormestinv" begin
    Random.seed!(1235)
    Ac = sprandn(20,20,.5) + im* sprandn(20,20,.5)
    Aci = ceil.(Int64, 100*sprand(20,20,.5)) + im*ceil.(Int64, sprand(20,20,.5))
    Ar = sprandn(20,20,.5)
    Ari = ceil.(Int64, 100*Ar)
    # NOTE: opnormestinv is probabilistic, so requires a fixed seed (set above in Random.seed!(1234))
    @test SparseArrays.opnormestinv(Ac,3) ≈ opnorm(inv(Array(Ac)),1) atol=1e-4
    @test SparseArrays.opnormestinv(Aci,3) ≈ opnorm(inv(Array(Aci)),1) atol=1e-4
    @test SparseArrays.opnormestinv(Ar) ≈ opnorm(inv(Array(Ar)),1) atol=1e-4
    @test_throws ArgumentError SparseArrays.opnormestinv(Ac,0)
    @test_throws ArgumentError SparseArrays.opnormestinv(Ac,21)
    @test_throws DimensionMismatch SparseArrays.opnormestinv(sprand(3,5,.9))
end

@testset "factorization" begin
    Random.seed!(123)
    local A
    A = sparse(Diagonal(rand(5))) + sprandn(5, 5, 0.2) + im*sprandn(5, 5, 0.2)
    A = A + copy(A')
    @test abs(det(factorize(Hermitian(A)))) ≈ abs(det(factorize(Array(A))))
    A = sparse(Diagonal(rand(5))) + sprandn(5, 5, 0.2) + im*sprandn(5, 5, 0.2)
    A = A*A'
    @test abs(det(factorize(Hermitian(A)))) ≈ abs(det(factorize(Array(A))))
    A = sparse(Diagonal(rand(5))) + sprandn(5, 5, 0.2)
    A = A + copy(transpose(A))
    @test abs(det(factorize(Symmetric(A)))) ≈ abs(det(factorize(Array(A))))
    A = sparse(Diagonal(rand(5))) + sprandn(5, 5, 0.2)
    A = A*transpose(A)
    @test abs(det(factorize(Symmetric(A)))) ≈ abs(det(factorize(Array(A))))
    @test factorize(triu(A)) == triu(A)
    @test isa(factorize(triu(A)), UpperTriangular{Float64, SparseMatrixCSC{Float64, Int}})
    @test factorize(tril(A)) == tril(A)
    @test isa(factorize(tril(A)), LowerTriangular{Float64, SparseMatrixCSC{Float64, Int}})
    C, b = A[:, 1:4], fill(1., size(A, 1))
    @test factorize(C)\b ≈ Array(C)\b
    @test_throws ErrorException eigen(A)
    @test_throws ErrorException inv(A)
end

end # Base.USE_GPL_LIBS

end # module
