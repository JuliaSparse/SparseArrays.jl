# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseLinalgSolversTests
using Test

using SparseArrays
using Random
using LinearAlgebra

@testset "explicit zeros" begin
    a = SparseMatrixCSC(2, 2, [1, 3, 5], [1, 2, 1, 2], [1.0, 0.0, 0.0, 1.0])
    @test lu(a)\[2.0, 3.0] ≈ [2.0, 3.0]
    @test cholesky(a)\[2.0, 3.0] ≈ [2.0, 3.0]
end

@testset "complex left-division" begin
    for i = 1:5
        a = I + 0.1*sprandn(5, 5, 0.2)
        b = randn(5,3) + im*randn(5,3)
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        a = I + 0.1*sprandn(5, 5, 0.2) + 0.1*im*sprandn(5, 5, 0.2)
        b = randn(5,3)
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())
    end
end

@testset "sparse matrix cond" begin
    Random.seed!(1235)
    local A = sparse(reshape([1.0], 1, 1))
    @test cond(A, 1) == 1.0
    @test_throws ArgumentError cond(A,2)
    @test_throws ArgumentError cond(A,3)
    Arect = spzeros(10, 6)
    @test_throws DimensionMismatch cond(Arect, 1)
    @test_throws ArgumentError cond(Arect,2)
    @test_throws DimensionMismatch cond(Arect, Inf)
    Ac = sprandn(20, 20,.5) + im*sprandn(20, 20,.5)
    Ar = sprandn(20, 20,.5) + eps()*I
    # For a discussion of the tolerance, see #14778
    @test 0.99 <= cond(Ar, 1) \ opnorm(Ar, 1) * opnorm(inv(Array(Ar)), 1) < 3
    @test 0.99 <= cond(Ac, 1) \ opnorm(Ac, 1) * opnorm(inv(Array(Ac)), 1) < 3
    @test 0.99 <= cond(Ar, Inf) \ opnorm(Ar, Inf) * opnorm(inv(Array(Ar)), Inf) < 3
    @test 0.99 <= cond(Ac, Inf) \ opnorm(Ac, Inf) * opnorm(inv(Array(Ac)), Inf) < 3
    #issue 680
    A22 = sparse(randn(2,2))
    @test 0.99 ≤ cond(Array(A22), 1) / cond(A22, 1) < 3
    @test 0.99 ≤ cond(Array(A22), Inf) / cond(A22, Inf) < 3
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
    #issue 680
    A33 = sparse(randn(3,3))
    @test SparseArrays.opnormestinv(A33,3) ≈ opnorm(inv(Array(A33)),1) atol=1e-4
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
    C, b = A[:, 1:4], fill(1., size(A, 1))
    @test factorize(C)\b ≈ Array(C)\b
end

@testset "type stability of linear solve" begin
    for relty in (Float16, Float32, Float64), elty in (relty, Complex{relty})
        A = sprand(elty, 2, 2, 1.0)
        B = randn(elty, 2, 2)
        b = randn(elty, 2)
        @inferred A \ b
        @inferred A \ B
    end
end

@testset "integer Hermitian solve uses the sparse solvers" begin
    # `\` and `factorize` used to send a Hermitian integer matrix to LinearAlgebra's
    # generic `factorize`, a dense Bunch-Kaufman in `Rational{BigInt}`
    for (A, T) in ((sparse([4 1 0; 1 4 1; 0 1 4]), Float64),
                   (sparse(Complex{Int}[4 1+im 0; 1-im 4 1+im; 0 1-im 4]), ComplexF64))
        elty = eltype(A)
        @test ishermitian(A)
        @test factorize(A) isa SparseArrays.UMFPACK.UmfpackLU{T}
        b, B = elty[1, 2, 3], elty[1 2; 3 4; 5 6]
        for M in (A, A', transpose(A))
            for rhs in (b, B)
                x = @inferred M \ rhs
                @test x isa Array{T}
                @test x ≈ Matrix(M) \ rhs
            end
            for rhs in (sparsevec(b), sparse(B))
                x = M \ rhs
                @test x isa Array{T}
                @test x ≈ Matrix(M) \ rhs
            end
        end
    end
    # `Rational` is not rerouted: it stays exact, like dense
    A = sparse(Rational{Int}[2 1; 1 2])
    b = Rational{Int}[1, 0]
    for M in (A, A', transpose(A))
        x = M \ b
        @test x isa Vector{Rational{Int}}
        @test M * x == b
        @test x == Matrix(M) \ b
    end
end

@testset "factorization of a fixed-pattern matrix" begin
    b = sprandn(10, 10, 0.99) + I
    a = SparseArrays.fixed(b)

    @test (lu(a) \ randn(10); true)
    @test b == a
    @test (qr(a + a') \ randn(10); true)
    @test b == a
end


@testset "ldiv! with and without a workspace, $name" for (name, fact, M) in (
        ("lu", lu, sparse([4.0 1 0; 1 4 1; 0 1 4] + im * [0 1 0; 0 0 1; 1 0 0])),
        ("cholesky", cholesky, sparse(ComplexF64[4 1+im 0; 1-im 4 1; 0 1 4])),
        ("qr", qr, sparse([4.0 1 0; 1 4 1; 0 1 4] + im * [0 1 0; 0 0 1; 1 0 0])))
    F = fact(M)
    b = ComplexF64[1, 2, 3]
    ws = fact === lu ? SparseArrays.UMFPACK.UmfpackWS(F) :
         fact === cholesky ? SparseArrays.CHOLMOD.CholmodWS(F) : SparseArrays.SPQR.SpqrWS(F)
    for (G, D) in ((F, Matrix(M)), (F', Matrix(M)'), (transpose(F), transpose(Matrix(M))))
        x = D \ b
        @test ldiv!(similar(b), G, b) ≈ x
        @test ldiv!(similar(b), G, b; workspace = ws) ≈ x
        @test ldiv!(G, copy(b)) ≈ x
        @test ldiv!(G, copy(b); workspace = ws) ≈ x
        @test ldiv!(similar([b b]), G, [b b]; workspace = ws) ≈ [x x]
    end
end

end # module
