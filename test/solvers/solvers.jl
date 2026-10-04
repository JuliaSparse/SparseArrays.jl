# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseLinalgSolversTests
using Test

using SparseArrays
using Random
using LinearAlgebra
include("../testhelpers.jl")

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
        @static if COMPREHENSIVE
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())

        a = I + 0.1*sprandn(5, 5, 0.2) + 0.1*im*sprandn(5, 5, 0.2)
        b = randn(5,3)
        @test (maximum(abs.(a\b - Array(a)\b)) < 1000*eps())
        @test (maximum(abs.(a'\b - Array(a')\b)) < 1000*eps())
        @test (maximum(abs.(transpose(a)\b - Array(transpose(a))\b)) < 1000*eps())
        end
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
    @static if COMPREHENSIVE
    @test 0.99 <= cond(Ac, 1) \ opnorm(Ac, 1) * opnorm(inv(Array(Ac)), 1) < 3
    @test 0.99 <= cond(Ar, Inf) \ opnorm(Ar, Inf) * opnorm(inv(Array(Ar)), Inf) < 3
    end
    @test 0.99 <= cond(Ac, Inf) \ opnorm(Ac, Inf) * opnorm(inv(Array(Ac)), Inf) < 3
    @static if COMPREHENSIVE
    #issue 680
    A22 = sparse(randn(2,2))
    @test 0.99 ≤ cond(Array(A22), 1) / cond(A22, 1) < 3
    @test 0.99 ≤ cond(Array(A22), Inf) / cond(A22, Inf) < 3
    end
end

@testset "sparse matrix opnormestinv" begin
    Random.seed!(1235)
    Ac = sprandn(20,20,.5) + im* sprandn(20,20,.5)
    @static if COMPREHENSIVE
    Aci = ceil.(Int64, 100*sprand(20,20,.5)) + im*ceil.(Int64, sprand(20,20,.5))
    end
    Ar = sprandn(20,20,.5)
    # NOTE: opnormestinv is probabilistic, so requires a fixed seed (set above in Random.seed!(1234))
    @test SparseArrays.opnormestinv(Ac,3) ≈ opnorm(inv(Array(Ac)),1) atol=1e-4
    @static if COMPREHENSIVE
    @test SparseArrays.opnormestinv(Aci,3) ≈ opnorm(inv(Array(Aci)),1) atol=1e-4
    end
    @test SparseArrays.opnormestinv(Ar) ≈ opnorm(inv(Array(Ar)),1) atol=1e-4
    @test_throws ArgumentError SparseArrays.opnormestinv(Ac,0)
    @test_throws ArgumentError SparseArrays.opnormestinv(Ac,21)
    @test_throws DimensionMismatch SparseArrays.opnormestinv(sprand(3,5,.9))
    @static if COMPREHENSIVE
    #issue 680
    A33 = sparse(randn(3,3))
    @test SparseArrays.opnormestinv(A33,3) ≈ opnorm(inv(Array(A33)),1) atol=1e-4
    end
end

@static if COMPREHENSIVE
@testset "factorization" begin
    Random.seed!(123)
    local A
    @static if COMPREHENSIVE
    A = sparse(Diagonal(rand(5))) + sprandn(5, 5, 0.2) + im*sprandn(5, 5, 0.2)
    A = A + copy(A')
    @test abs(det(factorize(Hermitian(A)))) ≈ abs(det(factorize(Array(A))))
    end
    A = sparse(Diagonal(rand(5))) + sprandn(5, 5, 0.2) + im*sprandn(5, 5, 0.2)
    A = A*A'
    @test abs(det(factorize(Hermitian(A)))) ≈ abs(det(factorize(Array(A))))
    A = sparse(Diagonal(rand(5))) + sprandn(5, 5, 0.2)
    A = A + copy(transpose(A))
    @test abs(det(factorize(Symmetric(A)))) ≈ abs(det(factorize(Array(A))))
    @static if COMPREHENSIVE
    A = sparse(Diagonal(rand(5))) + sprandn(5, 5, 0.2)
    A = A*transpose(A)
    @test abs(det(factorize(Symmetric(A)))) ≈ abs(det(factorize(Array(A))))
    end
    C, b = A[:, 1:4], fill(1., size(A, 1))
    @test factorize(C)\b ≈ Array(C)\b
end
end

@testset "\\ and factorize choose the same method" begin
    square = sparse((@static COMPREHENSIVE ? Float32 : Float64)[4 1 0; 0 4 2; 1 0 4])
    herm = sparse([4.0 1 0; 1 4 1; 0 1 4])
    tall = sparse([2.0 0; 1 3; 0 1])
    wide = sparse([2.0 1 0; 0 3 1])
    b = [1.0, 2, 3]
    for (A, F) in ((square, SparseArrays.UMFPACK.UmfpackLU), (herm, SparseArrays.CHOLMOD.Factor),
                   (tall, SparseArrays.SPQR.QRSparse), (wide, SparseArrays.SPQR.AdjointQRSparse))
        @test factorize(A) isa F
        rhs = b[1:size(A, 1)]
        @test A \ rhs ≈ Matrix(A) \ rhs
    end
    # A' of a wide A is tall, so its least squares solve needs `qr(A')`
    @test wide' \ b ≈ Matrix(wide') \ b
    @static if COMPREHENSIVE
    # the adjoint solve converts its result to the eltype the plain solve gives
    x = square' \ Float32.(b)
    @test x isa Vector{Float32} && x ≈ Matrix(square') \ b
    # only the LU solve converts its result, so a least squares solve takes any right-hand side
    @test tall \ Any[1.0, 2, 3] ≈ Matrix(tall) \ b
    end
    # when the Cholesky factorization of a Hermitian matrix fails, `factorize` returns
    # the LDLt factorization and `\` falls back to `lu`
    H = sparse([1.0 2 0; 2 1 0; 0 0 -3])
    F = factorize(H)
    @test F isa SparseArrays.CHOLMOD.Factor && !isposdef(F)
    @test H \ b ≈ Matrix(H) \ b
end

@static if COMPREHENSIVE
@testset "type stability of linear solve" begin
    for (elty, vecrhs) in ((Float64, true), (ComplexF64, false), (@static COMPREHENSIVE ? ((Float64, false),
            (ComplexF64, true), eachvalue((Float16, ComplexF16, Float32, ComplexF32), (true, false))...) : ())...)
        A = sprand(elty, 2, 2, 1.0)
        B = randn(elty, 2, 2)
        b = randn(elty, 2)
        @inferred A \ (vecrhs ? b : B)
    end
end
end

@testset "integer Hermitian solve uses the sparse solvers" begin
    # `\` and `factorize` used to send a Hermitian integer matrix to LinearAlgebra's
    # generic `factorize`, a dense Bunch-Kaufman in `Rational{BigInt}`
    As = ((sparse([4 1 0; 1 4 1; 0 1 4]), Float64),
          (sparse(Complex{Int}[4 1+im 0; 1-im 4 1+im; 0 1-im 4]), ComplexF64))
    for (A, T) in (@static COMPREHENSIVE ? As : As[1:1])
        @test ishermitian(A)
        @test factorize(A) isa SparseArrays.UMFPACK.UmfpackLU{T}
    end
    for ((A, T), wrap, dense) in (@static COMPREHENSIVE ?
            pairwise(As, (identity, adjoint, transpose), (true, false)) : ((As[1], identity, true),))
        elty = eltype(A)
        b, B = elty[1, 2, 3], elty[1 2; 3 4; 5 6]
        for M in (wrap(A),)
            for rhs in (dense ? (@static COMPREHENSIVE ? (b, B) : (b,)) : ())
                x = @inferred M \ rhs
                @test x isa Array{T}
                @test x ≈ Matrix(M) \ rhs
            end
            for rhs in (dense ? () : (sparsevec(b), sparse(B)))
                x = M \ rhs
                @test x isa Array{T}
                @test x ≈ Matrix(M) \ rhs
            end
        end
    end
    @static if COMPREHENSIVE
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
end

@testset "LibSuiteSparse names are not imported into SparseArrays" begin
    @test isdefined(SparseArrays.LibSuiteSparse, :cholmod_l_start)
    @test isdefined(SparseArrays.LibSuiteSparse, :umfpack_dl_symbolic)
    @test isdefined(SparseArrays.LibSuiteSparse, :CHOLMOD_OK)
    @test !isdefined(SparseArrays, :cholmod_l_start)
    @test !isdefined(SparseArrays, :umfpack_dl_symbolic)
    @test !isdefined(SparseArrays, :CHOLMOD_OK)
end

@testset "factorization of a fixed-pattern matrix" begin
    b = sprandn(10, 10, 0.99) + I
    a = SparseArrays.fixed(b)

    @test (lu(a) \ randn(10); true)
    @test b == a
    @test (qr(a + a') \ randn(10); true)
    @test b == a

    # `factorize` and `\` query `ishermitian`, which used to throw on a fixed matrix
    F = SparseArrays.fixed(sparse([4.0 1 0; 1 4 1; 0 1 4]))
    @test factorize(F) isa SparseArrays.CHOLMOD.Factor{Float64}
    @test F \ [1.0, 2, 3] ≈ Matrix(F) \ [1.0, 2, 3]
    # an indefinite Hermitian fixed matrix reaches `ldlt!`, which takes a `SparseMatrixCSC` only
    @test_broken factorize(SparseArrays.fixed(sparse([1.0 2 0; 2 1 2; 0 2 1]))) isa SparseArrays.CHOLMOD.Factor
end


@testset "ldiv! with and without a workspace, $name" for (name, fact, M) in (
        ("lu", lu, sparse([4.0 1 0; 1 4 1; 0 1 4] + im * [0 1 0; 0 0 1; 1 0 0])),
        ("cholesky", cholesky, sparse(ComplexF64[4 1+im 0; 1-im 4 1; 0 1 4])),
        ("qr", qr, sparse([4.0 1 0; 1 4 1; 0 1 4] + im * [0 1 0; 0 0 1; 1 0 0])))
    F = fact(M)
    b = ComplexF64[1, 2, 3]
    ws = fact === lu ? SparseArrays.UMFPACK.UmfpackWS(F) :
         fact === cholesky ? SparseArrays.CHOLMOD.CholmodWS(F) : SparseArrays.SPQR.SpqrWS(F)
    # the adjoint of a Cholesky factorization solves the same system as the factorization,
    # so its transpose is the wrapper that takes a path of its own
    for wrap in (identity, (@static COMPREHENSIVE ? (adjoint, transpose) : fact === cholesky ? (transpose,) : (adjoint,))...)
        G, D = wrap(F), wrap(Matrix(M))
        x = D \ b
        @static if COMPREHENSIVE
        @test ldiv!(similar(b), G, b) ≈ x
        end
        @test ldiv!(similar(b), G, b; workspace = ws) ≈ x
        @static if COMPREHENSIVE
        @test ldiv!(G, copy(b)) ≈ x
        end
        @test ldiv!(G, copy(b); workspace = ws) ≈ x
        @static if COMPREHENSIVE
        @test ldiv!(similar([b b]), G, [b b]; workspace = ws) ≈ [x x]
        end
    end
end

end # module
