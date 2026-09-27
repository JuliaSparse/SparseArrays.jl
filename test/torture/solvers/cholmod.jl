# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests and the remaining grid cases of the CHOLMOD wrappers; they
# run only when the `torture` suite is selected.

module TortureCHOLMODTests
using Test
using SparseArrays
using LinearAlgebra: I, cholesky, ldlt, ldlt!, issuccess, Symmetric, Hermitian
include("../../testhelpers.jl")

Tv = Float64

@testset "Issue 11745 - row and column pointers were not sorted in sparse(Factor)" begin
    A = Tv[10 1 1 1; 1 10 0 0; 1 0 10 0; 1 0 0 10]
    @test sparse(cholesky(sparse(A))) ≈ A
end

@testset "Issue 29367" begin
    if Int != Int32
        @test_nowarn cholesky(sparse(Int32[1,2,3,4], Int32[1,2,3,4], Tv[1,4,16,64]))
        @test_nowarn ldlt(sparse(Int32[1,2,3,4], Int32[1,2,3,4], Tv[1,4,16,64]))
    end
end

@testset "Issue #22335" begin
    local A, F
    A = sparse(1.0I, 3, 3)
    @test issuccess(cholesky(A))
    A[3, 3] = -1
    F = cholesky(A; check = false)
    @test !issuccess(F)
    @test issuccess(ldlt!(F, A))
    A[3, 3] = 1
    @test A[:, 3:-1:1]\fill(1., 3) == [1, 1, 1]
end

# The mixed real/complex pairs; the core suite keeps the matching pairs.
@testset "Issues #27860 & #28363" begin
    for typeA in (Tv, Complex{Tv}), typeB in (Tv, Complex{Tv}), transform in (identity, adjoint, transpose)
        typeA == typeB && continue
        A = sparse(typeA[2.0 0.1; 0.1 2.0])
        B = randn(typeB, 2, 2)
        @test A \ transform(B) ≈ cholesky(A) \ transform(B) ≈ Matrix(A) \ transform(B)
        C = randn(typeA, 2, 2)
        sC = sparse(C)
        sF = typeA <: Real ? cholesky(Symmetric(A)) : cholesky(Hermitian(A))
        @test cholesky(A) \ transform(sC) ≈ Matrix(A) \ transform(C)
        @test sF.PtL \ transform(A) ≈ sF.PtL \ Matrix(transform(A))
    end
end

end # module
