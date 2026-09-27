# This file is a part of Julia. License is MIT: https://julialang.org/license

# The full eltype and index-type grids of the UMFPACK wrapper tests; they run only when
# the `torture` suite is selected.

module TortureUMFPACKTests
using Test
using Random
using SparseArrays
using Serialization
using LinearAlgebra:
    LinearAlgebra, I, det, diag, issuccess, ldiv!, lu, lu!, Transpose, SingularException, Diagonal, logabsdet
using SparseArrays: nnz, sparse, sprandn, SparseMatrixCSC, UMFPACK, increment!
include("../../testhelpers.jl")

function umfpack_report(l::UMFPACK.UmfpackLU)
    UMFPACK.umfpack_report_numeric(l, 0)
    UMFPACK.umfpack_report_symbolic(l, 0)
    return
end

_isnull_numeric(F::UMFPACK.UmfpackLU) = F.numeric.p == C_NULL

const TransposeFact = isdefined(LinearAlgebra, :TransposeFactorization) ?
    LinearAlgebra.TransposeFactorization :
    Transpose

# based on deps/Suitesparse-4.0.2/UMFPACK/Demo/umfpack_di_demo.c
A0 = sparse(increment!([0,4,1,1,2,2,0,1,2,3,4,4]),
            increment!([0,4,0,2,1,2,1,4,3,2,1,2]),
            [2.,1.,3.,4.,-1.,-3.,3.,6.,2.,1.,4.,2.], 5, 5)

# The half-precision and single-precision complex eltypes; the core suite keeps Float64,
# ComplexF64 and Float32.
@testset "Do/do not reuse symbolic LU factorization" for reuse ∈ (true, false)
    A1 = sparse(increment!([0,4,1,1,2,2,0,1,2,3,4,4]),
                increment!([0,4,0,2,1,2,1,4,3,2,1,2]),
                [2.,1.,3.,4.,-1.,-3.,3.,9.,2.,1.,4.,2.], 5, 5)
    testtypes = [ComplexF32, Float16, ComplexF16]
    for Tv in testtypes
        for Ti in Base.uniontypes(UMFPACK.UMFITypes)
            A = convert(SparseMatrixCSC{Tv,Ti}, A0)
            B = convert(SparseMatrixCSC{Tv,Ti}, A1)
            b = Tv[8., 45., -3., 3., 19.]
            F = lu(A)
            umfpack_report(F)
            lu!(F, B; reuse_symbolic=reuse)
            umfpack_report(F)
            @test F\b ≈ B\b ≈ Matrix(B)\b

            # singular matrix
            C = copy(B)
            C[4, 3] = Tv(0)
            F = lu(A)
            umfpack_report(F)
            @test_throws SingularException lu!(F, C; reuse_symbolic=reuse)
            # change of nonzero pattern
            D = copy(B)
            D[5, 1] = Tv(1.0)
            F = lu(A)
            umfpack_report(F)
            if reuse
                @test_throws ArgumentError lu!(F, D; reuse_symbolic=reuse)
                # the stale numeric factorization of A has been dropped, so
                # anything needing it refactors D against A's symbolic and fails again
                @test_throws ArgumentError umfpack_report(F)
                @test_throws ArgumentError F\b
            else
                lu!(F, D; reuse_symbolic=reuse)
                umfpack_report(F)
                @test F\b ≈ D\b ≈ Matrix(D)\b
            end
        end
    end
end

@testset "rcond (#118) for $Tv, $Ti" for Tv in (Float64, ComplexF64), Ti in Base.uniontypes(UMFPACK.UMFITypes)
    # the number is min/max of |diag(U)| of the row-scaled matrix UMFPACK factorized
    F = lu(SparseMatrixCSC{Tv,Ti}(sparse(Tv[1 3; 0 1])))
    @test UMFPACK.rcond(F) === 0.25
    @test UMFPACK.rcond(F) === minimum(abs, diag(F.U)) / maximum(abs, diag(F.U))
    # row scaling is on by default, so a diagonal matrix is perfectly conditioned
    @test UMFPACK.rcond(lu(SparseMatrixCSC{Tv,Ti}(sparse(Diagonal(Tv[1, 2, 4]))))) === 1.0
    # 1-by-1 and singular special cases
    @test UMFPACK.rcond(lu(SparseMatrixCSC{Tv,Ti}(sparse(Diagonal(Tv[3]))))) === 1.0
    @test UMFPACK.rcond(lu(SparseMatrixCSC{Tv,Ti}(sparse(Tv[1 2; 0 0])); check=false)) === 0.0
    # a factor without a numeric decomposition gets one on demand
    G = UMFPACK.UmfpackLU(SparseMatrixCSC{Tv,Ti}(sparse(Tv[1 3; 0 1])))
    @test UMFPACK.rcond(G) === 0.25
    # lu! refreshes the estimate
    lu!(F, SparseMatrixCSC{Tv,Ti}(sparse(Tv[1 1; 0 1])))
    @test UMFPACK.rcond(F) === 0.5
end

@testset "F.Rs and logabsdet when UMFPACK divides by the scale factors, $Tv, $Ti" for
        Tv in (Float64, ComplexF64), Ti in Base.uniontypes(UMFPACK.UMFITypes)
    # UMFPACK stores reciprocal scale factors for badly scaled rows
    A = SparseMatrixCSC{Tv,Ti}(sparse(Tv[1e-20 2e-20 0; 0 1 3; 1 0 1]))
    F = lu(A)
    @test F.L * F.U ≈ (F.Rs .* A)[F.p, F.q]
    L, U, p, q, Rs = F.:(:)
    @test Rs == F.Rs
    @test all(logabsdet(F) .≈ logabsdet(Matrix(A)))
    @test det(F) ≈ det(Matrix(A))
    B = SparseMatrixCSC{Tv,Ti}(1e-15 * (sprandn(MersenneTwister(1), 50, 50, 0.1) + 10I))
    @test all(logabsdet(lu(B)) .≈ logabsdet(Matrix(B)))
end

@testset "factors are rebuilt on demand, $Tv, $Ti" for
        Tv in (Float64, ComplexF64), Ti in Base.uniontypes(UMFPACK.UMFITypes)
    A = SparseMatrixCSC{Tv,Ti}(sparse(Tv[4 1; 1 3]))
    for G in (UMFPACK.UmfpackLU(A), deserialize(seekstart(let io = IOBuffer(); serialize(io, lu(A)); io; end)))
        @test det(G) ≈ det(Matrix(A))
        @test nnz(G) == nnz(lu(A))
    end
    # a failed factorization stays failed across serialization
    S = SparseMatrixCSC{Tv,Ti}(sparse(Tv[1 2; 2 4]))
    io = IOBuffer(); serialize(io, lu(S; check=false)); seekstart(io)
    G = deserialize(io)
    @test !issuccess(G)
    @test occursin("Failed factorization", sprint(show, MIME"text/plain"(), G))
end

@testset "failed lu! drops the old numeric factorization, $Tv, $Ti" for
        Tv in (Float64, ComplexF64), Ti in Base.uniontypes(UMFPACK.UMFITypes)
    A = SparseMatrixCSC{Tv,Ti}(sparse(Tv[4 1 0; 1 4 1; 0 1 4]))
    B = SparseMatrixCSC{Tv,Ti}(sparse(Tv[5 1 0; 1 5 1; 0 1 5]))
    F = lu(A)
    @test_throws ArgumentError lu!(F, B; reuse_symbolic=false, q=[1, 1, 2])
    @test !issuccess(F)
    @test _isnull_numeric(F)
    # anything needing the factors refactorizes the matrix now held, B
    @test all(logabsdet(F) .≈ logabsdet(Matrix(B)))
    @test F \ ones(3) ≈ Matrix(B) \ ones(3)
end

@testset "ldiv! with strided and adjoint/transpose right-hand sides, $Tv, $Ti" for
        Tv in (Float64, ComplexF64), Ti in Base.uniontypes(UMFPACK.UMFITypes)
    A = SparseMatrixCSC{Tv,Ti}(sparse(Tv[4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4.5]))
    F = lu(A)
    Ad = Matrix(A)
    w = Tv.(collect(1.0:8.0))
    v = view(w, 1:2:8)
    @test ldiv!(F, v) ≈ Ad \ Tv.(1:2:8)
    @test w[2:2:8] == 2:2:8
    @test ldiv!(zeros(Tv, 4), F, view(Tv.(collect(1.0:8.0)), 1:2:8)) ≈ Ad \ Tv.(1:2:8)
    M = Tv.(reshape(1.0:24.0, 8, 3))
    Y = zeros(Tv, 8, 3)
    ldiv!(view(Y, 1:2:8, :), transpose(F), view(M, 2:2:8, :))
    @test Y[1:2:8, :] ≈ transpose(Ad) \ M[2:2:8, :]
    @test iszero(Y[2:2:8, :])
    for op in (adjoint, transpose), G in (F, F', transpose(F))
        B = Tv.(reshape(1.0:12.0, 3, 4))
        Bw = op(copy(B))
        @test ldiv!(G, Bw) === Bw
        @test Bw ≈ (G === F ? Ad : G isa TransposeFact ? transpose(Ad) : Ad') \ op(B)
    end
end

# Ten random right-hand sides; the core suite runs one.
@testset "UMFPACK's lu with custom permutation" begin
    A = sparse([1.0 0.0 0.9778920565882165 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 1.0 0.0 0.0 0.0 1.847311282254734 0.0 0.0 0.0 0.0;
    0.0 0.0 1.0 0.0 0.0 0.04863647201402087 0.0 0.0 0.0 -1.1593207405039443;
    0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.5145863988424498;
    0.0421803353935357 0.0 -1.2818900361848549 0.0 1.0 0.0 0.1116124255865398 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.5457237331767308;
    -0.4983003278517826 -0.9974658316950679 1.0734689365455168 -1.0511956770913033 0.0 -0.37409855916460416 1.999357231970987 0.0 0.0 -0.9620788056415616;
    -1.5784683379261246 0.0 0.0 0.0 -0.4147349268116999 0.0 0.8539293641597945 1.0 0.0 0.0;
    0.0 0.0 0.0 0.0 -0.039051958043171624 0.0 0.0 -0.3814599389272203 1.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0])
    q1 = [9, 8, 5, 1, 7, 2, 3, 4, 6, 10]
    q0 = q1 .- 1
    for i in 1:10
        b = randn(10)
        x = lu(A) \ b
        x0 = lu(A; q=q0) \ b
        x1 = lu(A; q=q1) \ b
        @test x ≈ x0
        @test x ≈ x1
    end
end

end # module
