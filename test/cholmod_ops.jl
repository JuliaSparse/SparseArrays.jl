# This file is a part of Julia. License is MIT: https://julialang.org/license

module CHOLMODOpsTests
# Core Sparse/Dense/Factor operations, factor extraction and regression tests for
# CHOLMOD. Split off from cholmod.jl so the two halves run on separate test workers.
using Test

@static if !Base.USE_GPL_LIBS
    @info "This Julia build excludes the use of SuiteSparse GPL libraries. Skipping CHOLMOD tests"
else

using SparseArrays.CHOLMOD
using SparseArrays.CHOLMOD: getcommon
using Random
using Serialization
using LinearAlgebra:
    I, cholesky, cholesky!, det, diag, eigmax, ishermitian, isposdef, issuccess,
    issymmetric, ldiv!, ldlt, ldlt!, logdet, norm, opnorm, Diagonal, Hermitian, Symmetric,
    PosDefException, ZeroPivotException, RowMaximum
using SparseArrays
using SparseArrays: getcolptr
using SparseArrays.LibSuiteSparse
using SparseArrays.LibSuiteSparse: cholmod_l_allocate_sparse, cholmod_allocate_sparse

# CHOLMOD tests
itypes = sizeof(Int) == 4 ? (Int32,) : (Int32, Int64)
for Ti ∈ itypes, Tv ∈ (Float32, Float64)
Random.seed!(123)

@testset "Core functionality ($elty, $elty2)" for
    elty in (Tv, Complex{Tv}),
    Tv2 in (Float32, Float64),
    elty2 in (Tv2, Complex{Tv2}),
    Ti ∈ itypes
    A1 = sparse(Ti[1:5; 1], Ti[1:5; 2], elty <: Real ? randn(Tv, 6) : complex.(randn(Tv, 6), randn(Tv, 6)))
    A2 = sparse(Ti[1:5; 1], Ti[1:5; 2], elty2 <: Real ? randn(Tv2, 6) : complex.(randn(Tv2, 6), randn(Tv2, 6)))
    A1pd = A1'A1 + 10I
    A1pdSparse = CHOLMOD.Sparse(
        size(A1pd, 1),
        size(A1pd, 2),
        SparseArrays.decrement(getcolptr(A1pd)),
        SparseArrays.decrement(rowvals(A1pd)),
        nonzeros(A1pd))

    ## High level interface
    @test isa(CHOLMOD.Sparse(3, 3, Ti[0,1,3,4], Ti[0,2,1,2], fill(one(Tv), 4)), CHOLMOD.Sparse) # Sparse doesn't require columns to be sorted
    for i ∈ axes(A1, 1)
        A1[i, i] = real(A1[i, i])
    end #Construct Hermitian matrix properly
    A1Sparse = CHOLMOD.Sparse(A1)
    A2Sparse = CHOLMOD.Sparse(A2)
    @test_throws BoundsError A1Sparse[6, 1]
    @test_throws BoundsError A1Sparse[1, 6]
    @test sparse(A1Sparse) == A1
    @test CHOLMOD.sparse(CHOLMOD.Sparse(Hermitian(A1, :L))) == Hermitian(A1, :L)
    @test CHOLMOD.sparse(CHOLMOD.Sparse(Hermitian(A1, :U))) == Hermitian(A1, :U)
    @test_throws ArgumentError convert(SparseMatrixCSC{elty,Ti}, A1pdSparse)
    if elty <: Real
        @test_throws ArgumentError convert(Symmetric{Tv,SparseMatrixCSC{Tv,Ti}}, A1Sparse)
    else
        @test_throws ArgumentError convert(Hermitian{Complex{Tv},SparseMatrixCSC{Complex{Tv},Ti}}, A1Sparse)
    end
    @test copy(A1Sparse) == A1Sparse
    @test size(A1Sparse, 3) == 1
    if elty <: Real # multiplication only defined for real matrices in CHOLMOD
        @test A1Sparse*A2Sparse ≈ A1*A2
        @test_throws DimensionMismatch CHOLMOD.Sparse(A1[:,1:4])*A2Sparse
        @test A1Sparse'A2Sparse ≈ A1'A2
        @test A1Sparse*A2Sparse' ≈ A1*A2'

        @test A1Sparse*A1Sparse ≈ A1*A1
        @test A1Sparse'A1Sparse ≈ A1'A1
        @test A1Sparse*A1Sparse' ≈ A1*A1'

        @test A1pdSparse*A1pdSparse ≈ A1pd*A1pd
        @test A1pdSparse'A1pdSparse ≈ A1pd'A1pd
        @test A1pdSparse*A1pdSparse' ≈ A1pd*A1pd'

        @test_throws DimensionMismatch A1Sparse*CHOLMOD.eye(4, 5, elty)
    end

    # Factor
    @test_throws ArgumentError cholesky(A1)
    @test_throws ArgumentError cholesky(A1)
    @test_throws ArgumentError cholesky(A1, shift=1.0)
    @test_throws ArgumentError ldlt(A1)
    @test_throws ArgumentError ldlt(A1, shift=1.0)
    C = A1 + copy(adjoint(A1))
    λmaxC = eigmax(Array(C))
    b = fill(one(Tv), size(A1, 1))
    @test_throws PosDefException cholesky(C - 2λmaxC*I)
    @test_throws PosDefException cholesky(C, shift=-2λmaxC)
    @test_throws ZeroPivotException ldlt(C - C[1,1]*I)
    @test_throws ZeroPivotException ldlt(C, shift=-real(C[1,1]))
    @test !isposdef(cholesky(C - 2λmaxC*I; check = false))
    @test !isposdef(cholesky(C, shift=-2λmaxC; check = false))
    @test !issuccess(ldlt(C - C[1,1]*I; check = false))
    @test !issuccess(ldlt(C, shift=-real(C[1,1]); check = false))
    F = cholesky(A1pd)
    tmp = IOBuffer()
    show(tmp, F)
    @test tmp.size > 0
    @test isa(CHOLMOD.Sparse(F), CHOLMOD.Sparse{elty})
    @test_throws DimensionMismatch F\CHOLMOD.Dense(fill(elty(1), 4))
    @test_throws DimensionMismatch F\CHOLMOD.Sparse(sparse(fill(elty(1), 4)))
    b = ones(elty2, 5)
    bT = ones(elty, 5)
    @test F'\bT ≈ Array(A1pd)'\b
    @test F'\sparse(bT) ≈ Array(A1pd)'\b
    @test transpose(F)\bT ≈ conj(A1pd)'\bT
    @test F\CHOLMOD.Sparse(sparse(bT)) ≈ A1pd\b
    @test logdet(F) ≈ logdet(Array(A1pd))
    @test det(F) == exp(logdet(F))
    let # to test supernodal, we must use a larger matrix
        Ftmp = SparseMatrixCSC{Tv, Ti}(sprandn(Tv, 100, 100, 0.1))
        Ftmp = Ftmp'Ftmp + 10I
        @test logdet(cholesky(Ftmp)) ≈ logdet(Array(Ftmp))
    end
    @test logdet(ldlt(A1pd)) ≈ logdet(Array(A1pd))
    @test isposdef(A1pd)
    @test !isposdef(A1)
    @test !isposdef(A1 + copy(A1') |> t -> t - 2eigmax(Array(t))*I)

    if elty <: Real
        @test CHOLMOD.issymmetric(Sparse(A1pd, 0))
        @test CHOLMOD.Sparse(cholesky(Symmetric(A1pd, :L))) == CHOLMOD.Sparse(cholesky(A1pd))
        F1 = CHOLMOD.Sparse(cholesky(Symmetric(A1pd, :L), shift=2))
        F2 = CHOLMOD.Sparse(cholesky(A1pd, shift=2))
        @test F1 == F2
        @test CHOLMOD.Sparse(ldlt(Symmetric(A1pd, :L))) == CHOLMOD.Sparse(ldlt(A1pd))
        F1 = CHOLMOD.Sparse(ldlt(Symmetric(A1pd, :L), shift=2))
        F2 = CHOLMOD.Sparse(ldlt(A1pd, shift=2))
        @test F1 == F2
    else
        @test !CHOLMOD.issymmetric(Sparse(A1pd, 0))
        @test CHOLMOD.ishermitian(Sparse(A1pd, 0))
        @test CHOLMOD.Sparse(cholesky(Hermitian(A1pd, :L))) == CHOLMOD.Sparse(cholesky(A1pd))
        F1 = CHOLMOD.Sparse(cholesky(Hermitian(A1pd, :L), shift=2))
        F2 = CHOLMOD.Sparse(cholesky(A1pd, shift=2))
        @test F1 == F2
        @test CHOLMOD.Sparse(ldlt(Hermitian(A1pd, :L))) == CHOLMOD.Sparse(ldlt(A1pd))
        F1 = CHOLMOD.Sparse(ldlt(Hermitian(A1pd, :L), shift=2))
        F2 = CHOLMOD.Sparse(ldlt(A1pd, shift=2))
        @test F1 == F2
    end

    ### cholesky!/ldlt!
    F = cholesky(A1pd)
    CHOLMOD.change_factor!(F, false, false, true, true)
    @test unsafe_load(pointer(F)).is_ll == 0
    CHOLMOD.change_factor!(F, true, false, true, true)
    @test CHOLMOD.Sparse(cholesky!(copy(F), A1pd)) ≈ CHOLMOD.Sparse(F) # surprisingly, this can cause small ulp size changes so we cannot test exact equality
    @test size(F, 2) == 5
    @test size(F, 3) == 1
    @test_throws ArgumentError size(F, 0)

    F = cholesky(A1pdSparse, shift=2)
    @test isa(CHOLMOD.Sparse(F), CHOLMOD.Sparse{elty, Ti})
    @test CHOLMOD.Sparse(cholesky!(copy(F), A1pd, shift=2.0)) ≈ CHOLMOD.Sparse(F) # surprisingly, this can cause small ulp size changes so we cannot test exact equality

    F = ldlt(A1pd)
    @test isa(CHOLMOD.Sparse(F), CHOLMOD.Sparse{elty, Ti})
    @test CHOLMOD.Sparse(ldlt!(copy(F), A1pd)) ≈ CHOLMOD.Sparse(F) # surprisingly, this can cause small ulp size changes so we cannot test exact equality

    F = ldlt(A1pdSparse, shift=2)
    @test isa(CHOLMOD.Sparse(F), CHOLMOD.Sparse{elty, Ti})
    @test CHOLMOD.Sparse(ldlt!(copy(F), A1pd, shift=2.0)) ≈ CHOLMOD.Sparse(F) # surprisingly, this can cause small ulp size changes so we cannot test exact equality

    @test isa(CHOLMOD.factor_to_sparse!(F), CHOLMOD.Sparse)
    @test_throws CHOLMOD.CHOLMODException CHOLMOD.factor_to_sparse!(F)

    ## Low level interface
    @test CHOLMOD.nnz(A1Sparse) == nnz(A1)
    @test CHOLMOD.speye(5, 5, elty) == Matrix(I, 5, 5)
    @test CHOLMOD.spzeros(5, 5, 5, elty) == zeros(elty, 5, 5)
    if elty <: Real && elty2 <: Real
        @test CHOLMOD.copy(A1Sparse, 0, 1) == A1Sparse
        @test CHOLMOD.horzcat(A1Sparse, A2Sparse, true) == [A1 A2]
        @test CHOLMOD.vertcat(A1Sparse, A2Sparse, true) == [A1; A2]
        svec = fill(one(elty2), 1)
        @test CHOLMOD.scale!(CHOLMOD.Dense(svec), CHOLMOD_SCALAR, A1Sparse) == A1Sparse
        svec = fill(one(elty2), 5)
        @test_throws DimensionMismatch CHOLMOD.scale!(CHOLMOD.Dense(svec), CHOLMOD_SCALAR, A1Sparse)
        @test CHOLMOD.scale!(CHOLMOD.Dense(svec), CHOLMOD_ROW, A1Sparse) == A1Sparse
        @test_throws DimensionMismatch CHOLMOD.scale!(CHOLMOD.Dense([svec; 1]), CHOLMOD_ROW, A1Sparse)
        @test CHOLMOD.scale!(CHOLMOD.Dense(svec), CHOLMOD_COL, A1Sparse) == A1Sparse
        @test_throws DimensionMismatch CHOLMOD.scale!(CHOLMOD.Dense([svec; 1]), CHOLMOD_COL, A1Sparse)
        @test CHOLMOD.scale!(CHOLMOD.Dense(svec), CHOLMOD_SYM, A1Sparse) == A1Sparse
        @test_throws DimensionMismatch CHOLMOD.scale!(CHOLMOD.Dense([svec; 1]), CHOLMOD_SYM, A1Sparse)
        @test_throws DimensionMismatch CHOLMOD.scale!(CHOLMOD.Dense(svec), CHOLMOD_SYM, CHOLMOD.Sparse(A1[:,1:4]))
        @test CHOLMOD.aat(A1Sparse, [0:size(A1,2)-1;], 1) ≈ A1*A1'
        @test CHOLMOD.aat(A1Sparse, [0:1;], 1) ≈ A1[:,1:2]*A1[:,1:2]'
        @test CHOLMOD.copy(A1Sparse, 0, 1) == A1Sparse
    else
        # These operations are not well-supportd for Complex, as CHOLMOD assumes input is Hermitian.
        @test_throws MethodError CHOLMOD.horzcat(A1Sparse, A2Sparse, true) == [A1 A2]
        @test_throws MethodError CHOLMOD.vertcat(A1Sparse, A2Sparse, true) == [A1; A2]
    end
    @test CHOLMOD.ssmult(A1Sparse, A2Sparse, 0, true, true) ≈ A1*A2
    d = fill(one(elty2), 5)
    @test A1Sparse*d ≈ A1*d
    @test A1Sparse'*d ≈ A1'*d
    @test A2Sparse*A2Sparse' ≈ A2*A2'

    @test CHOLMOD.Sparse(CHOLMOD.Dense(A1Sparse)) == A1Sparse
end

@testset "extract factors" begin
    Af = Tv.([4 12 -16; 12 37 -43; -16 -43 98])
    As = sparse(Af)
    Lf = Tv.([2 0 0; 6 1 0; -8 5 3])
    LDf = Tv.([4 0 0; 3 1 0; -4 5 9])  # D is stored along the diagonal
    L_f = Tv.([1 0 0; 3 1 0; -4 5 1])  # L by itself in LDLt of Af
    D_f = Tv.([4 0 0; 0 1 0; 0 0 9])
    p = [2,3,1]
    p_inv = [3,1,2]

    @testset "cholesky, no permutation $Tv" begin
        Fs = cholesky(As, perm=[1:3;])
        @test sort(collect(propertynames(Fs))) == sort([:L, :U, :PtL, :UP, :p, :ptr])
        @test Fs.p == [1:3;]
        @test sparse(Fs.L) ≈ Lf
        @test sparse(Fs) ≈ As
        @test_throws CHOLMOD.CHOLMODException("sparse: supported only for :L on LLt factorizations") sparse(Fs.U)
        @test_throws CHOLMOD.CHOLMODException("sparse: supported only for :L on LLt factorizations") sparse(Fs.PtL)
        @test_throws CHOLMOD.CHOLMODException("sparse: supported only for :L on LLt factorizations") sparse(Fs.UP)
        b = rand(Tv, 3)
        bs = sparse(b)
        @test Fs\b ≈ Af\b ≈ (Fs\bs)::SparseVector
        @test Fs.UP\(Fs.PtL\b) ≈ Af\b
        @test Fs.L\b ≈ Lf\b ≈ (Fs.L\bs)::SparseVector
        @test Fs.U\b ≈ Lf'\b ≈ (Fs.U\bs)::SparseVector
        @test Fs.L'\b ≈ Lf'\b ≈ (Fs.L'\bs)::SparseVector
        @test Fs.U'\b ≈ Lf\b ≈ (Fs.U'\bs)::SparseVector
        @test Fs.PtL\b ≈ Lf\b ≈ (Fs.PtL\bs)::SparseVector
        @test Fs.UP\b ≈ Lf'\b ≈ (Fs.UP\bs)::SparseVector
        @test Fs.PtL'\b ≈ Lf'\b ≈ (Fs.PtL'\bs)::SparseVector
        @test Fs.UP'\b ≈ Lf\b ≈ (Fs.UP'\bs)::SparseVector
        @test_throws CHOLMOD.CHOLMODException Fs.D
        @test_throws CHOLMOD.CHOLMODException Fs.LD
        @test_throws CHOLMOD.CHOLMODException Fs.DU
        @test_throws CHOLMOD.CHOLMODException Fs.PLD
        @test_throws CHOLMOD.CHOLMODException Fs.DUPt
    end

    @testset "cholesky, with permutation" begin
        Fs = cholesky(As, perm=p)
        @test Fs.p == p
        Afp = Af[p,p]
        Lfp = cholesky(Afp).L
        Ls = sparse(Fs.L)
        @test Ls ≈ Lfp
        @test Ls * Ls' ≈ Afp
        P = sparse(1:3, Fs.p, ones(Tv, 3))
        @test P' * Ls * Ls' * P ≈ As
        @test sparse(Fs) ≈ As
        b = rand(Tv, 3)
        bs = sparse(b)
        @test Fs\b ≈ Af\b ≈ (Fs\bs)::SparseVector
        @test Fs.UP\(Fs.PtL\b) ≈ Af\b
        @test Fs.L\b ≈ Lfp\b ≈ (Fs.L\bs)::SparseVector
        @test Fs.U'\b ≈ Lfp\b ≈ (Fs.U'\bs)::SparseVector
        @test Fs.U\b ≈ Lfp'\b ≈ (Fs.U\bs)::SparseVector
        @test Fs.L'\b ≈ Lfp'\b ≈ (Fs.L'\bs)::SparseVector
        @test Fs.PtL\b ≈ Lfp\b[p] ≈ (Fs.PtL\bs)::SparseVector
        @test Fs.UP\b ≈ (Lfp'\b)[p_inv] ≈ (Fs.UP\bs)::SparseVector
        @test Fs.PtL'\b ≈ (Lfp'\b)[p_inv] ≈ (Fs.PtL'\bs)::SparseVector
        @test Fs.UP'\b ≈ Lfp\b[p] ≈ (Fs.UP'\bs)::SparseVector
        @test_throws CHOLMOD.CHOLMODException Fs.PL
        @test_throws CHOLMOD.CHOLMODException Fs.UPt
        @test_throws CHOLMOD.CHOLMODException Fs.D
        @test_throws CHOLMOD.CHOLMODException Fs.LD
        @test_throws CHOLMOD.CHOLMODException Fs.DU
        @test_throws CHOLMOD.CHOLMODException Fs.PLD
        @test_throws CHOLMOD.CHOLMODException Fs.DUPt
    end

    @testset "ldlt, no permutation" begin
        Fs = ldlt(As, perm=[1:3;])
        @test sort(collect(propertynames(Fs))) == sort([:L, :U, :PtL, :UP, :D, :LD, :DU, :PtLD, :DUP, :p, :ptr])
        @test Fs.p == [1:3;]
        @test sparse(Fs.LD) ≈ LDf
        @test sparse(Fs) ≈ As
        @test_throws CHOLMOD.CHOLMODException("sparse: supported only for :LD on LDLt factorizations") sparse(Fs.L)
        @test_throws CHOLMOD.CHOLMODException("sparse: supported only for :LD on LDLt factorizations") sparse(Fs.U)
        @test_throws CHOLMOD.CHOLMODException("sparse: supported only for :LD on LDLt factorizations") sparse(Fs.PtL)
        @test_throws CHOLMOD.CHOLMODException("sparse: supported only for :LD on LDLt factorizations") sparse(Fs.UP)
        @test_throws CHOLMOD.CHOLMODException("sparse: supported only for :LD on LDLt factorizations") sparse(Fs.D)
        @test_throws CHOLMOD.CHOLMODException("sparse: supported only for :LD on LDLt factorizations") sparse(Fs.DU)
        @test_throws CHOLMOD.CHOLMODException("sparse: supported only for :LD on LDLt factorizations") sparse(Fs.PtLD)
        @test_throws CHOLMOD.CHOLMODException("sparse: supported only for :LD on LDLt factorizations") sparse(Fs.DUP)
        b = rand(Tv, 3)
        bs = sparse(b)
        @test Fs\b ≈ Af\b ≈ (Fs\bs)::SparseVector
        @test Fs.UP\(Fs.PtLD\b) ≈ Af\b
        @test Fs.DUP\(Fs.PtL\b) ≈ Af\b
        @test Fs.L\b ≈ L_f\b ≈ (Fs.L\bs)::SparseVector
        @test Fs.U\b ≈ L_f'\b ≈ (Fs.U\bs)::SparseVector
        @test Fs.L'\b ≈ L_f'\b
        @test Fs.U'\b ≈ L_f\b
        @test Fs.PtL\b ≈ L_f\b ≈ (Fs.PtL\bs)::SparseVector
        @test Fs.UP\b ≈ L_f'\b
        @test Fs.PtL'\b ≈ L_f'\b
        @test Fs.UP'\b ≈ L_f\b
        @test Fs.D\b ≈ D_f\b
        @test Fs.D'\b ≈ D_f\b
        @test Fs.LD\b ≈ D_f\(L_f\b)
        @test Fs.DU'\b ≈ D_f\(L_f\b)
        @test Fs.LD'\b ≈ L_f'\(D_f\b)
        @test Fs.DU\b ≈ L_f'\(D_f\b)
        @test Fs.PtLD\b ≈ D_f\(L_f\b)
        @test Fs.DUP'\b ≈ D_f\(L_f\b)
        @test Fs.PtLD'\b ≈ L_f'\(D_f\b)
        @test Fs.DUP\b ≈ L_f'\(D_f\b)
    end

    @testset "ldlt, with permutation" begin
        Fs = ldlt(As, perm=p)
        @test Fs.p == p
        @test sparse(Fs) ≈ As
        b = rand(Tv, 3)
        bs = sparse(b)
        Asp = As[p,p]
        LDp = sparse(ldlt(Asp, perm=[1,2,3]).LD)
        # LDp = sparse(Fs.LD)
        Lp, dp = CHOLMOD.getLd!(copy(LDp))
        Dp = sparse(Diagonal(dp))
        @test Fs\b ≈ Af\b ≈ (Fs\bs)::SparseVector
        @test Fs.UP\(Fs.PtLD\b) ≈ Af\b
        @test Fs.DUP\(Fs.PtL\b) ≈ Af\b
        @test Fs.L\b ≈ Lp\b ≈ (Fs.L\bs)::SparseVector
        @test Fs.U\b ≈ Lp'\b ≈ (Fs.U\bs)::SparseVector
        @test Fs.L'\b ≈ Lp'\b
        @test Fs.U'\b ≈ Lp\b
        @test Fs.PtL\b ≈ Lp\b[p] ≈ (Fs.PtL\bs)::SparseVector
        @test Fs.UP\b ≈ (Lp'\b)[p_inv]
        @test Fs.PtL'\b ≈ (Lp'\b)[p_inv]
        @test Fs.UP'\b ≈ Lp\b[p]
        @test Fs.LD\b ≈ Dp\(Lp\b)
        @test Fs.DU'\b ≈ Dp\(Lp\b)
        @test Fs.LD'\b ≈ Lp'\(Dp\b)
        @test Fs.DU\b ≈ Lp'\(Dp\b)
        @test Fs.PtLD\b ≈ Dp\(Lp\b[p])
        @test Fs.DUP'\b ≈ Dp\(Lp\b[p])
        @test Fs.PtLD'\b ≈ (Lp'\(Dp\b))[p_inv]
        @test Fs.DUP\b ≈ (Lp'\(Dp\b))[p_inv]
        @test_throws CHOLMOD.CHOLMODException Fs.DUPt
        @test_throws CHOLMOD.CHOLMODException Fs.PLD
    end

    @testset "Element promotion and type inference" begin
        @inferred cholesky(As)\fill(1, size(As, 1))
        @inferred ldlt(As)\fill(1, size(As, 1))
    end
end

@testset "Issue 11745 - row and column pointers were not sorted in sparse(Factor)" begin
    A = Tv[10 1 1 1; 1 10 0 0; 1 0 10 0; 1 0 0 10]
    @test sparse(cholesky(sparse(A))) ≈ A
end
GC.gc()

@testset "Issue 11747 - Wrong show method defined for FactorComponent" begin
    v = cholesky(sparse(Tv[ 10 1 1 1; 1 10 0 0; 1 0 10 0; 1 0 0 10])).L
    for s in (sprint(show, MIME("text/plain"), v), sprint(show, v))
        @test occursin("method:  simplicial", s)
        @test !occursin("#undef", s)
    end
end

@testset "Issue 29367" begin
    if Int != Int32
        @test_nowarn cholesky(sparse(Int32[1,2,3,4], Int32[1,2,3,4], Tv[1,4,16,64]))
        @test_nowarn ldlt(sparse(Int32[1,2,3,4], Int32[1,2,3,4], Tv[1,4,16,64]))
    end
end

@testset "Issue 14134" begin
    A = CHOLMOD.Sparse(sprandn(Tv, 10,5,0.1) + I |> t -> t't)
    b = IOBuffer()
    serialize(b, A)
    seekstart(b)
    Anew = deserialize(b)
    @test_throws ArgumentError show(Anew)
    @test_throws ArgumentError size(Anew)
    @test_throws ArgumentError Anew[1]
    @test_throws ArgumentError Anew[2,1]
    F = cholesky(A)
    serialize(b, F)
    seekstart(b)
    Fnew = deserialize(b)
    @test_throws ArgumentError Fnew\fill(1., 5)
    @test_throws ArgumentError show(Fnew)
    @test_throws ArgumentError size(Fnew)
    @test_throws ArgumentError diag(Fnew)
    @test_throws ArgumentError logdet(Fnew)
end

@testset "Issue #28985" begin
    @test typeof(cholesky(Tv.(sparse(I, 4, 4)))'\rand(Tv, 4)) == Array{Tv, 1}
    @test typeof(cholesky(Tv.(sparse(I, 4, 4)))'\rand(Tv, 4,1)) == Array{Tv, 2}
end

@testset "Issue with promotion during conversion to CHOLMOD.Dense" begin
    @test CHOLMOD.Dense(fill(1, 5)) == fill(1, 5, 1)
    @test CHOLMOD.Dense(fill(1f0, 5)) == fill(1, 5, 1)
    @test CHOLMOD.Dense(fill(1f0 + 0im, 5, 2)) == fill(1, 5, 2)
end

@testset "Further issue with promotion #14894" begin
    x = fill(1., 5)
    @test cholesky(sparse(Float16(1)I, 5, 5))\x == x
    @test cholesky(Symmetric(sparse(Float16(1)I, 5, 5)))\x == x
    @test cholesky(Hermitian(sparse(Complex{Float16}(1)I, 5, 5)))\x == x
    @test_throws TypeError cholesky(sparse(BigFloat(1)I, 5, 5))
    @test_throws TypeError cholesky(Symmetric(sparse(BigFloat(1)I, 5, 5)))
    @test_throws TypeError cholesky(Hermitian(sparse(Complex{BigFloat}(1)I, 5, 5)))
end

@testset "test \\ for Factor and StridedVecOrMat" begin
    x = rand(5)
    A = cholesky(sparse(Diagonal(x.\1)))
    @test A\view(fill(1.,10),1:2:10) ≈ x
    @test A\view(Matrix(1.0I, 5, 5), :, :) ≈ Matrix(Diagonal(x))
    @test A\view(Matrix(1.0I, 6, 5), 1:5, :) ≈ Matrix(Diagonal(x))
end

@testset "Test \\ for Factor and SparseVecOrMat" begin
    sparseI = sparse(1.0I, 100, 100)
    sparseb = sprandn(100, 0.5)
    sparseB = sprandn(100, 100, 0.5)
    chI = cholesky(sparseI)
    @test chI \ sparseb ≈ sparseb
    @test chI \ sparseB ≈ sparseB
    @test chI \ sparseI ≈ sparseI
end

@testset "Issue 630" begin
    sparseI = sparse(1.0I, 1, 1)
    @test cholesky(sparseI) \ sparse([1.0]) == [1]
    sparseI = sparse(1.0I, 2, 2)
    res = cholesky(sparseI) \ spzeros(2)
    @test isempty(nonzeros(res))
end

@testset "Real factorization and complex rhs" begin
    A = sprandn(5, 5, 0.4) |> t -> t't + I
    B = complex.(randn(5, 5), randn(5, 5))
    b = B[:,1]
    @test cholesky(A)\b ≈ A\b
    @test cholesky(A)\B ≈ A\B
    @test cholesky(A)\B' ≈ A\B'
    @test cholesky(A)\transpose(B) ≈ A\transpose(B)
    @test cholesky(A)'\b ≈ copy(A')\b
    @test cholesky(A)'\B ≈ copy(A')\B
    @test cholesky(A)'\B' ≈ copy(A')\B'
    @test cholesky(A)'\transpose(B) ≈ copy(A')\transpose(B)
end

@testset "Make sure that ldlt performs an LDLt (Issue #19032)" begin
    m, n = 400, 500
    A = sprandn(m, n, .2)
    M = [I copy(A'); A -I]
    b = M * fill(1., m+n)
    F = ldlt(M)
    s = unsafe_load(pointer(F))
    @test s.is_super == 0
    @test F\b ≈ fill(1., m+n)
    F2 = cholesky(M; check = false)
    @test !issuccess(F2)
    ldlt!(F2, M)
    @test issuccess(F2)
    @test F2\b ≈ fill(1., m+n)
end

@testset "Test that imaginary parts in Hermitian{T,SparseMatrixCSC{T}} are ignored" begin
    A = sparse([1,2,3,4,1], [1,2,3,4,2], [complex(2.0,1),2,2,2,1])
    Fs = cholesky(Hermitian(A))
    Fd = cholesky(Hermitian(Array(A)))
    @test sparse(Fs) ≈ Hermitian(A)
    @test Fs\fill(1., 4) ≈ Fd\fill(1., 4)
end

@testset "\\ '\\ and transpose(...)\\" begin
    # Test that \ and '\ and transpose(...)\ work for Symmetric and Hermitian. This is just
    # a dispatch exercise so it doesn't matter that the complex matrix has
    # zero imaginary parts
    Apre = sprandn(Tv, 10, 10, 0.2) - I
    for A in (Symmetric(Apre), Hermitian(Apre),
              Symmetric(Apre + 10I), Hermitian(Apre + 10I),
              Hermitian(complex(Apre)), Hermitian(complex(Apre) + 10I))
        local A, x, b
        x = fill(1, 10)
        b = A*x
        @test @inferred A\b ≈ x
        @test transpose(A)\b ≈ A'\b
    end
end

@testset "Check that Symmetric{SparseMatrixCSC} can be constructed from CHOLMOD.Sparse" begin
    Int === Int32 && Random.seed!(124)
    A = sprandn(Tv, 10, 10, 0.1)
    B = CHOLMOD.Sparse(A)
    C = B'B
    # Change internal representation to symmetric (upper/lower)
    o = fieldoffset(cholmod_sparse, findall(fieldnames(cholmod_sparse) .== :stype)[1])
    for uplo in (1, -1)
        unsafe_store!(Ptr{Int8}(pointer(C)), uplo, Int(o) + 1)
        @test convert(Symmetric{Tv,SparseMatrixCSC{Tv,Int}}, C) ≈ Symmetric(A'A)
    end
end

@testset "sparse right multiplication of Symmetric and Hermitian matrices #21431" begin
    S = sparse(1.0I, 2, 2)
    @test issparse(S*S*S)
    for T in (Symmetric, Hermitian)
        @test issparse(S*T(S)*S)
        @test issparse(S*(T(S)*S))
        @test issparse((S*T(S))*S)
    end
end

@testset "Test sparse low rank update for cholesky decomposition" begin
    A = SparseMatrixCSC{Tv,Int}(10, 5, [1,3,6,8,10,13], [6,7,1,2,9,3,5,1,7,6,7,9],
        Tv[-0.138843, 2.99571, -0.556814, 0.669704, -1.39252, 1.33814,
        1.02371, -0.502384, 1.10686, 0.262229, -1.6935, 0.525239])
    AtA = A'*A
    C0 = Tv[1., 2., 0, 0, 0]
    # Test both cholesky and LDLt with and without automatic permutations
    for F in (cholesky(AtA), cholesky(AtA, perm=1:5), ldlt(AtA), ldlt(AtA, perm=1:5))
        local F
        x0 = F\(b = ones(Tv, 5))
        #Test both sparse/dense and vectors/matrices
        for Ctest in (C0, sparse(C0), [C0 2*C0], sparse([C0 2*C0]))
            local x, C, F1
            C = copy(Ctest)
            F1 = copy(F)
            x = (AtA+C*C')\b

            #Test update
            F11 = CHOLMOD.lowrankupdate(F1, C)
            @test Array(sparse(F11)) ≈ AtA+C*C'
            @test F11\b ≈ x
            #Make sure we get back the same factor again
            F10 = CHOLMOD.lowrankdowndate(F11, C)
            @test Array(sparse(F10)) ≈ AtA
            @test F10\b ≈ x0

            #Test in-place update
            CHOLMOD.lowrankupdate!(F1, C)
            @test Array(sparse(F1)) ≈ AtA+C*C'
            @test F1\b ≈ x
            #Test in-place downdate
            CHOLMOD.lowrankdowndate!(F1, C)
            @test Array(sparse(F1)) ≈ AtA
            @test F1\b ≈ x0

            @test C == Ctest    #Make sure C didn't change
        end
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

@testset "Non-positive definite matrices" begin
    A = sparse(Tv[1 2; 2 1])
    B = sparse(Complex{Tv}[1 2; 2 1])
    for M in (A, B, Symmetric(A), Hermitian(B))
        F = cholesky(M; check = false)
        @test_throws PosDefException cholesky(M)
        @test_throws PosDefException cholesky!(F, M)
        @test !issuccess(cholesky(M; check = false))
        @test !issuccess(cholesky!(F, M; check = false))
    end
    A = sparse(Tv[0 0; 0 0])
    B = sparse(Complex{Tv}[0 0; 0 0])
    for M in (A, B, Symmetric(A), Hermitian(B))
        F = ldlt(M; check = false)
        @test_throws ZeroPivotException ldlt(M)
        @test_throws ZeroPivotException ldlt!(F, M)
        @test !issuccess(ldlt(M; check = false))
        @test !issuccess(ldlt!(F, M; check = false))
    end
end

@testset "Issues #27860 & #28363" begin
    for typeA in (Tv, Complex{Tv}), typeB in (Tv, Complex{Tv}), transform in (identity, adjoint, transpose)
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

@testset "Issue #33365" begin
    A = Sparse(spzeros(Tv, 0, 0))
    @test A * A' == A
    @test A' * A == A
    B = Sparse(spzeros(Tv, 0, 4))
    @test B * B' == Sparse(spzeros(Tv, 0, 0))
    @test B' * B == Sparse(spzeros(Tv, 4, 4))
    C = Sparse(spzeros(Tv, 3, 0))
    @test C * C' == Sparse(spzeros(Tv, 3, 3))
    @test C' * C == Sparse(spzeros(Tv, 0, 0))
end

@testset "permutation handling" begin
    @testset "default permutation" begin
        # Assemble arrow matrix
        A = sparse(5I,3,3)
        A[:,1] .= 1; A[1,:] .= A[:,1]

        # Ensure cholesky eliminates the fill-in
        @test cholesky(A).p[1] != 1
    end

    @testset "user-specified permutation" begin
        n = 100
        A = sprand(Tv, n,n,5/n) |> t -> t't + I
        @test cholesky(A, perm=1:n).p == 1:n
    end
end

@testset "sym indefinite poly alg" begin
    K = open(joinpath(@__DIR__, "matrices", "stiffness_sym_indef")) do io
        ml = readline(io)
        m = parse(Int, split(ml, "m = ")[2])
        nl = readline(io)
        n = parse(Int, split(nl, "n = ")[2])

        colptrl = readline(io)
        rowvall = readline(io)
        nzvall = readline(io)

        colptr = parse.(Int,     split(strip(split(colptrl, "colptr = ")[2], [']', '[']), ','))
        rowval = parse.(Int,     split(strip(split(rowvall, "rowval = ")[2], [']', '[']), ','))
        nzval =  parse.(Float64, split(strip(split(nzvall, "nzval = ")[2], [']', '[']), ','))

        SparseMatrixCSC(m, n, colptr, rowval, nzval)
    end

    f = ones(size(K, 1))
    u = K \ f
    residual = norm(f - K * u) / norm(f)
    @test residual < 1e-6
end

@testset "wrapped sparse matrices" begin
    A = I + sprand(Tv, 10, 10, 0.1); A = A'A
    @test issuccess(cholesky(view(A, :, :)))
    @test issuccess(cholesky(Symmetric(view(A, :, :))))
    @test_throws ErrorException cholesky(view(A, :, :), RowMaximum())
    # turn on once two-arg cholesky is made to forward any PivotingStrategy argument
    # @test_throws ErrorException cholesky(A, NoPivot())
    # @test_throws ErrorException cholesky(view(A, :, :), NoPivot())
end

@testset "solve with adjoint factorization and adjoint rhs" begin
    n = 10
    A = sprand(Tv, n, n, 1/n)
    A = A + A' + 10I

    B = rand(n, 2)
    Bt = Matrix(B')
    Bts = sparse(B')

    F = cholesky(A)'
    @test F \ B ≈ F \ Bt'
    @test F \ B ≈ F \ Bts'
    @test issparse(F \ Bts')
end

@testset "getindex with unsorted or unpacked buffers (#758), Ti = $Ti" for Ti ∈ itypes
    # the product of two matrices with sorted row indices need not be sorted
    A = sparse(Ti[2, 1, 2], Ti[1, 2, 2], Tv[1, 2, 3])
    S = CHOLMOD.Sparse(A)
    P = S'S
    @test Array(P) == Matrix(sparse(P)) == Matrix(A'A)
    @test P[1, 2] == (A'A)[1, 2]
    @test_throws BoundsError P[0, 1]
    @test_throws BoundsError P[1, 3]

    # a matrix that is both unsorted and unpacked: room for three entries is
    # reserved in every column, column 1 stores rows 3 and 1 in that order,
    # column 2 stores row 2 and column 3 is empty
    U = CHOLMOD.allocate_sparse(3, 3, 9, false, false, 0, Tv, Ti)
    s = unsafe_load(CHOLMOD.typedpointer(U))
    unsafe_wrap(Array, s.p, 4) .= Ti[0, 3, 6, 9]
    unsafe_wrap(Array, s.nz, 3) .= Ti[2, 1, 0]
    rowval = unsafe_wrap(Array, s.i, 9)
    nzval = unsafe_wrap(Array, Ptr{Tv}(s.x), 9)
    rowval[1], nzval[1] = 2, 10 # U[3, 1]
    rowval[2], nzval[2] = 0, 20 # U[1, 1]
    rowval[4], nzval[4] = 1, 30 # U[2, 2]
    @test Array(U) == Tv[20 0 0; 0 30 0; 10 0 0]
end

end # for Tv ∈ (Float32, Float64)

end # Base.USE_GPL_LIBS

end # module
