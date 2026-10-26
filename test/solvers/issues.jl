# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseSolverIssuesTests
using Test
using SparseArrays
using SparseArrays: CHOLMOD, UMFPACK
using LinearAlgebra
using Serialization
include("../testhelpers.jl")

@testset "Issue #9915, $Tv" for Tv in (Float32, Float64)
    sparseI = sparse(Tv(1.0)I, 2, 2)
    @test sparseI \ sparseI == sparseI
end

@testset "Issue 11745 - row and column pointers were not sorted in sparse(Factor)" begin
    Tv = Float64
    A = Tv[10 1 1 1; 1 10 0 0; 1 0 10 0; 1 0 0 10]
    @test sparse(cholesky(sparse(A))) ≈ A
end

@testset "Issue 29367" begin
    Tv = Float64
    if Int != Int32
        @test_nowarn cholesky(sparse(Int32[1,2,3,4], Int32[1,2,3,4], Tv[1,4,16,64]))
        @test_nowarn ldlt(sparse(Int32[1,2,3,4], Int32[1,2,3,4], Tv[1,4,16,64]))
    end
end

@testset "Issue 14134" begin
    Tv = Float64
    A = CHOLMOD.Sparse(fixture(Tv, 10, 5) + sparse(I, 10, 5) |> t -> t't)
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
    Tv = Float64
    @test typeof(cholesky(Tv.(sparse(I, 4, 4)))'\ones(Tv, 4)) == Array{Tv, 1}
    @test typeof(cholesky(Tv.(sparse(I, 4, 4)))'\ones(Tv, 4,1)) == Array{Tv, 2}
end

@testset "Issues #27860 & #28363" begin
    Tv = Float64
    # both mixed pairs of element types, and each transform once
    for (typeA, typeB, transform) in ((Tv, Complex{Tv}, adjoint), (Complex{Tv}, Tv, transpose), (Complex{Tv}, Complex{Tv}, identity))
        A = sparse(typeA[2.0 0.1; 0.1 2.0])
        B = typeB <: Real ? typeB[1 2; 3 4] : typeB[1+2im 2-im; 3+im 4-3im]
        @test A \ transform(B) ≈ cholesky(A) \ transform(B) ≈ Matrix(A) \ transform(B)
        C = typeA <: Real ? typeA[2 -1; 1 3] : typeA[2-im -1+3im; 1+im 3+2im]
        sC = sparse(C)
        sF = typeA <: Real ? cholesky(Symmetric(A)) : cholesky(Hermitian(A))
        @test cholesky(A) \ transform(sC) ≈ Matrix(A) \ transform(C)
        @test sF.PtL \ transform(A) ≈ sF.PtL \ Matrix(transform(A))
    end
end

@testset "Issue #33365" begin
    Tv = Float64
    A = CHOLMOD.Sparse(spzeros(Tv, 0, 0))
    @test A * A' == A
    @test A' * A == A
    B = CHOLMOD.Sparse(spzeros(Tv, 0, 4))
    @test B * B' == CHOLMOD.Sparse(spzeros(Tv, 0, 0))
    @test B' * B == CHOLMOD.Sparse(spzeros(Tv, 4, 4))
    C = CHOLMOD.Sparse(spzeros(Tv, 3, 0))
    @test C * C' == CHOLMOD.Sparse(spzeros(Tv, 3, 3))
    @test C' * C == CHOLMOD.Sparse(spzeros(Tv, 0, 0))
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

@testset "Issue 630" begin
    sparseI = sparse(1.0I, 1, 1)
    @test cholesky(sparseI) \ sparse([1.0]) == [1]
    sparseI = sparse(1.0I, 2, 2)
    res = cholesky(sparseI) \ spzeros(2)
    @test isempty(nonzeros(res))
end

@testset "Make sure that ldlt performs an LDLt (Issue #19032)" begin
    m, n = 400, 500
    # entries of at most one keep M well conditioned
    A = fixture(Float64, m, n) / (m + 3n)
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

@testset "symmetric indefinite solve fallback (#325)" begin
    # Well conditioned and symmetric indefinite with a tiny diagonal: `cholesky` fails, and
    # an unpivoted LDLt succeeds but is inaccurate, so `\` has to fall back to `lu` (#325)
    n = 50
    K = spdiagm(-1 => ones(n - 1), 0 => [isodd(i) ? 1e-16 : -1e-16 for i in 1:n], 1 => ones(n - 1))
    @test !issuccess(cholesky(Symmetric(K); check = false))

    f = ones(size(K, 1))
    u = K \ f
    residual = norm(f - K * u) / norm(f)
    @test residual < 1e-6
end

@testset "sparse right-hand side on a failed factorization throws (PR #1029)" begin
    Ti, Tv = Int, Float64
    A = SparseMatrixCSC{Tv,Ti}(Tv[2 0 0; 0 -3 0; 0 0 5])
    b = sparse(Tv[1, 2, 3])
    F = cholesky(A; check=false)
    @test !issuccess(F)
    @test_throws PosDefException F \ b
    @test_throws PosDefException F' \ b
    @test_throws PosDefException F \ sparse(Tv[1 0; 0 1; 1 1])
    G = ldlt(SparseMatrixCSC{Tv,Ti}(Tv[0 0; 0 1]); check=false)
    @test !issuccess(G)
    @test_throws ZeroPivotException G \ sparse(Tv[1, 2])
end

@testset "diag of a symbolic factorization throws (PR #1029)" begin
    Ti, Tv = Int, Float64
    F = CHOLMOD.symbolic(CHOLMOD.Sparse(SparseMatrixCSC{Tv,Ti}(Tv[4 1; 1 3])))
    @test_throws ArgumentError diag(F)
end

@testset "Issue #4523 - complex sparse \\" begin
    A, b = sparse((1.0 + im)I, 2, 2), fill(1., 2)
    @test A * (lu(A)\b) ≈ b

    @test det(sparse([1,3,3,1], [1,1,3,3], [1,1,1,1])) == 0
end

@testset "Issues #18246,18244 - lu sparse pivot" begin
    A = sparse(1.0I, 4, 4)
    A[1:2,1:2] = [-.01 -200; 200 .001]
    F = lu(A)
    UMFPACK.umfpack_report_numeric(F, 0)
    UMFPACK.umfpack_report_symbolic(F, 0)
    @test F.p == [3 ; 4 ; 2 ; 1]
end

@testset "lu!(F, S) does not reuse the symbolic factors of another size (PR #1029), $Ti" for Ti in core_itypes
    F = lu(SparseMatrixCSC{Float64,Ti}(sparse([4.0 1; 1 3])))
    C = sparse([4.0 1 0; 1 3 0; 0 0 7])
    lu!(F, C)
    @test size(F.L) == (3, 3)
    @test F \ [1.0, 2.0, 3.0] ≈ Matrix(C) \ [1.0, 2.0, 3.0]
    @test det(F) ≈ det(Matrix(C))
end

@testset "logabsdet after lu! with a matrix of another size (#1029, PR #1034), $Tv" for Tv in STD_ELTYPES
    F = lu(sparse(Diagonal(Tv[2, 3])))
    A = sparse(Diagonal(Tv[7, 11, 13]))
    lu!(F, A)
    @test all(logabsdet(F) .≈ logabsdet(Matrix(A)))
end

@testset "lu!(F, S) leaves F unchanged when the values of S do not convert" begin
    A = sparse(ComplexF64[4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4])
    F = lu(A)
    colptr, rowval, nzval = copy(F.colptr), copy(F.rowval), copy(F.nzval)
    S = SparseMatrixCSC{Any,Int}(2, 2, [1, 2, 4], [1, 1, 2], Any[1.0, "x", 1.0])
    @test_throws MethodError lu!(F, S)
    @test size(F) == (4, 4)
    @test (F.colptr, F.rowval, F.nzval) == (colptr, rowval, nzval)
    b = ComplexF64[1, 2, 3, 4]
    @test F \ b ≈ Matrix(A) \ b
end

@testset "cholesky and cholesky! with an index type other than CHOLMOD's" begin
    A = sparse([4.0 1 0; 1 4 1; 0 1 4])
    @test_throws ArgumentError cholesky(SparseMatrixCSC{Float64,UInt64}(A))
    F = cholesky(A)
    @test cholesky!(F, SparseMatrixCSC{Float64,Int16}(2A)) \ [1.0, 2.0, 3.0] ≈ (2A) \ [1.0, 2.0, 3.0]
end

@testset "ldlt! with a matrix of another size leaves an LLt factor unchanged" begin
    A = sparse([4.0 1 0; 1 4 1; 0 1 4])
    F = cholesky(A)
    @test_throws DimensionMismatch ldlt!(F, sparse(1.0I, 4, 4))
    @test :D ∉ propertynames(F)
    @test F \ ones(3) ≈ A \ ones(3)
end

@testset "Issues #26367 and #26368" begin
    A = sparse([0.0 1 0 0; 0 0 0 0])
    F = qr(A)
    Fd = qr(Matrix(A))
    Qs = sparse(F.Q)
    @test Matrix(F.Q) == Matrix(Fd.Q) == Matrix(I, 2, 2)
    @test Qs == sparse(Fd.Q) == Matrix(I, 2, 2)
    @test (sparse(I, 2, 2) * F.Q)::Matrix == Qs == sparse(I, 2, 2)
    @test (F.Q*F.R)::Matrix == A[F.prow,F.pcol]
end

@testset "Issue #585 for element type: $eltyA" for eltyA in (Float32, ComplexF32)
    A = sparse(eltyA[1 0; 0 1])
    F = qr(A)
    @test eltype(F.Q) == eltype(F.R) == eltyA
end

@testset "sparsevec of a CHOLMOD.Sparse checks its shape and stype" begin
    @test_throws DimensionMismatch sparsevec(CHOLMOD.Sparse(sparse([1.0 2; 0 3])))
    @test_throws ArgumentError sparsevec(CHOLMOD.Sparse(Symmetric(sparse([2.0 1; 1 2]))))
end

@testset "ldiv! of a real UmfpackLU with a ComplexF32 vector" begin
    A = sparse([2.0 1; 1 3])
    @test ldiv!(lu(A), ComplexF32[1, 2]) ≈ ComplexF32.(Matrix(A) \ [1, 2])
end

@testset "factorize of an indefinite Hermitian matrix whose LDLt needs pivoting" begin
    b = [1.0, 2.0]
    @test factorize(sparse([0.0 1; 1 0])) \ b ≈ [0.0 1; 1 0] \ b
end

@testset "cond of a singular matrix is Inf" begin
    @test cond(sparse([1.0 2; 3 6]), 1) === Inf
    # the triangular solve, not the factorization, finds this one singular
    @test cond(sparse([1.0 0; 2 0]), Inf) === Inf
end

end # module
