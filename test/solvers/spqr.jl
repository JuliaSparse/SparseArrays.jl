# This file is a part of Julia. License is MIT: https://julialang.org/license

module SPQRTests
using Test

using SparseArrays.SPQR
using SparseArrays.CHOLMOD
using LinearAlgebra: I, istril, istriu, lq, norm, qr, rank, rmul!, lmul!, ldiv!, factorize, Adjoint, Transpose, ColumnNorm, RowMaximum, NoPivot
using SparseArrays: SparseArrays, sparse, spzeros, SparseMatrixCSC
include("../testhelpers.jl")


const m, n = 100, 10

# The fixture has an empty row and an empty column. A diagonal gives it full rank, for the
# solves that are compared with dense ones: a rank-deficient system has a basic solution
# from SPQR and the minimum-norm one from LAPACK. The scaling keeps the entries of order
# one, for the absolute tolerances.
fullrank(::Type{T}, m, n) where {T} = (fixture(T, m, n) + sparse(1:min(m, n), 1:min(m, n), fill(T(10), min(m, n)), m, n)) / max(m, n)

@test size(qr(fixture(Float64, m, n)).Q) == (m, m)

@test repr("text/plain", qr(fixture(Float64, 4, 4)).Q) == "4×4 $(SparseArrays.SPQR.QRSparseQ{Float64, Int})"

# SPQR has one entry point per element type and per index type, so a comprehensive run takes
# each of them once: the standard case, and the complex element type with `Int32` indices
@testset "element type of A: $eltyA" for eltyA in (@static COMPREHENSIVE ? STD_ELTYPES : (Float64,)), iltyA in (@static COMPREHENSIVE ? (eltyA <: Real ? core_itypes : (Int32,)) : core_itypes)
    A = SparseMatrixCSC{eltyA, iltyA}(fullrank(eltyA, m, n))

    F = qr(A)
    @test size(F) == (m,n)
    # qr of an adjoint or transpose factorizes the sparse transpose with SPQR
    for X in (A', (@static COMPREHENSIVE ? (transpose(A),) : ())...)
        @test which(qr, Base.typesof(X)).module == SPQR
        G = qr(X; tol = 1e-3)
        @test G isa SPQR.QRSparse{eltyA, iltyA} && size(G) == (n, m)
        @test G.Q * G.R ≈ Matrix(X)[G.prow, G.pcol]
    end
    @test size(F, 1) == m
    @test size(F, 2) == n
    @test size(F, 3) == 1
    @test_throws ArgumentError size(F, 0)

    @testset "getindex" begin
        @test istriu(F.R)
        @test isperm(F.pcol)
        @test isperm(F.prow)
        @test @inferred((F -> F.pcol)(F)) isa Vector{iltyA}
        @test @inferred((F -> F.prow)(F)) isa Vector{iltyA}
        @test_throws isdefined(Base, :FieldError) ? FieldError : ErrorException F.T
    end

    @testset "apply Q" begin
        Q = F.Q
        Imm = Matrix{Float64}(I, m, m)
        @test Q' * (Q*Imm) ≈ Imm
        @test (Imm*Q) * Q' ≈ Imm
        @test ((Imm[:,1])' * Q')::Adjoint ≈ Q[:,1]'

        # test that Q'Pl*A*Pr = R
        R0 = Q'*Array(A[F.prow, F.pcol])
        @test mismatch(F.R, R0[1:n, :]; Ti=iltyA, approx=true) === nothing
        @test norm(R0[n + 1:end, :], 1) < 1e-12

        offsizeA = Matrix{Float64}(I, m+1, m+1)
        @test_throws DimensionMismatch lmul!(Q, offsizeA)
        @test_throws DimensionMismatch lmul!(adjoint(Q), offsizeA)
        @test_throws DimensionMismatch rmul!(offsizeA, Q)
        @test_throws DimensionMismatch rmul!(offsizeA, adjoint(Q))

        # products with an operand of another element type convert Q
        Qd = Q * Matrix{eltyA}(I, m, m)
        b, B = complex.(1.0:m, m:-1.0:1), complex.(reshape(1.0:3m, 3, m), reshape(3m:-1.0:1, 3, m))
        @test Q * b ≈ Qd * b
        @test Q' * b ≈ Qd' * b
        @test B * Q' ≈ B * Qd'
    end

    @testset "right-hand sides that are not strided arrays of the same element type" begin
        # a real wrapper and a complex view reach both right-hand side conversions; the other kinds vary only the array type
        kinds = [1, 5]
        @static COMPREHENSIVE && union!(kinds, eltyA <: Real ? (2, 4, 6) : (3, 7))
        rhs(k) = (reshape(collect(1.0:2k), 2, k)', transpose(reshape(collect(1.0:2k), 2, k)), fixturevec(Float64, k), 1:k,
                  view(complex.(reshape(1.0:2k, k, 2), reshape(2k:-1.0:1, k, 2)), :, 1),
                  complex.(reshape(1.0:2k, 2, k), reshape(2k:-1.0:1, 2, k))', ComplexF32.(complex.(1:k, k:-1:1)))[kinds]
        for X in rhs(m)
            @test A \ X ≈ Array(A) \ Array(X)
            @test F \ X ≈ Array(A) \ Array(X)
        end
        @static if COMPREHENSIVE
        C = A[1:9, :]   # wide
        for X in rhs(9)
            @test C \ X ≈ Array(C) \ Array(X)
            @test lq(C) \ X ≈ Array(C) \ Array(X)
        end
        end
        for X in rhs(n)
            @test F' \ X ≈ Array(A)' \ Array(X)
        end
    end

    @static if COMPREHENSIVE
    @testset "element type of B: $eltyB" for eltyB in (eltyA <: Real ? (Int, ComplexF64) : (ComplexF64,))
        if eltyB == Int
            B = reshape(mod1.(3 .* (1:2m), 10), m, 2)
        elseif eltyB <: Real
            B = reshape(collect(1.0:2m), m, 2)
        else
            B = complex.(reshape(1.0:2m, m, 2), reshape(2m:-1.0:1, m, 2))
        end

        @inferred A\B
        @test A\B[:,1] ≈ Array(A)\B[:,1]
        @test A\B ≈ Array(A)\B
        @test_throws DimensionMismatch A\B[1:m-1,:]
        C, x = A[1:9, :], fill(eltyB(1), 9)
        @test C*(C\x) ≈ x # Underdetermined system
        # A \ b returns the minimum-norm solution for a wide A, like dense (#301)
        @test C\x ≈ Array(C)\x
        @test C\B[1:9, :] ≈ Array(C)\B[1:9, :]
        @test factorize(C)\x ≈ Array(C)\x

        # Minimum-norm solution of the underdetermined A'x = b (#656)
        D = B[1:n, :]
        @test F'\D ≈ Array(A)'\D
        @test F'\D[:,1] ≈ Array(A)'\D[:,1]
        @test transpose(F)\D ≈ transpose(Array(A))\D
        eltyB == eltyA && @test ldiv!(zeros(eltyA, m, 2), transpose(F), D; workspace = SPQR.SpqrWS(F)) ≈ transpose(F)\D
        @test A'\D ≈ Array(A)'\D
        @test_throws DimensionMismatch F'\B
        # Least squares solve of the overdetermined C'y = x for the wide C
        y = B[1:n, 1]
        @test C'\y ≈ Array(C)'\y
        @static if COMPREHENSIVE
        @test transpose(C)\y ≈ transpose(Array(C))\y
        end
    end
    end

    @testset "lq (#114)" begin
        W = A[1:9, :]   # wide
        F = lq(W)
        @test F isa SPQR.AdjointQRSparse{eltyA} && size(F) == size(W)
        @test F.L isa SparseMatrixCSC{eltyA, iltyA} && istril(F.L)
        @test F.L * F.Q ≈ Matrix(W)[F.prow, F.pcol]
        @test @inferred((F -> F.pcol)(F)) isa Vector{iltyA}
        @test @inferred((F -> F.prow)(F)) isa Vector{iltyA}
        @test rank(F) == 9 && propertynames(F) == (:L, :Q, :prow, :pcol)
        @test F' isa SPQR.QRSparse{eltyA, iltyA}
        @test occursin("L factor", sprint(show, MIME"text/plain"(), F))
        b = eltyA <: Real ? reshape(collect(1.0:18), 9, 2) : complex.(reshape(1.0:18, 9, 2), reshape(18:-1.0:1, 9, 2))
        @test F \ b ≈ Matrix(W) \ b   # the minimum-norm solution, as for dense lq
        @test F \ b[:, 1] ≈ Matrix(W) \ b[:, 1]
        @test lq(W; tol = 1e-3) \ b ≈ Matrix(W) \ b
        @test_throws DimensionMismatch lq(A) \ ones(eltyA, m)   # overdetermined, as for dense lq
        c = eltyA <: Real ? collect(1.0:n) : complex.(1.0:n, n:-1.0:1)
        @test lq(A') \ c ≈ Matrix(A') \ c   # reuses qr(A)
        @test lq(transpose(A)) \ c ≈ Matrix(transpose(A)) \ c
    end

    # Make sure that conversion to Sparse doesn't use SuiteSparse's symmetric flag
    @test qr(SparseMatrixCSC{eltyA}(I, 5, 5)) \ fill(eltyA(1), 5) == fill(1, 5)
end

# the loop above covers the complex element type in a comprehensive run
@static if !COMPREHENSIVE
@testset "complex element type" begin
    A = fullrank(ComplexF64, m, n)
    F = qr(A)
    Ad = Array(A)
    @test F isa SPQR.QRSparse{ComplexF64, Int}
    @test istriu(F.R)
    R0 = F.Q' * Ad[F.prow, F.pcol]
    @test mismatch(F.R, R0[1:n, :]; Ti=Int, approx=true) === nothing
    @test norm(R0[n + 1:end, :], 1) < 1e-12
    # the transpose is not the adjoint
    X = transpose(A)
    G = qr(X)
    @test G isa SPQR.QRSparse{ComplexF64, Int}
    @test G.Q * G.R ≈ Matrix(X)[G.prow, G.pcol]
    B = complex.(reshape(1.0:2m, m, 2), reshape(2m:-1.0:1, m, 2))
    @test A\B ≈ Ad\B
    D = B[1:n, :]
    @test F'\D ≈ copy(Ad')\D
    @test transpose(F)\D ≈ copy(transpose(Ad))\D
    @test lq(X)\D ≈ copy(transpose(Ad))\D
    @test ldiv!(zeros(ComplexF64, m, 2), transpose(F), D; workspace = SPQR.SpqrWS(F)) ≈ transpose(F)\D
    W = A[1:9, :]
    L = lq(W)
    @test L.L * L.Q ≈ Ad[1:9, :][L.prow, L.pcol]
    @test L \ B[1:9, :] ≈ Ad[1:9, :] \ B[1:9, :]
end
end

@testset "basic solution of rank deficient ls" begin
    A = fullrank(Float64, m, 5)*fullrank(Float64, 5, n)   # of rank 5
    b = collect(1.0:m)
    xs = A\b
    xd = Array(A)\b

    # check that basic solution has more zeros
    @test count(!iszero, xs) < count(!iszero, xd)
    @test A*xs ≈ A*xd
end

@static if COMPREHENSIVE
@testset "Issue 26367" begin
    A = sparse([0.0 1 0 0; 0 0 0 0])
    @test Matrix(qr(A).Q) == Matrix(qr(Matrix(A)).Q) == Matrix(I, 2, 2)
    @test sparse(qr(A).Q) == sparse(qr(Matrix(A)).Q) == Matrix(I, 2, 2)
    @test (sparse(I, 2, 2) * qr(A).Q)::Matrix == sparse(qr(A).Q) == sparse(I, 2, 2)
end
end

@testset "thin Q products when SPQR stores fewer reflectors than columns" begin
    A = sparse([1:9; 3], [1:9; 5], collect(1.0:10), 10, 9)
    F = qr(A)
    @test size(F.Q.factors, 2) < size(A, 2)
    @test F.Q * F.R ≈ A[F.prow, F.pcol]
    @test F.Q * F.R[:, 1] ≈ A[F.prow, F.pcol][:, 1]
    @test Matrix(F.R)' * F.Q' ≈ A[F.prow, F.pcol]'
end

@static if COMPREHENSIVE
@testset "Issue 26368" begin
    A = sparse([0.0 1 0 0; 0 0 0 0])
    F = qr(A)
    @test (F.Q*F.R)::Matrix == A[F.prow,F.pcol]
end
end

@testset "products of Q with sparse operands (#121), size(A) = $(size(A))" for A in
        (fixture(ComplexF64, 5, 3), (@static COMPREHENSIVE ? (fixture(ComplexF64, 6, 20),) : ())...)
    local m, n = size(A)
    k = min(m, n)   # the rows of R and the columns of the thin Q
    F = qr(A)
    Q = F.Q
    T = eltype(A)
    # the identity from the issue, with a thin R for a tall A
    @test (Q * F.R)::Matrix ≈ A[F.prow, F.pcol]
    # one operand of each kind, including the thin shapes the dense-operand methods
    # accept, gives the product with an explicitly formed Q: a product of Q with the dense
    # copy of the operand takes the same method again, and would agree with it when wrong
    Qd = Q * Matrix{T}(I, m, m)
    B, C, b = fixture(T, m, 3), fixture(T, 3, m), fixturevec(T, m)
    for X in (B, (@static COMPREHENSIVE ? (sparse(B')', view(B, :, 1:2)) : ())..., fixture(T, k, 3))
        @test (Q * X)::Matrix ≈ Qd[:, 1:size(X, 1)] * Matrix(X)
    end
    for X in (C, (@static COMPREHENSIVE ? (transpose(sparse(transpose(C))), view(C, :, 1:m), view(B, :, 1:2)') : ())..., transpose(b), fixture(T, 3, k))
        @test (X * Q')::Matrix ≈ Matrix(X) * Qd[:, 1:size(X, 2)]'
    end
    @test (Q' * B)::Matrix ≈ Qd' * Matrix(B)
    @test (C * Q)::Matrix ≈ Matrix(C) * Qd
    for x in (b, (@static COMPREHENSIVE ? (view(B, :, 1), view(b, 1:m)) : ())...)
        @test (Q * x)::Vector ≈ Qd * Vector(x)
    end
    @test (Q' * b)::Vector ≈ Qd' * Vector(b)
    @static if COMPREHENSIVE
    @test (b' * Q)::Adjoint ≈ Vector(b)' * Qd
    end
    # and nothing else
    k == m || @test_throws DimensionMismatch Q' * fixture(T, k, 3)
    @test_throws DimensionMismatch Q * fixture(T, m + 1, 2)
end

@static if COMPREHENSIVE
@testset "Issue #585 for element type: $eltyA" for eltyA in (Float32, ComplexF32)
    A = sparse(eltyA[1 0; 0 1])
    F = qr(A)
    @test eltype(F.Q) == eltype(F.R) == eltyA
end
end

@testset "single-precision qr factorization works as expected: $eltyA" for eltyA in (Float32, ComplexF32)
    A = fixture(eltyA, m, n)
    F = qr(A)
    @test eltype(F.Q) == eltype(F.R) == eltyA
    @test Matrix(F.Q) * F.R ≈ A[F.prow, F.pcol]
    @static if COMPREHENSIVE
    # products with double-precision operands convert Q, in the same way for a complex Q
    b, B = collect(1.0:m), reshape(collect(1.0:3m), 3, m)
    eltyA <: Real && @test F.Q * b ≈ F.Q * eltyA.(b)
    eltyA <: Real && @test B * F.Q' ≈ eltyA.(B) * F.Q'
    end
end

@static if COMPREHENSIVE
@testset "select ordering overdetermined" begin
     A = fullrank(Float64, m, n)
     b = collect(1.0:m)
     xref = Array(A) \ b
     c = collect(1.0:n)
     cref = Array(A)' \ c
     for ordering ∈ SPQR.ORDERINGS
         QR = qr(A, ordering=ordering)
         @test isperm(QR.pcol)
         @test QR.Q * QR.R ≈ A[QR.prow, QR.pcol]
         x = QR \ b
         @test x ≈ xref
         @test QR' \ c ≈ cref
     end
     @test_throws ArgumentError qr(A, ordering=(@static COMPREHENSIVE ? Int32(10) : 10))
end
end

@static if COMPREHENSIVE
@testset "select ordering underdetermined" begin
     A = fullrank(Float64, n, m)
     b = A * ones(m)
     for ordering ∈ SPQR.ORDERINGS
         QR = qr(A, ordering=ordering)
         x = QR \ b
         # x ≂̸ Array(A) \ b; LAPACK returns a min-norm x while SPQR returns a basic x
         @test A * x ≈ b
     end
     @test_throws ArgumentError qr(A, ordering=(@static COMPREHENSIVE ? Int32(10) : 10))
end
end

# the element and index types meet the other way round from the loop over A above
@testset "ORDERING_FIXED with a dependent column, $Tv $Ti" for Tv in (@static COMPREHENSIVE ? STD_ELTYPES : (Float64,)), Ti in (@static COMPREHENSIVE ? (Tv <: Real ? (Int32,) : core_itypes) : core_itypes)
    # the second column is twice the first
    A = SparseMatrixCSC{Tv, Ti}([1 2 3; 4 8 6; 7 14 9; 1 2 5])
    F = qr(A; ordering=SPQR.ORDERING_FIXED)
    @test rank(F) == 2
    @test isperm(F.pcol) && istriu(F.R)
    @test F.Q * F.R ≈ A[F.prow, F.pcol]
    b = A * Tv[1, 2, 3]
    @test A * (F \ b) ≈ b
    c = A' * Tv[1, 2, 3, 4]
    @test A' * (F' \ c) ≈ c
    W = sparse(A')   # wide
    G = qr(W; ordering=SPQR.ORDERING_FIXED)
    @test G.Q * G.R ≈ W[G.prow, G.pcol]
    @test W * (G \ c) ≈ c
    # without dependent columns the ordering is the identity
    @test qr(A[:, [1, 3]]; ordering=SPQR.ORDERING_FIXED).pcol == 1:2
end

@testset "non-floating-point element types" begin
    @static if COMPREHENSIVE
    A = sparse(Complex{Int}[1 2; 3 4im])
    @test qr(A) isa SPQR.QRSparse{ComplexF64}
    @test qr(A) \ [1.0, 2.0] ≈ Matrix(A) \ [1.0, 2.0]
    @test rank(A) == 2
    end
    B = sparse([1 2; 3 4])
    F = qr(B; ordering=SPQR.ORDERING_NATURAL)
    @test F isa SPQR.QRSparse{Float64} && F.pcol == [1, 2]
    @test_throws ArgumentError qr(SparseMatrixCSC{Float64, Int16}(B); ordering=SPQR.ORDERING_NATURAL)
end

@testset "propertynames of QRSparse" begin
    A = sparse([0.0 1 0 0; 0 0 0 0])
    F = qr(A)
    @test propertynames(F) == (:R, :Q, :prow, :pcol)
    @test propertynames(F, true) == (:R, :Q, :prow, :pcol, :factors, :τ, :cpiv, :rpivinv, :_lock)
end

@testset "rank" begin
    # each factor holds a diagonal block of order 5, so the product has rank 5 exactly
    S = sparse([1:10; 6:10], [1:5; [2, 1, 4, 5, 3]; 1:5], collect(1.0:15), 10, 5) *
        sparse([1:5; 1:5; 1:5], [1:5; 6:10; 10:-1:6], collect(1.0:15), 5, 10)
    @test rank(qr(S; tol=1e-5)) == 5
    @test rank(S; tol=1e-5) == 5
    @test all(iszero, (rank(qr(spzeros(10, i))) for i in 1:10))
    @test all(iszero, (rank(spzeros(10, i)) for i in 0:10))
    @test size(qr(spzeros(3, 0))) == (3, 0)
    @test qr(spzeros(3, 0)) \ ones(3) == zeros(0)
end


@testset "sparse" begin
    # one off-diagonal entry in each row, smaller than the diagonal: nonsingular
    A = I + sparse(1:100, [51:100; 1:50], fill(0.5, 100), 100, 100)
    q = qr(A; ordering=SPQR.ORDERING_FIXED)
    Q = q.Q
    sQ = sparse(Q)
    @test mismatch(sQ, Matrix(Q)) === nothing
    Dq = qr(Matrix(A))
    @static if COMPREHENSIVE
    perm = inv(Matrix(I, size(A)...)[q.prow, :])
    f = sum(q.R; dims=2) ./ sum(Dq.R; dims=2)
    @test perm * (transpose(f) .* sQ) ≈ sparse(Dq.Q)
    end
    v, V = fixturevec(Float64, 100), fixture(Float64, 100, 100)
    @test Dq.Q * v ≈ Matrix(Dq.Q) * v
    @test Dq.Q * V ≈ Matrix(Dq.Q) * V
    @static if COMPREHENSIVE
    @test Dq.Q * V' ≈ Matrix(Dq.Q) * V'
    end
    @test V * Dq.Q ≈ V * Matrix(Dq.Q)
    @static if COMPREHENSIVE
    @test V' * Dq.Q ≈ V' * Matrix(Dq.Q)
    end
end

@testset "ldiv!" begin
    @testset "workspace reuse" begin
        A = fullrank(Float64, m, n)
        F = qr(A)
        b = collect(1.0:m)
        x = zeros(n)

        # without a workspace each call allocates its own
        ldiv!(x, F, b)
        @test x ≈ Array(A) \ b
        @test @allocated(ldiv!(x, F, b)) > 0

        # a caller-provided workspace is resized on first use and then reused
        ws = SPQR.SpqrWS(F)
        ldiv!(x, F, b; workspace = ws)
        @test !isempty(ws.w)
        b2 = collect(m:-1.0:1)
        ldiv!(x, F, b2; workspace = ws)
        @test x ≈ Array(A) \ b2
        @test @allocated(ldiv!(x, F, b2; workspace = ws)) == 0
    end

    @testset "dimension errors" begin
        A = fixture(Float64, m, n)
        F = qr(A)
        @test_throws DimensionMismatch ldiv!(zeros(n), F, zeros(m - 1))
        @test_throws DimensionMismatch ldiv!(zeros(n - 1), F, zeros(m))
        @test_throws DimensionMismatch ldiv!(zeros(n, 2), F, zeros(m, 3))
        @test_throws DimensionMismatch ldiv!(zeros(m), F', zeros(n - 1))
        @test_throws DimensionMismatch ldiv!(zeros(m - 1), F', zeros(n))
        @test_throws DimensionMismatch ldiv!(zeros(m, 2), F', zeros(n, 3))
        X = fill(7.0, n, 2)
        @test_throws DimensionMismatch ldiv!(X, F, zeros(m))
        @test all(==(7.0), X)
        X = fill(7.0, m, 2)
        @test_throws DimensionMismatch ldiv!(X, F', zeros(n))
        @test all(==(7.0), X)
        # A' is overdetermined when A is wide, which needs a factorization of A'
        @test_throws DimensionMismatch qr(fixture(Float64, n, m))' \ zeros(m)
    end

    @testset "aliased X and B" begin
        # B is gathered into the workspace before X is written, so X may alias B
        F = qr(fullrank(Float64, n, n))
        b = collect(1.0:n)
        x = F \ b
        @test ldiv!(b, F, b) == x
    end

    @testset "copying QRSparse" begin
        A = fixture(Float64, m, n)
        F = qr(A)
        F_copy = copy(F)

        # The lock must not be shared
        @test F._lock !== F_copy._lock
    end

    @testset "solves take the lock of F" begin
        A = fullrank(Float64, m, n)
        F = qr(A)
        b, c = collect(1.0:m), collect(1.0:n)
        x, y = F \ b, F' \ c
        calls = (() -> ldiv!(zeros(n), F, b) ≈ x,
                 () -> F \ b ≈ x,
                 (@static COMPREHENSIVE ? (() -> F \ complex.(b) ≈ x,) : ())...,
                 () -> ldiv!(zeros(m), F', c) ≈ y,
                 (@static COMPREHENSIVE ? (() -> F' \ c ≈ y,) : ())...)
        for f in calls
            lock(F._lock)
            t = @async f()
            yield()
            @test !istaskdone(t)
            # copying F and solving with the copy do not wait for the lock of F
            @test fetch(@async copy(F) \ b ≈ x && copy(F)' \ c ≈ y)
            unlock(F._lock)
            @test timedwait(() -> istaskdone(t), 60; pollint=0.001) === :ok
            @test fetch(t)
        end
    end

    @testset "copy and deepcopy are independent" begin
        A = fullrank(Float64, m, n)
        b, c = collect(1.0:m), collect(1.0:n)
        F = qr(A)
        x, y = F \ b, F' \ c
        G, H = copy(F), deepcopy(F)
        SparseArrays.nonzeros(F.R) .*= 2
        @test G \ b ≈ x
        @test H \ b ≈ x
        SparseArrays.nonzeros(F.R) ./= 2
        @test copy(F') \ c ≈ y
    end
end

@testset "no strategies" begin
    A = I + fixture(Float64, 10, 10)
    for i in (ColumnNorm, (@static COMPREHENSIVE ? (RowMaximum, NoPivot) : ())...)
        @test_throws ErrorException qr(A, i())
    end
end

end # module
