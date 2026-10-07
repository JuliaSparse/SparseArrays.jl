# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseSupernodalTests

using Test
using SparseArrays
using SparseArrays: Supernodal
using SparseArrays.Supernodal: supernodal_lu
using LinearAlgebra
include("testhelpers.jl")

# A k²×k² five-point grid with unsymmetric values (complex ones for a complex `T`), whose
# pattern is symmetric and diagonal dominant, so that no pivot is perturbed.
function grid(::Type{T}, k) where {T}
    o = T <: Complex ? T(0.3, 0.2) : T(0.3)
    D = spdiagm(-1 => fill(-1 + o, k - 1), 0 => fill(T(4), k), 1 => fill(-1 - o, k - 1))
    E = spdiagm(-1 => fill(T(-1), k - 1), 1 => fill(T(-1) + o, k - 1))
    return kron(sparse(T(1)I, k, k), D) + kron(E, sparse(T(1)I, k, k))
end

# `A` with its rows shifted cyclically, so that no diagonal entry is stored and the
# factorization needs the matching.
shifted(A) = A[circshift(1:size(A, 1), 1), :]

rhsvalue(::Type{T}, i, j) where {T} = T <: Complex ? T(sin(i + 3j), cos(2i - j)) : T(sin(i + 3j))
rhs(::Type{T}, n) where {T} = T[rhsvalue(T, i, 1) for i in 1:n]
rhs(::Type{T}, n, m) where {T} = T[rhsvalue(T, i, j) for i in 1:n, j in 1:m]

function check_solves(@nospecialize(A), @nospecialize(F))
    Ad = Matrix(A)
    T = eltype(A)
    b = rhs(T, size(A, 1))
    B = rhs(T, size(A, 1), 3)
    tol = sqrt(eps(real(T)))
    @test F \ b ≈ Ad \ b rtol = tol
    @test F \ B ≈ Ad \ B rtol = tol
    @test transpose(F) \ b ≈ transpose(Ad) \ b rtol = tol
    @test F' \ B ≈ Ad' \ B rtol = tol
    x = copy(b)
    @test ldiv!(F, x) === x
    @test x ≈ Ad \ b rtol = tol
    @test logabsdet(F)[1] ≈ logabsdet(Ad)[1] rtol = tol
    @test logabsdet(F)[2] ≈ logabsdet(Ad)[2] rtol = tol
    @test (F.Rs .* A)[F.p, F.q] ≈ F.L * F.U rtol = tol
end

@testset "supernodal LU, $T" for T in (STD_ELTYPES..., (@static COMPREHENSIVE ? (Float32, BigFloat) : ())...)
    A = grid(T, 12)
    F = supernodal_lu(A)
    @test F isa Factorization{T}
    @test issuccess(F)
    @test size(F) == size(A)
    check_solves(A, F)
    @test_throws DimensionMismatch F \ ones(T, size(A, 1) + 1)
end

@testset "matching, $T" for T in STD_ELTYPES
    A = shifted(grid(T, 10))
    F = supernodal_lu(A)
    @test F.matched
    check_solves(A, F)
end

@testset "wide panels" begin
    # A dense 300×300 corner gives panels wide enough for the blocked panel factorization
    # and the BLAS supernode updates.
    A = grid(Float64, 20)
    A[1:300, 1:300] .+= [cos(i * j) for i in 1:300, j in 1:300]
    A += 300I
    F = supernodal_lu(A)
    @test maximum(diff(F.xsup)) > 32
    check_solves(A, F)
end

@testset "real factorization, complex right-hand side" begin
    A = grid(Float64, 6)
    F = supernodal_lu(A)
    b = rhs(ComplexF64, size(A, 1))
    @test F \ b ≈ Matrix(A) \ b
    x = copy(b)
    @test ldiv!(F, x) ≈ Matrix(A) \ b
    @test transpose(F) \ b ≈ transpose(Matrix(A)) \ b
end

@testset "pivoting off the diagonal" begin
    # without the matching, the tiny diagonal entry cannot be a pivot
    A = sparse([1e-12 1 0; 1 1 1; 0 1 2])
    F = supernodal_lu(A; matching = false)
    @test F.p != F.q
    check_solves(A, F)
end

@testset "singular" begin
    A = sparse([1.0 1 0; 1 1 0; 0 0 1])
    @test_throws SingularException supernodal_lu(A)
    F = supernodal_lu(A; check = false)
    @test !issuccess(F)
    @test det(F) == 0
    @test_throws SingularException F \ [1.0, 2, 3]
    @test_throws SingularException supernodal_lu(sparse([1.0 0 0; 0 0 0; 0 0 1]))
    @test_throws SingularException supernodal_lu(sparse([1.0 2; 0 0]))
end

@testset "lu!" begin
    A = grid(Float64, 8)
    b = rhs(Float64, size(A, 1))
    F = supernodal_lu(A)
    G = copy(F)
    A2 = copy(A)
    nonzeros(A2) .*= 2
    @test lu!(F, A2) === F
    @test F \ b ≈ Matrix(A2) \ b
    @test G \ b ≈ Matrix(A) \ b
    A3 = shifted(A)
    lu!(F, A3)
    @test F.matched
    @test F \ b ≈ Matrix(A3) \ b
    A7 = grid(Float64, 7)
    lu!(F, A7)
    @test size(F) == size(A7)
    @test F \ ones(size(A7, 1)) ≈ Matrix(A7) \ ones(size(A7, 1))
    @test_throws SingularException lu!(F, spzeros(size(A)...))
    @test !issuccess(F)
end

@testset "lu, \\ and factorize, $T" for T in STD_ELTYPES
    A = grid(T, 6)
    Ad = Matrix(A)
    b = rhs(T, size(A, 1))
    @test lu(A) isa Supernodal.SupernodalLU{T}
    @test factorize(A) isa Supernodal.SupernodalLU{T}
    @test A \ b ≈ Ad \ b
    @test A' \ b ≈ Ad' \ b
    @test lu(transpose(A)) \ b ≈ transpose(Ad) \ b
    W = T <: Real ? Symmetric(A) : Hermitian(A)
    @test lu(W) isa Supernodal.SupernodalLU{T}
    @test lu(W) \ b ≈ Matrix(W) \ b
    S = copy(A)
    S[:, 2] = S[:, 1]
    @test_throws SingularException S \ b
end

@testset "rectangular, $shape" for (shape, m, n) in (("tall", 60, 25), ("wide", 25, 60))
    A = sparse([cos(i * j + i) for i in 1:m, j in 1:n] .* (rem.((1:m) .+ (1:n)', 3) .== 0)) +
        sparse(1:min(m, n), 1:min(m, n), 4.0, m, n)
    F = lu(A)
    @test F isa Supernodal.SupernodalLU{Float64}
    @test size(F) == (m, n)
    @test size(F.L) == (m, min(m, n))
    @test size(F.U) == (min(m, n), n)
    @test istril(F.L) && istriu(F.U)
    @test (F.Rs .* A)[F.p, F.q] ≈ F.L * F.U
    @test_throws DimensionMismatch F \ ones(m)
    # rank deficiency: a zero pivot for a tall matrix, a deferred column for a wide one
    D = copy(A)
    m > n ? (D[:, 3] .= 0) : (D[3, :] .= 0)
    dropzeros!(D)
    @test_throws SingularException lu(D)
    G = lu(D; check = false)
    @test !issuccess(G)
    @test (G.Rs .* D)[G.p, G.q] ≈ G.L * G.U
end

@testset "lu with a column order" begin
    A = grid(Float64, 6)
    n = size(A, 2)
    b = rhs(Float64, n)
    # the natural order of a banded matrix is already postordered
    @test lu(A; q = 1:n).q == 1:n
    @test lu(A; q = 0:(n - 1)).q == 1:n
    U = sparse([j >= i ? cos(i + 2j) : 0.0 for i in 1:n, j in 1:n]) + spdiagm(-1 => ones(n - 1)) + 4I
    for M in (A, U)
        F = lu(M; q = n:-1:1, control = zeros(20))
        @test isperm(F.q)
        @test F \ b ≈ Matrix(M) \ b
    end
    @test_throws DimensionMismatch lu(A; q = 1:(n - 1))
    @test_throws ArgumentError lu(A; q = fill(1, n))
end

@testset "unsymmetric strategy" begin
    # an upper triangle plus a subdiagonal: no off-diagonal entry has a stored transpose
    # but the subdiagonal ones
    n = 40
    A = sparse([j >= i ? cos(i + 2j) : 0.0 for i in 1:n, j in 1:n]) + spdiagm(-1 => ones(n - 1)) + 4I
    @test Supernodal._unsymmetric(A)
    @test !Supernodal._unsymmetric(grid(Float64, 6))
    # judged after a row permutation to a zero-free diagonal
    @test !Supernodal._unsymmetric(shifted(grid(Float64, 6)))
    @static if COMPREHENSIVE
    @test Supernodal.structural_matching(sparse([1.0 1 0; 1 1 0; 0 0 0])) === nothing
    end
    F = lu(A)
    @test !F.matched
    check_solves(A, F)
end

# The matrix of UMFPACK's umfpack_di_demo.c, with its solutions.
demo(::Type{T}) where {T} = sparse([1, 5, 2, 2, 3, 3, 1, 2, 3, 4, 5, 5], [1, 5, 1, 3, 2, 3, 2, 5, 4, 3, 2, 3],
    T[2, 1, 3, 4, -1, -3, 3, 6, 2, 1, 4, 2], 5, 5)

@testset "demo matrix, $T" for T in STD_ELTYPES
    A = demo(T)
    F = lu(A)
    @test det(F) ≈ det(Matrix(A))
    @test all(logabsdet(F) .≈ logabsdet(Matrix(A)))
    b = T[8, 45, -3, 3, 19]
    @test F \ b ≈ 1:5
    z = complex.(b)
    @test ldiv!(F, z) === z
    @test z ≈ 1:5
    @test ldiv!(similar(z), F, complex.(b)) ≈ 1:5
    b = T[8, 20, 13, 6, 17]
    @test F' \ b ≈ 1:5
    @test ldiv!(similar(z), F', complex.(b)) ≈ 1:5
    @test transpose(F) \ b ≈ 1:5
    @test lu(A') \ b ≈ 1:5
    @test lu(transpose(A)) \ b ≈ 1:5
    # element promotion and type inference
    @inferred F \ fill(1, 5)
end

@testset "empty matrices" begin
    for (m, n) in ((0, 0), (5, 0), (0, 5))
        F = lu(spzeros(m, n))
        @test size(F) == (m, n)
        @test size(F.L) == (m, min(m, n)) && size(F.U) == (min(m, n), n)
    end
end

@testset "Issue #15099, $Tin" for Tin in (ComplexF32, Int, (@static COMPREHENSIVE ? (Float32, Float16) : ())...)
    F = lu(sparse(fill(Tin(1), 1, 1)))
    Tout = float(Tin)
    @test F.p == F.q == [1]
    @test F.Rs == [1]
    @test mismatch(F.L, fill(Tout(1), 1, 1)) === nothing
    @test mismatch(F.U, fill(Tout(1), 1, 1)) === nothing
end

@testset "size and propertynames" begin
    F = lu(sparse(fill(2.0, 1, 1)))
    @test size(F) == (1, 1)
    @test size(F, 1) == 1 && size(F, 2) == 1 && size(F, 3) == 1
    @test_throws ArgumentError size(F, -1)
    @test propertynames(F) == (:L, :U, :p, :q, :Rs)
    @test :sn ∉ propertynames(F)
    @test :sn ∈ propertynames(F, true)
    @test occursin("SupernodalLU", sprint(show, MIME"text/plain"(), F))
    @test occursin("1×1", sprint(show, F))
end

@testset "aliased solution and right-hand side" begin
    A = sparse([2.0 1 0; 1 3 1; 0 1 4])
    F = lu(A)
    B = A * [1.0 2; 3 4; 5 6]
    @test ldiv!(view(B, :, 1), F, view(vec(B), 1:3)) ≈ [1, 3, 5]
    B = A * [1.0 2 3; 4 5 6; 7 8 9]
    @test ldiv!(view(B, :, 2:3), F, view(B, :, 1:2)) ≈ [1.0 2; 4 5; 7 8]
end

@testset "complex right-hand sides with a real factorization" begin
    N = 10
    A = N * I + fixture(Float64, N, N)
    X = zeros(ComplexF64, N, N)
    B = Matrix(fixture(ComplexF64, N, N))
    luA, lufA = lu(A), lu(Array(A))
    @test ldiv!(copy(X), luA, B) ≈ ldiv!(copy(X), lufA, B)
    @test ldiv!(X[:, 1], luA, B[:, 1]) ≈ ldiv!(copy(X), lufA, B)[:, 1]
    @static if COMPREHENSIVE
    @test ldiv!(copy(X), adjoint(luA), B) ≈ ldiv!(copy(X), adjoint(lufA), B)
    @test ldiv!(copy(X), transpose(luA), B) ≈ ldiv!(copy(X), transpose(lufA), B)
    end
end

@testset "singular matrix, $T" for T in (Float64, (@static COMPREHENSIVE ? (ComplexF64,) : ())...)
    A = sparse(T[1 2; 0 0])
    @test_throws SingularException lu(A)
    @test !issuccess(lu(A; check = false))
end

@testset "refactorization, reuse_symbolic = $reuse" for reuse in (true, false)
    A = demo(Float64)
    B = copy(A)
    nonzeros(B)[8] = 9
    b = [8.0, 45, -3, 3, 19]
    F = lu(A)
    lu!(F, B; reuse_symbolic = reuse)
    @test F \ b ≈ Matrix(B) \ b
    C = copy(B)
    C[4, 3] = 0
    @test_throws SingularException lu!(lu(A), C; reuse_symbolic = reuse)
    # a new nonzero pattern is analyzed afresh
    D = copy(B)
    D[5, 1] = 1
    F = lu(A)
    lu!(F, D; reuse_symbolic = reuse)
    @test F \ b ≈ Matrix(D) \ b
end

@testset "F.Rs and logabsdet of a badly scaled matrix" begin
    A = sparse([1e-20 2e-20 0; 0 1 3; 1 0 1])
    F = lu(A)
    @test F.L * F.U ≈ (F.Rs .* A)[F.p, F.q]
    @test all(logabsdet(F) .≈ logabsdet(Matrix(A)))
    @test det(F) ≈ det(Matrix(A))
    @static if COMPREHENSIVE
    B = 1e-15 * (fixture(Float64, 50, 50) + 50I)
    @test all(logabsdet(lu(B)) .≈ logabsdet(Matrix(B)))
    end
end

@testset "lu!(F, S) validates S before mutating F" begin
    A = sparse([4.0 1; 1 3])
    F = lu(A)
    L, U = F.L, F.U
    @test_throws ArgumentError lu!(F, sparse(ComplexF64[4 1 0; 1 3 0; 0 0 1im]))
    @test_throws ArgumentError lu!(F, A; q = [1, 1])
    @test size(F) == (2, 2)
    @test F.L == L && F.U == U
    @test F \ [1.0, 2.0] ≈ Matrix(A) \ [1.0, 2.0]
    # integer inputs convert, and the factorization follows a new size
    C = sparse([4 1 0; 1 3 0; 0 0 1])
    lu!(F, C)
    @test size(F) == (3, 3)
    @test F \ [1.0, 2.0, 3.0] ≈ Matrix(C) \ [1.0, 2.0, 3.0]
    Fc = lu(sparse(ComplexF64[4 1; 1 3]))
    lu!(Fc, C)
    @test Fc \ ComplexF64[1, 2, 3] ≈ Matrix(C) \ ComplexF64[1, 2, 3]
end

@testset "column orders with converted eltypes, $T" for T in (Float32, (@static COMPREHENSIVE ? (ComplexF32, Int) : ())...)
    A = sparse([4.0 1 0; 1 4 1; 0 1 4])
    b = [1.0, 2.0, 3.0]
    x = Matrix(A) \ b
    S = SparseMatrixCSC{T}(A)
    @test lu(S; q = [3, 2, 1]) \ b ≈ x
    @test lu(S; q = 3:-1:1) \ b ≈ x
    @static if COMPREHENSIVE
    @test lu(S; q = Int32[2, 1, 0]) \ b ≈ x
    end
    # an odd permutation
    F = lu(S; q = [1, 3, 2])
    @test all(logabsdet(F) .≈ logabsdet(Matrix(A)))
    @test lu(S; control = zeros(20)) \ b ≈ x
end

@testset "non-square det and \\ throw DimensionMismatch, $T, $m×$n" for
        T in STD_ELTYPES, (m, n) in FIXTURE_SHAPES[1:2]
    A = fixture(T, m, n)
    # the fixture is rank deficient, which lu reports and still factorizes
    F = lu(A; check = false)
    @test !issuccess(F)
    @test_throws DimensionMismatch det(F)
    @test_throws DimensionMismatch F \ ones(m)
    @test size(F.L) == (m, min(m, n)) && size(F.U) == (min(m, n), n)
    @test F.L * F.U ≈ (F.Rs .* A)[F.p, F.q]
end

@testset "ldiv! DimensionMismatch names the sizes and leaves the output unchanged" begin
    F = lu(sparse([4.0 1 0; 1 4 1; 0 1 4]))
    X = fill(7.0, 3)
    @test_throws DimensionMismatch ldiv!(X, F, [1.0, 2])
    @test_throws r"3×3.*2 rows" ldiv!(X, F, [1.0, 2])
    @test_throws r"\(3,\).*\(3, 1\)" ldiv!(X, F, reshape([1.0, 2, 3], 3, 1))
    @test X == fill(7.0, 3)
end

@testset "ldiv! with strided and adjoint/transpose right-hand sides, $T" for T in STD_ELTYPES
    A = sparse(T[4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4.5])
    F = lu(A)
    Ad = Matrix(A)
    w = T.(collect(1.0:8.0))
    v = view(w, 1:2:8)
    @test ldiv!(F, v) ≈ Ad \ T.(1:2:8)
    @test w[2:2:8] == 2:2:8
    @test ldiv!(zeros(T, 4), F, view(T.(collect(1.0:8.0)), 1:2:8)) ≈ Ad \ T.(1:2:8)
    # a matrix and a right-hand side that differ from their conjugates tell the transpose
    # from the adjoint solve, and check the imaginary part of the determinant
    Ac = fixture(ComplexF64, 4, 4) + 5I
    Fc = lu(Ac)
    bc = complex.(1.0:4.0, 4.0:-1.0:1.0)
    @test transpose(Ac) * ldiv!(zeros(ComplexF64, 4), transpose(Fc), bc) ≈ bc
    @test Ac' * ldiv!(zeros(ComplexF64, 4), Fc', bc) ≈ bc
    @test det(Fc) ≈ det(Matrix(Ac))
    M = T.(reshape(1.0:24.0, 8, 3))
    Y = zeros(T, 8, 3)
    ldiv!(view(Y, 1:2:8, :), transpose(F), view(M, 2:2:8, :))
    @test Y[1:2:8, :] ≈ transpose(Ad) \ M[2:2:8, :]
    @test iszero(Y[2:2:8, :])
    B = T.(reshape(1.0:12.0, 3, 4))
    Bw = adjoint(copy(B))
    @test ldiv!(F, Bw) === Bw
    @test Bw ≈ Ad \ B'
end

@static if COMPREHENSIVE
@testset "complex demo matrix" begin
    A = demo(Float64)
    Ac = complex.(A, A)
    x = fill(1.0 + im, 5)
    F = lu(Ac)
    @test F.L * F.U ≈ (F.Rs .* Ac)[F.p, F.q]
    @test Ac \ (Ac * x) ≈ x
    @test Ac' \ (Ac' * x) ≈ x
    @test transpose(Ac) \ (transpose(Ac) * x) ≈ x
end

@testset "Issue #4523 - complex sparse \\" begin
    A, b = sparse((1.0 + im)I, 2, 2), fill(1.0, 2)
    @test A * (lu(A) \ b) ≈ b
    @test det(sparse([1, 3, 3, 1], [1, 1, 3, 3], [1, 1, 1, 1])) == 0
end

@testset "Issues #18246, #18244 - lu sparse pivot" begin
    A = sparse(1.0I, 4, 4)
    A[1:2, 1:2] = [-0.01 -200; 200 0.001]
    F = lu(A)
    @test F.L * F.U ≈ (F.Rs .* A)[F.p, F.q]
    @test A * (F \ ones(4)) ≈ ones(4)
end

@testset "complex BigFloat" begin
    A = sparse(Complex{BigFloat}[2 1im; -1im 3])
    @test lu(A) \ Complex{BigFloat}[1, 2] ≈ Matrix(A) \ Complex{BigFloat}[1, 2]
end

@testset "lu with a custom column order" begin
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
        b = Float64.((1:10) .== i)
        x = lu(A) \ b
        @test lu(A; q = q0) \ b ≈ x
        @test lu(A; q = q1) \ b ≈ x
    end
end
end

@static if COMPREHENSIVE
@testset "supernodal LU, corner cases" begin
    F = supernodal_lu(spzeros(0, 0))
    @test issuccess(F)
    @test det(F) == 1
    @test F \ Float64[] == Float64[]
    A = SparseMatrixCSC{Float64,Int32}(grid(Float64, 5))
    F = supernodal_lu(A)
    @test F \ ones(size(A, 1)) ≈ Matrix(A) \ ones(size(A, 1))
    @test eltype(supernodal_lu(sparse([2 1; 1 3]))) == Float64
    @test_throws SingularException supernodal_lu(sparse(ones(2, 3)))
    A = shifted(grid(ComplexF64, 6))
    F = supernodal_lu(A)
    @test F' \ ones(size(A, 1)) ≈ Matrix(A)' \ ones(size(A, 1))
    @test transpose(F) \ ones(size(A, 1)) ≈ transpose(Matrix(A)) \ ones(size(A, 1))
    @test det(F) ≈ det(Matrix(A))
    @test lu!(F, A; reuse_symbolic = false) \ ones(size(A, 1)) ≈ Matrix(A) \ ones(size(A, 1))
    # The matching's scales span more than Float32's range here; they are clamped so that
    # they stay finite in the element type.
    d = Float32[isodd(i) ? 1f15 : 1f-15 for i in 1:39]
    A = shifted(spdiagm(1 => d, -1 => ones(Float32, 39), 0 => fill(1f-3, 40)))
    F = supernodal_lu(A)
    @test F.matched
    @test eltype(F.Rs) == Float32
    @test all(isfinite, F.Rs)
    x = F \ ones(Float32, 40)
    @test norm(A * x - ones(Float32, 40)) <= sqrt(eps(Float32)) * opnorm(A, Inf) * norm(x)
end
end

end # module
