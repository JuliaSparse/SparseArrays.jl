# This file is a part of Julia. License is MIT: https://julialang.org/license

# An app that CI builds with `juliac --trim=safe` and runs. SparseArrays extends Base's
# concatenation for dense numeric arrays, so loading it must keep dense concatenation
# trimmable, as Julia's own trim test checks; the other checks cover the main sparse
# operations.
module SparseTrim

using LinearAlgebra
using Random
using SparseArrays

check(ok::Bool) = ok ? 0 : 1
near(x, y) = maximum(abs, x - y) < 1e-10

function dense_cat()
    v = [1.0, 2.0, 3.0]
    m = [1.0 2.0; 3.0 4.0]
    a = [2.0 3.0 4.0; 5.0 6.0 7.0; 8.0 9.0 1.0]
    bad = 0
    bad += check(size(hcat(v, v, v)) == (3, 3))
    bad += check(size(vcat(m, m, m, m, m, m)) == (12, 2))
    bad += check(size(hcat(m, m, m, m, m, m)) == (2, 12))
    bad += check(size([m m; m m]) == (4, 4))
    bad += check(size(hvcat((2, 2), m, m, m, m)) == (4, 4))
    bad += check(size(cat(v, v; dims = 1)) == (6,))
    bad += check(size(cat(a, a; dims = 2)) == (3, 6))
    bad += check(size(cat(v, v, v, v, v, v; dims = (1,))) == (18,))
    bad += check(size(cat(m, m, m, m, m, m; dims = (1, 2))) == (12, 12))
    return bad
end

function sparse_cat()
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    D = Matrix(A)
    bad = 0
    bad += check(hcat(A, D) isa SparseMatrixCSC{Float64,Int})
    bad += check(Matrix(vcat(A, A)) == vcat(D, D))
    B = [A D; D A]
    bad += check(B isa SparseMatrixCSC{Float64,Int} && B[4, 1] == 2.0 && B[6, 4] == 4.0 && nnz(B) == 16)
    bad += check(hvcat((2, 2), A, A, A, A) isa SparseMatrixCSC{Float64,Int})
    bad += check(size([A D A; D A D]) == (6, 9))
    A32 = SparseMatrixCSC{Float32,Int32}(A)
    bad += check([A A32; A32 A] isa SparseMatrixCSC{Float64,Int})
    v = sparsevec([1, 3], [0.0, 1.0], 3)
    E = [A v; 1.0 2.0 3.0 0.0]
    bad += check(E isa SparseMatrixCSC{Float64,Int} && size(E) == (4, 4) && nnz(E) == 9 && E[4, 3] == 3.0)
    x = [1.0, 2.0, 3.0]
    bad += check([A x; x' 1.0] isa SparseMatrixCSC{Float64,Int})
    C = cat(A, D; dims = (1, 2))
    bad += check(C isa SparseMatrixCSC{Float64,Int} && size(C) == (6, 6) && C[4, 4] == 2.0 && C[1, 4] == 0.0)
    bad += check(size(cat(A, D, A, D, A, D; dims = (1, 2))) == (18, 18))
    bad += check(cat(A, A; dims = 1) isa SparseMatrixCSC{Float64,Int})
    return bad
end

function construction()
    bad = 0
    A = sparse([1, 2, 3, 3, 1], [1, 2, 1, 3, 1], [2.0, 3.0, 4.0, 5.0, 1.0], 3, 3)
    bad += check(A[1, 1] == 3.0 && nnz(A) == 4)
    bad += check(Matrix(A) == [3.0 0.0 0.0; 0.0 3.0 0.0; 4.0 0.0 5.0])
    bad += check(sparse([3.0 0.0; 0.0 1.0]) == sparse([1, 2], [1, 2], [3.0, 1.0]))
    bad += check(nnz(spzeros(4, 5)) == 0 && size(spzeros(ComplexF64, 2, 3)) == (2, 3))
    R = sprand(Xoshiro(1), 20, 20, 0.2)
    bad += check(size(R) == (20, 20) && all(x -> 0 <= x < 1, nonzeros(R)))
    T = spdiagm(-1 => [-1.0, -1.0], 0 => [4.0, 4.0, 4.0], 1 => [-1.0, -1.0])
    bad += check(Matrix(T) == [4.0 -1.0 0.0; -1.0 4.0 -1.0; 0.0 -1.0 4.0])
    bad += check(sparse(1.0I, 3, 3) == spdiagm(0 => ones(3)))
    is, js, vs = findnz(A)
    bad += check(is == [1, 3, 2, 3] && js == [1, 1, 2, 3] && vs == [3.0, 4.0, 3.0, 5.0])
    C = sparse([1, 2], [2, 1], [1.0 + 2.0im, 3.0 - 1.0im], 2, 2)
    bad += check(Matrix(C') == [0.0 3.0+1.0im; 1.0-2.0im 0.0])
    bad += check(SparseMatrixCSC{Float32,Int32}(A) == A)
    return bad
end

function indexing()
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    bad = 0
    bad += check(A[3, 1] == 4.0 && A[1, 3] == 0.0)
    bad += check(A[:, 1] == sparsevec([1, 3], [2.0, 4.0], 3))
    bad += check(A[2:3, :] == sparse([2, 1, 2], [1, 2, 3], [4.0, 3.0, 5.0], 2, 3))
    bad += check(A[[3, 1], [1, 3]] == sparse([1, 2, 1], [1, 1, 2], [4.0, 2.0, 5.0], 2, 2))
    bad += check(A[A .> 2.5] == [4.0, 3.0, 5.0])
    B = copy(A)
    B[1, 3] = 7.0
    B[2, 2] = 0.0
    bad += check(B[1, 3] == 7.0 && nnz(B) == 5 && nnz(A) == 4)
    dropzeros!(B)
    bad += check(nnz(B) == 4 && B[2, 2] == 0.0)
    B[:, 2] .= 1.0
    bad += check(B[:, 2] == [1.0, 1.0, 1.0])
    bad += check(Array(view(A, :, 3)) == [0.0, 0.0, 5.0])
    return bad
end

function algebra()
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    B = sparse([1, 3], [3, 1], [1.0, -1.0], 3, 3)
    D = Matrix(A)
    x = [1.0, 1.0, 1.0]
    bad = 0
    # a trimmed binary cannot load BLAS, so the expected values are written out rather
    # than computed with dense products
    bad += check(A * x == [2.0, 3.0, 9.0])
    bad += check(A' * x == [6.0, 3.0, 5.0])
    bad += check(transpose(A) * x == [6.0, 3.0, 5.0])
    bad += check(Matrix(A + A) == 2 .* D)
    bad += check((A .+ B) isa SparseMatrixCSC && Matrix(A .+ B) == D + Matrix(B))
    bad += check(2 .* A == A + A && A .* B == sparse([3], [1], [-4.0], 3, 3))
    bad += check(abs.(B) == sparse([1, 3], [3, 1], [1.0, 1.0], 3, 3))
    bad += check(map(x -> 2x, A) == 2A)
    bad += check(A - B == sparse([1, 2, 3, 3, 1], [1, 2, 1, 3, 3], [2.0, 3.0, 5.0, 5.0, -1.0], 3, 3))
    bad += check(A * B == sparse([1, 3, 3], [3, 1, 3], [2.0, -5.0, 4.0], 3, 3))
    bad += check(Matrix(A * D) == [4.0 0.0 0.0; 0.0 9.0 0.0; 28.0 0.0 25.0])
    bad += check(sum(A) == 14.0 && sum(A; dims = 1) == [6.0 3.0 5.0])
    bad += check(sum(A; dims = 2) == reshape([2.0, 3.0, 9.0], 3, 1))
    bad += check(maximum(A) == 5.0 && minimum(B) == -1.0 && prod(A) == 0.0)
    bad += check(norm(A, 1) == 14.0 && norm(A, Inf) == 5.0 && abs(norm(A) - sqrt(54.0)) < 1e-12)
    bad += check(opnorm(A, 1) == 6.0 && opnorm(A, Inf) == 9.0)
    bad += check(dot(A, A) == 54.0 && dot(x, A, x) == 14.0)
    bad += check(tr(A) == 10.0 && diag(A) == [2.0, 3.0, 5.0])
    C = sparse([1, 2], [1, 2], [1.0im, 2.0], 2, 2)
    bad += check(C * [1.0, 1.0im] == [1.0im, 2.0im] && Matrix(C') == [-1.0im 0.0; 0.0 2.0])
    return bad
end

function structure()
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    E = sparse([1, 2], [2, 1], [1.0, 1.0], 2, 2)
    bad = 0
    bad += check(copy(transpose(A)) == sparse([1, 2, 1, 3], [1, 2, 3, 3], [2.0, 3.0, 4.0, 5.0], 3, 3))
    bad += check(permute(A, [3, 2, 1], [1, 2, 3]) == sparse([3, 2, 1, 1], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3))
    K = kron(E, A)
    bad += check(size(K) == (6, 6) && nnz(K) == 8 && K[6, 1] == 4.0 && K[1, 4] == 2.0)
    Bd = blockdiag(A, E)
    bad += check(size(Bd) == (5, 5) && nnz(Bd) == 6 && Bd[4, 5] == 1.0)
    bad += check(triu(A) == sparse([1, 2, 3], [1, 2, 3], [2.0, 3.0, 5.0], 3, 3))
    bad += check(tril(A, -1) == sparse([3], [1], [4.0], 3, 3))
    bad += check(issymmetric(E) && !issymmetric(A) && ishermitian(E) && istril(A) && !istriu(A))
    bad += check(diag(A, -2) == [4.0])
    bad += check(fkeep!((i, j, v) -> v > 2.5, copy(A)) == sparse([2, 3, 3], [2, 1, 3], [3.0, 4.0, 5.0], 3, 3))
    return bad
end

function vectors()
    x = sparsevec([1, 3, 5], [1.0, 2.0, 3.0], 6)
    y = sparsevec([3, 4], [4.0, -1.0], 6)
    bad = 0
    bad += check(nnz(x) == 3 && x[3] == 2.0 && x[2] == 0.0)
    bad += check(x + y == sparsevec([1, 3, 4, 5], [1.0, 6.0, -1.0, 3.0], 6))
    bad += check(2 .* x == x + x && (x .* y) == sparsevec([3], [8.0], 6))
    bad += check(dot(x, y) == 8.0 && sum(x) == 6.0 && maximum(y) == 4.0)
    bad += check(norm(y, 1) == 5.0 && abs(norm(x) - sqrt(14.0)) < 1e-12)
    bad += check(Vector(x) == [1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
    bad += check(sparse([1.0, 0.0, 2.0]) == sparsevec([1, 3], [1.0, 2.0], 3))
    bad += check(x[2:5] == sparsevec([2, 4], [2.0, 3.0], 4))
    is, vs = findnz(y)
    bad += check(is == [3, 4] && vs == [4.0, -1.0])
    z = copy(x)
    z[2] = 5.0
    bad += check(nnz(z) == 4 && z[2] == 5.0)
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    v = sparsevec([1, 3], [1.0, 1.0], 3)
    bad += check(A * v == sparsevec([1, 3], [2.0, 9.0], 3))
    c = sparsevec([2], [1.0im], 3)
    bad += check(dot(c, c) == 1.0 && Vector(conj(c)) == [0.0, -1.0im, 0.0])
    return bad
end

function eltypes()
    bad = 0
    Ai = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2, 3, 4, 5], 3, 3)
    bad += check(Ai * [1, 1, 1] == [2, 3, 9] && Ai + Ai == 2Ai && Ai - Ai == spzeros(Int, 3, 3))
    bad += check(sum(Ai) == 14 && sum(Ai; dims = 2) == reshape([2, 3, 9], 3, 1) && maximum(Ai) == 5)
    bad += check(sum(Ai .* Ai) == 54 && Ai * Ai == sparse([1, 3, 2, 3], [1, 1, 2, 3], [4, 28, 9, 25], 3, 3))
    M = Ai .> 2
    bad += check(M isa SparseMatrixCSC{Bool,Int} && nnz(M) == nnz(Ai) && count(M) == 3 && Ai[M] == [4, 3, 5])
    Bi = copy(Ai)
    Bi[M] .= 1
    bad += check(sum(Bi) == 5 && Bi[3, 3] == 1)
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    A32 = SparseMatrixCSC{Float64,Int32}(A)
    AA = sparse([1, 3, 2, 3], [1, 1, 2, 3], [4.0, 28.0, 9.0, 25.0], 3, 3)
    bad += check(A32 * [1.0, 1.0, 1.0] == [2.0, 3.0, 9.0])
    bad += check(A32 * A32 isa SparseMatrixCSC{Float64,Int32} && A32 * A32 == AA)
    bad += check((A32 .+ A32) isa SparseMatrixCSC{Float64,Int32} && A32 .+ A32 == 2A)
    bad += check(A32[2:3, :] isa SparseMatrixCSC{Float64,Int32} && A32[2:3, :] == A[2:3, :])
    bad += check(A32[:, 1] isa SparseVector{Float64,Int32} && A32[3, 1] == 4.0)
    bad += check(hcat(A32, A32) isa SparseMatrixCSC{Float64,Int32} && [A32; A32] == [A; A])
    v32 = sparsevec(Int32[1, 3], [1.0, 2.0], 3)
    bad += check(A32 * v32 == sparsevec([1, 3], [2.0, 14.0], 3) && dot(v32, v32) == 5.0)
    bad += check((v32 .+ v32) isa SparseVector{Float64,Int32} && sum(v32 .* 3) == 9.0)
    F = SparseMatrixCSC{Float32,Int}(A)
    bad += check(F * Float32[1, 1, 1] == Float32[2, 3, 9] && F * F == SparseMatrixCSC{Float32,Int}(AA))
    bad += check(sum(F) == 14.0f0 && eltype(2 .* F) == Float32 && norm(F, 1) == 14.0f0)
    C = SparseMatrixCSC{ComplexF32,Int}(A) .* im
    bad += check(C isa SparseMatrixCSC{ComplexF32,Int} && sum(C) == 14.0f0im)
    bad += check(C * ComplexF32[1, 1, 1] == ComplexF32[2im, 3im, 9im])
    bad += check(C' * ComplexF32[1, 1, 1] == ComplexF32[-6im, -3im, -5im])
    bad += check(abs.(C) == F && eltype(abs.(C)) == Float32)
    return bad
end

function wrappers()
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    bad = 0
    U = sparse([1, 1, 2], [1, 2, 2], [2.0, 1.0, 3.0], 2, 2)
    bad += check(Symmetric(U) * [1.0, 1.0] == [3.0, 4.0] && sparse(Symmetric(U)) == sparse([2.0 1.0; 1.0 3.0]))
    H = sparse([1, 1, 2], [1, 2, 2], [2.0 + 0im, 1.0im, 3.0 + 0im], 2, 2)
    bad += check(Hermitian(H) * ComplexF64[1, 1] == [2.0 + 1.0im, 3.0 - 1.0im])
    bad += check(sparse(Hermitian(H)) == sparse(ComplexF64[2 im; -im 3]))
    Dg = Diagonal([1.0, 2.0, 3.0])
    bad += check(Dg * A isa SparseMatrixCSC && Matrix(Dg * A) == [2.0 0 0; 0 6 0; 12 0 15])
    bad += check(A * Dg isa SparseMatrixCSC && Matrix(A * Dg) == [2.0 0 0; 0 6 0; 4 0 15])
    Bd = Bidiagonal([1.0, 1.0, 1.0], [1.0, 1.0], :U)
    bad += check(Matrix(Bd * A) == [2.0 3 0; 4 3 5; 4 0 5] && Matrix(A * Bd) == [2.0 2 0; 0 3 3; 4 4 5])
    Td = Tridiagonal([1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0])
    bad += check(Matrix(Td * A) == [2.0 3 0; 6 3 5; 4 3 5] && Matrix(A * Td) == [2.0 2 0; 3 3 3; 4 9 5])
    V = view(A, :, 2:3)
    bad += check(V * [1.0, 1.0] == [0.0, 3.0, 5.0] && sum(V) == 8.0 && Matrix(V .* 2) == [0.0 0; 6 0; 0 10])
    c = view(A, :, 1)
    bad += check(dot(c, [1.0, 1.0, 1.0]) == 6.0 && sum(c) == 6.0 && maximum(view(A, :, 3)) == 5.0)
    bad += check(A' * view([1.0, 1.0, 1.0, 1.0], 2:4) == [6.0, 3.0, 5.0])
    x = sparsevec([1, 3, 5], [1.0, 2.0, 3.0], 6)
    bad += check(sum(view(x, 2:5)) == 5.0 && Vector(view(x, 3:4) .* 2) == [4.0, 0.0])
    Fx = SparseArrays.FixedSparseCSC(copy(A))
    Fx[1, 1] = 10.0
    bad += check(Fx[1, 1] == 10.0 && Fx * [1.0, 1.0, 1.0] == [10.0, 3.0, 9.0] && sum(Fx) == 22.0)
    threw = try
        Fx[1, 2] = 1.0
        false
    catch
        true
    end
    bad += check(threw && nnz(Fx) == 4)
    return bad
end

function triangular()
    # lower triangular, with A * [1, 2, 3] == [2, 6, 19] and A' * [1, 2, 3] == [14, 6, 15]
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    b = [2.0, 6.0, 19.0]
    bt = [14.0, 6.0, 15.0]
    x = [1.0, 2.0, 3.0]
    bad = 0
    bad += check(near(LowerTriangular(A) \ b, x) && near(UpperTriangular(copy(A')) \ bt, x))
    bad += check(near(LowerTriangular(A)' \ bt, x) && near(transpose(LowerTriangular(A)) \ bt, x))
    bad += check(near(UpperTriangular(A') \ bt, x))
    y = zeros(3)
    ldiv!(y, LowerTriangular(A), b)
    bad += check(near(y, x))
    z = copy(b)
    ldiv!(LowerTriangular(A), z)
    bad += check(near(z, x))
    bad += check(near(LowerTriangular(A) \ [2.0 2.0; 6.0 6.0; 19.0 19.0], [1.0 1.0; 2.0 2.0; 3.0 3.0]))
    bad += check(near(UnitLowerTriangular(A) \ [1.0, 2.0, 7.0], x))
    bad += check(near(LowerTriangular(A) \ sparsevec([1], [2.0], 3), [1.0, 0.0, -0.8]))
    return bad
end

# a sparse right-hand side: the structure functions, the LU factorization written in Julia
# and the sparse solution of `\`
function sparse_rhs()
    # not triangular, but a permutation of a triangular matrix, and N * [1, 2, 3, 4] == [4, 4, 6, 7]
    N = sparse([1, 2, 1, 3, 2, 4, 4], [1, 1, 2, 2, 3, 3, 4], [2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0], 4, 4)
    c = sparsevec([1, 2, 3, 4], [4.0, 4.0, 6.0, 7.0], 4)
    x = [1.0, 2.0, 3.0, 4.0]
    bad = 0
    d = dmperm(N)
    bad += check(sprank(N) == 4 && isperm(d.p) && length(d.colblocks) == 5)
    F = SparseArrays.sparselu(N)
    bad += check(near(F.L * F.U, N[F.p, F.q]))
    bad += check(near(F \ Vector(c), x) && near(F \ c, x))
    y = N \ c
    bad += check(y isa SparseVector{Float64,Int} && near(y, x))
    Y = N \ sparse(reshape(Vector(c), 4, 1))
    bad += check(Y isa SparseMatrixCSC{Float64,Int} && near(Y, reshape(x, 4, 1)))
    bad += check(near(N' \ sparsevec([1, 2, 3, 4], [4.0, 10.0, 6.0, 4.0], 4), x))
    return bad
end

function inplace()
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    x = [1.0, 1.0, 1.0]
    bad = 0
    y = zeros(3)
    mul!(y, A, x)
    bad += check(y == [2.0, 3.0, 9.0])
    mul!(y, A, x, 2.0, 1.0)
    bad += check(y == [6.0, 9.0, 27.0])
    Y = zeros(3, 2)
    mul!(Y, A, [1.0 0.0; 0.0 1.0; 1.0 1.0])
    bad += check(Y == [2.0 0.0; 0.0 3.0; 9.0 5.0])
    Z = zeros(2, 3)
    mul!(Z, [1.0 0.0 1.0; 0.0 1.0 1.0], A)
    bad += check(Z == [6.0 0.0 5.0; 4.0 3.0 5.0])
    B = copy(A)
    lmul!(2.0, B)
    bad += check(B == 2A)
    rmul!(B, 0.5)
    bad += check(B == A)
    lmul!(Diagonal([1.0, 2.0, 3.0]), B)
    bad += check(Matrix(B) == [2.0 0 0; 0 6 0; 12 0 15])
    bad += check(copyto!(zeros(3, 3), A) == Matrix(A) && copyto!(B, A) == A)
    fill!(view(B, :, 3), 1.0)
    bad += check(B[:, 3] == [1.0, 1.0, 1.0] && B[3, 1] == 4.0)
    bad += check(droptol!(copy(A), 2.5) == sparse([3, 2, 3], [1, 2, 3], [4.0, 3.0, 5.0], 3, 3))
    return bad
end

function colsums(A::SparseMatrixCSC)
    s = zeros(eltype(A), size(A, 2))
    w = zero(eltype(A))
    rows = rowvals(A)
    vals = nonzeros(A)
    for j in axes(A, 2), k in nzrange(A, j)
        s[j] += vals[k]
        w += rows[k] * vals[k]
    end
    return s, w
end

function kernels()
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    bad = 0
    s, w = colsums(A)
    bad += check(s == [6.0, 3.0, 5.0] && w == 35.0)
    s32, w32 = colsums(SparseMatrixCSC{Float64,Int32}(A))
    bad += check(s32 == [6.0, 3.0, 5.0] && w32 == 35.0)
    return bad
end

function search()
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    x = sparsevec([1, 3, 5], [1.0, 2.0, 3.0], 6)
    CI = CartesianIndex
    bad = 0
    bad += check(findall(!iszero, A) == [CI(1, 1), CI(3, 1), CI(2, 2), CI(3, 3)])
    bad += check(findall(A .> 2.5) == [CI(3, 1), CI(2, 2), CI(3, 3)])
    bad += check(findall(v -> v > 2.5, A) == [CI(3, 1), CI(2, 2), CI(3, 3)])
    bad += check(findmax(A) == (5.0, CI(3, 3)) && findmin(A) == (0.0, CI(2, 1)))
    bad += check(findmax(A; dims = 1) == ([4.0 3.0 5.0], [CI(3, 1) CI(2, 2) CI(3, 3)]))
    bad += check(findmin(A; dims = 2) == (reshape([0.0, 0.0, 0.0], 3, 1), reshape([CI(1, 2), CI(2, 1), CI(3, 2)], 3, 1)))
    bad += check(any(A .> 4.5) && all(A .>= 0) && count(!iszero, A) == 4 && argmax(A) == CI(3, 3))
    bad += check(findall(!iszero, x) == [1, 3, 5] && findmax(x) == (3.0, 5) && argmin(x) == 2)
    return bad
end

function reshaping()
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    x = sparsevec([1, 3, 5], [1.0, 2.0, 3.0], 6)
    bad = 0
    bad += check(Vector(vec(A)) == [2.0, 0, 4, 0, 3, 0, 0, 0, 5] && size(reshape(A, 1, 9)) == (1, 9))
    bad += check(reshape(A, 9, 1)[3, 1] == 4.0 && permutedims(A) == copy(transpose(A)))
    bad += check(Matrix(reverse(A; dims = 2)) == [0.0 0 2; 0 3 0; 5 0 4] && Vector(reverse(x)) == [0.0, 3, 0, 2, 0, 1])
    bad += check(Matrix(circshift(A, (1, 0))) == [4.0 0 5; 2 0 0; 0 3 0] && Vector(circshift(x, 1)) == [0.0, 1, 0, 2, 0, 3])
    bad += check(Matrix(diff(A; dims = 1)) == [-2.0 3 0; 4 -3 5])
    return bad
end

function broadcasting()
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    B = sparse([1, 3], [3, 1], [1.0, -1.0], 3, 3)
    x = sparsevec([1, 3, 5], [1.0, 2.0, 3.0], 6)
    y = sparsevec([3, 4], [4.0, -1.0], 6)
    bad = 0
    bad += check(Matrix(A .* 2 .+ 1) == [5.0 1 1; 1 7 1; 9 1 11])
    bad += check(Matrix(A .+ [1.0, 2.0, 3.0]) == [3.0 1 1; 2 5 2; 7 3 8])
    bad += check(Matrix(A .+ ones(3, 3)) == [3.0 1 1; 1 4 1; 5 1 6])
    bad += check(Matrix(((a, b, c) -> a * b + c).(A, B, A)) == [2.0 0 0; 0 3 0; 0 0 5])
    bad += check(Vector(x .+ y .* 2) == [1.0, 0, 10, -2, 3, 0] && count(A .== 0) == 5)
    return bad
end

function equality()
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    x = sparsevec([1, 3, 5], [1.0, 2.0, 3.0], 6)
    bad = 0
    bad += check(A == copy(A) && A == Matrix(A) && isequal(A, copy(A)) && A != 2A)
    bad += check(hash(A) == hash(Matrix(A)) && hash(x) == hash(Vector(x)) && isequal(x, Vector(x)))
    return bad
end

# A trimmed binary cannot yet load a `LazyLibrary`, which is how SuiteSparse is loaded, so
# `main` runs these checks only when given `solvers`. CI does not pass it: the build still
# verifies that the solvers trim, but they are not run.
function solvers()
    # tridiagonal(-1, 4, -1), with x = [1, 2, 3, 4]
    T = spdiagm(-1 => [-1.0, -1.0, -1.0], 0 => [4.0, 4.0, 4.0, 4.0], 1 => [-1.0, -1.0, -1.0])
    b = [2.0, 4.0, 6.0, 13.0]
    x = [1.0, 2.0, 3.0, 4.0]
    # not symmetric, with N * x == c for the same x
    N = sparse([1, 2, 1, 3, 2, 4, 4], [1, 1, 2, 2, 3, 3, 4], [2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0], 4, 4)
    c = [4.0, 4.0, 6.0, 7.0]
    bad = 0
    bad += check(near(T \ b, x))
    bad += check(near(N \ c, x))
    bad += check(near(lu(N) \ c, x))
    bad += check(near(qr(N) \ c, x))
    bad += check(near(cholesky(T) \ b, x))
    bad += check(near(ldlt(T) \ b, x))
    bad += check(abs(det(lu(T)) - 209.0) < 1e-10 && abs(logdet(cholesky(T)) - log(209.0)) < 1e-12)
    F = lu(N)
    lu!(F, 2N)
    bad += check(near(F \ c, x ./ 2))
    Z = sparse(ComplexF64[1.0im 1.0; 0.0 2.0])
    bad += check(near(Z \ ComplexF64[1.0 + 1.0im, 2.0], ComplexF64[1.0, 1.0]))
    return bad
end

function @main(args::Vector{String})::Cint
    bad = dense_cat() + sparse_cat() + construction() + indexing() + algebra() +
        structure() + vectors() + eltypes() + wrappers() + triangular() + inplace() +
        kernels() + search() + reshaping() + broadcasting() + equality() + sparse_rhs()
    if "solvers" in args
        bad += solvers()
    end
    println(Core.stdout, bad == 0 ? "ok" : "$bad checks failed")
    return bad == 0 ? 0 : 1
end

end
