# This file is a part of Julia. License is MIT: https://julialang.org/license

# An app built with `juliac --trim=safe` in CI. SparseArrays extends Base's concatenation
# for every numeric array, so loading it must keep dense concatenation trimmable, as
# Julia's own trim test checks; the sparse calls cover the package's common paths.
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
    # dense products would call BLAS, so the expected values are written out
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

# The solvers pass trim verification, but a trimmed binary cannot yet load a library
# through a `LazyLibrary`, as SuiteSparse is loaded, so CI builds them without running them.
function solvers()
    # tridiagonal(-1, 4, -1), with x = [1, 2, 3, 4]
    T = spdiagm(-1 => [-1.0, -1.0, -1.0], 0 => [4.0, 4.0, 4.0, 4.0], 1 => [-1.0, -1.0, -1.0])
    b = [2.0, 4.0, 6.0, 13.0]
    x = [1.0, 2.0, 3.0, 4.0]
    # not symmetric, with the same x
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
        structure() + vectors()
    if "solvers" in args
        bad += solvers()
    end
    println(Core.stdout, bad == 0 ? "ok" : "$bad checks failed")
    return bad == 0 ? 0 : 1
end

end
