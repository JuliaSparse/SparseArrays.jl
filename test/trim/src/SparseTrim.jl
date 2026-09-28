# This file is a part of Julia. License is MIT: https://julialang.org/license

# An app built with `juliac --trim=safe` in CI. SparseArrays extends Base's concatenation
# for every numeric array, so loading it must keep dense concatenation trimmable, as
# Julia's own trim test checks; the sparse calls cover the package's common paths.
module SparseTrim

using SparseArrays

check(ok::Bool) = ok ? 0 : 1

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

function sparse_ops()
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 4.0, 5.0], 3, 3)
    x = [1.0, 1.0, 1.0]
    D = Matrix(A)
    bad = 0
    # dense products would call BLAS, so the expected values are written out
    bad += check(A * x == [2.0, 3.0, 9.0])
    bad += check(A' * x == [6.0, 3.0, 5.0])
    bad += check(Matrix(A + A) == 2 .* D)
    bad += check(sum(A) == sum(D))
    bad += check(A[3, 1] == 4.0)
    bad += check(nnz(A) == 4)
    bad += check(hcat(A, D) isa SparseMatrixCSC{Float64,Int})
    bad += check(Matrix(vcat(A, A)) == vcat(D, D))
    bad += check(Vector(sparsevec([1, 3], [1.0, 2.0], 3)) == [1.0, 0.0, 2.0])
    return bad
end

function @main(args::Vector{String})::Cint
    bad = dense_cat() + sparse_ops()
    println(Core.stdout, bad == 0 ? "ok" : "$bad checks failed")
    return bad == 0 ? 0 : 1
end

end
