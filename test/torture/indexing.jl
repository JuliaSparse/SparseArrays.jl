# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for sparse indexing. Each testset names the issue it guards;
# they run only when the `torture` suite is selected.

module TortureIndexingTests
using Test
using SparseArrays
using LinearAlgebra
using Test: guardseed
include("../testhelpers.jl")
using Random

# an index type lowered by `to_indices`, like `InvertedIndices.Not`
struct AllBut; i::Int; end
Base.to_indices(A, inds, I::Tuple{AllBut,Vararg}) =
    (setdiff(inds[1], I[1].i), to_indices(A, Base.tail(inds), Base.tail(I))...)

@testset "Issue #30006" begin
    A = SparseMatrixCSC{Float64,Int32}(spzeros(3,3))
    A[:, 1] = [1, 2, 3]
    @test nnz(A) == 3
    @test nonzeros(A) == [1, 2, 3]
end

@testset "Issue #28963" begin
    @test_throws DimensionMismatch (spzeros(10,10)[:, :] = sprand(10,20,0.5))
end

@testset "issue #9917" begin
    @test sparse([]') == reshape(sparse([]), 1, 0)
    @test Array(sparse([])) == zeros(0)
    @test_throws BoundsError sparse([])[1]
    @test_throws BoundsError sparse([])[1] = 1
    x = sparse(1.0I, 100, 100)
    @test_throws BoundsError x[-10:10]
end

@testset "issue #14398" begin
    @test collect(view(sparse(I, 10, 10), 1:5, 1:5)') ≈ Matrix(I, 5, 5)
end

@testset "dropstored issue #20513" begin
    x = sparse(rand(3,3))
    SparseArrays.dropstored!(x, 1, 1)
    @test x[1, 1] == 0.0
    @test getcolptr(x) == [1, 3, 6, 9]
    SparseArrays.dropstored!(x, 2, 1)
    @test getcolptr(x) == [1, 2, 5, 8]
    @test x[2, 1] == 0.0
    SparseArrays.dropstored!(x, 2, 2)
    @test getcolptr(x) == [1, 2, 4, 7]
    @test x[2, 2] == 0.0
    SparseArrays.dropstored!(x, 2, 3)
    @test getcolptr(x) == [1, 2, 4, 6]
    @test x[2, 3] == 0.0
end

@testset "setindex issue #20657" begin
    local A = spzeros(3, 3)
    I = [1, 1, 1]; J = [1, 1, 1]
    A[I, 1] .= 1
    @test nnz(A) == 1
    A[1, J] .= 1
    @test nnz(A) == 1
    A[I, J] .= 1
    @test nnz(A) == 1
end

@testset "setindex with vector eltype (#29034)" begin
    A = sparse([1], [1], [Vector{Float64}(undef, 3)], 3, 3)
    A[1,1] = [1.0, 2.0, 3.0]
    @test A[1,1] == [1.0, 2.0, 3.0]
    @test_throws BoundsError setindex!(A, [4.0, 5.0, 6.0], 4, 3)
    @test_throws BoundsError setindex!(A, [4.0, 5.0, 6.0], 3, 4)
end

@testset "reverse search direction if step < 0 #21986" begin
    local A, B
    A = guardseed(1234) do
        sprand(5, 5, 1/5)
    end
    A = max.(A, copy(A'))
    LinearAlgebra.fillstored!(A, 1)
    B = A[5:-1:1, 5:-1:1]
    @test issymmetric(B)
end

@testset "Issue #29" begin
    s = sprand(6, 6, .2)
    li = LinearIndices(s)
    ci = CartesianIndices(s)
    @test s[li] == s[ci] == s[Matrix(li)] == s[Matrix(ci)]
end

# #20711
@testset "vec returns a view" begin
    local A = sparse(Matrix(1.0I, 3, 3))
    local v = vec(A)
    v[1] = 2
    @test A[1,1] == 2
end

@testset "-0.0 (issue #294, pr #296)" begin
    v = spzeros(1)
    v[1] = -0.0
    @test v[1] === -0.0

    m = spzeros(1, 1)
    m[1, 1] = -0.0
    @test m[1, 1] === -0.0
end

# From Base's arrayops.jl
@testset "CartesianIndex" begin
   a = spzeros(2,3)
    @test CartesianIndices(size(a)) == eachindex(a)
    a[CartesianIndex{2}(2,3)] = 5
    @test a[2,3] == 5
    b = view(a, 1:2, 2:3)
    b[CartesianIndex{2}(1,1)] = 7
    @test a[1,2] == 7
end

@testset "Assignment of singleton array to sparse array (julia #43644)" begin
    K = spzeros(3,3)
    b = zeros(3,3)
    b[3,:] = [1,2,3]
    K[3,1:3] += [1.0 2.0 3.0]'
    @test K == b
    K[3:3,1:3] += zeros(1, 3)
    @test K == b
    K[3,1:3] += zeros(3)
    @test K == b
    K[3,:] += zeros(3,1)
    @test K == b
    @test_throws DimensionMismatch K[3,1:2] += [1.0 2.0 3.0]'
end

# From Base's abstractarray.jl
@testset "itr, iterate" begin
    r = sparse(2:3:8)
    itr = eachindex(r)
    y = iterate(itr)
    @test y !== nothing
    y = iterate(itr, y[2])
    y = iterate(itr, y[2])
    @test y !== nothing
    val, state = y
    @test r[val] == 8
    @test iterate(itr, state) == nothing
end

# The real eltype of the `to_indices` grid; the core suite keeps ComplexF64.
@testset "indices lowered by to_indices (issue #42), $T" for T in (Float64,)
    A = sprand(T, 6, 6, 0.4); c = isodd.(1:6); x = A[:, 1]; m = A .!= 0
    for B in (A, A', transpose(A))
        for I in ((1, c), (c, 2), (c, c), (:, c), (c, :), (2:5, c), ([3, 1], c), (:, :),
                  (AllBut(2), AllBut(3)), (1, AllBut(3)), (AllBut(2), c), (Int32(2), Int32(3)))
            @test which(getindex, typeof.((B, I...))).module === SparseArrays
            @test B[I...] == B[to_indices(B, I)...] == Array(B)[I...]
            @test B[I...] isa Union{T,SparseVector{T,Int},SparseMatrixCSC{T,Int}}
        end
        @test B[5] == B[CartesianIndex(5, 1)] == B[5, 1, 1] == Array(B)[5]
        # masks of the wrong length throw as they do for dense arrays
        @test_throws BoundsError B[trues(7), 1]
        @test_throws BoundsError B[1, trues(7)]
    end
    @test A[to_indices(A, (m,))...] == A[to_indices(A, (vec(m),))...] == Array(A)[m]
    @test A[1:2, :][false:true, c] == Array(A)[1:2, :][false:true, c]
    @test which(getindex, typeof.((x, AllBut(2)))).module === SparseArrays
    @test x[AllBut(2)] == Array(x)[AllBut(2)]
    @test x[to_indices(x, (c,))...] == Array(x)[c]
    @test_throws BoundsError x[trues(7)]
end

# Ranged assignment into a large sparse matrix; the core "setindex" testset covers the
# same operations at 10 x 20.
@testset "setindex, large ranged assignment" begin
    ASZ = 1000
    TSZ = 800
    A = sprand(ASZ, 2*ASZ, 0.0001)
    B = copy(A)
    nA = count(!iszero, A)
    x = A[1:TSZ, 1:(2*TSZ)]
    nx = count(!iszero, x)
    A[1:TSZ, 1:(2*TSZ)] .= 0
    nB = count(!iszero, A)
    @test nB == (nA - nx)
    A[1:TSZ, 1:(2*TSZ)] = x
    @test count(!iszero, A) == nA
    @test A == B
    A[1:TSZ, 1:(2*TSZ)] .= 10
    @test count(!iszero, A) == nB + 2*TSZ*TSZ
    A[1:TSZ, 1:(2*TSZ)] = x
    @test count(!iszero, A) == nA
    @test A == B
end

# The densities of the getindex-algorithm grid that the core suite does not run.
@testset "test_getindex_algs" begin
    function test_getindex_algs(S, I, J)
        D = Matrix(S)
        @test S[I, J] == D[I, J]
        sortedI = sort(I)
        expected = D[sortedI, J]
        @test S[sortedI, J] == expected
        for alg in (SparseArrays.getindex_I_sorted_bsearch_A,
                    SparseArrays.getindex_I_sorted_bsearch_I,
                    SparseArrays.getindex_I_sorted_linear,
                    SparseArrays.getindex_I_sorted_nocache)
            @test alg(S, sortedI, J) == expected
        end
    end

    rng = MersenneTwister(12860)
    m, n = 128, 8
    indices = (Int[], [1], [m], [m, 1, m ÷ 2, 1],
               randperm(rng, m)[1:13], repeat(collect(1:m), 3))
    for density in (0.0001, 0.001, 0.1)
        S = sprand(rng, m, n, density)
        isempty(nonzeros(S)) || (nonzeros(S)[1] = 0)
        for I in indices, J in (Int[], [n, 1, n], randperm(rng, n))
            test_getindex_algs(S, I, J)
        end
    end
end

_length_or_count_or_five(::Colon) = 5
_length_or_count_or_five(x::AbstractVector{Bool}) = count(x)
_length_or_count_or_five(x) = length(x)

# The full product of index kinds; the core suite pairs every row index with two column
# indices and every column index with two row indices.
@testset "nonscalar setindex!, all index-kind pairs" begin
    for I in (1:4, :, 5:-1:2, [], trues(5), setindex!(falses(5), true, 2), 3),
        J in (2:4, :, 4:-1:1, [], setindex!(trues(5), false, 3), falses(5), 4)
        V = sparse(1 .+ zeros(_length_or_count_or_five(I)*_length_or_count_or_five(J)))
        M = sparse(1 .+ zeros(_length_or_count_or_five(I), _length_or_count_or_five(J)))
        if I isa Integer && J isa Integer
            @test_throws MethodError spzeros(5,5)[I, J] = V
            @test_throws MethodError spzeros(5,5)[I, J] = M
            continue
        end
        @test setindex!(spzeros(5, 5), V, I, J) == setindex!(zeros(5,5), V, I, J)
        @test setindex!(spzeros(5, 5), M, I, J) == setindex!(zeros(5,5), M, I, J)
        @test setindex!(spzeros(5, 5), Array(M), I, J) == setindex!(zeros(5,5), M, I, J)
        @test setindex!(spzeros(5, 5), Array(V), I, J) == setindex!(zeros(5,5), V, I, J)
    end
end

end # module
