# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseIndexingTests

using Test
using SparseArrays
using SparseArrays: getcolptr, nonzeroinds, _show_with_braille_patterns
using LinearAlgebra
using Random
using Test: guardseed
using InteractiveUtils: @which
using Dates
include("forbidproperties.jl")

# an index type lowered by `to_indices`, like `InvertedIndices.Not`
struct AllBut; i::Int; end
Base.to_indices(A, inds, I::Tuple{AllBut,Vararg}) =
    (setdiff(inds[1], I[1].i), to_indices(A, Base.tail(inds), Base.tail(I))...)

@testset "getindex" begin
    ni = 23
    nj = 32
    a116 = reshape(1:(ni*nj), ni, nj)
    s116 = sparse(a116)

    ad116 = diagm(0 => diag(a116))
    sd116 = sparse(ad116)

    for (aa116, ss116) in [(a116, s116), (ad116, sd116)]
        ij=11; i=3; j=2
        @test ss116[ij] == aa116[ij]
        @test ss116[(i,j)] == aa116[i,j]
        @test ss116[i,j] == aa116[i,j]
        @test ss116[i-1,j] == aa116[i-1,j]
        ss116[i,j] = 0
        @test ss116[i,j] == 0
        ss116 = sparse(aa116)

        @test ss116[:,:] == copy(ss116)

        @test convert(SparseMatrixCSC{Float32,Int32}, sd116)[2:5,:] == convert(SparseMatrixCSC{Float32,Int32}, sd116[2:5,:])

        # range indexing
        @test Array(ss116[i,:]) == aa116[i,:]
        @test Array(ss116[:,j]) == aa116[:,j]
        @test Array(ss116[i,1:2:end]) == aa116[i,1:2:end]
        @test Array(ss116[1:2:end,j]) == aa116[1:2:end,j]
        @test Array(ss116[i,end:-2:1]) == aa116[i,end:-2:1]
        @test Array(ss116[end:-2:1,j]) == aa116[end:-2:1,j]
        # float-range indexing is not supported

        # sorted vector indexing
        @test Array(ss116[i,[3:2:end-3;]]) == aa116[i,[3:2:end-3;]]
        @test Array(ss116[[3:2:end-3;],j]) == aa116[[3:2:end-3;],j]
        @test Array(ss116[i,[end-3:-2:1;]]) == aa116[i,[end-3:-2:1;]]
        @test Array(ss116[[end-3:-2:1;],j]) == aa116[[end-3:-2:1;],j]

        # unsorted vector indexing with repetition
        p = [4, 1, 2, 3, 2, 6]
        @test Array(ss116[p,:]) == aa116[p,:]
        @test Array(ss116[:,p]) == aa116[:,p]
        @test Array(ss116[p,p]) == aa116[p,p]

        # bool indexing
        li = bitrand(size(aa116,1))
        lj = bitrand(size(aa116,2))
        @test Array(ss116[li,j]) == aa116[li,j]
        @test Array(ss116[li,:]) == aa116[li,:]
        @test Array(ss116[i,lj]) == aa116[i,lj]
        @test Array(ss116[:,lj]) == aa116[:,lj]
        @test Array(ss116[li,lj]) == aa116[li,lj]

        # empty indices
        for empty in (1:0, Int[])
            @test Array(ss116[empty,:]) == aa116[empty,:]
            @test Array(ss116[:,empty]) == aa116[:,empty]
            @test Array(ss116[empty,lj]) == aa116[empty,lj]
            @test Array(ss116[li,empty]) == aa116[li,empty]
            @test Array(ss116[empty,empty]) == aa116[empty,empty]
        end

        # out of bounds indexing
        @test_throws BoundsError ss116[0, 1]
        @test_throws BoundsError ss116[end+1, 1]
        @test_throws BoundsError ss116[1, 0]
        @test_throws BoundsError ss116[1, end+1]
        for j in (1, 1:size(s116,2), 1:1, Int[1], trues(size(s116, 2)), 1:0, Int[])
            @test_throws BoundsError ss116[0:1, j]
            @test_throws BoundsError ss116[[0, 1], j]
            @test_throws BoundsError ss116[end:end+1, j]
            @test_throws BoundsError ss116[[end, end+1], j]
        end
        for i in (1, 1:size(s116,1), 1:1, Int[1], trues(size(s116, 1)), 1:0, Int[])
            @test_throws BoundsError ss116[i, 0:1]
            @test_throws BoundsError ss116[i, [0, 1]]
            @test_throws BoundsError ss116[i, end:end+1]
            @test_throws BoundsError ss116[i, [end, end+1]]
        end
    end

    # indexing by array of CartesianIndex (issue #30981)
    S = sprand(10, 10, 0.4)
    inds_sparse = S[findall(S .> 0.2)]
    M = Matrix(S)
    inds_dense = M[findall(M .> 0.2)]
    @test Array(inds_sparse) == inds_dense
    inds_out = Array([CartesianIndex(1, 1), CartesianIndex(0, 1)])
    @test_throws BoundsError S[inds_out]
    pop!(inds_out); push!(inds_out, CartesianIndex(1, 0))
    @test_throws BoundsError S[inds_out]
    pop!(inds_out); push!(inds_out, CartesianIndex(11, 1))
    @test_throws BoundsError S[inds_out]
    pop!(inds_out); push!(inds_out, CartesianIndex(1, 11))
    @test_throws BoundsError S[inds_out]

    @testset "indices lowered by to_indices (issue #42), $T" for T in (Float64, ComplexF64)
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

    # workaround issue #7197: comment out let-block
    #let S = SparseMatrixCSC(3, 3, UInt8[1,1,1,1], UInt8[], Int64[])
    S1290 = SparseMatrixCSC(3, 3, UInt8[1,1,1,1], UInt8[], Int64[])
        S1290[1,1] = 1
        S1290[5] = 2
        S1290[end] = 3
        @test S1290[end] == (S1290[1] + S1290[2,2])
        @test 6 == sum(diag(S1290))
        @test Array(S1290)[[3,1],1] == Array(S1290[[3,1],1])

        # check that indexing with an abstract array returns matrix
        # with same colptr and rowval eltypes as input. Tests PR 24548
        r1 = S1290[[5,9]]
        r2 = S1290[[1 2;5 9]]
        @test isa(r1, SparseVector{Int64,UInt8})
        @test isa(r2, SparseMatrixCSC{Int64,UInt8})
    # end

    @testset "empty sparse matrix indexing" begin
        for k = 0:3
            @test issparse(spzeros(k,0)[:])
            @test isempty(spzeros(k,0)[:])
            @test issparse(spzeros(0,k)[:])
            @test isempty(spzeros(0,k)[:])
        end
    end
end

@testset "setindex" begin
    a = spzeros(Int, 10, 10)
    @test count(!iszero, a) == count((!iszero).(a)) == 0
    @test count(!iszero, a') == count((!iszero).(a')) == 0
    @test count(!iszero, transpose(a)) == count(transpose((!iszero).(a))) == 0
    a[1,:] .= 1
    @test count(!iszero, a) == count((!iszero).(a)) == 10
    @test count(!iszero, a, init=2) == count((!iszero).(a), init=2) == 12
    @test count(!iszero, a, init=Int128(2))::Int128 == 12
    @test count(!iszero, a') == count(((!iszero).(a))') == 10
    @test count(!iszero, transpose(a)) == count(transpose((!iszero).(a))) == 10
    @test a[1,:] == sparse(fill(1,10))
    a[:,2] .= 2
    @test count(!iszero, a) == count((!iszero).(a)) == 19
    @test a[:,2] == sparse(fill(2,10))
    b = copy(a)

    # Zero-assignment behavior of setindex!(A, v, i, j)
    a[1,3] = 0
    @test nnz(a) == 19
    @test count(!iszero, a) == 18
    a[2,1] = 0
    @test nnz(a) == 19
    @test count(!iszero, a) == 18

    # Zero-assignment behavior of setindex!(A, v, I, J)
    a[1,:] .= 0
    @test nnz(a) == 19
    @test count(!iszero, a) == 9
    a[2,:] .= 0
    @test nnz(a) == 19
    @test count(!iszero, a) == 8
    a[:,1] .= 0
    @test nnz(a) == 19
    @test count(!iszero, a) == 8
    a[:,2] .= 0
    @test nnz(a) == 19
    @test count(!iszero, a) == 0
    a = copy(b)
    a[:,:] .= 0
    @test nnz(a) == 19
    @test count(!iszero, a) == 0

    # Zero-assignment behavior of setindex!(A, B::SparseMatrixCSC, I, J)
    a = copy(b)
    a[1:2,:] = spzeros(2, 10)
    @test nnz(a) == 19
    @test count(!iszero, a) == 8
    a[1:2,1:3] = sparse([1 0 1; 0 0 1])
    @test nnz(a) == 20
    @test count(!iszero, a) == 11
    a = copy(b)
    a[1:2,:] = let c = sparse(fill(1,2,10)); fill!(nonzeros(c), 0); c; end
    @test nnz(a) == 19
    @test count(!iszero, a) == 8
    a[1:2,1:3] = let c = sparse(fill(1,2,3)); c[1,2] = c[2,1] = c[2,2] = 0; c; end
    @test nnz(a) == 20
    @test count(!iszero, a) == 11

    a[1,:] = 1:10
    @test a[1,:] == sparse([1:10;])
    a[:,2] = 1:10
    @test a[:,2] == sparse([1:10;])

    a[1,1:0] = []
    @test a[1,:] == sparse([1; 1; 3:10])
    a[1:0,2] = []
    @test a[:,2] == sparse([1:10;])
    a[1,1:0] .= 0
    @test a[1,:] == sparse([1; 1; 3:10])
    a[1:0,2] .= 0
    @test a[:,2] == sparse([1:10;])
    a[1,1:0] .= 1
    @test a[1,:] == sparse([1; 1; 3:10])
    a[1:0,2] .= 1
    @test a[:,2] == sparse([1:10;])
    a[3,2:3] .= 1 # one stored, one new value
    @test a[3,2:3] == sparse([1; 1])
    a[5:6,1] .= 1 # only new values
    @test a[:,1] == sparse([1; 0; 0; 0; 1; 1; 0; 0; 0; 0;])
    a[2:4,2:3] .= 3 # two ranges
    @test nnz(a) == 24

    @test_throws BoundsError a[:,11] = spzeros(10,1)
    @test_throws BoundsError a[11,:] = spzeros(1,10)
    @test_throws BoundsError a[:,-1] = spzeros(10,1)
    @test_throws BoundsError a[-1,:] = spzeros(1,10)
    @test_throws BoundsError a[0:9] = spzeros(1,10)
    @test_throws BoundsError (a[:,11] .= 0; a)
    @test_throws BoundsError (a[11,:] .= 0; a)
    @test_throws BoundsError (a[:,-1] .= 0; a)
    @test_throws BoundsError (a[-1,:] .= 0; a)
    @test_throws BoundsError (a[0:9] .= 0; a)
    @test_throws BoundsError (a[:,11] .= 1; a)
    @test_throws BoundsError (a[11,:] .= 1; a)
    @test_throws BoundsError (a[:,-1] .= 1; a)
    @test_throws BoundsError (a[-1,:] .= 1; a)
    @test_throws BoundsError (a[0:9] .= 1; a)

    @test_throws DimensionMismatch a[1:2,1:2] = 1:3
    @test_throws DimensionMismatch a[1:2,1] = 1:3
    @test_throws DimensionMismatch a[1,1:2] = 1:3
    @test_throws DimensionMismatch a[1:2] = 1:3

    A = spzeros(Int, 10, 20)
    A[1:5,1:10] .= 10
    A[1:5,1:10] .= 10
    @test count(!iszero, A) == 50
    @test A[1:5,1:10] == fill(10, 5, 10)
    A[6:10,11:20] .= 0
    @test count(!iszero, A) == 50
    A[6:10,11:20] .= 20
    @test count(!iszero, A) == 100
    @test A[6:10,11:20] == fill(20, 5, 10)
    A[4:8,8:16] .= 15
    @test count(!iszero, A) == 121
    @test A[4:8,8:16] == fill(15, 5, 9)

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

    A = sparse(1I, 5, 5)
    lininds = 1:10
    X=reshape([trues(10); falses(15)],5,5)
    @test A[lininds] == A[X] == [1,0,0,0,0,0,1,0,0,0]
    A[lininds] = [1:10;]
    @test A[lininds] == A[X] == 1:10
    A[lininds] = zeros(Int, 10)
    @test nnz(A) == 13
    @test count(!iszero, A) == 3
    @test A[lininds] == A[X] == zeros(Int, 10)
    c = Vector(11:20); c[1] = c[3] = 0
    A[lininds] = c
    @test nnz(A) == 13
    @test count(!iszero, A) == 11
    @test A[lininds] == A[X] == c
    A = sparse(1I, 5, 5)
    A[lininds] = c
    @test nnz(A) == 12
    @test count(!iszero, A) == 11
    @test A[lininds] == A[X] == c

    let # prevent assignment to I from overwriting UniformSampling in enclosing scope
        S = sprand(50, 30, 0.5, x -> round.(Int, rand(x) * 100))
        I = sprand(Bool, 50, 30, 0.2)
        FS = Array(S)
        FI = Array(I)
        @test sparse(FS[FI]) == S[I] == S[FI]
        @test S[vec(FI)]::SparseVector == FS[vec(FI)]
        @test sum(S[FI]) + sum(S[.!FI]) == sum(S)
        @test count(!iszero, I) == count(I)

        sumS1 = sum(S)
        sumFI = sum(S[FI])
        nnzS1 = nnz(S)
        S[FI] .= 0
        sumS2 = sum(S)
        cnzS2 = count(!iszero, S)
        @test sum(S[FI]) == 0
        @test nnz(S) == nnzS1
        @test (sum(S) + sumFI) == sumS1

        S[FI] .= 10
        nnzS3 = nnz(S)
        @test sum(S) == sumS2 + 10*sum(FI)
        S[FI] .= 0
        @test sum(S) == sumS2
        @test nnz(S) == nnzS3
        @test count(!iszero, S) == cnzS2

        S[FI] .= [1:sum(FI);]
        @test sum(S) == sumS2 + sum(1:sum(FI))

        S = sprand(50, 30, 0.5, x -> round.(Int, rand(x) * 100))
        N = length(S) >> 2
        I = randperm(N) .* 4
        J = randperm(N)
        sumS1 = sum(S)
        sumS2 = sum(S[I])
        S[I] .= 0
        @test sum(S) == (sumS1 - sumS2)
        S[I] .= J
        @test sum(S) == (sumS1 - sumS2 + sum(J))
    end

    # setindex with a Matrix{Bool}
    Is = fill(false, 10, 10)
    Is[1, 1] = true
    Is[10, 10] = true
    A = sprand(10, 10, 0.2)
    A[Is] = [0.1, 0.5]
    @test A[1, 1] == 0.1
    @test A[10, 10] == 0.5
    A = spzeros(10, 10)
    A[Is] = [0.1, 0.5]
    @test nnz(A) == 2
end

@testset "dropstored!" begin
    A = spzeros(Int, 10, 10)
    # Introduce nonzeros in row and column two
    A[1,:] .= 1
    A[:,2] .= 2
    @test nnz(A) == 19

    # Test argument bounds checking for dropstored!(A, i, j)
    @test_throws BoundsError SparseArrays.dropstored!(A, 0, 1)
    @test_throws BoundsError SparseArrays.dropstored!(A, 1, 0)
    @test_throws BoundsError SparseArrays.dropstored!(A, 1, 11)
    @test_throws BoundsError SparseArrays.dropstored!(A, 11, 1)

    # Test argument bounds checking for dropstored!(A, I, J)
    @test_throws BoundsError SparseArrays.dropstored!(A, 0:1, 1:1)
    @test_throws BoundsError SparseArrays.dropstored!(A, 1:1, 0:1)
    @test_throws BoundsError SparseArrays.dropstored!(A, 10:11, 1:1)
    @test_throws BoundsError SparseArrays.dropstored!(A, 1:1, 10:11)

    # Test behavior of dropstored!(A, i, j)
    # --> Test dropping a single stored entry
    SparseArrays.dropstored!(A, 1, 2)
    @test nnz(A) == 18
    # --> Test dropping a single nonstored entry
    SparseArrays.dropstored!(A, 2, 1)
    @test nnz(A) == 18

    # Test behavior of dropstored!(A, I, J) and derivs.
    # --> Test dropping a single row including stored and nonstored entries
    SparseArrays.dropstored!(A, 1, :)
    @test nnz(A) == 9
    # --> Test dropping a single column including stored and nonstored entries
    SparseArrays.dropstored!(A, :, 2)
    @test nnz(A) == 0
    # --> Introduce nonzeros in rows one and two and columns two and three
    A[1:2,:] .= 1
    A[:,2:3] .= 2
    @test nnz(A) == 36
    # --> Test dropping multiple rows containing stored and nonstored entries
    SparseArrays.dropstored!(A, 1:3, :)
    @test nnz(A) == 14
    # --> Test dropping multiple columns containing stored and nonstored entries
    SparseArrays.dropstored!(A, :, 2:4)
    @test nnz(A) == 0
    # --> Introduce nonzeros in every other row
    A[1:2:9, :] .= 1
    @test nnz(A) == 50
    # --> Test dropping a block of the matrix towards the upper left
    SparseArrays.dropstored!(A, 2:5, 2:5)
    @test nnz(A) == 42
    # --> Test dropping all elements
    SparseArrays.dropstored!(A, :)
    @test nnz(A) == 0
    A[1:2:9, :] .= 1
    @test nnz(A) == 50
    SparseArrays.dropstored!(A, :, :)
    @test nnz(A) == 0
end

struct CountedReads <: AbstractVector{Int}
    v::Vector{Int}
    reads::Base.RefValue{Int}
end
Base.size(c::CountedReads) = size(c.v)
Base.IndexStyle(::Type{CountedReads}) = IndexLinear()
Base.getindex(c::CountedReads, i::Int) = (c.reads[] += 1; c.v[i])

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
    for density in (0.0, 0.0001, 0.001, 0.01, 0.1, 1.0)
        S = sprand(rng, m, n, density)
        isempty(nonzeros(S)) || (nonzeros(S)[1] = 0)
        for I in indices, J in (Int[], [n, 1, n], randperm(rng, n))
            test_getindex_algs(S, I, J)
        end
    end

    @testset "heuristic boundaries" begin
        # Keep the sparse-first condition false while crossing each strict threshold.
        for (m, n, stored_per_column, selected_rows) in (
                (1024, 8, 128, 128 + 255),
                (1024, 8, 128, 128 + 256),
                (1024, 8, 128, 128 + 257),
                (2048, 2, 1152, 1152 - 1023),
                (2048, 2, 1152, 1152 - 1024),
                (2048, 2, 1152, 1152 - 1025),
                (2048, 2, 16, 400))
            rows = round.(Int, range(1, m; length=stored_per_column))
            S = sparse(repeat(rows, n), repeat(1:n; inner=stored_per_column),
                       collect(1:(n * stored_per_column)), m, n)
            nonzeros(S)[1] = 0
            I = collect(1:selected_rows)
            test_getindex_algs(S, I, [n, 1, n])
        end
    end

    @testset "few columns do not allocate a cache of length size(A, 1)" begin
        m = 10^6
        S = sparse(1.0I, m, m)
        for I in ([2, 2, 5], collect(1:300)), J in (2, [2], [5, 2])
            S[I, J]
            @test (@allocated S[I, J]) < m
            @test S[I, J] == S[1:m, J][I, fill(:, ndims(J))...]
        end
    end

    @testset "more rows than stored entries does not walk I for every column" begin
        m, n = 10^4, 200
        S = sparse(collect(1:50:m), collect(1:n), 1.0, m, n)
        S[m, n] = 0
        @test m > nnz(S)
        I = CountedReads(collect(2:2:m-2), Ref(0))
        R = S[I, 1:n]
        @test I.reads[] < 20 * length(I)
        @test R == Matrix(S)[I.v, 1:n]
        # a short I still binary-searches the columns
        T = sparse(repeat(1:4:m, 2), repeat(1:2; inner=m÷4), 1.0, m, 2)
        @test T[[5, 5, 6, m-3], [2, 1]] == Matrix(T)[[5, 5, 6, m-3], [2, 1]]
    end
end

@testset "getindex bounds checking" begin
    S = sprand(10, 10, 0.1)
    @test_throws BoundsError S[[0,1,2], [1,2]]
    @test_throws BoundsError S[[1,2], [0,1,2]]
    @test_throws BoundsError S[[0,2,1], [1,2]]
    @test_throws BoundsError S[[2,1], [0,1,2]]
end

@testset "row indexing a SparseMatrixCSC with non-Int integer type" begin
    local A = sparse(UInt32[1,2,3], UInt32[1,2,3], [1.0,2.0,3.0])
    @test A[1,1:3] == A[1,:] == [1,0,0]
end

@testset "isstored" begin
    m = 5
    n = 4
    I = [1, 2, 5, 3]
    J = [2, 3, 4, 2]
    A = sparse(I, J, [1, 2, 3, 4], m, n)
    stored_indices = [CartesianIndex(i, j) for (i, j) in zip(I, J)]
    unstored_indices = [c for c in CartesianIndices((m, n)) if !(c in stored_indices)]
    for c in stored_indices
        @test Base.isstored(A, c[1], c[2]) == true
    end
    for c in unstored_indices
        @test Base.isstored(A, c[1], c[2]) == false
    end

    # `isstored` for adjoint and transposed matrices:
    for trans in (adjoint, transpose)
        B = trans(A)
        stored_indices = [CartesianIndex(j, i) for (j, i) in zip(J, I)]
        unstored_indices = [c for c in CartesianIndices((n, m)) if !(c in stored_indices)]
        for c in stored_indices
            @test Base.isstored(B, c[1], c[2]) == true
        end
        for c in unstored_indices
            @test Base.isstored(B, c[1], c[2]) == false
        end
    end
end


_length_or_count_or_five(::Colon) = 5
_length_or_count_or_five(x::AbstractVector{Bool}) = count(x)
_length_or_count_or_five(x) = length(x)

@testset "nonscalar setindex!" begin
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
    @test setindex!(spzeros(5, 5), 1:25, :) == setindex!(zeros(5,5), 1:25, :) == reshape(1:25, 5, 5)
    # a 1×n matrix value into a column is reshaped rather than silently zeroed, see #569
    @test setindex!(sparse(1.0I, 5, 5), reshape(1.0:5.0, 1, 5), :, 2) == setindex!(Matrix(1.0I, 5, 5), reshape(1.0:5.0, 1, 5), :, 2)
    @test setindex!(spzeros(5, 5), (25:-1:1).+spzeros(25), :) == setindex!(zeros(5,5), (25:-1:1).+spzeros(25), :) == reshape(25:-1:1, 5, 5)
    for X in (1:20, sparse(1:20), reshape(sparse(1:20), 20, 1), (1:20) .+ spzeros(20, 1), collect(1:20), collect(reshape(1:20, 20, 1)))
        @test setindex!(spzeros(5, 5), X, 6:25) == setindex!(zeros(5,5), 1:20, 6:25)
        @test setindex!(spzeros(5, 5), X, 21:-1:2) == setindex!(zeros(5,5), 1:20, 21:-1:2)
        b = trues(25)
        b[[6, 8, 13, 15, 23]] .= false
        @test setindex!(spzeros(5, 5), X, b) == setindex!(zeros(5, 5), X, b)
    end
end

let
    a116 = copy(reshape(1:16, 4, 4))
    s116 = sparse(a116)

    @testset "sparse ref" begin
        p = [4, 1, 2, 3, 2]
        @test Array(s116[p,:]) == a116[p,:]
        @test Array(s116[:,p]) == a116[:,p]
        @test Array(s116[p,p]) == a116[p,p]
    end

    @testset "sparse assignment" begin
        p = [4, 1, 3]
        a116[p, p] .= -1
        s116[p, p] .= -1
        @test a116 == s116

        p = [2, 1, 4]
        a116[p, p] = reshape(1:9, 3, 3)
        s116[p, p] = reshape(1:9, 3, 3)
        @test a116 == s116
    end
end

end # module
