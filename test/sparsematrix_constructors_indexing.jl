# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseMatrixConstructorIndexingTests

using Test
using SparseArrays
using SparseArrays: getcolptr, nonzeroinds, _show_with_braille_patterns
using LinearAlgebra
using Random
using Test: guardseed
using InteractiveUtils: @which
using Dates
include("forbidproperties.jl")

function same_structure(A, B)
    return all(getfield(A, f) == getfield(B, f) for f in (:m, :n, :colptr, :rowval))
end

@testset "uniform scaling should not change type #103" begin
    A = spzeros(Float32, Int8, 5, 5)
    B = I - A
    @test typeof(B) == typeof(A)
end

@testset "spzeros de-splatting" begin
    @test spzeros(Float64, Int64, (2, 2)) == spzeros(Float64, Int64, 2, 2)
    @test spzeros(Float64, Int32, (2, 2)) == spzeros(Float64, Int32, 2, 2)
    @test spzeros(Float32, (3, 2)) == spzeros(Float32, Int, 3, 2)
    @test spzeros((3, 2)) == spzeros((3, 2)...)
end

@testset "conversion to AbstractMatrix/SparseMatrix of same eltype" begin
    a = sprand(5, 5, 0.2)
    @test AbstractMatrix{eltype(a)}(a) == a
    @test SparseMatrixCSC{eltype(a)}(a) == a
    @test SparseMatrixCSC{eltype(a), Int}(a) == a
    @test SparseMatrixCSC{eltype(a)}(Array(a)) == a
    @test Array(SparseMatrixCSC{eltype(a), Int8}(a)) == Array(a)
    @test collect(a) == a
    # wrappers and views convert through the sparse kernels, not element by element
    c = sprand(ComplexF64, 5, 3, 0.4)
    for w in (a', transpose(c), view(c, :, 2:3), view(c, :, 1)', transpose(view(c, :, 1)))
        @test which(copyto!, Tuple{Matrix{eltype(w)}, typeof(w)}).module == SparseArrays
        @test Matrix(w)::Matrix{eltype(w)} == collect(w)
    end
    # issue #54
    b = SparseMatrixCSC{ComplexF64,Int32}(a)
    @test promote_type(typeof(a), typeof(b)) === SparseMatrixCSC{ComplexF64,Int}
    @test promote_type(typeof(a), Matrix{ComplexF64}) === Matrix{ComplexF64}
    @test promote_type(Matrix{Int}, typeof(a)) === Matrix{Float64}
    @test promote_type(SparseMatrixCSC{Int8,Int}, SparseMatrixCSC{Int16,Int}) === SparseMatrixCSC{Int16,Int}
    @test promote(a, b) == (a, b)
    @test eltype([a, b]) === SparseMatrixCSC{ComplexF64,Int}
end

@testset "sparse matrix construction" begin
    @test (A = fill(1.0+im,5,5); isequal(Array(sparse(A)), A))
    @test_throws ArgumentError sparse([1,2,3], [1,2], [1,2,3], 3, 3)
    @test_throws ArgumentError sparse([1,2,3], [1,2,3], [1,2], 3, 3)
    @test_throws ArgumentError sparse([1,2,3], [1,2,3], [1,2,3], 0, 1)
    @test_throws ArgumentError sparse([1,2,3], [1,2,3], [1,2,3], 1, 0)
    @test_throws ArgumentError sparse([1,2,4], [1,2,3], [1,2,3], 3, 3)
    @test_throws ArgumentError sparse([1,2,3], [1,2,4], [1,2,3], 3, 3)
    @test isequal(sparse(Int[], Int[], Int[], 0, 0), SparseMatrixCSC(0, 0, Int[1], Int[], Int[]))
    @test isequal(sparse(big.([1,1,1,2,2,3,4,5]),big.([1,2,3,2,3,3,4,5]),big.([1,2,4,3,5,6,7,8]), 6, 6),
        SparseMatrixCSC(6, 6, big.([1,2,4,7,8,9,9]), big.([1,1,2,1,2,3,4,5]), big.([1,2,3,4,5,6,7,8])))
    @test sparse(Any[1,2,3], Any[1,2,3], Any[1,1,1]) == sparse([1,2,3], [1,2,3], [1,1,1])
    @test sparse(Any[1,2,3], Any[1,2,3], Any[1,1,1], 5, 4) == sparse([1,2,3], [1,2,3], [1,1,1], 5, 4)
    # with combine
    @test sparse([1, 1, 2, 2, 2], [1, 2, 1, 2, 2], 1.0, 2, 2, +) == sparse([1, 1, 2, 2], [1, 2, 1, 2], [1.0, 1.0, 1.0, 2.0], 2, 2)
    @test sparse([1, 1, 2, 2, 2], [1, 2, 1, 2, 2], -1.0, 2, 2, *) == sparse([1, 1, 2, 2], [1, 2, 1, 2], [-1.0, -1.0, -1.0, 1.0], 2, 2)
    @test sparse(sparse(Int32.(1:5), Int32.(1:5), trues(5))') isa SparseMatrixCSC{Bool,Int32}
    # undef initializer
    sz = (3, 4)
    for m in (SparseMatrixCSC{Float32, Int16}(undef, sz...), SparseMatrixCSC{Float32, Int16}(undef, sz),
                 similar(SparseMatrixCSC{Float32, Int16}, sz))
        @test size(m) == sz
        @test eltype(m) === Float32
        @test m == spzeros(sz...)
    end
end

@testset "spzeros for pattern creation (structural zeros)" begin
    I = [1, 2, 3]
    J = [1, 3, 4]
    V = zeros(length(I))
    S = spzeros(I, J)
    S′ = sparse(I, J, V)
    @test S == S′
    @test same_structure(S, S′)
    @test eltype(S) == Float64
    S = spzeros(Float32, I, J)
    @test S == S′
    @test same_structure(S, S′)
    @test eltype(S) == Float32
    S = spzeros(I, J, 4, 5)
    S′ = sparse(I, J, V, 4, 5)
    @test S == S′
    @test same_structure(S, S′)
    @test eltype(S) == Float64
    S = spzeros(Float32, I, J, 4, 5)
    @test S == S′
    @test same_structure(S, S′)
    @test eltype(S) == Float32
end

@testset "repeat tests" begin
    A = sprand(6, 4, 0.5)
    A_full = Matrix(A)
    for m = 0:3
        @test issparse(repeat(A, m))
        @test repeat(A, m) == repeat(A_full, m)
        for n = 0:3
            @test issparse(repeat(A, m, n))
            @test repeat(A, m, n) == repeat(A_full, m, n)
        end
    end
end

@testset "copyto!" begin
    A = sprand(5, 5, 0.2)
    B = sprand(5, 5, 0.2)
    Ar = copyto!(A, B)
    @test Ar === A
    @test A == B
    @test pointer(nonzeros(A)) != pointer(nonzeros(B))
    @test pointer(rowvals(A)) != pointer(rowvals(B))
    @test pointer(getcolptr(A)) != pointer(getcolptr(B))
    # Test size(A) != size(B), but length(A) == length(B)
    B = sprand(25, 1, 0.2)
    copyto!(A, B)
    @test A[:] == B[:]
    # Test various size(A) / size(B) combinations
    for mA in [5, 10, 20], nA in [5, 10, 20], mB in [5, 10, 20], nB in [5, 10, 20]
        A = sprand(mA,nA,0.4)
        Aorig = copy(A)
        B = sprand(mB,nB,0.4)
        if mA*nA >= mB*nB
            copyto!(A,B)
            @assert(A[1:length(B)] == B[:])
            @assert(A[length(B)+1:end] == Aorig[length(B)+1:end])
        else
            @test_throws BoundsError copyto!(A,B)
        end
    end
    # Test eltype(A) != eltype(B), size(A) != size(B)
    A = sprand(5, 5, 0.2)
    Aorig = copy(A)
    B = sparse(rand(Float32, 3, 3))
    copyto!(A, B)
    @test A[1:9] == B[:]
    @test A[10:end] == Aorig[10:end]
    # Test eltype(A) != eltype(B), size(A) == size(B)
    A = sparse(rand(Float64, 3, 3))
    B = sparse(rand(Float32, 3, 3))
    copyto!(A, B)
    @test A == B
    # Test copyto!(dense, sparse)
    B = sprand(5, 5, 1.0)
    A = rand(5,5)
    A´ = similar(A)
    Ac = copyto!(A, B)
    @test Ac === A
    @test A == copyto!(A´, Matrix(B))
    # Test copyto!(dense, Rdest, sparse, Rsrc)
    A = rand(5,5)
    A´ = similar(A)
    Rsrc = CartesianIndices((3:4, 2:3))
    Rdest = CartesianIndices((2:3, 1:2))
    copyto!(A, Rdest, B, Rsrc)
    copyto!(A´, Rdest, Matrix(B), Rsrc)
    @test A[Rdest] == A´[Rdest] == Matrix(B)[Rsrc]
    # Test unaliasing of B´
    B´ = copy(B)
    copyto!(B´, Rdest, B´, Rsrc)
    @test Matrix(B´)[Rdest] == Matrix(B)[Rsrc]
    # Test that only elements at overlapping linear indices are overwritten
    A = sprand(3, 3, 1.0); B = ones(4, 4)
    Bc = copyto!(B, A)
    @test B[4, :] != B[:, 4] == ones(4)
    @test Bc === B
    # Allow no-op copyto! with empty source even for incompatible eltypes
    A = sparse(fill("", 0, 0))
    @test copyto!(B, A) == B

    # Test correct error for too small destination array
    @test_throws BoundsError copyto!(rand(2,2), sprand(3,3,0.2))
end

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

@testset "sparsevec from matrices" begin
    X = Matrix(1.0I, 5, 5)
    M = rand(5,4)
    C = spzeros(3,3)
    SX = sparse(X); SM = sparse(M)
    VX = vec(X); VSX = vec(SX)
    VM = vec(M); VSM1 = vec(SM); VSM2 = sparsevec(M)
    VC = vec(C)
    @test VX == VSX
    @test VM == VSM1
    @test VM == VSM2
    @test size(VC) == (9,)
    @test nnz(VC) == 0
    @test nnz(VSX) == 5
end

@testset "test_getindex_algs" begin
    function test_getindex_algs(S, I, J)
        D = Matrix(S)
        @test S[I, J] == D[I, J]
        sortedI = sort(I)
        expected = D[sortedI, J]
        @test S[sortedI, J] == expected
        for alg in (SparseArrays.getindex_I_sorted_bsearch_A,
                    SparseArrays.getindex_I_sorted_bsearch_I,
                    SparseArrays.getindex_I_sorted_linear)
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
end

@testset "getindex bounds checking" begin
    S = sprand(10, 10, 0.1)
    @test_throws BoundsError S[[0,1,2], [1,2]]
    @test_throws BoundsError S[[1,2], [0,1,2]]
    @test_throws BoundsError S[[0,2,1], [1,2]]
    @test_throws BoundsError S[[2,1], [0,1,2]]
end

@testset "test that sparse / sparsevec constructors work for AbstractMatrix subtypes" begin
    D = Diagonal(fill(1,10))
    sm = sparse(D)
    sv = sparsevec(D)

    @test count(!iszero, sm) == 10
    @test count(!iszero, sv) == 10

    @test count(!iszero, sparse(Diagonal(Int[]))) == 0
    @test count(!iszero, sparsevec(Diagonal(Int[]))) == 0
end

@testset "Sparse construction with empty/1x1 structured matrices" begin
    empty = spzeros(0, 0)

    @test sparse(Diagonal(zeros(0, 0))) == empty
    @test sparse(Bidiagonal(zeros(0, 0), :U)) == empty
    @test sparse(Bidiagonal(zeros(0, 0), :L)) == empty
    @test sparse(SymTridiagonal(zeros(0, 0))) == empty
    @test sparse(Tridiagonal(zeros(0, 0))) == empty

    one_by_one = rand(1,1)
    sp_one_by_one = sparse(one_by_one)

    @test sparse(Diagonal(one_by_one)) == sp_one_by_one
    @test sparse(Bidiagonal(one_by_one, :U)) == sp_one_by_one
    @test sparse(Bidiagonal(one_by_one, :L)) == sp_one_by_one
    @test sparse(Tridiagonal(one_by_one)) == sp_one_by_one

    s = SymTridiagonal(rand(1), rand(0))
    @test sparse(s) == s
end

@testset "avoid allocation for zeros in diagonal" begin
    x = [1, 0, 0, 5, 0]
    d = Diagonal(x)
    s = sparse(d)
    @test s == d
    @test nnz(s) == 2
end

@testset "error conditions for reshape, and dropdims" begin
    local A = sprand(Bool, 5, 5, 0.2)
    @test_throws DimensionMismatch reshape(A,(20, 2))
    @test_throws ArgumentError dropdims(A,dims=(1, 1))
end

@testset "float" begin
    local A
    A = sprand(Bool, 5, 5, 0.0)
    @test eltype(float(A)) == Float64  # issue #11658
    A = sprand(Bool, 5, 5, 0.2)
    @test float(A) == float(Array(A))
end

@testset "complex" begin
    A = sprand(Bool, 5, 5, 0.0)
    @test eltype(complex(A)) == Complex{Bool}
    A = sprand(Bool, 5, 5, 0.2)
    @test complex(A) == complex(Array(A))
end

@testset "one(A::SparseMatrixCSC)" begin
    @test_throws DimensionMismatch one(sparse([1 1 1; 1 1 1]))
    @test one(sparse([1 1; 1 1]))::SparseMatrixCSC == [1 0; 0 1]
end

struct MockTropical{T} <: Number begin
    n::T
    end
end
MockTropical{T}(x::MockTropical{T}) where {T} = x
Base.zero(::Type{MockTropical{T}}) where {T} = MockTropical{T}(typemin(T))
Base.zero(x::MockTropical{T}) where {T} = zero(MockTropical{T})
Base.one(::Type{MockTropical{T}}) where {T} = MockTropical{T}(zero(T))
Base.one(x::MockTropical{T}) where {T} = one(MockTropical{T})
Base.:*(a::MockTropical{T}, b::MockTropical{T}) where {T} = MockTropical{T}(a.n + b.n)
Base.:+(a::MockTropical{T}, b::MockTropical{T}) where {T} = MockTropical{T}(max(a.n, b.n))

@testset "issue #731" begin
    x = MockTropical{Float64}(1.0)
    B = sparse([1, 2], [1,2], [x, x])
    C = [one(x) zero(x); zero(x) one(x)]
    @test one(B) == C
    @test B^0 == C
end

@testset "sparsevec" begin
    local A = sparse(fill(1, 5, 5))
    @test sparsevec(A) == fill(1, 25)
    @test sparsevec([1:5;], 1) == fill(1, 5)
    @test_throws ArgumentError sparsevec([1:5;], [1:4;])
end

@testset "sparse" begin
    local A = sparse(fill(1, 5, 5))
    @test sparse(A) == A
    @test sparse([1:5;], [1:5;], 1) == sparse(1.0I, 5, 5)
end

@testset "droptol" begin
    A = guardseed(1234321) do
        triu(sprand(10, 10, 0.2))
    end
    @test getcolptr(SparseArrays.droptol!(A, 0.01)) == [1, 1, 1, 1, 2, 2, 2, 4, 4, 5, 5]
    @test isequal(SparseArrays.droptol!(sparse([1], [1], [1]), 1), SparseMatrixCSC(1, 1, Int[1, 1], Int[], Int[]))
end

@testset "dropzeros[!]" begin
    smalldim = 5
    largedim = 10
    nzprob = 0.4
    targetnumposzeros = 5
    targetnumnegzeros = 5
    for (m, n) in ((largedim, largedim), (smalldim, largedim), (largedim, smalldim))
        local A = sprand(m, n, nzprob)
        struczerosA = findall(x -> x == 0, A)
        poszerosinds = unique(rand(struczerosA, targetnumposzeros))
        negzerosinds = unique(rand(struczerosA, targetnumnegzeros))
        Aposzeros = copy(A)
        Aposzeros[poszerosinds] .= 2
        Anegzeros = copy(A)
        Anegzeros[negzerosinds] .= -2
        Abothsigns = copy(Aposzeros)
        Abothsigns[negzerosinds] .= -2
        map!(x -> x == 2 ? 0.0 : x, nonzeros(Aposzeros), nonzeros(Aposzeros))
        map!(x -> x == -2 ? -0.0 : x, nonzeros(Anegzeros), nonzeros(Anegzeros))
        map!(x -> x == 2 ? 0.0 : x == -2 ? -0.0 : x, nonzeros(Abothsigns), nonzeros(Abothsigns))
        for Awithzeros in (Aposzeros, Anegzeros, Abothsigns)
            # Basic functionality / dropzeros!
            @test dropzeros!(copy(Awithzeros)) == A
            # Basic functionality / dropzeros
            @test dropzeros(Awithzeros) == A
            # Check trimming works as expected
            @test length(nonzeros(dropzeros!(copy(Awithzeros)))) == length(nonzeros(A))
            @test length(rowvals(dropzeros!(copy(Awithzeros)))) == length(rowvals(A))
        end
    end
    # original lone dropzeros test
    local A = sparse([1 2 3; 4 5 6; 7 8 9])
    nonzeros(A)[2] = nonzeros(A)[6] = nonzeros(A)[7] = 0
    @test getcolptr(dropzeros!(A)) == [1, 3, 5, 7]
    # test for issue #5169, modified for new behavior following #15242/#14798
    @test nnz(sparse([1, 1], [1, 2], [0.0, -0.0])) == 2
    @test nnz(dropzeros!(sparse([1, 1], [1, 2], [0.0, -0.0]))) == 0
    # test for issue #5437, modified for new behavior following #15242/#14798
    @test nnz(sparse([1, 2, 3], [1, 2, 3], [0.0, 1.0, 2.0])) == 3
    @test nnz(dropzeros!(sparse([1, 2, 3],[1, 2, 3],[0.0, 1.0, 2.0]))) == 2
end

@testset "test created type of sprand{T}(::Type{T}, m::Integer, n::Integer, density::AbstractFloat)" begin
    m = sprand(Float32, 10, 10, 0.1)
    @test eltype(m) == Float32
    m = sprand(Float64, 10, 10, 0.1)
    @test eltype(m) == Float64
    m = sprand(Int32, 10, 10, 0.1)
    @test eltype(m) == Int32
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

@testset "similar should not alias the input sparse array" begin
    a = sparse(rand(3,3) .+ 0.1)
    b = similar(a, Float32, Int32)
    c = similar(b, Float32, Int32)
    SparseArrays.dropstored!(b, 1, 1)
    @test length(rowvals(c)) == 9
    @test length(nonzeros(c)) == 9
end

@testset "similar with type conversion" begin
    local A = sparse(1.0I, 5, 5)
    @test size(similar(A, ComplexF64, Int)) == (5, 5)
    @test typeof(similar(A, ComplexF64, Int)) == SparseMatrixCSC{ComplexF64, Int}
    @test size(similar(A, ComplexF64, Int8)) == (5, 5)
    @test typeof(similar(A, ComplexF64, Int8)) == SparseMatrixCSC{ComplexF64, Int8}
    @test similar(A, ComplexF64,(6, 6)) == spzeros(ComplexF64, 6, 6)
    @test convert(Matrix, A) == Array(A) # lolwut, are you lost, test?
end

@testset "similar for SparseMatrixCSC" begin
    local A = sparse(1.0I, 5, 5)
    # test similar without specifications (preserves stored-entry structure)
    simA = similar(A)
    @test typeof(simA) == typeof(A)
    @test size(simA) == size(A)
    @test getcolptr(simA) == getcolptr(A)
    @test rowvals(simA) == rowvals(A)
    @test length(nonzeros(simA)) == length(nonzeros(A))
    # test similar with entry type specification (preserves stored-entry structure)
    simA = similar(A, Float32)
    @test typeof(simA) == SparseMatrixCSC{Float32,eltype(getcolptr(A))}
    @test size(simA) == size(A)
    @test getcolptr(simA) == getcolptr(A)
    @test rowvals(simA) == rowvals(A)
    @test length(nonzeros(simA)) == length(nonzeros(A))
    # test similar with entry and index type specification (preserves stored-entry structure)
    simA = similar(A, Float32, Int8)
    @test typeof(simA) == SparseMatrixCSC{Float32,Int8}
    @test size(simA) == size(A)
    @test getcolptr(simA) == getcolptr(A)
    @test rowvals(simA) == rowvals(A)
    @test length(nonzeros(simA)) == length(nonzeros(A))
    # test similar with Dims{2} specification (preserves storage space only, not stored-entry structure)
    simA = similar(A, (6,6))
    @test typeof(simA) == typeof(A)
    @test size(simA) == (6,6)
    @test getcolptr(simA) == fill(1, 6+1)
    @test length(rowvals(simA)) == 0
    @test length(nonzeros(simA)) == 0
    # test similar with entry type and Dims{2} specification (empty storage space)
    simA = similar(A, Float32, (6,6))
    @test typeof(simA) == SparseMatrixCSC{Float32,eltype(getcolptr(A))}
    @test size(simA) == (6,6)
    @test getcolptr(simA) == fill(1, 6+1)
    @test length(rowvals(simA)) == 0
    @test length(nonzeros(simA)) == 0
    # test similar with entry type, index type, and Dims{2} specification (preserves storage space only)
    simA = similar(A, Float32, Int8, (6,6))
    @test typeof(simA) == SparseMatrixCSC{Float32, Int8}
    @test size(simA) == (6,6)
    @test getcolptr(simA) == fill(1, 6+1)
    @test length(rowvals(simA)) == 0
    @test length(nonzeros(simA)) == 0
    # test similar with Dims{1} specification (preserves nothing)
    simA = similar(A, (6,))
    @test typeof(simA) == SparseVector{eltype(nonzeros(A)),eltype(getcolptr(A))}
    @test size(simA) == (6,)
    @test length(nonzeroinds(simA)) == 0
    @test length(nonzeros(simA)) == 0
    # test similar with entry type and Dims{1} specification (preserves nothing)
    simA = similar(A, Float32, (6,))
    @test typeof(simA) == SparseVector{Float32,eltype(getcolptr(A))}
    @test size(simA) == (6,)
    @test length(nonzeroinds(simA)) == 0
    @test length(nonzeros(simA)) == 0
    # test similar with entry type, index type, and Dims{1} specification (preserves nothing)
    simA = similar(A, Float32, Int8, (6,))
    @test typeof(simA) == SparseVector{Float32,Int8}
    @test size(simA) == (6,)
    @test length(nonzeroinds(simA)) == 0
    @test length(nonzeros(simA)) == 0
    # test entry points to similar with entry type, index type, and non-Dims shape specification
    @test similar(A, Float32, Int8, 6, 6) == similar(A, Float32, Int8, (6, 6))
    @test similar(A, Float32, Int8, 6) == similar(A, Float32, Int8, (6,))
end

@testset "similar should preserve underlying storage type and uplo flag" begin
    m, n = 4, 3
    sparsemat = sprand(m, m, 0.5)
    for SymType in (Symmetric, Hermitian)
        symsparsemat = SymType(sparsemat)
        @test isa(similar(symsparsemat), typeof(symsparsemat))
        @test similar(symsparsemat).uplo == symsparsemat.uplo
        @test isa(similar(symsparsemat, Float32), SymType{Float32,<:SparseMatrixCSC{Float32}})
        @test similar(symsparsemat, Float32).uplo == symsparsemat.uplo
        @test isa(similar(symsparsemat, (n, n)), typeof(sparsemat))
        @test isa(similar(symsparsemat, Float32, (n, n)), SparseMatrixCSC{Float32})
    end
end

@testset "similar should preserve underlying storage type" begin
    local m, n = 4, 3
    sparsemat = sprand(m, m, 0.5)
    for TriType in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
        trisparsemat = TriType(sparsemat)
        @test isa(similar(trisparsemat), typeof(trisparsemat))
        @test isa(similar(trisparsemat, Float32), TriType{Float32,<:SparseMatrixCSC{Float32}})
        @test isa(similar(trisparsemat, (n, n)), typeof(sparsemat))
        @test isa(similar(trisparsemat, Float32, (n, n)), SparseMatrixCSC{Float32})
    end
end

@testset "sparse findprev/findnext operations" begin

    x = [0,0,0,0,1,0,1,0,1,1,0]
    x_sp = sparse(x)

    for i=1:length(x)
        @test findnext(!iszero, x,i) == findnext(!iszero, x_sp,i)
        @test findprev(!iszero, x,i) == findprev(!iszero, x_sp,i)
    end

    y = [7 0 0 0 0;
         1 0 1 0 0;
         1 7 0 7 1;
         0 0 1 0 0;
         1 0 1 1 0.0]
    y_sp = [x == 7 ? -0.0 : x for x in sparse(y)]
    y = Array(y_sp)
    @test isequal(y_sp[1,1], -0.0)

    for i in keys(y)
        @test findnext(!iszero, y,i) == findnext(!iszero, y_sp,i)
        @test findprev(!iszero, y,i) == findprev(!iszero, y_sp,i)
        @test findnext(iszero, y,i) == findnext(iszero, y_sp,i)
        @test findprev(iszero, y,i) == findprev(iszero, y_sp,i)
    end

    z_sp = sparsevec(Dict(1=>1, 5=>1, 8=>0, 10=>1))
    z = collect(z_sp)

    for i in keys(z)
        @test findnext(!iszero, z,i) == findnext(!iszero, z_sp,i)
        @test findprev(!iszero, z,i) == findprev(!iszero, z_sp,i)
    end

    # issue 32568
    for T = (UInt, BigInt)
        @test findnext(!iszero, x_sp, T(4)) isa keytype(x_sp)
        @test findnext(!iszero, x_sp, T(5)) isa keytype(x_sp)
        @test findprev(!iszero, x_sp, T(5)) isa keytype(x_sp)
        @test findprev(!iszero, x_sp, T(6)) isa keytype(x_sp)
        @test findnext(iseven, x_sp, T(4)) isa keytype(x_sp)
        @test findnext(iseven, x_sp, T(5)) isa keytype(x_sp)
        @test findprev(iseven, x_sp, T(4)) isa keytype(x_sp)
        @test findprev(iseven, x_sp, T(5)) isa keytype(x_sp)
        @test findnext(!iszero, z_sp, T(4)) isa keytype(z_sp)
        @test findnext(!iszero, z_sp, T(5)) isa keytype(z_sp)
        @test findprev(!iszero, z_sp, T(4)) isa keytype(z_sp)
        @test findprev(!iszero, z_sp, T(5)) isa keytype(z_sp)
    end

    # The sparse methods must actually extend `Base.findnext`/`Base.findprev` and skip
    # implicit zeros for predicates other than `!iszero`, e.g. the `!isequal(elt)` that
    # `Base.hash` uses to skip runs of equal values.
    @test SparseArrays.findnext === Base.findnext && SparseArrays.findprev === Base.findprev
    n = 10^9
    big = spzeros(n); big[1] = 1; big[n ÷ 2] = -0.0
    @test findprev(!isequal(0.0), big, n) == n ÷ 2
    @test findprev(!isequal(-0.0), big, n ÷ 2) == n ÷ 2 - 1   # implicit 0.0 is not isequal(-0.0)
    @test findnext(!isequal(0.0), big, 2) == n ÷ 2
    @test findnext(!isequal(0.0), big, n ÷ 2 + 1) === nothing
    # the predicate is evaluated once on the implicit zero and then on stored entries only
    calls = Ref(0)
    counted = x -> (calls[] += 1; !isequal(x, 0.0))
    @test findprev(counted, big, n) == n ÷ 2 && calls[] <= nnz(big) + 1
    calls[] = 0
    @test findnext(counted, big, 2) == n ÷ 2 && calls[] <= nnz(big) + 1
    for i in keys(y), f in (!isequal(0.0), !isequal(-0.0), !isequal(7.0), !isequal(NaN))
        @test findnext(f, y, i) == findnext(f, y_sp, i)
        @test findprev(f, y, i) == findprev(f, y_sp, i)
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

#testing the sparse matrix/vector access functions nnz, nzrange, rowvals, nonzeros
@testset "generic sparse matrix access functions" begin
    I = [1,3,4,5, 1,3,4,5, 1,3,4,5];
    J = [4,4,4,4, 5,5,5,5, 6,6,6,6];
    V = [14,34,44,54, 15,35,45,55, 16,36,46,56];
    A = sparse(I, J, V, 9, 9);
    AU = UpperTriangular(A)
    AL = LowerTriangular(A)
    b = SparseVector(9, I[1:4], V[1:4])
    c = view(A, :, 5)
    d = view(b, :)

    @testset "nnz $n" for (n, M, nz) in (("A", A, 12), ("AU", AU, 11), ("AL", AL, 3),
                                         ("b", b, 4), ("c", c, 4), ("d", d, 4))
        @test nnz(M) == nz
        @test_throws BoundsError nzrange(M, 0)
        @test_throws BoundsError nzrange(M, size(M, 2) + 1)
    end
    @testset "nzrange(A, $i)" for (i, nzr) in ((1,1:0),(4,1:4),(5,5:8),(6,9:12),(9,13:12))
        @test nzrange(A, i) == nzr
    end
    @testset "nzrange(AU, $i)" for (i, nzr) in ((2,1:0),(4,1:3),(5,5:8),(6,9:12),(8,13:12))
        @test nzrange(AU, i) == nzr
    end
    @testset "nzrange(AL, $i)" for (i, nzr) in ((3,1:0),(4,3:4),(5,8:8),(6,13:12),(7,13:12))
        @test nzrange(AL, i) == nzr
    end
    @test nzrange(b, 1) == 1:4
    @test nzrange(c, 1) == 1:4
    @test nzrange(d, 1) == 1:4

    @test rowvals(A) == I
    @test rowvals(AL) == I
    @test rowvals(AL) == I
    @test rowvals(b) == I[1:4]
    @test rowvals(c) == I[5:8]
    @test rowvals(d) == I[1:4]

    @test nonzeros(A) == V
    @test nonzeros(AU) == V
    @test nonzeros(AL) == V
    @test nonzeros(b) == V[1:4]
    @test nonzeros(c) == V[5:8]
    @test nonzeros(d) == V[1:4]
end

@testset "sprand" begin
    p=0.3; m=1000; n=2000;
    for s in 1:10
        # build a (dense) random matrix with randsubset + rand
        Random.seed!(s);
        v = randsubseq(1:m*n,p);
        x = zeros(m,n);
        x[v] .= rand(length(v));
        # redo the same with sprand
        Random.seed!(s);
        a = sprand(m,n,p);
        @test x == a
    end
end

@testset "copy a ReshapedArray of SparseMatrixCSC" begin
    A = sprand(20, 10, 0.2)
    rA = reshape(A, 10, 20)
    crA = copy(rA)
    @test reshape(crA, 20, 10) == A
end

@testset "SparseMatrixCSCView" begin
    A  = sprand(10, 10, 0.2)
    vA = view(A, :, 1:5) # a CSCView contains all rows and a UnitRange of the columns
    @test SparseArrays.getnzval(vA)  == SparseArrays.getnzval(A)
    @test SparseArrays.getrowval(vA) == SparseArrays.getrowval(A)
    @test SparseArrays.getcolptr(vA) == SparseArrays.getcolptr(A[:, 1:5])
end

@testset "fill! for SubArrays" begin
    a = sprand(10, 10, 0.2)
    b = copy(a)
    sa = view(a, 1:10, 2:3)
    sa_filled = fill!(sa, 0.0)
    # `fill!` should return the sub array instead of its parent.
    @test sa_filled === sa
    b[1:10, 2:3] .= 0.0
    @test a == b
    A = sparse([1], [1], [Vector{Float64}(undef, 3)], 3, 3)
    A[1,1] = [1.0, 2.0, 3.0]
    B = deepcopy(A)
    sA = view(A, 1:1, 1:2)
    fill!(sA, [4.0, 5.0, 6.0])
    for jj in 1:2
        B[1, jj] = [4.0, 5.0, 6.0]
    end
    @test A == B

    # https://github.com/JuliaSparse/SparseArrays.jl/pull/433
    struct Foo
       x::Int
    end
    Base.zero(::Type{Foo}) = Foo(0)
    Base.zero(::Foo) = zero(Foo)
    C = sparse([1], [1], [Foo(3)], 3, 3)
    sC = view(C, 1:1, 1:2)
    fill!(sC, zero(Foo))
    @test C[1:1, 1:2] == zeros(Foo, 1, 2)
end

using Base: swaprows!, swapcols!
@testset "swaprows!, swapcols!" begin
    S = sparse(
        [ 0   0  0  0  0   0
          0  -1  1  1  0   0
          0   0  0  1  1   0
          0   0  1  1  1  -1])

    for (f!, i, j) in
            ((swaprows!, 1, 2), # Test swapping rows where one row is fully sparse
             (swaprows!, 2, 3), # Test swapping rows of unequal length
             (swaprows!, 2, 4), # Test swapping non-adjacent rows
             (swapcols!, 1, 2), # Test swapping columns where one column is fully sparse
             (swapcols!, 2, 3), # Test swapping columns of unequal length
             (swapcols!, 2, 4)) # Test swapping non-adjacent columns
        Scopy = copy(S)
        Sdense = Array(S)
        f!(Scopy, i, j); f!(Sdense, i, j)
        @test Scopy == Sdense
    end

    for (A, i, j) in (
            (sparse([1.0  2.0  3.0;
                     0.0  0.0  0.0;
                     4.0  5.0  6.0]), 1, 2),
            (sparse([1.0  0.0  5.0;
                     0.0  2.0  0.0;
                     0.0  3.0  6.0;
                     7.0  4.0  0.0]), 1, 2),
            (sparse(reshape([1.0, 2.0, 3.0, 4.0, 0.0, 0.0], 6, 1)), 2, 6))
        Scopy = copy(A)
        Sdense = Array(A)
        swaprows!(Scopy, i, j); swaprows!(Sdense, i, j)
        @test Scopy == Sdense
    end
end

@testset "sprandn with type $T" for T in (Float64, Float32, Float16, ComplexF64, ComplexF32, ComplexF16)
    @test sprandn(T, 5, 5, 0.5) isa AbstractSparseMatrix{T}
end
@testset "sprandn with invalid type $T" for T in (AbstractFloat, Complex)
    @test_throws MethodError sprandn(T, 5, 5, 0.5)
end

@testset "count specializations" begin
    # count should throw for sparse arrays for which zero(eltype) does not exist
    @test_throws MethodError count(SparseMatrixCSC(2, 2, Int[1, 2, 3], Int[1, 2], Any[true, true]))
    @test_throws MethodError count(SparseVector(2, Int[1], Any[true]))
end

@testset "show" begin
    io = IOBuffer()

    A = spzeros(Float64, Int64, 0, 0)
    for (transform, showstring) in zip(
        (identity, adjoint, transpose), (
        "0×0 $SparseMatrixCSC{Float64, Int64} with 0 stored entries",
        "0×0 $Adjoint{Float64, $SparseMatrixCSC{Float64, Int64}} with 0 stored entries",
        "0×0 $Transpose{Float64, $SparseMatrixCSC{Float64, Int64}} with 0 stored entries"
        ))
        show(io, MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
    end

    A = sparse(Int64[1], Int64[1], [1.0])
    for (transform, showstring) in zip(
        (identity, adjoint, transpose), (
        "1×1 $SparseMatrixCSC{Float64, Int64} with 1 stored entry:\n 1.0",
        "1×1 $Adjoint{Float64, $SparseMatrixCSC{Float64, Int64}} with 1 stored entry:\n 1.0",
        "1×1 $Transpose{Float64, $SparseMatrixCSC{Float64, Int64}} with 1 stored entry:\n 1.0",
        ))
        show(io, MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
    end

    A = spzeros(Float32, Int64, 2, 2)
    for (transform, showstring) in zip(
        (identity, adjoint, transpose), (
        "2×2 $SparseMatrixCSC{Float32, Int64} with 0 stored entries:\n  ⋅    ⋅ \n  ⋅    ⋅ ",
        "2×2 $Adjoint{Float32, $SparseMatrixCSC{Float32, Int64}} with 0 stored entries:\n  ⋅    ⋅ \n  ⋅    ⋅ ",
        "2×2 $Transpose{Float32, $SparseMatrixCSC{Float32, Int64}} with 0 stored entries:\n  ⋅    ⋅ \n  ⋅    ⋅ ",
        ))
        show(io, MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
    end

    A = sparse(Int64[1, 1], Int64[1, 2], [1.0, 2.0])
    for (transform, showstring, braille) in zip(
        (identity, adjoint, transpose), (
        "1×2 $SparseMatrixCSC{Float64, Int64} with 2 stored entries:\n 1.0  2.0",
        "2×1 $Adjoint{Float64, $SparseMatrixCSC{Float64, Int64}} with 2 stored entries:\n 1.0\n 2.0",
        "2×1 $Transpose{Float64, $SparseMatrixCSC{Float64, Int64}} with 2 stored entries:\n 1.0\n 2.0",
        ),
        ("⎡⠁⠈⎤\n" *
         "⎣⠀⠀⎦",
         "⎡⠁⠀⎤\n" *
         "⎣⡀⠀⎦",
         "⎡⠁⠀⎤\n" *
         "⎣⡀⠀⎦"))
        show(io, MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test String(take!(io)) == braille
    end

    # every 1-dot braille pattern
    for (i, b) in enumerate(split("⠁⠂⠄⡀⠈⠐⠠⢀", ""))
        A = spzeros(Int64, Int64, 8, 4)
        A[mod1(i, 4), (i - 1) ÷ 4 + 1] = 1
        _show_with_braille_patterns(convert(IOContext, io), A)
        out = String(take!(io))
        @test occursin(b, out) == true
        for c in split("⠁⠂⠄⡀⠈⠐⠠⢀", "")
            b == c && continue
            @test occursin(c, out) == false
        end
    end

    # empty braille pattern Char(10240)
    A = spzeros(Int64, Int64, 4, 2)
    for transform in (identity, adjoint, transpose)
        expected = "⎡" * Char(10240)^2 * "⎤\n⎣" * Char(10240)^2 * "⎦"
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test String(take!(io)) == expected
    end

    A = sparse(Int64[1, 2, 4, 2, 3], Int64[1, 1, 1, 2, 2], Int64[1, 1, 1, 1, 1], 4, 2)
    for (transform, showstring, braille) in zip(
        (identity, adjoint, transpose), (
        "4×2 $SparseMatrixCSC{Int64, Int64} with 5 stored entries:\n 1  ⋅\n 1  1\n ⋅  1\n 1  ⋅",
        "2×4 $Adjoint{Int64, $SparseMatrixCSC{Int64, Int64}} with 5 stored entries:\n 1  1  ⋅  1\n ⋅  1  1  ⋅",
        "2×4 $Transpose{Int64, $SparseMatrixCSC{Int64, Int64}} with 5 stored entries:\n 1  1  ⋅  1\n ⋅  1  1  ⋅",
        ),
        ("⎡⠅⠠⎤\n" *
         "⎣⡀⠐⎦",
         "⎡⠉⠈⎤\n" *
         "⎣⢀⡀⎦",
         "⎡⠉⠈⎤\n" *
         "⎣⢀⡀⎦"))
        show(io, MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test String(take!(io)) == braille
    end

    A = sparse(Int64[1, 3, 2, 4], Int64[1, 1, 2, 2], Int64[1, 1, 1, 1], 7, 3)
    for (transform, showstring, braille) in zip(
        (identity, adjoint, transpose), (
        "7×3 $SparseMatrixCSC{Int64, Int64} with 4 stored entries:\n 1  ⋅  ⋅\n ⋅  1  ⋅\n 1  ⋅  ⋅\n ⋅  1  ⋅\n ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅",
        "3×7 $Adjoint{Int64, $SparseMatrixCSC{Int64, Int64}} with 4 stored entries:\n 1  ⋅  1  ⋅  ⋅  ⋅  ⋅\n ⋅  1  ⋅  1  ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅",
        "3×7 $Transpose{Int64, $SparseMatrixCSC{Int64, Int64}} with 4 stored entries:\n 1  ⋅  1  ⋅  ⋅  ⋅  ⋅\n ⋅  1  ⋅  1  ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅",
        ),
        ("⎡⢕⠀⎤\n" *
         "⎣⠀⠀⎦",
         "⎡⢁⢁⠀⠀⎤\n" *
         "⎣⠀⠀⠀⠀⎦",
         "⎡⢁⢁⠀⠀⎤\n" *
         "⎣⠀⠀⠀⠀⎦"))
        show(io, MIME"text/plain"(), transform(A))
        @test String(take!(io)) == showstring
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test String(take!(io)) == braille
    end

    A = sparse(Int64[1:10;], Int64[1:10;], fill(Float64(1), 10))
    brailleString = "⎡⠑⢄⠀⠀⠀⎤\n" *
                    "⎢⠀⠀⠑⢄⠀⎥\n" *
                    "⎣⠀⠀⠀⠀⠑⎦"
    for transform in (identity, adjoint, transpose)
        _show_with_braille_patterns(convert(IOContext, io), transform(A))
        @test String(take!(io)) == brailleString
    end

    # Issue #30589
    @test repr("text/plain", sparse([true true])) == "1×2 $SparseMatrixCSC{Bool, $Int} with 2 stored entries:\n 1  1"

    function _filled_sparse(m::Integer, n::Integer)
        C = CartesianIndices((m, n))[:]
        Is = [Int64(x[1]) for x in C]
        Js = [Int64(x[2]) for x in C]
        return sparse(Is, Js, true, m, n)
    end

    # vertical scaling
    ioc = IOContext(io, :displaysize => (5, 80), :limit => true)
    _show_with_braille_patterns(ioc, _filled_sparse(10, 10))
    @test String(take!(io)) == "⎡⣿⣿⎤\n" *
                               "⎣⣿⣿⎦"

    _show_with_braille_patterns(ioc, _filled_sparse(20, 10))
    @test String(take!(io)) == "⎡⣿⣿⎤\n" *
                               "⎣⣿⣿⎦"

    # horizontal scaling
    ioc = IOContext(io, :displaysize => (80, 4), :limit => true)
    _show_with_braille_patterns(ioc, _filled_sparse(8, 8))
    @test String(take!(io)) == "⎡⣿⣿⎤\n" *
                               "⎣⣿⣿⎦"

    _show_with_braille_patterns(ioc, _filled_sparse(8, 16))
    @test String(take!(io)) == "⎡⣿⣿⎤\n" *
                               "⎣⣿⣿⎦"

    # respect IOContext while displaying J
    I, J, V = shuffle(1:50), shuffle(1:50), [1:50;]
    S = sparse(I, J, V)
    I, J, V = I[sortperm(J)], sort(J), V[sortperm(J)]
    @test repr(S) == "sparse($I, $J, $V, $(size(S,1)), $(size(S,2)))"
    limctxt(x) = repr(x, context=:limit=>true)
    expstr = "sparse($(limctxt(I)), $(limctxt(J)), $(limctxt(V)), $(size(S,1)), $(size(S,2)))"
    @test limctxt(S) == expstr
end

@testset "issparse for specialized matrix types" begin
    m = sprand(10, 10, 0.1)
    @test issparse(Symmetric(m))
    @test issparse(Hermitian(m))
    @test issparse(LowerTriangular(m))
    @test issparse(LinearAlgebra.UnitLowerTriangular(m))
    @test issparse(UpperTriangular(m))
    @test issparse(LinearAlgebra.UnitUpperTriangular(m))
    @test issparse(adjoint(m))
    @test issparse(transpose(m))
    @test issparse(Symmetric(Array(m))) == false
    @test issparse(Hermitian(Array(m))) == false
    @test issparse(LowerTriangular(Array(m))) == false
    @test issparse(LinearAlgebra.UnitLowerTriangular(Array(m))) == false
    @test issparse(UpperTriangular(Array(m))) == false
    @test issparse(LinearAlgebra.UnitUpperTriangular(Array(m))) == false
    @test issparse(Base.ReshapedArray(m, (20, 5), ()))
    @test issparse(@view m[1:3, :])

    # greater nesting
    @test issparse(Symmetric(UpperTriangular(m)))
    @test issparse(Symmetric(UpperTriangular(Array(m)))) == false
end

@testset "equality ==" begin
    A1 = sparse(1.0I, 10, 10)
    A2 = sparse(1.0I, 10, 10)
    nonzeros(A1)[end]=0
    @test A1!=A2
    nonzeros(A1)[end]=1
    @test A1==A2
    A1[1:4,end] .= 1
    @test A1!=A2
    nonzeros(A1)[end-4:end-1].=0
    @test A1==A2
    A2[1:4,end-1] .= 1
    @test A1!=A2
    nonzeros(A2)[end-5:end-2].=0
    @test A1==A2
    A2[2:3,1] .= 1
    @test A1!=A2
    nonzeros(A2)[2:3].=0
    @test A1==A2
    A1[2:5,1] .= 1
    @test A1!=A2
    nonzeros(A1)[2:5].=0
    @test A1==A2
    @test sparse([1,1,0])!=sparse([0,1,1])
end

@testset "expandptr" begin
    local A = sparse(1.0I, 5, 5)
    @test SparseArrays.expandptr(getcolptr(A)) == 1:5
    A[1,2] = 1
    @test SparseArrays.expandptr(getcolptr(A)) == [1; 2; 2; 3; 4; 5]
    @test_throws ArgumentError SparseArrays.expandptr([2; 3])
end

@testset "sparse! and spzeros!" begin
    using SparseArrays: sparse!, spzeros!, getcolptr, getrowval, nonzeros

    function allocate_arrays(m, n)
        N = round(Int, 0.5 * m * n)
        Tv, Ti = Float64, Int
        I = Ti[rand(1:m) for _ in 1:N]; I = Ti[I; I]
        J = Ti[rand(1:n) for _ in 1:N]; J = Ti[J; J]
        V = Tv.(I)
        csrrowptr = Vector{Ti}(undef, m + 1)
        csrcolval = Vector{Ti}(undef, length(I))
        csrnzval = Vector{Tv}(undef, length(I))
        klasttouch = Vector{Ti}(undef, n)
        csccolptr = Vector{Ti}(undef, n + 1)
        cscrowval = Vector{Ti}()
        cscnzval = Vector{Tv}()
        return I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr, cscrowval, cscnzval
    end

    for (m, n) in ((10, 5), (5, 10), (10, 10))
        # Passing csr vectors
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval = allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval)
        @test S == S!
        @test same_structure(S, S!)

        I, J, _, klasttouch, csrrowptr, csrcolval = allocate_arrays(m, n)
        S  = spzeros(I, J, m, n)
        S! = spzeros!(Float64, I, J, m, n, klasttouch, csrrowptr, csrcolval)
        @test S == S!
        @test iszero(S!)
        @test same_structure(S, S!)

        # Passing csr vectors + csccolptr
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr = allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr)
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === csccolptr

        I, J, _, klasttouch, csrrowptr, csrcolval, _, csccolptr = allocate_arrays(m, n)
        S  = spzeros(I, J, m, n)
        S! = spzeros!(Float64, I, J, m, n, klasttouch, csrrowptr, csrcolval, csccolptr)
        @test S == S!
        @test iszero(S!)
        @test same_structure(S, S!)
        @test getcolptr(S!) === csccolptr

        # Passing csr vectors, and csc vectors
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr, cscrowval, cscnzval =
            allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval,
                     csccolptr, cscrowval, cscnzval)
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === csccolptr
        @test getrowval(S!) === cscrowval
        @test nonzeros(S!) === cscnzval

        I, J, _, klasttouch, csrrowptr, csrcolval, _, csccolptr, cscrowval, cscnzval =
            allocate_arrays(m, n)
        S  = spzeros(I, J, m, n)
        S! = spzeros!(Float64, I, J, m, n, klasttouch, csrrowptr, csrcolval,
                      csccolptr, cscrowval, cscnzval)
        @test S == S!
        @test iszero(S!)
        @test same_structure(S, S!)
        @test getcolptr(S!) === csccolptr
        @test getrowval(S!) === cscrowval
        @test nonzeros(S!) === cscnzval

        # Passing csr vectors, and csc vectors of insufficient lengths
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr, cscrowval, cscnzval =
            allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval,
                     resize!(csccolptr, 0), resize!(cscrowval, 0), resize!(cscnzval, 0))
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === csccolptr
        @test getrowval(S!) === cscrowval
        @test nonzeros(S!) === cscnzval

        I, J, _, klasttouch, csrrowptr, csrcolval, _, csccolptr, cscrowval, cscnzval =
            allocate_arrays(m, n)
        S  = spzeros(I, J, m, n)
        S! = spzeros!(Float64, I, J, m, n, klasttouch, csrrowptr, csrcolval,
                      resize!(csccolptr, 0), resize!(cscrowval, 0), resize!(cscnzval, 0))
        @test S == S!
        @test iszero(S!)
        @test same_structure(S, S!)
        @test getcolptr(S!) === csccolptr
        @test getrowval(S!) === cscrowval
        @test nonzeros(S!) === cscnzval

        # Passing csr vectors, and csc vectors aliased with I, J, V
        I, J, V, klasttouch, csrrowptr, csrcolval, csrnzval = allocate_arrays(m, n)
        S  = sparse(I, J, V, m, n)
        S! = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V

        I, J, V, klasttouch, csrrowptr, csrcolval = allocate_arrays(m, n)
        S  = spzeros(I, J, m, n)
        S! = spzeros!(Float64, I, J, m, n, klasttouch, csrrowptr, csrcolval, I, J, V)
        @test S == S!
        @test iszero(S!)
        @test same_structure(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V

        # Test reuse of I, J, V for the matrix buffers in
        # sparse!(I, J, V), sparse!(I, J, V, m, n), sparse!(I, J, V, m, n, combine),
        # spzeros!(T, I, J), and spzeros!(T, I, J, m, n).
        I, J, V = allocate_arrays(m, n)
        S = sparse(I, J, V)
        S! = sparse!(I, J, V)
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V
        I, J, V = allocate_arrays(m, n)
        S = sparse(I, J, V, 2m, 2n)
        S! = sparse!(I, J, V, 2m, 2n)
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V
        I, J, V = allocate_arrays(m, n)
        S = sparse(I, J, V, 2m, 2n, *)
        S! = sparse!(I, J, V, 2m, 2n, *)
        @test S == S!
        @test same_structure(S, S!)
        @test getcolptr(S!) === I
        @test getrowval(S!) === J
        @test nonzeros(S!) === V
        for T in (Float32, Float64)
            I, J, = allocate_arrays(m, n)
            S = spzeros(T, I, J)
            S! = spzeros!(T, I, J)
            @test S == S!
            @test same_structure(S, S!)
            @test eltype(S) == eltype(S!) == T
            @test getcolptr(S!) === I
            @test getrowval(S!) === J
            I, J, = allocate_arrays(m, n)
            S = spzeros(T, I, J, 2m, 2n)
            S! = spzeros!(T, I, J, 2m, 2n)
            @test S == S!
            @test same_structure(S, S!)
            @test eltype(S) == eltype(S!) == T
            @test getcolptr(S!) === I
            @test getrowval(S!) === J
        end
    end
end

@testset "reverse" begin
    @testset "$name" for (name, S) in (("standard", sparse([2,2,4], [1,2,5], [-19, 73, -7])),
                            ("sprand", sprand(Float32, 15, 18, 0.2)),
                            ("zeros", spzeros(Int8, 20, 40)),
                            ("fixed", SparseArrays.fixed(sparse([2,2,4], [1,2,5], [-19, 73, -7]))))
        w = collect(S)
        revS = reverse(S)
        @test revS == reverse(w)
        @test nnz(revS) == nnz(S)
        if S isa SparseMatrixCSC
            S2 = copy(S)
            reverse!(S2)
            @test S2 == revS
            @test nnz(S2) == nnz(S)
        end
        for dims in 1:2
            revS = reverse(S; dims)
            @test revS == reverse(w; dims)
            @test nnz(revS) == nnz(S)
            if S isa SparseMatrixCSC
                S2 = copy(S)
                reverse!(S2; dims)
                @test S2 == revS
                @test nnz(S2) == nnz(S)
            end
        end
        revS = reverse(S, dims=(1,2))
        @test revS == reverse(w, dims=(1,2))
        @test nnz(revS) == nnz(S)
        if S isa SparseMatrixCSC
            S2 = copy(S)
            reverse!(S2, dims=(1,2))
            @test S2 == revS
            @test nnz(S2) == nnz(S)
        end
    end
end

end
