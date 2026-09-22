
# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseConcatenationTests

using Test
using SparseArrays
using LinearAlgebra

@testset "concatenation tests" begin
    sp33 = sparse(1.0I, 3, 3)
    se33 = SparseMatrixCSC{Float64}(I, 3, 3)
    do33 = fill(1.,3)
    @testset "horizontal concatenation" begin
        @test [se33 se33] == [Array(se33) Array(se33)]
        @test length(nonzeros([sp33 0I])) == 3
    end

    @testset "vertical concatenation" begin
        @test [se33; se33] == [Array(se33); Array(se33)]
        se33_32bit = convert(SparseMatrixCSC{Float32,Int32}, se33)
        @test [se33; se33_32bit] == [Array(se33); Array(se33_32bit)]
        @test length(nonzeros([sp33; 0I])) == 3
    end

    se44 = sparse(1.0I, 4, 4)
    sz42 = spzeros(4, 2)
    sz41 = spzeros(4, 1)
    sz34 = spzeros(3, 4)
    se77 = sparse(1.0I, 7, 7)
    @testset "h+v concatenation" begin
        @test @inferred(hvcat((3, 2), se44, sz42, sz41, sz34, se33)) == se77 # [se44 sz42 sz41; sz34 se33]
        @test length(nonzeros([sp33 0I; 1I 0I])) == 6
    end

    @testset "blockdiag concatenation" begin
        @test blockdiag(se33, se33) == sparse(1:6,1:6,fill(1.,6))
        @test blockdiag() == spzeros(0, 0)
        @test nnz(blockdiag()) == 0
    end

    @testset "Diagonal of sparse matrices" begin
        s = sparse([1 2; 3 4])
        D = Diagonal([s, s])
        @test D[1, 1] == s
        @test D[1, 2] == zero(s)
        @test isa(D[2, 1], SparseMatrixCSC)
    end

    @testset "concatenation promotion" begin
        sz41_f32 = spzeros(Float32, 4, 1)
        se33_i32 = sparse(Int32(1)I, 3, 3)
        @test [se44 sz42 sz41_f32; sz34 se33_i32] == se77
    end

    @testset "mixed sparse-dense concatenation" begin
        sz33 = spzeros(3, 3)
        de33 = Matrix(1.0I, 3, 3)
        @test [se33 de33; sz33 se33] == Array([se33 se33; sz33 se33 ])
    end

    # check splicing + concatenation on random instances, with nested vcat and also side-checks sparse ref
    @testset "splicing + concatenation on random instances" begin
        for i = 1 : 10
            a = sprand(5, 4, 0.5)
            @test [a[1:2,1:2] a[1:2,3:4]; a[3:5,1] [a[3:4,2:4]; a[5:5,2:4]]] == a
        end
    end

    # should all yield sparse arrays
    @testset "concatenations of combinations of special and other matrix types" begin
        N = 4
        diagmat = Diagonal(1:N)
        bidiagmat = Bidiagonal(1:N, 1:(N-1), :U)
        tridiagmat = Tridiagonal(1:(N-1), 1:N, 1:(N-1))
        symtridiagmat = SymTridiagonal(1:N, 1:(N-1))
        specialmats = (diagmat, bidiagmat, tridiagmat, symtridiagmat)
        # Test concatenating pairwise combinations of special matrices with sparse matrices,
        # dense matrices, or dense vectors
        spmat = spdiagm(0 => fill(1., N))
        dmat  = Array(spmat)
        spvec = sparse(fill(1., N))
        dvec  = Array(spvec)
        for specialmat in specialmats
            # --> Tests applicable only to pairs of matrices
            @test issparse(vcat(specialmat, spmat))
            @test issparse(vcat(spmat, specialmat))
            @test sparse_vcat(specialmat, dmat)::SparseMatrixCSC == vcat(specialmat, spmat)
            @test sparse_vcat(dmat, specialmat)::SparseMatrixCSC == vcat(spmat, specialmat)
            # --> Tests applicable also to pairs including vectors
            for (smatorvec, dmatorvec) in ((spmat, dmat), (spvec, dvec))
                @test issparse(hcat(specialmat, smatorvec))
                @test sparse_hcat(specialmat, dmatorvec)::SparseMatrixCSC == hcat(specialmat, smatorvec)
                @test issparse(hcat(smatorvec, specialmat))
                @test sparse_hcat(dmatorvec, specialmat)::SparseMatrixCSC == hcat(smatorvec, specialmat)
                @test issparse(hvcat((2,), specialmat, smatorvec))
                @test sparse_hvcat((2,), specialmat, dmatorvec)::SparseMatrixCSC == hvcat((2,), specialmat, smatorvec)
                @test issparse(hvcat((2,), smatorvec, specialmat))
                @test sparse_hvcat((2,), dmatorvec, specialmat)::SparseMatrixCSC == hvcat((2,), smatorvec, specialmat)
                @test issparse(cat(specialmat, smatorvec; dims=(1,2)))
                @test issparse(cat(smatorvec, specialmat; dims=(1,2)))
            end
        end
    end

    # Test that concatenations of annotated sparse/special matrix types with other matrix
    # types yield sparse arrays, and that the code which effects that does not make concatenations
    # strictly involving un/annotated dense matrices yield sparse arrays
    @testset "concatenations of annotated types" begin
        N = 4
        # The tested annotation types
        testfull = Bool(parse(Int,(get(ENV, "JULIA_TESTFULL", "0"))))
        utriannotations = (UpperTriangular, UnitUpperTriangular)
        ltriannotations = (LowerTriangular, UnitLowerTriangular)
        triannotations = (utriannotations..., ltriannotations...)
        symannotations = (Symmetric, Hermitian)
        annotations = testfull ? (triannotations..., symannotations...) : (LowerTriangular, Symmetric)
        # Concatenations involving these types, un/annotated, should yield sparse arrays
        spvec = spzeros(N)
        spmat = sparse(1.0I, N, N)
        diagmat = Diagonal(1:N)
        bidiagmat = Bidiagonal(1:N, 1:(N-1), :U)
        tridiagmat = Tridiagonal(1:(N-1), 1:N, 1:(N-1))
        symtridiagmat = SymTridiagonal(1:N, 1:(N-1))
        sparseconcatmats = testfull ? (spmat, diagmat, bidiagmat, tridiagmat, symtridiagmat) : (spmat, diagmat)
        # Concatenations involving strictly these types, un/annotated, should yield dense arrays
        densevec = Array(spvec)
        densemat = Array(spmat)
        # Annotated collections
        annodmats = [annot(densemat) for annot in annotations]
        annospcmats = [annot(spmat) for annot in annotations]
        # Test that concatenations of pairwise combinations of annotated sparse/special
        # yield sparse matrices
        for annospcmata in annospcmats, annospcmatb in annospcmats
            @test issparse(vcat(annospcmata, annospcmatb))
            @test issparse(hcat(annospcmata, annospcmatb))
            @test issparse(hvcat((2,), annospcmata, annospcmatb))
            @test issparse(cat(annospcmata, annospcmatb; dims=(1,2)))
        end
        # Test that concatenations of pairwise combinations of annotated sparse/special
        # matrices and other matrix/vector types yield sparse matrices
        for annospcmat in annospcmats
            # --> Tests applicable to pairs including only matrices
            for othermat in (densemat, annodmats..., sparseconcatmats...)
                @test issparse(vcat(annospcmat, othermat))
                @test issparse(vcat(othermat, annospcmat))
            end
            for (smat, dmat) in zip(annospcmats, annodmats), specialmat in sparseconcatmats
                @test sparse_hcat(dmat, specialmat)::SparseMatrixCSC == hcat(smat, specialmat)
                @test sparse_hcat(specialmat, dmat)::SparseMatrixCSC == hcat(specialmat, smat)
                @test sparse_vcat(dmat, specialmat)::SparseMatrixCSC == vcat(smat, specialmat)
                @test sparse_vcat(specialmat, dmat)::SparseMatrixCSC == vcat(specialmat, smat)
                @test sparse_hvcat((2,), dmat, specialmat)::SparseMatrixCSC == hvcat((2,), smat, specialmat)
                @test sparse_hvcat((2,), specialmat, dmat)::SparseMatrixCSC == hvcat((2,), specialmat, smat)
            end
            # --> Tests applicable to pairs including other vectors or matrices
            for other in (spvec, densevec, densemat, annodmats..., sparseconcatmats...)
                @test issparse(hcat(annospcmat, other))
                @test issparse(hcat(other, annospcmat))
                @test issparse(hvcat((2,), annospcmat, other))
                @test issparse(hvcat((2,), other, annospcmat))
                @test issparse(cat(annospcmat, other; dims=(1,2)))
                @test issparse(cat(other, annospcmat; dims=(1,2)))
            end
        end
        # The preceding tests should cover multi-way combinations of those types, but for good
        # measure test a few multi-way combinations involving those types
        @test issparse(vcat(spmat, densemat, annospcmats[1], annodmats[2]))
        @test issparse(vcat(densemat, spmat, annodmats[1], annospcmats[2]))
        @test issparse(hcat(spvec, annodmats[1], annospcmats[1], densevec, diagmat))
        @test issparse(hcat(annodmats[2], annospcmats[2], spvec, densevec, diagmat))
        @test issparse(hvcat((5,), diagmat, densevec, spvec, annodmats[1], annospcmats[1]))
        @test issparse(hvcat((5,), spvec, annodmats[2], diagmat, densevec, annospcmats[2]))
        @test issparse(cat(annodmats[1], diagmat, annospcmats[2], densevec, spvec; dims=(1,2)))
        @test issparse(cat(spvec, diagmat, densevec, annospcmats[1], annodmats[2]; dims=(1,2)))
    end

    @testset "hcat and vcat involving UniformScaling" begin
        @test_throws ArgumentError hcat(I)
        @test_throws ArgumentError [I I]
        @test_throws ArgumentError vcat(I)
        @test_throws ArgumentError [I; I]
        @test_throws ArgumentError [I I; I]

        A = SparseMatrixCSC(rand(3,4))
        B = SparseMatrixCSC(rand(3,3))
        C = SparseMatrixCSC(rand(0,3))
        D = SparseMatrixCSC(rand(2,0))
        E = SparseMatrixCSC(rand(1,3))
        F = SparseMatrixCSC(rand(3,1))
        α = rand()
        @test (hcat(A, 2I, I(3)))::SparseMatrixCSC == hcat(A, Matrix(2I, 3, 3), Matrix(I, 3, 3))
        @test (hcat(E, α))::SparseMatrixCSC == hcat(E, [α])
        @test (hcat(E, α, 2I))::SparseMatrixCSC == hcat(E, [α], fill(2, 1, 1))
        @test (vcat(A, 2I))::SparseMatrixCSC == (vcat(A, 2I(4)))::SparseMatrixCSC == vcat(A, Matrix(2I, 4, 4))
        @test (vcat(F, α))::SparseMatrixCSC == vcat(F, [α])
        @test (vcat(F, α, 2I))::SparseMatrixCSC == (vcat(F, α, 2I(1)))::SparseMatrixCSC == vcat(F, [α], fill(2, 1, 1))
        @test (hcat(C, 2I))::SparseMatrixCSC == C
        @test_throws DimensionMismatch hcat(C, α)
        @test (vcat(D, 2I))::SparseMatrixCSC == D
        @test_throws DimensionMismatch vcat(D, α)
        @test (hcat(I, 3I, A, 2I))::SparseMatrixCSC == hcat(Matrix(I, 3, 3), Matrix(3I, 3, 3), A, Matrix(2I, 3, 3))
        @test (vcat(I, 3I, A, 2I))::SparseMatrixCSC == vcat(Matrix(I, 4, 4), Matrix(3I, 4, 4), A, Matrix(2I, 4, 4))
        @test (hvcat((2,1,2), B, 2I, I(6), 3I, 4I))::SparseMatrixCSC ==
            hvcat((2,1,2), B, Matrix(2I, 3, 3), Matrix(I, 6, 6), Matrix(3I, 3, 3), Matrix(4I, 3, 3))
        @test hvcat((3,1), C, C, I, 3I)::SparseMatrixCSC == hvcat((2,1), C, C, Matrix(3I, 6, 6))
        @test hvcat((2,2,2), I, 2I, 3I, 4I, C, C)::SparseMatrixCSC ==
            hvcat((2,2,2), Matrix(I, 3, 3), Matrix(2I, 3, 3), Matrix(3I, 3, 3), Matrix(4I, 3, 3), C, C)
        @test hvcat((2,2,4), C, C, I(3), 2I, 3I, 4I, 5I, D)::SparseMatrixCSC ==
            hvcat((2,2,4), C, C, Matrix(I, 3, 3), Matrix(2I, 3, 3),
                Matrix(3I, 2, 2), Matrix(4I, 2, 2), Matrix(5I, 2, 2), D)
        @test (hvcat((2,3,2), B, 2I(3), C, C, I, 3I, 4I))::SparseMatrixCSC ==
            hvcat((2,2,2), B, Matrix(2I, 3, 3), C, C, Matrix(3I, 3, 3), Matrix(4I, 3, 3))
        @test hvcat((3,2,1), C, C, I, B, 3I(3), 2I)::SparseMatrixCSC ==
            hvcat((2,2,1), C, C, B, Matrix(3I, 3, 3), Matrix(2I, 6, 6))
        @test (hvcat((1,2), A, E, α))::SparseMatrixCSC == hvcat((1,2), A, E, [α]) == hvcat((1,2), A, E, α*I)
        @test (hvcat((2,2), α, E, F, 3I))::SparseMatrixCSC == hvcat((2,2), [α], E, F, Matrix(3I, 3, 3))
        @test (hvcat((2,2), 3I, F, E, α))::SparseMatrixCSC == hvcat((2,2), Matrix(3I, 3, 3), F, E, [α])
    end
end


# Test that concatenations of combinations of sparse matrices with sparse matrices or dense
# matrices/vectors yield sparse arrays
@testset "sparse and dense concatenations" begin
    N = 4
    densevec = fill(1., N)
    densemat = diagm(0 => densevec)
    spmat = spdiagm(0 => densevec)
    # Test that concatenations of pairs of sparse matrices yield sparse arrays
    @test issparse(vcat(spmat, spmat))
    @test issparse(hcat(spmat, spmat))
    @test issparse(@inferred(hvcat((2,), spmat, spmat)))
    @test issparse(cat(spmat, spmat; dims=(1,2)))
    # Test that concatenations of a sparse matrice with a dense matrix/vector yield sparse arrays
    @test issparse(vcat(spmat, densemat))
    @test issparse(vcat(densemat, spmat))
    for densearg in (densevec, densemat)
        @test issparse(hcat(spmat, densearg))
        @test issparse(hcat(densearg, spmat))
        @test issparse(hvcat((2,), spmat, densearg))
        @test issparse(hvcat((2,), densearg, spmat))
        @test issparse(cat(spmat, densearg; dims=(1,2)))
        @test issparse(cat(densearg, spmat; dims=(1,2)))
    end
end


@testset "issue #19304" begin
    @inferred hcat(sparse(rand(2,1)), I)
    @inferred hcat(sparse(rand(2,1)), 1.0I)
    @inferred hcat(sparse(rand(2,1)), Matrix(I, 2, 2))
    @inferred hcat(sparse(rand(2,1)), Matrix(1.0I, 2, 2))
end


@testset "Concatenation" begin
    let m = 80, n = 100
        A = Vector{SparseVector{Float64,Int}}(undef, n)
        tnnz = 0
        for i = 1:length(A)
            A[i] = sprand(m, 0.3)
            tnnz += nnz(A[i])
        end

        H = hcat(A...)
        @test isa(H, SparseMatrixCSC{Float64,Int})
        @test size(H) == (m, n)
        @test nnz(H) == tnnz
        Hr = zeros(m, n)
        for j = 1:n
            Hr[:,j] = Array(A[j])
        end
        @test Array(H) == Hr

        V = vcat(A...)
        @test isa(V, SparseVector{Float64,Int})
        @test length(V) == m * n
        Vr = vec(Hr)
        @test Array(V) == Vr
        Vnum = vcat(A..., zero(Float64))
        Vnum2 = sparse_vcat(map(Array, A)..., zero(Float64))
        @test Vnum isa SparseVector{Float64,Int}
        @test Vnum2 isa SparseVector{Float64,Int}
        @test length(Vnum) == length(Vnum2) == m*n + 1
        @test Array(Vnum) == Array(Vnum2) == [Vr; 0]
        Vnum = vcat(zero(Float64), A...)
        Vnum2 = sparse_vcat(zero(Float64), map(Array, A)...)
        @test Vnum isa SparseVector{Float64,Int}
        @test Vnum2 isa SparseVector{Float64,Int}
        @test length(Vnum) == length(Vnum2) == m*n + 1
        @test Array(Vnum) == Array(Vnum2) == [0; Vr]
        # case with rowwise a Number as first element, should still yield a sparse matrix
        x = sparsevec([1], [3.0], 1)
        X = [3.0 x; 3.0 x]
        @test issparse(X)
    end

    @testset "stack (#498)" begin
        A = [sprand(80, 0.3) for _ in 1:100]
        H = hcat(A...)
        S = @inferred stack(A)
        @test S isa SparseMatrixCSC{Float64,Int}
        @test S == H
        S1 = stack(A; dims=1)
        @test S1 isa SparseMatrixCSC{Float64,Int}
        @test S1 == permutedims(H)
        @test_throws ArgumentError stack(A; dims=3)
        @test stack(x for x in A if true) == H
        @test stack(eachcol(H)) == H
        # slices with different element and index types promote
        SB = stack([sparsevec(Int32[1], Int32[2], 3), sparsevec([3], [0.5], 3)])
        @test SB isa SparseMatrixCSC{Float64,Int}
        @test SB == [2 0; 0 0; 0 0.5]
        # a container with more than one axis stacks into a dense array
        @test stack(reshape(A, 2, :)) == reshape(Array(H), 80, 2, :)
        @test_throws ArgumentError stack(SparseVector{Float64,Int}[])
        @test_throws DimensionMismatch stack([sparsevec([1], [1.0], 3), sparsevec([1], [1.0], 4)])
    end

@testset "concatenation of sparse vectors with other types" begin
        # Test that concatenations of combinations of sparse vectors with various other
        # matrix/vector types yield sparse arrays
        let N = 4
            spvec = spzeros(N)
            spmat = spzeros(N, 1)
            densevec = fill(1., N)
            densemat = fill(1., N, 1)
            diagmat = Diagonal(densevec)
            # inferrability (https://github.com/JuliaSparse/SparseArrays.jl/pull/92)
            cat_with_constdims(args...) = cat(args...; dims=(1,2))
            # Test that concatenations of pairwise combinations of sparse vectors with dense
            # vectors/matrices, sparse matrices, or special matrices yield sparse arrays
            for othervecormat in (densevec, densemat, spmat)
                @test issparse(vcat(spvec, othervecormat))
                @test issparse(vcat(othervecormat, spvec))
            end
            for othervecormat in (densevec, densemat, spmat, diagmat)
                @test issparse(hcat(spvec, othervecormat))
                @test issparse(hcat(othervecormat, spvec))
                @test issparse(hvcat((2,), spvec, othervecormat))
                @test issparse(hvcat((2,), othervecormat, spvec))
                @test issparse(cat(spvec, othervecormat; dims=(1,2)))
                @test issparse(cat(othervecormat, spvec; dims=(1,2)))

                @test issparse(@inferred cat_with_constdims(spvec, othervecormat))
                @test issparse(@inferred cat_with_constdims(othervecormat, spvec))
            end
            # The preceding tests should cover multi-way combinations of those types, but for good
            # measure test a few multi-way combinations involving those types
            @test issparse(vcat(spvec, densevec, spmat, densemat))
            @test issparse(vcat(densevec, spvec, densemat, spmat))
            @test issparse(hcat(spvec, densemat, spmat, densevec, diagmat))
            @test issparse(hcat(densemat, spmat, spvec, densevec, diagmat))
            @test issparse(hvcat((5,), diagmat, densevec, spvec, densemat, spmat))
            @test issparse(hvcat((5,), spvec, densemat, diagmat, densevec, spmat))
            @test issparse(cat(densemat, diagmat, spmat, densevec, spvec; dims=(1,2)))
            @test issparse(cat(spvec, diagmat, densevec, spmat, densemat; dims=(1,2)))

            @test issparse(@inferred cat_with_constdims(densemat, diagmat, spmat, densevec, spvec))
            @test issparse(@inferred cat_with_constdims(spvec, diagmat, densevec, spmat, densemat))
        end
        @testset "vertical concatenation of SparseVectors with different el- and ind-type (#22225)" begin
            spv6464 = SparseVector(0, Int64[], Int64[])
            @test isa(vcat(spv6464, SparseVector(0, Int64[], Int32[])), SparseVector{Int64,Int64})
            @test isa(vcat(spv6464, SparseVector(0, Int32[], Int64[])), SparseVector{Int64,Int64})
            @test isa(vcat(spv6464, SparseVector(0, Int32[], Int32[])), SparseVector{Int64,Int64})
        end
        @testset "horizontal concatenation of SparseVectors with different el- and ind-type (#22225)" begin
            spv6464 = SparseVector(0, Int64[], Int64[])
            @test isa(hcat(spv6464, SparseVector(0, Int64[], Int32[])), SparseMatrixCSC{Int64,Int64})
            @test isa(hcat(spv6464, SparseVector(0, Int32[], Int64[])), SparseMatrixCSC{Int64,Int64})
            @test isa(hcat(spv6464, SparseVector(0, Int32[], Int32[])), SparseMatrixCSC{Int64,Int64})
        end
    end
end

# An array type from another package that owns the `vcat`/`hcat`/`hvcat` of its own
# arrays with anything; those methods must not become ambiguous when SparseArrays is
# loaded (#431)
struct ConcatArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
ConcatArray(x::AbstractArray) = ConcatArray(Array(x))
Base.size(x::ConcatArray) = size(x.data)
Base.getindex(x::ConcatArray, i::Int) = x.data[i]
Base.IndexStyle(::Type{<:ConcatArray}) = IndexLinear()
Base.vcat(x::AbstractMatrix, y::ConcatArray{<:Any,2}) = ConcatArray(vcat(x, y.data))
Base.hcat(x::AbstractMatrix, y::ConcatArray{<:Any,2}) = ConcatArray(hcat(x, y.data))
Base.hvcat(rows::Tuple{Vararg{Int}}, x::AbstractMatrix, y::ConcatArray{<:Any,2}) =
    ConcatArray(hvcat(rows, x, y.data))
Base.vcat(x::AbstractVector, y::ConcatArray{<:Any,1}) = ConcatArray(vcat(x, y.data))
Base.hcat(x::AbstractVector, y::ConcatArray{<:Any,1}) = ConcatArray(hcat(x, y.data))

@testset "no ambiguities with concatenation methods of other array types (#431)" begin
    A = ConcatArray([1 2; 3 4])
    v = ConcatArray([1, 2])
    S = sparse([1 0; 0 1])
    for x in (A, [5 6; 7 8], S)
        @test vcat(x, A)::ConcatArray == vcat(Array(x), A.data)
        @test hcat(x, A)::ConcatArray == hcat(Array(x), A.data)
        @test hvcat((2,), x, A)::ConcatArray == hvcat((2,), Array(x), A.data)
    end
    for x in (v, [5, 6], sparse([1, 0]))
        @test vcat(x, v)::ConcatArray == vcat(Array(x), v.data)
        @test hcat(x, v)::ConcatArray == hcat(Array(x), v.data)
    end
    # with the sparse array first, the generic fallback still yields a sparse result
    @test vcat(A, S)::SparseMatrixCSC == vcat(A.data, Array(S))
    @test hcat(A, S)::SparseMatrixCSC == hcat(A.data, Array(S))
    @test hvcat((2,), A, S)::SparseMatrixCSC == hvcat((2,), A.data, Array(S))
    @test vcat(A, [5 6; 7 8])::Matrix == vcat(A.data, [5 6; 7 8])
    @test vcat(v, sparse([1, 0]))::SparseVector == vcat(v.data, [1, 0])
end

@testset "concatenation with non-numeric eltypes stays dense (#71)" begin
    S = sparse([1 0 0])
    M = fill("a", 1, 3)
    @test vcat(M, S)::Matrix == vcat(M, Array(S))
    @test hcat(M, S)::Matrix == hcat(M, Array(S))
    @test hvcat((1, 1), M, S)::Matrix == hvcat((1, 1), M, Array(S))
    @test vcat(fill("a", 3), sparse([1, 0, 0]))::Vector == vcat(fill("a", 3), [1, 0, 0])
    # with a UniformScaling, the array type is chosen by `promote_to_array_type`
    A = fill("a", 2, 2); Z = spzeros(2, 2)
    @test hcat(I, A, Z)::Matrix == hcat(Matrix(I, 2, 2), A, Array(Z))
    @test vcat(I, A, Z)::Matrix == vcat(Matrix(I, 2, 2), A, Array(Z))
    @test hvcat((3,), I, A, Z)::Matrix == hvcat((3,), Matrix(I, 2, 2), A, Array(Z))
    @test hcat(I, Z, spzeros(2, 2))::SparseMatrixCSC == hcat(Matrix(I, 2, 2), Array(Z), zeros(2, 2))
end

@testset "concatenation with a leading number fills its block like dense (#383)" begin
    M = sparse([1 2]); V = sparse([1, 2]); dM = Array(M); dV = Array(V)
    @test vcat(1, M)::SparseMatrixCSC == vcat(1, dM)
    @test vcat(1.5, M)::SparseMatrixCSC{Float64} == vcat(1.5, dM)
    @test vcat(1, M, M)::SparseMatrixCSC == vcat(1, dM, dM)
    @test vcat(1, V)::SparseVector == vcat(1, dV)
    @test vcat(V, 3)::SparseVector == vcat(dV, 3)
    @test hcat(1, M)::SparseMatrixCSC == hcat(1, dM)
    @test hcat(1, M, 3)::SparseMatrixCSC == hcat(1, dM, 3)
    @test hvcat((2,), 1, M)::SparseMatrixCSC == hvcat((2,), 1, dM)
    @test cat(1, M; dims=1)::SparseMatrixCSC == cat(1, dM; dims=1)
    @test cat(1, M; dims=(1, 2))::SparseMatrixCSC == cat(1, dM; dims=(1, 2))
    @test [1; M]::SparseMatrixCSC == [1; dM]
    @test sparse_vcat(1, dM)::SparseMatrixCSC == vcat(1, dM)
    @test sparse_hcat(1, dM, 3)::SparseMatrixCSC == hcat(1, dM, 3)
    @test sparse_hvcat((2,), 1, dM)::SparseMatrixCSC == hvcat((2,), 1, dM)
    @test sparse_vcat(1, 2)::SparseVector == [1, 2]
    @test sparse_hcat(1, 2)::SparseMatrixCSC == [1 2]
    # shape mismatches throw as for dense
    @test_throws DimensionMismatch vcat(M, 3)
    @test_throws DimensionMismatch vcat(1, M, 3)
    @test_throws DimensionMismatch hcat(1, V)
    @test_throws DimensionMismatch hcat(V, 3)
end

end # module SparseConcatenationTests
