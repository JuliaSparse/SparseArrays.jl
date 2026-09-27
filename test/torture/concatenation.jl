# This file is a part of Julia. License is MIT: https://julialang.org/license

# Long-tail regression tests for concatenation. Each testset names the issue it guards;
# they run only when the `torture` suite is selected.

module TortureConcatenationTests
using Test
using SparseArrays
using LinearAlgebra
include("../testhelpers.jl")

# From Base's abstractarray.jl
@testset "julia #17088" begin
    n = 10
    M = rand(n, n)
    @testset "vector of vectors" begin
        v = [[M]; [M]] # using vcat
        @test size(v) == (2,)
        @test !issparse(v)
    end
    @testset "matrix of vectors" begin
        m1 = [[M] [M]] # using hcat
        m2 = [[M] [M];] # using hvcat
        @test m1 == m2
        @test size(m1) == (1,2)
        @test !issparse(m1)
        @test !issparse(m2)
    end
end

@testset "13130 and 16661" begin
    @test issparse([sprand(10,10,.1) sprand(10,.1)])
    @test issparse([sprand(10,1,.1); sprand(10,.1)])

    @test issparse([sprand(10,10,.1) rand(10)])
    @test issparse([sprand(10,1,.1)  rand(10)])
    @test issparse([sprand(10,2,.1) sprand(10,1,.1) rand(10)])
    @test issparse([sprand(10,1,.1); rand(10)])

    @test issparse([sprand(10,.1)  rand(10)])
    @test issparse([sprand(10,.1); rand(10)])
end

# Ten random draws; the core suite runs one.
@testset "splicing + concatenation on random instances" begin
    for i = 1 : 10
        a = sprand(5, 4, 0.5)
        @test [a[1:2,1:2] a[1:2,3:4]; a[3:5,1] [a[3:4,2:4]; a[5:5,2:4]]] == a
    end
end

# The full annotation and special-matrix lists; the core suite runs the reduced lists
# unless `JULIA_TESTFULL` is set.
@testset "concatenations of annotated types, full lists" begin
    N = 4
    # The tested annotation types
    testfull = true
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

end # module
