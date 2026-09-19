# This file is a part of Julia. License is MIT: https://julialang.org/license

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
