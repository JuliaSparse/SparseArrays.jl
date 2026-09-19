# This file is a part of Julia. License is MIT: https://julialang.org/license

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
