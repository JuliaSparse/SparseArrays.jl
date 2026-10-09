
# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseConcatenationTests

using Test
using SparseArrays
using LinearAlgebra
include("testhelpers.jl")

@testset "concatenation tests" begin
    sp33 = sparse(1.0I, 3, 3)
    se33 = SparseMatrixCSC{Float64}(I, 3, 3)
    do33 = fill(1.,3)
    # a block with another element and index type, which promote
    sc33 = SparseMatrixCSC{ComplexF64,Int16}(fixture(ComplexF64, 3, 3))
    # blocks whose row and column counts differ
    sf53 = fixture(Float64, 5, 3)
    sf35 = fixture(Float64, 3, 5)
    @testset "horizontal concatenation" begin
        @test mismatch([sf53 sf53], [Array(sf53) Array(sf53)]; Ti=Int) === nothing
        @test_throws DimensionMismatch [sf53 sf35]
        @static if COMPREHENSIVE
        @test length(nonzeros([sp33 0I])) == 3
        @test mismatch([se33 sc33], [Array(se33) Array(sc33)]; Ti=Int) === nothing
        end
    end

    @testset "vertical concatenation" begin
        @test mismatch([sf53; sf53], [Array(sf53); Array(sf53)]; Ti=Int) === nothing
        # the block with the narrower index type comes first, so that taking the first
        # block's index type differs from promoting
        @test mismatch([sc33; sf53], [Array(sc33); Array(sf53)]; Ti=Int) === nothing
        @test_throws DimensionMismatch [sf53; sf35]
        @static if COMPREHENSIVE
        se33_32bit = convert(SparseMatrixCSC{Float32,Int32}, se33)
        @test [se33; se33_32bit] == [Array(se33); Array(se33_32bit)]
        @test length(nonzeros([sp33; 0I])) == 3
        end
    end

    se44 = sparse(1.0I, 4, 4)
    sz42 = spzeros(4, 2)
    sz41 = spzeros(4, 1)
    sz34 = spzeros(3, 4)
    se77 = sparse(1.0I, 7, 7)
    @testset "h+v concatenation" begin
        @test mismatch(@inferred(hvcat((3, 2), se44, sz42, sz41, sz34, se33)), Array(se77); Ti=Int) === nothing # [se44 sz42 sz41; sz34 se33]
        @test length(nonzeros([sp33 0I; 1I 0I])) == 6
    end

    @testset "h+v concatenation with block rows unknown to inference" begin
        A = sparse([1, 3, 2], [1, 1, 3], [1.0, 2.0, 0.0], 3, 3)  # a stored zero
        B = @static COMPREHENSIVE ? SparseMatrixCSC{Float32,Int32}(fixture(Float64, 3, 2)) : A[:, 2:3]
        C = fixture(Float64, 3, 8)
        rows = Base.inferencebarrier((3, 1))
        H = hvcat(rows, A, B, A, C)
        @test H isa SparseMatrixCSC{Float64,Int}
        @test mismatch(H, hvcat((3, 1), Matrix(A), Matrix(B), Matrix(A), Matrix(C)); Ti=Int) === nothing
        @test nnz(H) == 2nnz(A) + nnz(B) + nnz(C)
        @test mismatch(hvcat(rows, A, Matrix(B), A, C), Array(H); Ti=Int) === nothing
        @test mismatch(hvcat(Base.inferencebarrier((2, 2)), spzeros(0, 2), spzeros(0, 1), A[:, 1:2], A[:, 3:3]), Array(A); Ti=Int) === nothing
        @test_throws DimensionMismatch hvcat(Base.inferencebarrier((2,)), A, spzeros(4, 2))
        @test_throws DimensionMismatch hvcat(Base.inferencebarrier((2, 1)), A, A, C[:, 1:5])
        # a later block row wider than the first, which the copy loop must not be reached with
        @test_throws DimensionMismatch hvcat(Base.inferencebarrier((1, 1)), A, C)
        @test_throws DimensionMismatch hvcat(Base.inferencebarrier((2, 2)), A, A, A)
        @test_throws DimensionMismatch hvcat(Base.inferencebarrier((1, 1)), A, A, A)
        @test_throws ArgumentError hvcat(Base.inferencebarrier((0, 2)), A, A)
    end

    @testset "h+v concatenation with vector and number blocks" begin
        A = sparse([1.0 0; 0 2])
        vz = sparsevec([1, 2], [0.0, 1.0])  # a stored zero
        H = [A vz; 1.0 0.0 -0.0]
        @test H isa SparseMatrixCSC{Float64,Int}
        @test mismatch(H, [Matrix(A) Vector(vz); 1.0 0.0 -0.0]) === nothing
        # the stored zero of `vz` stays, `0.0` is not stored and `-0.0` is, as in `setindex!`
        @test findnz(H) == ([1, 3, 2, 1, 2, 3], [1, 1, 2, 3, 3, 3], [1.0, 1.0, 2.0, 0.0, 1.0, -0.0])
        @test mismatch(hvcat(Base.inferencebarrier((2, 3)), A, vz, 1.0, 0.0, -0.0), Array(H); Ti=Int) === nothing
        @static if COMPREHENSIVE
        @test [A [3.0, 4.0]; 1 2 3] == [Matrix(A) [3.0, 4.0]; 1 2 3]
        end
        @test [A vz; 1 im 3] isa SparseMatrixCSC{ComplexF64,Int}
        @static if COMPREHENSIVE
        @test [A vz; missing 2 3] isa SparseMatrixCSC{Union{Missing,Float64},Int}
        A32 = SparseMatrixCSC{Float64,Int32}(A)
        v32 = SparseVector{Float64,Int32}(vz)
        @test [A32 v32; v32' 1.0] isa SparseMatrixCSC{Float64,Int32}
        @test [1.0 v32'; v32 A32] isa SparseMatrixCSC{Float64,Int}  # a leading number widens
        end
        @test_throws DimensionMismatch [A vz; 1 2]
        @static if COMPREHENSIVE
        @test_throws DimensionMismatch hvcat(Base.inferencebarrier((2, 2)), A, vz, 1.0)
        end
    end

    @testset "cat with dims unknown to inference" begin
        A = fixture(Float64, 5, 3)
        D = reshape(Float64.(1:15), 5, 3)
        v = fixturevec(Float64, 5)
        for dims in (1, 2, (@static COMPREHENSIVE ? ((1, 2), Val(2)) : ())...)
            C = cat(A, D; dims = Base.inferencebarrier(dims))
            @test C isa SparseMatrixCSC{Float64,Int}
            @test mismatch(C, cat(Matrix(A), D; dims)) === nothing
        end
        @static if COMPREHENSIVE
        @test cat(v, v; dims = Base.inferencebarrier(1)) isa SparseVector{Float64,Int}
        end
        C3 = cat(A, Matrix(A); dims = Base.inferencebarrier((1, 3)))
        @test C3 isa Array{Float64,3}
        @test C3 == cat(Matrix(A), Matrix(A); dims = (1, 3))
    end

    @testset "blockdiag concatenation" begin
        # with blocks that are not square the row and the column offsets differ
        @test mismatch(blockdiag(sf53, sf35), [Array(sf53) zeros(5, 5); zeros(3, 3) Array(sf35)]; Ti=Int) === nothing
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

    @static if COMPREHENSIVE
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
    end

    # check splicing + concatenation, with nested vcat and also side-checks sparse ref
    @testset "splicing + concatenation" begin
        for T in (Float64, (@static COMPREHENSIVE ? (ComplexF64,) : ())...)
            a = fixture(T, 5, 4)
            @test mismatch([a[1:2,1:2] a[1:2,3:4]; a[3:5,1] [a[3:4,2:4]; a[5:5,2:4]]], Array(a); Ti=Int) === nothing
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
        fdiagmat = Diagonal(dvec)
        @test issparse(vcat(fdiagmat, spmat))
        @test sparse_vcat(dmat, fdiagmat)::SparseMatrixCSC == vcat(spmat, fdiagmat)
        @test issparse(hcat(spvec, fdiagmat))
        @test sparse_hcat(fdiagmat, dvec)::SparseMatrixCSC == hcat(fdiagmat, spvec)
        @test sparse_hvcat((2,), dmat, fdiagmat)::SparseMatrixCSC == hvcat((2,), spmat, fdiagmat)
        @test issparse(cat(fdiagmat, spvec; dims=(1,2)))
        @static if COMPREHENSIVE
        # `Diagonal` is covered by `fdiagmat` above
        for specialmat in specialmats[2:end]
            # --> Tests applicable only to pairs of matrices
            @test issparse(vcat(specialmat, spmat))
            @test issparse(vcat(spmat, specialmat))
            @test sparse_vcat(specialmat, dmat)::SparseMatrixCSC == vcat(specialmat, spmat)
            @test sparse_vcat(dmat, specialmat)::SparseMatrixCSC == vcat(spmat, specialmat)
            # --> Tests applicable also to pairs including vectors
            # the partner only decides how the other block converts, so one special type
            # takes the vector partner and the others the matrix
            for (smatorvec, dmatorvec) in (specialmat === tridiagmat ? ((spvec, dvec),) : ((spmat, dmat),))
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
    end

    # Test that concatenations of annotated sparse/special matrix types with other matrix
    # types yield sparse arrays, and that the code which effects that does not make concatenations
    # strictly involving un/annotated dense matrices yield sparse arrays
    @testset "concatenations of annotated types" begin
        N = 4
        # The tested annotation types
        utriannotations = (UpperTriangular, UnitUpperTriangular)
        ltriannotations = (LowerTriangular, UnitLowerTriangular)
        triannotations = (utriannotations..., ltriannotations...)
        symannotations = (Symmetric, Hermitian)
        annotations = (triannotations..., symannotations...)
        # Concatenations involving these types, un/annotated, should yield sparse arrays
        spvec = spzeros(N)
        spmat = sparse(1.0I, N, N)
        diagmat = Diagonal(1:N)
        bidiagmat = Bidiagonal(1:N, 1:(N-1), :U)
        tridiagmat = Tridiagonal(1:(N-1), 1:N, 1:(N-1))
        symtridiagmat = SymTridiagonal(1:N, 1:(N-1))
        sparseconcatmats = (spmat, diagmat, bidiagmat, tridiagmat, symtridiagmat)
        # Concatenations involving strictly these types, un/annotated, should yield dense arrays
        densevec = Array(spvec)
        densemat = Array(spmat)
        Lsp, Ssp = LowerTriangular(spmat), Symmetric(spmat)
        Ld, Sd = LowerTriangular(densemat), Symmetric(densemat)
        fdiagmat = Diagonal(fill(1., N))
        hvcat2 = (a, b) -> hvcat((2,), a, b)
        cat12 = (a, b) -> cat(a, b; dims=(1,2))
        ops = (vcat, hcat, hvcat2, cat12)
        sides = (true, false)
        @static if COMPREHENSIVE
        # Annotated collections
        annodmats = [annot(densemat) for annot in annotations]
        annospcmats = [annot(spmat) for annot in annotations]
        end
        # Test that concatenations of pairwise combinations of annotated sparse/special
        # yield sparse matrices. The concatenation methods see an annotation only through
        # `issparse` and the conversion to `SparseMatrixCSC`, one block at a time, so each
        # annotation, partner, operation and side appears, not their combinations
        for (a, b, op) in (@static COMPREHENSIVE ? eachvalue(annospcmats, reverse(annospcmats), ops) : ((Lsp, Ssp, vcat),))
            @test issparse(op(a, b))
        end
        # Test that concatenations of pairwise combinations of annotated sparse/special
        # matrices and other matrix/vector types yield sparse matrices
        cases = ((Lsp, densevec, hcat, true), (Lsp, spvec, cat12, true),
                 (Ssp, Sd, vcat, false), (Ssp, Ld, hvcat2, true))
        # the grids may pick one of `cases` again, which then runs once
        for (a, other, op, lead) in (cases..., (@static COMPREHENSIVE ? filter(c -> !any(s -> s === c, cases), Any[
                eachvalue(annospcmats, (densemat, sparseconcatmats...), ops, sides)...,
                # a vector does not `vcat` with a matrix
                eachvalue(annospcmats[1:2:end], (spvec, densevec), (hcat, hvcat2, cat12), sides)...,
                # the operations in the other order, so that each meets both sides
                eachvalue(annospcmats, reverse(annodmats), reverse(ops), sides)...]) : ())...)
            @test issparse(lead ? op(a, other) : op(other, a))
        end
        @test sparse_hcat(Ld, fdiagmat)::SparseMatrixCSC == hcat(Lsp, fdiagmat)
        @test sparse_vcat(spmat, Sd)::SparseMatrixCSC == vcat(spmat, Ssp)
        @static if COMPREHENSIVE
        # The `sparse_*` entry points on the annotated dense matrices match the annotated sparse ones
        sparse_hvcat2 = (a, b) -> sparse_hvcat((2,), a, b)
        entries = ((sparse_hcat, hcat), (sparse_vcat, vcat), (sparse_hvcat2, hvcat2))
        for (i, specialmat, (sparse_op, op), lead) in eachvalue(eachindex(annotations), sparseconcatmats, entries, sides)
            smat, dmat = annospcmats[i], annodmats[i]
            if lead
                @test sparse_op(dmat, specialmat)::SparseMatrixCSC == op(smat, specialmat)
            else
                @test sparse_op(specialmat, dmat)::SparseMatrixCSC == op(specialmat, smat)
            end
        end
        # The preceding tests should cover multi-way combinations of those types, but for good
        # measure test a few multi-way combinations involving those types
        @test issparse(vcat(spmat, densemat, annospcmats[1], annodmats[2]))
        @test issparse(hcat(spvec, annodmats[1], annospcmats[1], densevec, diagmat))
        @test issparse(cat(annodmats[1], diagmat, annospcmats[2], densevec, spvec; dims=(1,2)))
        end
    end

    @testset "hcat and vcat involving UniformScaling" begin
        @test_throws ArgumentError [I I]
        @static if COMPREHENSIVE
        @test_throws ArgumentError hcat(I)
        @test_throws ArgumentError vcat(I)
        @test_throws ArgumentError [I; I]
        @test_throws ArgumentError [I I; I]
        end

        A = fixture(Float64, 3, 4)
        B = fixture(Float64, 3, 3)
        C = spzeros(0, 3)
        D = spzeros(2, 0)
        E = fixture(Float64, 1, 3)
        F = fixture(Float64, 3, 1)
        α = 0.75
        @static if COMPREHENSIVE
        @test (hcat(A, 2I, I(3)))::SparseMatrixCSC == hcat(A, Matrix(2I, 3, 3), Matrix(I, 3, 3))
        @test (hcat(E, α))::SparseMatrixCSC == hcat(E, [α])
        @test (hcat(E, α, 2I))::SparseMatrixCSC == hcat(E, [α], fill(2, 1, 1))
        end
        @test (vcat(A, 2I))::SparseMatrixCSC == (vcat(A, 2I(4)))::SparseMatrixCSC == vcat(A, Matrix(2I, 4, 4))
        @static if COMPREHENSIVE
        @test (vcat(F, α))::SparseMatrixCSC == vcat(F, [α])
        @test (vcat(F, α, 2I))::SparseMatrixCSC == (vcat(F, α, 2I(1)))::SparseMatrixCSC == vcat(F, [α], fill(2, 1, 1))
        end
        @test (hcat(C, 2I))::SparseMatrixCSC == C
        @test_throws DimensionMismatch hcat(C, α)
        @static if COMPREHENSIVE
        @test (vcat(D, 2I))::SparseMatrixCSC == D
        @test_throws DimensionMismatch vcat(D, α)
        @test (hcat(I, 3I, A, 2I))::SparseMatrixCSC == hcat(Matrix(I, 3, 3), Matrix(3I, 3, 3), A, Matrix(2I, 3, 3))
        @test (vcat(I, 3I, A, 2I))::SparseMatrixCSC == vcat(Matrix(I, 4, 4), Matrix(3I, 4, 4), A, Matrix(2I, 4, 4))
        @test hvcat((3,1), C, C, I, 3I)::SparseMatrixCSC == hvcat((2,1), C, C, Matrix(3I, 6, 6))
        @test hvcat((2,2,4), C, C, I(3), 2I, 3I, 4I, 5I, D)::SparseMatrixCSC ==
            hvcat((2,2,4), C, C, Matrix(I, 3, 3), Matrix(2I, 3, 3),
                Matrix(3I, 2, 2), Matrix(4I, 2, 2), Matrix(5I, 2, 2), D)
        @test (hvcat((1,2), A, E, α))::SparseMatrixCSC == hvcat((1,2), A, E, [α]) == hvcat((1,2), A, E, α*I)
        @test (hvcat((2,2), α, E, F, 3I))::SparseMatrixCSC == hvcat((2,2), [α], E, F, Matrix(3I, 3, 3))
        end
        # the `sparse_*` entry points size a `UniformScaling` from its neighbours like the plain ones
        dA, dB = Array(A), Array(B)
        @test sparse_hcat(A, I)::SparseMatrixCSC == sparse_hcat(dA, I)::SparseMatrixCSC == hcat(A, I)
        @static if COMPREHENSIVE
        @test sparse_hcat(I, A, 2I)::SparseMatrixCSC == sparse_hcat(I, dA, 2I)::SparseMatrixCSC == hcat(I, A, 2I)
        end
        @test sparse_vcat(3I, A)::SparseMatrixCSC == sparse_vcat(3I, dA)::SparseMatrixCSC == vcat(3I, A)
        @test sparse_hvcat((2,2), B, I, I, B)::SparseMatrixCSC == sparse_hvcat((2,2), dB, I, I, dB)::SparseMatrixCSC ==
            hvcat((2,2), B, I, I, B)
        @static if COMPREHENSIVE
        @test sparse_hvcat((3,1), C, C, I, 3I)::SparseMatrixCSC == hvcat((3,1), C, C, I, 3I)
        end
        @test_throws ArgumentError sparse_hcat(I)
        @static if COMPREHENSIVE
        @test_throws ArgumentError sparse_vcat(I, 2I)
        @test_throws ArgumentError sparse_hvcat((1,1), I, I)
        end
        @test_throws DimensionMismatch sparse_hcat(A, I, E)
    end
end


# Test that concatenations of combinations of sparse matrices with sparse matrices or dense
# matrices/vectors yield sparse arrays
@static if COMPREHENSIVE
@testset "sparse and dense concatenations" begin
    N = 4
    densevec = fill(1., N)
    densemat = diagm(0 => densevec)
    spmat = spdiagm(0 => densevec)
    @static if COMPREHENSIVE
    # Test that concatenations of pairs of sparse matrices yield sparse arrays
    @test issparse(vcat(spmat, spmat))
    @test issparse(hcat(spmat, spmat))
    @test issparse(@inferred(hvcat((2,), spmat, spmat)))
    @test issparse(cat(spmat, spmat; dims=(1,2)))
    end
    # Test that concatenations of a sparse matrice with a dense matrix/vector yield sparse arrays
    @test issparse(vcat(spmat, densemat))
    @test issparse(vcat(densemat, spmat))
    for densearg in (densevec, (@static COMPREHENSIVE ? (densemat,) : ())...)
        @test issparse(hcat(spmat, densearg))
        @test issparse(hcat(densearg, spmat))
        @test issparse(hvcat((2,), spmat, densearg))
        @test issparse(hvcat((2,), densearg, spmat))
        @test issparse(cat(spmat, densearg; dims=(1,2)))
        @test issparse(cat(densearg, spmat; dims=(1,2)))
    end
end
end

@testset "block literals mixing sparse and dense blocks infer" begin
    S = fixture(Float64, 4, 4)
    C = fixture(ComplexF64, 4, 4)
    A = reshape(Float64.(1:16), 4, 4)
    v = fixturevec(Float64, 4)
    w = Float64.(1:4)
    dS, dC, dv = Array(S), Array(C), Array(v)
    # the literals are wrapped so that `@inferred` sees the constant `rows` of the syntax
    lit22 = (X, Y) -> [X Y; Y Y]
    border = (X, y, z) -> [X y; z' 1]
    litI = (X, Y) -> [X I; Y X]
    @test mismatch(@inferred(lit22(S, A)), [dS A; A A]; Ti=Int) === nothing
    @static if COMPREHENSIVE
    @test mismatch(@inferred(lit22(A, C)), [A dC; dC dC]; Ti=Int) === nothing
    @test mismatch(@inferred(lit22(v, w)), [dv w; w w]; Ti=Int) === nothing
    @test mismatch(@inferred(border(S, v, w)), [dS dv; w' 1]; Ti=Int) === nothing
    end
    @test mismatch(@inferred(litI(C, A)), [dC I; A dC]; Ti=Int) === nothing
    # LinearAlgebra replaces `UniformScaling` blocks by sparse identities and calls `hvcat`
    # back with `rows` no longer a constant; `@inferred` here sees only the type of `rows`
    rows = (2, 2)
    @test mismatch(@inferred(hvcat(rows, S, A, A, A)), [dS A; A A]; Ti=Int) === nothing
    @static if COMPREHENSIVE
    B = sparse(I, 4, 4)
    @test mismatch(@inferred(hvcat(rows, S, B, B, S)), [dS I; I dS]; Ti=Int) === nothing
    # a dense block converts to `Int` indices
    S32 = SparseMatrixCSC{Float64,Int32}(S)
    @test mismatch(@inferred(lit22(S32, A)), [dS A; A A]; Ti=Int) === nothing
    end
end

@testset "issue #19304" begin
    @inferred hcat(fixture(Float64, 2, 1), I)
    @static if COMPREHENSIVE
    @inferred hcat(fixture(Float64, 2, 1), 1.0I)
    @inferred hcat(fixture(Float64, 2, 1), Matrix(I, 2, 2))
    @inferred hcat(fixture(Float64, 2, 1), Matrix(1.0I, 2, 2))
    end
end


@testset "Concatenation" begin
    let m = 80, n = (@static COMPREHENSIVE ? 100 : 10)  # 100 reach Base's long-tuple methods
        A = Vector{SparseVector{Float64,Int}}(undef, n)
        tnnz = 0
        for i = 1:length(A)
            # the pattern and the value differ from one vector to the next
            A[i] = sparsevec(mod1(i, 7):3:m, Float64(i), m)
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
        @test mismatch(H, Hr) === nothing

        V = vcat(A...)
        @test isa(V, SparseVector{Float64,Int})
        @test length(V) == m * n
        Vr = vec(Hr)
        @test mismatch(V, Vr) === nothing
        Vnum = vcat(A..., zero(Float64))
        Vnum2 = sparse_vcat(map(Array, A)..., zero(Float64))
        @test Vnum isa SparseVector{Float64,Int}
        @test Vnum2 isa SparseVector{Float64,Int}
        @test length(Vnum) == length(Vnum2) == m*n + 1
        @test mismatch(Vnum, [Vr; 0]) === nothing
        @test mismatch(Vnum2, [Vr; 0]) === nothing
        @static if COMPREHENSIVE
        Vnum = vcat(zero(Float64), A...)
        Vnum2 = sparse_vcat(zero(Float64), map(Array, A)...)
        @test Vnum isa SparseVector{Float64,Int}
        @test Vnum2 isa SparseVector{Float64,Int}
        @test length(Vnum) == length(Vnum2) == m*n + 1
        @test Array(Vnum) == Array(Vnum2) == [0; Vr]
        end
        # case with rowwise a Number as first element, should still yield a sparse matrix
        x = sparsevec([1], [3.0], 1)
        X = [3.0 x; 3.0 x]
        @test issparse(X)
        # the element and index types of the vectors promote
        y = SparseVector{ComplexF64,Int16}(x)
        @test isa(vcat(x, y), SparseVector{ComplexF64,Int})
        @test isa(hcat(x, y), SparseMatrixCSC{ComplexF64,Int})
    end

    @testset "stack (#498)" begin
        A = [sparsevec(mod1(i, 7):3:80, Float64(i), 80) for i in 1:(@static COMPREHENSIVE ? 100 : 10)]
        H = hcat(A...)
        S = @inferred stack(A)
        @test S isa SparseMatrixCSC{Float64,Int}
        @test mismatch(S, Array(H)) === nothing
        S1 = stack(A; dims=1)
        @test S1 isa SparseMatrixCSC{Float64,Int}
        @test mismatch(S1, permutedims(Array(H))) === nothing
        @test_throws ArgumentError stack(A; dims=3)
        @static if COMPREHENSIVE
        @test stack(x for x in A if true) == H
        @test stack(eachcol(H)) == H
        # slices with different element and index types promote
        SB = stack([sparsevec(Int32[1], Int32[2], 3), sparsevec([3], [0.5], 3)])
        @test SB isa SparseMatrixCSC{Float64,Int}
        @test SB == [2 0; 0 0; 0 0.5]
        # a container with more than one axis stacks into a dense array
        @test stack(reshape(A, 2, :)) == reshape(Array(H), 80, 2, :)
        end
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
            for othervecormat in (densevec, (@static COMPREHENSIVE ? (densemat, spmat) : ())...)
                @test issparse(vcat(spvec, othervecormat))
                @test issparse(vcat(othervecormat, spvec))
            end
            for othervecormat in (densevec, (@static COMPREHENSIVE ? (densemat, spmat, diagmat) : ())...)
                @test issparse(hcat(spvec, othervecormat))
                @test issparse(hcat(othervecormat, spvec))
                @test issparse(hvcat((2,), spvec, othervecormat))
                @test issparse(hvcat((2,), othervecormat, spvec))
                @test issparse(cat(spvec, othervecormat; dims=(1,2)))
                @test issparse(cat(othervecormat, spvec; dims=(1,2)))

                @test issparse(@inferred cat_with_constdims(spvec, othervecormat))
                @test issparse(@inferred cat_with_constdims(othervecormat, spvec))
            end
            @static if COMPREHENSIVE
            # The preceding tests should cover multi-way combinations of those types, but for good
            # measure test a few multi-way combinations involving those types
            @test issparse(vcat(spvec, densevec, spmat, densemat))
            @test issparse(cat(densemat, diagmat, spmat, densevec, spvec; dims=(1,2)))
            end

            @test issparse(@inferred cat_with_constdims(densemat, diagmat, spmat, densevec, spvec))
        end
        @static if COMPREHENSIVE
        @testset "vertical concatenation of SparseVectors with different el- and ind-type (#22225)" begin
            spv6464 = SparseVector(0, Int64[], Int64[])
            @test isa(vcat(spv6464, SparseVector(0, Int32[], Int32[])), SparseVector{Int64,Int64})
        end
        @testset "horizontal concatenation of SparseVectors with different el- and ind-type (#22225)" begin
            spv6464 = SparseVector(0, Int64[], Int64[])
            @test isa(hcat(spv6464, SparseVector(0, Int32[], Int32[])), SparseMatrixCSC{Int64,Int64})
        end
        end
    end
end

# An array type from another package that owns the `vcat`/`hcat`/`hvcat` of its own
# arrays with anything; those methods must not become ambiguous when SparseArrays is
# loaded (#431)
@static if COMPREHENSIVE
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
end

@testset "concatenation with non-numeric eltypes stays dense (#71)" begin
    S = sparse([1 0 0])
    M = fill("a", 1, 3)
    @test vcat(M, S)::Matrix == vcat(M, Array(S))
    @static if COMPREHENSIVE
    @test hcat(M, S)::Matrix == hcat(M, Array(S))
    @test hvcat((1, 1), M, S)::Matrix == hvcat((1, 1), M, Array(S))
    @test vcat(fill("a", 3), sparse([1, 0, 0]))::Vector == vcat(fill("a", 3), [1, 0, 0])
    # with a UniformScaling, the array type is chosen by `promote_to_array_type`
    A = fill("a", 2, 2); Z = spzeros(2, 2)
    @test hcat(I, A, Z)::Matrix == hcat(Matrix(I, 2, 2), A, Array(Z))
    @test vcat(I, A, Z)::Matrix == vcat(Matrix(I, 2, 2), A, Array(Z))
    @test hvcat((3,), I, A, Z)::Matrix == hvcat((3,), Matrix(I, 2, 2), A, Array(Z))
    @test hcat(I, Z, spzeros(2, 2))::SparseMatrixCSC == hcat(Matrix(I, 2, 2), Array(Z), zeros(2, 2))
    # with the sparse array first
    @test vcat(S, M)::Matrix == vcat(Array(S), M)
    end
    @test hcat(S, M)::Matrix == hcat(Array(S), M)
    @static if COMPREHENSIVE
    @test hvcat((2, 2), S, M, M, S)::Matrix == hvcat((2, 2), Array(S), M, M, Array(S))
    end
    @test cat(S, M; dims=1)::Matrix == cat(Array(S), M; dims=1)
    @static if COMPREHENSIVE
    @test cat(S, M; dims=3)::Array{Any,3} == cat(Array(S), M; dims=3)
    end
    s = sparse([1, 0, 0]); t = fill("a", 3)
    @test vcat(s, t)::Vector == vcat([1, 0, 0], t)
    @test hcat(s, t)::Matrix == hcat([1, 0, 0], t)
    @static if COMPREHENSIVE
    @test cat(s, t; dims=2)::Matrix == cat([1, 0, 0], t; dims=2)
    @test vcat(s', permutedims(t))::Matrix == vcat([1 0 0], permutedims(t))
    @test cat(transpose(s), permutedims(t); dims=1)::Matrix == cat([1 0 0], permutedims(t); dims=1)
    # a numeric array of more than two dimensions still takes Base's `cat`, also with the
    # same eltype as the sparse array
    @test cat(S, ones(1, 3, 2); dims=3)::Array{Float64,3} == cat(Array(S), ones(1, 3, 2); dims=3)
    @test cat(S, ones(Int, 1, 3, 2); dims=3)::Array{Int,3} == cat(Array(S), ones(Int, 1, 3, 2); dims=3)
    @test cat(s, ones(Int, 3, 1, 1); dims=2)::Array{Int,3} == cat([1, 0, 0], ones(Int, 3, 1, 1); dims=2)
    @test cat(S, S; dims=3)::Array{Int,3} == cat(Array(S), Array(S); dims=3)
    end
end

@testset "concatenation with a leading number fills its block like dense (#383)" begin
    M = sparse([1 2]); V = sparse([1, 2]); dM = Array(M); dV = Array(V)
    @test mismatch(vcat(1, M), vcat(1, dM)) === nothing
    @static if COMPREHENSIVE
    @test vcat(1.5, M)::SparseMatrixCSC{Float64} == vcat(1.5, dM)
    @test vcat(1, M, M)::SparseMatrixCSC == vcat(1, dM, dM)
    end
    @test mismatch(vcat(1, V), vcat(1, dV)) === nothing
    @static if COMPREHENSIVE
    @test vcat(V, 3)::SparseVector == vcat(dV, 3)
    @test hcat(1, M)::SparseMatrixCSC == hcat(1, dM)
    end
    @test mismatch(hcat(1, M, 3), hcat(1, dM, 3)) === nothing
    @static if COMPREHENSIVE
    @test hvcat((2,), 1, M)::SparseMatrixCSC == hvcat((2,), 1, dM)
    @test cat(1, M; dims=1)::SparseMatrixCSC == cat(1, dM; dims=1)
    @test cat(1, M; dims=(1, 2))::SparseMatrixCSC == cat(1, dM; dims=(1, 2))
    @test [1; M]::SparseMatrixCSC == [1; dM]
    @test sparse_vcat(1, dM)::SparseMatrixCSC == vcat(1, dM)
    @test sparse_hcat(1, dM, 3)::SparseMatrixCSC == hcat(1, dM, 3)
    @test sparse_hvcat((2,), 1, dM)::SparseMatrixCSC == hvcat((2,), 1, dM)
    end
    @test mismatch(sparse_vcat(1, 2), [1, 2]) === nothing
    @static if COMPREHENSIVE
    @test sparse_hcat(1, 2)::SparseMatrixCSC == [1 2]
    # a leading number widens a narrow index type as it did before
    v8 = sparsevec(Int8[127], [2], 127); A8 = sparse(Int8[127], Int8[1], [2], 127, 2)
    @test vcat(1, v8)::SparseVector{Int,Int} == vcat(1, Array(v8))
    @test sparse_vcat(1, v8)::SparseVector{Int,Int} == vcat(1, Array(v8))
    @test vcat(1, A8)::SparseMatrixCSC{Int,Int} == vcat(1, Array(A8))
    # a vector whose length is not an `Int` stacked on a matrix
    w8 = sparsevec(Int8[1], [2.0], 2)
    @test mismatch(vcat(w8, ones(1, 1)), vcat(Array(w8), ones(1, 1)), Ti=Int8) === nothing
    @test mismatch(vcat(sparse(ones(1, 1)), w8), vcat(ones(1, 1), Array(w8)), Ti=Int) === nothing
    @test mismatch([w8; ones(1, 1); w8], [Array(w8); ones(1, 1); Array(w8)]) === nothing
    end
    # shape mismatches throw as for dense
    @test_throws DimensionMismatch vcat(M, 3)
    @static if COMPREHENSIVE
    @test_throws DimensionMismatch vcat(1, M, 3)
    @test_throws DimensionMismatch hcat(1, V)
    @test_throws DimensionMismatch hcat(V, 3)
    end
end

end # module SparseConcatenationTests
