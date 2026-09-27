# This file is a part of Julia. License is MIT: https://julialang.org/license

# The full Cartesian grids behind the factored triangular tests of `test/triangular.jl`:
# sparse-vector solves over every eltype pair, products over every eltype pair and
# transform pair, and the product/solve fixtures over sizes and index types. They run
# only when the `torture` suite is selected.

module TortureTriangularTests

using Test
using SparseArrays
using SparseArrays: getcolptr, rowvals, nonzeros
using LinearAlgebra
using Random
include("../testhelpers.jl")

const WRAPPERS = (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
const TRANSFORMS = (identity, adjoint, transpose)

@testset "sparse-vector triangular solves over every eltype pair" begin
    m = 10
    sprmat = sprand(m, m, 0.2)
    sparsefloatmat = I + sprmat/(2m)
    sparsecomplexmat = I + SparseMatrixCSC(m, m, getcolptr(sprmat), rowvals(sprmat), complex.(nonzeros(sprmat), nonzeros(sprmat))/(4m))
    sparseintmat = 10m*I + SparseMatrixCSC(m, m, getcolptr(sprmat), rowvals(sprmat), round.(Int, nonzeros(sprmat)*10))

    denseintmat = I*10m + rand(1:m, m, m)
    densefloatmat = I + randn(m, m)/(2m)
    densecomplexmat = I + randn(ComplexF64, m, m)/(4m)

    inttypes = (Int32, Int64, BigInt)
    floattypes = (Float32, Float64, BigFloat)
    complextypes = (ComplexF32, ComplexF64)
    eltypes = (inttypes..., floattypes..., complextypes...)

    function check_solve(mat, spvec)
        fspvec = Array(spvec)
        T = typeof(zero(eltype(mat))*zero(eltype(spvec)) + zero(eltype(mat))*zero(eltype(spvec)))
        if !(mat isa Union{UnitLowerTriangular,UnitUpperTriangular})
            T = typeof(zero(T)/one(eltype(mat)))
        end
        @test (mat \ spvec)::Vector{T} ≈ mat \ fspvec
        if eltype(spvec) == T
            @test ldiv!(mat, copy(spvec)) ≈ ldiv!(mat, copy(fspvec))
        end
    end

    @testset "$eltypemat matrix, $eltypevec vector" for eltypemat in eltypes, eltypevec in eltypes
        (densemat, sparsemat) = eltypemat in inttypes ? (denseintmat, sparseintmat) :
                                eltypemat in floattypes ? (densefloatmat, sparsefloatmat) :
                                eltypemat in complextypes && (densecomplexmat, sparsecomplexmat)
        densemat = convert(Matrix{eltypemat}, densemat)
        sparsemat = convert(SparseMatrixCSC{eltypemat}, sparsemat)
        z = eltypevec <: Complex ? eltypevec(2 + 3im) : eltypevec(2)
        spvecs = (spzeros(eltypevec, m),
                  SparseVector(m, [1], [z]),
                  SparseVector(m, [m], [z]),
                  SparseVector(m, [3, 7], [z, -z]),
                  SparseVector(m, [1, 3, m], [zero(eltypevec), z, zero(eltypevec)]))
        for backing in (densemat, sparsemat), tri in WRAPPERS, transform in TRANSFORMS
            mat = transform(tri(backing))
            for spvec in spvecs
                check_solve(mat, spvec)
            end
            if backing isa Matrix
                @test which(\, Tuple{typeof(mat), typeof(first(spvecs))}).module === SparseArrays
                @test which(ldiv!, Tuple{typeof(mat), typeof(first(spvecs))}).module === SparseArrays
            end
        end
    end
end

@testset "sparse times triangular over every eltype and transform pair" begin
    _sparse_test_matrix(n, T) = T == Int ? sparse(rand(0:4, n, n)) : sprandn(T, n, n, 0.6)
    _triangular_test_matrix(n, TA, T) = T == Int ? TA(rand(0:9, n, n)) : TA(randn(T, n, n))
    n = 5
    @testset "$T1 sparse, $T2 $TM" for T1 in (Int, Float64, ComplexF32), T2 in (Int, Float64, ComplexF32), TM in WRAPPERS
        S = _sparse_test_matrix(n, T1)
        MS = Matrix(S)
        T = _triangular_test_matrix(n, TM, T2)
        MT = Matrix(T)
        for transT in TRANSFORMS, transS in TRANSFORMS
            @test (transT(T) * transS(S))::DenseMatrix ≈ transT(MT) * transS(MS)
            @test (transS(S) * transT(T))::DenseMatrix ≈ transS(MS) * transT(MT)
        end
    end
end

@testset "triangular sparse products and solves over sizes and index types" begin
    rng = MersenneTwister(0)
    @testset "n = $n, $T" for n in (1, 2, 5, 100, 127, 1000), T in (Float64, ComplexF64)
        density = min(1.0, 0.01 + 2/n)
        B = ones(n)
        X = T.(reshape(1.0:3n, 3, n) ./ n)
        s = sprandn(rng, T, n, 0.05)
        sd = Vector(s)
        A0 = sprand(rng, T, n, n, density)
        A0[1, 1] = T <: Complex ? T(2 + im) : T(2) # a stored diagonal entry, conjugated by adjoint
        MA0 = Matrix(A0)
        # the solves use a diagonally dominant matrix to keep rounding away from the tolerance
        A1 = A0 - Diagonal(diag(A0)) + 2I
        MA1 = Matrix(A1)
        transforms = T <: Complex ? (TRANSFORMS..., adjoint ∘ transpose, transpose ∘ adjoint) : TRANSFORMS
        for wr in WRAPPERS, tr in transforms
            MAW = tr(wr(MA0))
            refB, refs, refA, refX = MAW * B, MAW * sd, MAW * MA0, X * MAW
            # the solve reference is the materialized matrix: LinearAlgebra's dense solve with a
            # conjugating wrapper (`adjoint ∘ transpose`) of `UnitLowerTriangular` is wrong
            refsolve = Matrix(tr(wr(MA1))) \ B
            for Ti in (Int32, Int64)
                A = SparseMatrixCSC{T,Ti}(A0)
                AW = tr(wr(A))
                @test AW * B ≈ refB
                @test AW * s ≈ refs
                @test AW * A ≈ refA
                @test (X * AW)::Matrix ≈ refX
                @test rmul!(copy(X), AW) ≈ refX
                @test mul!(similar(X), view(X, [1, 2, 3], :), AW) ≈ refX
                tr === identity && @test AW * AW isa wr
                # a view of all rows and a unit range of columns
                vAW = tr(wr(view([zero(A)+I A], :, (n+1):2n)))
                @test vAW * B ≈ refB
                @test vAW * A ≈ refA
                @test X * vAW ≈ refX

                AS = SparseMatrixCSC{T,Ti}(A1)
                ASW = tr(wr(AS))
                @test ASW \ B ≈ refsolve
                @test !issparse(ASW \ B)
                vASW = tr(wr(view([zero(AS)+I AS], :, (n+1):2n)))
                @test vASW \ B ≈ refsolve
            end
        end
    end
end

@testset "Int8 index type at the capacity boundary (#816)" begin
    rows, cols = [1, 2, 5], [2, 1, 7]
    @testset "n = $n, $T" for n in (1, 2, 7, 127), T in (Float64, ComplexF64)
        keep = (rows .<= n) .& (cols .<= n)
        vals = T <: Complex ? T[2 + im, 3, 4 - im] : T[2, 3, 4]
        A8 = sparse(Int8.(rows[keep]), Int8.(cols[keep]), vals[keep], n, n)
        MA8 = Matrix(A8)
        B = ones(n)
        X = T.(reshape(1.0:3n, 3, n) ./ n)
        for wr in WRAPPERS, tr in TRANSFORMS
            AW = tr(wr(A8))
            MAW = tr(wr(MA8))
            @test AW * A8 ≈ MAW * MA8
            tr === identity && @test AW * A8 isa SparseMatrixCSC{T,Int8}
            @test AW * spzeros(T, n) == zeros(T, n)
            @test AW * B ≈ MAW * B
            @test X * AW ≈ X * MAW
            if wr in (UnitUpperTriangular, UnitLowerTriangular)
                @test AW \ B ≈ MAW \ B
            else
                @test_throws SingularException AW \ B
            end
        end
    end
end

end # module
