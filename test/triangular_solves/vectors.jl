# This file is a part of Julia. License is MIT: https://julialang.org/license

@testset "ldiv with different element types (#40171)" begin
    sA = sparse(Int16.(1:4), Int16.(1:4), ones(4))
    @test all(ldiv!(LowerTriangular(sA), ones(4)) .≈ 1.)
end
@testset "ldiv ops with triangular matrices and sparse vecs (#14005)" begin
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
    coretypes = (Int64, Float64, ComplexF64)

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

    @testset "wrapper dispatch and active-index boundaries" for T in (Float64, ComplexF64)
        densemat, sparsemat = T == Float64 ? (densefloatmat, sparsefloatmat) :
                                            (densecomplexmat, sparsecomplexmat)
        z = T == Float64 ? T(2) : T(2 + 3im)
        spvecs = (spzeros(T, m),
                  SparseVector(m, [1], [z]),
                  SparseVector(m, [m], [z]),
                  SparseVector(m, [3, 7], [z, -z]),
                  SparseVector(m, [1, 3, m], [zero(T), z, zero(T)]))
        for backing in (densemat, sparsemat), tri in (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular),
            transform in (identity, adjoint, transpose)
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

    @testset "scalar promotion" for eltypemat in eltypes
        (densemat, sparsemat) = eltypemat in inttypes ? (denseintmat, sparseintmat) :
                                eltypemat in floattypes ? (densefloatmat, sparsefloatmat) :
                                eltypemat in complextypes && (densecomplexmat, sparsecomplexmat)
        densemat = convert(Matrix{eltypemat}, densemat)
        sparsemat = convert(SparseMatrixCSC{eltypemat}, sparsemat)
        for eltypevec in eltypes
            (eltypemat in coretypes && eltypevec in coretypes) ||
                eltypemat == Float64 || eltypevec == Float64 || continue
            vals = eltypevec <: Complex ? eltypevec[2 + 3im, -1 + 2im, 3 - im] : eltypevec[2, -1, 3]
            spvec = SparseVector(m, [2, 5, 8], vals)
            for backing in (densemat, sparsemat), tri in (LowerTriangular, UnitLowerTriangular)
                check_solve(tri(backing), spvec)
            end
        end
    end
end
@testset "#16716" begin
    origmat = [-1.5 -0.7; 0.0 1.0]
    transmat = copy(origmat')
    utmat = UpperTriangular(origmat)
    ltmat = LowerTriangular(transmat)
    uutmat = LinearAlgebra.UnitUpperTriangular(origmat)
    ultmat = LinearAlgebra.UnitLowerTriangular(transmat)

    zerospvec = spzeros(Float64, 2)
    zerodvec = zeros(Float64, 2)

    for mat in (utmat, ltmat, uutmat, ultmat)
        @test isequal(\(mat, zerospvec), zerodvec)
        @test isequal(\(adjoint(mat), zerospvec), zerodvec)
        @test isequal(\(transpose(mat), zerospvec), zerodvec)
        @test isequal(ldiv!(mat, copy(zerospvec)), zerospvec)
        @test isequal(ldiv!(adjoint(mat), copy(zerospvec)), zerospvec)
        @test isequal(ldiv!(transpose(mat), copy(zerospvec)), zerospvec)
    end
end
