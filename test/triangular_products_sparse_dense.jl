# This file is a part of Julia. License is MIT: https://julialang.org/license

@testset "multiplication of triangular sparse and dense matrices" begin
    n = 7
    B = rand(n, 3)
    _triangular_sparse_matrix(n, ULT, T) = T == Int ? ULT(sparse(rand(0:10, n, n))) : ULT(sprandn(T, n, n, 0.4))
    for T in (Int, Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64)
        for AT in (adjoint, transpose)
            for TR in (UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular)
                TS = AT(_triangular_sparse_matrix(n, TR, T))
                @test isa(TS * B, DenseMatrix)
                @test TS * B ≈ Matrix(TS)*B
            end
        end
    end
end
