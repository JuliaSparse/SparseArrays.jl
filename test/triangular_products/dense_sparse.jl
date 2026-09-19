# This file is a part of Julia. License is MIT: https://julialang.org/license

@testset "multiplication of sparse matrix and triangular matrix" begin
    _sparse_test_matrix(n, T) =  T == Int ? sparse(rand(0:4, n, n)) : sprandn(T, n, n, 0.6)
    _triangular_test_matrix(n, TA, T) = T == Int ? TA(rand(0:9, n, n)) : TA(randn(T, n, n))

    function test_triangular_product(S, T)
        @test (T * S)::DenseMatrix ≈ Matrix(T) * Matrix(S)
        @test (S * T)::DenseMatrix ≈ Matrix(S) * Matrix(T)
    end

    n = 5
    @testset "wrappers" begin
        for ElType in (Float64, ComplexF32)
            S = _sparse_test_matrix(n, ElType)
            for TM in (LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular)
                T = _triangular_test_matrix(n, TM, ElType)
                for transT in (identity, adjoint, transpose), transS in (identity, adjoint, transpose)
                    test_triangular_product(transS(S), transT(T))
                end
            end
        end
    end
    @testset "promotion" begin
        for T1 in (Int, Float64, ComplexF32), T2 in (Int, Float64, ComplexF32)
            S = _sparse_test_matrix(n, T1)
            for TM in (LowerTriangular, UpperTriangular)
                test_triangular_product(S, _triangular_test_matrix(n, TM, T2))
            end
        end
    end
end
