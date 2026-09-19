# This file is a part of Julia. License is MIT: https://julialang.org/license

@testset "Triangular and SparseVector multiplications" begin
    n = 10
    types = (Int, Float64, ComplexF64)
    tritypes = (LowerTriangular, UnitUpperTriangular)
    for ta in types
        for tri in tritypes
            if ta == Int
                T = tri(rand(1:9, n, n))
            else
                T = tri(randn(ta, n, n))
            end
            for tb in types
                if tb == Int
                    x = sparse(rand(0:4, n))
                else
                    x = sprandn(tb, n, 0.6)
                end
                @test T * x ≈ Array(T) * Array(x)
                @test T' * x ≈ Array(T)' * Array(x)
                @test transpose(T) * x ≈ transpose(Array(T)) * Array(x)
                @test x' * T ≈ Array(x)' * Array(T)
                @test x' * T' ≈ Array(x)' * Array(T)'
                @test x' * transpose(T) ≈ Array(x)' * transpose(Array(T))
            end
        end
    end

    # 0-dimensional case
    x = sparse(zeros(0))
    for tri in tritypes
        T = tri(zeros(0, 0))
        @test T*x == Array(T) * Array(x)
        @test T' * x == Array(T)' * Array(x)
        @test transpose(T) * x == transpose(Array(T)) * Array(x)
        @test x' * T == Array(x)' * Array(T)
        @test x' * T' == Array(x)' * Array(T)'
        @test x' * transpose(T) == Array(x)' * transpose(Array(T))
    end
end
