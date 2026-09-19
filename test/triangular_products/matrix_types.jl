# This file is a part of Julia. License is MIT: https://julialang.org/license

@testset "Multiplying with triangular sparse matrices #35609 #35610" begin
    n = 10
    A = sprand(n, n, 5/n)
    U = UpperTriangular(A)
    L = LowerTriangular(A)
    AM = Matrix(A)
    UM = Matrix(U)
    LM = Matrix(L)
    Y = A * U
    @test Y ≈ AM * UM
    @test typeof(Y) == typeof(A)
    Y = A * L
    @test Y ≈ AM * LM
    @test typeof(Y) == typeof(A)
    Y = U * A
    @test Y ≈ UM * AM
    @test typeof(Y) == typeof(A)
    Y = L * A
    @test Y ≈ LM * AM
    @test typeof(Y) == typeof(A)
    Y = U * U
    @test Y ≈ UM * UM
    @test typeof(Y) == typeof(U)
    Y = L * L
    @test Y ≈ LM * LM
    @test typeof(Y) == typeof(L)
    Y = L * U
    @test Y ≈ LM * UM
    @test typeof(Y) == typeof(A)
    Y = U * L
    @test Y ≈ UM * LM
    @test typeof(Y) == typeof(A)
end
