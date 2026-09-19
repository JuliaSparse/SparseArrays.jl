# This file is a part of Julia. License is MIT: https://julialang.org/license

@testset "multiplication of Triangular sparse matrices with sparse vectors #35642" begin
    n = 10
    A = sprand(n, n, 5/n)
    U = UpperTriangular(A)
    L = LowerTriangular(A)
    x = sprand(n, 5/n)
    y = view(A, :, 6)
    z = view(x, :)
    ty = typeof
    @testset "matvec multiplication $(ty(X)) * $(ty(v))" for X in (U, L), v in (x, y, z)
        @test X * v ≈ Matrix(X) * Vector(v)
        @test typeof(X * v) == typeof(x)
    end
end
