# This file is a part of Julia. License is MIT: https://julialang.org/license

@testset "issue #13792, use sparse triangular solvers for sparse triangular solves" begin
    local A, n, x
    n = 100
    A, b = sprandn(n, n, 0.5) + sqrt(n)*I, fill(1., n)
    @test LowerTriangular(A)\(LowerTriangular(A)*b) ≈ b
    @test UpperTriangular(A)\(UpperTriangular(A)*b) ≈ b
    A[2,2] = 0
    dropzeros!(A)
    @test_throws LinearAlgebra.SingularException LowerTriangular(A)\b
    @test_throws LinearAlgebra.SingularException UpperTriangular(A)\b
end
