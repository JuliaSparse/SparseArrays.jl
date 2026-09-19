# This file is a part of Julia. License is MIT: https://julialang.org/license

@testset "issue #14816" begin
    m = 5
    intmat = fill(1, m, m)
    ltintmat = LowerTriangular(rand(1:5, m, m))
    @test \(transpose(ltintmat), sparse(intmat)) ≈ \(transpose(ltintmat), intmat)
end
