# This file is a part of Julia. License is MIT: https://julialang.org/license

@testset "issue #19304" begin
    @inferred hcat(sparse(rand(2,1)), I)
    @inferred hcat(sparse(rand(2,1)), 1.0I)
    @inferred hcat(sparse(rand(2,1)), Matrix(I, 2, 2))
    @inferred hcat(sparse(rand(2,1)), Matrix(1.0I, 2, 2))
end
