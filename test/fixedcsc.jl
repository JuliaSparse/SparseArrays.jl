using Test, SparseArrays, LinearAlgebra
using SparseArrays: FixedSparseCSC, getcolptr, rowvals, nonzeros
@testset "main" begin
    A = sprandn(10, 10, 0.3)
    F = FixedSparseCSC(copy(A))
    nonzeros(F) .= 0
    dropzeros!(F)
    H = F ./ 1
    @test typeof(H) == typeof(F)
    @test getcolptr(H) == getcolptr(F)
    @test rowvals(H) == rowvals(F)
    H = map!(zero, copy(F), F)
    @test rowvals(H) == rowvals(F)
    @test getcolptr(H) == getcolptr(F)
    function f(F, A)
        @allocated F .= A .+ A
    end
    f(F, A)
    @test f(F, A) == 0


end

@testset "#190" begin
    J = fixed(sparse(Diagonal(ones(4))))
    W = fixed(sparse(Diagonal(ones(4))))
    J[4, 4] = 0
    gamma = 1.0
    W .= gamma .* J
    @test W == J

    x = fixed(sprandn(10, 10, 0.1))
    @test (x .= x .* 0; true)
    @test (x .= 0; true)
    @test (fill!(x, false); true)
end