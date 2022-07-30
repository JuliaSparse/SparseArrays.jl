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
end