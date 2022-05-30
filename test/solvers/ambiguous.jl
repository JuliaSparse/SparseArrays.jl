using SparseArrays
using Test

@testset "detect_ambiguities" begin
    @test_nowarn detect_ambiguities(SparseArrays; recursive=true, ambiguous_bottom=false)
end
