# This file is a part of Julia. License is MIT: https://julialang.org/license

# Test that concatenations of combinations of sparse matrices with sparse matrices or dense
# matrices/vectors yield sparse arrays
@testset "sparse and dense concatenations" begin
    N = 4
    densevec = fill(1., N)
    densemat = diagm(0 => densevec)
    spmat = spdiagm(0 => densevec)
    # Test that concatenations of pairs of sparse matrices yield sparse arrays
    @test issparse(vcat(spmat, spmat))
    @test issparse(hcat(spmat, spmat))
    @test issparse(@inferred(hvcat((2,), spmat, spmat)))
    @test issparse(cat(spmat, spmat; dims=(1,2)))
    # Test that concatenations of a sparse matrice with a dense matrix/vector yield sparse arrays
    @test issparse(vcat(spmat, densemat))
    @test issparse(vcat(densemat, spmat))
    for densearg in (densevec, densemat)
        @test issparse(hcat(spmat, densearg))
        @test issparse(hcat(densearg, spmat))
        @test issparse(hvcat((2,), spmat, densearg))
        @test issparse(hvcat((2,), densearg, spmat))
        @test issparse(cat(spmat, densearg; dims=(1,2)))
        @test issparse(cat(densearg, spmat; dims=(1,2)))
    end
end
