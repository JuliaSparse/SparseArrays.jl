# This file is a part of Julia. License is MIT: https://julialang.org/license
using Test, LinearAlgebra, SparseArrays
@testset "detect_ambiguities" begin
    @test_nowarn detect_ambiguities(SparseArrays; recursive=true, ambiguous_bottom=false)
    @test_nowarn detect_ambiguities(LinearAlgebra; recursive=true, ambiguous_bottom=false)
end

## This was the older version that was disabled

# let ambig = detect_ambiguities(SparseArrays; recursive=true)
#     @test isempty(ambig)
#     ambig = Set{Any}(((m1.sig, m2.sig) for (m1, m2) in ambig))
#     expect = []
#     good = true
#     while !isempty(ambig)
#         sigs = pop!(ambig)
#         i = findfirst(==(sigs), expect)
#         if i === nothing
#             println(stderr, "push!(expect, (", sigs[1], ", ", sigs[2], "))")
#             good = false
#             continue
#         end
#         deleteat!(expect, i)
#     end
#     @test isempty(expect)
#     @test good
# end
