# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseAmbiguityTests

using Test, LinearAlgebra, SparseArrays

include("aquahelper.jl")

with_aqua() do Aqua
    @testset "ambiguities" begin
        Aqua.test_ambiguities(SparseArrays)
    end
end

end # module
