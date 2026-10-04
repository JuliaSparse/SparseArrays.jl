# This file is a part of Julia. License is MIT: https://julialang.org/license

module SparseAquaTests

using Test, LinearAlgebra, SparseArrays

include("aquahelper.jl")

# The ambiguity check is the `ambiguous` suite.
with_aqua() do Aqua
    @testset "code quality" begin
        Aqua.test_all(SparseArrays; ambiguities=false)
    end
end

end # module
