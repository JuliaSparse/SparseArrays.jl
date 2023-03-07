# This file is a part of Julia. License is MIT: https://julialang.org/license

import Pkg

# Because julia CI doesn't run stdlib tests via `Pkg.test` test deps must be manually installed if missing
if Base.find_package("Aqua") === nothing
    @debug "Installing Aqua.jl for SparseArrays.jl tests"
    iob = IOBuffer()
    try
        Pkg.add("Aqua", io=iob) # Needed for custom julia version resolve tests
    catch
        println(String(take!(iob)))
        rethrow()
    end
end

using Test, LinearAlgebra, SparseArrays, Aqua

@testset "code quality" begin
    @testset "Method ambiguity" begin
        # Aqua.test_ambiguities([SparseArrays, Base, Core])
    end
    @testset "Unbound type parameters" begin
        @test_broken Aqua.detect_unbound_args_recursively(SparseArrays) == []
    end
    @testset "Undefined exports" begin
        Aqua.test_undefined_exports(SparseArrays) == []
    end
    @testset "Compare Project.toml and test/Project.toml" begin
        Aqua.test_project_extras(SparseArrays)
    end
    @testset "Stale dependencies" begin
        Aqua.test_stale_deps(SparseArrays)
    end
    @testset "Compat bounds" begin
        Aqua.test_deps_compat(SparseArrays)
    end
    @testset "Project.toml formatting" begin
        Aqua.test_project_toml_formatting(SparseArrays)
    end
    @testset "Piracy" begin
        @test_broken Aqua.Piracy.hunt(SparseArrays) == Method[]
    end
end

@testset "detect_ambiguities" begin
    @test_nowarn detect_ambiguities(SparseArrays; recursive=true, ambiguous_bottom=false)
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
