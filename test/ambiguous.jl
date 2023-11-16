# This file is a part of Julia. License is MIT: https://julialang.org/license

###
# We'll restore the original env at the end of this testgroup.
original_depot_path = copy(Base.DEPOT_PATH)
original_load_path = copy(Base.LOAD_PATH)
original_env = copy(ENV)
original_project = Base.active_project()
###

import Pkg

# Because julia CI doesn't run stdlib tests via `Pkg.test` test deps must be manually installed if missing
if Base.find_package("Aqua") === nothing
    @debug "Installing Aqua.jl for SparseArrays.jl tests"
    iob = IOBuffer()
    Pkg.activate(; temp = true)
    try
        # TODO: make this version tie to compat in Project.toml
        # or do this another safer way
        Pkg.add(name="Aqua", version="0.8", io=iob) # Needed for custom julia version resolve tests
    catch
        println(String(take!(iob)))
        rethrow()
    end
end

using Test, LinearAlgebra, SparseArrays, Aqua

@testset "code quality" begin
    Aqua.test_all(SparseArrays; unbound_args=(; broken=true), piracies=(; broken=true))
end

let ambig = detect_ambiguities(SparseArrays; recursive=true)
    @test isempty(ambig)
    ambig = Set{Any}(((m1.sig, m2.sig) for (m1, m2) in ambig))
    expect = []
    good = true
    while !isempty(ambig)
        sigs = pop!(ambig)
        i = findfirst(==(sigs), expect)
        if i === nothing
            println(stderr, "push!(expect, (", sigs[1], ", ", sigs[2], "))")
            good = false
            continue
        end
        deleteat!(expect, i)
    end
    @test isempty(expect)
    @test good
end

###
# Now we restore the original env, as promised
empty!(Base.DEPOT_PATH)
empty!(Base.LOAD_PATH)
append!(Base.DEPOT_PATH, original_depot_path)
append!(Base.LOAD_PATH, original_load_path)

for k in setdiff(collect(keys(ENV)), collect(keys(original_env)))
    delete!(ENV, k)
end
for (k, v) in pairs(original_env)
    ENV[k] = v
end

Base.set_active_project(original_project)
###
