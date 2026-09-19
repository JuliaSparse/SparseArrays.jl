# This file is a part of Julia. License is MIT: https://julialang.org/license

verbose = "--verbose" in ARGS
suites = filter(!=("--verbose"), ARGS)
length(suites) == 1 || error("Usage: julia +nightly --project .ci/measure-tests.jl SUITE [--verbose]")
suite = only(suites)
occursin(r"^[A-Za-z0-9_]+$", suite) || error("Expected a test suite name without .jl")
root = dirname(@__DIR__)
testfile = joinpath(root, "test", suite * ".jl")
isfile(testfile) || error("Unknown test suite: $suite")
ENV["JULIA_TEST_VERBOSE"] = string(verbose)

using Test, Random, SparseArrays, LinearAlgebra

samefile(pathof(SparseArrays), joinpath(root, "src", "SparseArrays.jl")) ||
    error("SparseArrays must load from this checkout; launch Julia with --project=$root")
Random.seed!(1234)
BLAS.set_num_threads(1)
Base.cumulative_compile_timing(true)
stats = @timed @testset "$suite" verbose=verbose begin
    include(testfile)
end
println("MEASURE\t", suite, "\ttime=", stats.time, "\tcompile=", stats.compile_time,
        "\tgc=", stats.gctime, "\tbytes=", stats.bytes,
        "\tversion=", VERSION, "\tsource=", pathof(SparseArrays))
