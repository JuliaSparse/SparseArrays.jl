# This file is a part of Julia. License is MIT: https://julialang.org/license

using Test, LinearAlgebra, SparseArrays

testfiles = [file * ".jl" for file in readlines(joinpath(@__DIR__, "testgroups")) if file != ""]

if Base.USE_GPL_LIBS
    push!(testfiles, "threads_suite.jl")
end

# ParallelTestRunner comes from the Pkg.test target; Julia base CI runs this
# file without it and falls back to the serial path.
if Base.find_package("ParallelTestRunner") !== nothing
    using ParallelTestRunner
    # Auto CPU thread count detection in ParallelTestRunner is bad
    push!(ARGS, "--jobs=$(Sys.CPU_THREADS)")
    testsuite = Dict{String,Expr}(splitext(f)[1] => :(include($(joinpath(@__DIR__, f))))
                                  for f in testfiles)
    runtests(SparseArrays, ARGS; testsuite)
else
    foreach(include, testfiles)
end
