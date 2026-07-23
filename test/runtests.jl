# This file is a part of Julia. License is MIT: https://julialang.org/license

using Test, LinearAlgebra, SparseArrays

include("util/gha.jl")

testfiles = ["allowscalar.jl", "fixed.jl", "higherorderfns.jl",
             "sparsematrix_constructors_indexing.jl", "sparsematrix_ops.jl",
             "sparsevector.jl", "issues.jl"]

if Base.USE_GPL_LIBS
    append!(testfiles, ["cholmod.jl", "umfpack.jl", "spqr.jl", "linalg.jl",
                        "linalg_solvers.jl"])
    if Sys.iswindows() && is_github_actions_ci()
        @warn "Skipping `threads` tests on Windows on GitHub Actions CI"
    else
        push!(testfiles, "threads_suite.jl")
    end
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
