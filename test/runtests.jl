# This file is a part of Julia. License is MIT: https://julialang.org/license

using Test, LinearAlgebra, SparseArrays

testfiles = ["allowscalar.jl", "fixed.jl", "higherorderfns_core.jl",
             "sparsematrix_constructors_indexing.jl", "sparsematrix_ops.jl",
             "sparsevector_core.jl", "issues.jl", "linalg_core.jl", "linalg_products.jl",
             "threads_suite.jl", "triangular.jl", "concatenation.jl"]

if Base.USE_GPL_LIBS
    append!(testfiles, ["cholmod.jl", "umfpack.jl", "spqr.jl",
                        "linalg_solvers.jl"])
end

# ParallelTestRunner comes from the Pkg.test target; Julia base CI runs this
# file without it and falls back to the serial path.
if Base.find_package("ParallelTestRunner") !== nothing
    using ParallelTestRunner
    # Auto CPU thread count detection in ParallelTestRunner is bad
    push!(ARGS, "--jobs=$(Sys.CPU_THREADS)")
    get(ENV, "CI", "false") == "true" && push!(ARGS, "--verbose")
    testsuite = Dict{String,Expr}(splitext(f)[1] => :(include($(joinpath(@__DIR__, f))))
                                  for f in testfiles)
    runtests(SparseArrays, ARGS; testsuite)
else
    foreach(include, testfiles)
end
