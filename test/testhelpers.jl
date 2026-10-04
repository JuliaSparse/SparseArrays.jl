# This file is a part of Julia. License is MIT: https://julialang.org/license

# Included by every suite module at its top. The shared types and helpers live in the
# `SparseTestHelpers` module, loaded into `Main` once per process. `ambiguous.jl` does not
# include this file, so that the helpers stay out of its checks.

isdefined(Main, :SparseTestHelpers) ||
    Base.include(Main, joinpath(@__DIR__, "SparseTestHelpers.jl"))
using Main.SparseTestHelpers

# Test time is compilation, and a good part of it is the suites' own code: long testset
# bodies that run once. This compiles the including suite module without optimization. The
# methods under test belong to other modules and are compiled as usual.
Base.Experimental.@compiler_options optimize=0
