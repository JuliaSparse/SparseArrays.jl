#!/usr/bin/env julia
# This file is a part of Julia. License is MIT: https://julialang.org/license

# The wrappers in src/solvers/wrappers.jl are generated from the SuiteSparse headers of the
# version named in gen/Makefile, and must match the SuiteSparse_jll the package is built
# against, which is the compat entry in Project.toml.

using TOML

const root = normpath(joinpath(@__DIR__, ".."))

const compat = TOML.parsefile(joinpath(root, "Project.toml"))["compat"]["SuiteSparse_jll"]

const ver = let m = match(r"^VER=(\S+)$"m, read(joinpath(root, "gen", "Makefile"), String))
    m === nothing && error("no `VER=` line in gen/Makefile")
    m[1]
end

if compat != ver
    error("gen/Makefile has VER=$ver but Project.toml has SuiteSparse_jll = \"$compat\"; keep them in sync")
end
println("gen/Makefile VER=$ver matches the Project.toml SuiteSparse_jll compat")
