#! /bin/bash julia --project generator.jl
# -*- julia -*-

using Pkg
using Pkg.Artifacts
using Clang.Generators
using Clang.Generators.JLLEnvs
using SuiteSparse_jll
using JuliaFormatter

cd(@__DIR__)

# headers
include_dir = joinpath(SuiteSparse_jll.artifact_dir, "include", "suitesparse") |> normpath

if !isfile(joinpath(include_dir, "cholmod.h"))
    include_dir = joinpath(SuiteSparse_jll.artifact_dir, "include") |> normpath
end

cholmod_h = joinpath(include_dir, "cholmod.h")
@assert isfile(cholmod_h)

SuiteSparseQR_C_h = joinpath(include_dir, "SuiteSparseQR_C.h")
@assert isfile(SuiteSparseQR_C_h)

umfpack_h = joinpath(include_dir, "umfpack.h")
@assert isfile(umfpack_h)

# load common option
options = load_options(joinpath(@__DIR__, "generator.toml"))

# we only generate a single wrapper for all platforms, because the headers are currently not
# platform dependent. since this package is part of the default Julia distribution, we also
# need to make sure that it can handle all platforms, including new ones that are not yet
# supported by BinaryBuilder (the easiest solution here is to always use a single wrapper).
options["general"]["output_file_path"] = joinpath(@__DIR__, "..", "src/solvers/wrappers.jl")
args = get_default_args()
push!(args, "-I$include_dir")

header_files = [cholmod_h, SuiteSparseQR_C_h, umfpack_h]

ctx = create_context(header_files, args, options)

build!(ctx)

path = options["general"]["output_file_path"]
format_file(path, YASStyle())
