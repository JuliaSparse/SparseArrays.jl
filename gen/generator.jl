#! /bin/bash julia --project generator.jl
using Pkg
using Pkg.Artifacts
using Clang.Generators
using Clang.Generators.JLLEnvs
using SuiteSparse_jll

cd(@__DIR__)

# headers
include_dir = joinpath(SuiteSparse_jll.artifact_dir, "include") |> normpath
cholmod_h = joinpath(include_dir, "cholmod.h")
@assert isfile(cholmod_h)

SuiteSparseQR_C_h = joinpath(include_dir, "SuiteSparseQR_C.h")
@assert isfile(SuiteSparseQR_C_h)

umfpack_h = joinpath(include_dir, "umfpack.h")
@assert isfile(umfpack_h)

amd_h = joinpath(include_dir, "amd.h")
@assert isfile(amd_h)

colamd_h = joinpath(include_dir, "colamd.h")
@assert isfile(colamd_h)

ccolamd_h = joinpath(include_dir, "ccolamd.h")
@assert isfile(ccolamd_h)

# load common option
options = load_options(joinpath(@__DIR__, "generator.toml"))

# run generator for all platforms
for target in JLLEnvs.JLL_ENV_TRIPLES
    @info "processing $target"

    options["general"]["output_file_path"] = joinpath(@__DIR__, "..", "src/solvers/lib", "$target.jl")

    args = get_default_args(target)
    push!(args, "-I$include_dir")
    if startswith(target, "x86_64") || startswith(target, "powerpc64le") || startswith(target, "aarch64")
        push!(args, "-DSUN64 -DLONGBLAS='long long' -D'SuiteSparse_long_max=9223372036854775801'")
    end

    header_files = [cholmod_h, SuiteSparseQR_C_h, amd_h, colamd_h, ccolamd_h, umfpack_h]

    ctx = create_context(header_files, args, options)

    build!(ctx)
end
