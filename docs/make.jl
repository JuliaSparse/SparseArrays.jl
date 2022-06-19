using SparseArrays
using Documenter: DocMeta, makedocs, deploydocs

DocMeta.setdocmeta!(SparseArrays, :DocTestSetup, :(using SparseArrays; using LinearAlgebra); recursive=true)

makedocs(
    modules = [SparseArrays],
    sitename = "SparseArrays",
    pages = Any[
        "SparseArrays" => "index.md",
        "Solvers" => "solvers.md",
        ];
    # strict = true,
    strict = Symbol[:doctest],
    )

deploydocs(repo = "github.com/JuliaLang/SparseArrays.jl.git")
