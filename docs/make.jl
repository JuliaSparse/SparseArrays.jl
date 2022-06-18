using SparseArrays
using Documenter: DocMeta, makedocs, deploydocs

DocMeta.setdocmeta!(SparseArrays, :DocTestSetup, :(using SparseArrays; using LinearAlgebra); recursive=true)

makedocs(
    modules = [SparseArrays, SparseArrays.Solvers],
    sitename = "SparseArrays",
    pages = Any[
        "SparseArrays" => "index.md",
        "SparseArrays.Solvers" => "Sparse Solvers",
        ];
    # strict = true,
    strict = Symbol[:doctest],
    )

deploydocs(repo = "github.com/JuliaLang/SparseArrays.jl.git")
