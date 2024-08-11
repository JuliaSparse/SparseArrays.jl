using SparseArrays
using Documenter: DocMeta, makedocs, deploydocs

DocMeta.setdocmeta!(SparseArrays, :DocTestSetup, :(using SparseArrays; using LinearAlgebra); recursive=true)

makedocs(
    modules = [SparseArrays],
    sitename = "SparseArrays",
    pages = Any[
        "SparseArrays" => "index.md",
        "Sparse Linear Algebra API" => "solvers.md",
        ];
    warnonly = [:missing_docs, :cross_references],
    )

deploydocs(repo = "github.com/JuliaSparse/SparseArrays.jl.git")
