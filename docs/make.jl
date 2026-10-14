using SparseArrays
using Documenter: Documenter, DocMeta, makedocs, deploydocs

DocMeta.setdocmeta!(SparseArrays, :DocTestSetup, :(using SparseArrays; using LinearAlgebra); recursive=true)

makedocs(
    modules = [SparseArrays],
    sitename = "SparseArrays",
    pages = Any[
        "SparseArrays" => "index.md",
        ];
    # The same limits as Julia's manual, which renders `index.md` as one page.
    format = Documenter.HTML(size_threshold = 800 * 2^10, size_threshold_warn = 200 * 2^10),
    warnonly = [:cross_references],
    checkdocs = :public,
    )

deploydocs(repo = "github.com/JuliaSparse/SparseArrays.jl.git")
