using Documenter, SparseArrays

makedocs(
    modules = [SparseArrays],
    sitename = "SparseArrays",
    pages = Any[
        "SparseArrays" => "index.md"
        ]
    )

deploydocs(repo = "github.com/JuliaLang/SparseArrays.jl.git")
