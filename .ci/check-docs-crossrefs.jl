#!/usr/bin/env julia
# This file is a part of Julia. License is MIT: https://julialang.org/license

# Julia's manual includes `docs/src/index.md` and the docstrings it lists, but not
# `docs/src/solvers.md`. An `@ref` from the former to a section or docstring that only
# the solvers page provides resolves in this package's own docs build and fails
# `makedocs` in JuliaLang/julia. This lint fails on such a cross-reference in
# `index.md` or in a docstring outside `src/solvers/`.

using LinearAlgebra

const index_page = joinpath("docs", "src", "index.md")
const solvers_page = joinpath("docs", "src", "solvers.md")
const solver_dir = "solvers"

const header = r"^#+\s+(?:\[(.*)\]\(@id\s+([^)\s]+)\)|(.*\S))\s*$"
const cross_reference = r"\[([^\]]*)\]\(@ref\s*([^)]*)\)"

const is_gha = something(tryparse(Bool, get(ENV, "GITHUB_ACTIONS", "false")), false)

# The section titles and `@id` labels of a page, and the entries of its `@docs` blocks.
function page_targets(path)
    sections, docstrings = Set{String}(), Set{String}()
    fence = nothing
    for line in eachline(path)
        if startswith(line, "```")
            fence = fence === nothing ? line : nothing
        elseif fence === nothing
            m = match(header, line)
            m === nothing || union!(sections, filter(!isnothing, m.captures))
        elseif startswith(fence, "```@docs")
            isempty(strip(line)) || push!(docstrings, strip(line))
        end
    end
    return sections, docstrings
end

# `SparseArrays.CHOLMOD.Factor` is referred to as `CHOLMOD.Factor` or `Factor`. A bare
# name that Base or LinearAlgebra also define, such as `lu`, resolves in Julia's manual.
function docstring_names(entry)
    parts = split(replace(entry, r"^SparseArrays\." => "", r"\(.*$" => ""), '.')
    names = [join(parts[i:end], '.') for i in eachindex(parts)]
    sym = Symbol(strip(last(parts), ':'))
    (isdefined(Base, sym) || isdefined(LinearAlgebra, sym)) && pop!(names)
    return names
end

function solver_only_targets(repo)
    sections, docstrings = page_targets(joinpath(repo, solvers_page))
    index_sections, index_docstrings = page_targets(joinpath(repo, index_page))
    names = Set{String}()
    for entry in setdiff(docstrings, index_docstrings)
        union!(names, docstring_names(entry))
    end
    return union!(names, setdiff(sections, index_sections))
end

function reference_target(text, target)
    target = replace(strip(target), r"^SparseArrays\." => "")
    isempty(target) || return target
    code = match(r"^`(.*)`$"s, text)
    code === nothing && return replace(text, r"\s+" => " ")
    return replace(code[1], r"^SparseArrays\." => "", r"[({].*$"s => "")
end

function files_to_check(repo)
    paths = [index_page]
    for (dir, _, names) in walkdir(joinpath(repo, "src"))
        rel = relpath(dir, joinpath(repo, "src"))
        (rel == solver_dir || startswith(rel, solver_dir * "/")) && continue
        for name in names
            endswith(name, ".jl") && push!(paths, relpath(joinpath(dir, name), repo))
        end
    end
    return sort!(paths)
end

function check_file(repo, path, targets)
    errors = Tuple{Int,String}[]
    content = read(joinpath(repo, path), String)
    for m in eachmatch(cross_reference, content)
        reference_target(m[1], m[2]) in targets || continue
        lineno = count(==('\n'), SubString(content, 1, m.offset)) + 1
        push!(errors, (lineno, replace(m.match, r"\s+" => " ")))
    end
    return errors
end

function check_docs_crossrefs()
    repo = dirname(@__DIR__)
    targets = solver_only_targets(repo)
    nerrors = 0
    for path in files_to_check(repo), (lineno, text) in check_file(repo, path, targets)
        nerrors += 1
        msg = "`$text` links to $solvers_page, which Julia's manual does not include; use plain text"
        println(stderr, "$path:$lineno -- $msg")
        is_gha && println(stdout, "::error title=Docs cross-reference check,file=", path, ",line=", lineno, "::", msg)
    end
    if nerrors == 0
        println(stderr, "Docs cross-reference check found no issues.")
        exit(0)
    else
        println(stderr, "Docs cross-reference check found $nerrors issues.")
        exit(1)
    end
end

check_docs_crossrefs()
