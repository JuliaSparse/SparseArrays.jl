#!/usr/bin/env julia
# This file is a part of Julia. License is MIT: https://julialang.org/license

# Everything that needs the GPL SuiteSparse libraries lives in `src/solvers/` and
# `test/solvers/`, so the other suites pass on a build without GPL libraries (see
# AGENTS.md). This lint fails when any other Julia file under `src/` or `test/`
# names a solver module, a solver factorization type or a SuiteSparse library, or
# factorizes a sparse literal. It cannot see a factorization of a sparse variable,
# so it is a backstop for review, not a replacement.

const roots = ("src", "test")
const solver_dir = "solvers"
const solver_names = r"\b(CHOLMOD|UMFPACK|SPQR|LibSuiteSparse|SuiteSparse_jll|UmfpackLU|QRSparse|libcholmod|libumfpack|libspqr|libsuitesparseconfig)\b"
const sparse_factorization = r"\b(lu|qr|cholesky|ldlt|factorize)!?\(\s*(sparse|sprand|sprandn|spdiagm|spzeros|SparseMatrixCSC|SparseVector)\b"
# `src/SparseArrays.jl` is the one place outside `src/solvers/` that loads the solvers.
const allowed = r"^\s*(include\(\"solvers/|(using|import) \.LibSuiteSparse\b)"

const is_gha = something(tryparse(Bool, get(ENV, "GITHUB_ACTIONS", "false")), false)

function files_to_check(repo)
    paths = String[]
    for root in roots, (dir, _, names) in walkdir(joinpath(repo, root))
        rel = relpath(dir, joinpath(repo, root))
        (rel == solver_dir || startswith(rel, solver_dir * "/")) && continue
        for name in names
            endswith(name, ".jl") && push!(paths, relpath(joinpath(dir, name), repo))
        end
    end
    return sort!(paths)
end

function check_file(repo, path)
    errors = Tuple{Int,String}[]
    in_docstring = false
    for (lineno, line) in enumerate(eachline(joinpath(repo, path)))
        toggles = isodd(count("\"\"\"", line))
        if in_docstring || toggles
            toggles && (in_docstring = !in_docstring)
            continue
        end
        (startswith(lstrip(line), '#') || occursin(allowed, line)) && continue
        m = something(match(solver_names, line), match(sparse_factorization, line), Some(nothing))
        m === nothing || push!(errors, (lineno, m.match))
    end
    return errors
end

function check_gpl_usage()
    repo = dirname(@__DIR__)
    nerrors = 0
    for path in files_to_check(repo), (lineno, text) in check_file(repo, path)
        nerrors += 1
        msg = "`$text` outside $solver_dir/; solver code and tests belong in src/$solver_dir/ and test/$solver_dir/"
        println(stderr, "$path:$lineno -- $msg")
        is_gha && println(stdout, "::error title=GPL usage check,file=", path, ",line=", lineno, "::", msg)
    end
    if nerrors == 0
        println(stderr, "GPL usage check found no issues.")
        exit(0)
    else
        println(stderr, "GPL usage check found $nerrors issues.")
        exit(1)
    end
end

check_gpl_usage()
