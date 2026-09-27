# This file is a part of Julia. License is MIT: https://julialang.org/license

using Test, LinearAlgebra, SparseArrays

# The suite inventory, shared by ParallelTestRunner and the serial fallback. A suite is
# named by its path under `test/` without the extension. Command-line selectors match
# those names by prefix, and a `!prefix` selector excludes instead.
testfiles = ["allowscalar.jl", "fixed.jl", "higherorderfns.jl",
             "sparsematrix.jl", "constructors.jl", "indexing.jl", "reductions.jl",
             "sparsevector.jl", "linalg.jl", "matmul.jl",
             "triangular.jl", "concatenation.jl", "ambiguous.jl"]

@static if Base.USE_GPL_LIBS
    append!(testfiles, "solvers/" .* ["cholmod.jl", "umfpack.jl", "spqr.jl", "solvers.jl", "threads.jl"])
end

# The torture suite: long-tail regression tests and exhaustive grids that the core suite
# above does not need. It is listed here so that the `torture` selector reaches it on
# both paths, and skipped otherwise.
torturefiles = "torture/" .* ["constructors.jl", "indexing.jl", "sparsematrix.jl",
                              "higherorderfns.jl", "sparsevector.jl", "reductions.jl",
                              "linalg.jl", "matmul.jl", "concatenation.jl"]
append!(testfiles, torturefiles)

suitename(f) = splitext(f)[1]

# Suites that run only when a selector names them. The Aqua and ambiguity checks in
# `ambiguous.jl` get their own CI job; the torture suite runs on manual dispatch.
skipped_by_default(name) = name == "ambiguous" || startswith(name, "torture/")

matches(name, selectors) = any(sel -> startswith(name, sel), selectors)

# ParallelTestRunner comes from the Pkg.test target; Julia base CI runs this
# file without it and falls back to the serial path.
if Base.find_package("ParallelTestRunner") !== nothing
    using ParallelTestRunner
    # ParallelTestRunner sizes its worker pool from the CPU count and free memory; use the
    # CPU count unless the caller chose a job count, on the command line or in the
    # environment. `--jobs` may be given only once, so it must not be added twice.
    if !any(a -> a == "--jobs" || startswith(a, "--jobs="), ARGS) && !haskey(ENV, "PTR_NUM_JOBS")
        push!(ARGS, "--jobs=$(Sys.CPU_THREADS)")
    end
    get(ENV, "CI", "false") == "true" && push!(ARGS, "--verbose")
    testsuite = Dict{String,Expr}(suitename(f) => :(include($(joinpath(@__DIR__, f))))
                                  for f in testfiles)
    args = parse_args(ARGS)
    filter_tests!(testsuite, args)
    # `--list` shows the whole catalog; otherwise a skipped-by-default suite runs only
    # when an include selector names it.
    if args.list === nothing
        includes = filter(!startswith("!"), args.positionals)
        for name in collect(keys(testsuite))
            skipped_by_default(name) && !matches(name, includes) && delete!(testsuite, name)
        end
    end
    runtests(SparseArrays, args; testsuite)
else
    names = map(suitename, testfiles)
    options = filter(startswith("-"), ARGS)
    if options == ["--list"]
        println("Available tests:")
        foreach(name -> println(" - ", name), sort(names))
    elseif !isempty(options)
        error("the serial test runner takes suite selectors only, not $(join(options, " ")); " *
              "run the tests through Pkg.test for ParallelTestRunner's options")
    else
        selectors = filter(!startswith("-"), ARGS)
        excludes = lstrip.(filter(startswith("!"), selectors), '!')
        includes = filter(!startswith("!"), selectors)
        selected = filter(names) do name
            (isempty(includes) ? !skipped_by_default(name) : matches(name, includes)) &&
                !matches(name, excludes)
        end
        isempty(selected) && error("no test suite matches $(repr(selectors)); the suites are $(join(names, ", "))")
        for f in testfiles
            suitename(f) in selected && include(f)
        end
    end
end
