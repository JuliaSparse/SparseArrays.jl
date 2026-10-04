# This file is a part of Julia. License is MIT: https://julialang.org/license

# Under `Pkg.test` Aqua comes from the test target. Julia CI doesn't run stdlib tests via
# `Pkg.test`, so there it is installed here, at the version `Project.toml` allows, into a
# temporary environment; the original environment is restored when `f` returns or throws.
const installed_aqua = Base.find_package("Aqua") === nothing
if installed_aqua
    import Pkg
end

const project = Base.parsed_toml(joinpath(dirname(@__DIR__), "Project.toml"))
const aqua_id = Base.PkgId(Base.UUID(project["extras"]["Aqua"]), "Aqua")

# Call `f(Aqua)` with Aqua loaded.
function with_aqua(f)
    original_depot_path = copy(Base.DEPOT_PATH)
    original_load_path = copy(Base.LOAD_PATH)
    original_env = copy(ENV)
    original_project = Base.active_project()
    try
        if installed_aqua
            @debug "Installing Aqua.jl for SparseArrays.jl tests"
            iob = IOBuffer()
            Pkg.activate(; temp = true)
            try
                Pkg.add(name="Aqua", version=project["compat"]["Aqua"], io=iob) # Needed for custom julia version resolve tests
            catch
                println(String(take!(iob)))
                rethrow()
            end
        end
        Base.invokelatest(f, Base.require(aqua_id))
    finally
        if installed_aqua
            empty!(Base.DEPOT_PATH)
            empty!(Base.LOAD_PATH)
            append!(Base.DEPOT_PATH, original_depot_path)
            append!(Base.LOAD_PATH, original_load_path)

            for k in setdiff(collect(keys(ENV)), collect(keys(original_env)))
                delete!(ENV, k)
            end
            for (k, v) in pairs(original_env)
                ENV[k] = v
            end

            Base.set_active_project(original_project)
        end
    end
end
