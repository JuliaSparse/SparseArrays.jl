module LibSuiteSparse

using SuiteSparse_jll

const TRUE  = Int32(1)
const FALSE = Int32(0)

include("wrappers.jl")

const SUITESPARSE_MIN_VERSION = v"6.0.0"
const BUILD_VERSION = VersionNumber(
    SUITESPARSE_MAIN_VERSION,
    SUITESPARSE_SUB_VERSION,
    SUITESPARSE_SUBSUB_VERSION
)

public init_suitesparse

"""
    LibSuiteSparse.init_suitesparse

Internal function which is used to initialize the SuiteSparse libraries to the correct
    memory management functions. Any package which directly wraps one of the following
    SuiteSparse libraries *must* ensure that this function is called before the use of that
    library: AMD, CAMD, COLAMD, CCOLAMD, UMFPACK, CXSparse, CHOLMOD, KLU, BTF, LDL, RBio,
    SPQR, SPEX, and ParU

# Notes:
- Currently this function only sets the memory management functions of SuiteSparse_config,
    however there are also override functions for `printf`, `hypot`, and `divcomplex`.
- SuiteSparse_config, and this initialization function, is not a dependency of CSparse,
    GraphBLAS, or LAGraph.
"""
const init_suitesparse = Base.OncePerProcess{Nothing}() do
    try
        ### Check if the linked library is compatible with the Julia code
        if Libdl.dlsym_e(Libdl.dlopen("libsuitesparseconfig"), :SuiteSparse_version) != C_NULL
            current_version_array = Vector{Cint}(undef, 3)
            SuiteSparse_version(current_version_array)
            current_version = VersionNumber(current_version_array...)
        else # SuiteSparse < 4.2.0 does not include SuiteSparse_version()
            current_version = v"0.0.0"
        end


        if current_version < SUITESPARSE_MIN_VERSION
            @warn """
                SuiteSparse version incompatibility

                Julia was compiled with SuiteSparse version $BUILD_VERSION. It is
                currently linked with a version older than
                $(SUITESPARSE_MIN_VERSION). This might cause Julia to
                terminate when working with sparse matrix factorizations,
                e.g. solving systems of equations with \\.

                It is recommended that you use Julia with a recent version
                of SuiteSparse, or download the generic binaries
                from www.julialang.org, which ship with the correct
                versions of all dependencies.
                """
        elseif BUILD_VERSION.major != current_version.major
            @warn """
                SuiteSparse version incompatibility

                Julia was compiled with SuiteSparse version $BUILD_VERSION. It is
                currently linked with version $current_version.
                This might cause Julia to terminate when working with
                sparse matrix factorizations, e.g. solving systems of
                equations with \\.

                It is recommended that you use Julia with the same major
                version of SuiteSparse as the one used during the build, or
                download the generic binaries from www.julialang.org,
                which ship with the correct versions of all dependencies.
                """
        end

        current_version >= v"6.0.0" && SuiteSparse_start()

        # Register gc tracked allocator if SuiteSparse is new enough
        if current_version >= v"7.0.0"
            SuiteSparse_config_malloc_func_set(cglobal(:jl_malloc, Ptr{Cvoid}))
            SuiteSparse_config_calloc_func_set(cglobal(:jl_calloc, Ptr{Cvoid}))
            SuiteSparse_config_realloc_func_set(cglobal(:jl_realloc, Ptr{Cvoid}))
            SuiteSparse_config_free_func_set(cglobal(:jl_free, Ptr{Cvoid}))
        elseif current_version >= v"4.2.0"
            cnfg = cglobal((:SuiteSparse_config, libsuitesparseconfig), Ptr{Cvoid})
            unsafe_store!(cnfg, cglobal(:jl_malloc, Ptr{Cvoid}), 1)
            unsafe_store!(cnfg, cglobal(:jl_calloc, Ptr{Cvoid}), 2)
            unsafe_store!(cnfg, cglobal(:jl_realloc, Ptr{Cvoid}), 3)
            unsafe_store!(cnfg, cglobal(:jl_free, Ptr{Cvoid}), 4)
        end

        current_version >= v"6.0.0" && atexit() do
            SuiteSparse_finish()
        end

    catch ex
        @error "Error during initialization of module LibSuiteSparse" exception=ex,catch_backtrace()
    end
end

# exports
const PREFIXES = ["cholmod_", "CHOLMOD_", "umfpack_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
