module LibSuiteSparse

import SuiteSparse_jll
import Libdl
using Libdl: LazyLibrary

const TRUE  = Int32(1)
const FALSE = Int32(0)

## Library loading
#
# SparseArrays owns the `LazyLibrary` handles for the SuiteSparse libraries it calls
# rather than using SuiteSparse_jll's, so the whole set can be redirected to another
# directory (#250). Without an override each library resolves to the same file
# SuiteSparse_jll would open. Paths are resolved at first `dlopen`, so the override
# only has to be in place before the first solver call.

const SPARSEARRAYS_UUID = Base.UUID("2f01184e-e22b-5df5-ae63-d93ebab69eaf")
const LIBDIR_ENV = "JULIA_SUITESPARSE_LIBDIR"
const LIBDIR_PREFERENCE = "suitesparse_libdir"

# nothing = not resolved yet, "" = no override
const _libdir = Ref{Union{Nothing,String}}(nothing)
const _libdir_lock = ReentrantLock()

# Path SuiteSparse_jll would load for library `name`, without loading it.
function _jll_path(name::Symbol)
    lib = getfield(SuiteSparse_jll, name)
    lib isa LazyLibrary && return string(lib.path)
    # Older stubs dlopen in `__init__` and record the resolved path in `<name>_path`.
    path = getfield(SuiteSparse_jll, Symbol(name, :_path))::String
    return isempty(path) ? String(lib) : path
end

function _override_dir()
    # Nothing loaded while generating a pkgimage may be cached in it.
    Base.generating_output() && return ""
    @lock _libdir_lock begin
        dir = _libdir[]
        if dir === nothing
            dir = get(ENV, LIBDIR_ENV, "")
            if isempty(dir)
                pref = get(Base.get_preferences(SPARSEARRAYS_UUID), LIBDIR_PREFERENCE, nothing)
                dir = pref isa AbstractString ? String(pref) : ""
            end
            isempty(dir) || (dir = abspath(expanduser(dir)))
            _libdir[] = dir
        end
        return dir::String
    end
end

struct SuiteSparseLibPath
    name::Symbol
end
function Base.string(p::SuiteSparseLibPath)
    path = _jll_path(p.name)
    dir = _override_dir()
    return isempty(dir) ? path : joinpath(dir, basename(path))
end
Base.print(io::IO, p::SuiteSparseLibPath) = print(io, string(p))

# The BLAS/LAPACK and compiler runtime libraries SuiteSparse_jll declares as dependencies.
# Loading libblastrampoline through its LazyLibrary (rather than letting the OS loader pull
# it in as a DT_NEEDED of libcholmod) is what runs LinearAlgebra's BLAS forwarding callback.
const _system_deps = LazyLibrary[getfield(SuiteSparse_jll, n)
    for n in (:libblastrampoline, :libstdcxx, :libgcc_s)
    if isdefined(SuiteSparse_jll, n) && getfield(SuiteSparse_jll, n) isa LazyLibrary]

const libsuitesparseconfig = LazyLibrary(SuiteSparseLibPath(:libsuitesparseconfig))
const libamd     = LazyLibrary(SuiteSparseLibPath(:libamd);     dependencies = [libsuitesparseconfig])
const libcamd    = LazyLibrary(SuiteSparseLibPath(:libcamd);    dependencies = [libsuitesparseconfig])
const libcolamd  = LazyLibrary(SuiteSparseLibPath(:libcolamd);  dependencies = [libsuitesparseconfig])
const libccolamd = LazyLibrary(SuiteSparseLibPath(:libccolamd); dependencies = [libsuitesparseconfig])
const libcholmod = LazyLibrary(SuiteSparseLibPath(:libcholmod);
    dependencies = [libsuitesparseconfig, libamd, libcamd, libccolamd, libcolamd, _system_deps...])
const libspqr    = LazyLibrary(SuiteSparseLibPath(:libspqr);
    dependencies = [libsuitesparseconfig, libcholmod, _system_deps...])
const libumfpack = LazyLibrary(SuiteSparseLibPath(:libumfpack);
    dependencies = [libsuitesparseconfig, libamd, libcholmod, _system_deps...])

const SUITESPARSE_LIBRARIES = (; libsuitesparseconfig, libamd, libcamd, libcolamd, libccolamd,
                                 libcholmod, libspqr, libumfpack)

_isloaded(lib::LazyLibrary) = (@atomic :acquire lib.handle) != C_NULL

"""
    LibSuiteSparse.libdir()

Return the directory the SuiteSparse libraries are loaded from, or will be loaded from
if none has been loaded yet. See [`set_libdir!`](@ref LibSuiteSparse.set_libdir!).
"""
function libdir()
    dir = _override_dir()
    return isempty(dir) ? dirname(_jll_path(:libsuitesparseconfig)) : dir
end

"""
    LibSuiteSparse.set_libdir!(dir)
    LibSuiteSparse.set_libdir!(nothing)

Load the SuiteSparse libraries from `dir` instead of the copies bundled with Julia, or
restore the bundled copies with `nothing`. `dir` must contain the whole set of libraries
(`libsuitesparseconfig`, `libamd`, `libcamd`, `libcolamd`, `libccolamd`, `libcholmod`,
`libspqr` and `libumfpack`) under the same file names as the bundled ones, built from
the same major SuiteSparse version.

The libraries are loaded on first use, so this must be called before the first solver
call. It throws once any of them has been loaded. The directory can also be set with the
`$LIBDIR_ENV` environment variable or the `$LIBDIR_PREFERENCE` preference of SparseArrays;
`set_libdir!` takes precedence over both.
"""
function set_libdir!(dir::Union{AbstractString,Nothing})
    @lock _libdir_lock begin
        loaded = [String(k) for (k, lib) in pairs(SUITESPARSE_LIBRARIES) if _isloaded(lib)]
        isempty(loaded) || throw(ArgumentError(
            "the SuiteSparse library directory cannot be changed after loading $(join(loaded, ", "))"))
        if dir === nothing
            _libdir[] = ""
        else
            isdir(dir) || throw(ArgumentError("not a directory: $dir"))
            _libdir[] = abspath(String(dir))
        end
    end
    return libdir()
end

include("wrappers.jl")

const SUITESPARSE_MIN_VERSION = v"6.0.0"
const BUILD_VERSION = VersionNumber(
    SUITESPARSE_MAIN_VERSION,
    SUITESPARSE_SUB_VERSION,
    SUITESPARSE_SUBSUB_VERSION
)

public init_suitesparse, libdir, set_libdir!

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
        if Libdl.dlsym_e(Libdl.dlopen(libsuitesparseconfig), :SuiteSparse_version) != C_NULL
            current_version_array = Vector{Cint}(undef, 3)
            SuiteSparse_version(current_version_array)
            (major, minor, patch) = current_version_array
            current_version = VersionNumber(major, minor, patch)
        else # SuiteSparse < 4.2.0 does not include SuiteSparse_version()
            current_version = v"0.0.0"
        end


        if current_version < SUITESPARSE_MIN_VERSION
            @warn """
                SuiteSparse version incompatibility

                Julia was compiled with SuiteSparse version $BUILD_VERSION. It is
                currently linked with a version older than
                $(SUITESPARSE_MIN_VERSION) from $(Libdl.dlpath(libsuitesparseconfig)).
                This might cause Julia to
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
                currently linked with version $current_version from
                $(Libdl.dlpath(libsuitesparseconfig)).
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
    return nothing
end

# exports
const PREFIXES = ["cholmod_", "CHOLMOD_", "umfpack_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
