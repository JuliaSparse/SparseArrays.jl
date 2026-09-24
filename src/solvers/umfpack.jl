# This file is a part of Julia. License is MIT: https://julialang.org/license

module UMFPACK

export UmfpackLU

public rcond

import Base: (\), getproperty, show, size
using LinearAlgebra
using LinearAlgebra: AdjOrTrans, HermOrSym
import LinearAlgebra: Factorization, AdjointFactorization, TransposeFactorization,
    checksquare, det, logabsdet, lu, lu!, ldiv!

using SparseArrays
using SparseArrays: getcolptr, AbstractSparseMatrixCSC
import SparseArrays: nnz

import Serialization: AbstractSerializer, deserialize, serialize
using Serialization

import ..increment, ..increment!, ..decrement, ..decrement!

using ..LibSuiteSparse
import ..LibSuiteSparse:
    umfpack_dl_defaults,
    umfpack_dl_report_control,
    umfpack_dl_report_info,
    ## Type of solve
    UMFPACK_A,        # Ax=b
    UMFPACK_At,       # adjoint(A)x=b
    UMFPACK_Aat,      # transpose(A)x=b
    UMFPACK_Pt_L,     # adjoint(P)Lx=b
    UMFPACK_L,        # Lx=b
    UMFPACK_Lt_P,     # adjoint(L)Px=b
    UMFPACK_Lat_P,    # transpose(L)Px=b
    UMFPACK_Lt,       # adjoint(L)x=b
    UMFPACK_Lat,      # transpose(L)x=b
    UMFPACK_U_Qt,     # U*adjoint(Q)x=b
    UMFPACK_U,        # Ux=b
    UMFPACK_Q_Ut,     # Q*adjoint(U)x=b
    UMFPACK_Q_Uat,    # Q*transpose(U)x=b
    UMFPACK_Ut,       # adjoint(U)x=b
    UMFPACK_Uat,      # transpose(U)x=b
    ## Sizes of Control and Info arrays for returning information from solver
    UMFPACK_INFO,
    UMFPACK_CONTROL,
    # index of the info array in ZERO BASED indexing
    UMFPACK_RCOND,
    # index of the control arrays in ZERO BASED indexing
    UMFPACK_PRL,
    UMFPACK_DENSE_ROW,
    UMFPACK_DENSE_COL,
    UMFPACK_PIVOT_TOLERANCE,
    UMFPACK_BLOCK_SIZE,
    UMFPACK_ORDERING,
    UMFPACK_FIXQ,
    UMFPACK_AMD_DENSE,
    UMFPACK_AGGRESSIVE,
    UMFPACK_SINGLETONS,
    UMFPACK_ALLOC_INIT,
    UMFPACK_SYM_PIVOT_TOLERANCE,
    UMFPACK_SCALE,
    UMFPACK_FRONT_ALLOC_INIT,
    UMFPACK_DROPTOL,
    UMFPACK_IRSTEP,
    ## Status codes
    UMFPACK_OK,
    UMFPACK_WARNING_singular_matrix,
    UMFPACK_WARNING_determinant_underflow,
    UMFPACK_WARNING_determinant_overflow,
    UMFPACK_ERROR_out_of_memory,
    UMFPACK_ERROR_invalid_Numeric_object,
    UMFPACK_ERROR_invalid_Symbolic_object,
    UMFPACK_ERROR_argument_missing,
    UMFPACK_ERROR_n_nonpositive,
    UMFPACK_ERROR_invalid_matrix,
    UMFPACK_ERROR_different_pattern,
    UMFPACK_ERROR_invalid_system,
    UMFPACK_ERROR_invalid_permutation,
    UMFPACK_ERROR_internal_error,
    UMFPACK_ERROR_file_IO,
    UMFPACK_ERROR_ordering_failed

# Julia uses one based indexing so here we are
const JL_UMFPACK_PRL = UMFPACK_PRL + 1
const JL_UMFPACK_DENSE_ROW = UMFPACK_DENSE_ROW + 1
const JL_UMFPACK_DENSE_COL = UMFPACK_DENSE_COL + 1
const JL_UMFPACK_PIVOT_TOLERANCE = UMFPACK_PIVOT_TOLERANCE + 1
const JL_UMFPACK_BLOCK_SIZE = UMFPACK_BLOCK_SIZE + 1
const JL_UMFPACK_ORDERING = UMFPACK_ORDERING + 1
const JL_UMFPACK_FIXQ = UMFPACK_FIXQ + 1
const JL_UMFPACK_AMD_DENSE = UMFPACK_AMD_DENSE + 1
const JL_UMFPACK_AGGRESSIVE = UMFPACK_AGGRESSIVE + 1
const JL_UMFPACK_SINGLETONS = UMFPACK_SINGLETONS + 1
const JL_UMFPACK_ALLOC_INIT = UMFPACK_ALLOC_INIT + 1
const JL_UMFPACK_SYM_PIVOT_TOLERANCE = UMFPACK_SYM_PIVOT_TOLERANCE + 1
const JL_UMFPACK_SCALE = UMFPACK_SCALE + 1
const JL_UMFPACK_FRONT_ALLOC_INIT = UMFPACK_FRONT_ALLOC_INIT + 1
const JL_UMFPACK_DROPTOL = UMFPACK_DROPTOL + 1
const JL_UMFPACK_IRSTEP = UMFPACK_IRSTEP + 1
const JL_UMFPACK_RCOND = UMFPACK_RCOND + 1

struct MatrixIllConditionedException <: Exception
    msg::String
end

function umferror(status::Integer)
    if status==UMFPACK_OK
        return
    elseif status==UMFPACK_WARNING_singular_matrix
        throw(LinearAlgebra.SingularException(0))
    elseif status==UMFPACK_WARNING_determinant_underflow
        throw(MatrixIllConditionedException("the determinant is nonzero but underflowed"))
    elseif status==UMFPACK_WARNING_determinant_overflow
        throw(MatrixIllConditionedException("the determinant overflowed"))
    elseif status==UMFPACK_ERROR_out_of_memory
        throw(OutOfMemoryError())
    elseif status==UMFPACK_ERROR_invalid_Numeric_object
        throw(ArgumentError("invalid UMFPack numeric object"))
    elseif status==UMFPACK_ERROR_invalid_Symbolic_object
        throw(ArgumentError("invalid UMFPack symbolic object"))
    elseif status==UMFPACK_ERROR_argument_missing
        throw(ArgumentError("a required argument to UMFPack is missing"))
    elseif status==UMFPACK_ERROR_n_nonpositive
        throw(ArgumentError("the number of rows or columns of the matrix must be greater than zero"))
    elseif status==UMFPACK_ERROR_invalid_matrix
        throw(ArgumentError("invalid matrix"))
    elseif status==UMFPACK_ERROR_different_pattern
        throw(ArgumentError("pattern of the matrix changed"))
    elseif status==UMFPACK_ERROR_invalid_system
        throw(ArgumentError("invalid sys argument provided to UMFPack solver"))
    elseif status==UMFPACK_ERROR_invalid_permutation
        throw(ArgumentError("invalid permutation"))
    elseif status==UMFPACK_ERROR_file_IO
        throw(ErrorException("error saving / loading UMFPack decomposition"))
    elseif status==UMFPACK_ERROR_ordering_failed
        throw(ErrorException("the ordering method failed"))
    elseif status==UMFPACK_ERROR_internal_error
        throw(ErrorException("an internal error has occurred, of unknown cause"))
    else
        throw(ErrorException("unknown UMFPack error code: $status"))
    end
end

macro isok(A)
    :(umferror($(esc(A))))
end

if Sys.WORD_SIZE == 64
    const UmfpackIndexTypes = (:Int32, :Int64)
    const UMFITypes = Union{Int32, Int64}
else
    const UmfpackIndexTypes = (:Int32,)
    const UMFITypes = Int32
end

const UMFVTypes = Union{Float64,ComplexF64}

## UMFPACK

function show_umf_ctrl(control::Vector{Float64}, level::Real = 2.0)
    old_prt::Float64 = control[1]
    control[1] = Float64(level)
    umfpack_dl_report_control(control)
    control[1] = old_prt
end

function show_umf_info(control::Vector{Float64}, info::Vector{Float64}, level::Real = 2.0)
    old_prt::Float64 = control[1]
    control[1] = Float64(level)
    umfpack_dl_report_info(control, info)
    control[1] = old_prt
end

mutable struct Numeric{Tv,Ti}
    p::Ptr{Cvoid}
    function Numeric{Tv, Ti}(p) where {Tv<:UMFVTypes, Ti<:UMFITypes}
        return finalizer(new{Tv, Ti}(p)) do num
            umfpack_free_numeric(num, Tv, Ti)
            num.p = C_NULL
        end
    end
end
Base.unsafe_convert(::Type{Ptr{Cvoid}}, num::Numeric) = num.p

mutable struct Symbolic{Tv, Ti}
    p::Ptr{Cvoid}
    function Symbolic{Tv, Ti}(p) where {Tv<:UMFVTypes, Ti<:UMFITypes}
        return finalizer(new{Tv, Ti}(p)) do sym
            umfpack_free_symbolic(sym, Tv, Ti)
            sym.p = C_NULL
        end
    end
end
Base.unsafe_convert(::Type{Ptr{Cvoid}}, num::Symbolic) = num.p

_isnull(x::Union{Symbolic, Numeric}) = x.p == C_NULL
_isnotnull(x::Union{Symbolic, Numeric}) = x.p != C_NULL
"""
    UMFPACK.UmfpackWS(F::UmfpackLU)

Scratch space for `ldiv!(x, F, b; workspace)`, which makes repeated solves allocation-free.
Without it, `ldiv!` allocates its scratch space on each call. A workspace grows as needed,
so it can be reused with any factorization, but not by two calls at once.
"""
struct UmfpackWS{T<:UMFITypes}
    Wi::Vector{T}
    W::Vector{Float64}
end

UmfpackWS{T}(Wisize::Integer, Wsize::Integer) where {T<:UMFITypes} =
    UmfpackWS{T}(Vector{T}(undef, Wisize), Vector{Float64}(undef, Wsize))

UmfpackWS(S::AbstractSparseMatrixCSC{Tv,Ti}, refinement::Bool) where {Tv,Ti} = UmfpackWS{Ti}(
    Vector{Ti}(undef, size(S, 2)),
    Vector{Float64}(undef, workspace_W_size(S, refinement)))

function Base.resize!(W::UmfpackWS, S, refinement::Bool; expand_only=false)
    (!expand_only || length(W.Wi) < size(S, 2)) && resize!(W.Wi, size(S, 2))
    ws = workspace_W_size(S, refinement)
    (!expand_only || length(W.W) < ws) && resize!(W.W, ws)
    return W
end

Base.similar(w::UmfpackWS) = UmfpackWS(similar(w.Wi), similar(w.W))

## Should this type be immutable?
"""
    UMFPACK.UmfpackLU{Tv,Ti} <: Factorization{Tv}

The LU factorization of a sparse matrix computed by UMFPACK, returned by
[`lu`](@ref SparseArrays.UMFPACK.lu). `Tv` is `Float64` or `ComplexF64`, and `Ti` is
`Int32` or `Int64` (only `Int32` on 32-bit systems). `F` holds a zero-based copy of the
factorized matrix together with UMFPACK's opaque symbolic and numeric objects, which are
released by finalizers.

The factors are copied out of UMFPACK on each property access:

| Property | Description                                   |
|:---------|:----------------------------------------------|
| `F.L`    | unit lower triangular `SparseMatrixCSC`       |
| `F.U`    | upper triangular `SparseMatrixCSC`            |
| `F.p`    | row permutation `Vector`                      |
| `F.q`    | column permutation `Vector`                   |
| `F.Rs`   | `Vector` of row scaling factors               |
| `F.:(:)` | the tuple `(L, U, p, q, Rs)`, extracted in one call |

They satisfy `F.L * F.U == (F.Rs .* A)[F.p, F.q]`.

`F` supports `\\`, `ldiv!`, [`det`](@ref), `logabsdet`, [`issuccess`](@ref), `nnz`,
`adjoint`, `transpose`, [`UMFPACK.rcond`](@ref SparseArrays.UMFPACK.rcond) and
refactorization with `lu!`. A serialized `UmfpackLU` carries the matrix rather than the
factors, which are recomputed when it is deserialized.

`ldiv!` takes an optional [`UMFPACK.UmfpackWS`](@ref SparseArrays.UMFPACK.UmfpackWS) to
avoid allocating. Solves do not modify `F`, so several tasks can solve with one `F` at
once; see [Multithreading and thread safety](@ref). The exception is an `F` without
factors, such as one built with `UmfpackLU(S)` or left by a failed `lu!`: its first use
computes them.

# Examples
```jldoctest
julia> A = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0]);

julia> F = lu(A);

julia> F.L * F.U ≈ (F.Rs .* A)[F.p, F.q]
true

julia> L, U, p, q, Rs = F.:(:);

julia> L * U ≈ (Rs .* A)[p, q]
true
```
"""
mutable struct UmfpackLU{Tv<:UMFVTypes,Ti<:UMFITypes} <: Factorization{Tv}
    symbolic::Symbolic{Tv, Ti}
    numeric::Numeric{Tv, Ti}
    m::Int
    n::Int
    colptr::Vector{Ti}                  # 0-based column pointers
    rowval::Vector{Ti}                  # 0-based row indices
    nzval::Vector{Tv}
    status::Int
    control::Vector{Float64}
    info::Vector{Float64}
end

function UmfpackLU(S::AbstractSparseMatrixCSC{Tv, Ti};
    control=get_umfpack_control(Tv, Ti)) where
    {Tv<:UMFVTypes,Ti<:UMFITypes}

    zerobased = getcolptr(S)[1] == 0
    return UmfpackLU(Symbolic{Tv, Ti}(C_NULL), Numeric{Tv, Ti}(C_NULL),
                    size(S, 1), size(S, 2),
                    zerobased ? copy(getcolptr(S)) : decrement(getcolptr(S)),
                    zerobased ? copy(rowvals(S)) : decrement(rowvals(S)),
                    copy(nonzeros(S)), 0,
                    copy(control), Vector{Float64}(undef, UMFPACK_INFO)
    )
end

workspace_W_size(F::UmfpackLU) = workspace_W_size(F, has_refinement(F))
workspace_W_size(S::Union{UmfpackLU{<:AbstractFloat}, AbstractSparseMatrixCSC{<:AbstractFloat}}, refinement::Bool) = refinement ? 5 * size(S, 2) : size(S, 2)
workspace_W_size(S::Union{UmfpackLU{<:Complex}, AbstractSparseMatrixCSC{<:Complex}}, refinement::Bool) = refinement ? 10 * size(S, 2) : 4 * size(S, 2)

const UMFAdjOrTransLU = Union{TransposeFactorization{<:Any, <:UmfpackLU}, AdjointFactorization{<:Any, <:UmfpackLU}}
has_refinement(F::UMFAdjOrTransLU) = has_refinement(parent(F))
has_refinement(F::UmfpackLU) = has_refinement(F.control)
has_refinement(control::AbstractVector) = control[JL_UMFPACK_IRSTEP] > 0

UmfpackWS(F::UmfpackLU{Tv, Ti}, refinement::Bool=has_refinement(F)) where {Tv, Ti} = UmfpackWS(
        Vector{Ti}(undef, size(F, 2)),
        Vector{Float64}(undef, workspace_W_size(F, refinement)))
UmfpackWS(F::UMFAdjOrTransLU, refinement::Bool=has_refinement(F)) = UmfpackWS(parent(F), refinement)

"""
    copy(F::UmfpackLU)::UmfpackLU

Return an independent copy of `F`, with its own matrix, symbolic and numeric factors,
`control` and `info`; refactorizing either one with [`lu!`](@ref) does not affect
the other. `deepcopy(F)` does the same.
"""
Base.copy(F::UmfpackLU) = UmfpackLU(
        umfpack_copy_symbolic(F.symbolic),
        umfpack_copy_numeric(F.numeric),
        F.m, F.n,
        copy(F.colptr),
        copy(F.rowval),
        copy(F.nzval),
        F.status,
        copy(F.control),
        copy(F.info)
    )
# The workspace argument is accepted for compatibility; solves take their workspace from
# `ldiv!`.
Base.copy(F::UmfpackLU, ::UmfpackWS) = copy(F)
Base.copy(F::T) where {T <: UMFAdjOrTransLU} = T(copy(parent(F)))
Base.copy(F::T, ::UmfpackWS) where {T <: UMFAdjOrTransLU} = T(copy(parent(F)))

# The default deepcopy would duplicate the raw symbolic and numeric pointers into
# wrappers without finalizers, leaving them dangling once the original is freed.
function Base.deepcopy_internal(F::UmfpackLU, stackdict::IdDict)
    haskey(stackdict, F) && return stackdict[F]::typeof(F)
    G = copy(F)
    stackdict[F] = G
    return G
end

Base.transpose(F::UmfpackLU) = TransposeFactorization(F)

show_umf_ctrl(F::UmfpackLU, level::Real=2.0) = show_umf_ctrl(F.control, level)


show_umf_info(F::UmfpackLU, level::Real=2.0) = show_umf_info(F.control, F.info, level)


"""
    lu(A::AbstractSparseMatrixCSC; check = true, q = nothing, control = get_umfpack_control()) -> F::UmfpackLU

Compute the LU factorization of a sparse matrix `A`.

For sparse `A` with real or complex element type, the return type of `F` is
`UmfpackLU{Tv, Ti}`, with `Tv` = [`Float64`](@ref) or `ComplexF64` respectively and
`Ti` is an integer type ([`Int32`](@ref) or [`Int64`](@ref)).

When `check = true`, an error is thrown if the decomposition fails.
When `check = false`, responsibility for checking the decomposition's
validity (via [`issuccess`](@ref)) lies with the user.

The column permutation `q` can either be a permutation vector or `nothing`. If no permutation vector
is provided or `q` is `nothing`, UMFPACK's default is used. If the permutation is not zero-based, a
zero-based copy is made.

The `control` vector defaults to the Julia SparseArrays package's default configuration for UMFPACK (NB: this is modified from the UMFPACK defaults to
disable iterative refinement), but can be changed by passing a vector of length `UMFPACK_CONTROL`, see the UMFPACK manual for possible configurations.
For example to reenable iterative refinement:

    umfpack_control = SparseArrays.UMFPACK.get_umfpack_control(Float64, Int64) # read Julia default configuration for a Float64 sparse matrix
    SparseArrays.UMFPACK.show_umf_ctrl(umfpack_control) # optional - display values
    umfpack_control[SparseArrays.UMFPACK.JL_UMFPACK_IRSTEP] = 2.0 # reenable iterative refinement (2 is UMFPACK default max iterative refinement steps)

    Alu = lu(A; control = umfpack_control)
    x = Alu \\ b   # solve Ax = b, including UMFPACK iterative refinement

The individual components of the factorization `F` can be accessed by indexing:

| Component | Description                         |
|:----------|:------------------------------------|
| `L`       | `L` (lower triangular) part of `LU` |
| `U`       | `U` (upper triangular) part of `LU` |
| `p`       | row permutation `Vector`            |
| `q`       | column permutation `Vector`         |
| `Rs`      | `Vector` of scaling factors         |
| `:`       | `(L,U,p,q,Rs)` components           |

The relation between `F` and `A` is

`F.L*F.U == (F.Rs .* A)[F.p, F.q]`

`F` further supports the following functions:

- [`\\`](@ref)
- [`det`](@ref)

See also [`lu!`](@ref)

!!! note
    `lu(A::AbstractSparseMatrixCSC)` uses the UMFPACK[^ACM832] library that is part of
    [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse).
    As this library only supports sparse matrices with [`Float64`](@ref) or
    `ComplexF64` elements, `lu` converts `A` into a copy that is of type
    `SparseMatrixCSC{Float64}` or `SparseMatrixCSC{ComplexF64}` as appropriate.

[^ACM832]: Davis, Timothy A. (2004b). Algorithm 832: UMFPACK V4.3---an Unsymmetric-Pattern Multifrontal Method. ACM Trans. Math. Softw., 30(2), 196–199. [doi:10.1145/992200.992206](https://doi.org/10.1145/992200.992206)
"""
function lu(S::AbstractSparseMatrixCSC{Tv, Ti};
    check::Bool = true, q=nothing, control=get_umfpack_control(Tv, Ti)) where
    {Tv<:UMFVTypes,Ti<:UMFITypes}
    res = UmfpackLU(S; control)
    umfpack_numeric!(res; q)
    check && (issuccess(res) || throw(LinearAlgebra.SingularException(0)))
    return res
end
lu(A::AbstractSparseMatrixCSC{<:Union{Float16,Float32},Ti}; kws...) where {Ti<:UMFITypes} =
    lu(convert(SparseMatrixCSC{Float64,Ti}, A); kws...)
lu(A::AbstractSparseMatrixCSC{<:Union{ComplexF16,ComplexF32},Ti}; kws...) where {Ti<:UMFITypes} =
    lu(convert(SparseMatrixCSC{ComplexF64,Ti}, A); kws...)
lu(A::Union{AbstractSparseMatrixCSC{T},AbstractSparseMatrixCSC{Complex{T}}};
   kws...) where {T<:AbstractFloat} =
    throw(ArgumentError(string("matrix type ", typeof(A), " not supported. ",
    "Try lu(convert(SparseMatrixCSC{Float64/ComplexF64,Int}, A)) for ",
    "sparse floating point LU using UMFPACK or lu(Array(A)) for generic ",
    "dense LU.")))
lu(A::AbstractSparseMatrixCSC; kws...) = lu(float(A); kws...)

# We could do this as lu(A') = lu(A)' with UMFPACK, but the user could want to do one over the other
lu(A::AdjOrTrans{T,S}; kws...) where {T<:UMFVTypes, S<:AbstractSparseMatrixCSC{T}} =
    lu(copy(A); kws...)
lu(A::HermOrSym{<:Any,<:Union{AbstractSparseMatrixCSC,SubArray{<:Any,2,<:AbstractSparseMatrixCSC}}}; kws...) = lu(sparse(A); kws...)

LinearAlgebra._lu(A::AbstractSparseMatrixCSC; kwargs...) =
    lu(A; kwargs...)
LinearAlgebra._lu(::AbstractSparseMatrixCSC, ::LinearAlgebra.PivotingStrategy; kwargs...) =
    error("Pivoting Strategies are not supported by `SparseMatrixCSC`s")

"""
    lu!(F::UmfpackLU, A::AbstractSparseMatrixCSC; check=true, reuse_symbolic=true, q=nothing) -> F::UmfpackLU

Compute the LU factorization of a sparse matrix `A`, reusing the symbolic
factorization of an already existing LU factorization stored in `F`.
Unless `reuse_symbolic` is set to false, the sparse matrix `A` must have an
identical nonzero pattern as the matrix used to create the LU factorization `F`,
otherwise an error is thrown. If the size of `A` and `F` differ, all vectors will
be resized accordingly.

When `check = true`, an error is thrown if the decomposition fails.
When `check = false`, responsibility for checking the decomposition's
validity (via [`issuccess`](@ref)) lies with the user.

The column permutation `q` can either be a permutation vector or `nothing`. If no permutation vector
is provided or `q` is `nothing`, UMFPACK's default is used. If the permutation is not zero based, a
zero based copy is made.

See also [`lu`](@ref)

!!! note
    `lu!(F::UmfpackLU, A::AbstractSparseMatrixCSC)` uses the UMFPACK library that is part of
    SuiteSparse. As this library only supports sparse matrices with [`Float64`](@ref) or
    `ComplexF64` elements, `lu!` will automatically convert the types to those set by the LU
    factorization or `SparseMatrixCSC{ComplexF64}` as appropriate.

!!! compat "Julia 1.5"
    `lu!` for `UmfpackLU` requires at least Julia 1.5.

# Examples
```jldoctest
julia> A = sparse(Float64[1.0 2.0; 0.0 3.0]);

julia> F = lu(A);

julia> B = sparse(Float64[1.0 1.0; 0.0 1.0]);

julia> lu!(F, B);

julia> F \\ ones(2)
2-element Vector{Float64}:
 0.0
 1.0
```
"""
function lu!(F::UmfpackLU{Tv, Ti}, S::AbstractSparseMatrixCSC;
  check::Bool=true, reuse_symbolic::Bool=true, q=nothing) where {Tv, Ti}
    zerobased = getcolptr(S)[1] == 0
    if max(size(S)..., length(nonzeros(S))) >= typemax(Ti)
        throw(ArgumentError("matrix of size $(size(S)) with $(length(nonzeros(S))) stored entries does not fit the $Ti indices of $(typeof(F)); use lu(S) instead"))
    end
    if Tv <: Real && !(eltype(S) <: Real)
        throw(ArgumentError("cannot refactorize the real $(typeof(F)) with a matrix of eltype $(eltype(S)); use lu(S) instead"))
    end

    F.m = size(S, 1)
    F.n = size(S, 2)

    resize!(F.colptr, length(getcolptr(S)))
    if zerobased
        F.colptr .= getcolptr(S)
    else
        F.colptr .= getcolptr(S) .- one(Ti)
    end

    resize!(F.rowval, length(rowvals(S)))
    if zerobased
        F.rowval .= rowvals(S)
    else
        F.rowval .= rowvals(S) .- one(Ti)
    end

    resize!(F.nzval, length(nonzeros(S)))
    F.nzval .= nonzeros(S)

    return lu!(F; reuse_symbolic, check, q)
end

function lu!(F::UmfpackLU{Tv, Ti}; check::Bool=true, reuse_symbolic::Bool=true,
  q=nothing) where {Tv, Ti}
    if !reuse_symbolic && _isnotnull(F.symbolic)
        umfpack_free_symbolic(F.symbolic, Tv, Ti)
        F.symbolic = Symbolic{Tv, Ti}(C_NULL)
    end
    umfpack_numeric!(F; reuse_numeric = false, q)
    check && (issuccess(F) || throw(LinearAlgebra.SingularException(0)))
    return F
end

size(F::UmfpackLU) = (F.m, F.n)
function size(F::UmfpackLU, dim::Integer)
    if dim < 1
        throw(ArgumentError("size: dimension $dim out of range"))
    elseif dim == 1
        return Int(F.m)
    elseif dim == 2
        return Int(F.n)
    else
        return 1
    end
end

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::UmfpackLU)
    if !issuccess(F)
        print(io, "Failed factorization of type $(typeof(F))")
    elseif _isnull(F.numeric)
        summary(io, F)
        print(io, "\nfactors not computed yet; they are computed on first use")
    else
        summary(io, F); println(io)
        println(io, "L factor:")
        show(io, mime, F.L)
        println(io, "\nU factor:")
        show(io, mime, F.U)
    end
end

function serialize(s::AbstractSerializer, L::UmfpackLU{Tv, Ti}) where {Tv, Ti}
    # TODO: If we can get a C FILE handle we can serialize umfpack_numeric and
    # umfpack_symbolic. using the save_{numeric | symbolic} functions.
    Serialization.serialize_type(s, typeof(L))
    serialize(s, L.m)
    serialize(s, L.n)
    serialize(s, L.colptr)
    serialize(s, L.rowval)
    serialize(s, L.nzval)
    serialize(s, L.control)
    serialize(s, L.info)
    serialize(s, L.status)
end
function deserialize(s::AbstractSerializer, ::Type{UmfpackLU{Tv,Ti}}) where {Tv,Ti}
    # TODO: If we can get a C FILE handle we can deserialize umfpack_numeric and
    # umfpack_symbolic. using the load_{numeric | symbolic} functions.
    m        = deserialize(s)
    n        = deserialize(s)
    colptr   = deserialize(s)
    rowval   = deserialize(s)
    nzval    = deserialize(s)
    control  = deserialize(s)
    info     = deserialize(s)
    status   = deserialize(s)
    F = UmfpackLU{Tv,Ti}(Symbolic{Tv, Ti}(C_NULL), Numeric{Tv, Ti}(C_NULL),
        m, n, colptr, rowval, nzval, status, control, info)
    # Factorize now, so that solves with the deserialized object do not write it.
    if status == UMFPACK_OK || status == UMFPACK_WARNING_singular_matrix
        umfpack_numeric!(F)
    end
    return F
end

function _zerobased_perm(::Type{Ti}, q::AbstractVector{<:Integer}, n::Integer) where {Ti}
    length(q) == n ||
        throw(DimensionMismatch("permutation q has length $(length(q)), but the matrix has $n columns"))
    return !isempty(q) && minimum(q) == 1 ? Vector{Ti}(q) .- one(Ti) : convert(Vector{Ti}, q)
end

# compute the sign/parity of a permutation
function _signperm(p)
    n = length(p)
    result = 0
    todo = trues(n)
    while any(todo)
        k = findfirst(todo)
        todo[k] = false
        result += 1 # increment element count
        j = p[k]
        while j != k
            result += 1 # increment element count
            todo[j] = false
            j = p[j]
        end
        result += 1 # increment cycle count
    end
    return ifelse(isodd(result), -1, 1)
end

# UMFPACK reports whether its scale factors multiply (`do_recip`) or divide the rows of A;
# `F.Rs` always multiplies.
_scale_factors!(Rs, do_recip) = do_recip == 0 ? map!(inv, Rs, Rs) : Rs

# UMFPACK needs contiguous vectors; solve through contiguous copies otherwise.
function _unit_stride_solve!(x, lu, b, typ, workspace::UmfpackWS)
    xc = stride(x, 1) == 1 ? x : similar(x, length(x))
    bc = stride(b, 1) == 1 ? b : collect(b)
    solve!(xc, lu, bc, typ; workspace)
    xc === x || copyto!(x, xc)
    return x
end

## Wrappers for UMFPACK functions

# generate the name of the C function according to the value and integer types
umf_nm(nm,Tv,Ti) = "umfpack_" * (Tv === :Float64 ? "d" : "z") * (Ti === :Int64 ? "l_" : "i_") * nm

for itype in UmfpackIndexTypes
    sym_r = Symbol(umf_nm("symbolic", :Float64, itype))
    symq_r = Symbol(umf_nm("qsymbolic", :Float64, itype))
    sym_c = Symbol(umf_nm("symbolic", :ComplexF64, itype))
    symq_c = Symbol(umf_nm("qsymbolic", :ComplexF64, itype))
    num_r = Symbol(umf_nm("numeric", :Float64, itype))
    num_c = Symbol(umf_nm("numeric", :ComplexF64, itype))
    sol_r = Symbol(umf_nm("solve", :Float64, itype))
    sol_c = Symbol(umf_nm("solve", :ComplexF64, itype))
    wsol_r = Symbol(umf_nm("wsolve", :Float64, itype))
    wsol_c = Symbol(umf_nm("wsolve", :ComplexF64, itype))
    det_r = Symbol(umf_nm("get_determinant", :Float64, itype))
    det_z = Symbol(umf_nm("get_determinant", :ComplexF64, itype))
    lunz_r = Symbol(umf_nm("get_lunz", :Float64, itype))
    lunz_z = Symbol(umf_nm("get_lunz", :ComplexF64, itype))
    get_num_r = Symbol(umf_nm("get_numeric", :Float64, itype))
    get_num_z = Symbol(umf_nm("get_numeric", :ComplexF64, itype))
    @eval begin
        function umfpack_symbolic!(U::UmfpackLU{Float64,$itype}, q::Union{Nothing, AbstractVector{<:Integer}})
            _isnotnull(U.symbolic) && return U
            tmp = Ref{Ptr{Cvoid}}(C_NULL)
            if q === nothing
                @isok $sym_r(U.m, U.n, U.colptr, U.rowval, U.nzval, tmp, U.control, U.info)
            else
                qq = _zerobased_perm($itype, q, U.n)
                @isok $symq_r(U.m, U.n, U.colptr, U.rowval, U.nzval, qq, tmp, U.control, U.info)
            end
            U.symbolic = Symbolic{Float64, $itype}(tmp[])

            return U
        end
        function umfpack_symbolic!(U::UmfpackLU{ComplexF64,$itype}, q::Union{Nothing, AbstractVector{<:Integer}})
            _isnotnull(U.symbolic) && return U
            tmp = Ref{Ptr{Cvoid}}(C_NULL)
            if q === nothing
                @isok $sym_c(U.m, U.n, U.colptr, U.rowval, real(U.nzval), imag(U.nzval), tmp,
                             U.control, U.info)
            else
                qq = _zerobased_perm($itype, q, U.n)
                @isok $symq_c(U.m, U.n, U.colptr, U.rowval, real(U.nzval), imag(U.nzval), qq, tmp, U.control, U.info)
            end
            U.symbolic = Symbolic{ComplexF64, $itype}(tmp[])
            return U
        end
        function umfpack_numeric!(U::UmfpackLU{Float64,$itype}; reuse_numeric=true, q=nothing)
            (reuse_numeric && _isnotnull(U.numeric)) && return U
            # Free the previous factorization eagerly (through the shared
            # wrapper, so copies see a null numeric and refactor) and drop
            # it before the symbolic and numeric calls, so that a failure
            # in either does not leave a stale numeric object behind.
            umfpack_free_numeric(U.numeric, Float64, $itype)
            U.numeric = Numeric{Float64, $itype}(C_NULL)
            U.status = UMFPACK_ERROR_invalid_Numeric_object
            _isnull(U.symbolic) && umfpack_symbolic!(U, q)
            tmp = Ref{Ptr{Cvoid}}(C_NULL)
            status = $num_r(U.colptr, U.rowval, U.nzval, U.symbolic, tmp, U.control, U.info)
            U.status = status
            U.numeric = Numeric{Float64, $itype}(tmp[])
            if status != UMFPACK_WARNING_singular_matrix
                umferror(status)
            end
            return U
        end
        function umfpack_numeric!(U::UmfpackLU{ComplexF64,$itype}; reuse_numeric=true, q=nothing)
            (reuse_numeric && _isnotnull(U.numeric)) && return U
            umfpack_free_numeric(U.numeric, ComplexF64, $itype)
            U.numeric = Numeric{ComplexF64, $itype}(C_NULL)
            U.status = UMFPACK_ERROR_invalid_Numeric_object
            _isnull(U.symbolic) && umfpack_symbolic!(U, q)
            tmp = Ref{Ptr{Cvoid}}(C_NULL)
            status = $num_c(U.colptr, U.rowval, real(U.nzval), imag(U.nzval), U.symbolic, tmp,
                U.control, U.info)
            U.status = status
            U.numeric = Numeric{ComplexF64, $itype}(tmp[])
            if status != UMFPACK_WARNING_singular_matrix
                umferror(status)
            end
            return U
        end
        function solve!(x::StridedVector{Float64},
            lu::UmfpackLU{Float64,$itype}, b::StridedVector{Float64},
            typ::Integer; workspace::Union{Nothing,UmfpackWS{$itype}} = nothing)
            if x === b
                throw(ArgumentError("output array must not be aliased with input array"))
            end
            workspace === nothing && (workspace = UmfpackWS(lu))
            if stride(x, 1) != 1 || stride(b, 1) != 1
                return _unit_stride_solve!(x, lu, b, typ, workspace)
            end
            resize!(workspace, lu, has_refinement(lu); expand_only = true)
            umfpack_numeric!(lu)
            (size(b, 1) == lu.m) && (size(b) == size(x)) || throw(DimensionMismatch())

            @isok $wsol_r(typ, lu.colptr, lu.rowval, lu.nzval,
                x, b, lu.numeric, lu.control,
                C_NULL, workspace.Wi, workspace.W)
            return x
        end
        function solve!(x::StridedVector{ComplexF64},
            lu::UmfpackLU{ComplexF64,$itype}, b::StridedVector{ComplexF64},
            typ::Integer; workspace::Union{Nothing,UmfpackWS{$itype}} = nothing)
            if x === b
                throw(ArgumentError("output array must not be aliased with input array"))
            end
            workspace === nothing && (workspace = UmfpackWS(lu))
            if stride(x, 1) != 1 || stride(b, 1) != 1
                return _unit_stride_solve!(x, lu, b, typ, workspace)
            end
            resize!(workspace, lu, has_refinement(lu); expand_only = true)
            umfpack_numeric!(lu)
            (size(b, 1) == lu.m) && (size(b) == size(x)) || throw(DimensionMismatch())
            @isok $wsol_c(typ, lu.colptr, lu.rowval, lu.nzval, C_NULL, x, C_NULL, b,
                C_NULL, lu.numeric, lu.control, C_NULL, workspace.Wi, workspace.W)
            return x
        end
        function det(lu::UmfpackLU{Float64,$itype})
            checksquare(lu)
            mx = Ref{Float64}(zero(Float64))
            umfpack_numeric!(lu)
            @isok $det_r(mx, C_NULL, lu.numeric, C_NULL)
            mx[]
        end

        function det(lu::UmfpackLU{ComplexF64,$itype})
            mx = Ref{Float64}(zero(Float64))
            mz = Ref{Float64}(zero(Float64))
            checksquare(lu)
            umfpack_numeric!(lu)
            @isok $det_z(mx, mz, C_NULL, lu.numeric, C_NULL)
            complex(mx[], mz[])
        end
        function logabsdet(F::UmfpackLU{T, $itype}) where {T<:Union{Float64,ComplexF64}} # return log(abs(det)) and sign(det)
            n = checksquare(F)
            umfpack_numeric!(F)
            issuccess(F) || return log(zero(real(T))), zero(T)
            U = F.U
            Rs = F.Rs
            p = F.p
            q = F.q
            s = _signperm(p)*_signperm(q)*one(real(T))
            P = one(T)
            abs_det = zero(real(T))
            # det(A) = prod(diag(U)) / prod(Rs); Rs > 0, and the logs avoid overflow
            @inbounds for i in 1:n
                u_ii = U[i, i]
                P *= sign(u_ii)
                abs_det += log(abs(u_ii)) - log(Rs[i])
            end
            return abs_det, s * P
        end
        function umf_lunz(lu::UmfpackLU{Float64,$itype})
            lnz = Ref{$itype}(zero($itype))
            unz = Ref{$itype}(zero($itype))
            n_row = Ref{$itype}(zero($itype))
            n_col = Ref{$itype}(zero($itype))
            nz_diag = Ref{$itype}(zero($itype))
            umfpack_numeric!(lu)
            @isok $lunz_r(lnz, unz, n_row, n_col, nz_diag, lu.numeric)
            (lnz[], unz[], n_row[], n_col[], nz_diag[])
        end
        function umf_lunz(lu::UmfpackLU{ComplexF64,$itype})
            lnz = Ref{$itype}(zero($itype))
            unz = Ref{$itype}(zero($itype))
            n_row = Ref{$itype}(zero($itype))
            n_col = Ref{$itype}(zero($itype))
            nz_diag = Ref{$itype}(zero($itype))
            umfpack_numeric!(lu)
            @isok $lunz_z(lnz, unz, n_row, n_col, nz_diag, lu.numeric)
            (lnz[], unz[], n_row[], n_col[], nz_diag[])
        end
        function getproperty(lu::UmfpackLU{Float64, $itype}, d::Symbol)
            if d === :L
                umfpack_numeric!(lu)        # ensure the numeric decomposition exists
                (lnz, unz, n_row, n_col, nz_diag) = umf_lunz(lu)
                Lp = Vector{$itype}(undef, n_row + 1)
                # L is returned in CSR (compressed sparse row) format
                Lj = Vector{$itype}(undef, lnz)
                Lx = Vector{Float64}(undef, lnz)
                @isok $get_num_r(
                            Lp, Lj, Lx,
                            C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, lu.numeric)
                return copy(transpose(SparseMatrixCSC(min(n_row, n_col), n_row,
                                                      increment!(Lp), increment!(Lj), Lx)))
            elseif d === :U
                umfpack_numeric!(lu)        # ensure the numeric decomposition exists
                (lnz, unz, n_row, n_col, nz_diag) = umf_lunz(lu)
                Up = Vector{$itype}(undef, n_col + 1)
                Ui = Vector{$itype}(undef, unz)
                Ux = Vector{Float64}(undef, unz)
                @isok $get_num_r(
                            C_NULL, C_NULL, C_NULL,
                            Up, Ui, Ux,
                            C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, lu.numeric)
                return  SparseMatrixCSC(min(n_row, n_col), n_col, increment!(Up),
                                        increment!(Ui), Ux)
            elseif d === :p
                umfpack_numeric!(lu)        # ensure the numeric decomposition exists
                (lnz, unz, n_row, n_col, nz_diag) = umf_lunz(lu)
                P  = Vector{$itype}(undef, n_row)
                @isok $get_num_r(
                            C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL,
                            P, C_NULL, C_NULL,
                            C_NULL, C_NULL, lu.numeric)
                return increment!(P)
            elseif d === :q
                umfpack_numeric!(lu)        # ensure the numeric decomposition exists
                (lnz, unz, n_row, n_col, nz_diag) = umf_lunz(lu)
                Q  = Vector{$itype}(undef, n_col)
                @isok $get_num_r(
                            C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL,
                            C_NULL, Q, C_NULL,
                            C_NULL, C_NULL, lu.numeric)
                return increment!(Q)
            elseif d === :Rs
                umfpack_numeric!(lu)        # ensure the numeric decomposition exists
                (lnz, unz, n_row, n_col, nz_diag) = umf_lunz(lu)
                Rs = Vector{Float64}(undef, n_row)
                do_recip = Ref{$itype}(0)
                @isok $get_num_r(
                            C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL,
                            do_recip, Rs, lu.numeric)
                return _scale_factors!(Rs, do_recip[])
            elseif d === :(:)
                umfpack_numeric!(lu)        # ensure the numeric decomposition exists
                (lnz, unz, n_row, n_col, nz_diag) = umf_lunz(lu)
                Lp = Vector{$itype}(undef, n_row + 1)
                # L is returned in CSR (compressed sparse row) format
                Lj = Vector{$itype}(undef, lnz)
                Lx = Vector{Float64}(undef, lnz)
                Up = Vector{$itype}(undef, n_col + 1)
                Ui = Vector{$itype}(undef, unz)
                Ux = Vector{Float64}(undef, unz)
                P  = Vector{$itype}(undef, n_row)
                Q  = Vector{$itype}(undef, n_col)
                Rs = Vector{Float64}(undef, n_row)
                do_recip = Ref{$itype}(0)
                @isok $get_num_r(
                            Lp, Lj, Lx,
                            Up, Ui, Ux,
                            P, Q, C_NULL,
                            do_recip, Rs, lu.numeric)
                return (copy(transpose(SparseMatrixCSC(min(n_row, n_col), n_row,
                                                       increment!(Lp), increment!(Lj),
                                                       Lx))),
                        SparseMatrixCSC(min(n_row, n_col), n_col, increment!(Up),
                                        increment!(Ui), Ux),
                        increment!(P), increment!(Q), _scale_factors!(Rs, do_recip[]))
            else
                return getfield(lu, d)
            end
        end
        function getproperty(lu::UmfpackLU{ComplexF64, $itype}, d::Symbol)
            if d === :L
                umfpack_numeric!(lu)        # ensure the numeric decomposition exists
                (lnz, unz, n_row, n_col, nz_diag) = umf_lunz(lu)
                Lp = Vector{$itype}(undef, n_row + 1)
                # L is returned in CSR (compressed sparse row) format
                Lj = Vector{$itype}(undef, lnz)
                Lx = Vector{Float64}(undef, lnz)
                Lz = Vector{Float64}(undef, lnz)
                @isok $get_num_z(
                            Lp, Lj, Lx, Lz,
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, lu.numeric)
                return copy(transpose(SparseMatrixCSC(min(n_row, n_col), n_row,
                                                      increment!(Lp), increment!(Lj),
                                                      complex.(Lx, Lz))))
            elseif d === :U
                umfpack_numeric!(lu)        # ensure the numeric decomposition exists
                (lnz, unz, n_row, n_col, nz_diag) = umf_lunz(lu)
                Up = Vector{$itype}(undef, n_col + 1)
                Ui = Vector{$itype}(undef, unz)
                Ux = Vector{Float64}(undef, unz)
                Uz = Vector{Float64}(undef, unz)
                @isok $get_num_z(
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            Up, Ui, Ux, Uz,
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, lu.numeric)
                return SparseMatrixCSC(min(n_row, n_col), n_col, increment!(Up),
                                       increment!(Ui), complex.(Ux, Uz))
            elseif d === :p
                umfpack_numeric!(lu)        # ensure the numeric decomposition exists
                (lnz, unz, n_row, n_col, nz_diag) = umf_lunz(lu)
                P  = Vector{$itype}(undef, n_row)
                @isok $get_num_z(
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            P, C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, lu.numeric)
                return increment!(P)
            elseif d === :q
                umfpack_numeric!(lu)        # ensure the numeric decomposition exists
                (lnz, unz, n_row, n_col, nz_diag) = umf_lunz(lu)
                Q  = Vector{$itype}(undef, n_col)
                @isok $get_num_z(
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            C_NULL, Q, C_NULL, C_NULL,
                            C_NULL, C_NULL, lu.numeric)
                return increment!(Q)
            elseif d === :Rs
                umfpack_numeric!(lu)        # ensure the numeric decomposition exists
                (lnz, unz, n_row, n_col, nz_diag) = umf_lunz(lu)
                Rs = Vector{Float64}(undef, n_row)
                do_recip = Ref{$itype}(0)
                @isok $get_num_z(
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            do_recip, Rs, lu.numeric)
                return _scale_factors!(Rs, do_recip[])
            elseif d === :(:)
                umfpack_numeric!(lu)        # ensure the numeric decomposition exists
                (lnz, unz, n_row, n_col, nz_diag) = umf_lunz(lu)
                Lp = Vector{$itype}(undef, n_row + 1)
                # L is returned in CSR (compressed sparse row) format
                Lj = Vector{$itype}(undef, lnz)
                Lx = Vector{Float64}(undef, lnz)
                Lz = Vector{Float64}(undef, lnz)
                Up = Vector{$itype}(undef, n_col + 1)
                Ui = Vector{$itype}(undef, unz)
                Ux = Vector{Float64}(undef, unz)
                Uz = Vector{Float64}(undef, unz)
                P  = Vector{$itype}(undef, n_row)
                Q  = Vector{$itype}(undef, n_col)
                Rs = Vector{Float64}(undef, n_row)
                do_recip = Ref{$itype}(0)
                @isok $get_num_z(
                            Lp, Lj, Lx, Lz,
                            Up, Ui, Ux, Uz,
                            P, Q, C_NULL, C_NULL,
                            do_recip, Rs, lu.numeric)
                return (copy(transpose(SparseMatrixCSC(min(n_row, n_col), n_row,
                                                       increment!(Lp), increment!(Lj),
                                                       complex.(Lx, Lz)))),
                        SparseMatrixCSC(min(n_row, n_col), n_col, increment!(Up),
                                        increment!(Ui), complex.(Ux, Uz)),
                        increment!(P), increment!(Q), _scale_factors!(Rs, do_recip[]))
            else
                return getfield(lu, d)
            end
        end
    end
end

# backward compatibility
umfpack_extract(lu::UmfpackLU) = getproperty(lu, :(:))

function nnz(lu::UmfpackLU)
    lnz, unz, = umf_lunz(lu)
    return Int(lnz + unz)
end

LinearAlgebra.issuccess(lu::UmfpackLU) = lu.status == UMFPACK_OK

"""
    rcond(F::UmfpackLU) -> Float64

Return UMFPACK's rough estimate of the reciprocal condition number of the
factorized matrix, computed from the diagonal of the factor alone: the smallest
entry of `abs.(diag(F.U))` divided by the largest.

This is much cheaper than a norm-based estimate such as `cond(A, 1)`, but also
much cruder, and it describes the matrix UMFPACK actually factorized rather
than `A` itself. UMFPACK scales the rows of `A` before factorizing by default
(see `F.Rs`), so for instance every diagonal matrix reports `1`. Unlike the
Cholesky-based [`CHOLMOD.rcond`](@ref SparseArrays.CHOLMOD.rcond), the value
is neither an upper nor a lower bound on `1 / cond(A, 2)`. Use it to detect a
singular or badly pivoted factorization, not to measure conditioning.

Returns `0` if the matrix is singular, and `1` if the matrix is 1-by-1.

# Examples
```jldoctest
julia> F = lu(sparse([1.0 3.0; 0.0 1.0]));

julia> SparseArrays.UMFPACK.rcond(F)
0.25

julia> minimum(abs, diag(F.U)) / maximum(abs, diag(F.U))
0.25

julia> SparseArrays.UMFPACK.rcond(lu(sparse([1.0 2.0; 0.0 0.0]); check=false))
0.0
```
"""
function rcond(F::UmfpackLU)
    umfpack_numeric!(F)        # ensure the numeric decomposition exists
    return F.info[JL_UMFPACK_RCOND]
end

### Solve with Factorization

ldiv!(lu::UmfpackLU{T}, B::StridedVecOrMat{T}; workspace=nothing) where {T<:UMFVTypes} =
    ldiv!(B, lu, copy(B); workspace)
ldiv!(translu::TransposeFactorization{T,<:UmfpackLU{T}}, B::StridedVecOrMat{T}; workspace=nothing) where {T<:UMFVTypes} =
    ldiv!(B, translu, copy(B); workspace)
ldiv!(adjlu::AdjointFactorization{T,<:UmfpackLU{T}}, B::StridedVecOrMat{T}; workspace=nothing) where {T<:UMFVTypes} =
    ldiv!(B, adjlu, copy(B); workspace)
ldiv!(lu::UmfpackLU{Float64}, B::StridedVecOrMat{<:Complex}; workspace=nothing) =
    ldiv!(B, lu, copy(B); workspace)
ldiv!(translu::TransposeFactorization{Float64,<:UmfpackLU{Float64}}, B::StridedVecOrMat{<:Complex}; workspace=nothing) =
    ldiv!(B, translu, copy(B); workspace)
ldiv!(adjlu::AdjointFactorization{Float64,<:UmfpackLU{Float64}}, B::StridedVecOrMat{<:Complex}; workspace=nothing) =
    ldiv!(B, adjlu, copy(B); workspace)

function ldiv!(lu::Union{UmfpackLU,UMFAdjOrTransLU}, B::AdjOrTrans{<:Any,<:StridedVecOrMat}; workspace=nothing)
    X = Matrix(B)
    ldiv!(lu, X; workspace)
    return copyto!(B, X)
end

ldiv!(X::StridedVecOrMat{T}, lu::UmfpackLU{T}, B::StridedVecOrMat{T}; workspace=nothing) where {T<:UMFVTypes} =
    _Aq_ldiv_B!(X, lu, B, UMFPACK_A, workspace)
ldiv!(X::StridedVecOrMat{T}, translu::TransposeFactorization{T,<:UmfpackLU{T}}, B::StridedVecOrMat{T}; workspace=nothing) where {T<:UMFVTypes} =
    (lu = parent(translu); _Aq_ldiv_B!(X, lu, B, UMFPACK_Aat, workspace))
ldiv!(X::StridedVecOrMat{T}, adjlu::AdjointFactorization{T,<:UmfpackLU{T}}, B::StridedVecOrMat{T}; workspace=nothing) where {T<:UMFVTypes} =
    (lu = parent(adjlu); _Aq_ldiv_B!(X, lu, B, UMFPACK_At, workspace))
ldiv!(X::StridedVecOrMat{Tb}, lu::UmfpackLU{Float64}, B::StridedVecOrMat{Tb}; workspace=nothing) where {Tb<:Complex} =
    _Aq_ldiv_B!(X, lu, B, UMFPACK_A, workspace)
ldiv!(X::StridedVecOrMat{Tb}, translu::TransposeFactorization{Float64,<:UmfpackLU{Float64}}, B::StridedVecOrMat{Tb}; workspace=nothing) where {Tb<:Complex} =
    (lu = parent(translu); _Aq_ldiv_B!(X, lu, B, UMFPACK_Aat, workspace))
ldiv!(X::StridedVecOrMat{Tb}, adjlu::AdjointFactorization{Float64,<:UmfpackLU{Float64}}, B::StridedVecOrMat{Tb}; workspace=nothing) where {Tb<:Complex} =
    (lu = parent(adjlu); _Aq_ldiv_B!(X, lu, B, UMFPACK_At, workspace))

function _Aq_ldiv_B!(X::StridedVecOrMat, lu::UmfpackLU, B::StridedVecOrMat, transposeoptype,
                     workspace::Union{Nothing,UmfpackWS})
    checksquare(lu)
    if size(X, 2) != size(B, 2)
        throw(DimensionMismatch("input and output arrays must have same number of columns"))
    end
    _AqldivB_kernel!(X, lu, B, transposeoptype, workspace === nothing ? UmfpackWS(lu) : workspace)
    return X
end
function _AqldivB_kernel!(x::StridedVector{T}, lu::UmfpackLU{T},
                          b::StridedVector{T}, transposeoptype, workspace) where {T<:UMFVTypes}
    solve!(x, lu, b, transposeoptype; workspace)
end
function _AqldivB_kernel!(X::StridedMatrix{T}, lu::UmfpackLU{T},
                          B::StridedMatrix{T}, transposeoptype, workspace) where {T<:UMFVTypes}
    for col in axes(X, 2)
        solve!(view(X, :, col), lu, view(B, :, col), transposeoptype; workspace)
    end
end
function _AqldivB_kernel!(x::StridedVector{Tb}, lu::UmfpackLU{Float64},
                          b::StridedVector{Tb}, transposeoptype, workspace) where Tb<:Complex
    r = similar(b, Float64)
    i = similar(b, Float64)
    c = real.(b)
    solve!(r, lu, c, transposeoptype; workspace)
    c .= imag.(b)
    solve!(i, lu, c, transposeoptype; workspace)
    map!(complex, x, r, i)
end
function _AqldivB_kernel!(X::StridedMatrix{Tb}, lu::UmfpackLU{Float64},
                          B::StridedMatrix{Tb}, transposeoptype, workspace) where Tb<:Complex
    r = similar(B, Float64, size(B, 1))
    i = similar(B, Float64, size(B, 1))
    c = similar(B, Float64, size(B, 1))
    for j in axes(B, 2)
        c .= real.(view(B, :, j))
        solve!(r, lu, c, transposeoptype; workspace)
        c .= imag.(view(B, :, j))
        solve!(i, lu, c, transposeoptype; workspace)
        map!(complex, view(X, :, j), r, i)
    end
end

for Tv in (:Float64, :ComplexF64), Ti in UmfpackIndexTypes
    # Used by the finalizers too. These are idempotent: the C
    # routine nulls the pointer it is handed (a temporary `Ref`), so we null
    # the wrapper's own pointer as well, making a second call a no-op rather
    # than a double free.
    _free_symbolic = Symbol(umf_nm("free_symbolic", Tv, Ti))
    @eval function umfpack_free_symbolic(symbolic::Symbolic, ::Type{$Tv}, ::Type{$Ti})
        if _isnotnull(symbolic)
            r = Ref(symbolic.p)
            symbolic.p = C_NULL
            $_free_symbolic(r)
        end
        return symbolic
    end
    _free_numeric = Symbol(umf_nm("free_numeric", Tv, Ti))
    @eval function umfpack_free_numeric(numeric::Numeric, ::Type{$Tv}, ::Type{$Ti})
        if _isnotnull(numeric)
            r = Ref(numeric.p)
            numeric.p = C_NULL
            $_free_numeric(r)
        end
        return numeric
    end

    # A null object copies to a null object. The copy is wrapped before the status is
    # checked, so that its finalizer owns whatever UMFPACK returned.
    _copy_symbolic = Symbol(umf_nm("copy_symbolic", Tv, Ti))
    @eval function umfpack_copy_symbolic(symbolic::Symbolic{$Tv,$Ti})
        _isnull(symbolic) && return Symbolic{$Tv,$Ti}(C_NULL)
        tmp = Ref{Ptr{Cvoid}}(C_NULL)
        status = $_copy_symbolic(tmp, symbolic)
        res = Symbolic{$Tv,$Ti}(tmp[])
        umferror(status)
        return res
    end
    _copy_numeric = Symbol(umf_nm("copy_numeric", Tv, Ti))
    @eval function umfpack_copy_numeric(numeric::Numeric{$Tv,$Ti})
        _isnull(numeric) && return Numeric{$Tv,$Ti}(C_NULL)
        tmp = Ref{Ptr{Cvoid}}(C_NULL)
        status = $_copy_numeric(tmp, numeric)
        res = Numeric{$Tv,$Ti}(tmp[])
        umferror(status)
        return res
    end

    _report_symbolic = Symbol(umf_nm("report_symbolic", Tv, Ti))
    @eval function umfpack_report_symbolic(lu::UmfpackLU{$Tv,$Ti}, level::Real=4; q=nothing)
        umfpack_symbolic!(lu, q)
        control = copy(lu.control)
        control[JL_UMFPACK_PRL] = level
        @isok $_report_symbolic(lu.symbolic, control)
        return lu
    end
    _report_numeric = Symbol(umf_nm("report_numeric", Tv, Ti))
    @eval function umfpack_report_numeric(lu::UmfpackLU{$Tv,$Ti}, level::Real=4; q=nothing)
        umfpack_numeric!(lu; q)
        control = copy(lu.control)
        control[JL_UMFPACK_PRL] = level
        @isok $_report_numeric(lu.numeric, control)
        return lu
    end
    # the control and info arrays
    _defaults = Symbol(umf_nm("defaults", Tv, Ti))
    @eval function get_umfpack_control(::Type{$Tv}, ::Type{$Ti})
        LibSuiteSparse.init_suitesparse()
        control = Vector{Float64}(undef, UMFPACK_CONTROL)
        $_defaults(control)
        # Put julia's config here
        # disable iterative refinement by default Issue #122
        control[JL_UMFPACK_IRSTEP] = 0

        return control
    end
end
end # UMFPACK module
