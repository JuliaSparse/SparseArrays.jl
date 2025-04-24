# This file is a part of Julia. License is MIT: https://julialang.org/license

module UMFPACK

export UmfpackLU

import Base: (\), getproperty, show, size
using LinearAlgebra
using LinearAlgebra: AdjOrTrans
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
Working space for Umfpack so `ldiv!` doesn't allocate.

To use multiple threads, each thread should have their own workspace this can be done using`copy(::UmfpackLU)`


The constructor is overloaded so to create appropriate sized working space based on the lu
factorization or the sparse matrix and the refinement setting.
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
mutable struct UmfpackLU{Tv<:UMFVTypes,Ti<:UMFITypes} <: Factorization{Tv}
    symbolic::Symbolic{Tv, Ti}
    numeric::Numeric{Tv, Ti}
    m::Int
    n::Int
    colptr::Vector{Ti}                  # 0-based column pointers
    rowval::Vector{Ti}                  # 0-based row indices
    nzval::Vector{Tv}
    status::Int
    workspace::UmfpackWS{Ti}
    control::Vector{Float64}
    info::Vector{Float64}
    lock::ReentrantLock
end

function UmfpackLU(S::AbstractSparseMatrixCSC{Tv, Ti};
    control=get_umfpack_control(Tv, Ti)) where
    {Tv<:UMFVTypes,Ti<:UMFITypes}

    zerobased = getcolptr(S)[1] == 0
    return UmfpackLU(Symbolic{Tv, Ti}(C_NULL), Numeric{Tv, Ti}(C_NULL),
                    size(S, 1), size(S, 2),
                    zerobased ? copy(getcolptr(S)) : decrement(getcolptr(S)),
                    zerobased ? copy(rowvals(S)) : decrement(rowvals(S)),
                    copy(nonzeros(S)), 0, UmfpackWS(S, has_refinement(control)),
                    copy(control), Vector{Float64}(undef, UMFPACK_INFO),
                    ReentrantLock()
    )
end

workspace_W_size(F::UmfpackLU) = workspace_W_size(F, has_refinement(F))
workspace_W_size(S::Union{UmfpackLU{<:AbstractFloat}, AbstractSparseMatrixCSC{<:AbstractFloat}}, refinement::Bool) = refinement ? 5 * size(S, 2) : size(S, 2)
workspace_W_size(S::Union{UmfpackLU{<:Complex}, AbstractSparseMatrixCSC{<:Complex}}, refinement::Bool) = refinement ? 10 * size(S, 2) : 4 * size(S, 2)

const ATLU = Union{TransposeFactorization{<:Any, <:UmfpackLU}, AdjointFactorization{<:Any, <:UmfpackLU}}
has_refinement(F::ATLU) = has_refinement(F.parent)
has_refinement(F::UmfpackLU) = has_refinement(F.control)
has_refinement(control::AbstractVector) = control[JL_UMFPACK_IRSTEP] > 0

# auto magick resize, should this only expand and not shrink?
function getworkspace(F::UmfpackLU)
    @lock F.lock begin
        return resize!(F.workspace, F, has_refinement(F); expand_only = true)
    end
end

UmfpackWS(F::UmfpackLU{Tv, Ti}, refinement::Bool=has_refinement(F)) where {Tv, Ti} = UmfpackWS(
        Vector{Ti}(undef, size(F, 2)),
        Vector{Float64}(undef, workspace_W_size(F, refinement)))
UmfpackWS(F::ATLU, refinement::Bool=has_refinement(F)) = UmfpackWS(F.parent, refinement)

"""
    copy(F::UmfpackLU, [ws::UmfpackWS])::UmfpackLU
A shallow copy of UmfpackLU to use in multithreaded solve applications.
This function duplicates the working space, control, info and lock fields.
"""
# Not using similar helps if the actual needed size has changed as it would need to be resized again
Base.copy(F::UmfpackLU{Tv, Ti}, ws=UmfpackWS(F)) where {Tv, Ti} =
    UmfpackLU(
        F.symbolic,
        F.numeric,
        F.m, F.n,
        F.colptr,
        F.rowval,
        F.nzval,
        F.status,
        ws,
        copy(F.control),
        copy(F.info),
        ReentrantLock()
    )
Base.copy(F::T, ws=UmfpackWS(F)) where {T <: ATLU} =
    T(copy(parent(F), ws))

Base.transpose(F::UmfpackLU) = TransposeFactorization(F)

show_umf_ctrl(F::UmfpackLU, level::Real=2.0) =
    @lock F.lock show_umf_ctrl(F.control, level)


show_umf_info(F::UmfpackLU, level::Real=2.0) =
    @lock F.lock show_umf_info(F.control, F.info, level)


"""
    lu(A::AbstractSparseMatrixCSC; check = true, q = nothing, control = get_umfpack_control()) -> F::UmfpackLU

Compute the LU factorization of a sparse matrix `A`.

For sparse `A` with real or complex element type, the return type of `F` is
`UmfpackLU{Tv, Ti}`, with `Tv` = [`Float64`](@ref) or `ComplexF64` respectively and
`Ti` is an integer type ([`Int32`](@ref) or [`Int64`](@ref)).

When `check = true`, an error is thrown if the decomposition fails.
When `check = false`, responsibility for checking the decomposition's
validity (via [`issuccess`](@ref)) lies with the user.

The permutation `q` can either be a permutation vector or `nothing`. If no permutation vector
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
| `p`       | right permutation `Vector`          |
| `q`       | left permutation `Vector`           |
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
lu(A::AbstractSparseMatrixCSC{<:Union{Float16,Float32},Ti};
   check::Bool = true) where {Ti<:UMFITypes} =
    lu(convert(SparseMatrixCSC{Float64,Ti}, A); check = check)
lu(A::AbstractSparseMatrixCSC{<:Union{ComplexF16,ComplexF32},Ti};
   check::Bool = true) where {Ti<:UMFITypes} =
    lu(convert(SparseMatrixCSC{ComplexF64,Ti}, A); check = check)
lu(A::Union{AbstractSparseMatrixCSC{T},AbstractSparseMatrixCSC{Complex{T}}};
   check::Bool = true) where {T<:AbstractFloat} =
    throw(ArgumentError(string("matrix type ", typeof(A), " not supported. ",
    "Try lu(convert(SparseMatrixCSC{Float64/ComplexF64,Int}, A)) for ",
    "sparse floating point LU using UMFPACK or lu(Array(A)) for generic ",
    "dense LU.")))
lu(A::AbstractSparseMatrixCSC; check::Bool = true) = lu(float(A); check = check)

# We could do this as lu(A') = lu(A)' with UMFPACK, but the user could want to do one over the other
lu(A::AdjOrTrans{T,S}; check::Bool = true) where {T<:UMFVTypes, S<:AbstractSparseMatrixCSC{T}} =
    lu(copy(A); check)

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

The permutation `q` can either be a permutation vector or `nothing`. If no permutation vector
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

    F.m = size(S, 1)
    F.n = size(S, 2)

    # resize workspace if needed
    resize!(F.workspace, S, has_refinement(F))

    resize!(F.colptr, length(getcolptr(S)))
    if zerobased
        F.colptr .= getcolptr(S)
    else
        F.colptr .= getcolptr(S) .- one(eltype(S))
    end

    resize!(F.rowval, length(rowvals(S)))
    if zerobased
        F.rowval .= rowvals(S)
    else
        F.rowval .= rowvals(S) .- one(eltype(S))
    end

    resize!(F.nzval, length(nonzeros(S)))
    F.nzval .= nonzeros(S)
    return lu!(F; reuse_symbolic, check, q)
end

function lu!(F::UmfpackLU; check::Bool=true, reuse_symbolic::Bool=true, q=nothing)
    if !reuse_symbolic && _isnotnull(F.symbolic)
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
    if _isnotnull(F.numeric)
        if issuccess(F)
            summary(io, F); println(io)
            println(io, "L factor:")
            show(io, mime, F.L)
            println(io, "\nU factor:")
            show(io, mime, F.U)
        else
            print(io, "Failed factorization of type $(typeof(F))")
        end
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
    serialize(s, length(L.workspace.Wi))
    serialize(s, length(L.workspace.W))
    serialize(s, L.control)
    serialize(s, L.info)
end
function deserialize(s::AbstractSerializer, ::Type{UmfpackLU{Tv,Ti}}) where {Tv,Ti}
    # TODO: If we can get a C FILE handle we can deserialize umfpack_numeric and
    # umfpack_symbolic. using the load_{numeric | symbolic} functions.
    m        = deserialize(s)
    n        = deserialize(s)
    colptr   = deserialize(s)
    rowval   = deserialize(s)
    nzval    = deserialize(s)
    Wisize   = deserialize(s)
    Wsize    = deserialize(s)
    control  = deserialize(s)
    info     = deserialize(s)
    return UmfpackLU{Tv,Ti}(Symbolic{Tv, Ti}(C_NULL), Numeric{Tv, Ti}(C_NULL),
        m, n, colptr, rowval, nzval, 0,
        UmfpackWS{Ti}(Wisize, Wsize), control, info, ReentrantLock())
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
        function umfpack_symbolic!(U::UmfpackLU{Float64,$itype}, q::Union{Nothing, StridedVector{$itype}})
            _isnotnull(U.symbolic) && return U
            @lock U.lock begin
                tmp = Ref{Ptr{Cvoid}}(C_NULL)
                if q === nothing
                    @isok $sym_r(U.m, U.n, U.colptr, U.rowval, U.nzval, tmp, U.control, U.info)
                else
                    qq = minimum(q) == 1 ? q .- one(eltype(q)) : q
                    @isok $symq_r(U.m, U.n, U.colptr, U.rowval, U.nzval, qq, tmp, U.control, U.info)
                end
                U.symbolic = Symbolic{Float64, $itype}(tmp[])

            end
            return U
        end
        function umfpack_symbolic!(U::UmfpackLU{ComplexF64,$itype}, q::Union{Nothing, StridedVector{$itype}})
            _isnotnull(U.symbolic) && return U
            @lock U.lock begin
                tmp = Ref{Ptr{Cvoid}}(C_NULL)
                if q === nothing
                    @isok $sym_c(U.m, U.n, U.colptr, U.rowval, real(U.nzval), imag(U.nzval), tmp,
                                 U.control, U.info)
                else
                    qq = minimum(q) == 1 ? q .- one(eltype(q)) : q
                    @isok $symq_c(U.m, U.n, U.colptr, U.rowval, real(U.nzval), imag(U.nzval), qq, tmp, U.control, U.info)
                end
                U.symbolic = Symbolic{ComplexF64, $itype}(tmp[])
            end
            return U
        end
        function umfpack_numeric!(U::UmfpackLU{Float64,$itype}; reuse_numeric=true, q=nothing)
            @lock U.lock begin
                (reuse_numeric && _isnotnull(U.numeric)) && return U
                if _isnull(U.symbolic)
                    umfpack_symbolic!(U, q)
                end
                tmp = Ref{Ptr{Cvoid}}(C_NULL)
                status = $num_r(U.colptr, U.rowval, U.nzval, U.symbolic, tmp, U.control, U.info)
                U.status = status
                if status != UMFPACK_WARNING_singular_matrix
                    umferror(status)
                end
                U.numeric = Numeric{Float64, $itype}(tmp[])
            end
            return U
        end
        function umfpack_numeric!(U::UmfpackLU{ComplexF64,$itype}; reuse_numeric=true, q=nothing)
            @lock U.lock begin
                (reuse_numeric && _isnotnull(U.numeric)) && return U
                _isnull(U.symbolic) && umfpack_symbolic!(U, q)
                tmp = Ref{Ptr{Cvoid}}(C_NULL)
                status = $num_c(U.colptr, U.rowval, real(U.nzval), imag(U.nzval), U.symbolic, tmp,
                    U.control, U.info)
                U.status = status
                if status != UMFPACK_WARNING_singular_matrix
                    umferror(status)
                end
                U.numeric = Numeric{ComplexF64, $itype}(tmp[])
            end
            return U
        end
        function solve!(x::StridedVector{Float64},
            lu::UmfpackLU{Float64,$itype}, b::StridedVector{Float64},
            typ::Integer; workspace = getworkspace(lu))
            if x === b
                throw(ArgumentError("output array must not be aliased with input array"))
            end
            if stride(x, 1) != 1 || stride(b, 1) != 1
                throw(ArgumentError("in and output vectors must have unit strides"))
            end
            if size(lu, 2) > length(workspace.Wi)
                throw(ArgumentError("Wi should be larger than `size(Af, 2)`"))
            end
            if workspace_W_size(lu) > length(workspace.W)
                throw(ArgumentError("W should be larger than `workspace_W_size(Af)`"))
            end
            @lock lu.lock begin
                umfpack_numeric!(lu)
                (size(b, 1) == lu.m) && (size(b) == size(x)) || throw(DimensionMismatch())

                @isok $wsol_r(typ, lu.colptr, lu.rowval, lu.nzval,
                    x, b, lu.numeric, lu.control,
                    lu.info, workspace.Wi, workspace.W)
            end
            return x
        end
        function solve!(x::StridedVector{ComplexF64},
            lu::UmfpackLU{ComplexF64,$itype}, b::StridedVector{ComplexF64},
            typ::Integer; workspace = getworkspace(lu))
            if x === b
                throw(ArgumentError("output array must not be aliased with input array"))
            end
            if stride(x, 1) != 1 || stride(b, 1) != 1
                throw(ArgumentError("in and output vectors must have unit strides"))
            end
            if size(lu, 2) > length(workspace.Wi)
                throw(ArgumentError("Wi should be at least larger than `size(Af, 2)`"))
            end
            if workspace_W_size(lu) > length(workspace.W)
                throw(ArgumentError("W should be larger than `workspace_W_size(Af)`"))
            end
            @lock lu.lock begin
                umfpack_numeric!(lu)
                (size(b, 1) == lu.m) && (size(b) == size(x)) || throw(DimensionMismatch())
                @isok $wsol_c(typ, lu.colptr, lu.rowval, lu.nzval, C_NULL, x, C_NULL, b,
                    C_NULL, lu.numeric, lu.control, lu.info, workspace.Wi, workspace.W)
            end
            return x
        end
        function det(lu::UmfpackLU{Float64,$itype})
            mx = Ref{Float64}(zero(Float64))
            @lock lu.lock @isok($det_r(mx, C_NULL, lu.numeric, lu.info))
            mx[]
        end

        function det(lu::UmfpackLU{ComplexF64,$itype})
            mx = Ref{Float64}(zero(Float64))
            mz = Ref{Float64}(zero(Float64))
            @lock lu.lock @isok($det_z(mx, mz, C_NULL, lu.numeric, lu.info))
            complex(mx[], mz[])
        end
        function logabsdet(F::UmfpackLU{T, $itype}) where {T<:Union{Float64,ComplexF64}} # return log(abs(det)) and sign(det)
            n = checksquare(F)
            issuccess(F) || return log(zero(real(T))), zero(T)
            U = F.U
            Rs = F.Rs
            p = F.p
            q = F.q
            s = _signperm(p)*_signperm(q)*one(real(T))
            P = one(T)
            abs_det = zero(real(T))
            @inbounds for i in 1:n
                dg_ii = U[i, i] / Rs[i]
                P *= sign(dg_ii)
                abs_det += log(abs(dg_ii))
            end
            return abs_det, s * P
        end
        function umf_lunz(lu::UmfpackLU{Float64,$itype})
            lnz = Ref{$itype}(zero($itype))
            unz = Ref{$itype}(zero($itype))
            n_row = Ref{$itype}(zero($itype))
            n_col = Ref{$itype}(zero($itype))
            nz_diag = Ref{$itype}(zero($itype))
            @isok $lunz_r(lnz, unz, n_row, n_col, nz_diag, lu.numeric)
            (lnz[], unz[], n_row[], n_col[], nz_diag[])
        end
        function umf_lunz(lu::UmfpackLU{ComplexF64,$itype})
            lnz = Ref{$itype}(zero($itype))
            unz = Ref{$itype}(zero($itype))
            n_row = Ref{$itype}(zero($itype))
            n_col = Ref{$itype}(zero($itype))
            nz_diag = Ref{$itype}(zero($itype))
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
                @isok $get_num_r(
                            C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL,
                            C_NULL, Rs, lu.numeric)
                return Rs
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
                @isok $get_num_r(
                            Lp, Lj, Lx,
                            Up, Ui, Ux,
                            P, Q, C_NULL,
                            C_NULL, Rs, lu.numeric)
                return (copy(transpose(SparseMatrixCSC(min(n_row, n_col), n_row,
                                                       increment!(Lp), increment!(Lj),
                                                       Lx))),
                        SparseMatrixCSC(min(n_row, n_col), n_col, increment!(Up),
                                        increment!(Ui), Ux),
                        increment!(P), increment!(Q), Rs)
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
                @isok $get_num_z(
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            C_NULL, C_NULL, C_NULL, C_NULL,
                            C_NULL, Rs, lu.numeric)
                return Rs
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
                @isok $get_num_z(
                            Lp, Lj, Lx, Lz,
                            Up, Ui, Ux, Uz,
                            P, Q, C_NULL, C_NULL,
                            C_NULL, Rs, lu.numeric)
                return (copy(transpose(SparseMatrixCSC(min(n_row, n_col), n_row,
                                                       increment!(Lp), increment!(Lj),
                                                       complex.(Lx, Lz)))),
                        SparseMatrixCSC(min(n_row, n_col), n_col, increment!(Up),
                                        increment!(Ui), complex.(Ux, Uz)),
                        increment!(P), increment!(Q), Rs)
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

### Solve with Factorization

ldiv!(lu::UmfpackLU{T}, B::StridedVecOrMat{T}) where {T<:UMFVTypes} =
    ldiv!(B, lu, copy(B))
ldiv!(translu::TransposeFactorization{T,<:UmfpackLU{T}}, B::StridedVecOrMat{T}) where {T<:UMFVTypes} =
    ldiv!(B, translu, copy(B))
ldiv!(adjlu::AdjointFactorization{T,<:UmfpackLU{T}}, B::StridedVecOrMat{T}) where {T<:UMFVTypes} =
    ldiv!(B, adjlu, copy(B))
ldiv!(lu::UmfpackLU{Float64}, B::StridedVecOrMat{<:Complex}) =
    ldiv!(B, lu, copy(B))
ldiv!(translu::TransposeFactorization{Float64,<:UmfpackLU{Float64}}, B::StridedVecOrMat{<:Complex}) =
    ldiv!(B, translu, copy(B))
ldiv!(adjlu::AdjointFactorization{Float64,<:UmfpackLU{Float64}}, B::StridedVecOrMat{<:Complex}) =
    ldiv!(B, adjlu, copy(B))

ldiv!(X::StridedVecOrMat{T}, lu::UmfpackLU{T}, B::StridedVecOrMat{T}) where {T<:UMFVTypes} =
    _Aq_ldiv_B!(X, lu, B, UMFPACK_A)
ldiv!(X::StridedVecOrMat{T}, translu::TransposeFactorization{T,<:UmfpackLU{T}}, B::StridedVecOrMat{T}) where {T<:UMFVTypes} =
    (lu = translu.parent; _Aq_ldiv_B!(X, lu, B, UMFPACK_Aat))
ldiv!(X::StridedVecOrMat{T}, adjlu::AdjointFactorization{T,<:UmfpackLU{T}}, B::StridedVecOrMat{T}) where {T<:UMFVTypes} =
    (lu = adjlu.parent; _Aq_ldiv_B!(X, lu, B, UMFPACK_At))
ldiv!(X::StridedVecOrMat{Tb}, lu::UmfpackLU{Float64}, B::StridedVecOrMat{Tb}) where {Tb<:Complex} =
    _Aq_ldiv_B!(X, lu, B, UMFPACK_A)
ldiv!(X::StridedVecOrMat{Tb}, translu::TransposeFactorization{Float64,<:UmfpackLU{Float64}}, B::StridedVecOrMat{Tb}) where {Tb<:Complex} =
    (lu = translu.parent; _Aq_ldiv_B!(X, lu, B, UMFPACK_Aat))
ldiv!(X::StridedVecOrMat{Tb}, adjlu::AdjointFactorization{Float64,<:UmfpackLU{Float64}}, B::StridedVecOrMat{Tb}) where {Tb<:Complex} =
    (lu = adjlu.parent; _Aq_ldiv_B!(X, lu, B, UMFPACK_At))

function _Aq_ldiv_B!(X::StridedVecOrMat, lu::UmfpackLU, B::StridedVecOrMat, transposeoptype)
    if size(X, 2) != size(B, 2)
        throw(DimensionMismatch("input and output arrays must have same number of columns"))
    end
    _AqldivB_kernel!(X, lu, B, transposeoptype)
    return X
end
function _AqldivB_kernel!(x::StridedVector{T}, lu::UmfpackLU{T},
                          b::StridedVector{T}, transposeoptype) where {T<:UMFVTypes}
    solve!(x, lu, b, transposeoptype)
end
function _AqldivB_kernel!(X::StridedMatrix{T}, lu::UmfpackLU{T},
                          B::StridedMatrix{T}, transposeoptype) where {T<:UMFVTypes}
    for col in axes(X, 2)
        solve!(view(X, :, col), lu, view(B, :, col), transposeoptype)
    end
end
function _AqldivB_kernel!(x::StridedVector{Tb}, lu::UmfpackLU{Float64},
                          b::StridedVector{Tb}, transposeoptype) where Tb<:Complex
    r = similar(b, Float64)
    i = similar(b, Float64)
    c = real.(b)
    solve!(r, lu, c, transposeoptype)
    c .= imag.(b)
    solve!(i, lu, c, transposeoptype)
    map!(complex, x, r, i)
end
function _AqldivB_kernel!(X::StridedMatrix{Tb}, lu::UmfpackLU{Float64},
                          B::StridedMatrix{Tb}, transposeoptype) where Tb<:Complex
    r = similar(B, Float64, size(B, 1))
    i = similar(B, Float64, size(B, 1))
    c = similar(B, Float64, size(B, 1))
    for j in axes(B, 2)
        c .= real.(view(B, :, j))
        solve!(r, lu, c, transposeoptype)
        c .= imag.(view(B, :, j))
        solve!(i, lu, c, transposeoptype)
        map!(complex, view(X, :, j), r, i)
    end
end

for Tv in (:Float64, :ComplexF64), Ti in UmfpackIndexTypes
    # no lock version for the finalizer
    _free_symbolic = Symbol(umf_nm("free_symbolic", Tv, Ti))
    @eval function umfpack_free_symbolic(symbolic::Symbolic, ::Type{$Tv}, ::Type{$Ti})
        if _isnotnull(symbolic)
            r = Ref(symbolic.p)
            $_free_symbolic(r)
        end
    end
    _free_numeric = Symbol(umf_nm("free_numeric", Tv, Ti))
    @eval function umfpack_free_numeric(numeric::Numeric, ::Type{$Tv}, ::Type{$Ti})
        if _isnotnull(numeric)
            r = Ref(numeric.p)
            $_free_numeric(r)
        end
    end

    _report_symbolic = Symbol(umf_nm("report_symbolic", Tv, Ti))
    @eval umfpack_report_symbolic(lu::UmfpackLU{$Tv,$Ti}, level::Real=4; q=nothing) =
        @lock lu.lock begin
            umfpack_symbolic!(lu, q)
            old_prl = lu.control[JL_UMFPACK_PRL]
            lu.control[JL_UMFPACK_PRL] = level
            @isok $_report_symbolic(lu.symbolic, lu.control)
            lu.control[JL_UMFPACK_PRL] = old_prl
            lu
        end
    _report_numeric = Symbol(umf_nm("report_numeric", Tv, Ti))
    @eval umfpack_report_numeric(lu::UmfpackLU{$Tv,$Ti}, level::Real=4; q=nothing) =
        @lock lu.lock begin
            umfpack_numeric!(lu; q)
            old_prl = lu.control[JL_UMFPACK_PRL]
            lu.control[JL_UMFPACK_PRL] = level
            @isok $_report_numeric(lu.numeric, lu.control)
            lu.control[JL_UMFPACK_PRL] = old_prl
            lu
        end
    # the control and info arrays
    _defaults = Symbol(umf_nm("defaults", Tv, Ti))
    @eval function get_umfpack_control(::Type{$Tv}, ::Type{$Ti})
        control = Vector{Float64}(undef, UMFPACK_CONTROL)
        $_defaults(control)
        # Put julia's config here
        # disable iterative refinement by default Issue #122
        control[JL_UMFPACK_IRSTEP] = 0

        return control
    end
end
end # UMFPACK module
