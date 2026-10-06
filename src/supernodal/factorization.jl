# This file is a part of Julia. License is MIT: https://julialang.org/license

# Supernodal LU factorization of the sparse matrix `A`. Integer and other
# non-floating-point element types are converted with `float`. `ordering` and `matching`
# are as for `_analyze_and_factor`, and `q` is a column order to use in place of the
# fill-reducing one, one- or zero-based. `check = true` throws `SingularException` when the
# factorization did not succeed.
function supernodal_lu(
        A::AbstractSparseMatrixCSC; check::Bool = true, ordering::Symbol = :auto,
        matching::Union{Symbol, Bool} = :auto, q::Union{Nothing, AbstractVector{<:Integer}} = nothing
    )
    qinit = _column_order(q, size(A, 2))
    S = convert(SparseMatrixCSC{float(eltype(A)), SparseArrays.indtype(A)}, A)
    F = _analyze_and_factor(S, ordering, matching, qinit)
    check && _checksuccess(F)
    return F
end

function _column_order(q::Union{Nothing, AbstractVector{<:Integer}}, n::Int)
    q === nothing && return nothing
    length(q) == n || throw(DimensionMismatch(
        "the column order has length $(length(q)) but the matrix has $n columns"))
    p = Vector{Int}(q)
    n > 0 && minimum(p) == 0 && (p .+= 1)
    isperm(p) || throw(ArgumentError("the column order is not a permutation of 1:$n or 0:$(n - 1)"))
    return p
end

# A factorization fails when the matrix is structurally singular, a pivot is zero or a
# pivot is not finite.
LinearAlgebra.issuccess(F::SupernodalLU) = F.finite && !F.matchfail && F.zeropiv == 0

_checksuccess(F::SupernodalLU) = issuccess(F) || throw(SingularException(F.zeropiv))

Base.size(F::SupernodalLU) = (F.m, F.n)
Base.size(F::SupernodalLU, d::Integer) =
    d < 1 ? throw(ArgumentError("dimension must be ≥ 1, got $d")) : d <= 2 ? size(F)[d] : 1

# Entries stored in L and U, including the unit diagonal and the explicit zeros of the
# dense supernode blocks.
SparseArrays.nnz(F::SupernodalLU) = length(F.sn.nzval) + length(F.Ux)

function Base.show(io::IO, F::SupernodalLU)
    print(io, summary(F), " of a ", F.m, "×", F.n, " sparse matrix with ", nnz(F),
        " stored entries in L and U")
    return nothing
end
Base.show(io::IO, ::MIME"text/plain", F::SupernodalLU) = show(io, F)

Base.propertynames(F::SupernodalLU, private::Bool = false) =
    (:L, :U, :p, :q, :Rs, (private ? fieldnames(typeof(F)) : ())...)

# `(F.Rs .* A)[F.p, F.q] ≈ F.L * F.U`, with L m×min(m, n) and U min(m, n)×n. The column
# scaling of the matching is folded into U.
function Base.getproperty(F::SupernodalLU, d::Symbol)
    if d === :L || d === :U
        return @lock getfield(F, :lock) _extract_factor(F, d)
    elseif d === :p
        return @lock getfield(F, :lock) getfield(F, :rowperm)[_rowpositions(F)[2]]
    elseif d === :q
        return copy(getfield(F, :qfac))
    elseif d === :Rs
        return copy(getfield(F, :rscale))
    else
        return getfield(F, d)
    end
end

# The position of every row of B in L, and the rows in position order: the pivot rows,
# then for a tall matrix the rows without a pivot, in increasing order.
function _rowpositions(F::SupernodalLU)
    m = F.m
    pivrow = F.pivrow
    pos = zeros(Int, m)
    for (k, r) in enumerate(pivrow)
        pos[r] = k
    end
    order = copy(pivrow)
    for r in 1:m
        if pos[r] == 0
            push!(order, r)
            pos[r] = length(order)
        end
    end
    return pos, order
end

function _extract_factor(F::SupernodalLU{Tv, Ti}, which::Symbol) where {Tv, Ti}
    (; m, n, npiv, xsup, sn, Up, Ui, Ux, cscale) = F
    colperm = F.qfac
    npos = min(m, n)
    pos = _rowpositions(F)[1]
    Is = Ti[]
    Js = Ti[]
    Vs = Tv[]
    for s in 1:nsuper(F)
        f = xsup[s]
        nc = xsup[s + 1] - f
        rows = _rows(sn, s)
        nr = length(rows)
        V = _vals(sn, s)
        for a in 1:nc
            j = f + a - 1
            off = (a - 1) * nr
            if which === :L
                push!(Is, j); push!(Js, j); push!(Vs, one(Tv))
                for b in (a + 1):nr
                    push!(Is, pos[rows[b]]); push!(Js, j); push!(Vs, V[off + b])
                end
            else
                for b in 1:a
                    push!(Is, f + b - 1); push!(Js, j); push!(Vs, V[off + b] / cscale[colperm[j]])
                end
            end
        end
    end
    if which === :L
        # the positions without a pivot, of the rows a wide rank-deficient matrix leaves
        for k in (npiv + 1):npos
            push!(Is, k); push!(Js, k); push!(Vs, one(Tv))
        end
        return SparseArrays.sparse(Is, Js, Vs, m, npos)
    end
    for j in 1:n
        cj = cscale[colperm[j]]
        for p in Up[j]:(Up[j + 1] - 1)
            push!(Is, Ui[p]); push!(Js, j); push!(Vs, Ux[p] / cj)
        end
    end
    return SparseArrays.sparse(Is, Js, Vs, npos, n)
end

function Base.copy(F::SupernodalLU{Tv}) where {Tv}
    @lock F.lock begin
        return typeof(F)(
            F.m, F.n, F.colperm, F.panels, copy(F.qfac), F.npiv, F.nnzl, copy(F.pivrow), copy(F.xsup), copy(F.sn),
            copy(F.Up), copy(F.Ui), copy(F.Ux), F.zeropiv, F.finite,
            F.matchfail, F.ordering, F.matching, F.qinit, F.colptr, F.rowval, F.rowperm, F.rscale,
            F.cscale, F.matched, F.B, map(similar, F.work), ReentrantLock()
        )
    end
end

"""
    lu!(F::SupernodalLU, A::AbstractSparseMatrixCSC; check = true, reuse_symbolic = true, q = nothing) -> F

Refactorize `F`, the result of [`lu`](@ref) of a sparse matrix, with `A`. When `A` has the
size and sparsity pattern of the matrix `F` factorizes, `reuse_symbolic = true` and no `q`
is given, the ordering, matching and scaling of `F` are kept and only the numerical
factorization is redone, with pivoting as before; otherwise `A` is analyzed afresh, in the
column order `q` when given. `A` is converted to the element type of `F`. `lu!(F)` redoes
the numerical factorization of the matrix `F` holds.

`check` and `q` are as for [`lu`](@ref).

# Examples
```jldoctest
julia> A = sparse([2.0 1 0; 0 3 1; 1 0 4]);

julia> F = lu(A);

julia> lu!(F, 2A);

julia> F \\ [3.0, 4.0, 5.0] ≈ (2A) \\ [3.0, 4.0, 5.0]
true
```
"""
function LinearAlgebra.lu!(
        F::SupernodalLU{Tv, Ti}, A::AbstractSparseMatrixCSC; check::Bool = true,
        reuse_symbolic::Bool = true, q::Union{Nothing, AbstractVector{<:Integer}} = nothing
    ) where {Tv, Ti}
    Tv <: Real && !(eltype(A) <: Real) && throw(ArgumentError(
        "cannot refactorize the real $(typeof(F)) with a matrix of eltype $(eltype(A)); use lu(A) instead"))
    qinit = q === nothing ? nothing : _column_order(q, size(A, 2))
    S = convert(SparseMatrixCSC{Tv, Ti}, A)
    @lock F.lock begin
        if reuse_symbolic && q === nothing && size(S) == size(F) &&
                getcolptr(S) == F.colptr && rowvals(S) == F.rowval
            _refactor!(F, S)
        else
            G = _analyze_and_factor(S, F.ordering, F.matching, q === nothing ? F.qinit : qinit)
            for f in fieldnames(SupernodalLU)
                f === :lock || setfield!(F, f, getfield(G, f))
            end
        end
        check && _checksuccess(F)
    end
    return F
end

function LinearAlgebra.lu!(F::SupernodalLU; check::Bool = true)
    @lock F.lock begin
        _factor!(F)
        check && _checksuccess(F)
    end
    return F
end

function _checksolve(F::SupernodalLU, B::AbstractVecOrMat)
    F.m == F.n || throw(DimensionMismatch(
        "a solve needs the LU factorization of a square matrix, not of a $(F.m)×$(F.n) one; use qr instead"))
    require_one_based_indexing(B)
    size(B, 1) == F.n || throw(DimensionMismatch(
        "the factorization is $(F.m)×$(F.n) but the right-hand side has $(size(B, 1)) rows"))
    return nothing
end

# X := op(F) \ B, checked before X is written
function _checksolve(F::SupernodalLU, X::AbstractVecOrMat, B::AbstractVecOrMat)
    _checksolve(F, B)
    size(X) == size(B) || throw(DimensionMismatch(
        "the output has size $(size(X)) but the right-hand side has size $(size(B))"))
    return nothing
end

function LinearAlgebra.ldiv!(F::SupernodalLU, B::AbstractVecOrMat)
    _checksolve(F, B)
    return @lock F.lock _ldiv!(F, B, identity)
end
function LinearAlgebra.ldiv!(Ft::TransposeFactorization{<:Any, <:SupernodalLU}, B::AbstractVecOrMat)
    F = parent(Ft)
    _checksolve(F, B)
    return @lock F.lock _ldiv!(F, B, transpose)
end
function LinearAlgebra.ldiv!(Fa::AdjointFactorization{<:Any, <:SupernodalLU}, B::AbstractVecOrMat)
    F = parent(Fa)
    _checksolve(F, B)
    return @lock F.lock _ldiv!(F, B, adjoint)
end

const _SupernodalOp = Union{SupernodalLU, TransposeFactorization{<:Any, <:SupernodalLU},
    AdjointFactorization{<:Any, <:SupernodalLU}}
for TX in (AbstractVector, AbstractMatrix), TB in (AbstractVector, AbstractMatrix)
    @eval function LinearAlgebra.ldiv!(X::$TX, F::_SupernodalOp, B::$TB)
        _checksolve(F isa SupernodalLU ? F : parent(F), X, B)
        return ldiv!(F, copyto!(X, B))
    end
end

function _permsign(p::AbstractVector{Int})
    seen = falses(length(p))
    s = 1
    for i in eachindex(p)
        seen[i] && continue
        j = i
        len = 0
        while !seen[j]
            seen[j] = true
            j = p[j]
            len += 1
        end
        iseven(len) && (s = -s)
    end
    return s
end

# P·B·Q = L·U with B = (Dr·A·Dc)[rowperm, :], so det(A) is the product of the pivots
# times the signs of the three permutations, divided by the scalings.
function LinearAlgebra.logabsdet(F::SupernodalLU{Tv}) where {Tv}
    LinearAlgebra.checksquare(F)
    @lock F.lock begin
        R = real(Tv)
        la = zero(R)
        sg = one(Tv)
        (; xsup, sn) = F
        for s in 1:nsuper(F)
            nr = sn.rowptr[s + 1] - sn.rowptr[s]
            V = _vals(sn, s)
            for a in 1:(xsup[s + 1] - xsup[s])
                d = V[(a - 1) * nr + a]
                iszero(d) && return (R(-Inf), zero(Tv))
                la += log(abs(d))
                sg *= sign(d)
            end
        end
        for x in F.rscale
            la -= log(x)
        end
        for x in F.cscale
            la -= log(x)
        end
        sg *= _permsign(F.pivrow) * _permsign(F.qfac) * _permsign(F.rowperm)
        return (la, sg)
    end
end
