# This file is a part of Julia. License is MIT: https://julialang.org/license

# Solve phase. With P·B[:, q] = L·U, a solve with B runs L forward over the supernodes in
# the row space of B, then U backward in pivot positions. A solve with Bᵀ runs Uᵀ forward
# in pivot positions, then Lᵀ backward in the row space. `f` is applied to every factor
# entry: `conj` for the adjoint, `identity` otherwise.

# x := x - the contribution of supernode s to L \ x, in the row space of B: a unit lower
# solve on its block rows, then a product onto the rows below.
function _lsolve_super!(x::AbstractVector, F::SupernodalLU, s::Int, seg::AbstractVector, tbuf::AbstractVector)
    rows = _rows(F.sn, s)
    V = _vals(F.sn, s)
    nr = length(rows)
    nc = F.xsup[s + 1] - F.xsup[s]
    @inbounds for a in 1:nc
        seg[a] = x[rows[a]]
    end
    if nr * nc <= 4 * UPDATE_SCALAR_MAX
        @inbounds for a in 1:nc
            sa = seg[a]
            iszero(sa) && continue
            off = (a - 1) * nr
            for b in (a + 1):nc
                seg[b] -= V[off + b] * sa
            end
            for b in (nc + 1):nr
                x[rows[b]] -= V[off + b] * sa
            end
        end
    else
        M = reshape(V, nr, nc)
        sv = view(seg, 1:nc)
        ldiv!(UnitLowerTriangular(view(M, 1:nc, 1:nc)), sv)
        nb = nr - nc
        if nb > 0
            tv = view(tbuf, 1:nb)
            mul!(tv, view(M, (nc + 1):nr, 1:nc), sv)
            @inbounds for b in 1:nb
                x[rows[nc + b]] -= tbuf[b]
            end
        end
    end
    @inbounds for a in 1:nc
        x[rows[a]] = seg[a]
    end
    return x
end

# z := U \ z in pivot positions.
function _usolve!(z::AbstractVector, F::SupernodalLU)
    (; xsup, sn, Up, Ui, Ux) = F
    @inbounds for s in nsuper(F):-1:1
        f = xsup[s]
        l = xsup[s + 1] - 1
        nc = l - f + 1
        nr = sn.rowptr[s + 1] - sn.rowptr[s]
        V = _vals(sn, s)
        for a in nc:-1:1
            off = (a - 1) * nr
            za = z[f + a - 1] / V[off + a]
            z[f + a - 1] = za
            iszero(za) && continue
            for b in 1:(a - 1)
                z[f + b - 1] -= V[off + b] * za
            end
        end
        for j in l:-1:f
            zj = z[j]
            iszero(zj) && continue
            for p in Up[j]:(Up[j + 1] - 1)
                z[Ui[p]] -= Ux[p] * zj
            end
        end
    end
    return z
end

# z := Uᵀ \ z in pivot positions, with `f` applied to the entries of U.
function _utsolve!(z::AbstractVector, F::SupernodalLU, f::Fn) where {Fn}
    (; xsup, sn, Up, Ui, Ux) = F
    @inbounds for s in 1:nsuper(F)
        fs = xsup[s]
        l = xsup[s + 1] - 1
        nr = sn.rowptr[s + 1] - sn.rowptr[s]
        V = _vals(sn, s)
        for j in fs:l
            acc = z[j]
            for p in Up[j]:(Up[j + 1] - 1)
                acc -= f(Ux[p]) * z[Ui[p]]
            end
            a = j - fs + 1
            off = (a - 1) * nr
            for b in 1:(a - 1)
                acc -= f(V[off + b]) * z[fs + b - 1]
            end
            z[j] = acc / f(V[off + a])
        end
    end
    return z
end

# w := Lᵀ \ w in the row space of B, with `f` applied to the entries of L.
function _ltsolve!(w::AbstractVector, F::SupernodalLU, f::Fn) where {Fn}
    (; xsup, sn) = F
    @inbounds for s in nsuper(F):-1:1
        rows = _rows(sn, s)
        V = _vals(sn, s)
        nr = length(rows)
        nc = xsup[s + 1] - xsup[s]
        for a in nc:-1:1
            off = (a - 1) * nr
            acc = w[rows[a]]
            for b in (a + 1):nr
                acc -= f(V[off + b]) * w[rows[b]]
            end
            w[rows[a]] = acc
        end
    end
    return w
end

# Workspaces with the element type of the right-hand side: the factorization's own when
# it matches, fresh ones otherwise.
function _workspace(F::SupernodalLU{Tv}, ::Type{T}) where {Tv, T}
    T === Tv && return F.work
    return ntuple(_ -> Vector{T}(undef, F.n), 4)
end

# b := op(A) \ b, through B = (Dr·A·Dc)[rowperm, :].
function _ldiv_vec!(F::SupernodalLU, b::AbstractVector, op::Op, ws) where {Op}
    w, z, seg, tbuf = ws
    (; n, pivrow, rowperm, rscale, cscale) = F
    q = F.qfac
    if op === identity
        @inbounds for i in 1:n
            r = rowperm[i]
            w[i] = rscale[r] * b[r]
        end
        for s in 1:nsuper(F)
            _lsolve_super!(w, F, s, seg, tbuf)
        end
        @inbounds for k in 1:n
            z[k] = w[pivrow[k]]
        end
        _usolve!(z, F)
        @inbounds for j in 1:n
            c = q[j]
            b[c] = cscale[c] * z[j]
        end
    else
        fn = op === adjoint ? conj : identity
        @inbounds for k in 1:n
            c = q[k]
            z[k] = cscale[c] * b[c]
        end
        _utsolve!(z, F, fn)
        @inbounds for k in 1:n
            w[pivrow[k]] = z[k]
        end
        _ltsolve!(w, F, fn)
        @inbounds for i in 1:n
            r = rowperm[i]
            b[r] = rscale[r] * w[i]
        end
    end
    return b
end

function _ldiv!(F::SupernodalLU, B::AbstractVecOrMat{T}, op::Op) where {T, Op}
    F.zeropiv == 0 || throw(SingularException(F.zeropiv))
    ws = _workspace(F, T)
    B isa AbstractVector && return _ldiv_vec!(F, B, op, ws)
    for col in axes(B, 2)
        _ldiv_vec!(F, view(B, :, col), op, ws)
    end
    return B
end
