# This file is a part of Julia. License is MIT: https://julialang.org/license

# Construction of and conversion to `SparseMatrixCSC`.

# converting between SparseMatrixCSC types
SparseMatrixCSC(S::AbstractSparseMatrixCSC) = copy(S)
AbstractMatrix{Tv}(A::AbstractSparseMatrixCSC) where {Tv} = SparseMatrixCSC{Tv}(A)
SparseMatrixCSC{Tv}(S::AbstractSparseMatrixCSC{Tv}) where {Tv} = copy(S)
SparseMatrixCSC{Tv}(S::AbstractSparseMatrixCSC) where {Tv} = SparseMatrixCSC{Tv,eltype(getcolptr(S))}(S)
SparseMatrixCSC{Tv,Ti}(S::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = copy(S)
function SparseMatrixCSC{Tv,Ti}(S::AbstractSparseMatrixCSC) where {Tv,Ti}
    eltypeTicolptr = Vector{Ti}(getcolptr(S))
    eltypeTirowval = Vector{Ti}(rowvals(S))
    eltypeTvnzval = Vector{Tv}(nonzeros(S))
    return SparseMatrixCSC(size(S, 1), size(S, 2), eltypeTicolptr, eltypeTirowval, eltypeTvnzval)
end

# converting from other matrix types to SparseMatrixCSC (also see sparse())
SparseMatrixCSC(M::Matrix) = sparse(M)
SparseMatrixCSC(T::Tridiagonal{Tv}) where Tv = SparseMatrixCSC{Tv,Int}(T)
function SparseMatrixCSC{Tv,Ti}(T::Tridiagonal) where {Tv,Ti}
    m = length(T.d)
    m == 0 && return SparseMatrixCSC{Tv,Ti}(0, 0, ones(Ti, 1), Ti[], Tv[])
    m == 1 && return SparseMatrixCSC{Tv,Ti}(1, 1, Ti[1, 2], Ti[1], Tv[T.d[1]])

    colptr = Vector{Ti}(undef, m+1)
    colptr[1] = 1
    @inbounds for i=1:m-1
        colptr[i+1] = 3i
    end
    colptr[end] = 3m-1

    rowval = Vector{Ti}(undef, 3m-2)
    rowval[1] = 1
    rowval[2] = 2
    @inbounds for i=2:m-1, j=-1:1
        rowval[3i+j-2] = i+j
    end
    rowval[end-1] = m - 1
    rowval[end] = m

    nzval = Vector{Tv}(undef, 3m-2)
    @inbounds for i=1:(m-1)
        nzval[3i-2] = T.d[i]
        nzval[3i-1] = T.dl[i]
        nzval[3i]   = T.du[i]
    end
    nzval[end] = T.d[end]

    return SparseMatrixCSC(m, m, colptr, rowval, nzval)
end
SparseMatrixCSC(T::SymTridiagonal{Tv}) where Tv = SparseMatrixCSC{Tv,Int}(T)
function SparseMatrixCSC{Tv,Ti}(T::SymTridiagonal) where {Tv,Ti}
    m = length(T.dv)
    m == 0 && return SparseMatrixCSC{Tv,Ti}(0, 0, ones(Ti, 1), Ti[], Tv[])
    m == 1 && return SparseMatrixCSC{Tv,Ti}(1, 1, Ti[1, 2], Ti[1], Tv[T.dv[1]])

    colptr = Vector{Ti}(undef, m+1)
    colptr[1] = 1
    @inbounds for i=1:m-1
        colptr[i+1] = 3i
    end
    colptr[end] = 3m-1

    rowval = Vector{Ti}(undef, 3m-2)
    rowval[1] = 1
    rowval[2] = 2
    @inbounds for i=2:m-1, j=-1:1
        rowval[3i+j-2] = i+j
    end
    rowval[end-1] = m - 1
    rowval[end] = m

    nzval = Vector{Tv}(undef, 3m-2)
    @inbounds for i=1:(m-1)
        nzval[3i-2] = T.dv[i]
        nzval[3i-1] = T.ev[i]
        nzval[3i]   = T.ev[i]
    end
    nzval[end] = T.dv[end]

    return SparseMatrixCSC(m, m, colptr, rowval, nzval)
end
SparseMatrixCSC(B::Bidiagonal{Tv}) where Tv = SparseMatrixCSC{Tv,Int}(B)
function SparseMatrixCSC{Tv,Ti}(B::Bidiagonal) where {Tv,Ti}
    m = length(B.dv)
    m == 0 && return SparseMatrixCSC{Tv,Ti}(0, 0, ones(Ti, 1), Ti[], Tv[])

    colptr = Vector{Ti}(undef, m+1)
    colptr[1] = 1
    @inbounds for i=1:m-1
        colptr[i+1] = B.uplo == 'U' ? 2i : 2i+1
    end
    colptr[end] = 2m

    rowval = Vector{Ti}(undef, 2m-1)
    @inbounds for i=1:m-1
        rowval[2i-1] = i
        rowval[2i]   = B.uplo == 'U' ? i : i+1
    end
    rowval[end] = m

    nzval = Vector{Tv}(undef, 2m-1)
    nzval[1] = B.dv[1]
    @inbounds for i=1:m-1
        nzval[2i-1] = B.dv[i]
        nzval[2i]   = B.ev[i]
    end
    nzval[end] = B.dv[end]

    return SparseMatrixCSC(m, m, colptr, rowval, nzval)
end
SparseMatrixCSC(D::Diagonal{Tv}) where Tv = SparseMatrixCSC{Tv,Int}(D)
function SparseMatrixCSC{Tv,Ti}(D::Diagonal) where {Tv,Ti}
    m = length(D.diag)
    m == 0 && return SparseMatrixCSC{Tv,Ti}(zeros(Tv, 0, 0))

    nz = count(_isnotzero, D.diag)
    nz_counter = 1

    rowval = Vector{Ti}(undef, nz)
    nzval =  Vector{Tv}(undef, nz)

    nz == 0 && return SparseMatrixCSC{Tv,Ti}(m, m, ones(Ti, m+1), rowval, nzval)

    colptr = Vector{Ti}(undef, m+1)

    @inbounds for i=1:m
        if _isnotzero(D.diag[i])
            colptr[i] = nz_counter
            rowval[nz_counter] = i
            nzval[nz_counter]  = D.diag[i]
            nz_counter += 1
        else
            colptr[i] = nz_counter
        end
    end
    colptr[end] = nz_counter

    return SparseMatrixCSC{Tv,Ti}(m, m, colptr, rowval, nzval)
end

SparseMatrixCSC(M::AbstractMatrix{Tv}) where {Tv} = SparseMatrixCSC{Tv,Int}(M)
SparseMatrixCSC{Tv}(M::AbstractMatrix) where {Tv} = SparseMatrixCSC{Tv,Int}(M)
function SparseMatrixCSC{Tv,Ti}(M::AbstractMatrix) where {Tv,Ti}
    require_one_based_indexing(M)
    I = Ti[]
    V = Tv[]
    i = 0
    for v in M
        i += 1
        if _isnotzero(v)
            push!(I, i)
            push!(V, v)
        end
    end
    return sparse_sortedlinearindices!(I, V, size(M)...)
end

function SparseMatrixCSC{Tv,Ti}(M::StridedMatrix) where {Tv,Ti}
    nz = count(_isnotzero, M)
    colptr = zeros(Ti, size(M, 2) + 1)
    nzval = Vector{Tv}(undef, nz)
    rowval = Vector{Ti}(undef, nz)
    colptr[1] = 1
    cnt = 1
    @inbounds for j in axes(M, 2)
        for i in axes(M, 1)
            v = M[i, j]
            if _isnotzero(v)
                rowval[cnt] = i
                nzval[cnt] = v
                cnt += 1
            end
        end
        colptr[j+1] = cnt
    end
    return SparseMatrixCSC(size(M, 1), size(M, 2), colptr, rowval, nzval)
end
SparseMatrixCSC(M::Adjoint{<:Any,<:AbstractSparseMatrixCSC}) = copy(M)
SparseMatrixCSC(M::Transpose{<:Any,<:AbstractSparseMatrixCSC}) = copy(M)
SparseMatrixCSC{Tv}(M::Adjoint{Tv,<:AbstractSparseMatrixCSC{Tv}}) where {Tv} = copy(M)
SparseMatrixCSC{Tv}(M::Transpose{Tv,<:AbstractSparseMatrixCSC{Tv}}) where {Tv} = copy(M)
SparseMatrixCSC{Tv,Ti}(M::Adjoint{Tv,<:AbstractSparseMatrixCSC{Tv,Ti}}) where {Tv,Ti} = copy(M)
SparseMatrixCSC{Tv,Ti}(M::Transpose{Tv,<:AbstractSparseMatrixCSC{Tv,Ti}}) where {Tv,Ti} = copy(M)

# converting from adjoint or transpose sparse matrices to sparse matrices with different eltype
SparseMatrixCSC{Tv}(M::Adjoint{<:Any,<:AbstractSparseMatrixCSC}) where {Tv} = SparseMatrixCSC{Tv}(copy(M))
SparseMatrixCSC{Tv}(M::Transpose{<:Any,<:AbstractSparseMatrixCSC}) where {Tv} = SparseMatrixCSC{Tv}(copy(M))
SparseMatrixCSC{Tv,Ti}(M::Adjoint{<:Any,<:AbstractSparseMatrixCSC}) where {Tv,Ti} = SparseMatrixCSC{Tv,Ti}(copy(M))
SparseMatrixCSC{Tv,Ti}(M::Transpose{<:Any,<:AbstractSparseMatrixCSC}) where {Tv,Ti} = SparseMatrixCSC{Tv,Ti}(copy(M))

# we can only view AbstractQs as columns
SparseMatrixCSC(Q::AbstractQ{Tv}) where {Tv} = SparseMatrixCSC{Tv,Int}(Q)
SparseMatrixCSC{Tv}(Q::AbstractQ{Tv}) where {Tv} = SparseMatrixCSC{Tv,Int}(Q)
SparseMatrixCSC{Tv,Ti}(Q::AbstractQ) where {Tv,Ti} = sparse_with_lmul(Tv, Ti, Q)

"""
    sparse_with_lmul(Tv, Ti, Q) -> SparseMatrixCSC

Helper function that creates a `SparseMatrixCSC{Tv,Ti}` representation of `Q`, where `Q` is
supposed to not have fast `getindex` or not admit an iteration protocol at all, but instead
a fast `lmul!(Q, v)` for dense vectors `v`. The prime example for such `Q`s is the Q factor
of a (sparse) QR decomposition.
"""
function sparse_with_lmul(Tv, Ti, Q)
    colptr = zeros(Ti, size(Q, 2) + 1)
    nzval = Tv[]
    rowval = Ti[]
    col = zeros(eltype(Q), size(Q, 1))

    colptr[1] = 1
    ind = 1
    for j in axes(Q, 2)
        fill!(col, false)
        col[j] = one(Tv)
        lmul!(Q, col)
        for (i, v) in enumerate(col)
            if _isnotzero(v)
                push!(nzval, v)
                push!(rowval, i)
                ind += 1
            end
        end
        colptr[j + 1] = ind
    end
    return SparseMatrixCSC{Tv,Ti}(size(Q)..., colptr, rowval, nzval)
end

convert(T::Type{<:AbstractSparseMatrixCSC}, m::AbstractMatrix) = m isa T ? m : T(m)

# mirror Base's Array rule: promote the eltype only if at least one container wouldn't
# change, otherwise join the container types (see Base.el_same)
function promote_rule(::Type{SparseMatrixCSC{Tv1,Ti1}}, ::Type{SparseMatrixCSC{Tv2,Ti2}}) where {Tv1,Ti1,Tv2,Ti2}
    Ti = promote_type(Ti1, Ti2)
    return Base.el_same(promote_type(Tv1, Tv2), SparseMatrixCSC{Tv1,Ti}, SparseMatrixCSC{Tv2,Ti})
end
promote_rule(::Type{Matrix{Tv1}}, ::Type{<:SparseMatrixCSC{Tv2}}) where {Tv1,Tv2} =
    Base.el_same(promote_type(Tv1, Tv2), Matrix{Tv1}, Matrix{Tv2})

convert(T::Type{<:Diagonal},       m::AbstractSparseMatrixCSC) = m isa T ? m :
    isdiag(m) ? T(m) : throw(ArgumentError("matrix cannot be represented as Diagonal"))
convert(T::Type{<:SymTridiagonal}, m::AbstractSparseMatrixCSC) = m isa T ? m :
    issymmetric(m) && isbanded(m, -1, 1) ? T(m) : throw(ArgumentError("matrix cannot be represented as SymTridiagonal"))
convert(T::Type{<:Tridiagonal},    m::AbstractSparseMatrixCSC) = m isa T ? m :
    isbanded(m, -1, 1) ? T(m) : throw(ArgumentError("matrix cannot be represented as Tridiagonal"))
convert(T::Type{<:LowerTriangular}, m::AbstractSparseMatrixCSC) = m isa T ? m :
    istril(m) ? T(m) : throw(ArgumentError("matrix cannot be represented as LowerTriangular"))
convert(T::Type{<:UpperTriangular}, m::AbstractSparseMatrixCSC) = m isa T ? m :
    istriu(m) ? T(m) : throw(ArgumentError("matrix cannot be represented as UpperTriangular"))

float(S::SparseMatrixCSC) = SparseMatrixCSC(size(S, 1), size(S, 2), getcolptr(S), rowvals(S), float(nonzeros(S)))
complex(S::SparseMatrixCSC) = SparseMatrixCSC(size(S, 1), size(S, 2), getcolptr(S), rowvals(S), complex(nonzeros(S)))

"""
    sparse(A::Union{AbstractVector, AbstractMatrix})

Convert a vector or matrix `A` into a sparse array.
Numerical zeros in `A` are turned into structural zeros.

# Examples
```jldoctest
julia> A = Matrix(1.0I, 3, 3)
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0

julia> sparse(A)
3×3 SparseMatrixCSC{Float64, Int64} with 3 stored entries:
 1.0   ⋅    ⋅
  ⋅   1.0   ⋅
  ⋅    ⋅   1.0

julia> [1.0, 0.0, 1.0]
3-element Vector{Float64}:
 1.0
 0.0
 1.0

julia> sparse([1.0, 0.0, 1.0])
3-element SparseVector{Float64, Int64} with 2 stored entries:
  [1]  =  1.0
  [3]  =  1.0
```
"""
sparse(A::AbstractMatrix{Tv}) where {Tv} = convert(SparseMatrixCSC{Tv}, A)

sparse(S::AbstractSparseMatrixCSC) = copy(S)

sparse(Q::AbstractQ) = SparseMatrixCSC(Q)

sparse(T::SymTridiagonal) = SparseMatrixCSC(T)

sparse(T::Tridiagonal) = SparseMatrixCSC(T)

sparse(B::Bidiagonal) = SparseMatrixCSC(B)

sparse(D::Diagonal) = SparseMatrixCSC(D)

"""
    sparse(I, J, V,[ m, n, combine])

Create a sparse matrix `S` of dimensions `m x n` such that `S[I[k], J[k]] = V[k]`. The
`combine` function is used to combine duplicates. If `m` and `n` are not specified, they
are set to `maximum(I)` and `maximum(J)` respectively. If the `combine` function is not
supplied, `combine` defaults to `+` unless the elements of `V` are Booleans in which case
`combine` defaults to `|`. All elements of `I` must satisfy `1 <= I[k] <= m`, and all
elements of `J` must satisfy `1 <= J[k] <= n`. Numerical zeros in (`I`, `J`, `V`) are
retained as structural nonzeros; to drop numerical zeros, use [`dropzeros!`](@ref).

For additional documentation and an expert driver, see `SparseArrays.sparse!`.

# Examples
```jldoctest
julia> Is = [1; 2; 3];

julia> Js = [1; 2; 3];

julia> Vs = [1; 2; 3];

julia> sparse(Is, Js, Vs)
3×3 SparseMatrixCSC{Int64, Int64} with 3 stored entries:
 1  ⋅  ⋅
 ⋅  2  ⋅
 ⋅  ⋅  3
```
"""
function sparse(I::AbstractVector{Ti}, J::AbstractVector{Ti}, V::AbstractVector{Tv}, m::Integer, n::Integer, combine::F) where {Tv,Ti<:Integer,F}
    require_one_based_indexing(I, J, V)
    coolen = length(I)
    if length(J) != coolen || length(V) != coolen
        throw(ArgumentError(string("the first three arguments' lengths must match, ",
              "length(I) (=$(length(I))) == length(J) (= $(length(J))) == length(V) (= ",
              "$(length(V)))")))
    end
    if Base.hastypemax(Ti) && coolen >= typemax(Ti)
        throw(ArgumentError("the index type $Ti cannot hold $coolen elements; use a larger index type"))
    end
    if m == 0 || n == 0 || coolen == 0
        if coolen != 0
            if n == 0
                throw(ArgumentError("column indices J[k] must satisfy 1 <= J[k] <= n"))
            elseif m == 0
                throw(ArgumentError("row indices I[k] must satisfy 1 <= I[k] <= m"))
            end
        end
        SparseMatrixCSC(m, n, fill(one(Ti), n+1), Vector{Ti}(), Vector{Tv}())
    else
        # Allocate storage for CSR form
        csrrowptr = Vector{Ti}(undef, m+1)
        csrcolval = Vector{Ti}(undef, coolen)
        csrnzval = Vector{Tv}(undef, coolen)

        # Allocate storage for the CSC form's column pointers and a necessary workspace
        csccolptr = Vector{Ti}(undef, n+1)
        klasttouch = Vector{Ti}(undef, n)

        # Allocate empty arrays for the CSC form's row and nonzero value arrays
        # The parent method called below automagically resizes these arrays
        cscrowval = Vector{Ti}()
        cscnzval = Vector{Tv}()

        sparse!(I, J, V, m, n, combine, klasttouch,
                csrrowptr, csrcolval, csrnzval,
                csccolptr, cscrowval, cscnzval)
    end
end

sparse(I::AbstractVector, J::AbstractVector, V::AbstractVector, m::Integer, n::Integer, combine::F) where {F} =
    sparse(AbstractVector{Int}(I), AbstractVector{Int}(J), V, m, n, combine)

"""
    sparse!(I::AbstractVector{Ti}, J::AbstractVector{Ti}, V::AbstractVector{Tv},
            m::Integer, n::Integer, combine, klasttouch::Vector{Ti},
            csrrowptr::Vector{Ti}, csrcolval::Vector{Ti}, csrnzval::Vector{Tv},
            [csccolptr::Vector{Ti}], [cscrowval::Vector{Ti}, cscnzval::Vector{Tv}] ) where {Tv,Ti<:Integer}

Parent of and expert driver for [`sparse`](@ref);
see [`sparse`](@ref) for basic usage. This method
allows the user to provide preallocated storage for `sparse`'s intermediate objects and
result as described below. This capability enables more efficient successive construction
of [`SparseMatrixCSC`](@ref)s from coordinate representations, and also enables extraction
of an unsorted-column representation of the result's transpose at no additional cost.

This method consists of three major steps: (1) Counting-sort the provided coordinate
representation into an unsorted-row CSR form including repeated entries. (2) Sweep through
the CSR form, simultaneously calculating the desired CSC form's column-pointer array,
detecting repeated entries, and repacking the CSR form with repeated entries combined;
this stage yields an unsorted-row CSR form with no repeated entries. (3) Counting-sort the
preceding CSR form into a fully-sorted CSC form with no repeated entries.

Input arrays `csrrowptr`, `csrcolval`, and `csrnzval` constitute storage for the
intermediate CSR forms and require `length(csrrowptr) >= m + 1`,
`length(csrcolval) >= length(I)`, and `length(csrnzval >= length(I))`. Input
array `klasttouch`, workspace for the second stage, requires `length(klasttouch) >= n`.
Optional input arrays `csccolptr`, `cscrowval`, and `cscnzval` constitute storage for the
returned CSC form `S`. If necessary, these are resized automatically to satisfy
`length(csccolptr) = n + 1`, `length(cscrowval) = nnz(S)` and `length(cscnzval) = nnz(S)`; hence, if `nnz(S)` is
unknown at the outset, passing in empty vectors of the appropriate type (`Vector{Ti}()`
and `Vector{Tv}()` respectively) suffices, or calling the `sparse!` method
neglecting `cscrowval` and `cscnzval`.

On return, `csrrowptr`, `csrcolval`, and `csrnzval` contain an unsorted-column
representation of the result's transpose.

You may reuse the input arrays' storage (`I`, `J`, `V`) for the output arrays
(`csccolptr`, `cscrowval`, `cscnzval`). For example, you may call
`sparse!(I, J, V, csrrowptr, csrcolval, csrnzval, I, J, V)`.
Note that they will be resized to satisfy the conditions above.

For the sake of efficiency, this method performs no argument checking beyond
`1 <= I[k] <= m` and `1 <= J[k] <= n`. Use with care. Testing with `--check-bounds=yes`
is wise.

This method runs in `O(m, n, length(I))` time. The HALFPERM algorithm described in
F. Gustavson, "Two fast algorithms for sparse matrices: multiplication and permuted
transposition," ACM TOMS 4(3), 250-269 (1978) inspired this method's use of a pair of
counting sorts.
"""
function sparse!(I::AbstractVector{Ti}, J::AbstractVector{Ti}, V::AbstractVector{Tv},
        m::Integer, n::Integer, combine, klasttouch::Vector{Tj},
        csrrowptr::Vector{Tj}, csrcolval::Vector{Ti}, csrnzval::Vector{Tv},
        csccolptr::Vector{Ti}, cscrowval::Vector{Ti}, cscnzval::Vector{Tv}) where {Tv,Ti<:Integer,Tj<:Integer}

    require_one_based_indexing(I, J, V)
    sparse_check_Ti(m, n, Ti)
    sparse_check_length("I", I, 0, Tj)

    # This method is also used internally by spzeros! to build the sparsity pattern without
    # caring about the values. This is communicated by passing combine=nothing and in this
    # case V and csrnzval should *not* be accessed. When called from spzeros! they will both
    # alias cscnzval, which will be resized and filled with zero(Tv).
    only_sparsity_pattern = combine === nothing

    # Compute the CSR form's row counts and store them shifted forward by one in csrrowptr
    fill!(csrrowptr, Tj(0))
    coolen = length(I)
    length(J) >= coolen || throw(ArgumentError("J need length >= length(I) = $coolen"))
    only_sparsity_pattern || length(V) >= coolen || throw(ArgumentError("V need length >= length(I) = $coolen"))

    @inbounds for k in 1:coolen
        Ik = I[k]
        if 1 > Ik || m < Ik
            throw(ArgumentError("row indices I[k] must satisfy 1 <= I[k] <= m"))
        end
        csrrowptr[Ik+1] += Tj(1)
    end

    # Compute the CSR form's rowptrs and store them shifted forward by one in csrrowptr
    countsum = Tj(1)
    csrrowptr[1] = Tj(1)
    @inbounds for i in 2:(m+1)
        overwritten = csrrowptr[i]
        csrrowptr[i] = countsum
        countsum += overwritten
    end

    # Counting-sort the column and nonzero values from J and V into csrcolval and csrnzval
    # Tracking write positions in csrrowptr corrects the row pointers
    @inbounds for k in 1:coolen
        Ik, Jk = I[k], J[k]
        if Ti(1) > Jk || Ti(n) < Jk
            throw(ArgumentError("column indices J[k] must satisfy 1 <= J[k] <= n"))
        end
        csrk = csrrowptr[Ik+1]
        @assert csrk >= Tj(1) "index into csrcolval exceeds typemax(Ti)"
        csrrowptr[Ik+1] = csrk + Tj(1)
        csrcolval[csrk] = Jk
        if !only_sparsity_pattern
            csrnzval[csrk] = V[k]
        end
    end
    # This completes the unsorted-row, has-repeats CSR form's construction

    # The output array csccolptr can now be resized safely even if aliased with I
    resize!(csccolptr, n + 1)

    # Sweep through the CSR form, simultaneously (1) calculating the CSC form's column
    # counts and storing them shifted forward by one in csccolptr; (2) detecting repeated
    # entries; and (3) repacking the CSR form with the repeated entries combined.
    #
    # Minimizing extraneous communication and nonlocality of reference, primarily by using
    # only a single auxiliary array in this step, is the key to this method's performance.
    fill!(csccolptr, Ti(0))
    fill!(klasttouch, Tj(0))
    writek = Tj(1)
    newcsrrowptri = Ti(1)
    origcsrrowptri = Tj(1)
    origcsrrowptrip1 = csrrowptr[2]
    @inbounds for i in 1:m
        for readk in origcsrrowptri:(origcsrrowptrip1-Tj(1))
            j = csrcolval[readk]
            if klasttouch[j] < newcsrrowptri
                klasttouch[j] = writek
                if writek != readk
                    csrcolval[writek] = j
                    if !only_sparsity_pattern
                        csrnzval[writek] = csrnzval[readk]
                    end
                end
                writek += Tj(1)
                csccolptr[j+1] += Ti(1)
            elseif !only_sparsity_pattern
                klt = klasttouch[j]
                csrnzval[klt] = combine(csrnzval[klt], csrnzval[readk])
            end
        end
        newcsrrowptri = writek
        origcsrrowptri = origcsrrowptrip1
        origcsrrowptrip1 != writek && (csrrowptr[i+1] = writek)
        i < m && (origcsrrowptrip1 = csrrowptr[i+2])
    end

    # Compute the CSC form's colptrs and store them shifted forward by one in csccolptr
    countsum = Tj(1)
    csccolptr[1] = Ti(1)
    @inbounds for j in 2:(n+1)
        overwritten = csccolptr[j]
        csccolptr[j] = countsum
        countsum += overwritten
        Base.hastypemax(Ti) && (countsum <= typemax(Ti) || throw(ArgumentError("more than typemax(Ti)-1 == $(typemax(Ti)-1) entries")))
    end

    # Now knowing the CSC form's entry count, resize cscrowval and cscnzval
    # Note: This is done unconditionally to appease the buffer checks in the SparseMatrixCSC
    #       constructor. If these checks are lifted this resizing is only needed if the
    #       buffers are too short. csccolptr is resized above.
    cscnnz = countsum - Tj(1)
    resize!(cscrowval, cscnnz)
    resize!(cscnzval, cscnnz)

    # Finally counting-sort the row and nonzero values from the CSR form into cscrowval and
    # cscnzval. Tracking write positions in csccolptr corrects the column pointers.
    @inbounds for i in 1:m
        for csrk in csrrowptr[i]:(csrrowptr[i+1]-Tj(1))
            j = csrcolval[csrk]
            csck = csccolptr[j+1]
            csccolptr[j+1] = csck + Ti(1)
            cscrowval[csck] = i
            cscnzval[csck] = only_sparsity_pattern ? zero(Tv) : csrnzval[csrk]
        end
    end

    SparseMatrixCSC(m, n, csccolptr, cscrowval, cscnzval)
end
function sparse!(I::AbstractVector{Ti}, J::AbstractVector{Ti},
        V::AbstractVector{Tv}, m::Integer, n::Integer, combine, klasttouch::Vector{Tj},
        csrrowptr::Vector{Tj}, csrcolval::Vector{Ti}, csrnzval::Vector{Tv},
        csccolptr::Vector{Ti}) where {Tv,Ti<:Integer,Tj<:Integer}
    sparse!(I, J, V, m, n, combine, klasttouch,
            csrrowptr, csrcolval, csrnzval,
            csccolptr, Vector{Ti}(), Vector{Tv}())
end
function sparse!(I::AbstractVector{Ti}, J::AbstractVector{Ti},
        V::AbstractVector{Tv}, m::Integer, n::Integer, combine, klasttouch::Vector{Tj},
        csrrowptr::Vector{Tj}, csrcolval::Vector{Ti}, csrnzval::Vector{Tv}) where {Tv,Ti<:Integer,Tj<:Integer}
    sparse!(I, J, V, m, n, combine, klasttouch,
            csrrowptr, csrcolval, csrnzval,
            Vector{Ti}(undef, n+1), Vector{Ti}(), Vector{Tv}())
end

"""
    SparseArrays.sparse!(I, J, V, [m, n, combine]) -> SparseMatrixCSC

Variant of `sparse!` that re-uses the input vectors (`I`, `J`, `V`) for the final matrix
storage. After construction the input vectors will alias the matrix buffers; `S.colptr ===
I`, `S.rowval === J`, and `S.nzval === V` holds, and they will be `resize!`d as necessary.

Note that some work buffers will still be allocated. Specifically, this method is a
convenience wrapper around `sparse!(I, J, V, m, n, combine, klasttouch, csrrowptr,
csrcolval, csrnzval, csccolptr, cscrowval, cscnzval)` where this method allocates
`klasttouch`, `csrrowptr`, `csrcolval`, and `csrnzval` of appropriate size, but reuses `I`,
`J`, and `V` for `csccolptr`, `cscrowval`, and `cscnzval`.

Arguments `m`, `n`, and `combine` defaults to `maximum(I)`, `maximum(J)`, and `+`,
respectively.

!!! compat "Julia 1.10"
    This method requires Julia version 1.10 or later.
"""
function sparse!(I::AbstractVector{Ti}, J::AbstractVector{Ti}, V::AbstractVector{Tv},
                 m::Integer=dimlub(I), n::Integer=dimlub(J), combine::Function=+) where {Tv, Ti<:Integer}
    klasttouch = Vector{Ti}(undef, n)
    csrrowptr  = Vector{Ti}(undef, m + 1)
    csrcolval  = Vector{Ti}(undef, length(I))
    csrnzval   = Vector{Tv}(undef, length(I))
    sparse!(I, J, V, Int(m), Int(n), combine, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)
end

dimlub(I) = isempty(I) ? 0 : Int(maximum(I)) #least upper bound on required sparse matrix dimension

sparse(I,J,v::Number) = sparse(I, J, fill(v,length(I)))

sparse(I,J,V::AbstractVector) = sparse(I, J, V, dimlub(I), dimlub(J))

sparse(I,J,v::Number,m,n) = sparse(I, J, fill(v,length(I)), Int(m), Int(n))

sparse(I,J,V::AbstractVector,m,n) = sparse(I, J, V, Int(m), Int(n), +)

sparse(I,J,V::AbstractVector{Bool},m,n) = sparse(I, J, V, Int(m), Int(n), |)

sparse(I,J,v::Number,m,n,combine::Function) = sparse(I, J, fill(v,length(I)), Int(m), Int(n), combine)

function sparse_sortedlinearindices!(I::Vector{Ti}, V::Vector, m::Int, n::Int) where Ti
    length(I) == length(V) || throw(ArgumentError("I and V should have the same length"))
    nnz = length(V)
    colptr = Vector{Ti}(undef, n + 1)
    j, colm = 1, 0
    @inbounds for col = 1:n+1
        colptr[col] = j
        while j <= nnz && (I[j] -= colm) <= m
            j += 1
        end
        j <= nnz && (I[j] += colm)
        colm += m
    end
    return SparseMatrixCSC(m, n, colptr, I, V)
end

"""
    sprand([rng],[T::Type],m,[n],p::AbstractFloat)
    sprand([rng],m,[n],p::AbstractFloat,[rfn=rand])

Create a random length `m` sparse vector or `m` by `n` sparse matrix, in
which the probability of any element being nonzero is independently given by
`p` (and hence the mean density of nonzeros is also exactly `p`).
The optional `rng` argument specifies a random number generator, see [Random Numbers](@ref).
The optional `T` argument specifies the element type, which defaults to `Float64`.

By default, nonzero values are sampled from a uniform distribution using
the [`rand`](@ref) function, i.e. by `rand(T)`, or `rand(rng, T)` if `rng`
is supplied; for the default `T=Float64`, this corresponds to nonzero values
sampled uniformly in `[0,1)`.

You can sample nonzero values from a different distribution by passing a
custom `rfn` function instead of `rand`.   This should be a function `rfn(k)`
that returns an array of `k` random numbers sampled from the desired distribution;
alternatively, if `rng` is supplied, it should instead be a function `rfn(rng, k)`.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(1234))
julia> sprand(Bool, 2, 2, 0.5)
2×2 SparseMatrixCSC{Bool, Int64} with 2 stored entries:
 1  1
 ⋅  ⋅

julia> sprand(Float64, 3, 0.75)
3-element SparseVector{Float64, Int64} with 2 stored entries:
  [1]  =  0.795547
  [2]  =  0.49425
```
"""
function sprand(r::AbstractRNG, m::Integer, n::Integer, density::AbstractFloat, rfn::Function, ::Type{T}=eltype(rfn(r, 1))) where T
    m, n = Int(m), Int(n)
    (m < 0 || n < 0) && throw(ArgumentError("invalid Array dimensions"))
    0 <= density <= 1 || throw(ArgumentError("$density not in [0,1]"))
    I = randsubseq(r, 1:(m*n), density)
    return sparse_sortedlinearindices!(I, convert(Vector{T}, rfn(r,length(I))), m, n)
end

sprand(m::Integer, n::Integer, density::AbstractFloat, rfn::Function, ::Type{T} = eltype(rfn(1))) where {T} =
    sprand(default_rng(), m, n, density, (r, i) -> rfn(i))

truebools(r::AbstractRNG, n::Integer) = fill(true, n)

sprand(m::Integer, n::Integer, density::AbstractFloat) = sprand(default_rng(), m, n, density)

sprand(r::AbstractRNG, m::Integer, n::Integer, density::AbstractFloat) =
    sprand(r, m, n, density, rand, Float64)
sprand(r::AbstractRNG, ::Type{T}, m::Integer, n::Integer, density::AbstractFloat) where {T} =
    sprand(r, m, n, density, (r, i) -> rand(r, T, i), T)
sprand(r::AbstractRNG, ::Type{Bool}, m::Integer, n::Integer, density::AbstractFloat) =
    sprand(r, m, n, density, truebools, Bool)
sprand(::Type{T}, m::Integer, n::Integer, density::AbstractFloat) where {T} =
    sprand(default_rng(), T, m, n, density)

"""
    sprandn([rng][,Type],m[,n],p::AbstractFloat)

Create a random sparse vector of length `m` or sparse matrix of size `m` by `n`
with the specified (independent) probability `p` of any entry being nonzero,
where nonzero values are sampled from the normal distribution. The optional `rng`
argument specifies a random number generator, see [Random Numbers](@ref).

!!! compat "Julia 1.1"
    Specifying the output element type `Type` requires at least Julia 1.1.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> sprandn(2, 2, 0.75)
2×2 SparseMatrixCSC{Float64, Int64} with 3 stored entries:
 -1.20577     ⋅
  0.311817  -0.234641
```
"""
sprandn(r::AbstractRNG, m::Integer, n::Integer, density::AbstractFloat) =
    sprand(r, m, n, density, randn, Float64)
sprandn(m::Integer, n::Integer, density::AbstractFloat) =
    sprandn(default_rng(), m, n, density)
sprandn(r::AbstractRNG, ::Type{T}, m::Integer, n::Integer, density::AbstractFloat) where {T} =
    sprand(r, m, n, density, (r, i) -> randn(r, T, i), T)
sprandn(::Type{T}, m::Integer, n::Integer, density::AbstractFloat) where {T} =
    sprandn(default_rng(), T, m, n, density)

LinearAlgebra.fillstored!(S::AbstractSparseMatrixCSC, x) = (fill!(nzvalview(S), x); S)

"""
    spzeros([type,]m[,n])

Create a sparse vector of length `m` or sparse matrix of size `m x n`. This
sparse array will not contain any nonzero values, and no storage is allocated
for them. The type defaults to [`Float64`](@ref) if not specified.

This does not make the call allocation-free: the empty index and value buffers
are still allocated, and a matrix additionally allocates a column pointer of
`n + 1` entries, so an `m x n` matrix uses memory proportional to `n`.

# Examples
```jldoctest
julia> spzeros(3, 3)
3×3 SparseMatrixCSC{Float64, Int64} with 0 stored entries:
  ⋅    ⋅    ⋅
  ⋅    ⋅    ⋅
  ⋅    ⋅    ⋅

julia> spzeros(Float32, 4)
4-element SparseVector{Float32, Int64} with 0 stored entries
```
"""
spzeros(m::Integer, n::Integer) = spzeros(Float64, m, n)
spzeros(::Type{Tv}, m::Integer, n::Integer) where {Tv} = spzeros(Tv, Int, m, n)
function spzeros(::Type{Tv}, ::Type{Ti}, m::Integer, n::Integer) where {Tv, Ti}
    ((m < 0) || (n < 0)) && throw(ArgumentError("invalid Array dimensions"))
    SparseMatrixCSC(m, n, fill(one(Ti), n+1), Vector{Ti}(), Vector{Tv}())
end
# de-splatting variants
function spzeros(::Type{Tv}, ::Type{Ti}, sz::Tuple{Integer,Integer}) where {Tv, Ti}
    spzeros(Tv, Ti, sz[1], sz[2])
end
spzeros(::Type{Tv}, sz::Tuple{Integer,Integer}) where {Tv} = spzeros(Tv, Int, sz[1], sz[2])
spzeros(sz::Tuple{Integer,Integer}) = spzeros(Float64, Int, sz[1], sz[2])

"""
    spzeros([type], I::AbstractVector, J::AbstractVector, [m, n])

Create a sparse matrix `S` of dimensions `m x n` with structural zeros at `S[I[k], J[k]]`.

This method can be used to construct the sparsity pattern of the matrix, and is more
efficient than using e.g. `sparse(I, J, zeros(length(I)))`.

For additional documentation and an expert driver, see `SparseArrays.spzeros!`.

!!! compat "Julia 1.10"
    This methods requires Julia version 1.10 or later.
"""
spzeros(I::AbstractVector, J::AbstractVector) = spzeros(Float64, I, J)
spzeros(I::AbstractVector, J::AbstractVector, m::Integer, n::Integer) = spzeros(Float64, I, J, m, n)
spzeros(::Type{Tv}, I::AbstractVector, J::AbstractVector) where {Tv} = spzeros(Tv, I, J, dimlub(I), dimlub(J))
function spzeros(::Type{Tv}, I::AbstractVector, J::AbstractVector, m::Integer, n::Integer) where {Tv}
    return spzeros(Tv, AbstractVector{Int}(I), AbstractVector{Int}(J), m, n)
end
function spzeros(::Type{Tv}, I::AbstractVector{Ti}, J::AbstractVector{Ti}, m::Integer, n::Integer) where {Tv, Ti<:Integer}
    if length(I) != length(J)
        throw(ArgumentError("length(I) = $(length(I)) does not match length(J) = $(length(J))"))
    end
    klasttouch = Vector{Ti}(undef, n)
    csrrowptr = Vector{Ti}(undef, m+1)
    csrcolval = Vector{Ti}(undef, length(I))
    return spzeros!(Tv, I, J, m, n, klasttouch, csrrowptr, csrcolval)
end

"""
    spzeros!(::Type{Tv}, I::AbstractVector{Ti}, J::AbstractVector{Ti}, m::Integer, n::Integer,
             klasttouch::Vector{Ti}, csrrowptr::Vector{Ti}, csrcolval::Vector{Ti},
             [csccolptr::Vector{Ti}], [cscrowval::Vector{Ti}, cscnzval::Vector{Tv}]) where {Tv,Ti<:Integer}

Parent of and expert driver for `spzeros(I, J)` allowing user to provide preallocated
storage for intermediate objects. This method is to `spzeros` what `SparseArrays.sparse!` is
to `sparse`. See documentation for `SparseArrays.sparse!` for details and required buffer
lengths.

!!! compat "Julia 1.10"
    This methods requires Julia version 1.10 or later.
"""
function spzeros!(::Type{Tv}, I::AbstractVector{Ti}, J::AbstractVector{Ti}, m::Integer, n::Integer,
        klasttouch::Vector{Ti}, csrrowptr::Vector{Ti}, csrcolval::Vector{Ti},
        csccolptr::Vector{Ti}=Ti[], cscrowval::Vector{Ti}=Ti[], cscnzval::Vector{Tv}=Tv[]
    ) where {Tv, Ti<:Integer}
    # We can pass V = csrnzval = cscnzval since V and csrnzval are unused in sparse! if used
    # to only build the sparsity pattern (which is indicated by passing combine=nothing).
    return sparse!(I, J, cscnzval, m, n, nothing, klasttouch,
                   csrrowptr, csrcolval, cscnzval, csccolptr, cscrowval, cscnzval)
end

"""
    SparseArrays.spzeros!(::Type{Tv}, I, J, [m, n]) -> SparseMatrixCSC{Tv}

Variant of `spzeros!` that re-uses the input vectors `I` and `J` for the final matrix
storage. After construction the input vectors will alias the matrix buffers; `S.colptr ===
I` and `S.rowval === J` holds, and they will be `resize!`d as necessary.

Note that some work buffers will still be allocated. Specifically, this method is a
convenience wrapper around `spzeros!(Tv, I, J, m, n, klasttouch, csrrowptr, csrcolval,
csccolptr, cscrowval)` where this method allocates `klasttouch`, `csrrowptr`, and
`csrcolval` of appropriate size, but reuses `I` and `J` for `csccolptr` and `cscrowval`.

Arguments `m` and `n` defaults to `maximum(I)` and `maximum(J)`.

!!! compat "Julia 1.10"
    This method requires Julia version 1.10 or later.
"""
function spzeros!(::Type{Tv}, I::AbstractVector{Ti}, J::AbstractVector{Ti},
                  m::Integer=dimlub(I), n::Integer=dimlub(J)) where {Tv, Ti <: Integer}
    klasttouch = Vector{Ti}(undef, n)
    csrrowptr  = Vector{Ti}(undef, m + 1)
    csrcolval  = Vector{Ti}(undef, length(I))
    return spzeros!(Tv, I, J, Int(m), Int(n), klasttouch, csrrowptr, csrcolval, I, J)
end

import Base._one
function Base._one(unit::T, S::AbstractSparseMatrixCSC) where T
    size(S, 1) == size(S, 2) || throw(DimensionMismatch("multiplicative identity only defined for square matrices"))
    return _spscaling(T, Int, unit, size(S, 1), size(S, 2))
end

## SparseMatrixCSC construction from UniformScaling
SparseMatrixCSC{Tv,Ti}(s::UniformScaling, m::Integer, n::Integer) where {Tv,Ti} = SparseMatrixCSC{Tv,Ti}(s, Dims((m, n)))
SparseMatrixCSC{Tv}(s::UniformScaling, m::Integer, n::Integer) where {Tv} = SparseMatrixCSC{Tv}(s, Dims((m, n)))
SparseMatrixCSC(s::UniformScaling, m::Integer, n::Integer) = SparseMatrixCSC(s, Dims((m, n)))
SparseMatrixCSC{Tv}(s::UniformScaling, dims::Dims{2}) where {Tv} = SparseMatrixCSC{Tv,Int}(s, dims)
SparseMatrixCSC(s::UniformScaling, dims::Dims{2}) = SparseMatrixCSC{eltype(s)}(s, dims)
function SparseMatrixCSC{Tv,Ti}(s::UniformScaling, dims::Dims{2}) where {Tv,Ti}
    @boundscheck first(dims) < 0 && throw(ArgumentError("first dimension invalid ($(first(dims)) < 0)"))
    @boundscheck last(dims) < 0 && throw(ArgumentError("second dimension invalid ($(last(dims)) < 0)"))
    return _spscaling(Tv, Ti, s.λ, dims...)
end

function _spscaling(::Type{Tv}, ::Type{Ti}, λ, m, n) where {Tv,Ti<:Integer}
    iszero(λ) && return spzeros(Tv, Ti, m, n)
    k = min(m, n)
    nzval = fill!(Vector{Tv}(undef, k), Tv(λ))
    rowval = copyto!(Vector{Ti}(undef, k), 1:k)
    colptr = copyto!(Vector{Ti}(undef, n + 1), 1:(k + 1))
    for i in (k + 2):(n + 1)
        colptr[i] = (k + 1)
    end
    return SparseMatrixCSC{Tv,Ti}(m, n, colptr, rowval, nzval)
end

sparse(s::UniformScaling, dims::Dims{2}) = SparseMatrixCSC(s, dims)
sparse(s::UniformScaling, m::Integer, n::Integer) = sparse(s, Dims((m, n)))
sparse(::Type{Tv}, s::UniformScaling, m::Integer, n::Integer) where {Tv} = SparseMatrixCSC{Tv}(s, Dims((m, n)))
sparse(::Type{Tv}, ::Type{Ti}, s::UniformScaling, m::Integer, n::Integer) where {Tv, Ti} = SparseMatrixCSC{Tv, Ti}(s, Dims((m, n)))

_nnz(v::AbstractSparseVector) = nnz(v)
_nnz(v::AbstractVector) = length(v)

function _indices(v::AbstractSparseVector, row, col)
    ix = nonzeroinds(v)
    return (row .+ ix, col .+ ix)
end
function _indices(v::AbstractVector, row, col)
    veclen = length(v)
    return (row+1:row+veclen, col+1:col+veclen)
end

_nzvals(v::AbstractSparseVector) = nonzeros(v)
_nzvals(v::AbstractVector) = v

# Promoted element type of the diagonals, mirroring `Base.promote_eltypeof`
spdiagm_eltype(p::Pair) = eltype(p.second)
spdiagm_eltype(p::Pair, q::Pair, rest::Pair...) =
    (@inline; promote_type(promote_type(eltype(p.second), eltype(q.second)),
                           spdiagm_eltype(rest...)))
spdiagm_eltype(p::Pair, q::Pair) = promote_type(eltype(p.second), eltype(q.second))
spdiagm_eltype(::Pair{<:Integer,<:AbstractVector{T}}, ::Pair{<:Integer,<:AbstractVector{T}}...) where {T} = T

function spdiagm_internal(kv::Pair{<:Integer,<:AbstractVector}...)
    ncoeffs = 0
    for p in kv
        ncoeffs += _nnz(p.second)
    end
    I = Vector{Int}(undef, ncoeffs)
    J = Vector{Int}(undef, ncoeffs)
    V = Vector{spdiagm_eltype(kv...)}(undef, ncoeffs)
    i = 0
    m = 0
    n = 0
    for p in kv
        k = p.first
        v = p.second
        if k < 0
            row = -k
            col = 0
        elseif k > 0
            row = 0
            col = k
        else
            row = 0
            col = 0
        end
        numel = _nnz(v)
        r = 1+i:numel+i
        I[r], J[r] = _indices(v, row, col)
        copyto!(view(V, r), _nzvals(v))
        veclen = length(v)
        m = max(m, row + veclen)
        n = max(n, col + veclen)
        i += numel
    end
    return I, J, V, m, n
end

"""
    spdiagm(kv::Pair{<:Integer,<:AbstractVector}...)
    spdiagm(m::Integer, n::Integer, kv::Pair{<:Integer,<:AbstractVector}...)

Construct a sparse diagonal matrix from `Pair`s of vectors and diagonals.
Each vector `kv.second` will be placed on the `kv.first` diagonal.  By
default, the matrix is square and its size is inferred
from `kv`, but a non-square size `m`×`n` (padded with zeros as needed)
can be specified by passing `m,n` as the first arguments.

# Examples
```jldoctest
julia> spdiagm(-1 => [1,2,3,4], 1 => [4,3,2,1])
5×5 SparseMatrixCSC{Int64, Int64} with 8 stored entries:
 ⋅  4  ⋅  ⋅  ⋅
 1  ⋅  3  ⋅  ⋅
 ⋅  2  ⋅  2  ⋅
 ⋅  ⋅  3  ⋅  1
 ⋅  ⋅  ⋅  4  ⋅
```
"""
spdiagm(kv::Pair{<:Integer,<:AbstractVector}...) = _spdiagm(nothing, kv...)
spdiagm(m::Integer, n::Integer, kv::Pair{<:Integer,<:AbstractVector}...) = _spdiagm((Int(m),Int(n)), kv...)

"""
    spdiagm(v::AbstractVector)
    spdiagm(m::Integer, n::Integer, v::AbstractVector)

Construct a sparse matrix with elements of the vector as diagonal elements.
By default (no given `m` and `n`), the matrix is square and its size is given
by `length(v)`, but a non-square size `m`×`n` can be specified by passing `m`
and `n` as the first arguments.

!!! compat "Julia 1.6"
    These functions require at least Julia 1.6.

# Examples
```jldoctest
julia> spdiagm([1,2,3])
3×3 SparseMatrixCSC{Int64, Int64} with 3 stored entries:
 1  ⋅  ⋅
 ⋅  2  ⋅
 ⋅  ⋅  3

julia> spdiagm(sparse([1,0,3]))
3×3 SparseMatrixCSC{Int64, Int64} with 2 stored entries:
 1  ⋅  ⋅
 ⋅  ⋅  ⋅
 ⋅  ⋅  3
```
"""
spdiagm(v::AbstractVector) = _spdiagm(nothing, 0 => v)
spdiagm(m::Integer, n::Integer, v::AbstractVector) = _spdiagm((Int(m), Int(n)), 0 => v)

_spdiagm(size) = spzeros(Bool, something(size, (0,0))...) # eltype as `diagm(m, n)`
function _spdiagm(size, kv::Pair{<:Integer,<:AbstractVector}...)
    I, J, V, mmax, nmax = spdiagm_internal(kv...)
    mnmax = max(mmax, nmax)
    m, n = something(size, (mnmax,mnmax))
    (m ≥ mmax && n ≥ nmax) || throw(DimensionMismatch("invalid size=$size"))
    return sparse(I, J, V, m, n)
end
