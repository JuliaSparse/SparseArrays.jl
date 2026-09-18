```@meta
EditURL = "https://github.com/JuliaSparse/SparseArrays.jl/blob/master/docs/src/index.md"
```

# Sparse Arrays

```@meta
DocTestSetup = :(using SparseArrays, LinearAlgebra)
```

Julia has support for sparse vectors and [sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix)
in the `SparseArrays` stdlib module. Sparse arrays are arrays that contain enough zeros that storing them in a special data structure leads to savings in space and execution time, compared to dense arrays.

External packages which implement different sparse storage types, multidimensional sparse arrays, and more can be found in [Noteworthy External Sparse Packages](@ref)

## [Compressed Sparse Column (CSC) Sparse Matrix Storage](@id man-csc)

In Julia, sparse matrices are stored in the [Compressed Sparse Column (CSC) format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_.28CSC_or_CCS.29).
Julia sparse matrices have the type [`SparseMatrixCSC{Tv,Ti}`](@ref), where `Tv` is the
type of the stored values, and `Ti` is the integer type for storing column pointers and
row indices. The internal representation of `SparseMatrixCSC` is as follows:

```julia
struct SparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    colptr::Vector{Ti}      # Column j is in colptr[j]:(colptr[j+1]-1)
    rowval::Vector{Ti}      # Row indices of stored values
    nzval::Vector{Tv}       # Stored values, typically nonzeros
end
```

The compressed sparse column storage makes it easy and quick to access the elements in the column
of a sparse matrix, whereas accessing the sparse matrix by rows is considerably slower. Operations
such as insertion of previously unstored entries one at a time in the CSC structure tend to be slow. This is
because all elements of the sparse matrix that are beyond the point of insertion have to be moved
one place over.

All operations on sparse matrices are carefully implemented to exploit the CSC data structure
for performance, and to avoid expensive operations.

If you have data in CSC format from a different application or
library, and wish to import it in Julia, make sure that you use
1-based indexing. The row indices in every column need to be sorted,
and if they are not, the matrix will display incorrectly.  If your
`SparseMatrixCSC` object contains unsorted row indices, one quick way
to sort them is by doing a double transpose. Since the transpose operation
is lazy, make a copy to materialize each transpose.

In some applications, it is convenient to store explicit zero values in a `SparseMatrixCSC`. These
*are* accepted by functions in `Base` (but there is no guarantee that they will be preserved in
mutating operations). Such explicitly stored zeros are treated as structural nonzeros by many
routines. The [`nnz`](@ref) function returns the number of elements explicitly stored in the
sparse data structure, including non-structural zeros. In order to count the exact number of
numerical nonzeros, use [`count(!iszero, x)`](@ref), which inspects every stored element of a sparse
matrix. [`dropzeros`](@ref), and the in-place [`dropzeros!`](@ref), can be used to
remove stored zeros from the sparse matrix.

```jldoctest
julia> A = sparse([1, 1, 2, 3], [1, 3, 2, 3], [0, 1, 2, 0])
3×3 SparseMatrixCSC{Int64, Int64} with 4 stored entries:
 0  ⋅  1
 ⋅  2  ⋅
 ⋅  ⋅  0

julia> dropzeros(A)
3×3 SparseMatrixCSC{Int64, Int64} with 2 stored entries:
 ⋅  ⋅  1
 ⋅  2  ⋅
 ⋅  ⋅  ⋅
```

## Sparse Vector Storage

Sparse vectors are stored in a close analog to compressed sparse column format for sparse
matrices. In Julia, sparse vectors have the type [`SparseVector{Tv,Ti}`](@ref) where `Tv`
is the type of the stored values and `Ti` the integer type for the indices. The internal
representation is as follows:

```julia
struct SparseVector{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}
    n::Int              # Length of the sparse vector
    nzind::Vector{Ti}   # Indices of stored values
    nzval::Vector{Tv}   # Stored values, typically nonzeros
end
```

Like [`SparseMatrixCSC`](@ref), the `SparseVector` type can also contain explicitly
stored zeros. (See [Sparse Matrix Storage](@ref man-csc).).

## Sparse Vector and Matrix Constructors

The simplest way to create a sparse array is to use a function equivalent to the [`zeros`](@ref)
function that Julia provides for working with dense arrays. To produce a
sparse array instead, you can use the same name with an `sp` prefix:

```jldoctest
julia> spzeros(3)
3-element SparseVector{Float64, Int64} with 0 stored entries
```

The [`sparse`](@ref) function is often a handy way to construct sparse arrays. For
example, to construct a sparse matrix we can input a vector `I` of row indices, a vector
`J` of column indices, and a vector `V` of stored values (this is also known as the
[COO (coordinate) format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_.28COO.29)).
`sparse(I,J,V)` then constructs a sparse matrix such that `S[I[k], J[k]] = V[k]`. The
equivalent sparse vector constructor is [`sparsevec`](@ref), which takes the (row) index
vector `I` and the vector `V` with the stored values and constructs a sparse vector `R`
such that `R[I[k]] = V[k]`.

```jldoctest sparse_function
julia> I = [1, 4, 3, 5]; J = [4, 7, 18, 9]; V = [1, 2, -5, 3];

julia> S = sparse(I,J,V)
5×18 SparseMatrixCSC{Int64, Int64} with 4 stored entries:
⎡⠀⠈⠀⠀⠀⠀⠀⠀⢀⎤
⎣⠀⠀⠀⠂⡀⠀⠀⠀⠀⎦

julia> R = sparsevec(I,V)
5-element SparseVector{Int64, Int64} with 4 stored entries:
  [1]  =  1
  [3]  =  -5
  [4]  =  2
  [5]  =  3
```

The inverse of the [`sparse`](@ref) and [`sparsevec`](@ref) functions is
[`findnz`](@ref), which retrieves the inputs used to create the sparse array (including stored entries equal to zero).
[`findall(!iszero, x)`](@ref) returns the Cartesian indices of non-zero entries in `x`
(not including stored entries equal to zero).

```jldoctest sparse_function
julia> findnz(S)
([1, 4, 5, 3], [4, 7, 9, 18], [1, 2, 3, -5])

julia> findall(!iszero, S)
4-element Vector{CartesianIndex{2}}:
 CartesianIndex(1, 4)
 CartesianIndex(4, 7)
 CartesianIndex(5, 9)
 CartesianIndex(3, 18)

julia> findnz(R)
([1, 3, 4, 5], [1, -5, 2, 3])

julia> findall(!iszero, R)
4-element Vector{Int64}:
 1
 3
 4
 5
```

Another way to create a sparse array is to convert a dense array into a sparse array using
the [`sparse`](@ref) function:

```jldoctest
julia> sparse(Matrix(1.0I, 5, 5))
5×5 SparseMatrixCSC{Float64, Int64} with 5 stored entries:
 1.0   ⋅    ⋅    ⋅    ⋅
  ⋅   1.0   ⋅    ⋅    ⋅
  ⋅    ⋅   1.0   ⋅    ⋅
  ⋅    ⋅    ⋅   1.0   ⋅
  ⋅    ⋅    ⋅    ⋅   1.0

julia> sparse([1.0, 0.0, 1.0])
3-element SparseVector{Float64, Int64} with 2 stored entries:
  [1]  =  1.0
  [3]  =  1.0
```

You can go in the other direction using the [`Array`](@ref) constructor. The [`issparse`](@ref)
function can be used to query if a matrix is sparse.

```jldoctest
julia> issparse(spzeros(5))
true
```

## Sparse matrix operations

Arithmetic operations on sparse matrices also work as they do on dense matrices. Indexing of,
assignment into, and concatenation of sparse matrices work in the same way as dense matrices.
Indexing operations, especially assignment, are expensive, when carried out one element at a time.
In many cases it may be better to convert the sparse matrix into `(I,J,V)` format using [`findnz`](@ref),
manipulate the values or the structure in the dense vectors `(I,J,V)`, and then reconstruct
the sparse matrix.

## [Performance tips](@id man-sparse-performance)

Sparse code is fast when its cost is proportional to the number of stored entries, and the
[CSC layout](@ref man-csc) determines which operations have that property.

### Iterate over stored entries by column

The stored entries of column `j` are located at the positions [`nzrange(A, j)`](@ref nzrange) of
[`rowvals(A)`](@ref rowvals) and [`nonzeros(A)`](@ref nonzeros). Looping over the columns, and over
that range within each column, visits every stored entry once and in memory order, without
searching. For example, a matrix-vector product can be written as:

```jldoctest perftips
julia> A = sparse([1, 1, 2, 3], [1, 3, 2, 3], [1.0, 2.0, 3.0, 4.0]);

julia> function mymul(A::SparseMatrixCSC, x::Vector)
           y = zeros(promote_type(eltype(A), eltype(x)), size(A, 1))
           rows, vals = rowvals(A), nonzeros(A)
           for j in axes(A, 2), k in nzrange(A, j)
               y[rows[k]] += vals[k] * x[j]
           end
           return y
       end;

julia> mymul(A, [1.0, 10.0, 100.0]) == A * [1.0, 10.0, 100.0]
true
```

In contrast, scalar indexing `A[i, j]` has to search column `j` for row `i`, and a loop over all
`(i, j)` costs time proportional to `length(A)` rather than `nnz(A)`.

### Build a matrix from its entries in one call

Storing a value at a position that has no stored entry yet moves all the later entries of `rowvals(A)`
and `nonzeros(A)` and updates the column pointers, so filling `spzeros(m, n)` one element at a time takes
time proportional to `nnz(A)` for each insertion. `sizehint!(A, n)` reserves room for
`n` stored entries, which avoids reallocating those vectors but not moving the entries. Instead, collect
the row indices, column indices and values in three vectors and call [`sparse`](@ref) once. Entries with
the same position are added together, or combined with the function passed as the last argument:

```jldoctest perftips
julia> I = [1, 1, 2]; J = [1, 1, 2]; V = [1.0, 2.0, 5.0];

julia> sparse(I, J, V)
2×2 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
 3.0   ⋅
  ⋅   5.0

julia> sparse(I, J, V, 2, 2, max)
2×2 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
 2.0   ⋅
  ⋅   5.0
```

### Slice columns, not rows

`A[:, j]` copies one contiguous range of the stored entries. `A[i, :]` has to search every column for
row `i`, so its cost grows with the number of columns even when the row is empty. When an algorithm
works row by row, transpose the matrix once and work on the columns of the result. `permutedims(A)`
and `copy(transpose(A))` (equivalently `sparse(transpose(A))`) build the transposed matrix in time
proportional to `nnz(A)`.

```jldoctest perftips
julia> At = permutedims(A);

julia> At[:, 1] == A[1, :]
true
```

### Choose a smaller index type

The index type `Ti` is used for the column pointers and for one row index per stored entry. With `Float64`
values, `Int32` indices reduce the memory per stored entry from 16 to 12 bytes. `Ti` is taken from the
index vectors given to [`sparse`](@ref), can be given to [`spzeros`](@ref), and an existing matrix is
converted with the type constructor. Both dimensions have to be at most `typemax(Ti)`, and the number of
stored entries has to be less than `typemax(Ti)`.

```jldoctest perftips
julia> A32 = SparseMatrixCSC{Float64,Int32}(A)
3×3 SparseMatrixCSC{Float64, Int32} with 4 stored entries:
 1.0   ⋅   2.0
  ⋅   3.0   ⋅
  ⋅    ⋅   4.0

julia> typeof(sparse(Int32[1, 2], Int32[1, 2], [1.0, 2.0])) == typeof(spzeros(Float64, Int32, 2, 2))
true

julia> spzeros(Float64, Int8, 200, 200)
ERROR: ArgumentError: number of rows (m = 200) does not fit in Ti = Int8
[...]
```

Operations between matrices with different index types promote to the wider one, so use one index type
consistently.

### Lazy `transpose` and `adjoint`

`transpose(A)` and `A'` do not copy; they return `Transpose` and `Adjoint` wrappers around `A`.
The wrappers are handled without materializing the transpose in products with dense vectors and matrices
(including `mul!`), in products with sparse vectors, in `\`, in `==`, and in `A'[i, :]`, which
is a column slice of `A`. Products with another sparse matrix, broadcasting (which includes `+` and `-`),
`map`, concatenation, `kron` and `findnz` first copy the wrapper into a new `SparseMatrixCSC`; this is
proportional to `nnz(A)` but is repeated on every call. Other functions, such as `sum`, `norm`, and
indexing other than by row, reach generic `AbstractMatrix` methods that visit the wrapper element by element.
When a transposed matrix is used more than once, or is passed to code that is not one of the products or
solves above, materialize it with `copy`, or with `sparse` when the argument may or may not be a wrapper.

```jldoctest perftips
julia> A' * [1.0, 10.0, 100.0]
3-element Vector{Float64}:
   1.0
  30.0
 402.0

julia> copy(A')
3×3 SparseMatrixCSC{Float64, Int64} with 4 stored entries:
 1.0   ⋅    ⋅
  ⋅   3.0   ⋅
 2.0   ⋅   4.0
```

### Keep results sparse and free of stored zeros

A result stays sparse only if the operation maps zeros to zeros. `A .+ 1` and `exp.(A)` return a
`SparseMatrixCSC` in which every entry is stored, which is slower and larger than a `Matrix`; apply such
functions to `nonzeros(A)` instead when only the stored entries are meant. Assigning zero to a stored entry,
and cancellation in a matrix product, leave explicitly stored zeros behind. They are harmless for correctness but
are visited by every kernel. [`dropzeros!`](@ref) removes them, [`droptol!`](@ref) removes entries of small
magnitude, and [`fkeep!`](@ref) keeps the entries for which a predicate of `(i, j, v)` is true, all in place.

```jldoctest perftips
julia> B = copy(A); B[1, 1] = 0; nnz(B)
4

julia> nnz(dropzeros!(B))
3

julia> fkeep!((i, j, v) -> i == j, B)
3×3 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
  ⋅    ⋅    ⋅
  ⋅   3.0   ⋅
  ⋅    ⋅   4.0
```

## Correspondence of dense and sparse methods

The following table gives a correspondence between built-in methods on sparse matrices and their
corresponding methods on dense matrix types. In general, methods that generate sparse matrices
differ from their dense counterparts in that the resulting matrix follows the same sparsity pattern
as a given sparse matrix `S`, or that the resulting sparse matrix has density `d`, i.e. each matrix
element has a probability `d` of being non-zero.

Details can be found in the [Sparse Vectors and Matrices](@ref stdlib-sparse-arrays)
section of the standard library reference.

| Sparse                     | Dense                  | Description                                                                                                                                                           |
|:-------------------------- |:---------------------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`spzeros(m,n)`](@ref)     | [`zeros(m,n)`](@ref)   | Creates a *m*-by-*n* matrix of zeros. ([`spzeros(m,n)`](@ref) is empty.)                                                                                              |
| [`sparse(I,n,n)`](@ref)  | [`Matrix(I,n,n)`](@ref)| Creates a *n*-by-*n* identity matrix.                                                                                                                                 |
| [`sparse(A)`](@ref)        | [`Array(S)`](@ref)   | Interconverts between dense and sparse formats.                                                                                                                       |
| [`sprand(m,n,d)`](@ref)    | [`rand(m,n)`](@ref)    | Creates a *m*-by-*n* random matrix (of density *d*) with iid non-zero elements distributed uniformly on the half-open interval ``[0, 1)``.                            |
| [`sprandn(m,n,d)`](@ref)   | [`randn(m,n)`](@ref)   | Creates a *m*-by-*n* random matrix (of density *d*) with iid non-zero elements distributed according to the standard normal (Gaussian) distribution.                  |
| [`sprandn(rng,m,n,d)`](@ref) | [`randn(rng,m,n)`](@ref) | Creates a *m*-by-*n* random matrix (of density *d*) with iid non-zero elements generated with the `rng` random number generator                                   |


```@meta
DocTestSetup = nothing
```

# [SparseArrays API](@id stdlib-sparse-arrays)

```@docs
SparseArrays.AbstractSparseArray
SparseArrays.AbstractSparseVector
SparseArrays.AbstractSparseMatrix
SparseArrays.SparseVector
SparseArrays.SparseMatrixCSC
SparseArrays.sparse
SparseArrays.sparse!
SparseArrays.sparsevec
Base.similar(::SparseArrays.AbstractSparseMatrixCSC, ::Type)
SparseArrays.issparse
SparseArrays.nnz
SparseArrays.findnz
SparseArrays.spzeros
SparseArrays.spzeros!
SparseArrays.spdiagm
SparseArrays.sparse_hcat
SparseArrays.sparse_vcat
SparseArrays.sparse_hvcat
SparseArrays.blockdiag
SparseArrays.sprand
SparseArrays.sprandn
SparseArrays.nonzeros
SparseArrays.rowvals
SparseArrays.nzrange
SparseArrays.droptol!
SparseArrays.dropzeros!
SparseArrays.dropzeros
SparseArrays.fkeep!
SparseArrays.permute
permute!{Tv, Ti, Tp <: Integer, Tq <: Integer}(::SparseMatrixCSC{Tv,Ti}, ::SparseMatrixCSC{Tv,Ti}, ::AbstractArray{Tp,1}, ::AbstractArray{Tq,1})
SparseArrays.halfperm!
SparseArrays.ftranspose!
```

```@meta
DocTestSetup = nothing
```

# Noteworthy External Sparse Packages

Several other Julia packages provide sparse matrix implementations that should be mentioned:

1. [SuiteSparseGraphBLAS.jl](https://github.com/JuliaSparse/SuiteSparseGraphBLAS.jl) is a wrapper over the fast, multithreaded SuiteSparse:GraphBLAS C library. On CPU this is typically the fastest option, often significantly outperforming MKLSparse.

2. [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) exposes the [CUSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html) library for GPU sparse matrix operations.

3. [SparseMatricesCSR.jl](https://github.com/gridap/SparseMatricesCSR.jl) provides a Julia native implementation of the Compressed Sparse Rows (CSR) format.

4. [MKLSparse.jl](https://github.com/JuliaSparse/MKLSparse.jl) accelerates SparseArrays sparse-dense matrix operations using Intel's MKL library.

5. [SparseArrayKit.jl](https://github.com/Jutho/SparseArrayKit.jl) available for multidimensional sparse arrays.

6. [LuxurySparse.jl](https://github.com/QuantumBFS/LuxurySparse.jl) provides static sparse array formats, as well as a coordinate format.

7. [ExtendableSparse.jl](https://github.com/j-fu/ExtendableSparse.jl) enables fast insertion into sparse matrices using a lazy approach to new stored indices.

8. [Finch.jl](https://github.com/willow-ahrens/Finch.jl) supports extensive multidimensional sparse array formats and operations through a mini tensor language and compiler, all in native Julia. Support for COO, CSF, CSR, CSC and more, as well as operations like broadcast, reduce, etc. and custom operations.

External packages providing sparse direct solvers:
1. [KLU.jl](https://github.com/JuliaSparse/KLU.jl)
2. [Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl/)

External packages providing solvers for iterative solution of eigensystems and singular value decompositions:
1. [ArnoldiMethods.jl](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl)
2. [KrylovKit](https://github.com/Jutho/KrylovKit.jl)
3. [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl)

External packages for working with graphs:
1. [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl)
