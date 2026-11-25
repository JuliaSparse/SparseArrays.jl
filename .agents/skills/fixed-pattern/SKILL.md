---
name: fixed-pattern
description: Rules for FixedSparseCSC and FixedSparseVector (sparse arrays with a fixed pattern) in SparseArrays.jl. Use when writing or changing a kernel that writes into a sparse argument, changes a sparsity pattern, or adds a method on a fixed type.
---

# Sparse arrays with a fixed pattern

Read "Sparse arrays with a fixed pattern" in `docs/src/index.md` first; it states the three
rules users see. The helpers are in `src/fixed.jl`.

A fixed-pattern array is a destination, not a kind of value.

- **A write stores inside the pattern or throws** before changing anything. A kernel that
  writes its argument checks up front, and stored entries the write does not fill become
  zero. Writing through `nonzeros` is always allowed.
- **A function whose purpose is to change the pattern** (`dropzeros!`, `fkeep!`, `tril!`,
  `sort!`, `permute!`, ...) calls `_checkpatternwritable(A, :name!)` on the argument it
  changes.
- **As an operand, a fixed array reads as the plain array.** No result is fixed except
  `copy`, which shares the pattern. An out-of-place kernel never checks `_is_fixed` on an
  input, and starts from `_writablecopy` when it mutates a copy.
- Do not add a method that makes another operation return a fixed array.
- A method on a fixed type is only for a function whose generic method in Base or
  LinearAlgebra never reaches a kernel of this package that could check.
- Fixed-pattern arrays stay experimental and unexported.
