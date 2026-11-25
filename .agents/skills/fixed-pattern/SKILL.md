---
name: fixed-pattern
description: Rules for sparse arrays with a fixed sparsity pattern. Use when writing or changing a kernel that writes into a sparse argument, changes a sparsity pattern, or adds a method on a fixed-pattern type.
---

# Sparse arrays with a fixed pattern

Read the manual's section on sparse arrays with a fixed pattern first; it states the rules
users see. A fixed-pattern array is a destination, not a kind of value.

- **A write stores inside the pattern or throws** before changing anything. A kernel that
  writes its argument checks up front; stored entries the write does not fill become zero.
  Writing the stored values directly is always allowed.
- **A function whose purpose is to change the pattern** (dropping, filtering, triangular
  parts, sorting, permuting in place) calls the package's pattern-writability check on the
  argument it changes.
- **As an operand, a fixed array reads as the plain array.** No result is fixed except
  `copy`, which shares the pattern. An out-of-place kernel never asks whether an input is
  fixed, and starts from a writable copy when it mutates a copy.
- Do not add a method that makes another operation return a fixed array.
- A method on a fixed type is only for a function whose generic method in Base or
  LinearAlgebra never reaches a kernel of this package that could check.
- Fixed-pattern arrays stay experimental and unexported.
