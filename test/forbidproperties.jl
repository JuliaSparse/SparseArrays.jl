# This file is a part of Julia. License is MIT: https://julialang.org/license

using SparseArrays
# Only define each `getproperty` override if it isn't already defined. This
# file is `include`d from multiple sibling test modules, and without this
# guard each subsequent include would re-set the same `Base` methods,
# producing spurious method-overwrite warnings (depending on `--warn-overwrite`).
# `hasmethod` is not sufficient here because the generic `Base.getproperty`
# fallback for `Any` always exists; we instead check whether `which` returns
# that fallback method.
let fallback = which(Base.getproperty, Tuple{Any, Symbol})
    if which(Base.getproperty, Tuple{SparseMatrixCSC, Symbol}) === fallback
        Base.getproperty(::SparseMatrixCSC, ::Symbol) = error("use accessor function")
    end
    if which(Base.getproperty, Tuple{SparseVector, Symbol}) === fallback
        Base.getproperty(::SparseVector, ::Symbol) = error("use accessor function")
    end
end
