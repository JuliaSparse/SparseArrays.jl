---
name: manual-page
description: Edit SparseArrays.jl documentation - docs/src/index.md, docstrings and their @docs and @ref entries - without breaking Julia's own docs build. Use when adding, moving or changing documentation or a docstring that the manual lists.
---

# The manual page

Julia's manual includes `docs/src/index.md` and the docstrings it lists, and no other
page. Its build, not this repository's, is the one that has to pass.

- All documentation goes in `docs/src/index.md`. There is no second page.
- An `@ref` to anything outside that page breaks Julia's docs build.
- A `@docs` entry for a function that LinearAlgebra's page lists too, such as `lu` or
  `cholesky`, breaks it unless the entry carries the method signature of the sparse
  docstring. Write that signature with the solver's own index-type alias
  (`UMFPACK.UMFITypes`, `CHOLMOD.ITypes`), which is `Int32` alone on a 32-bit build.
- The page's `@contents` block names it as both `index.md` and `SparseArrays.md`, its
  file name in Julia's manual. Under any other name the table of contents is empty.
- Public surface changes are documented; internal helpers are not API and get no docs
  entry.
- Build and doctest locally with
  `julia +nightly --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'`.
- A documentation-only PR body needs just the source of the material.
