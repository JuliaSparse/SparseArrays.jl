---
name: manual-page
description: Edit the package's manual page and the docstrings it lists without breaking Julia's own docs build. Use when adding, moving or changing documentation or a docstring that the manual lists.
---

# The manual page

Julia's manual includes this package's single manual page and the docstrings it lists,
and nothing else. Julia's docs build, not this repository's, is the one that has to pass.

- All documentation goes on that one page. There is no second page.
- An `@ref` to anything outside the page breaks Julia's build.
- A `@docs` entry for a function that LinearAlgebra's page also lists (`lu`, `cholesky`)
  breaks it unless the entry carries the exact method signature of the sparse docstring.
  Write that signature with the solver's own index-type alias, which is `Int32` alone on a
  32-bit build, so the entry matches on every platform.
- The page's `@contents` block names the page both by its file name here and by its file
  name in Julia's manual. Under any other name the table of contents is empty.
- Internal helpers get no entry.
- Build the docs and run the doctests locally before a PR.
- A documentation-only PR body needs just the source of the material.
