---
name: suitesparse-wrappers
description: Regenerate the SuiteSparse wrappers from the C headers, upgrade SuiteSparse, or upgrade the generator's own dependencies (Clang.jl, JuliaFormatter). Use for any change to the generator or to the generated wrappers.
---

# SuiteSparse wrappers

The low-level SuiteSparse bindings are generated from the C headers. Never edit the
generated code by hand: change the generator's inputs (its prologue, configuration or
script) and regenerate with `make` in the generator directory.

## Regenerating

- Set the SuiteSparse version, the `SuiteSparse_jll` build number and the archive checksum
  in the generator's Makefile, then run `make`. Downloads and extracted headers are cached
  per version and build.
- The headers come from one pinned `SuiteSparse_jll` archive, never from the running
  Julia's copy, which is a fixed stdlib. They are platform independent, so one generated
  file serves every platform.
- Keep the generator's SuiteSparse version in sync with the package's `SuiteSparse_jll`
  compat entry.
- Headers are wrapped in the listed order, so a header comes before any header that
  includes it; otherwise its wrappers move in the output. Header-to-library mapping is by
  pattern on the path, so a pattern must not match a longer name by accident.
- Before changing the generator, regenerate and confirm the committed output is
  reproduced byte for byte, so that the diff afterwards is only what your change causes.
- Drop a macro Clang.jl cannot handle through the generator's ignore list.
- BSD-licensed libraries among the headers (AMD, COLAMD) may be called from outside the
  GPL code; the rest may not.

## Upgrading SuiteSparse

1. Update SuiteSparse in Yggdrasil.
2. Update the version here and regenerate.
3. Run BumpStdlibs to update the SparseArrays.jl version in Julia master.
4. Update the relevant stdlibs in Julia to pull in the new releases.

The bump cannot be tested here until nightly bundles the new jll, so its CI is expected to
be red. Do not delete the merged bump PR's branch.

## Upgrading Clang.jl or JuliaFormatter

Change the compat bound in the generator's project (a breaking Clang.jl release may need
the script adapted), regenerate, and commit the output with the compat change. A diff in
the output is the new release's output change; say so in the PR.
