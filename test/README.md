# Test organization

`runtests.jl` lists the scheduling units used by both ParallelTestRunner and the
serial Base-CI fallback. File selectors such as `sparsevector` and `linalg` select
these units, including their feature files.

The `triangular_products`, `triangular_solves`, and `concatenation` directories
contain feature tests and their issue regressions. Their parent suites include
them in place, preserving test order, imports, and the existing scheduling units.
Files in these directories are included fragments, not standalone runner tasks.
The sparse triangular product and solve grid shares one fixture and stays together
in `triangular_products/sparse.jl`.

Scheduling changes should be measured with the same worker count and test inventory.
Group existing units in a trial runner before changing the default inventory;
creating a feature file does not require adding a worker task.
