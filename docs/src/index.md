```@meta
CurrentModule = LinearAlgebraMPI
```

```@eval
using Markdown
using Pkg
using LinearAlgebraMPI
v = string(pkgversion(LinearAlgebraMPI))
md"# LinearAlgebraMPI.jl $v"
```

**Pure Julia distributed linear algebra with MPI.**

## Overview

LinearAlgebraMPI.jl provides distributed matrix and vector types for parallel computing with MPI. It offers a pure Julia implementation of distributed linear algebra, with MUMPS for sparse direct solves.

## Key Features

- **Distributed Types**: `VectorMPI`, `MatrixMPI`, and `SparseMatrixMPI` for row-partitioned distributed storage
- **MUMPS Solver**: Direct solves using MUMPS for sparse linear systems
- **Row-wise Operations**: `map_rows` for efficient distributed row operations
- **Seamless Integration**: Works with standard Julia linear algebra operations
- **Plan Caching**: Efficient repeated operations through memoized communication plans

## Quick Example

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays

# Create distributed sparse matrix
A = SparseMatrixMPI{Float64}(sprandn(100, 100, 0.1) + 10I)
b = VectorMPI(randn(100))

# Solve linear system
x = A \ b

# Row-wise operations
norms = map_rows(row -> norm(row), MatrixMPI(randn(50, 10)))

println(io0(), "Solution computed!")
```

**Run with MPI:**

```bash
mpiexec -n 4 julia --project example.jl
```

## Documentation Contents

```@contents
Pages = ["installation.md", "guide.md", "examples.md", "api.md"]
Depth = 2
```

## Related Packages

- **[MultiGridBarrierMPI.jl](https://github.com/sloisel/MultiGridBarrierMPI.jl)**: Multigrid barrier methods using LinearAlgebraMPI
- **[MultiGridBarrier.jl](https://github.com/sloisel/MultiGridBarrier.jl)**: Core multigrid barrier method implementation
- **MPI.jl**: Julia MPI bindings

## Requirements

- Julia 1.10 or later (LTS version)
- MPI installation (OpenMPI, MPICH, or Intel MPI)
- MUMPS for sparse direct solves

## License

This package is licensed under the MIT License.
