# LinearAlgebraMPI.jl

**Distributed sparse matrix operations using MPI for parallel computing across multiple ranks.**

LinearAlgebraMPI.jl provides a high-performance implementation of distributed sparse matrices in Julia, enabling parallel sparse linear algebra operations across multiple MPI processes. The package is designed for large-scale computations where matrices are too large to fit on a single node or where parallel speedup is desired.

## Features

- **Row-partitioned sparse matrices**: Matrices are distributed by rows across MPI ranks
- **Matrix multiplication**: Efficient sparse matrix-matrix product with memoized communication plans
- **Addition and subtraction**: Element-wise operations with automatic data redistribution
- **Transpose operations**: Both eager and lazy transpose support
- **Conjugate and adjoint**: Full support for complex matrices
- **Scalar multiplication**: Efficient scalar-matrix products
- **Norms**: Frobenius norm, 1-norm, infinity norm, and general p-norms
- **Operator norms**: 1-norm and infinity-norm of operators
- **Type stability**: Generic implementation supporting `Float64`, `ComplexF64`, and other numeric types
- **Plan caching**: Communication plans are memoized for repeated operations with the same sparsity pattern

## Quick Example

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays

# Create a sparse matrix (must be identical on all ranks)
A = sprand(1000, 1000, 0.01)
B = sprand(1000, 1000, 0.01)

# Distribute matrices across MPI ranks
Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)

# Perform distributed operations
C = Adist * Bdist    # Matrix multiplication
D = Adist + Bdist    # Addition
E = Adist - Bdist    # Subtraction
F = 2.0 * Adist      # Scalar multiplication

# Compute norms
frobenius_norm = norm(Adist)
max_col_sum = opnorm(Adist, 1)

MPI.Finalize()
```

## Package Overview

```@contents
Pages = ["getting-started.md", "examples.md", "api.md", "internals.md"]
Depth = 2
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/your-username/LinearAlgebraMPI.jl")
```

Or in the Julia REPL package mode:

```
pkg> add https://github.com/your-username/LinearAlgebraMPI.jl
```

## Requirements

- Julia 1.10+
- MPI.jl with a working MPI implementation
- SparseArrays.jl
- LinearAlgebra.jl
- Blake3Hash.jl

## License

MIT License

## Index

```@index
```
