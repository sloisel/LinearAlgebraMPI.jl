# LinearAlgebraMPI.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sloisel.github.io/LinearAlgebraMPI.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sloisel.github.io/LinearAlgebraMPI.jl/dev/)
[![Build Status](https://github.com/sloisel/LinearAlgebraMPI.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sloisel/LinearAlgebraMPI.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sloisel/LinearAlgebraMPI.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sloisel/LinearAlgebraMPI.jl)

**Author:** S. Loisel

Distributed sparse matrix and vector operations using MPI for Julia. This package provides efficient parallel linear algebra operations across multiple MPI ranks.

## Features

- **Distributed sparse matrices** (`SparseMatrixMPI{T}`) with row-partitioning across MPI ranks
- **Distributed dense vectors** (`VectorMPI{T}`) with flexible partitioning
- **Matrix-matrix multiplication** (`A * B`) with memoized communication plans
- **Matrix-vector multiplication** (`A * x`, `mul!(y, A, x)`)
- **Sparse direct solvers**: LU and LDLT factorization using MUMPS
- **Lazy transpose** with optimized multiplication rules
- **Matrix addition/subtraction** (`A + B`, `A - B`)
- **Vector operations**: norms, reductions, arithmetic with automatic partition alignment
- Support for both `Float64` and `ComplexF64` element types

## Installation

```julia
using Pkg
Pkg.add("LinearAlgebraMPI")
```

## Quick Start

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays

# Create a sparse matrix (must be identical on all ranks)
A_global = sprand(1000, 1000, 0.01)
A = SparseMatrixMPI{Float64}(A_global)

# Create a vector
x_global = rand(1000)
x = VectorMPI(x_global)

# Matrix-vector multiplication
y = A * x

# Matrix-matrix multiplication
B_global = sprand(1000, 500, 0.01)
B = SparseMatrixMPI{Float64}(B_global)
C = A * B

# Transpose operations
At = transpose(A)
D = At * B  # Materializes transpose as needed

# Solve linear systems
using LinearAlgebra
A_sym = A + transpose(A) + 10I  # Make symmetric positive definite
A_sym_dist = SparseMatrixMPI{Float64}(A_sym)
F = ldlt(A_sym_dist)  # LDLT factorization
x_sol = solve(F, y)   # Solve A_sym * x_sol = y
# F is automatically cleaned up when garbage collected
```

## Running with MPI

```bash
mpiexec -n 4 julia your_script.jl
```

## Documentation

For detailed documentation, see the [stable docs](https://sloisel.github.io/LinearAlgebraMPI.jl/stable/) or [dev docs](https://sloisel.github.io/LinearAlgebraMPI.jl/dev/).
