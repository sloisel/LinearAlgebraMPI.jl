# LinearAlgebraMPI.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sloisel.github.io/LinearAlgebraMPI.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sloisel.github.io/LinearAlgebraMPI.jl/dev/)
[![Build Status](https://github.com/sloisel/LinearAlgebraMPI.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sloisel/LinearAlgebraMPI.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sloisel/LinearAlgebraMPI.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sloisel/LinearAlgebraMPI.jl)

**Author:** S. Loisel

Distributed sparse matrix and vector operations using MPI for Julia. This package provides efficient parallel linear algebra operations across multiple MPI ranks.

## Features

- **Distributed sparse matrices** (`SparseMatrixMPI{T,Ti,AV}`) with row-partitioning across MPI ranks
- **Distributed dense vectors** (`VectorMPI{T,AV}`) with flexible partitioning
- **Matrix-matrix multiplication** (`A * B`) with memoized communication plans
- **Matrix-vector multiplication** (`A * x`, `mul!(y, A, x)`)
- **Sparse direct solvers**: LU and LDLT factorization using MUMPS
- **Lazy transpose** with optimized multiplication rules
- **Matrix addition/subtraction** (`A + B`, `A - B`)
- **Vector operations**: norms, reductions, arithmetic with automatic partition alignment
- Support for both `Float64` and `ComplexF64` element types
- **GPU acceleration** via Metal.jl (macOS) or CUDA.jl (Linux/Windows) with automatic CPU staging for MPI
- **Multi-GPU sparse direct solver** via cuDSS with NCCL communication (CUDA only)

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
```

## GPU Support

LinearAlgebraMPI supports GPU acceleration via Metal.jl (macOS) or CUDA.jl (Linux/Windows). GPU support is optional - extensions are loaded as weak dependencies.

### Metal (macOS)

```julia
using Metal  # Load Metal BEFORE MPI for GPU detection
using MPI
MPI.Init()
using LinearAlgebraMPI

# Create CPU vectors/matrices as usual
x_cpu = VectorMPI(Float32.(rand(1000)))
A = SparseMatrixMPI{Float32}(sprand(Float32, 1000, 1000, 0.01))

# Convert to GPU
x_gpu = mtl(x_cpu)  # Returns VectorMPI with MtlVector storage

# GPU operations work transparently
y_gpu = A * x_gpu   # Sparse A*x with GPU vector (CPU staging for computation)
z_gpu = x_gpu + x_gpu  # Vector addition on GPU

# Convert back to CPU
y_cpu = cpu(y_gpu)
```

### CUDA (Linux/Windows)

```julia
using CUDA  # Load CUDA BEFORE MPI
using MPI
MPI.Init()
using LinearAlgebraMPI

# Convert to GPU
x_cpu = VectorMPI(rand(1000))
x_gpu = cu(x_cpu)  # Returns VectorMPI with CuVector storage

# GPU operations work transparently
y_gpu = A * x_gpu
z_gpu = x_gpu + x_gpu

# Convert back to CPU
y_cpu = cpu(y_gpu)
```

### cuDSS Multi-GPU Solver (CUDA only)

For multi-GPU distributed sparse direct solves, use `CuDSSFactorizationMPI`:

```julia
using CUDA, MPI
MPI.Init()
using LinearAlgebraMPI

# Each rank uses one GPU
CUDA.device!(MPI.Comm_rank(MPI.COMM_WORLD) % length(CUDA.devices()))

# Create distributed sparse matrix
A = SparseMatrixMPI{Float64}(make_spd_matrix(1000))
b = VectorMPI(rand(1000))

# Multi-GPU factorization using cuDSS + NCCL
F = cudss_ldlt(A)  # or cudss_lu(A)
x = F \ b
finalize!(F)  # Clean up cuDSS resources
```

**Requirements**: cuDSS 0.4+ with MGMN (Multi-GPU Multi-Node) support, NCCL for inter-GPU communication.

### How it works

- **Vectors**: `VectorMPI{T,AV}` where `AV` is `Vector{T}` (CPU), `MtlVector{T}` (Metal), or `CuVector{T}` (CUDA)
- **Sparse matrices**: `SparseMatrixMPI{T,Ti,AV}` where `AV` determines storage for nonzero values
- **Dense matrices**: `MatrixMPI{T,AM}` where `AM` is `Matrix{T}`, `MtlMatrix{T}`, or `CuMatrix{T}`
- **MPI communication**: Always uses CPU buffers (staged automatically)
- **Element types**: Metal requires `Float32`; CUDA supports `Float32` and `Float64`

### Supported GPU operations

| Operation | Metal | CUDA |
|-----------|-------|------|
| `v + w`, `v - w` | Native | Native |
| `α * v` (scalar) | Native | Native |
| `A * x` (sparse) | CPU staging | CPU staging |
| `A * x` (dense) | CPU staging | CPU staging |
| `transpose(A) * x` | CPU staging | CPU staging |
| Broadcasting (`abs.(v)`) | Native | Native |
| `cudss_lu(A)`, `cudss_ldlt(A)` | N/A | Multi-GPU native |

## Running with MPI

```bash
mpiexec -n 4 julia your_script.jl
```

## Documentation

For detailed documentation, see the [stable docs](https://sloisel.github.io/LinearAlgebraMPI.jl/stable/) or [dev docs](https://sloisel.github.io/LinearAlgebraMPI.jl/dev/).
