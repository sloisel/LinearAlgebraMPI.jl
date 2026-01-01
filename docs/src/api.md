# API Reference

This page provides detailed documentation for all exported types and functions in LinearAlgebraMPI.jl.

!!! note "MPI Collective Operations"
    Unless otherwise noted, all functions are MPI collective operations. Every MPI rank must call these functions together.

## Distributed Types

### VectorMPI

```@docs
VectorMPI
```

### MatrixMPI

```@docs
MatrixMPI
```

### SparseMatrixMPI

```@docs
SparseMatrixMPI
```

### SparseMatrixCSR

```@docs
SparseMatrixCSR
```

## Local Constructors

These constructors create distributed types from local data without global communication.

```@docs
VectorMPI_local
MatrixMPI_local
SparseMatrixMPI_local
```

## Row-wise Operations

```@docs
map_rows
map_rows_gpu
```

## Linear System Solvers

```@docs
solve
solve!
finalize!
```

## Partition Utilities

```@docs
uniform_partition
repartition
```

## Cache Management

```@docs
clear_plan_cache!
clear_mumps_analysis_cache!
```

## IO Utilities

```@docs
io0
```

## Type Mappings

### Native to MPI Conversions

| Native Type | MPI Type | Description |
|-------------|----------|-------------|
| `Vector{T}` | `VectorMPI{T,AV}` | Distributed vector |
| `Matrix{T}` | `MatrixMPI{T,AM}` | Distributed dense matrix |
| `SparseMatrixCSC{T,Ti}` | `SparseMatrixMPI{T,Ti,AV}` | Distributed sparse matrix |

The `AV` and `AM` type parameters specify the underlying storage (`Vector{T}`/`Matrix{T}` for CPU, `MtlVector{T}`/`MtlMatrix{T}` for Metal GPU).

### MPI to Native Conversions

| MPI Type | Native Type | Function |
|----------|-------------|----------|
| `VectorMPI{T,AV}` | `Vector{T}` | `Vector(v)` |
| `MatrixMPI{T,AM}` | `Matrix{T}` | `Matrix(A)` |
| `SparseMatrixMPI{T,Ti,AV}` | `SparseMatrixCSC{T,Ti}` | `SparseMatrixCSC(A)` |

## Supported Operations

### VectorMPI Operations

- Arithmetic: `+`, `-`, `*` (scalar)
- Linear algebra: `norm`, `dot`, `conj`
- Indexing: `v[i]` (global index)
- Conversion: `Vector(v)`

### MatrixMPI Operations

- Arithmetic: `*` (scalar), matrix-vector product
- Transpose: `transpose(A)`
- Indexing: `A[i, j]` (global indices)
- Conversion: `Matrix(A)`

### SparseMatrixMPI Operations

- Arithmetic: `+`, `-`, `*` (scalar, matrix-vector, matrix-matrix)
- Transpose: `transpose(A)`
- Linear solve: `A \ b`, `Symmetric(A) \ b`
- Utilities: `nnz`, `norm`, `issymmetric`
- Conversion: `SparseMatrixCSC(A)`

## Factorization Types

LinearAlgebraMPI uses MUMPS for sparse direct solves:

- `lu(A)`: LU factorization (general matrices)
- `ldlt(A)`: LDLT factorization (symmetric matrices, faster)

Both return factorization objects that support:
- `F \ b`: Solve with factorization
- `finalize!(F)`: Release MUMPS resources
