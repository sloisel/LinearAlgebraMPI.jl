# Getting Started

This guide will walk you through the basics of using LinearAlgebraMPI.jl for distributed sparse matrix computations.

## Prerequisites

Before using LinearAlgebraMPI.jl, ensure you have:

1. A working MPI installation (OpenMPI, MPICH, or Intel MPI)
2. MPI.jl configured to use your MPI installation

You can verify your MPI setup with:

```julia
using MPI
MPI.Init()
println("Rank $(MPI.Comm_rank(MPI.COMM_WORLD)) of $(MPI.Comm_size(MPI.COMM_WORLD))")
MPI.Finalize()
```

Run with:
```bash
mpiexec -n 4 julia --project=. your_script.jl
```

## Creating Distributed Matrices

### From a Global Sparse Matrix

The most common way to create a distributed matrix is from an existing `SparseMatrixCSC`:

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays

# Create a sparse matrix - MUST be identical on all ranks
n = 100
A = spdiagm(0 => 2.0*ones(n), 1 => -ones(n-1), -1 => -ones(n-1))

# Distribute across MPI ranks
Adist = SparseMatrixMPI{Float64}(A)
```

**Important**: All MPI ranks must have identical copies of the input matrix when constructing `SparseMatrixMPI`. The matrix is then automatically partitioned by rows across ranks.

### Understanding Row Partitioning

The matrix is partitioned roughly equally by rows. For example, with 4 ranks and a 100x100 matrix:

- Rank 0: rows 1-25
- Rank 1: rows 26-50
- Rank 2: rows 51-75
- Rank 3: rows 76-100

You can inspect the partition:

```julia
println("Row partition: ", Adist.row_partition)
# Output: [1, 26, 51, 76, 101]
```

## Basic Operations

### Matrix Multiplication

```julia
# Both matrices must be distributed
Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)

# Multiply
Cdist = Adist * Bdist
```

The multiplication automatically handles the necessary communication between ranks.

### Addition and Subtraction

```julia
Cdist = Adist + Bdist
Ddist = Adist - Bdist
```

If A and B have different row partitions, B's rows are redistributed to match A's partition.

### Scalar Multiplication

```julia
Cdist = 2.5 * Adist
Cdist = Adist * 2.5  # Equivalent
```

### Transpose

LinearAlgebraMPI.jl supports both lazy and eager transpose:

```julia
# Lazy transpose (no communication)
At = transpose(Adist)

# Eager transpose (materializes the transposed matrix)
plan = TransposePlan(Adist)
At_materialized = execute_plan!(plan, Adist)
```

Lazy transposes are automatically materialized when needed in operations.

### Adjoint (Conjugate Transpose)

For complex matrices:

```julia
Adist = SparseMatrixMPI{ComplexF64}(A)
Aadj = Adist'  # Conjugate transpose (lazy)
```

### Computing Norms

```julia
# Frobenius norm (default)
f_norm = norm(Adist)

# 1-norm (sum of absolute values)
one_norm = norm(Adist, 1)

# Infinity norm (maximum absolute value)
inf_norm = norm(Adist, Inf)

# General p-norm
p_norm = norm(Adist, 3)

# Operator norms
col_norm = opnorm(Adist, 1)   # Max column sum
row_norm = opnorm(Adist, Inf) # Max row sum
```

## Running MPI Programs

### Command Line

```bash
mpiexec -n 4 julia --project=. my_program.jl
```

### Program Structure

A typical LinearAlgebraMPI.jl program follows this pattern:

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Create matrices (identical on all ranks)
A = create_my_matrix()  # Your matrix creation function
B = create_my_matrix()

# Distribute
Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)

# Compute
Cdist = Adist * Bdist

# Get results (e.g., norm is computed globally)
result_norm = norm(Cdist)

if rank == 0
    println("Result norm: $result_norm")
end

MPI.Finalize()
```

## Performance Tips

### Reuse Communication Plans

For repeated operations with the same sparsity pattern, LinearAlgebraMPI.jl automatically caches communication plans:

```julia
# First multiplication creates and caches the plan
C1 = Adist * Bdist

# Subsequent multiplications with same A, B reuse the cached plan
C2 = Adist * Bdist  # Uses cached plan - much faster
```

### Clear Cache When Done

If you're done with a set of matrices and want to free memory:

```julia
clear_plan_cache!()
```

### Use Deterministic Test Data

For testing, avoid random matrices since they'll differ across ranks:

```julia
# Bad - different on each rank
A = sprand(100, 100, 0.01)

# Good - deterministic, identical on all ranks
I = [1:100; 1:99; 2:100]
J = [1:100; 2:100; 1:99]
V = [2.0*ones(100); -0.5*ones(99); -0.5*ones(99)]
A = sparse(I, J, V, 100, 100)
```

## Next Steps

- See [Examples](@ref) for more detailed usage examples
- Read the [API Reference](@ref) for complete function documentation
- Understand the [Internals](@ref) for implementation details
