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

**Important**: All MPI ranks must have the same matrix **size** when constructing distributed types. However, each rank only extracts its own local rows, so the actual **data** only needs to be correct for each rank's portion.

### Understanding Row Partitioning

The matrix is partitioned roughly equally by rows. For example, with 4 ranks and a 100x100 matrix:

- Rank 0: rows 1-25
- Rank 1: rows 26-50
- Rank 2: rows 51-75
- Rank 3: rows 76-100

### Internal Storage: CSR Format

Internally, each rank stores its local rows in CSR (Compressed Sparse Row) format using the `SparseMatrixCSR` type. This enables efficient row-wise iteration, which is essential for a row-partitioned distributed matrix.

In Julia, `SparseMatrixCSR{T,Ti}` is a type alias for `Transpose{T, SparseMatrixCSC{T,Ti}}`. This type has a dual interpretation:
- **Semantic view**: A lazy transpose of a CSC matrix
- **Storage view**: Row-major (CSR) access to the data

You don't need to worry about this for normal usage - it's handled automatically. But if you're accessing the internal storage (e.g., `A.A.parent`), be aware that it stores the transposed data in CSC format, which gives CSR access through the wrapper.

### Efficient Local-Only Construction

For large matrices, you can avoid replicating data across all ranks by only populating each rank's local portion:

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Global dimensions
m, n = 1000, 1000

# Compute which rows this rank owns
rows_per_rank = div(m, nranks)
remainder = mod(m, nranks)
my_row_start = 1 + rank * rows_per_rank + min(rank, remainder)
my_row_end = my_row_start + rows_per_rank - 1 + (rank < remainder ? 1 : 0)

# Create a sparse matrix with correct size, but only populate local rows
I, J, V = Int[], Int[], Float64[]
for i in my_row_start:my_row_end
    # Example: tridiagonal matrix
    if i > 1
        push!(I, i); push!(J, i-1); push!(V, -1.0)
    end
    push!(I, i); push!(J, i); push!(V, 2.0)
    if i < m
        push!(I, i); push!(J, i+1); push!(V, -1.0)
    end
end
A = sparse(I, J, V, m, n)

# The constructor extracts only local rows - other rows are ignored
Adist = SparseMatrixMPI{Float64}(A)
```

This pattern is useful when:
- The global matrix is too large to fit in memory on each rank
- You're generating matrix entries programmatically
- You want to minimize memory usage during construction

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

```julia
# Transpose is lazy (no communication until needed)
At = transpose(Adist)

# Use in multiplication - automatically materializes when needed
Cdist = At * Bdist
```

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

println(io0(), "Result norm: $result_norm")
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

For testing with the simple "replicate everywhere" pattern, avoid random matrices since they'll differ across ranks:

```julia
# Bad - different random values on each rank
A = sprand(100, 100, 0.01)

# Good - deterministic formula, same on all ranks
I = [1:100; 1:99; 2:100]
J = [1:100; 2:100; 1:99]
V = [2.0*ones(100); -0.5*ones(99); -0.5*ones(99)]
A = sparse(I, J, V, 100, 100)
```

Alternatively, use the [local-only construction pattern](#Efficient-Local-Only-Construction) where each rank generates only its own rows.

## MUMPS Solver Threading

LinearAlgebraMPI uses the MUMPS (MUltifrontal Massively Parallel Solver) library for sparse direct solves via `lu()` and `ldlt()`. MUMPS has two independent threading mechanisms that can be tuned for performance.

### Threading Parameters

**OpenMP threads (`OMP_NUM_THREADS`)**
- Controls MUMPS's algorithm-level parallelism
- The multifrontal method builds an elimination tree of "frontal matrices"
- OpenMP threads process independent subtrees in parallel
- This is coarse-grained: different threads work on different parts of the matrix

**BLAS threads (`OPENBLAS_NUM_THREADS`)**
- Controls parallelism inside dense matrix operations
- When MUMPS factors a frontal matrix, it calls BLAS routines (DGEMM, etc.)
- OpenBLAS can parallelize these dense operations
- This is fine-grained: threads cooperate on the same dense block

**Note on BLAS libraries**: Julia and MUMPS use separate OpenBLAS libraries (`libopenblas64_.dylib` for Julia's ILP64 interface, `libopenblas.dylib` for MUMPS's LP64 interface). Both libraries read `OPENBLAS_NUM_THREADS` at initialization, so this environment variable affects both.

### Recommended Configuration

For behavior that closely matches Julia's built-in sparse solver (UMFPACK):

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=<number_of_cores>
```

This configuration uses only BLAS-level threading, which is the same strategy Julia's built-in solver uses.

### Performance Comparison

The following table compares MUMPS (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=10`) against Julia's built-in sparse solver (also using the same settings) on a 2D Laplacian problem. Benchmarks were run on a 2025 M4 MacBook Pro with 10 CPU cores:

| n | Julia (ms) | MUMPS (ms) | Ratio |
|---|------------|------------|-------|
| 9 | 0.004 | 0.041 | 9.7x |
| 100 | 0.023 | 0.070 | 3.0x |
| 992 | 0.269 | 0.418 | 1.6x |
| 10,000 | 4.28 | 5.60 | 1.31x |
| 99,856 | 51.2 | 56.9 | 1.11x |
| 1,000,000 | 665 | 666 | 1.0x |

Key observations:
- At small problem sizes, MUMPS has initialization overhead (~0.04ms)
- At large problem sizes (n ≥ 100,000), MUMPS is within 11% of Julia's built-in solver
- At n = 1,000,000, MUMPS matches Julia's speed exactly (1.0x ratio)

### Default Behavior

For optimal performance, set threading environment variables **before starting Julia**:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=10  # or your number of CPU cores
julia your_script.jl
```

This is necessary because OpenBLAS creates its thread pool during library initialization, before LinearAlgebraMPI has a chance to configure it. LinearAlgebraMPI attempts to set sensible defaults programmatically, but this may not always take effect if the thread pool is already initialized.

You can also add these to your shell profile (`.bashrc`, `.zshrc`, etc.) or Julia's `startup.jl`:

```julia
# In ~/.julia/config/startup.jl
ENV["OMP_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = string(Sys.CPU_THREADS)
```

### Advanced: Combined Threading

For some problems, combining OpenMP and BLAS threads can be faster:

```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

This configuration achieved 14% faster performance than Julia's built-in solver on a 1M DOF 2D Laplacian in testing. However, the optimal configuration depends on your specific problem structure and hardware.

**Important caveat**: `OPENBLAS_NUM_THREADS` is a process-wide setting that affects both MUMPS and Julia's built-in sparse solver (UMFPACK). If you set `OPENBLAS_NUM_THREADS=4` to optimize MUMPS, Julia's built-in solver will also be limited to 4 BLAS threads, potentially slowing it down. This is another reason why `OMP_NUM_THREADS=1` with full BLAS threading is the recommended default—it ensures consistent behavior for all solvers in your program.

## Next Steps

- See [Examples](@ref) for more detailed usage examples
- Read the [API Reference](@ref) for complete function documentation
