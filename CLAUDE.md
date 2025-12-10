# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Run all tests (spawns MPI processes automatically via test harness)
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a specific MPI test directly (for debugging)
mpiexec -n 4 julia --project=. test/test_matrix_multiplication.jl
mpiexec -n 4 julia --project=. test/test_transpose.jl
mpiexec -n 4 julia --project=. test/test_addition.jl
mpiexec -n 4 julia --project=. test/test_lazy_transpose.jl
mpiexec -n 4 julia --project=. test/test_vector_multiplication.jl

# Precompile the package
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

## Architecture

LinearAlgebraMPI implements distributed sparse matrix operations using MPI for parallel computing across multiple ranks. Supports both `Float64` and `ComplexF64` element types.

### Core Data Structures

**SparseMatrixMPI{T}**
- Rows are partitioned across MPI ranks
- Stores the **transpose** of local rows as `AT::SparseMatrixCSC` (columns in AT = local rows)
- `row_partition`: Array of size `nranks + 1` defining which rows each rank owns (1-indexed boundaries)
- `col_partition`: Array of size `nranks + 1` defining column partition (used for transpose operations)
- `col_indices`: Column indices that appear in the local part (determines which rows of B are needed during multiplication)
- `structural_hash`: Blake3 hash of the matrix structure for consistency verification across ranks

**MatrixPlan{T}**
- Communication plan for gathering rows from another SparseMatrixMPI
- Memoized based on structural hashes of A and B (see `_plan_cache`)
- Pre-allocates all send/receive buffers for allocation-free `execute_plan!` calls

**TransposePlan{T}**
- Communication plan for computing matrix transpose
- Redistributes nonzeros based on `col_partition` becoming `row_partition`
- Also pre-allocates buffers for reusable execution

**VectorMPI{T}**
- Distributed dense vector partitioned across MPI ranks
- `partition`: Array of size `nranks + 1` defining which elements each rank owns (1-indexed boundaries)
- `v`: Local vector elements owned by this rank
- `structural_hash`: Blake3 hash of the partition for caching
- Can have any partition (not required to match matrix partitions)

**VectorPlan{T}**
- Communication plan for gathering vector elements needed for `A * x`
- Gathers `x[A.col_indices]` from appropriate ranks based on `x.partition`
- Memoized based on structural hashes of A and x (see `_vector_plan_cache`)
- Pre-allocates all send/receive buffers for allocation-free `execute_plan!` calls

### Matrix Multiplication Flow

1. **Plan creation** (`MatrixPlan` constructor): Uses `Alltoall` and `Alltoallv` to exchange row requests and sparse structure (colptr, rowval)
2. **Value exchange** (`execute_plan!`): Point-to-point `Isend`/`Irecv` to send actual matrix values
3. **Local computation**: `C^T = B^T * A^T` using reindexed local sparse matrices
4. **Result construction**: Directly wraps local result with inherited row partition from A

### Matrix-Vector Multiplication Flow

For `y = A * x` where `A::SparseMatrixMPI` and `x::VectorMPI`:

1. **Plan creation** (`VectorPlan` constructor): Uses `Alltoall` to exchange element request counts, then point-to-point to exchange indices
2. **Value exchange** (`execute_plan!`): Point-to-point `Isend`/`Irecv` to gather `x[A.col_indices]` into a local `gathered` vector
3. **Local computation**: Reindex `A.AT` to use local indices, then compute `transpose(A_AT_reindexed) * gathered`
4. **Result construction**: Result vector `y` inherits `A.row_partition`

Note: Uses `transpose()` (not adjoint `'`) to correctly handle complex values without conjugation.

### Vector Operations

- `conj(v)` - Returns new VectorMPI with conjugated values (materialized)
- `transpose(v)` - Returns lazy `Transpose` wrapper
- `v'` (adjoint) - Returns `transpose(conj(v))` where `conj(v)` is materialized
- `transpose(v) * A` - Computes `transpose(transpose(A) * v)`, returns transposed VectorMPI
- `v' * A` - Computes `transpose(transpose(A) * conj(v))`, returns transposed VectorMPI

### Lazy Transpose

`transpose(A)` returns `Transpose{T, SparseMatrixMPI{T}}` (lazy wrapper). Materialization happens automatically when needed:
- `transpose(A) * transpose(B)` → `transpose(B * A)` (stays lazy)
- `transpose(A) * B` or `A * transpose(B)` → materializes via `TransposePlan`

### Key Design Patterns

- All ranks must have identical copies of input matrices/vectors when constructing `SparseMatrixMPI` or `VectorMPI` from global data
- Row/element ownership: `searchsortedlast(partition, index) - 1` gives the owning rank (0-indexed)
- Communication uses MPI collective operations (`Alltoall`, `Alltoallv`) and point-to-point (`Isend`, `Irecv`)
- Communication tags: 1-3 for MatrixPlan, 10-11 for TransposePlan, 20-21 for VectorPlan
- Tests run under `mpiexec` via the test harness in `test/runtests.jl`
- Test matrices must be deterministic (not random) to ensure consistency across MPI ranks
