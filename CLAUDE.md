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
mpiexec -n 4 julia --project=. test/test_dense_matrix.jl
mpiexec -n 4 julia --project=. test/test_sparse_api.jl
mpiexec -n 4 julia --project=. test/test_blocks.jl
mpiexec -n 4 julia --project=. test/test_utilities.jl
mpiexec -n 4 julia --project=. test/test_local_constructors.jl
mpiexec -n 4 julia --project=. test/test_indexing.jl
mpiexec -n 4 julia --project=. test/test_factorization.jl

# Precompile the package
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

## Architecture

LinearAlgebraMPI implements distributed sparse and dense matrix operations using MPI for parallel computing across multiple ranks. Supports both `Float64` and `ComplexF64` element types.

### Core design principle

Distributed matrices (sparse and dense) and vectors should not be allgathered in the main library, although gathering constructors are available to the user. It is normal to have to move data from one rank to another, and for this, point-to-point communication (e.g. Isend/Irecv) should be favored. For communication that is performance-sensitive, like algebraic operations between matrices, a plan should be generated and cached based upon the hash of the relevant structures.

### MPI programmiing pitfalls.

Many operations in this module are collective and should not be run on a subset of all the ranks. The programming pattern `if rank == 0 println(...)` is almost always wrong and one should use instead `println(io0(),...)`, without the `if rank == 0`, or else MPI desynchronization is almost guaranteed.

### Core Data Structures

**SparseMatrixMPI{T}**
- Rows are partitioned across MPI ranks
- `A::Transpose{T,SparseMatrixCSC{T,Int}}`: Local rows wrapped in a Transpose for type clarity
  - `A.parent` is the underlying CSC storage with shape `(length(col_indices), local_nrows)`
  - Columns in `A.parent` correspond to local rows; this layout enables efficient row-wise iteration
  - Storage is **compressed**: `A.parent.rowval` uses local column indices (1:length(col_indices)), not global
- `row_partition`: Array of size `nranks + 1` defining which rows each rank owns (1-indexed boundaries)
- `col_partition`: Array of size `nranks + 1` defining column partition (used for transpose operations)
- `col_indices`: Sorted global column indices that appear in the local part (local→global mapping)
- `structural_hash`: Optional Blake3 hash of the matrix structure (computed lazily via `_ensure_hash`)
- `cached_transpose`: Cached materialized transpose (invalidated on modification)

**MatrixMPI{T}**
- Distributed dense matrix partitioned by rows across MPI ranks
- `A::Matrix{T}`: Local rows stored directly (NOT transposed), size = `(local_nrows, ncols)`
- `row_partition`: Array of size `nranks + 1` defining which rows each rank owns
- `col_partition`: Array of size `nranks + 1` defining column partition (for transpose)
- `structural_hash`: Optional Blake3 hash (computed lazily)

**VectorMPI{T}**
- Distributed dense vector partitioned across MPI ranks
- `partition`: Array of size `nranks + 1` defining which elements each rank owns (1-indexed boundaries)
- `v`: Local vector elements owned by this rank
- `structural_hash`: Blake3 hash of the partition (computed on construction)
- Can have any partition (not required to match matrix partitions)

**MatrixPlan{T}**
- Communication plan for gathering rows from another SparseMatrixMPI
- Memoized based on structural hashes of A and B (see `_plan_cache`)
- Pre-allocates all send/receive buffers for allocation-free `execute_plan!` calls

**TransposePlan{T}**
- Communication plan for computing matrix transpose
- Redistributes nonzeros based on `col_partition` becoming `row_partition`
- Pre-allocates buffers for reusable execution

**VectorPlan{T}**
- Communication plan for gathering vector elements needed for `A * x`
- Gathers `x[A.col_indices]` from appropriate ranks based on `x.partition`
- Memoized based on structural hashes of A and x (see `_vector_plan_cache`)
- Pre-allocates all send/receive buffers for allocation-free `execute_plan!` calls

### Factorization

Factorization uses MUMPS (MUltifrontal Massively Parallel Solver) with distributed matrix input (ICNTL(18)=3).

**MUMPSFactorizationMPI{T}** (internal type, not exported)
- Wraps a MUMPS object for distributed factorization
- Created by `lu(A)` for general matrices or `ldlt(A)` for symmetric matrices
- Stores COO arrays (irn_loc, jcn_loc, a_loc) to prevent GC while MUMPS holds pointers

**Automatic cleanup:** Factorization objects are automatically cleaned up when garbage collected.
The cleanup is synchronized across MPI ranks when the next factorization is created. Example:

```julia
F = lu(A)
x = F \ b
# F is automatically cleaned up when GC'd and next factorization is created
```

Manual `finalize!(F)` is still available for explicit control (must be called on all ranks together).

### Local Constructors

For efficient construction when data is already distributed:
- `VectorMPI_local(v_local)`: Create from local vector portion
- `SparseMatrixMPI_local(transpose(AT_local))`: Create from local rows
- `MatrixMPI_local(A_local)`: Create from local dense rows

These infer the global partition via MPI.Allgather of local sizes.

### Matrix Multiplication Flow

1. **Plan creation** (`MatrixPlan` constructor): Uses `Alltoall` and `Alltoallv` to exchange row requests and sparse structure (colptr, rowval)
2. **Value exchange** (`execute_plan!`): Point-to-point `Isend`/`Irecv` to send actual matrix values
3. **Local computation**: `C^T = B^T * A^T` using reindexed local sparse matrices
4. **Result construction**: Directly wraps local result with inherited row partition from A

### Matrix-Vector Multiplication Flow

For `y = A * x` where `A::SparseMatrixMPI` and `x::VectorMPI`:

1. **Plan creation** (`VectorPlan` constructor): Uses `Alltoall` to exchange element request counts, then point-to-point to exchange indices
2. **Value exchange** (`execute_plan!`): Point-to-point `Isend`/`Irecv` to gather `x[A.col_indices]` into a local `gathered` vector
3. **Local computation**: Compute `transpose(A.A.parent) * gathered` (A.A.parent already uses local indices)
4. **Result construction**: Result vector `y` inherits `A.row_partition`

Note: Uses `transpose()` (not adjoint `'`) to correctly handle complex values without conjugation.

### Vector Operations

- `conj(v)` - Returns new VectorMPI with conjugated values (materialized)
- `transpose(v)` - Returns lazy `Transpose` wrapper
- `v'` (adjoint) - Returns `transpose(conj(v))` where `conj(v)` is materialized
- `transpose(v) * A` - Computes `transpose(transpose(A) * v)`, returns transposed VectorMPI
- `v' * A` - Computes `transpose(transpose(A) * conj(v))`, returns transposed VectorMPI
- `u + v`, `u - v` - Automatic partition alignment if partitions differ

### Lazy Transpose

`transpose(A)` returns `Transpose{T, SparseMatrixMPI{T}}` (lazy wrapper). Materialization happens automatically when needed:
- `transpose(A) * transpose(B)` → `transpose(B * A)` (stays lazy)
- `transpose(A) * B` or `A * transpose(B)` → materializes via `TransposePlan`

### Indexing Operations

All distributed types support getindex and setindex! with cross-rank communication:
- `v[i]`, `v[i:j]`, `v[indices::VectorMPI]` for VectorMPI
- `A[i,j]`, `A[i:j, k:l]`, `A[:, j]`, `A[rows::VectorMPI, cols]` for MatrixMPI and SparseMatrixMPI
- Setting values triggers structural modification for sparse matrices when inserting new nonzeros

### Block Operations

- `cat(A, B; dims=...)` - Concatenate matrices/vectors along specified dimensions
- `blockdiag(A, B, ...)` - Create block diagonal matrix from multiple matrices

### Key Design Patterns

- All ranks must have identical copies of input matrices/vectors when constructing from global data
- Row/element ownership: `searchsortedlast(partition, index) - 1` gives the owning rank (0-indexed)
- Communication uses MPI collective operations (`Alltoall`, `Alltoallv`) and point-to-point (`Isend`, `Irecv`)
- Communication tags are organized by operation type (1-3, 10-11, 20-22, 30-35, 40-51, 60-125)
- Tests run under `mpiexec` via the test harness in `test/runtests.jl`
- Test matrices must be deterministic (not random) to ensure consistency across MPI ranks
