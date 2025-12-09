# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Run tests (spawns MPI processes automatically)
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a specific MPI test directly (for debugging)
mpiexec -n 4 julia --project=. test/test_matrix_multiplication.jl

# Precompile the package
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

## Architecture

LinearAlgebraMPI implements distributed sparse matrix operations using MPI for parallel computing across multiple ranks.

### Core Data Structures

**SparseMatrixMPI{T}**
- Rows are partitioned across MPI ranks
- Stores the **transpose** of local rows as `AT::SparseMatrixCSC` (columns in AT = local rows)
- `row_partition`: Array of size `nranks + 1` defining which rows each rank owns
- `col_indices`: Column indices that appear in the local part (used to determine which rows of B are needed during multiplication)
- `structural_hash`: Blake3 hash of the matrix structure for consistency verification

**MatrixPlan{T}**
- Communication plan for gathering rows from another SparseMatrixMPI
- Created before matrix multiplication to set up sparse structure and buffers
- Contains pre-allocated send buffers and the gathered sparse matrix structure

### Matrix Multiplication Flow

1. **Plan creation** (`MatrixPlan` constructor): Uses `Alltoall` and `Alltoallv` to exchange row requests and sparse structure (colptr, rowval)
2. **Value exchange**: Second round of `Alltoallv` to send actual matrix values
3. **Local computation**: Sparse matrix-matrix product on gathered data
4. **Result construction**: Directly wraps local result with inherited row partition from A

### Key Design Patterns

- All ranks must have identical copies of input matrices when constructing `SparseMatrixMPI` from a global matrix
- Row ownership is determined by `find_owner()` using the `row_partition` array
- Communication uses MPI collective operations with `UBuffer` (`Alltoall`, `Alltoallv`)
- Tests run under `mpiexec` via the test harness in `test/runtests.jl`
- Test matrices should be deterministic (not random) to ensure consistency across MPI ranks
