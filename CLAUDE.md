# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commit Messages

Do NOT add "Co-Authored-By" lines or any other self-attribution to commit messages. Do NOT advertise Claude or Anthropic in commits. Keep commit messages focused on describing the changes only.

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

# GPU tests run automatically when Metal.jl is available
# Tests are parameterized over (scalar type, backend) configurations

# Precompile the package
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

## MPI Configuration

By default, MPI.jl uses MPItrampoline_jll. On some Linux clusters, this causes MUMPS to hang during the solve phase. If you experience hangs with multi-rank MUMPS tests, switch to MPICH_jll:

```julia
using MPIPreferences
MPIPreferences.use_jll_binary("MPICH_jll")
```

This creates/updates `LocalPreferences.toml` (which is gitignored). Restart Julia after changing MPI preferences.

## GPU Support

GPU acceleration is supported via Metal.jl (macOS) or CUDA.jl (Linux/Windows) as package extensions.

### Type Parameters

- `VectorMPI{T,AV}` where `AV` is `Vector{T}` (CPU), `MtlVector{T}` (Metal), or `CuVector{T}` (CUDA)
- `MatrixMPI{T,AM}` where `AM` is `Matrix{T}` (CPU), `MtlMatrix{T}` (Metal), or `CuMatrix{T}` (CUDA)
- `SparseMatrixMPI{T,Ti,AV}` where `AV` is `Vector{T}` (CPU), `MtlVector{T}`, or `CuVector{T}` for the `nzval` array
- Type aliases: `VectorMPI_CPU{T}`, `MatrixMPI_CPU{T}`, `SparseMatrixMPI_CPU{T,Ti}` for CPU-backed types

### Creating Zero Arrays

Use `Base.zeros` with the full parametric type or type alias:

```julia
# CPU zero arrays
v = zeros(VectorMPI{Float64,Vector{Float64}}, 100)
v = zeros(VectorMPI_CPU{Float64}, 100)  # Equivalent using type alias

A = zeros(MatrixMPI{Float64,Matrix{Float64}}, 50, 30)
A = zeros(MatrixMPI_CPU{Float64}, 50, 30)

S = zeros(SparseMatrixMPI{Float64,Int,Vector{Float64}}, 100, 100)
S = zeros(SparseMatrixMPI_CPU{Float64,Int}, 100, 100)

# GPU zero arrays (requires Metal.jl or CUDA.jl loaded)
using Metal
v_gpu = zeros(VectorMPI{Float32,MtlVector{Float32}}, 100)
A_gpu = zeros(MatrixMPI{Float32,MtlMatrix{Float32}}, 50, 30)

# Or with CUDA
using CUDA
v_gpu = zeros(VectorMPI{Float64,CuVector{Float64}}, 100)
A_gpu = zeros(MatrixMPI{Float64,CuMatrix{Float64}}, 50, 30)
```

### CPU Staging

MPI communication always uses CPU buffers (no GPU-aware MPI). GPU data is staged through CPU:

1. GPU vector data copied to CPU staging buffer
2. MPI communication on CPU buffers
3. Results copied back to GPU

Plans (`VectorPlan`, `DenseMatrixVectorPlan`, `DenseTransposeVectorPlan`) include:
- `gathered::AV` - buffer matching input type
- `gathered_cpu::Vector{T}` - CPU staging buffer
- `send_bufs`, `recv_bufs` - always CPU for MPI

### Sparse Operations with GPU Vectors

Sparse matrices remain on CPU (Julia's `SparseMatrixCSC` doesn't support GPU arrays). For `A * x` where `x` is GPU:
1. Gather `x` elements via CPU staging
2. Compute sparse multiply on CPU
3. Copy result to GPU via `_create_output_like()`

### Extension Files

- `ext/LinearAlgebraMPIMetalExt.jl` - Metal extension with `mtl()` and `cpu()` functions
- `ext/LinearAlgebraMPICUDAExt.jl` - CUDA extension with `cu()` and `cpu()` functions, plus cuDSS multi-GPU solver
- Loaded automatically when `using Metal` or `using CUDA` before `using LinearAlgebraMPI`

### CUDA-Specific: cuDSS Multi-GPU Solver

The CUDA extension includes `CuDSSFactorizationMPI` for distributed sparse direct solves using NVIDIA's cuDSS library with NCCL inter-GPU communication:

```julia
using CUDA, MPI
MPI.Init()
using LinearAlgebraMPI

# Each MPI rank should use a different GPU
CUDA.device!(MPI.Comm_rank(MPI.COMM_WORLD) % length(CUDA.devices()))

# Create factorization (LDLT for symmetric, LU for general)
F = cudss_ldlt(A)  # or cudss_lu(A)
x = F \ b
finalize!(F)  # Required: clean up cuDSS resources
```

**Important cuDSS notes:**
- Requires cuDSS 0.4+ with MGMN (Multi-GPU Multi-Node) support
- NCCL communicator is bootstrapped automatically from MPI
- `finalize!(F)` must be called to avoid MPI desync during cleanup
- Known issue: tridiagonal matrices with 3+ rows per rank may fail (cuDSS bug reported to NVIDIA)

### Writing Unified CPU/GPU Functions

**IMPORTANT:** Never write separate CPU and GPU code paths with `if AV <: Vector` or `if A.nzval isa Vector` branches. Use unified helper functions instead.

**No mixed CPU/GPU operations:** Operations between CPU and GPU arrays are forbidden. Both operands must be on the same backend. Functions should error on mixed backends:
```julia
if (A_is_cpu != B_is_cpu)
    error("Mixed CPU/GPU operations not supported")
end
```

**Helper functions for unified code:**

1. `_values_to_backend(cpu_values::Vector, template)` - Convert CPU values to template's backend:
   - CPU template: returns `cpu_values` directly (no copy)
   - GPU template: `copyto!(similar(template, T, length(cpu_values)), cpu_values)`

2. `_to_target_backend(v::Vector, ::Type{AV})` - Convert CPU index vector to target type:
   - `Type{Vector{T}}`: returns `v` directly (no copy)
   - `Type{MtlVector{T}}` or `Type{CuVector{T}}`: returns GPU copy

**Pattern for result construction (unified):**
```julia
# Values: use helper (no-op for CPU, copy for GPU)
nzval = _values_to_backend(nzval_cpu, A.nzval)

# Structure arrays: cache once, reuse forever
# For CPU: caches reference to original (no allocation)
# For GPU: caches GPU copy (allocated once)
if plan.cached_rowptr_target === nothing
    plan.cached_rowptr_target = _to_target_backend(plan.rowptr, AV)
end
if plan.cached_colval_target === nothing
    plan.cached_colval_target = _to_target_backend(plan.colval, AV)
end
```

**Pattern for values that change each call (e.g., gathered B values in A*B):**
```julia
# First call: create cache (reference for CPU, GPU buffer for GPU)
if plan.cached_values === nothing
    plan.cached_values = _values_to_backend(plan.cpu_values, A.nzval)
end
# Sync: no-op for CPU (cache === source), copy for GPU
if !(plan.cached_values === plan.cpu_values)
    copyto!(plan.cached_values, plan.cpu_values)
end
```

## Architecture

LinearAlgebraMPI implements distributed sparse and dense matrix operations using MPI for parallel computing across multiple ranks. Supports both `Float64` and `ComplexF64` element types.

### Core design principle

Distributed matrices (sparse and dense) and vectors should not be allgathered in the main library, although gathering constructors are available to the user. It is normal to have to move data from one rank to another, and for this, point-to-point communication (e.g. Isend/Irecv) should be favored. For communication that is performance-sensitive, like algebraic operations between matrices, a plan should be generated and cached based upon the hash of the relevant structures.

### MPI programmiing pitfalls.

Many operations in this module are collective and should not be run on a subset of all the ranks. The programming pattern `if rank == 0 println(...)` is almost always wrong and one should use instead `println(io0(),...)`, without the `if rank == 0`, or else MPI desynchronization is almost guaranteed.

### Core Data Structures

**SparseMatrixCSR{T,Ti}** (Type Alias)
- `SparseMatrixCSR{T,Ti} = Transpose{T, SparseMatrixCSC{T,Ti}}` - type alias for CSR storage
- In Julia, `Transpose{SparseMatrixCSC}` has a **dual life**:
  - **Semantic view**: A lazy transpose of a CSC matrix (what `transpose(A)` returns)
  - **Storage view**: Row-major (CSR) access to sparse data
- Use `SparseMatrixCSR` when intent is CSR storage, use `transpose(A)` for mathematical transpose
- `SparseMatrixCSR(A::SparseMatrixCSC)` converts CSC to CSR representing the **same** matrix
- `SparseMatrixCSC(A::SparseMatrixCSR)` converts CSR to CSC representing the **same** matrix
- For `B = SparseMatrixCSR(A)`, `B[i,j] == A[i,j]` (same matrix, different storage)

**SparseMatrixMPI{T,Ti,AV}**
- Rows are partitioned across MPI ranks
- Type parameters: `T` = element type, `Ti` = index type, `AV` = array type for values (`Vector{T}` or `MtlVector{T}`)
- Local rows stored in CSR-like format with separate arrays:
  - `rowptr::Vector{Ti}` - row pointers (always CPU)
  - `colval::Vector{Ti}` - LOCAL column indices (1:ncols_compressed, always CPU)
  - `nzval::AV` - nonzero values (can be CPU or GPU)
  - `nrows_local::Int` - number of local rows
  - `ncols_compressed::Int` - = length(col_indices)
- `row_partition`: Array of size `nranks + 1` defining which rows each rank owns (1-indexed boundaries)
- `col_partition`: Array of size `nranks + 1` defining column partition (used for transpose operations)
- `col_indices`: Sorted global column indices that appear in the local part (local→global mapping)
- `structural_hash`: Optional Blake3 hash of the matrix structure (computed lazily via `_ensure_hash`)
- `cached_transpose`: Cached materialized transpose (invalidated on modification)
- `cached_symmetric`: Cached result of `issymmetric` check

**MatrixMPI{T,AM}**
- Distributed dense matrix partitioned by rows across MPI ranks
- Type parameter `AM<:AbstractMatrix{T}`: `Matrix{T}` (CPU) or `MtlMatrix{T}` (GPU)
- `A::AM`: Local rows stored directly (NOT transposed), size = `(local_nrows, ncols)`
- `row_partition`: Array of size `nranks + 1` defining which rows each rank owns (always CPU)
- `col_partition`: Array of size `nranks + 1` defining column partition (always CPU)
- `structural_hash`: Optional Blake3 hash (computed lazily)

**VectorMPI{T,AV}**
- Distributed dense vector partitioned across MPI ranks
- Type parameter `AV<:AbstractVector{T}`: `Vector{T}` (CPU) or `MtlVector{T}` (GPU)
- `partition`: Array of size `nranks + 1` defining which elements each rank owns (always CPU)
- `v::AV`: Local vector elements owned by this rank
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

**VectorPlan{T,AV}**
- Communication plan for gathering vector elements needed for `A * x`
- Type parameter `AV`: matches input vector type for GPU support
- Gathers `x[A.col_indices]` from appropriate ranks based on `x.partition`
- Memoized based on structural hashes of A and x plus array type (see `_vector_plan_cache`)
- `gathered::AV`: buffer matching input type, `gathered_cpu::Vector{T}`: CPU staging
- `send_bufs`, `recv_bufs`: always CPU for MPI communication

### Factorization

Factorization uses MUMPS (MUltifrontal Massively Parallel Solver) with distributed matrix input (ICNTL(18)=3).

**MUMPSFactorizationMPI{Tin, AVin, Tinternal}** (internal type, not exported)
- Wraps a MUMPS object for distributed factorization
- Type parameters: `Tin` = input element type, `AVin` = input array type, `Tinternal` = MUMPS-compatible type (Float64 or ComplexF64)
- Created by `lu(A)` for general matrices or `ldlt(A)` for symmetric matrices
- Automatically converts input to MUMPS-compatible types and back (e.g., Float32 GPU → Float64 CPU → solve → Float32 GPU)
- Stores COO arrays (irn_loc, jcn_loc, a_loc) to prevent GC while MUMPS holds pointers

```julia
F = lu(A)
x = F \ b
```

### Local Constructors

For efficient construction when data is already distributed:
- `VectorMPI_local(v_local)`: Create from local vector portion
- `SparseMatrixMPI_local(SparseMatrixCSR(local_csc))`: Create from local rows in CSR format
- `SparseMatrixMPI_local(transpose(AT_local))`: Alternative using explicit transpose wrapper
- `MatrixMPI_local(A_local)`: Create from local dense rows

These infer the global partition via MPI.Allgather of local sizes.

When building from triplets (I, J, V), the efficient pattern is to build M^T directly as CSC by swapping indices, then wrap in lazy transpose for CSR:
```julia
AT_local = sparse(local_J, local_I, local_V, ncols, local_nrows)  # M^T as CSC
SparseMatrixMPI_local(transpose(AT_local))  # M in CSR format
```
This avoids an unnecessary physical transpose operation.

### Matrix Multiplication Flow

1. **Plan creation** (`MatrixPlan` constructor): Uses `Alltoall` and `Alltoallv` to exchange row requests and sparse structure (colptr, rowval)
2. **Value exchange** (`execute_plan!`): Point-to-point `Isend`/`Irecv` to send actual matrix values
3. **Local computation**: `C^T = B^T * A^T` using reindexed local sparse matrices
4. **Result construction**: Directly wraps local result with inherited row partition from A

### Matrix-Vector Multiplication Flow

For `y = A * x` where `A::SparseMatrixMPI` and `x::VectorMPI`:

1. **Plan creation** (`VectorPlan` constructor): Uses `Alltoall` to exchange element request counts, then point-to-point to exchange indices
2. **Value exchange** (`execute_plan!`): Point-to-point `Isend`/`Irecv` to gather `x[A.col_indices]` into a local `gathered` vector
3. **Local computation**: Compute `transpose(_get_csc(A)) * gathered` where `_get_csc(A)` reconstructs a SparseMatrixCSC from the internal CSR arrays
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

`transpose(A)` returns `Transpose{T, SparseMatrixMPI{T,Ti,AV}}` (lazy wrapper). Materialization happens automatically when needed:
- `transpose(A) * transpose(B)` → `transpose(B * A)` (stays lazy)
- `transpose(A) * B` or `A * transpose(B)` → materializes via `TransposePlan`
- `SparseMatrixMPI(transpose(A))` → explicitly materialize the transpose (cached bidirectionally)

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
