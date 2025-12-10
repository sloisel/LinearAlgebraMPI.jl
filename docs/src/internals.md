# Internals

This page describes the internal architecture and implementation details of LinearAlgebraMPI.jl. Understanding these details can help with debugging, performance optimization, and extending the library.

## Data Structures

### SparseMatrixMPI Storage

`SparseMatrixMPI{T}` stores a distributed sparse matrix with the following key design decisions:

1. **Row partitioning**: Rows are distributed roughly equally across MPI ranks
2. **Transposed storage**: Local rows are stored as the **transpose** (`AT::SparseMatrixCSC`)
3. **Global indices**: Column indices in `AT.rowval` are global (not local) indices

#### Why Store the Transpose?

Storing the transpose (columns of `AT` = rows of `A`) has several advantages:

- **Efficient row access**: CSC format provides O(1) access to column slices. Since we store `AT`, we get O(1) access to rows of `A`.
- **Natural for multiplication**: In `A * B`, we need to iterate over rows of `A` and columns of `B`. With `AT` stored, iterating over rows of `A` is efficient.
- **Simplified communication**: When gathering rows from `B` for multiplication, we can directly send column slices of `B.AT`.

#### Field Descriptions

```julia
struct SparseMatrixMPI{T}
    structural_hash::NTuple{32,UInt8}  # Blake3 hash for plan caching
    row_partition::Vector{Int}          # [1, r1_end+1, r2_end+1, ..., nrows+1]
    col_partition::Vector{Int}          # Similar for columns (for transpose)
    col_indices::Vector{Int}            # Unique sorted column indices in local part
    AT::SparseMatrixCSC{T,Int}          # Transposed local rows
end
```

**Invariants:**
- `row_partition[1] = 1`
- `row_partition[end] = nrows + 1`
- `size(AT, 2) = row_partition[rank+2] - row_partition[rank+1]` (local row count)
- `AT.rowval` contains global column indices

### Structural Hashing

Each `SparseMatrixMPI` has a 256-bit Blake3 hash of its structure. This hash is:

1. **Computed collectively**: Uses `Allgather` to ensure identical hash on all ranks
2. **Structure-only**: Includes partition, indices, `colptr`, `rowval` (not values)
3. **Used for plan caching**: Same structure = same communication pattern

The hash computation:
1. Each rank hashes its local data (row_partition, col_indices, AT.colptr, AT.rowval)
2. All local hashes are gathered via `MPI.Allgather`
3. The gathered hashes are hashed together to produce the global hash

## Matrix Multiplication Algorithm

### Overview

Computing `C = A * B` requires three phases:

1. **Plan creation**: Determine which rows of `B` are needed and exchange sparsity structure
2. **Value exchange**: Send/receive the actual matrix values
3. **Local computation**: Multiply the gathered data locally

### Phase 1: Plan Creation

When creating a `MatrixPlan(A, B)`:

1. **Identify needed rows**: `A.col_indices` tells us which columns of `A` (= rows of `B`) we need
2. **Determine owners**: For each needed row, find which rank owns it using `B.row_partition`
3. **Exchange requests**: Use `Alltoall` to exchange counts, then point-to-point for row lists
4. **Exchange structure**: Send `colptr` and `rowval` for requested rows
5. **Build gathered structure**: Construct `plan.AT` with zeros (sparsity pattern only)
6. **Setup buffers**: Pre-allocate send/receive buffers and track offsets

### Phase 2: Value Exchange (execute_plan!)

1. **Local copy**: Copy values for locally-owned rows directly into `plan.AT.nzval`
2. **Pack and send**: For each remote rank, pack requested values into buffer and `Isend`
3. **Receive**: `Irecv` values from ranks we requested data from
4. **Unpack**: Copy received values into `plan.AT.nzval` at correct offsets

### Phase 3: Local Computation

```julia
# C^T = B^T * A^T = plan.AT * A.AT_reindexed
result_AT = plan.AT * A_AT_reindexed
```

The key insight is that we need to reindex `A.AT.rowval` from global indices to local indices (1:n_gathered) since `plan.AT` has only the gathered rows, not all of `B`.

### Plan Caching

Plans are cached in a global dictionary keyed by `(A.structural_hash, B.structural_hash, T)`. This means:

- Repeated multiplications with the same structure reuse plans
- The structure can be different values (plans only depend on sparsity)
- Plans must be cleared manually with `clear_plan_cache!()` when done

## Transpose Algorithm

### TransposePlan Creation

The transpose of `A` (with row_partition `R` and col_partition `C`) has:
- `row_partition = C` (columns become rows)
- `col_partition = R` (rows become columns)

Algorithm:
1. **Categorize nonzeros**: For each `A[i,j]`, determine which rank owns row `j` in `A^T`
2. **Exchange structure**: Send `(row, col)` pairs to destination ranks
3. **Build sparse structure**: Construct CSC format for the transposed matrix
4. **Setup communication**: Track permutations for scattering received values

### execute_plan! for Transpose

1. Copy local values (entries that stay on the same rank)
2. Pack and send values to destination ranks
3. Receive values and scatter into result using pre-computed permutation

## Addition/Subtraction

For `A + B` or `A - B`:

1. **Get addition plan**: Gather B's rows to match A's partition
2. **Execute plan**: Redistribute B's values
3. **Local operation**: Use SparseArrays' built-in `A.AT + plan.AT` or `A.AT - plan.AT`

The result inherits A's partition.

## Lazy Transpose

Lazy transpose uses Julia's `LinearAlgebra.Transpose` wrapper:

```julia
transpose(A::SparseMatrixMPI{T}) = Transpose(A)
```

Operations with lazy transposes are handled by specialized methods:

- `transpose(A) * transpose(B)` returns `transpose(B * A)` (computed lazily)
- `transpose(A) * B` materializes `A^T` first, then multiplies
- `A * transpose(B)` materializes `B^T` first, then multiplies
- `a * transpose(A)` returns `transpose(a * A)`

## Communication Patterns

### Alltoall Pattern

Used for exchanging counts and sizes:
```julia
recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)
```

### Point-to-Point Pattern

Used for variable-size data:
```julia
# Non-blocking send
req = MPI.Isend(msg, comm; dest=r, tag=1)

# Non-blocking receive
req = MPI.Irecv!(buf, comm; source=r, tag=1)

# Wait for completion
MPI.Waitall(reqs)
```

### Collective Reduction

Used for norms:
```julia
global_sum = MPI.Allreduce(local_sum, MPI.SUM, comm)
global_max = MPI.Allreduce(local_max, MPI.MAX, comm)
```

## Memory Management

### Pre-allocated Buffers

Plans pre-allocate all buffers during construction:
- `send_bufs`: One per destination rank
- `recv_bufs`: One per source rank
- `plan.AT.nzval`: Output values array

This makes `execute_plan!` allocation-free for the communication portions.

### Plan Reuse

Plans store all communication metadata:
- Which ranks to communicate with
- Buffer sizes and offsets
- Value permutations

Reusing plans avoids the expensive setup phase.

## Testing Considerations

### Deterministic Test Data

Tests must use deterministic data to ensure all ranks have identical input:

```julia
# Bad: different on each rank
A = sprand(100, 100, 0.01)

# Good: deterministic
I = [1:n; 1:n-1; 2:n]
J = [1:n; 2:n; 1:n-1]
V = [2.0*ones(n); -0.5*ones(n-1); -0.5*ones(n-1)]
A = sparse(I, J, V, n, n)
```

### Test Harness

Tests run under `mpiexec` via the test harness:
```julia
mpiexec -n 4 julia --project=. test/test_*.jl
```

The `runtests.jl` file automates this using `run_mpi_test()`.

## Performance Considerations

### When to Clear Cache

Clear the plan cache when:
- You're done with a set of matrices
- Memory usage is a concern
- You're about to work with different matrices

```julia
clear_plan_cache!()
```

### Optimal Matrix Sizes

- **Too small**: Communication overhead dominates
- **Too large per rank**: Memory limits
- **Sweet spot**: Large enough matrices with good sparsity distribution

### Sparsity Pattern Impact

- **Banded matrices**: Good locality, less communication
- **Random sparse**: More communication, but still efficient
- **Block diagonal**: Excellent if blocks align with partitions

## Extending the Library

### Adding New Operations

To add a new operation:

1. Determine the communication pattern
2. Create a plan type if needed
3. Implement the plan constructor
4. Implement `execute_plan!`
5. Implement the high-level operation (e.g., `Base.:*(...)`)
6. Add caching if the operation will be repeated

### Supporting New Element Types

The library is generic over element type `T`. To support a new type:

1. Ensure it works with SparseArrays
2. Ensure MPI can serialize it (or provide custom serialization)
3. Ensure Blake3Hash can hash it

## Debugging Tips

### Verify Identical Input

```julia
A_hash = compute_structural_hash(...)
# Hash should be identical on all ranks
```

### Check Partitions

```julia
println("Rank $rank: rows $(A.row_partition[rank+1]) to $(A.row_partition[rank+2]-1)")
```

### Trace Communication

```julia
for r in plan.rank_ids
    println("Rank $rank sends to $r: $(length(plan.send_bufs[i])) values")
end
```
