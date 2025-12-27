# SparseMatrixMPI type and sparse matrix operations

# ============================================================================
# Merge-sort style helpers for sorted column index arrays
# ============================================================================

"""
    merge_sorted_unique!(result, a, b)

Merge two sorted arrays into a sorted array of unique elements.
Returns the result array (which may be shorter than allocated).
"""
function merge_sorted_unique!(result::Vector{Int}, a::Vector{Int}, b::Vector{Int})
    i, j, k = 1, 1, 0
    @inbounds while i <= length(a) && j <= length(b)
        if a[i] < b[j]
            k += 1
            result[k] = a[i]
            i += 1
        elseif a[i] > b[j]
            k += 1
            result[k] = b[j]
            j += 1
        else  # equal
            k += 1
            result[k] = a[i]
            i += 1
            j += 1
        end
    end
    @inbounds while i <= length(a)
        k += 1
        result[k] = a[i]
        i += 1
    end
    @inbounds while j <= length(b)
        k += 1
        result[k] = b[j]
        j += 1
    end
    return resize!(result, k)
end

"""
    merge_sorted_unique(a, b)

Merge two sorted arrays into a new sorted array of unique elements.
O(n+m) time, no sorting or Dict needed.
"""
function merge_sorted_unique(a::Vector{Int}, b::Vector{Int})
    result = Vector{Int}(undef, length(a) + length(b))
    return merge_sorted_unique!(result, a, b)
end

"""
    build_subset_mapping!(mapping, subset, superset)

Build a mapping from subset indices to superset positions.
Both arrays must be sorted, and subset must be a subset of superset.
mapping[i] = position of subset[i] in superset.
O(|subset| + |superset|) time with linear scan.
"""
function build_subset_mapping!(mapping::Vector{Int}, subset::Vector{Int}, superset::Vector{Int})
    j = 1  # position in superset
    @inbounds for i in 1:length(subset)
        while superset[j] < subset[i]
            j += 1
        end
        # Now superset[j] == subset[i]
        mapping[i] = j
    end
    return mapping
end

"""
    build_subset_mapping(subset, superset)

Build a mapping from subset indices to superset positions.
Returns a new vector where mapping[i] = position of subset[i] in superset.
"""
function build_subset_mapping(subset::Vector{Int}, superset::Vector{Int})
    mapping = Vector{Int}(undef, length(subset))
    return build_subset_mapping!(mapping, subset, superset)
end

# ============================================================================

"""
    compute_structural_hash(row_partition, col_indices, rowptr, colval, comm) -> Blake3Hash

Compute a structural hash that is identical across all ranks.

1. Hash local data: row_partition, col_indices, rowptr, colval
2. Allgather all local hashes
3. Hash the gathered hashes to produce a global hash
"""
function compute_structural_hash(row_partition::Vector{Int}, col_indices::Vector{Int},
    rowptr::Vector{<:Integer}, colval::Vector{<:Integer}, comm::MPI.Comm)::Blake3Hash
    # Step 1: Compute rank-local hash
    # IMPORTANT: Prefix each vector with its length to disambiguate boundaries.
    # Without length prefixes, different structures could hash to the same value
    # if the concatenated bytes happen to match.
    ctx = Blake3Ctx()
    update!(ctx, reinterpret(UInt8, Int[length(row_partition)]))
    update!(ctx, reinterpret(UInt8, row_partition))
    update!(ctx, reinterpret(UInt8, Int[length(col_indices)]))
    update!(ctx, reinterpret(UInt8, col_indices))
    update!(ctx, reinterpret(UInt8, Int[length(rowptr)]))
    update!(ctx, reinterpret(UInt8, rowptr))
    update!(ctx, reinterpret(UInt8, Int[length(colval)]))
    update!(ctx, reinterpret(UInt8, colval))
    local_hash = digest(ctx)

    # Step 2: Allgather all local hashes
    all_hashes = MPI.Allgather(local_hash, comm)

    # Step 3: Hash them together to produce global hash
    ctx2 = Blake3Ctx()
    update!(ctx2, all_hashes)
    return Blake3Hash(digest(ctx2))
end

# Backwards-compatible overload that takes SparseMatrixCSC
function compute_structural_hash(row_partition::Vector{Int}, col_indices::Vector{Int},
    AT::SparseMatrixCSC, comm::MPI.Comm)::Blake3Hash
    return compute_structural_hash(row_partition, col_indices, AT.colptr, AT.rowval, comm)
end

"""
    compress_AT(AT::SparseMatrixCSC{T,Ti}, col_indices::Vector{Int}) where {T,Ti}

Compress AT from global column indices to local indices 1:length(col_indices).
Returns a new SparseMatrixCSC with m = length(col_indices).

The col_indices array provides the local→global mapping: col_indices[local_idx] = global_col.
"""
function compress_AT(AT::SparseMatrixCSC{T,Ti}, col_indices::Vector{Int}) where {T,Ti}
    if isempty(col_indices)
        return SparseMatrixCSC(0, AT.n, AT.colptr, Int[], T[])
    end
    # col_indices is sorted, use binary search instead of Dict
    compressed_rowval = [searchsortedfirst(col_indices, r) for r in AT.rowval]
    return SparseMatrixCSC(length(col_indices), AT.n, AT.colptr, compressed_rowval, AT.nzval)
end

"""
    reindex_to_union_cached(AT, col_to_union_map, union_size)

Optimized reindex using precomputed mapping vector.
col_to_union_map[local_idx] gives the union index for local column index local_idx.
"""
function reindex_to_union_cached(AT::SparseMatrixCSC{T,Ti}, col_to_union_map::Vector{Int}, union_size::Int) where {T,Ti}
    if isempty(AT.rowval)
        return SparseMatrixCSC(union_size, AT.n, AT.colptr, Int[], T[])
    end
    new_rowval = [col_to_union_map[r] for r in AT.rowval]
    return SparseMatrixCSC(union_size, AT.n, AT.colptr, new_rowval, AT.nzval)
end

"""
    compress_AT_cached(AT, compress_map)

Optimized compress_AT using precomputed mapping vector.
compress_map[global_idx] gives the local index for global column index global_idx.
"""
function compress_AT_cached(AT::SparseMatrixCSC{T,Ti}, compress_map::Vector{Int}, local_size::Int) where {T,Ti}
    if isempty(AT.rowval)
        return SparseMatrixCSC(local_size, AT.n, AT.colptr, Int[], T[])
    end
    compressed_rowval = [compress_map[r] for r in AT.rowval]
    return SparseMatrixCSC(local_size, AT.n, AT.colptr, compressed_rowval, AT.nzval)
end

"""
    _rebuild_AT_with_insertions(AT, col_indices, insertions, row_offset) -> (new_AT, new_col_indices)

Rebuild AT with a batch of insertions for structural setindex!.

Each insertion is (global_i, global_j, val) where:
- global_i is the global row index (must be in this rank's partition)
- global_j is the global column index (may or may not be in col_indices yet)
- val is the value to set

Returns updated AT and col_indices. If insertions target existing entries, values are updated.
If insertions target new structural positions, entries are added.

Note: AT stores rows in its columns (transpose layout), so:
- AT column = local row index
- AT row = local column index (into col_indices)
"""
function _rebuild_AT_with_insertions(AT::SparseMatrixCSC{T,Ti}, col_indices::Vector{Int},
    insertions::Vector{Tuple{Int,Int,T}},
    row_offset::Int) where {T,Ti}
    if isempty(insertions)
        return AT, col_indices
    end

    n_local_rows = AT.n  # Number of local rows

    # Collect all global columns needed (existing + new)
    new_global_cols = Set{Int}()
    for (_, global_j, _) in insertions
        push!(new_global_cols, global_j)
    end

    # Build expanded col_indices (merge existing and new, maintain sorted order)
    expanded_col_indices = sort(unique(vcat(col_indices, collect(new_global_cols))))

    # Collect all entries: (AT_col, AT_row, val) = (local_row, local_col_in_expanded, val)
    # Using a Dict to handle duplicates (later values win)
    entries = Dict{Tuple{Int,Int},T}()

    # Add existing entries from AT (reindex to expanded col_indices)
    # expanded_col_indices is sorted, use binary search
    for at_col in 1:n_local_rows
        for k in AT.colptr[at_col]:(AT.colptr[at_col+1]-1)
            old_local_col = AT.rowval[k]
            global_col = col_indices[old_local_col]
            new_local_col = searchsortedfirst(expanded_col_indices, global_col)
            entries[(at_col, new_local_col)] = AT.nzval[k]
        end
    end

    # Add new insertions (may overwrite existing)
    for (global_i, global_j, val) in insertions
        local_row = global_i - row_offset + 1  # AT column
        local_col = searchsortedfirst(expanded_col_indices, global_j)  # AT row
        entries[(local_row, local_col)] = val
    end

    # Build new CSC arrays using standard COO→CSC algorithm
    # Sort entries by (AT_col, AT_row)
    sorted_entries = sort(collect(entries), by=x -> (x[1][1], x[1][2]))

    nnz = length(sorted_entries)

    # Count entries per column
    col_counts = zeros(Int, n_local_rows)
    for ((at_col, _), _) in sorted_entries
        col_counts[at_col] += 1
    end

    # Build colptr from cumulative counts
    new_colptr = ones(Int, n_local_rows + 1)
    for j in 1:n_local_rows
        new_colptr[j+1] = new_colptr[j] + col_counts[j]
    end

    # Fill rowval and nzval
    new_rowval = Vector{Int}(undef, nnz)
    new_nzval = Vector{T}(undef, nnz)
    col_pos = copy(new_colptr[1:end-1])  # Current position in each column

    for ((at_col, at_row), val) in sorted_entries
        pos = col_pos[at_col]
        new_rowval[pos] = at_row
        new_nzval[pos] = val
        col_pos[at_col] += 1
    end

    new_AT = SparseMatrixCSC(length(expanded_col_indices), n_local_rows, new_colptr, new_rowval, new_nzval)

    return new_AT, expanded_col_indices
end

"""
    SparseMatrixMPI{T,Ti,AV}

A distributed sparse matrix partitioned by rows across MPI ranks.

# Type Parameters
- `T`: Element type (e.g., `Float64`, `ComplexF64`)
- `Ti`: Index type (e.g., `Int`, `Int32`), defaults to `Int`
- `AV<:AbstractVector{T}`: Storage type for nonzero values (`Vector{T}` for CPU, `MtlVector{T}` for GPU)

# Fields
- `structural_hash::Blake3Hash`: 256-bit Blake3 hash of the structural pattern
- `row_partition::Vector{Int}`: Row partition boundaries, length = nranks + 1
- `col_partition::Vector{Int}`: Column partition boundaries, length = nranks + 1 (placeholder for transpose)
- `col_indices::Vector{Int}`: Global column indices that appear in the local part (local→global mapping)
- `rowptr::Vector{Ti}`: Row pointers for CSR format (always CPU)
- `colval::Vector{Ti}`: LOCAL column indices 1:length(col_indices) for each nonzero (always CPU)
- `nzval::AV`: Nonzero values (CPU or GPU)
- `nrows_local::Int`: Number of local rows
- `ncols_compressed::Int`: Number of unique columns = length(col_indices)
- `cached_transpose`: Cached materialized transpose (bidirectionally linked)

# Invariants
- `col_indices`, `row_partition`, and `col_partition` are sorted
- `row_partition[nranks+1]` = total number of rows
- `col_partition[nranks+1]` = total number of columns
- `nrows_local == row_partition[rank+1] - row_partition[rank]` (number of local rows)
- `ncols_compressed == length(col_indices)` (compressed column dimension)
- `colval` contains local indices in `1:ncols_compressed`
- `rowptr` has length `nrows_local + 1`

# Storage Details
The local rows are stored in CSR format (Compressed Sparse Row), which enables efficient
row-wise iteration - essential for a row-partitioned distributed matrix.

The CSR storage consists of:
- `rowptr`: Row pointers where row i has nonzeros at positions rowptr[i]:(rowptr[i+1]-1)
- `colval`: LOCAL column indices (1:ncols_compressed), not global indices
- `nzval`: Nonzero values
- `col_indices[local_idx]` maps local→global column indices

This compression avoids "hypersparse" storage where the column dimension would be
the global number of columns even if only a few columns have nonzeros locally.

# GPU Support
Structure arrays (`rowptr`, `colval`) always stay on CPU for MPI communication and indexing.
Only `nzval` can live on GPU, with type determined by `AV`:
- `Vector{T}`: CPU storage
- `MtlVector{T}`: Metal GPU storage (macOS)

Use `mtl(A)` to convert to GPU, `cpu(A)` to convert back.
"""
mutable struct SparseMatrixMPI{T,Ti,AV<:AbstractVector{T}} <: AbstractMatrix{T}
    structural_hash::OptionalBlake3Hash
    row_partition::Vector{Int}      # Always CPU
    col_partition::Vector{Int}      # Always CPU
    col_indices::Vector{Int}        # Always CPU (local→global column mapping)
    rowptr::Vector{Ti}              # Always CPU - row pointers (for MPI operations)
    colval::Vector{Ti}              # Always CPU - LOCAL column indices (for MPI operations)
    nzval::AV                       # CPU or GPU - nonzero values
    nrows_local::Int                # Number of local rows
    ncols_compressed::Int           # = length(col_indices)
    cached_transpose::Union{Nothing,SparseMatrixMPI{T,Ti,AV}}
    cached_symmetric::Union{Nothing,Bool}  # Cache for issymmetric result
    # Structure arrays on target backend (same backend as nzval)
    # For CPU: these are the same objects as rowptr/colval
    # For GPU: these are GPU copies
    rowptr_target::AbstractVector{Ti}
    colval_target::AbstractVector{Ti}
end

# Type alias for CPU version (backwards compatibility)
const SparseMatrixMPI_CPU{T,Ti} = SparseMatrixMPI{T,Ti,Vector{T}}

"""
    _get_csc(A::SparseMatrixMPI) -> SparseMatrixCSC

Reconstruct a SparseMatrixCSC view of the local storage.
The returned CSC has shape (ncols_compressed, nrows_local) where:
- columns correspond to local rows
- rows correspond to compressed column indices

This is the transpose of the logical matrix structure - use transpose(_get_csc(A))
for sparse matrix-vector multiply.

Note: For GPU arrays, this copies nzval to CPU first.
"""
function _get_csc(A::SparseMatrixMPI{T,Ti,AV})::SparseMatrixCSC{T,Ti} where {T,Ti,AV}
    nzval_cpu = _ensure_cpu(A.nzval)
    return SparseMatrixCSC(A.ncols_compressed, A.nrows_local, A.rowptr, A.colval, nzval_cpu)
end

"""
    _get_csc_cpu(A::SparseMatrixMPI, nzval_cpu) -> SparseMatrixCSC

Reconstruct a SparseMatrixCSC using pre-converted CPU nzval.
Avoids redundant GPU→CPU copy when caller has already ensured CPU nzval.
"""
function _get_csc_cpu(A::SparseMatrixMPI{T,Ti}, nzval_cpu::Vector{T})::SparseMatrixCSC{T,Ti} where {T,Ti}
    return SparseMatrixCSC(A.ncols_compressed, A.nrows_local, A.rowptr, A.colval, nzval_cpu)
end

"""
    SparseMatrixMPI{T}(A::SparseMatrixCSC{T,Ti}; comm=MPI.COMM_WORLD, row_partition=..., col_partition=...) where {T,Ti}

Create a SparseMatrixMPI from a global sparse matrix A, partitioning it by rows across MPI ranks.

Each rank extracts only its local rows from `A`, so:

- **Simple usage**: Pass identical `A` to all ranks
- **Efficient usage**: Pass a matrix with correct `size(A)` on all ranks,
  but only populate the rows that each rank owns (other rows are ignored)

# Keyword Arguments
- `comm::MPI.Comm`: MPI communicator (default: `MPI.COMM_WORLD`)
- `row_partition::Vector{Int}`: Row partition boundaries (default: `uniform_partition(size(A,1), nranks)`)
- `col_partition::Vector{Int}`: Column partition boundaries (default: `uniform_partition(size(A,2), nranks)`)

Use `uniform_partition(n, nranks)` to compute custom partitions.
"""
function SparseMatrixMPI{T}(A::SparseMatrixCSC{T,Ti};
    comm::MPI.Comm=MPI.COMM_WORLD,
    row_partition::Vector{Int}=uniform_partition(size(A, 1), MPI.Comm_size(comm)),
    col_partition::Vector{Int}=uniform_partition(size(A, 2), MPI.Comm_size(comm))) where {T,Ti}
    rank = MPI.Comm_rank(comm)

    # Local row range (1-indexed, Julia style)
    row_start = row_partition[rank+1]
    row_end = row_partition[rank+2] - 1

    # Extract local rows and convert to CSR format for efficient row iteration
    local_A = A[row_start:row_end, :]

    # Delegate to SparseMatrixMPI_local which handles compression and col_indices
    return SparseMatrixMPI_local(SparseMatrixCSR(local_A); comm=comm, col_partition=col_partition)
end

"""
    SparseMatrixMPI_local(A_local::SparseMatrixCSR{T,Ti}; comm=MPI.COMM_WORLD, col_partition=...) where {T,Ti}
    SparseMatrixMPI_local(A_local::Adjoint{T,SparseMatrixCSC{T,Ti}}; comm=MPI.COMM_WORLD, col_partition=...) where {T,Ti}

Create a SparseMatrixMPI from a local sparse matrix on each rank.

Unlike `SparseMatrixMPI{T}(A_global)` which takes a global matrix and partitions it,
this constructor takes only the local rows of the matrix that each rank owns.
The row partition is computed by gathering the local row counts from all ranks.

The input `A_local` must be a `SparseMatrixCSR{T,Ti}` (or `Adjoint` of `SparseMatrixCSC{T,Ti}`) where:
- `A_local.parent.n` = number of local rows on this rank
- `A_local.parent.m` = global number of columns (must match on all ranks)
- `A_local.parent.rowval` = global column indices

All ranks must have local matrices with the same number of columns (block widths must match).
A collective error is raised if the column counts don't match.

Note: For `Adjoint` inputs, the values are conjugated to match the adjoint semantics.

# Keyword Arguments
- `comm::MPI.Comm`: MPI communicator (default: `MPI.COMM_WORLD`)
- `col_partition::Vector{Int}`: Column partition boundaries (default: `uniform_partition(A_local.parent.m, nranks)`)

# Example
```julia
# Create local rows in CSR format
# Rank 0 owns rows 1-2 of a 5×3 matrix, Rank 1 owns rows 3-5
local_csc = sparse([1, 1, 2], [1, 2, 3], [1.0, 2.0, 3.0], 2, 3)  # 2 local rows, 3 cols
A = SparseMatrixMPI_local(SparseMatrixCSR(local_csc))
```
"""
function SparseMatrixMPI_local(A_local::SparseMatrixCSR{T,Ti};
    comm::MPI.Comm=MPI.COMM_WORLD,
    col_partition::Vector{Int}=uniform_partition(A_local.parent.m, MPI.Comm_size(comm))) where {T,Ti}
    nranks = MPI.Comm_size(comm)

    AT_local = A_local.parent  # The underlying CSC storage
    local_nrows = AT_local.n   # Columns in CSC = rows in matrix
    ncols_global = AT_local.m  # Rows in CSC = columns in matrix (global)

    # Gather local row counts and column counts from all ranks
    local_info = Int32[local_nrows, ncols_global]
    all_info = MPI.Allgather(local_info, comm)

    # Extract row counts and verify column counts match
    all_row_counts = [all_info[2*(r-1)+1] for r in 1:nranks]
    all_col_counts = [all_info[2*(r-1)+2] for r in 1:nranks]

    # Check that all column counts are the same
    if !all(c == all_col_counts[1] for c in all_col_counts)
        error("SparseMatrixMPI_local: All ranks must have the same number of columns. " *
              "Got column counts: $all_col_counts")
    end

    # Build row_partition from row counts (inferred via Allgather)
    row_partition = Vector{Int}(undef, nranks + 1)
    row_partition[1] = 1
    for r in 1:nranks
        row_partition[r+1] = row_partition[r] + all_row_counts[r]
    end

    # Identify which columns have nonzeros in our local part
    # AT_local.rowval contains global column indices
    col_indices = isempty(AT_local.rowval) ? Int[] : unique(sort(AT_local.rowval))

    # Compress AT_local: convert global column indices to local indices 1:length(col_indices)
    compressed_AT = compress_AT(AT_local, col_indices)

    # Extract explicit CSR arrays from the compressed AT (which is in CSC format storing the transpose)
    # In CSC format: colptr→rowptr, rowval→colval, nzval stays the same
    rowptr = compressed_AT.colptr
    colval = compressed_AT.rowval
    nzval = compressed_AT.nzval
    nrows_local = local_nrows
    ncols_compressed = length(col_indices)

    # Structural hash computed lazily on first use via _ensure_hash
    # For CPU matrices, rowptr_target and colval_target are the same objects as rowptr/colval
    return SparseMatrixMPI{T,Ti,Vector{T}}(nothing, row_partition, col_partition, col_indices,
        rowptr, colval, nzval, nrows_local, ncols_compressed, nothing, nothing, rowptr, colval)
end

# Adjoint version: conjugate values during construction
function SparseMatrixMPI_local(A_local::Adjoint{T,SparseMatrixCSC{T,Ti}};
    comm::MPI.Comm=MPI.COMM_WORLD,
    col_partition::Vector{Int}=uniform_partition(A_local.parent.m, MPI.Comm_size(comm))) where {T,Ti}
    # Convert adjoint to transpose with conjugated values
    AT_parent = A_local.parent
    AT_conj = SparseMatrixCSC(AT_parent.m, AT_parent.n, copy(AT_parent.colptr),
        copy(AT_parent.rowval), conj.(AT_parent.nzval))
    return SparseMatrixMPI_local(transpose(AT_conj); comm=comm, col_partition=col_partition)
end

"""
    MatrixPlan{T,Ti}

A communication plan for gathering rows from an SparseMatrixMPI.

# Fields
- `rank_ids::Vector{Int}`: Ranks that requested data from us (0-indexed)
- `send_ranges::Vector{Vector{UnitRange{Int}}}`: For each rank, ranges into B.nzval to send
- `send_bufs::Vector{Vector{T}}`: Pre-allocated send buffers for each rank
- `send_reqs::Vector{MPI.Request}`: Pre-allocated send request handles
- `recv_rank_ids::Vector{Int}`: Ranks we need to receive data from (0-indexed)
- `recv_bufs::Vector{Vector{T}}`: Pre-allocated receive buffers for each rank
- `recv_reqs::Vector{MPI.Request}`: Pre-allocated receive request handles
- `recv_offsets::Vector{Int}`: Starting offsets into AT.nzval for each recv_rank_ids
- `local_ranges::Vector{Tuple{UnitRange{Int}, Int}}`: (src_range, dst_offset) for local copies
- `AT::SparseMatrixCSC{T,Ti}`: Transposed matrix structure for gathered rows (values zeroed)
"""
mutable struct MatrixPlan{T,Ti,AIV<:AbstractVector{Ti}}
    rank_ids::Vector{Int}
    send_ranges::Vector{Vector{UnitRange{Int}}}
    send_bufs::Vector{Vector{T}}
    send_reqs::Vector{MPI.Request}
    recv_rank_ids::Vector{Int}
    recv_bufs::Vector{Vector{T}}
    recv_reqs::Vector{MPI.Request}
    recv_offsets::Vector{Int}
    local_ranges::Vector{Tuple{UnitRange{Int},Int}}
    AT::SparseMatrixCSC{T,Ti}
    # Cached hash for product result (computed lazily on first execution)
    product_structural_hash::OptionalBlake3Hash
    product_col_indices::Union{Nothing, Vector{Int}}
    product_row_partition::Union{Nothing, Vector{Int}}
    product_compress_map::Union{Nothing, Vector{Int}}  # global_col -> local_col mapping for compress_AT
    # Symbolic multiplication data (computed lazily on first execution)
    # For C = plan.AT * _get_csc(A), these indices satisfy:
    # C.nzval[Ci[k]] += plan.AT.nzval[Ai[k]] * A.nzval[Bi[k]]
    # Arrays are partitioned into layers where Ci values don't repeat within a layer,
    # allowing parallel execution without atomics.
    # AIV is the index array type (Vector{Ti} for CPU, MtlVector{Ti} for GPU)
    sym_Ai::Union{Nothing, AIV}  # indices into plan.AT.nzval (on target backend)
    sym_Bi::Union{Nothing, AIV}  # indices into A.nzval (on target backend)
    sym_Ci::Union{Nothing, AIV}  # indices into result.nzval (on target backend)
    sym_layer_starts::Union{Nothing, Vector{Ti}}  # layer_starts[l] = first index of layer l (always CPU for kernel control)
    sym_colptr::Union{Nothing, Vector{Ti}}  # colptr for result (always CPU for result construction)
    sym_rowval::Union{Nothing, Vector{Ti}}  # rowval for result (always CPU, compressed local indices)
end

"""
    MatrixPlan(row_indices::Vector{Int}, B::SparseMatrixMPI{T,Ti}, ::Type{AIV}) where {T,Ti,AIV}

Create a communication plan to gather rows specified by row_indices from B.
Assumes row_indices is sorted. AIV is the index array type for symbolic multiply
(Vector{Ti} for CPU, MtlVector{Ti} for GPU).

The plan proceeds in 3 steps:
1. For each row i in row_indices, determine owner. If remote, use isend to request structure.
2. Receive requests from other ranks, add to rank_ids, isend structure responses.
3. Receive structure info, build plan.AT with zeros (sparsity pattern of B[row_indices,:]).
"""
function MatrixPlan(row_indices::Vector{Int}, B::SparseMatrixMPI{T,Ti}, ::Type{AIV}) where {T,Ti,AIV<:AbstractVector{Ti}}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    my_row_start = B.row_partition[rank+1]

    # Step 1: Determine which rows we need from which ranks
    # Group row_indices by owner rank
    rows_needed_from = [Int[] for _ in 1:nranks]
    for row in row_indices
        owner = searchsortedlast(B.row_partition, row) - 1
        push!(rows_needed_from[owner+1], row)
    end

    # Send row requests to remote ranks (aggregate per rank)
    send_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if r != rank && !isempty(rows_needed_from[r+1])
            # Send: [count, row1, row2, ...]
            msg = vcat([length(rows_needed_from[r+1])], rows_needed_from[r+1])
            req = MPI.Isend(msg, comm; dest=r, tag=1)
            push!(send_reqs, req)
        end
    end

    # Step 2: Receive requests from other ranks
    # First, probe to find out who is sending to us
    rank_ids = Int[]  # ranks that requested data from us (0-indexed)
    rows_requested_by = Dict{Int,Vector{Int}}()  # rank => rows requested

    # We need to receive from any rank that has rows_needed_from us
    # Use Iprobe to check for incoming messages
    pending_recvs = nranks - 1 - length(rows_needed_from[rank+1] == [] ? 0 : 0)  # rough estimate

    # Actually, we know exactly who will send to us: ranks r where rows_needed_from[rank+1]
    # on rank r is non-empty. But we don't know that directly.
    # Use a barrier-free approach: each rank sends count first, then we know.

    # Simpler approach: use Alltoall to exchange counts first
    send_counts = [r == rank ? 0 : length(rows_needed_from[r+1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Wait for our sends to complete
    MPI.Waitall(send_reqs)

    # Now receive the actual row requests
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int,Vector{Int}}()
    for r in 0:(nranks-1)
        if r != rank && recv_counts[r+1] > 0
            buf = Vector{Int}(undef, recv_counts[r+1] + 1)  # +1 for count
            req = MPI.Irecv!(buf, comm; source=r, tag=1)
            push!(recv_reqs, req)
            recv_bufs[r] = buf
        end
    end

    MPI.Waitall(recv_reqs)

    # Parse received requests
    for (r, buf) in recv_bufs
        cnt = buf[1]
        rows = buf[2:cnt+1]
        push!(rank_ids, r)
        rows_requested_by[r] = rows
    end
    sort!(rank_ids)

    # Prepare structure responses and send them
    # For each requesting rank, send: [ncols, colptr..., nrowvals, rowvals...]
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int,Vector{Int}}()  # keep buffers alive

    for r in rank_ids
        requested = rows_requested_by[r]
        local_cols = [row - my_row_start + 1 for row in requested]

        ncols = length(local_cols)
        new_colptr = Vector{Int}(undef, ncols + 1)
        new_colptr[1] = 1
        rowvals = Int[]

        for (i, col) in enumerate(local_cols)
            start_idx = B.rowptr[col]
            end_idx = B.rowptr[col+1] - 1
            col_nnz = end_idx - start_idx + 1
            new_colptr[i+1] = new_colptr[i] + col_nnz
            # B.colval contains LOCAL indices, convert to global using B.col_indices
            for idx in start_idx:end_idx
                push!(rowvals, B.col_indices[B.colval[idx]])
            end
        end

        msg = vcat([ncols], new_colptr, [length(rowvals)], rowvals)
        struct_send_bufs[r] = msg
        req = MPI.Isend(msg, comm; dest=r, tag=2)
        push!(struct_send_reqs, req)
    end

    # Step 3: Receive structure info from ranks we requested from
    recv_rank_ids = Int[]  # ranks we receive from (0-indexed)
    struct_recv_bufs = Dict{Int,Vector{Int}}()

    # First exchange structure message sizes
    struct_send_sizes = zeros(Int, nranks)
    for r in rank_ids
        struct_send_sizes[r+1] = length(struct_send_bufs[r])
    end
    struct_recv_sizes = MPI.Alltoall(MPI.UBuffer(struct_send_sizes, 1), comm)

    # Receive structure from ranks we need data from
    struct_recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if r != rank && !isempty(rows_needed_from[r+1])
            push!(recv_rank_ids, r)
            buf = Vector{Int}(undef, struct_recv_sizes[r+1])
            req = MPI.Irecv!(buf, comm; source=r, tag=2)
            push!(struct_recv_reqs, req)
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)
    sort!(recv_rank_ids)

    # Build plan.AT structure
    # Process row_indices in order, combining local and remote structure
    # nrows_AT is global ncols of B (not compressed size)
    nrows_AT = B.col_partition[end] - 1
    n_total_cols = length(row_indices)

    # First pass: count total nnz
    total_nnz = 0
    for row in row_indices
        owner = searchsortedlast(B.row_partition, row) - 1
        if owner == rank
            local_col = row - my_row_start + 1
            total_nnz += B.rowptr[local_col+1] - B.rowptr[local_col]
        else
            # Find position in request to that owner
            req_idx = findfirst(==(row), rows_needed_from[owner+1])
            data = struct_recv_bufs[owner]
            ncols = data[1]
            colptr = data[2:ncols+2]
            total_nnz += colptr[req_idx+1] - colptr[req_idx]
        end
    end

    # Allocate AT structure
    combined_colptr = Vector{Int}(undef, n_total_cols + 1)
    combined_rowval = Vector{Int}(undef, total_nnz)
    combined_nzval = zeros(T, total_nnz)

    combined_colptr[1] = 1
    nnz_idx = 1

    # Track local copy info and recv offsets
    # Track local copy ranges: (src_range, dst_offset)
    local_ranges = Tuple{UnitRange{Int},Int}[]

    # Track receive offsets per rank (consecutive values from each rank)
    recv_val_counts = Dict{Int,Int}()

    # Initialize recv tracking
    for r in recv_rank_ids
        recv_val_counts[r] = 0
    end

    # Second pass: fill structure and track offsets
    for (out_col, row) in enumerate(row_indices)
        owner = searchsortedlast(B.row_partition, row) - 1

        if owner == rank
            # Local row
            local_col = row - my_row_start + 1
            start_idx = B.rowptr[local_col]
            end_idx = B.rowptr[local_col+1] - 1
            col_nnz = end_idx - start_idx + 1

            combined_colptr[out_col+1] = combined_colptr[out_col] + col_nnz
            # B.colval contains LOCAL indices, convert to global using B.col_indices
            for (i, idx) in enumerate(start_idx:end_idx)
                combined_rowval[nnz_idx+i-1] = B.col_indices[B.colval[idx]]
            end
            # Track for local copy
            if col_nnz > 0
                push!(local_ranges, (start_idx:end_idx, nnz_idx))
            end
            nnz_idx += col_nnz
        else
            # Remote row
            req_idx = findfirst(==(row), rows_needed_from[owner+1])
            data = struct_recv_bufs[owner]
            ncols = data[1]
            colptr = data[2:ncols+2]
            rowvals_start = ncols + 4
            rowvals = data[rowvals_start:end]

            col_start = colptr[req_idx]
            col_end = colptr[req_idx+1] - 1
            col_nnz = col_end - col_start + 1

            combined_colptr[out_col+1] = combined_colptr[out_col] + col_nnz
            combined_rowval[nnz_idx:nnz_idx+col_nnz-1] = rowvals[col_start:col_end]
            recv_val_counts[owner] += col_nnz
            nnz_idx += col_nnz
        end
    end

    # Compute recv offsets (values arrive consecutively per rank, in order of rows_needed_from)
    # We need to map where each rank's data goes in AT.nzval
    # Values for rank r go to positions corresponding to rows_needed_from[r+1] in order

    recv_counts_vec = Int[]
    recv_offsets_vec = Int[]

    for r in recv_rank_ids
        push!(recv_counts_vec, recv_val_counts[r])
        # Find the first row from this rank in row_indices to get starting offset
        first_row = rows_needed_from[r+1][1]
        first_col = findfirst(==(first_row), row_indices)
        push!(recv_offsets_vec, combined_colptr[first_col])
    end

    # Prepare send info: for each rank in rank_ids, compute ranges into B.nzval
    send_ranges_vec = Vector{Vector{UnitRange{Int}}}()
    send_bufs = Vector{Vector{T}}()

    for r in rank_ids
        requested = rows_requested_by[r]
        ranges = UnitRange{Int}[]
        total_len = 0
        for row in requested
            local_col = row - my_row_start + 1
            start_idx = B.rowptr[local_col]
            end_idx = B.rowptr[local_col+1] - 1
            if end_idx >= start_idx
                push!(ranges, start_idx:end_idx)
                total_len += end_idx - start_idx + 1
            end
        end
        push!(send_ranges_vec, ranges)
        push!(send_bufs, Vector{T}(undef, total_len))
    end

    # Pre-allocate receive buffers and request vectors
    recv_bufs = [Vector{T}(undef, cnt) for cnt in recv_counts_vec]
    send_reqs = Vector{MPI.Request}(undef, length(rank_ids))
    recv_reqs = Vector{MPI.Request}(undef, length(recv_rank_ids))

    plan_AT = SparseMatrixCSC(nrows_AT, n_total_cols, combined_colptr, combined_rowval, combined_nzval)

    return MatrixPlan{T,Ti,AIV}(
        rank_ids, send_ranges_vec, send_bufs, send_reqs,
        recv_rank_ids, recv_bufs, recv_reqs, recv_offsets_vec,
        local_ranges,
        plan_AT,
        nothing, nothing, nothing, nothing,  # product: hash, col_indices, row_partition, compress_map
        nothing, nothing, nothing, nothing, nothing, nothing  # symbolic multiply: Ai, Bi, Ci, layer_starts, colptr, rowval
    )
end

"""
    _index_array_type(::Type{AV}, ::Type{Ti}) where {AV,Ti}

Map value array type AV to index array type with element type Ti.
For CPU (Vector{T}), returns Vector{Ti}. For GPU arrays, returns the
corresponding GPU array type with Ti elements.
"""
_index_array_type(::Type{Vector{T}}, ::Type{Ti}) where {T,Ti} = Vector{Ti}
# GPU versions are added by extensions

"""
    _to_target_backend(v::Vector{Ti}, ::Type{Vector{T}}) where {Ti,T}

Convert a CPU index vector to the target backend.
For CPU target, returns the same vector (no copy).
For GPU target (e.g., MtlVector{T}), returns a GPU copy.
"""
_to_target_backend(v::Vector{Ti}, ::Type{Vector{T}}) where {Ti,T} = v
# GPU versions are added by extensions

"""
    MatrixPlan(A::SparseMatrixMPI{T,Ti,AV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}

Create a memoized communication plan for A * B.
The plan is cached based on the structural hashes of A, B, and the array type AV.
"""
function MatrixPlan(A::SparseMatrixMPI{T,Ti,AV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    AIV = _index_array_type(AV, Ti)
    key = (_ensure_hash(A), _ensure_hash(B), T, Ti, AV)
    if haskey(_plan_cache, key)
        return _plan_cache[key]::MatrixPlan{T,Ti,AIV}
    end
    plan = MatrixPlan(A.col_indices, B, AIV)
    _plan_cache[key] = plan
    return plan
end

"""
    execute_plan!(plan::MatrixPlan{T,Ti,AIV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AIV,AV}

Execute a communication plan to gather rows from B into plan.AT.
After execution, plan.AT contains the values from B for the requested rows.
This function handles both CPU and GPU matrices by staging GPU data to CPU for MPI.
"""
function execute_plan!(plan::MatrixPlan{T,Ti,AIV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AIV,AV}
    comm = MPI.COMM_WORLD

    # Ensure B.nzval is on CPU for MPI communication
    B_nzval_cpu = _ensure_cpu(B.nzval)

    # Step 1: Copy local values into plan.AT
    for (src_range, dst_off) in plan.local_ranges
        plan.AT.nzval[dst_off:dst_off+length(src_range)-1] = view(B_nzval_cpu, src_range)
    end

    # Step 2: Fill send buffers and send to ranks that requested from us
    for (i, r) in enumerate(plan.rank_ids)
        buf = plan.send_bufs[i]
        offset = 1
        for rng in plan.send_ranges[i]
            n = length(rng)
            buf[offset:offset+n-1] = view(B_nzval_cpu, rng)
            offset += n
        end
        plan.send_reqs[i] = MPI.Isend(buf, comm; dest=r, tag=3)
    end

    # Step 3: Receive values from ranks we need data from
    for (i, r) in enumerate(plan.recv_rank_ids)
        plan.recv_reqs[i] = MPI.Irecv!(plan.recv_bufs[i], comm; source=r, tag=3)
    end

    # Wait for receives to complete
    MPI.Waitall(plan.recv_reqs)

    # Copy received values directly into plan.AT.nzval
    for i in eachindex(plan.recv_rank_ids)
        offset = plan.recv_offsets[i]
        buf = plan.recv_bufs[i]
        plan.AT.nzval[offset:offset+length(buf)-1] = buf
    end

    # Wait for sends to complete
    MPI.Waitall(plan.send_reqs)

    return plan.AT
end
"""
    _compute_symbolic_multiply!(plan::MatrixPlan{T,Ti}, A_parent::SparseMatrixCSC{T,Ti}) where {T,Ti}

Compute the symbolic multiplication for C = plan.AT * A_parent.
Stores in plan: sym_Ai, sym_Bi, sym_Ci, sym_colptr, sym_rowval.

For CSC multiplication C = plan.AT * A_parent where:
- plan.AT has shape (m, k)
- A_parent has shape (k, n)
- C has shape (m, n)

For each column j of C (1 to n):
  For each nonzero A_parent[p, j] at position p:
    For each nonzero plan.AT[i, p] in column p:
      C[i, j] += plan.AT[i, p] * A_parent[p, j]
"""
function _compute_symbolic_multiply!(plan::MatrixPlan{T,Ti,AIV}, A_parent::SparseMatrixCSC{T,Ti}) where {T,Ti,AIV}
    AT = plan.AT  # shape (m, k)
    B = A_parent  # shape (k, n), we call it B for clarity in the multiply

    m = AT.m
    n = B.n

    # First pass: compute symbolic structure of C
    # For each column j, collect unique row indices that will have nonzeros

    # Use a temporary array to mark which rows have nonzeros in current column
    row_marker = zeros(Ti, m)  # 0 means not present, >0 means present
    marker_val = Ti(0)

    # Count nnz per column and collect row indices
    col_nnz = zeros(Ti, n)
    all_row_indices = Vector{Ti}[]  # row indices for each column

    for j in 1:n
        marker_val += 1
        row_indices_j = Ti[]

        # For each nonzero B[p, j]
        for bp in B.colptr[j]:(B.colptr[j+1]-1)
            p = B.rowval[bp]  # column index in AT

            # For each nonzero AT[i, p]
            for ap in AT.colptr[p]:(AT.colptr[p+1]-1)
                i = AT.rowval[ap]

                # Mark row i if not already marked for this column
                if row_marker[i] != marker_val
                    row_marker[i] = marker_val
                    push!(row_indices_j, i)
                end
            end
        end

        # Sort row indices for CSC format
        sort!(row_indices_j)
        push!(all_row_indices, row_indices_j)
        col_nnz[j] = length(row_indices_j)
    end

    # Build colptr
    colptr = Vector{Ti}(undef, n + 1)
    colptr[1] = 1
    for j in 1:n
        colptr[j+1] = colptr[j] + col_nnz[j]
    end
    nnz_C = colptr[n+1] - 1

    # Build rowval
    rowval = Vector{Ti}(undef, nnz_C)
    idx = 1
    for j in 1:n
        for i in all_row_indices[j]
            rowval[idx] = i
            idx += 1
        end
    end

    # Build row_to_idx mapping for each column (to find C[i,j] index quickly)
    # For column j, row_to_idx[i] = index into C.nzval, or 0 if not present
    row_to_idx = zeros(Ti, m)

    # Second pass: compute index triplets (Ai, Bi, Ci)
    Ai = Ti[]
    Bi = Ti[]
    Ci = Ti[]

    for j in 1:n
        # Build row_to_idx for this column
        for idx in colptr[j]:(colptr[j+1]-1)
            row_to_idx[rowval[idx]] = idx
        end

        # For each nonzero B[p, j]
        for bp in B.colptr[j]:(B.colptr[j+1]-1)
            p = B.rowval[bp]

            # For each nonzero AT[i, p]
            for ap in AT.colptr[p]:(AT.colptr[p+1]-1)
                i = AT.rowval[ap]
                c_idx = row_to_idx[i]

                push!(Ai, ap)
                push!(Bi, bp)
                push!(Ci, c_idx)
            end
        end

        # Clear row_to_idx for this column
        for idx in colptr[j]:(colptr[j+1]-1)
            row_to_idx[rowval[idx]] = 0
        end
    end

    # Partition into layers where Ci values don't repeat within a layer.
    # This allows parallel execution without atomics.
    if isempty(Ci)
        layer_starts = Ti[1]
    else
        # Assign each triplet to a layer using greedy coloring
        # layer_of[k] = which layer triplet k belongs to
        n_triplets = length(Ci)
        layer_of = zeros(Ti, n_triplets)

        # For each output position, track which layer it was last used in
        # (0 means not used yet)
        last_layer_used = zeros(Ti, nnz_C)
        num_layers = Ti(0)

        for k in 1:n_triplets
            c = Ci[k]
            # Find smallest layer where c hasn't been used
            # Since last_layer_used[c] is the last layer where c appeared,
            # we need layer = last_layer_used[c] + 1
            layer = last_layer_used[c] + 1
            layer_of[k] = layer
            last_layer_used[c] = layer
            num_layers = max(num_layers, layer)
        end

        # Reorder triplets by layer
        # First, count how many triplets in each layer
        layer_counts = zeros(Ti, num_layers)
        for k in 1:n_triplets
            layer_counts[layer_of[k]] += 1
        end

        # Compute layer_starts (cumulative sum)
        layer_starts = Vector{Ti}(undef, num_layers + 1)
        layer_starts[1] = 1
        for l in 1:num_layers
            layer_starts[l+1] = layer_starts[l] + layer_counts[l]
        end

        # Build reordered arrays
        # next_pos[l] = next position to write for layer l
        next_pos = copy(layer_starts[1:num_layers])

        Ai_sorted = Vector{Ti}(undef, n_triplets)
        Bi_sorted = Vector{Ti}(undef, n_triplets)
        Ci_sorted = Vector{Ti}(undef, n_triplets)

        for k in 1:n_triplets
            l = layer_of[k]
            pos = next_pos[l]
            Ai_sorted[pos] = Ai[k]
            Bi_sorted[pos] = Bi[k]
            Ci_sorted[pos] = Ci[k]
            next_pos[l] += 1
        end

        Ai = Ai_sorted
        Bi = Bi_sorted
        Ci = Ci_sorted
    end

    # Store in plan, converting to target backend (AIV)
    # For CPU (AIV=Vector{Ti}), this is a no-op
    # For GPU (AIV=MtlVector{Ti}), this copies to GPU
    plan.sym_Ai = AIV(Ai)
    plan.sym_Bi = AIV(Bi)
    plan.sym_Ci = AIV(Ci)
    plan.sym_layer_starts = layer_starts
    plan.sym_colptr = colptr
    plan.sym_rowval = rowval

    return nothing
end

# ============================================================================
# Unified Symbolic Multiply Kernel (KernelAbstractions - works on CPU and GPU)
# ============================================================================

"""
    _symbolic_multiply_layer_kernel!

KernelAbstractions kernel for one layer of symbolic sparse matrix multiplication.
Within each layer, Ci values are unique, so no atomics are needed.
C.nzval[Ci[k]] += AT_nzval[Ai[k]] * B_nzval[Bi[k]] for k in layer_start:(layer_end-1)
"""
@kernel function _symbolic_multiply_layer_kernel!(C_nzval, @Const(Ai), @Const(Bi), @Const(Ci),
                                                   @Const(AT_nzval), @Const(B_nzval),
                                                   layer_start::Int, layer_size::Int)
    idx = @index(Global)
    if idx <= layer_size
        k = layer_start + idx - 1
        @inbounds C_nzval[Ci[k]] += AT_nzval[Ai[k]] * B_nzval[Bi[k]]
    end
end

"""
    _execute_symbolic_multiply!(nzval, plan, AT_nzval, B_nzval)

Unified symbolic multiplication using layer-based parallelism.
The triplets (Ai, Bi, Ci) are partitioned into layers where Ci values
don't repeat within a layer, allowing parallel execution without atomics.
Each layer is processed with a parallel KernelAbstractions kernel.

Works on both CPU and GPU with the same code path.
"""
function _execute_symbolic_multiply!(nzval::AbstractVector{T}, plan::MatrixPlan{T,Ti,AIV},
                                      AT_nzval::AbstractVector{T}, B_nzval::AbstractVector{T}) where {T,Ti,AIV}
    # Zero the output
    fill!(nzval, zero(T))

    Ai = plan.sym_Ai
    Bi = plan.sym_Bi
    Ci = plan.sym_Ci
    layer_starts = plan.sym_layer_starts

    if isempty(Ai)
        return nzval
    end

    backend = KernelAbstractions.get_backend(nzval)

    # Index arrays are already on the target backend (converted in _compute_symbolic_multiply!)
    # No adapt() needed - use directly

    # Process each layer (layers are processed sequentially, elements within layer in parallel)
    num_layers = length(layer_starts) - 1
    kernel = _symbolic_multiply_layer_kernel!(backend)

    for l in 1:num_layers
        layer_start = Int(layer_starts[l])
        layer_end = Int(layer_starts[l + 1])
        layer_size = layer_end - layer_start

        if layer_size > 0
            kernel(nzval, Ai, Bi, Ci, AT_nzval, B_nzval,
                   layer_start, layer_size; ndrange=layer_size)
        end
    end

    KernelAbstractions.synchronize(backend)
    return nzval
end

"""
    Base.*(A::SparseMatrixMPI{T,Ti,AV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}

Multiply two distributed sparse matrices A * B using symbolic multiplication.

Uses a unified KernelAbstractions kernel that works on both CPU and GPU.
MPI communication always uses CPU staging, then data is adapted to the
target backend (A.nzval's backend) for the symbolic multiply computation.

Note: A and B must have the same storage type (both CPU or both GPU).
"""
function Base.:*(A::SparseMatrixMPI{T,Ti,AV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    comm = MPI.COMM_WORLD

    # Get memoized communication plan and execute it
    plan = MatrixPlan(A, B)
    execute_plan!(plan, B)

    # Compute symbolic multiplication if not already cached
    # C^T = B^T * A^T = plan.AT * _get_csc(A)
    if plan.sym_Ai === nothing
        _compute_symbolic_multiply!(plan, _get_csc(A))

        # Compute col_indices and compress_map from symbolic rowval (global column indices)
        if isempty(plan.sym_rowval)
            result_col_indices = Int[]
            compress_map = Int[]
        else
            result_col_indices = unique(sort(Int.(plan.sym_rowval)))

            # Build compress_map: compress_map[global_col] = local_col
            max_col = maximum(result_col_indices)
            compress_map = zeros(Int, max_col)
            for (local_idx, global_idx) in enumerate(result_col_indices)
                compress_map[global_idx] = local_idx
            end
        end

        # Cache col_indices and compress_map
        plan.product_col_indices = result_col_indices
        plan.product_compress_map = compress_map
        plan.product_row_partition = A.row_partition

        # Compress sym_rowval in place: convert global to local indices
        for k in eachindex(plan.sym_rowval)
            plan.sym_rowval[k] = compress_map[plan.sym_rowval[k]]
        end

        # Compute and cache structural hash
        nnz_result = plan.sym_colptr[end] - 1
        temp_nzval = Vector{T}(undef, nnz_result)
        ncols_result = length(plan.sym_colptr) - 1
        temp_csc = SparseMatrixCSC(length(result_col_indices), ncols_result,
                                    plan.sym_colptr, plan.sym_rowval, temp_nzval)
        plan.product_structural_hash = compute_structural_hash(A.row_partition, result_col_indices,
                                                                temp_csc, comm)
    end

    # Allocate result nzval on same backend as A
    nnz_result = plan.sym_colptr[end] - 1
    nzval = similar(A.nzval, nnz_result)

    # Adapt gathered B values (plan.AT.nzval is CPU) to A's backend
    AT_nzval = A.nzval isa Vector ? plan.AT.nzval : copyto!(similar(A.nzval, length(plan.AT.nzval)), plan.AT.nzval)

    # Execute symbolic multiplication (unified kernel)
    _execute_symbolic_multiply!(nzval, plan, AT_nzval, A.nzval)

    # C = A * B has rows from A and columns from B
    # Extract CSR components directly (rowptr=sym_colptr, colval=sym_rowval)
    nrows_local = length(plan.sym_colptr) - 1
    ncols_compressed = length(plan.product_col_indices)

    # Convert structure arrays to target backend
    rowptr_target = _to_target_backend(plan.sym_colptr, AV)
    colval_target = _to_target_backend(plan.sym_rowval, AV)
    return SparseMatrixMPI{T,Ti,AV}(plan.product_structural_hash, A.row_partition, B.col_partition,
        plan.product_col_indices, plan.sym_colptr, plan.sym_rowval, nzval,
        nrows_local, ncols_compressed, nothing, nothing, rowptr_target, colval_target)
end

"""
    AdditionPlan{T,Ti}

A precomputed plan for adding/subtracting two sparse matrices with identical structure.
Stores the result structure (colptr, rowval) and index mappings for SIMD-optimized execution.

The plan preserves structural zeros, ensuring predictable output structure for hash caching.
"""
mutable struct AdditionPlan{T,Ti,AIV<:AbstractVector{Ti}}
    # Result matrix structure (always CPU, used for result construction)
    colptr::Vector{Ti}
    rowval::Vector{Ti}

    # Index mappings for SIMD execution (on target backend - CPU or GPU):
    # For entries only in A: result.nzval[A_only_dst] = A.nzval[A_only_src]
    A_only_src::AIV
    A_only_dst::AIV
    # For entries only in B: result.nzval[B_only_dst] = B.nzval[B_only_src]
    B_only_src::AIV
    B_only_dst::AIV
    # For entries in both: result.nzval[both_dst] = A.nzval[both_A_src] + B.nzval[both_B_src]
    both_A_src::AIV
    both_B_src::AIV
    both_dst::AIV

    # Result metadata (always CPU)
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    col_indices::Vector{Int}

    # Cached structural hash (computed lazily)
    structural_hash::OptionalBlake3Hash
end

"""
    AdditionPlan(A::SparseMatrixMPI{T,Ti,AV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}

Create a plan for computing A + B (or A - B).
Assumes B has already been repartitioned to match A's row_partition.

The plan performs symbolic addition to determine the result structure,
then precomputes index mappings for efficient SIMD execution.
Index arrays are stored on the target backend (CPU or GPU) based on AV.
"""
function AdditionPlan(A::SparseMatrixMPI{T,Ti,AV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    AIV = _index_array_type(AV, Ti)
    # Get the local CSC matrices (stored as transpose)
    A_csc = _get_csc(A)  # (ncols_compressed x nrows_local)
    B_csc = _get_csc(B)  # (ncols_compressed x nrows_local)

    nrows_local = A_csc.n  # number of local rows (columns of CSC)

    # Merge column indices
    union_cols = merge_sorted_unique(A.col_indices, B.col_indices)
    ncols_union = length(union_cols)

    # Build mappings from A/B col_indices to union_cols
    A_col_map = build_subset_mapping(A.col_indices, union_cols)
    B_col_map = build_subset_mapping(B.col_indices, union_cols)

    # Symbolic addition: compute result structure
    # For each local row (column of CSC), merge the nonzero column indices

    # First pass: count nnz per local row to build colptr
    colptr = Vector{Ti}(undef, nrows_local + 1)
    colptr[1] = 1

    for j in 1:nrows_local
        # Get A's nonzeros in this row (mapped to union space)
        A_start, A_end = A_csc.colptr[j], A_csc.colptr[j+1] - 1
        # Get B's nonzeros in this row (mapped to union space)
        B_start, B_end = B_csc.colptr[j], B_csc.colptr[j+1] - 1

        # Count merged entries (both lists are sorted)
        count = Ti(0)
        ai, bi = A_start, B_start
        while ai <= A_end && bi <= B_end
            a_col = A_col_map[A_csc.rowval[ai]]
            b_col = B_col_map[B_csc.rowval[bi]]
            if a_col < b_col
                count += 1
                ai += 1
            elseif b_col < a_col
                count += 1
                bi += 1
            else  # a_col == b_col
                count += 1
                ai += 1
                bi += 1
            end
        end
        count += (A_end - ai + 1) + (B_end - bi + 1)
        colptr[j+1] = colptr[j] + count
    end

    nnz_result = colptr[nrows_local + 1] - 1
    rowval = Vector{Ti}(undef, nnz_result)

    # Prepare index mapping vectors
    A_only_src = Ti[]
    A_only_dst = Ti[]
    B_only_src = Ti[]
    B_only_dst = Ti[]
    both_A_src = Ti[]
    both_B_src = Ti[]
    both_dst = Ti[]

    # Second pass: fill rowval and build index mappings
    for j in 1:nrows_local
        A_start, A_end = A_csc.colptr[j], A_csc.colptr[j+1] - 1
        B_start, B_end = B_csc.colptr[j], B_csc.colptr[j+1] - 1

        result_idx = colptr[j]
        ai, bi = A_start, B_start

        while ai <= A_end && bi <= B_end
            a_col = A_col_map[A_csc.rowval[ai]]
            b_col = B_col_map[B_csc.rowval[bi]]
            if a_col < b_col
                rowval[result_idx] = a_col
                push!(A_only_src, ai)
                push!(A_only_dst, result_idx)
                result_idx += 1
                ai += 1
            elseif b_col < a_col
                rowval[result_idx] = b_col
                push!(B_only_src, bi)
                push!(B_only_dst, result_idx)
                result_idx += 1
                bi += 1
            else  # a_col == b_col
                rowval[result_idx] = a_col
                push!(both_A_src, ai)
                push!(both_B_src, bi)
                push!(both_dst, result_idx)
                result_idx += 1
                ai += 1
                bi += 1
            end
        end

        # Remaining entries from A
        while ai <= A_end
            rowval[result_idx] = A_col_map[A_csc.rowval[ai]]
            push!(A_only_src, ai)
            push!(A_only_dst, result_idx)
            result_idx += 1
            ai += 1
        end

        # Remaining entries from B
        while bi <= B_end
            rowval[result_idx] = B_col_map[B_csc.rowval[bi]]
            push!(B_only_src, bi)
            push!(B_only_dst, result_idx)
            result_idx += 1
            bi += 1
        end
    end

    # Compute structural hash now (at plan time) to avoid computing during execution
    # Create a temporary CSC to compute hash (values don't matter for structural hash)
    temp_nzval = Vector{T}(undef, nnz_result)
    temp_csc = SparseMatrixCSC(length(union_cols), nrows_local, colptr, rowval, temp_nzval)
    structural_hash = compute_structural_hash(A.row_partition, union_cols, temp_csc, MPI.COMM_WORLD)

    # Convert index arrays to target backend (AIV)
    # For CPU (AIV=Vector{Ti}), this is a no-op
    # For GPU (AIV=MtlVector{Ti}), this copies to GPU
    return AdditionPlan{T,Ti,AIV}(
        colptr, rowval,
        AIV(Ti.(A_only_src)), AIV(Ti.(A_only_dst)),
        AIV(Ti.(B_only_src)), AIV(Ti.(B_only_dst)),
        AIV(Ti.(both_A_src)), AIV(Ti.(both_B_src)), AIV(Ti.(both_dst)),
        A.row_partition, A.col_partition, union_cols,
        structural_hash
    )
end

# ============================================================================
# Unified Addition/Subtraction Kernels (KernelAbstractions - works on CPU and GPU)
# ============================================================================

"""
    _copy_a_only_kernel!

KernelAbstractions kernel to copy A-only entries: nzval[dst[i]] = A[src[i]]
"""
@kernel function _copy_a_only_kernel!(nzval, @Const(A_nzval), @Const(src), @Const(dst))
    i = @index(Global)
    @inbounds nzval[dst[i]] = A_nzval[src[i]]
end

"""
    _copy_b_only_kernel!

KernelAbstractions kernel to copy B-only entries: nzval[dst[i]] = B[src[i]]
"""
@kernel function _copy_b_only_kernel!(nzval, @Const(B_nzval), @Const(src), @Const(dst))
    i = @index(Global)
    @inbounds nzval[dst[i]] = B_nzval[src[i]]
end

"""
    _negate_b_only_kernel!

KernelAbstractions kernel to negate B-only entries: nzval[dst[i]] = -B[src[i]]
"""
@kernel function _negate_b_only_kernel!(nzval, @Const(B_nzval), @Const(src), @Const(dst))
    i = @index(Global)
    @inbounds nzval[dst[i]] = -B_nzval[src[i]]
end

"""
    _add_both_kernel!

KernelAbstractions kernel to add entries in both: nzval[dst[i]] = A[A_src[i]] + B[B_src[i]]
"""
@kernel function _add_both_kernel!(nzval, @Const(A_nzval), @Const(B_nzval),
                                    @Const(A_src), @Const(B_src), @Const(dst))
    i = @index(Global)
    @inbounds nzval[dst[i]] = A_nzval[A_src[i]] + B_nzval[B_src[i]]
end

"""
    _sub_both_kernel!

KernelAbstractions kernel to subtract entries in both: nzval[dst[i]] = A[A_src[i]] - B[B_src[i]]
"""
@kernel function _sub_both_kernel!(nzval, @Const(A_nzval), @Const(B_nzval),
                                    @Const(A_src), @Const(B_src), @Const(dst))
    i = @index(Global)
    @inbounds nzval[dst[i]] = A_nzval[A_src[i]] - B_nzval[B_src[i]]
end

"""
    execute_addition!(nzval, plan, A_nzval, B_nzval)

Execute an addition plan: result = A + B.
Uses unified KernelAbstractions kernels that work on both CPU and GPU.
"""
function execute_addition!(nzval::AbstractVector{T}, plan::AdditionPlan{T,Ti,AIV},
                           A_nzval::AbstractVector{T}, B_nzval::AbstractVector{T}) where {T,Ti,AIV}
    backend = KernelAbstractions.get_backend(nzval)

    # Index arrays are already on the target backend (converted in AdditionPlan constructor)
    # No adapt() needed - use directly

    # Copy A-only entries
    if !isempty(plan.A_only_src)
        kernel = _copy_a_only_kernel!(backend)
        kernel(nzval, A_nzval, plan.A_only_src, plan.A_only_dst; ndrange=length(plan.A_only_src))
    end

    # Copy B-only entries
    if !isempty(plan.B_only_src)
        kernel = _copy_b_only_kernel!(backend)
        kernel(nzval, B_nzval, plan.B_only_src, plan.B_only_dst; ndrange=length(plan.B_only_src))
    end

    # Add entries in both
    if !isempty(plan.both_dst)
        kernel = _add_both_kernel!(backend)
        kernel(nzval, A_nzval, B_nzval, plan.both_A_src, plan.both_B_src, plan.both_dst; ndrange=length(plan.both_dst))
    end

    KernelAbstractions.synchronize(backend)
    return nzval
end

"""
    execute_subtraction!(nzval, plan, A_nzval, B_nzval)

Execute a subtraction plan: result = A - B.
Uses unified KernelAbstractions kernels that work on both CPU and GPU.
"""
function execute_subtraction!(nzval::AbstractVector{T}, plan::AdditionPlan{T,Ti,AIV},
                              A_nzval::AbstractVector{T}, B_nzval::AbstractVector{T}) where {T,Ti,AIV}
    backend = KernelAbstractions.get_backend(nzval)

    # Index arrays are already on the target backend (converted in AdditionPlan constructor)
    # No adapt() needed - use directly

    # Copy A-only entries
    if !isempty(plan.A_only_src)
        kernel = _copy_a_only_kernel!(backend)
        kernel(nzval, A_nzval, plan.A_only_src, plan.A_only_dst; ndrange=length(plan.A_only_src))
    end

    # Negate B-only entries
    if !isempty(plan.B_only_src)
        kernel = _negate_b_only_kernel!(backend)
        kernel(nzval, B_nzval, plan.B_only_src, plan.B_only_dst; ndrange=length(plan.B_only_src))
    end

    # Subtract entries in both
    if !isempty(plan.both_dst)
        kernel = _sub_both_kernel!(backend)
        kernel(nzval, A_nzval, B_nzval, plan.both_A_src, plan.both_B_src, plan.both_dst; ndrange=length(plan.both_dst))
    end

    KernelAbstractions.synchronize(backend)
    return nzval
end

"""
    _get_addition_plan(A::SparseMatrixMPI{T,Ti,AV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}

Get or create a cached addition plan for A + B.
B must already be repartitioned to match A's row_partition.
The plan's index arrays are stored on the target backend (CPU or GPU) based on AV.
"""
function _get_addition_plan(A::SparseMatrixMPI{T,Ti,AV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    AIV = _index_array_type(AV, Ti)
    key = (A.structural_hash, B.structural_hash, T, Ti, AV)
    if haskey(_addition_plan_cache, key)
        return _addition_plan_cache[key]::AdditionPlan{T,Ti,AIV}
    end
    plan = AdditionPlan(A, B)
    _addition_plan_cache[key] = plan
    return plan
end

"""
    Base.+(A::SparseMatrixMPI{T,Ti,AV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}

Add two distributed sparse matrices using plan-based execution.
Preserves structural zeros for predictable output structure.
The result has A's row partition and storage type.

Uses unified KernelAbstractions kernels that work on both CPU and GPU.
"""
function Base.:+(A::SparseMatrixMPI{T,Ti,AV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    # Repartition B to match A's row partition (uses CPU staging for MPI, returns CPU)
    B_repart = repartition(B, A.row_partition)

    # Ensure both have structural hashes (uses CPU nzval for hashing)
    _ensure_hash(A)
    _ensure_hash(B_repart)

    # Get or create plan (computed from CPU structures)
    plan = _get_addition_plan(A, B_repart)

    # Allocate result nzval on same backend as A
    nnz_result = plan.colptr[end] - 1
    nzval = similar(A.nzval, nnz_result)

    # Execute addition directly into result buffer (unified kernel)
    # B_repart.nzval has same backend as A.nzval (repartition preserves backend)
    execute_addition!(nzval, plan, A.nzval, B_repart.nzval)

    # Extract CSR components directly (rowptr=colptr, colval=rowval from plan)
    nrows_local = length(plan.colptr) - 1
    ncols_compressed = length(plan.col_indices)

    # Convert structure arrays to target backend
    rowptr_target = _to_target_backend(plan.colptr, AV)
    colval_target = _to_target_backend(plan.rowval, AV)
    return SparseMatrixMPI{T,Ti,AV}(plan.structural_hash, plan.row_partition, plan.col_partition,
        plan.col_indices, plan.colptr, plan.rowval, nzval,
        nrows_local, ncols_compressed, nothing, nothing, rowptr_target, colval_target)
end

"""
    Base.-(A::SparseMatrixMPI{T,Ti,AV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}

Subtract two distributed sparse matrices using plan-based execution.
Preserves structural zeros for predictable output structure.
The result has A's row partition and storage type.

Uses unified KernelAbstractions kernels that work on both CPU and GPU.
"""
function Base.:-(A::SparseMatrixMPI{T,Ti,AV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    # Repartition B to match A's row partition (uses CPU staging for MPI, returns CPU)
    B_repart = repartition(B, A.row_partition)

    # Ensure both have structural hashes (uses CPU nzval for hashing)
    _ensure_hash(A)
    _ensure_hash(B_repart)

    # Get or create plan (computed from CPU structures)
    plan = _get_addition_plan(A, B_repart)

    # Allocate result nzval on same backend as A
    nnz_result = plan.colptr[end] - 1
    nzval = similar(A.nzval, nnz_result)

    # Execute subtraction directly into result buffer (unified kernel)
    # B_repart.nzval has same backend as A.nzval (repartition preserves backend)
    execute_subtraction!(nzval, plan, A.nzval, B_repart.nzval)

    # Extract CSR components directly (rowptr=colptr, colval=rowval from plan)
    nrows_local = length(plan.colptr) - 1
    ncols_compressed = length(plan.col_indices)

    # Convert structure arrays to target backend
    rowptr_target = _to_target_backend(plan.colptr, AV)
    colval_target = _to_target_backend(plan.rowval, AV)
    return SparseMatrixMPI{T,Ti,AV}(plan.structural_hash, plan.row_partition, plan.col_partition,
        plan.col_indices, plan.colptr, plan.rowval, nzval,
        nrows_local, ncols_compressed, nothing, nothing, rowptr_target, colval_target)
end

"""
    TransposePlan{T,Ti}

A communication plan for computing the transpose of an SparseMatrixMPI.

The transpose of A (with row_partition R and col_partition C) will have:
- row_partition = C (columns of A become rows of A^T)
- col_partition = R (rows of A become columns of A^T)

# Fields
- `rank_ids::Vector{Int}`: Ranks we send data to (0-indexed)
- `send_indices::Vector{Vector{Int}}`: For each rank, indices into A.nzval to send
- `send_bufs::Vector{Vector{T}}`: Pre-allocated send buffers
- `send_reqs::Vector{MPI.Request}`: Pre-allocated send request handles
- `recv_rank_ids::Vector{Int}`: Ranks we receive data from (0-indexed)
- `recv_bufs::Vector{Vector{T}}`: Pre-allocated receive buffers
- `recv_reqs::Vector{MPI.Request}`: Pre-allocated receive request handles
- `recv_perm::Vector{Vector{Int}}`: For each recv rank, permutation into AT.nzval
- `local_src_indices::Vector{Int}`: Source indices for local copy
- `local_dst_indices::Vector{Int}`: Destination indices for local copy
- `AT::SparseMatrixCSC{T,Ti}`: Transposed matrix structure (values zeroed)
- `row_partition::Vector{Int}`: Row partition for the transposed matrix
- `col_partition::Vector{Int}`: Col partition for the transposed matrix
- `col_indices::Vector{Int}`: Column indices for the transposed matrix
"""
mutable struct TransposePlan{T,Ti}
    rank_ids::Vector{Int}
    send_indices::Vector{Vector{Int}}
    send_bufs::Vector{Vector{T}}
    send_reqs::Vector{MPI.Request}
    recv_rank_ids::Vector{Int}
    recv_bufs::Vector{Vector{T}}
    recv_reqs::Vector{MPI.Request}
    recv_perm::Vector{Vector{Int}}
    local_src_indices::Vector{Int}
    local_dst_indices::Vector{Int}
    AT::SparseMatrixCSC{T,Ti}
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    col_indices::Vector{Int}
    # Precomputed compress_map: compress_map[global_col] = local_col
    compress_map::Vector{Int}
    # Cached structural hash for transpose result (computed lazily on first execution)
    structural_hash::OptionalBlake3Hash
end

"""
    TransposePlan(A::SparseMatrixMPI{T,Ti}) where {T,Ti}

Create a communication plan for computing A^T.

The algorithm:
1. For each nonzero A[i,j] (stored as _get_csc(A)[j, local_i]), determine which rank
   owns row j in A^T (using A.col_partition). Package (i,j) pairs by destination rank.
2. Exchange structure via point-to-point communication.
3. Build the transposed sparse structure and communication buffers.
"""
function TransposePlan(A::SparseMatrixMPI{T,Ti}) where {T,Ti}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    my_row_start = A.row_partition[rank+1]
    nrows_A = A.row_partition[end] - 1

    # The transpose has swapped partitions
    result_row_partition = A.col_partition
    result_col_partition = A.row_partition

    # Step 1: For each nonzero in _get_csc(A), determine destination rank for transpose
    # _get_csc(A)[j, local_col] corresponds to A[global_row, j] where global_row = my_row_start + local_col - 1
    # In A^T, this is A^T[j, global_row], so it goes to the rank owning row j per col_partition

    # Group nonzeros by destination rank: (global_row, j, src_nzval_idx)
    send_to = [Tuple{Int,Int,Int}[] for _ in 1:nranks]

    for local_col in 1:A.nrows_local
        global_row = my_row_start + local_col - 1
        for idx in A.rowptr[local_col]:(A.rowptr[local_col+1]-1)
            local_j = A.colval[idx]  # LOCAL column index in A (compressed)
            j = A.col_indices[local_j]  # Convert to GLOBAL column index
            dest_rank = searchsortedlast(A.col_partition, j) - 1
            push!(send_to[dest_rank+1], (global_row, j, idx))
        end
    end

    # Step 2: Exchange counts via Alltoall
    send_counts = [length(send_to[r+1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Step 3: Send structure (row, col pairs) to each destination rank
    # Build rank_ids and send_indices in order of rank
    rank_ids = Int[]
    send_indices_map = Dict{Int,Vector{Int}}()
    struct_send_bufs = Dict{Int,Vector{Int}}()
    struct_send_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if send_counts[r+1] > 0 && r != rank
            push!(rank_ids, r)
            indices = [t[3] for t in send_to[r+1]]
            send_indices_map[r] = indices
            # Pack (row, col) pairs: in A^T, row=j, col=global_row
            msg = Vector{Int}(undef, 2 * length(send_to[r+1]))
            for (k, (global_row, j, _)) in enumerate(send_to[r+1])
                msg[2k-1] = j          # row in A^T
                msg[2k] = global_row   # col in A^T
            end
            struct_send_bufs[r] = msg
            req = MPI.Isend(msg, comm; dest=r, tag=10)
            push!(struct_send_reqs, req)
        end
    end

    # Step 4: Receive structure from other ranks
    recv_rank_ids = Int[]
    struct_recv_bufs = Dict{Int,Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r+1] > 0 && r != rank
            push!(recv_rank_ids, r)
            buf = Vector{Int}(undef, 2 * recv_counts[r+1])
            req = MPI.Irecv!(buf, comm; source=r, tag=10)
            push!(struct_recv_reqs, req)
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Step 5: Build the transposed sparse matrix structure
    my_AT_row_start = result_row_partition[rank+1]
    my_AT_row_end = result_row_partition[rank+2] - 1
    local_ncols = my_AT_row_end - my_AT_row_start + 1
    result_AT_nrows = nrows_A

    # Collect all (j, i, source_rank, source_idx) entries
    # j = row in A^T (our local row), i = col in A^T
    entries = Tuple{Int,Int,Int,Int}[]

    # From remote ranks - track index within each rank's buffer
    for r in recv_rank_ids
        buf = struct_recv_bufs[r]
        n = div(length(buf), 2)
        for k in 1:n
            j = buf[2k-1]   # row in A^T
            i = buf[2k]     # col in A^T
            push!(entries, (j, i, r, k))
        end
    end

    # From local - entries staying on this rank
    local_entries_src = Int[]
    for (global_row, j, idx) in send_to[rank+1]
        push!(local_entries_src, idx)
        push!(entries, (j, global_row, rank, length(local_entries_src)))
    end

    # Sort entries by (local_col_in_result_AT, row_in_result_AT) for CSC format
    sort!(entries, by=e -> (e[1] - my_AT_row_start + 1, e[2]))

    # Build CSC structure for result.AT
    # result.AT has dimensions (nrows_A, local_ncols)
    colptr = zeros(Int, local_ncols + 1)
    rowval = Vector{Int}(undef, length(entries))
    nzval = zeros(T, length(entries))

    # Count entries per column (store in colptr[col+1] for cumsum)
    for (j, _, _, _) in entries
        local_col = j - my_AT_row_start + 1
        colptr[local_col+1] += 1
    end

    # Cumsum to convert counts to pointers, starting from 1
    colptr[1] = 1
    for c in 2:(local_ncols+1)
        colptr[c] += colptr[c-1]
    end

    # Fill rowval using column cursors
    col_cursors = copy(colptr[1:end-1])
    entry_to_nzval_idx = Vector{Int}(undef, length(entries))

    for (ent_idx, (j, i, _, _)) in enumerate(entries)
        local_col = j - my_AT_row_start + 1
        pos = col_cursors[local_col]
        rowval[pos] = i
        entry_to_nzval_idx[ent_idx] = pos
        col_cursors[local_col] += 1
    end

    result_AT = SparseMatrixCSC(result_AT_nrows, local_ncols, colptr, rowval, nzval)

    # Step 6: Build communication plan for values
    # Pre-allocate recv_perm with correct sizes - use indexed assignment not push!
    # because entries are sorted but recv_bufs come in original send order
    recv_perm = [Vector{Int}(undef, recv_counts[r+1]) for r in recv_rank_ids]
    local_src_indices = Int[]
    local_dst_indices = Int[]

    # Map rank -> index in recv_rank_ids using Vector (nranks is small)
    recv_rank_to_idx = zeros(Int, nranks)
    for (i, r) in enumerate(recv_rank_ids)
        recv_rank_to_idx[r+1] = i
    end

    for (ent_idx, (_, _, src_rank, src_idx)) in enumerate(entries)
        dst_idx = entry_to_nzval_idx[ent_idx]
        if src_rank == rank
            push!(local_src_indices, local_entries_src[src_idx])
            push!(local_dst_indices, dst_idx)
        else
            # Use indexed assignment: src_idx is the position in recv_buf from src_rank
            recv_perm[recv_rank_to_idx[src_rank+1]][src_idx] = dst_idx
        end
    end

    # Build send_indices in same order as rank_ids
    send_indices_final = [send_indices_map[r] for r in rank_ids]

    # Allocate buffers
    send_bufs = [Vector{T}(undef, length(inds)) for inds in send_indices_final]
    recv_bufs = [Vector{T}(undef, recv_counts[r+1]) for r in recv_rank_ids]
    send_reqs = Vector{MPI.Request}(undef, length(rank_ids))
    recv_reqs = Vector{MPI.Request}(undef, length(recv_rank_ids))

    # Compute col_indices for result
    result_col_indices = isempty(rowval) ? Int[] : unique(sort(rowval))

    # Precompute compress_map: compress_map[global_col] = local_col
    if isempty(result_col_indices)
        compress_map = Int[]
    else
        max_col = maximum(result_col_indices)
        compress_map = zeros(Int, max_col)
        for (local_idx, global_idx) in enumerate(result_col_indices)
            compress_map[global_idx] = local_idx
        end
    end

    return TransposePlan{T,Ti}(
        rank_ids, send_indices_final, send_bufs, send_reqs,
        recv_rank_ids, recv_bufs, recv_reqs, recv_perm,
        local_src_indices, local_dst_indices,
        result_AT, result_row_partition, result_col_partition, result_col_indices,
        compress_map,
        nothing  # structural_hash (computed lazily on first execution)
    )
end

"""
    execute_plan!(plan::TransposePlan{T,Ti}, A::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}

Execute a transpose plan to compute A^T.
Returns an SparseMatrixMPI representing the transpose.
Handles both CPU and GPU matrices by staging through CPU for MPI.

Note: The returned matrix has its own copy of the sparse data, so the plan
can be safely reused for subsequent transposes.
"""
function execute_plan!(plan::TransposePlan{T,Ti}, A::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    comm = MPI.COMM_WORLD

    # Ensure A.nzval is on CPU for MPI communication
    A_nzval_cpu = _ensure_cpu(A.nzval)

    # Step 1: Copy local values (allocation-free loop)
    local_src = plan.local_src_indices
    local_dst = plan.local_dst_indices
    @inbounds for i in eachindex(local_src, local_dst)
        plan.AT.nzval[local_dst[i]] = A_nzval_cpu[local_src[i]]
    end

    # Step 2: Fill send buffers and send (allocation-free loops)
    @inbounds for i in eachindex(plan.rank_ids)
        r = plan.rank_ids[i]
        send_idx = plan.send_indices[i]
        buf = plan.send_bufs[i]
        for k in eachindex(send_idx)
            buf[k] = A_nzval_cpu[send_idx[k]]
        end
        plan.send_reqs[i] = MPI.Isend(buf, comm; dest=r, tag=11)
    end

    # Step 3: Receive values
    @inbounds for i in eachindex(plan.recv_rank_ids)
        plan.recv_reqs[i] = MPI.Irecv!(plan.recv_bufs[i], comm; source=plan.recv_rank_ids[i], tag=11)
    end

    MPI.Waitall(plan.recv_reqs)

    # Step 4: Scatter received values into result.AT.nzval (allocation-free loops)
    @inbounds for i in eachindex(plan.recv_rank_ids)
        perm = plan.recv_perm[i]
        buf = plan.recv_bufs[i]
        for k in eachindex(perm)
            plan.AT.nzval[perm[k]] = buf[k]
        end
    end

    MPI.Waitall(plan.send_reqs)

    # Create a copy of AT so the plan can be reused
    # plan.AT has global column indices in rowval, compress to local indices
    result_AT = SparseMatrixCSC(
        plan.AT.m, plan.AT.n,
        copy(plan.AT.colptr),
        copy(plan.AT.rowval),
        copy(plan.AT.nzval)
    )

    # Compress result_AT: convert global column indices to local indices (using precomputed map)
    compressed_result_AT = compress_AT_cached(result_AT, plan.compress_map, length(plan.col_indices))

    # Use cached hash if available, otherwise compute and cache
    if plan.structural_hash === nothing
        plan.structural_hash = compute_structural_hash(plan.row_partition, plan.col_indices, compressed_result_AT, comm)
    end

    # Extract CSR components from compressed_result_AT
    nrows_local = compressed_result_AT.n
    ncols_compressed = length(plan.col_indices)

    # Copy result to GPU if input was GPU
    result_nzval = A.nzval isa Vector ? compressed_result_AT.nzval : copyto!(similar(A.nzval, length(compressed_result_AT.nzval)), compressed_result_AT.nzval)

    # Convert structure arrays to target backend
    rowptr_target = _to_target_backend(compressed_result_AT.colptr, AV)
    colval_target = _to_target_backend(compressed_result_AT.rowval, AV)
    return SparseMatrixMPI{T,Ti,AV}(plan.structural_hash, plan.row_partition, plan.col_partition,
        plan.col_indices, compressed_result_AT.colptr, compressed_result_AT.rowval, result_nzval,
        nrows_local, ncols_compressed, nothing, nothing, rowptr_target, colval_target)
end

"""
    SparseMatrixMPI{T}(At::Transpose{T, SparseMatrixMPI{T,Ti,AV}}) where {T,Ti,AV}

Materialize a lazy transpose of a SparseMatrixMPI, using cached result if available.
If the transpose has been computed before, returns the cached result.
Otherwise, computes the transpose via TransposePlan and caches it bidirectionally
(A.cached_transpose = Y and Y.cached_transpose = A).

# Example
```julia
A = SparseMatrixMPI{Float64}(sparse(...))
At = transpose(A)           # Lazy transpose wrapper
At_mat = SparseMatrixMPI(At) # Materialize the transpose
```
"""
function SparseMatrixMPI{T}(At::Transpose{T, SparseMatrixMPI{T,Ti,AV}}) where {T,Ti,AV}
    A = At.parent
    # Check if already cached
    if A.cached_transpose !== nothing
        return A.cached_transpose
    end

    # Compute the transpose
    plan = TransposePlan(A)
    Y = execute_plan!(plan, A)

    # Cache bidirectionally
    A.cached_transpose = Y
    Y.cached_transpose = A

    return Y
end

# Convenience: allow SparseMatrixMPI(transpose(A)) without specifying type parameter
SparseMatrixMPI(At::Transpose{T, SparseMatrixMPI{T,Ti,AV}}) where {T,Ti,AV} = SparseMatrixMPI{T}(At)

# VectorPlan constructor for sparse A * x (adds method to VectorPlan from vectors.jl)

"""
    VectorPlan(A::SparseMatrixMPI{T,Ti}, x::VectorMPI{T,AV}) where {T,Ti,AV}

Create a communication plan to gather x[A.col_indices] for matrix-vector multiplication.
The gathered buffer will have the same storage type as the input vector x.
"""
function VectorPlan(A::SparseMatrixMPI{T,Ti}, x::VectorMPI{T,AV}) where {T,Ti,AV}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    col_indices = A.col_indices
    n_gathered = length(col_indices)

    my_x_start = x.partition[rank+1]

    # Step 1: Group col_indices by owner rank in x's partition
    # Track (global_idx, dst_idx_in_gathered) per owner
    needed_from = [Tuple{Int,Int}[] for _ in 1:nranks]
    for (dst_idx, global_idx) in enumerate(col_indices)
        owner = searchsortedlast(x.partition, global_idx) - 1
        # Clamp to handle edge case where index equals last partition boundary
        if owner >= nranks
            owner = nranks - 1
        end
        push!(needed_from[owner+1], (global_idx, dst_idx))
    end

    # Step 2: Exchange counts via Alltoall
    send_counts = [length(needed_from[r+1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Step 3: Send requested indices to each owner rank
    struct_send_bufs = Dict{Int,Vector{Int}}()
    struct_send_reqs = MPI.Request[]
    recv_rank_ids = Int[]
    recv_perm_map = Dict{Int,Vector{Int}}()  # rank => dst indices in gathered

    for r in 0:(nranks-1)
        if send_counts[r+1] > 0 && r != rank
            push!(recv_rank_ids, r)
            indices = [t[1] for t in needed_from[r+1]]
            dst_indices = [t[2] for t in needed_from[r+1]]
            recv_perm_map[r] = dst_indices
            struct_send_bufs[r] = indices
            req = MPI.Isend(indices, comm; dest=r, tag=20)
            push!(struct_send_reqs, req)
        end
    end

    # Step 4: Receive requests from other ranks
    send_rank_ids = Int[]
    struct_recv_bufs = Dict{Int,Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r+1] > 0 && r != rank
            push!(send_rank_ids, r)
            buf = Vector{Int}(undef, recv_counts[r+1])
            req = MPI.Irecv!(buf, comm; source=r, tag=20)
            push!(struct_recv_reqs, req)
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Step 5: Convert received global indices to local indices for sending
    send_indices_map = Dict{Int,Vector{Int}}()
    for r in send_rank_ids
        global_indices = struct_recv_bufs[r]
        local_indices = [idx - my_x_start + 1 for idx in global_indices]
        send_indices_map[r] = local_indices
    end

    # Step 6: Handle local elements (elements we own)
    local_src_indices = Int[]
    local_dst_indices = Int[]
    for (global_idx, dst_idx) in needed_from[rank+1]
        local_idx = global_idx - my_x_start + 1
        push!(local_src_indices, local_idx)
        push!(local_dst_indices, dst_idx)
    end

    # Step 7: Build final arrays and buffers
    sort!(send_rank_ids)
    sort!(recv_rank_ids)

    send_indices_final = [send_indices_map[r] for r in send_rank_ids]
    recv_perm_final = [recv_perm_map[r] for r in recv_rank_ids]

    # MPI buffers are always on CPU
    send_bufs = [Vector{T}(undef, length(inds)) for inds in send_indices_final]
    recv_bufs = [Vector{T}(undef, send_counts[r+1]) for r in recv_rank_ids]
    send_reqs = Vector{MPI.Request}(undef, length(send_rank_ids))
    recv_reqs = Vector{MPI.Request}(undef, length(recv_rank_ids))

    # CPU staging buffer (always needed for MPI)
    gathered_cpu = Vector{T}(undef, n_gathered)

    # Gathered buffer matches source vector's storage type
    gathered = similar(x.v, n_gathered)

    return VectorPlan{T,AV}(
        send_rank_ids, send_indices_final, send_bufs, send_reqs,
        recv_rank_ids, recv_bufs, recv_reqs, recv_perm_final,
        local_src_indices, local_dst_indices, gathered, gathered_cpu,
        nothing, nothing  # result_partition_hash, result_partition (computed lazily)
    )
end

"""
    get_vector_plan(A::SparseMatrixMPI{T,Ti}, x::VectorMPI{T,AV}) where {T,Ti,AV}

Get a memoized VectorPlan for A * x.
The plan is cached based on the structural hashes of A and x, plus the array type.
"""
function get_vector_plan(A::SparseMatrixMPI{T,Ti}, x::VectorMPI{T,AV}) where {T,Ti,AV}
    key = (_ensure_hash(A), x.structural_hash, T, AV)
    if haskey(_vector_plan_cache, key)
        return _vector_plan_cache[key]::VectorPlan{T,AV}
    end
    plan = VectorPlan(A, x)
    _vector_plan_cache[key] = plan
    return plan
end

# Matrix-vector multiplication

"""
    LinearAlgebra.mul!(y::VectorMPI{T,AV}, A::SparseMatrixMPI{T,Ti}, x::VectorMPI{T,AV}) where {T,Ti,AV}

In-place sparse matrix-vector multiplication: y = A * x.

The algorithm:
1. Gather x[A.col_indices] from all ranks using VectorPlan
2. Compute y.v = transpose(_get_csc(A)) * gathered

_get_csc(A) is already compressed with local indices 1:length(A.col_indices),
so gathered has length matching A.ncols_compressed and can be used directly.

For GPU vectors, the multiply is performed on CPU and the result is copied to y.v.
"""
function LinearAlgebra.mul!(y::VectorMPI{T,AV}, A::SparseMatrixMPI{T,Ti}, x::VectorMPI{T,AV}) where {T,Ti,AV}
    # Get memoized plan and execute it
    plan = get_vector_plan(A, x)
    execute_plan!(plan, x)

    # _get_csc(A) is already compressed with local indices 1:length(A.col_indices)
    # gathered has length length(A.col_indices), matching A.ncols_compressed
    # y = A * x => y^T = x^T * A^T => y.v = transpose(_get_csc(A)) * gathered
    # Use transpose() not ' to avoid conjugation for complex types

    # Sparse matrix multiply uses CPU buffer (sparse matrix is on CPU)
    gathered_cpu = _gathered_cpu_buffer(plan)
    y_local_cpu = Vector{T}(undef, length(y.v))
    LinearAlgebra.mul!(y_local_cpu, transpose(_get_csc(A)), gathered_cpu)

    # Copy result to output (no-op for CPU, GPU copy for GPU)
    copyto!(y.v, y_local_cpu)
    return y
end

# Helper to get CPU buffer for sparse multiply
_gathered_cpu_buffer(plan::VectorPlan{T,Vector{T}}) where T = plan.gathered
_gathered_cpu_buffer(plan::VectorPlan{T,AV}) where {T,AV} = plan.gathered_cpu

# Note: _create_output_like is defined in vectors.jl (shared by sparse and dense)

# ============================================================================
# Unified SpMV Kernel (KernelAbstractions - works on CPU and GPU)
# ============================================================================

"""
    _spmv_kernel!

KernelAbstractions kernel for sparse matrix-vector multiplication: y = A * x
where A is in CSR format. Works on both CPU and GPU backends.
"""
@kernel function _spmv_kernel!(y, @Const(rowptr), @Const(colval), @Const(nzval), @Const(x))
    row = @index(Global)
    if row <= length(y)
        @inbounds begin
            acc = zero(eltype(y))
            for j in rowptr[row]:(rowptr[row+1]-1)
                acc += nzval[j] * x[colval[j]]
            end
            y[row] = acc
        end
    end
end

"""
    _spmv!(y, rowptr, colval, nzval, x)

Unified sparse matrix-vector multiplication using KernelAbstractions.
Works on both CPU and GPU - the backend is determined from the output array y.
Structure arrays (rowptr, colval) should already be on the target backend.
"""
function _spmv!(y::AbstractVector{T}, rowptr::AbstractVector, colval::AbstractVector,
                nzval::AbstractVector{T}, x::AbstractVector{T}) where T
    backend = KernelAbstractions.get_backend(y)
    # Structure arrays should already be on target backend (no adapt needed)
    kernel = _spmv_kernel!(backend)
    kernel(y, rowptr, colval, nzval, x; ndrange=length(y))
    KernelAbstractions.synchronize(backend)
    return y
end

"""
    Base.:*(A::SparseMatrixMPI{T,Ti,AV}, x::VectorMPI{T,AVX}) where {T,Ti,AV,AVX}

Sparse matrix-vector multiplication returning a new VectorMPI.
The result has the same row partition as A.

Uses a unified KernelAbstractions kernel that works on both CPU and GPU.
MPI communication always uses CPU staging, then data is adapted to the
target backend (A.nzval's backend) for the SpMV computation.
"""
function Base.:*(A::SparseMatrixMPI{T,Ti,AV}, x::VectorMPI{T,AVX}) where {T,Ti,AV,AVX}
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_rows = A.row_partition[rank+2] - A.row_partition[rank+1]

    # Get the plan and use cached partition hash if available
    plan = get_vector_plan(A, x)
    if plan.result_partition_hash === nothing
        plan.result_partition_hash = compute_partition_hash(A.row_partition)
        plan.result_partition = copy(A.row_partition)
    end

    # Execute the plan to gather vector elements (MPI on CPU)
    execute_plan!(plan, x)

    # Get gathered data on CPU (MPI communication is always CPU)
    gathered_cpu = _gathered_cpu_buffer(plan)

    # Adapt gathered data to A's backend
    gathered = A.nzval isa Vector ? gathered_cpu : copyto!(similar(A.nzval, length(gathered_cpu)), gathered_cpu)
    y_local = similar(A.nzval, local_rows)

    # Unified SpMV kernel using pre-computed target arrays (works on CPU or GPU)
    # A.rowptr_target and A.colval_target are on the same backend as A.nzval
    _spmv!(y_local, A.rowptr_target, A.colval_target, A.nzval, gathered)

    # Determine result vector type: use A's storage type
    return VectorMPI{T,AV}(
        plan.result_partition_hash,
        plan.result_partition,
        y_local
    )
end

"""
    *(vt::Transpose{<:Any, VectorMPI{T}}, A::SparseMatrixMPI{T,Ti}) where {T,Ti}

Compute transpose(v) * A as transpose(transpose(A) * v).
Returns a transposed VectorMPI.
"""
function Base.:*(vt::Transpose{<:Any,VectorMPI{T}}, A::SparseMatrixMPI{T,Ti}) where {T,Ti}
    v = vt.parent
    # transpose(v) * A = transpose(transpose(A) * v)
    A_transposed = SparseMatrixMPI(transpose(A))
    result = A_transposed * v
    return transpose(result)
end

# AbstractMatrix interface for SparseMatrixMPI

"""
    Base.size(A::SparseMatrixMPI)

Return the size of the distributed matrix as (nrows, ncols).
"""
function Base.size(A::SparseMatrixMPI)
    nrows = A.row_partition[end] - 1
    ncols = A.col_partition[end] - 1
    return (nrows, ncols)
end

Base.size(A::SparseMatrixMPI, d::Integer) = size(A)[d]

Base.eltype(::SparseMatrixMPI{T,Ti}) where {T,Ti} = T
Base.eltype(::Type{SparseMatrixMPI{T,Ti}}) where {T,Ti} = T

# Norms

"""
    LinearAlgebra.norm(A::SparseMatrixMPI, p::Real=2)

Compute the p-norm of A treated as a vector of elements.
- `p=2` (default): Frobenius norm (sqrt of sum of squared absolute values)
- `p=1`: Sum of absolute values
- `p=Inf`: Maximum absolute value
"""
function LinearAlgebra.norm(A::SparseMatrixMPI{T}, p::Real=2) where T
    comm = MPI.COMM_WORLD
    local_vals = A.nzval

    if p == 2
        local_sum = sum(abs2, local_vals; init=zero(real(T)))
        global_sum = MPI.Allreduce(local_sum, MPI.SUM, comm)
        return sqrt(global_sum)
    elseif p == 1
        local_sum = sum(abs, local_vals; init=zero(real(T)))
        return MPI.Allreduce(local_sum, MPI.SUM, comm)
    elseif p == Inf
        local_max = isempty(local_vals) ? zero(real(T)) : maximum(abs, local_vals)
        return MPI.Allreduce(local_max, MPI.MAX, comm)
    else
        # General p-norm
        local_sum = sum(x -> abs(x)^p, local_vals; init=zero(real(T)))
        global_sum = MPI.Allreduce(local_sum, MPI.SUM, comm)
        return global_sum^(1 / p)
    end
end

"""
    LinearAlgebra.opnorm(A::SparseMatrixMPI, p::Real=1)

Compute the induced operator norm of A.
- `p=1`: Maximum absolute column sum
- `p=Inf`: Maximum absolute row sum

Note: `opnorm(A, 2)` (spectral norm) is not implemented as it requires SVD.
"""
function LinearAlgebra.opnorm(A::SparseMatrixMPI{T}, p::Real=1) where T
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    if p == Inf
        # Maximum absolute row sum
        # Each rank owns some rows, compute max row sum locally then reduce
        # _get_csc(A) is transposed, so columns of _get_csc(A) are rows of A
        local_nrows = A.nrows_local
        local_max = zero(real(T))
        for col in 1:local_nrows
            row_sum = zero(real(T))
            for idx in A.rowptr[col]:(A.rowptr[col+1]-1)
                row_sum += abs(A.nzval[idx])
            end
            local_max = max(local_max, row_sum)
        end
        return MPI.Allreduce(local_max, MPI.MAX, comm)

    elseif p == 1
        # Maximum absolute column sum
        # Columns are distributed, need to sum contributions from all ranks
        ncols = A.col_partition[end] - 1

        # Compute local contribution to each column sum
        # A.colval contains LOCAL column indices (compressed)
        # Map local→global using A.col_indices
        local_col_sums = zeros(real(T), ncols)
        col_indices = A.col_indices
        for (idx, local_col) in enumerate(A.colval)
            global_col = col_indices[local_col]
            local_col_sums[global_col] += abs(A.nzval[idx])
        end

        # Sum across all ranks
        global_col_sums = MPI.Allreduce(local_col_sums, MPI.SUM, comm)
        return maximum(global_col_sums; init=zero(real(T)))

    else
        error("opnorm(A, $p) is not implemented. Use p=1 or p=Inf.")
    end
end

# Lazy transpose support

"""
    Base.transpose(A::SparseMatrixMPI{T,Ti}) where {T,Ti}

Return a lazy transpose wrapper around A.
"""
Base.transpose(A::SparseMatrixMPI{T,Ti}) where {T,Ti} = Transpose(A)

"""
    conj(A::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}

Return a new SparseMatrixMPI with conjugated values.
"""
function Base.conj(A::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    # Conjugate values (works on CPU or GPU via broadcasting)
    new_nzval = conj.(A.nzval)

    # Structural hash is the same since structure didn't change
    # Reuse rowptr_target/colval_target since structure is identical
    return SparseMatrixMPI{T,Ti,AV}(A.structural_hash, A.row_partition, A.col_partition,
        A.col_indices, A.rowptr, A.colval, new_nzval,
        A.nrows_local, A.ncols_compressed, nothing, nothing, A.rowptr_target, A.colval_target)
end

"""
    Base.adjoint(A::SparseMatrixMPI{T,Ti}) where {T,Ti}

Return transpose(conj(A)), i.e., the conjugate transpose.
For real types, this is just transpose (no conjugation needed).
For complex types, this creates a conjugated copy then transposes.
"""
Base.adjoint(A::SparseMatrixMPI{T,Ti}) where {T<:Real,Ti} = transpose(A)
Base.adjoint(A::SparseMatrixMPI{T,Ti}) where {T<:Complex,Ti} = transpose(conj(A))

# Scalar multiplication

"""
    *(a::Number, A::SparseMatrixMPI{T,Ti}) where {T,Ti}

Scalar times matrix.
"""
function Base.:*(a::Number, A::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    RT = promote_type(typeof(a), T)
    # Scale values (works on CPU or GPU via broadcasting)
    new_nzval = RT.(a .* A.nzval)
    # Determine the new AV type based on the result
    AVR = typeof(new_nzval)

    # Convert structure arrays to target backend (may differ if type changed)
    rowptr_target = _to_target_backend(A.rowptr, AVR)
    colval_target = _to_target_backend(A.colval, AVR)
    return SparseMatrixMPI{RT,Ti,AVR}(A.structural_hash, A.row_partition, A.col_partition,
        A.col_indices, A.rowptr, A.colval, new_nzval,
        A.nrows_local, A.ncols_compressed, nothing, A.cached_symmetric, rowptr_target, colval_target)
end

"""
    *(A::SparseMatrixMPI{T,Ti}, a::Number) where {T,Ti}

Matrix times scalar.
"""
Base.:*(A::SparseMatrixMPI{T,Ti}, a::Number) where {T,Ti} = a * A

"""
    -(A::SparseMatrixMPI{T,Ti}) where {T,Ti}

Unary negation of a sparse matrix.
"""
Base.:-(A::SparseMatrixMPI{T,Ti}) where {T,Ti} = (-1) * A

# Type alias for transpose of SparseMatrixMPI
const TransposedSparseMatrixMPI{T,Ti,AV} = Transpose{T,SparseMatrixMPI{T,Ti,AV}}

"""
    *(a::Number, At::TransposedSparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}

Scalar times transposed matrix: a * transpose(A) = transpose(a * A).
"""
Base.:*(a::Number, At::TransposedSparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV} = transpose(a * At.parent)

"""
    *(At::TransposedSparseMatrixMPI{T,Ti,AV}, a::Number) where {T,Ti,AV}

Transposed matrix times scalar: transpose(A) * a = transpose(a * A).
"""
Base.:*(At::TransposedSparseMatrixMPI{T,Ti,AV}, a::Number) where {T,Ti,AV} = transpose(a * At.parent)

# Lazy transpose multiplication methods

"""
    *(At::Transpose, Bt::Transpose)

Compute transpose(A) * transpose(B) = transpose(B.parent * A.parent) lazily.
Returns a Transpose wrapper around the product B.parent * A.parent.
"""
function Base.:*(At::TransposedSparseMatrixMPI{T,Ti,AV}, Bt::TransposedSparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    A = At.parent
    B = Bt.parent
    return transpose(B * A)
end

"""
    *(At::Transpose, B::SparseMatrixMPI)

Compute transpose(A) * B by materializing the transpose of A first.
"""
function Base.:*(At::TransposedSparseMatrixMPI{T,Ti,AV}, B::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    A = At.parent
    A_transposed = SparseMatrixMPI(transpose(A))
    return A_transposed * B
end

"""
    *(A::SparseMatrixMPI, Bt::Transpose)

Compute A * transpose(B) by materializing the transpose of B first.
"""
function Base.:*(A::SparseMatrixMPI{T,Ti,AV}, Bt::TransposedSparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    B = Bt.parent
    B_transposed = SparseMatrixMPI(transpose(B))
    return A * B_transposed
end

"""
    Base.:*(At::TransposedSparseMatrixMPI{T,Ti,AV}, x::VectorMPI{T}) where {T,Ti,AV}

Compute transpose(A) * x by materializing the transpose of A first.
"""
function Base.:*(At::TransposedSparseMatrixMPI{T,Ti,AV}, x::VectorMPI{T,AVX}) where {T,Ti,AV,AVX}
    A = At.parent
    A_transposed = SparseMatrixMPI(transpose(A))
    return A_transposed * x
end

# ============================================================================
# Mixed Sparse-Dense Operations
# ============================================================================

"""
    Base.:*(A::SparseMatrixMPI{T,Ti,AV}, B::MatrixMPI{T,AM}) where {T,Ti,AV,AM}

Compute sparse matrix times dense matrix by column-by-column multiplication.
Returns a MatrixMPI with the same row partition as A and same backend as B.
"""
function Base.:*(A::SparseMatrixMPI{T,Ti,AV}, B::MatrixMPI{T,AM}) where {T,Ti,AV,AM}
    m = size(A, 1)
    n = size(B, 2)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Multiply column by column using existing sparse * vector
    columns = Vector{VectorMPI{T}}(undef, n)
    for k in 1:n
        # Extract k-th column of B as VectorMPI
        b_col = B[:, k]
        # Multiply sparse * vector
        columns[k] = A * b_col
    end

    # Get partition from first column result
    result_partition = columns[1].partition
    local_m = result_partition[rank+2] - result_partition[rank+1]

    # Build local matrix from column results on CPU
    local_result_cpu = Matrix{T}(undef, local_m, n)
    for k in 1:n
        local_result_cpu[:, k] = Array(columns[k].v)
    end

    # Convert back to original backend if needed
    local_result = B.A isa Matrix ? local_result_cpu : copyto!(similar(B.A, local_m, n), local_result_cpu)

    return MatrixMPI_local(local_result)
end

"""
    Base.:*(At::TransposedSparseMatrixMPI{T,Ti,AV}, B::MatrixMPI{T,AM}) where {T,Ti,AV,AM}

Compute transpose(A) * B by materializing the transpose of A first.
"""
function Base.:*(At::TransposedSparseMatrixMPI{T,Ti,AV}, B::MatrixMPI{T,AM}) where {T,Ti,AV,AM}
    A = At.parent
    A_transposed = SparseMatrixMPI(transpose(A))
    return A_transposed * B
end

# ============================================================================
# Extended SparseMatrixCSC API - Structural Queries
# ============================================================================

"""
    nnz(A::SparseMatrixMPI)

Return the total number of stored entries in the distributed sparse matrix.
Uses MPI.Allreduce to sum local counts across all ranks.
"""
function nnz(A::SparseMatrixMPI)
    comm = MPI.COMM_WORLD
    local_nnz = length(A.nzval)
    return MPI.Allreduce(local_nnz, MPI.SUM, comm)
end

"""
    issparse(::SparseMatrixMPI)

Return `true` (SparseMatrixMPI is always sparse).
"""
issparse(::SparseMatrixMPI) = true

# ============================================================================
# Extended SparseMatrixCSC API - Copy
# ============================================================================

"""
    Base.copy(A::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}

Create a deep copy of the distributed sparse matrix.
"""
function Base.copy(A::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    new_rowptr = copy(A.rowptr)
    new_colval = copy(A.colval)
    # Convert structure arrays to target backend
    rowptr_target = _to_target_backend(new_rowptr, AV)
    colval_target = _to_target_backend(new_colval, AV)
    return SparseMatrixMPI{T,Ti,AV}(
        A.structural_hash,
        copy(A.row_partition),
        copy(A.col_partition),
        copy(A.col_indices),
        new_rowptr,
        new_colval,
        copy(A.nzval),
        A.nrows_local,
        A.ncols_compressed,
        nothing,
        A.cached_symmetric,  # preserve symmetry cache on copy
        rowptr_target,
        colval_target
    )
end

# ============================================================================
# Extended SparseMatrixCSC API - Element-wise Operations
# ============================================================================

# Helper function for zero-preserving element-wise operations
function _map_nzval(f, A::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    # Infer result type from a sample element to handle empty arrays correctly
    # (Julia 1.10 infers Any for f.(empty_array) with lambdas)
    RT = typeof(f(zero(T)))

    # Create output array with correct type (handles empty arrays correctly)
    new_nzval = similar(A.nzval, RT)

    # Apply f to nzval (works on CPU or GPU via broadcasting)
    new_nzval .= f.(A.nzval)

    AVR = typeof(new_nzval)
    # Convert structure arrays to target backend (may differ if type changed)
    rowptr_target = _to_target_backend(A.rowptr, AVR)
    colval_target = _to_target_backend(A.colval, AVR)
    return SparseMatrixMPI{RT,Ti,AVR}(A.structural_hash, A.row_partition, A.col_partition,
        A.col_indices, A.rowptr, A.colval, new_nzval,
        A.nrows_local, A.ncols_compressed, nothing, A.cached_symmetric, rowptr_target, colval_target)
end

"""
    Base.abs(A::SparseMatrixMPI{T}) where T

Return a new SparseMatrixMPI with absolute values of all stored elements.
"""
Base.abs(A::SparseMatrixMPI) = _map_nzval(abs, A)

"""
    Base.abs2(A::SparseMatrixMPI{T}) where T

Return a new SparseMatrixMPI with squared absolute values of all stored elements.
"""
Base.abs2(A::SparseMatrixMPI) = _map_nzval(abs2, A)

"""
    Base.real(A::SparseMatrixMPI{T}) where T

Return a new SparseMatrixMPI containing the real parts of all stored elements.
"""
Base.real(A::SparseMatrixMPI) = _map_nzval(real, A)

"""
    Base.imag(A::SparseMatrixMPI{T}) where T

Return a new SparseMatrixMPI containing the imaginary parts of all stored elements.
"""
Base.imag(A::SparseMatrixMPI) = _map_nzval(imag, A)

"""
    Base.floor(A::SparseMatrixMPI)

Return a new SparseMatrixMPI with floor applied to all stored elements.
"""
Base.floor(A::SparseMatrixMPI) = _map_nzval(floor, A)

"""
    Base.ceil(A::SparseMatrixMPI)

Return a new SparseMatrixMPI with ceil applied to all stored elements.
"""
Base.ceil(A::SparseMatrixMPI) = _map_nzval(ceil, A)

"""
    Base.round(A::SparseMatrixMPI)

Return a new SparseMatrixMPI with round applied to all stored elements.
"""
Base.round(A::SparseMatrixMPI) = _map_nzval(round, A)

"""
    Base.map(f, A::SparseMatrixMPI{T}) where T

Apply function `f` to each stored (nonzero) element of A.
Returns a new SparseMatrixMPI. The function `f` should be zero-preserving (f(0) ≈ 0)
for the result to maintain proper sparse semantics.
"""
Base.map(f, A::SparseMatrixMPI) = _map_nzval(f, A)

# ============================================================================
# Extended SparseMatrixCSC API - Reductions
# ============================================================================

"""
    Base.sum(A::SparseMatrixMPI{T}; dims=nothing) where T

Compute the sum of elements in the distributed sparse matrix.

- `dims=nothing` (default): Sum all stored elements
- `dims=1`: Sum over rows, returns a VectorMPI of length n (column sums)
- `dims=2`: Sum over columns, returns a VectorMPI of length m (row sums)

Note: When dims=nothing, only stored (nonzero) values contribute to the sum.
"""
function Base.sum(A::SparseMatrixMPI{T,Ti,AV}; dims=nothing) where {T,Ti,AV}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    if dims === nothing
        # Sum all elements
        local_sum = sum(A.nzval; init=zero(T))
        return MPI.Allreduce(local_sum, MPI.SUM, comm)
    elseif dims == 1
        # Sum over rows: result is length-n vector (column sums)
        # A.colval contains LOCAL column indices (compressed)
        # Map local→global using A.col_indices
        # Stage nzval to CPU to avoid GPU scalar indexing
        nzval_cpu = Array(A.nzval)
        local_col_sums = zeros(T, n)
        col_indices = A.col_indices
        for (idx, local_col) in enumerate(A.colval)
            global_col = col_indices[local_col]
            local_col_sums[global_col] += nzval_cpu[idx]
        end
        global_col_sums = MPI.Allreduce(local_col_sums, MPI.SUM, comm)
        # Create uniform partition for the result vector
        partition = uniform_partition(n, nranks)
        local_range = partition[rank + 1]:(partition[rank + 2] - 1)
        local_sums = global_col_sums[local_range]
        # Convert to match input array type
        result_v = similar(A.nzval, length(local_sums))
        copyto!(result_v, local_sums)
        hash = compute_partition_hash(partition)
        return VectorMPI{T,typeof(result_v)}(hash, partition, result_v)
    elseif dims == 2
        # Sum over columns: result is length-m vector (row sums)
        local_nrows = A.nrows_local
        local_row_sums = zeros(T, local_nrows)
        # Stage nzval to CPU to avoid GPU scalar indexing
        nzval_cpu = Array(A.nzval)

        for local_col in 1:local_nrows
            for nz_idx in A.rowptr[local_col]:(A.rowptr[local_col+1]-1)
                local_row_sums[local_col] += nzval_cpu[nz_idx]
            end
        end

        # Result has A's row partition (partition is immutable, no need to copy)
        hash = compute_partition_hash(A.row_partition)
        # Convert to match input array type
        result_v = similar(A.nzval, local_nrows)
        copyto!(result_v, local_row_sums)
        return VectorMPI{T,typeof(result_v)}(hash, A.row_partition, result_v)
    else
        throw(ArgumentError("dims must be nothing, 1, or 2"))
    end
end

"""
    Base.maximum(A::SparseMatrixMPI{T}) where T

Compute the maximum stored element of the distributed sparse matrix.
Warning: Only considers stored values. If the matrix has implicit zeros and all
stored values are negative, the true maximum (zero) may not be returned.
"""
function Base.maximum(A::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD
    local_max = isempty(A.nzval) ? typemin(real(T)) : maximum(real, A.nzval)
    return MPI.Allreduce(local_max, MPI.MAX, comm)
end

"""
    Base.minimum(A::SparseMatrixMPI{T}) where T

Compute the minimum stored element of the distributed sparse matrix.
Warning: Only considers stored values. If the matrix has implicit zeros and all
stored values are positive, the true minimum (zero) may not be returned.
"""
function Base.minimum(A::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD
    local_min = isempty(A.nzval) ? typemax(real(T)) : minimum(real, A.nzval)
    return MPI.Allreduce(local_min, MPI.MIN, comm)
end

"""
    mean(A::SparseMatrixMPI{T}) where T

Compute the mean of all elements (including implicit zeros) in the distributed sparse matrix.
"""
function mean(A::SparseMatrixMPI{T}) where T
    m, n = size(A)
    total_elements = m * n
    return sum(A) / total_elements
end

"""
    _find_nzval_at_global_col(A::SparseMatrixMPI{T}, local_row::Int, global_col::Int) where T

Find the value at (local_row, global_col) in the local sparse storage.
Returns the value if found (including explicit zeros), or nothing if the
global column is not in this rank's col_indices.

Note: Returns nothing only when global_col is not in col_indices at all.
For structural zeros within existing columns, returns zero(T).
"""
function _find_nzval_at_global_col(A::SparseMatrixMPI{T}, local_row::Int, global_col::Int) where T
    col_indices = A.col_indices

    # Binary search to find local column index for global_col
    local_col_idx = searchsortedfirst(col_indices, global_col)
    if local_col_idx > length(col_indices) || col_indices[local_col_idx] != global_col
        return nothing  # global_col not in our local columns
    end

    # Use direct CSC indexing - returns 0 for structural zeros
    return _get_csc(A)[local_col_idx, local_row]
end

# CPU version that takes pre-staged nzval to avoid GPU scalar indexing
function _find_nzval_at_global_col_cpu(A::SparseMatrixMPI{T}, local_row::Int, global_col::Int, nzval_cpu::Vector{T}) where T
    col_indices = A.col_indices

    # Binary search to find local column index for global_col
    local_col_idx = searchsortedfirst(col_indices, global_col)
    if local_col_idx > length(col_indices) || col_indices[local_col_idx] != global_col
        return nothing  # global_col not in our local columns
    end

    # Create a temporary CSC matrix with CPU nzval for indexing
    csc = SparseMatrixCSC(A.ncols_compressed, A.nrows_local, A.rowptr, A.colval, nzval_cpu)
    return csc[local_col_idx, local_row]
end

"""
    tr(A::SparseMatrixMPI{T}) where T

Compute the trace (sum of main diagonal elements) of A.
"""
function tr(A::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    m, n = size(A)
    my_row_start = A.row_partition[rank+1]
    my_row_end = A.row_partition[rank+2] - 1

    local_trace = zero(T)
    for local_row in 1:(my_row_end-my_row_start+1)
        global_row = my_row_start + local_row - 1
        # Diagonal element is at (global_row, global_row) if within bounds
        if global_row <= n
            val = _find_nzval_at_global_col(A, local_row, global_row)
            if val !== nothing
                local_trace += val
            end
        end
    end

    return MPI.Allreduce(local_trace, MPI.SUM, comm)
end

# ============================================================================
# Extended SparseMatrixCSC API - Structural Operations
# ============================================================================

"""
    dropzeros(A::SparseMatrixMPI{T,Ti}) where {T,Ti}

Return a copy of A with explicitly stored zeros removed.
"""
function dropzeros(A::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV}
    comm = MPI.COMM_WORLD

    # Use SparseArrays.dropzeros on local AT
    new_AT = dropzeros(_get_csc(A))

    # Recompute col_indices since structure may have changed
    new_col_indices = isempty(new_AT.rowval) ? Int[] : unique(sort(new_AT.rowval))

    # Recompute structural hash since structure changed
    structural_hash = compute_structural_hash(A.row_partition, new_col_indices, new_AT, comm)

    # Extract CSR components from new_AT
    nrows_local = new_AT.n
    ncols_compressed = length(new_col_indices)

    # Convert nzval to match input array type (CPU or GPU)
    result_nzval = similar(A.nzval, length(new_AT.nzval))
    copyto!(result_nzval, new_AT.nzval)

    return SparseMatrixMPI{T,Ti,typeof(result_nzval)}(structural_hash, copy(A.row_partition), copy(A.col_partition),
        new_col_indices, new_AT.colptr, new_AT.rowval, result_nzval,
        nrows_local, ncols_compressed, nothing, nothing, new_AT.colptr, new_AT.rowval)
end

# ============================================================================
# Extended SparseMatrixCSC API - Diagonal and Triangular Operations
# ============================================================================

"""
    diag(A::SparseMatrixMPI{T}, k::Integer=0) where T

Extract the k-th diagonal of A as a VectorMPI.
- k=0: main diagonal
- k>0: k-th superdiagonal
- k<0: |k|-th subdiagonal

The result partition is derived from A's row partition: diagonal element d
is at row (row_offset + d), so the rank owning that row owns element d.
This is a purely local operation with no MPI communication.
"""
function diag(A::SparseMatrixMPI{T,Ti,AV}, k::Integer=0) where {T,Ti,AV}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Compute diagonal length and offsets
    if k >= 0
        diag_len = min(m, n - k)
        row_offset = 0
        col_offset = k
    else
        diag_len = min(m + k, n)
        row_offset = -k
        col_offset = 0
    end

    if diag_len <= 0
        # Empty diagonal - preserve array type
        partition = ones(Int, nranks + 1)
        hash = compute_partition_hash(partition)
        empty_v = similar(A.nzval, 0)
        return VectorMPI{T,typeof(empty_v)}(hash, partition, empty_v)
    end

    # Compute result partition from A's row partition (no communication needed)
    # Diagonal element d is at row (row_offset + d)
    # Rank r owns rows [A.row_partition[r+1], A.row_partition[r+2] - 1]
    # So rank r owns diagonal elements d where (row_offset + d) is in that range
    # i.e., d in [A.row_partition[r+1] - row_offset, A.row_partition[r+2] - row_offset - 1]
    result_partition = Vector{Int}(undef, nranks + 1)
    for r in 0:(nranks-1)
        first_d = A.row_partition[r+1] - row_offset
        result_partition[r+1] = clamp(first_d, 1, diag_len + 1)
    end
    result_partition[nranks+1] = diag_len + 1

    # My diagonal element range
    my_diag_start = result_partition[rank+1]
    my_diag_end = result_partition[rank+2] - 1
    my_diag_len = max(0, my_diag_end - my_diag_start + 1)

    # Extract local diagonal elements (no communication!)
    # Stage nzval to CPU to avoid GPU scalar indexing
    my_row_start = A.row_partition[rank+1]
    local_diag_cpu = Vector{T}(undef, my_diag_len)
    nzval_cpu = Array(A.nzval)

    for i in 1:my_diag_len
        d = my_diag_start + i - 1
        local_row = (row_offset + d) - my_row_start + 1
        global_col = col_offset + d
        val = _find_nzval_at_global_col_cpu(A, local_row, global_col, nzval_cpu)
        local_diag_cpu[i] = val === nothing ? zero(T) : val
    end

    # Convert to match input array type
    local_diag = similar(A.nzval, my_diag_len)
    copyto!(local_diag, local_diag_cpu)

    hash = compute_partition_hash(result_partition)
    return VectorMPI{T,typeof(local_diag)}(hash, result_partition, local_diag)
end

"""
    triu(A::SparseMatrixMPI{T,Ti}, k::Integer=0) where {T,Ti}

Return the upper triangular part of A, starting from the k-th diagonal.
- k=0: include main diagonal
- k>0: exclude k-1 diagonals below the k-th superdiagonal
- k<0: include |k| subdiagonals
"""
function triu(A::SparseMatrixMPI{T,Ti,AV}, k::Integer=0) where {T,Ti,AV}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    my_row_start = A.row_partition[rank+1]
    col_indices = A.col_indices

    # Stage nzval to CPU to avoid GPU scalar indexing
    nzval_cpu = Array(A.nzval)

    # Build new sparse structure keeping only upper triangular entries
    # Entry (i, j) is kept if j >= i + k, i.e., j - i >= k

    new_colptr = Vector{Int}(undef, A.nrows_local + 1)
    new_colptr[1] = 1

    # First pass: count entries per column
    nnz_per_col = zeros(Int, A.nrows_local)
    for local_col in 1:A.nrows_local
        global_row = my_row_start + local_col - 1
        for nz_idx in A.rowptr[local_col]:(A.rowptr[local_col+1]-1)
            local_j = A.colval[nz_idx]
            j = col_indices[local_j]  # convert to global column index
            # Keep if j >= global_row + k
            if j >= global_row + k
                nnz_per_col[local_col] += 1
            end
        end
    end

    # Build colptr
    for col in 1:length(nnz_per_col)
        new_colptr[col+1] = new_colptr[col] + nnz_per_col[col]
    end

    total_nnz = new_colptr[end] - 1
    new_rowval = Vector{Int}(undef, total_nnz)
    new_nzval_cpu = Vector{T}(undef, total_nnz)

    # Second pass: fill entries (keep local indices in new_rowval)
    idx = 1
    for local_col in 1:A.nrows_local
        global_row = my_row_start + local_col - 1
        for nz_idx in A.rowptr[local_col]:(A.rowptr[local_col+1]-1)
            local_j = A.colval[nz_idx]
            j = col_indices[local_j]  # convert to global column index
            if j >= global_row + k
                new_rowval[idx] = local_j  # keep local index
                new_nzval_cpu[idx] = nzval_cpu[nz_idx]
                idx += 1
            end
        end
    end

    # new_rowval contains local indices from A. Convert to global col_indices and compress.
    if isempty(new_rowval)
        new_col_indices = Int[]
        new_rowptr = new_colptr
        new_colval = Int[]
        new_nzval_final_cpu = T[]
    else
        local_used = unique(sort(new_rowval))
        new_col_indices = col_indices[local_used]  # convert to global
        # Compress: local_used is sorted, use binary search instead of Dict
        compressed_rowval = [searchsortedfirst(local_used, r) for r in new_rowval]
        new_rowptr = new_colptr
        new_colval = compressed_rowval
        new_nzval_final_cpu = new_nzval_cpu
    end

    structural_hash = compute_structural_hash(A.row_partition, new_col_indices, new_rowptr, new_colval, comm)

    nrows_local = A.nrows_local
    ncols_compressed = length(new_col_indices)

    # Convert nzval to match input array type
    new_nzval_final = similar(A.nzval, length(new_nzval_final_cpu))
    copyto!(new_nzval_final, new_nzval_final_cpu)

    return SparseMatrixMPI{T,Ti,typeof(new_nzval_final)}(structural_hash, copy(A.row_partition), copy(A.col_partition),
        new_col_indices, new_rowptr, new_colval, new_nzval_final,
        nrows_local, ncols_compressed, nothing, nothing, new_rowptr, new_colval)
end

"""
    tril(A::SparseMatrixMPI{T,Ti}, k::Integer=0) where {T,Ti}

Return the lower triangular part of A, starting from the k-th diagonal.
- k=0: include main diagonal
- k>0: include k superdiagonals
- k<0: exclude |k|-1 diagonals above the |k|-th subdiagonal
"""
function tril(A::SparseMatrixMPI{T,Ti,AV}, k::Integer=0) where {T,Ti,AV}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    my_row_start = A.row_partition[rank+1]
    col_indices = A.col_indices

    # Stage nzval to CPU to avoid GPU scalar indexing
    nzval_cpu = Array(A.nzval)

    # Keep entry (i, j) if j <= i + k

    new_colptr = Vector{Int}(undef, A.nrows_local + 1)
    new_colptr[1] = 1

    nnz_per_col = zeros(Int, A.nrows_local)
    for local_col in 1:A.nrows_local
        global_row = my_row_start + local_col - 1
        for nz_idx in A.rowptr[local_col]:(A.rowptr[local_col+1]-1)
            local_j = A.colval[nz_idx]
            j = col_indices[local_j]  # convert to global column index
            if j <= global_row + k
                nnz_per_col[local_col] += 1
            end
        end
    end

    for col in 1:length(nnz_per_col)
        new_colptr[col+1] = new_colptr[col] + nnz_per_col[col]
    end

    total_nnz = new_colptr[end] - 1
    new_rowval = Vector{Int}(undef, total_nnz)
    new_nzval_cpu = Vector{T}(undef, total_nnz)

    idx = 1
    for local_col in 1:A.nrows_local
        global_row = my_row_start + local_col - 1
        for nz_idx in A.rowptr[local_col]:(A.rowptr[local_col+1]-1)
            local_j = A.colval[nz_idx]
            j = col_indices[local_j]  # convert to global column index
            if j <= global_row + k
                new_rowval[idx] = local_j  # keep local index
                new_nzval_cpu[idx] = nzval_cpu[nz_idx]
                idx += 1
            end
        end
    end

    # new_rowval contains local indices from A. Convert to global col_indices and compress.
    if isempty(new_rowval)
        new_col_indices = Int[]
        new_rowptr = new_colptr
        new_colval = Int[]
        new_nzval_final_cpu = T[]
    else
        local_used = unique(sort(new_rowval))
        new_col_indices = col_indices[local_used]  # convert to global
        # Compress: local_used is sorted, use binary search instead of Dict
        compressed_rowval = [searchsortedfirst(local_used, r) for r in new_rowval]
        new_rowptr = new_colptr
        new_colval = compressed_rowval
        new_nzval_final_cpu = new_nzval_cpu
    end

    structural_hash = compute_structural_hash(A.row_partition, new_col_indices, new_rowptr, new_colval, comm)

    nrows_local = A.nrows_local
    ncols_compressed = length(new_col_indices)

    # Convert nzval to match input array type
    new_nzval_final = similar(A.nzval, length(new_nzval_final_cpu))
    copyto!(new_nzval_final, new_nzval_final_cpu)

    return SparseMatrixMPI{T,Ti,typeof(new_nzval_final)}(structural_hash, copy(A.row_partition), copy(A.col_partition),
        new_col_indices, new_rowptr, new_colval, new_nzval_final,
        nrows_local, ncols_compressed, nothing, nothing, new_rowptr, new_colval)
end

# ============================================================================
# Helpers for Distributed Operations (cat, blockdiag, etc.)
# ============================================================================

# Note: _partition_rows_by_owner is defined in dense.jl (included before this file)

"""
    _gather_rows_from_sparse(A::SparseMatrixMPI{T}, global_rows::AbstractVector{Int}) where T

Gather specific rows from a SparseMatrixMPI by global row index.

Returns a Vector of tuples `(global_row, global_col, value)` containing all nonzeros
in the requested rows. Uses point-to-point communication (Isend/Irecv) to minimize
data transfer - only exchanges rows that are actually needed.

# Arguments
- `A::SparseMatrixMPI{T}`: The distributed sparse matrix
- `global_rows::AbstractVector{Int}`: Global row indices to gather

# Returns
Vector{Tuple{Int,Int,T}} of (row, col, value) triplets for the requested rows.

Communication tags used: 30 (structure), 31 (values)
"""
function _gather_rows_from_sparse(A::SparseMatrixMPI{T}, global_rows::AbstractVector{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # NOTE: Do NOT early return here even if global_rows is empty!
    # All ranks must participate in MPI collectives below.

    my_row_start = A.row_partition[rank+1]

    # Step 1: Partition rows by owner
    rows_by_owner = _partition_rows_by_owner(global_rows, A.row_partition)

    # Step 2: Exchange row request counts via Alltoall
    send_counts = Int32[haskey(rows_by_owner, r) ? length(rows_by_owner[r]) : 0 for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Step 3: Send row requests to owner ranks
    row_send_reqs = MPI.Request[]
    row_recv_bufs = Dict{Int,Vector{Int32}}()
    row_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if send_counts[r+1] > 0 && r != rank
            rows_to_request = Int32.(rows_by_owner[r])
            req = MPI.Isend(rows_to_request, comm; dest=r, tag=30)
            push!(row_send_reqs, req)
        end
    end

    # Step 4: Receive row requests from other ranks
    for r in 0:(nranks-1)
        if recv_counts[r+1] > 0 && r != rank
            buf = Vector{Int32}(undef, recv_counts[r+1])
            req = MPI.Irecv!(buf, comm; source=r, tag=30)
            push!(row_recv_reqs, req)
            row_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(row_recv_reqs)
    MPI.Waitall(row_send_reqs)

    # Step 5: For each requesting rank, extract and send triplets for requested rows
    # First, compute how many triplets we'll send to each rank
    triplet_counts_to_send = Dict{Int,Int}()
    triplets_to_send = Dict{Int,Vector{Tuple{Int32,Int32,T}}}()

    # Ensure nzval is on CPU for scalar indexing (GPU arrays don't support scalar indexing)
    nzval_cpu = _ensure_cpu(A.nzval)

    for r in 0:(nranks-1)
        if recv_counts[r+1] > 0 && r != rank
            requested_rows = row_recv_bufs[r]
            triplets = Tuple{Int32,Int32,T}[]

            for global_row in requested_rows
                local_row = global_row - my_row_start + 1
                # _get_csc(A) has columns = local rows, so iterate column `local_row`
                for nz_idx in A.rowptr[local_row]:(A.rowptr[local_row+1]-1)
                    local_col = A.colval[nz_idx]
                    global_col = A.col_indices[local_col]
                    val = nzval_cpu[nz_idx]
                    push!(triplets, (Int32(global_row), Int32(global_col), val))
                end
            end

            triplet_counts_to_send[r] = length(triplets)
            triplets_to_send[r] = triplets
        end
    end

    # Step 6: Exchange triplet counts
    send_triplet_counts = Int32[get(triplet_counts_to_send, r, 0) for r in 0:(nranks-1)]
    recv_triplet_counts = MPI.Alltoall(MPI.UBuffer(send_triplet_counts, 1), comm)

    # Step 7: Send triplets (pack as flat arrays: rows, cols, vals)
    triplet_send_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if send_triplet_counts[r+1] > 0 && r != rank
            triplets = triplets_to_send[r]
            n_triplets = length(triplets)

            # Pack into arrays
            rows_buf = Int32[t[1] for t in triplets]
            cols_buf = Int32[t[2] for t in triplets]
            vals_buf = T[t[3] for t in triplets]

            req1 = MPI.Isend(rows_buf, comm; dest=r, tag=31)
            req2 = MPI.Isend(cols_buf, comm; dest=r, tag=32)
            req3 = MPI.Isend(vals_buf, comm; dest=r, tag=33)
            push!(triplet_send_reqs, req1, req2, req3)
        end
    end

    # Step 8: Receive triplets
    triplet_recv_bufs = Dict{Int,Tuple{Vector{Int32},Vector{Int32},Vector{T}}}()
    triplet_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_triplet_counts[r+1] > 0 && r != rank
            n = recv_triplet_counts[r+1]
            rows_buf = Vector{Int32}(undef, n)
            cols_buf = Vector{Int32}(undef, n)
            vals_buf = Vector{T}(undef, n)

            req1 = MPI.Irecv!(rows_buf, comm; source=r, tag=31)
            req2 = MPI.Irecv!(cols_buf, comm; source=r, tag=32)
            req3 = MPI.Irecv!(vals_buf, comm; source=r, tag=33)
            push!(triplet_recv_reqs, req1, req2, req3)

            triplet_recv_bufs[r] = (rows_buf, cols_buf, vals_buf)
        end
    end

    # Step 9: Collect local triplets for rows we own
    result = Tuple{Int,Int,T}[]

    if haskey(rows_by_owner, rank)
        for global_row in rows_by_owner[rank]
            local_row = global_row - my_row_start + 1
            for nz_idx in A.rowptr[local_row]:(A.rowptr[local_row+1]-1)
                local_col = A.colval[nz_idx]
                global_col = A.col_indices[local_col]
                val = nzval_cpu[nz_idx]
                push!(result, (global_row, global_col, val))
            end
        end
    end

    MPI.Waitall(triplet_recv_reqs)

    # Step 10: Unpack received triplets
    for r in 0:(nranks-1)
        if recv_triplet_counts[r+1] > 0 && r != rank
            rows_buf, cols_buf, vals_buf = triplet_recv_bufs[r]
            for i in eachindex(rows_buf)
                push!(result, (Int(rows_buf[i]), Int(cols_buf[i]), vals_buf[i]))
            end
        end
    end

    MPI.Waitall(triplet_send_reqs)

    return result
end

# ============================================================================
# Extended SparseMatrixCSC API - Diagonal Matrix Construction
# ============================================================================

"""
    _compute_spdiagm_size(kv::Pair{<:Integer, <:VectorMPI}...)

Compute the output matrix size for spdiagm from diagonal specifications.

For diagonal k with length n:
- If k >= 0: matrix is (n, n+k) minimum to fit
- If k < 0: matrix is (n+|k|, n) minimum to fit

Returns the maximum dimensions needed to fit all diagonals.
"""
function _compute_spdiagm_size(kv::Pair{<:Integer,<:VectorMPI}...)
    m = 0
    n = 0
    for (k, v) in kv
        vec_len = length(v)
        if k >= 0
            m = max(m, vec_len)
            n = max(n, vec_len + k)
        else
            m = max(m, vec_len - k)  # vec_len + |k|
            n = max(n, vec_len)
        end
    end
    return m, n
end

"""
    _diag_target_partition(row_partition::Vector{Int}, k::Integer, vec_len::Int) -> Vector{Int}

Compute the target partition for repartitioning a vector v of length vec_len
for use in diagonal k of a matrix with the given row_partition.

For diagonal k:
- k >= 0: vec_idx = global_row (element v[i] goes to position (i, i+k))
- k < 0: vec_idx = global_row + k (element v[i] goes to position (i-k, i))

The target partition ensures each rank owns the vector elements it needs for its rows.
"""
function _diag_target_partition(row_partition::Vector{Int}, k::Integer, vec_len::Int)
    nranks = length(row_partition) - 1
    target = Vector{Int}(undef, nranks + 1)

    for r in 0:(nranks-1)
        if k >= 0
            # vec_idx = global_row, so we need v starting at row_partition[r+1]
            target[r+1] = min(row_partition[r+1], vec_len + 1)
        else
            # vec_idx = global_row + k, so we need v starting at row_partition[r+1] + k
            target[r+1] = clamp(row_partition[r+1] + k, 1, vec_len + 1)
        end
    end
    target[nranks+1] = vec_len + 1

    return target
end

"""
    spdiagm(kv::Pair{<:Integer, <:VectorMPI}...)

Construct a sparse diagonal SparseMatrixMPI from pairs of diagonals and VectorMPI vectors.

Uses `repartition` to redistribute vector elements to the ranks that need them.
This provides plan caching for repeated operations and a fast path when partitions
already match (no communication needed).

# Example
```julia
v1 = VectorMPI([1.0, 2.0, 3.0])
v2 = VectorMPI([4.0, 5.0])
A = spdiagm(0 => v1, 1 => v2)  # Main diagonal and first superdiagonal
```
"""
function spdiagm(kv::Pair{<:Integer,<:VectorMPI}...)
    isempty(kv) && error("spdiagm requires at least one diagonal")

    # Fast path for single main diagonal
    if length(kv) == 1 && first(kv)[1] == 0
        return spdiagm(first(kv)[2])
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Determine element type and capture array type from first vector
    T = eltype(first(kv)[2])
    first_v = first(kv)[2]
    AV = typeof(first_v.v)  # Capture array type (Vector or MtlVector)
    for (_, v) in kv
        T = promote_type(T, eltype(v))
    end

    # Step 1: Compute output dimensions and row partition
    # Julia's spdiagm(kv...) returns a square matrix by default
    m, n = _compute_spdiagm_size(kv...)
    sz = max(m, n)
    m = n = sz
    row_partition = uniform_partition(m, nranks)

    my_row_start = row_partition[rank+1]
    my_row_end = row_partition[rank+2] - 1
    local_nrows = my_row_end - my_row_start + 1

    # Step 2: Repartition each vector to match the rows that need it
    # Uses plan caching and has fast path when partitions already match
    # Stage to CPU to avoid GPU scalar indexing
    repartitioned_cpu = Dict{Int, Vector{T}}()
    repartitioned_partitions = Dict{Int, Vector{Int}}()
    for (k, v) in kv
        target = _diag_target_partition(row_partition, k, length(v))
        v_repart = repartition(v, target)
        repartitioned_cpu[k] = Array(v_repart.v)
        repartitioned_partitions[k] = v_repart.partition
    end

    # Step 3: Build local triplets using vectorized operations
    local_I = Int[]
    local_J = Int[]
    local_V = T[]

    for (k, v) in kv
        vec_len = length(v)
        v_cpu = repartitioned_cpu[k]
        partition = repartitioned_partitions[k]
        my_v_start = partition[rank+1]
        my_v_end = partition[rank+2] - 1
        local_v_len = my_v_end - my_v_start + 1

        # Compute which local rows have valid diagonal entries
        # For k >= 0: global_row i -> col i+k, uses v[i]
        # For k < 0:  global_row i -> col i+k, uses v[i+k]
        if k >= 0
            # v[i] goes to (i, i+k), so row = vec_idx, col = vec_idx + k
            # We have v[my_v_start:my_v_end], these go to rows my_v_start:my_v_end
            first_row = max(my_row_start, my_v_start)
            last_row = min(my_row_end, my_v_end)
        else
            # v[i] goes to (i-k, i), so row = vec_idx - k, col = vec_idx
            # Row r uses v[r+k], col = r+k
            first_row = max(my_row_start, my_v_start - k)
            last_row = min(my_row_end, my_v_end - k)
        end

        if first_row <= last_row
            nentries = last_row - first_row + 1
            rows = first_row:last_row
            cols = rows .+ k

            # Filter to valid column range
            valid = (1 .<= cols .<= n)
            valid_rows = rows[valid]
            valid_cols = cols[valid]

            # Local row indices and vector indices
            local_rows = valid_rows .- my_row_start .+ 1
            if k >= 0
                v_indices = valid_rows .- my_v_start .+ 1
            else
                v_indices = (valid_rows .+ k) .- my_v_start .+ 1
            end

            append!(local_I, local_rows)
            append!(local_J, valid_cols)
            append!(local_V, v_cpu[v_indices])
        end
    end

    # Step 4: Build M^T directly as CSC (swap I↔J), then wrap in lazy transpose for CSR
    AT_local = isempty(local_I) ?
        SparseMatrixCSC(n, local_nrows, ones(Int, local_nrows + 1), Int[], T[]) :
        sparse(local_J, local_I, local_V, n, local_nrows)

    # Build CPU result first
    result_cpu = SparseMatrixMPI_local(transpose(AT_local); comm=comm)

    # Convert nzval to match input array type if GPU
    if AV !== Vector{T}
        result_nzval = similar(first_v.v, length(result_cpu.nzval))
        copyto!(result_nzval, result_cpu.nzval)
        return SparseMatrixMPI{T,Int,typeof(result_nzval)}(
            result_cpu.structural_hash, result_cpu.row_partition, result_cpu.col_partition,
            result_cpu.col_indices, result_cpu.rowptr, result_cpu.colval, result_nzval,
            result_cpu.nrows_local, result_cpu.ncols_compressed, nothing, nothing,
            result_cpu.rowptr_target, result_cpu.colval_target)
    end
    return result_cpu
end

"""
    spdiagm(m::Integer, n::Integer, kv::Pair{<:Integer, <:VectorMPI}...)

Construct an m×n sparse diagonal SparseMatrixMPI from pairs of diagonals and VectorMPI vectors.

Uses `repartition` to redistribute vector elements to the ranks that need them.
This provides plan caching for repeated operations and a fast path when partitions
already match (no communication needed).

# Example
```julia
v = VectorMPI([1.0, 2.0])
A = spdiagm(4, 4, 0 => v)  # 4×4 matrix with [1,2] on main diagonal
```
"""
function spdiagm(m::Integer, n::Integer, kv::Pair{<:Integer,<:VectorMPI}...)
    isempty(kv) && error("spdiagm requires at least one diagonal")

    # Fast path for single main diagonal with square matrix matching vector length
    if length(kv) == 1 && first(kv)[1] == 0 && m == n == length(first(kv)[2])
        return spdiagm(first(kv)[2])
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Determine element type and capture array type from first vector
    T = eltype(first(kv)[2])
    first_v = first(kv)[2]
    AV = typeof(first_v.v)  # Capture array type (Vector or MtlVector)
    for (_, v) in kv
        T = promote_type(T, eltype(v))
    end

    # Step 1: Compute output row partition
    row_partition = uniform_partition(m, nranks)

    my_row_start = row_partition[rank+1]
    my_row_end = row_partition[rank+2] - 1
    local_nrows = my_row_end - my_row_start + 1

    # Step 2: Repartition each vector to match the rows that need it
    # Stage to CPU to avoid GPU scalar indexing
    repartitioned_cpu = Dict{Int, Vector{T}}()
    repartitioned_partitions = Dict{Int, Vector{Int}}()
    for (k, v) in kv
        target = _diag_target_partition(row_partition, k, length(v))
        v_repart = repartition(v, target)
        repartitioned_cpu[k] = Array(v_repart.v)
        repartitioned_partitions[k] = v_repart.partition
    end

    # Step 3: Build local triplets
    local_I = Int[]
    local_J = Int[]
    local_V = T[]

    for (k, v) in kv
        vec_len = length(v)
        v_cpu = repartitioned_cpu[k]
        partition = repartitioned_partitions[k]
        my_v_start = partition[rank+1]

        for local_row_idx in 1:local_nrows
            global_row = my_row_start + local_row_idx - 1

            if k >= 0
                vec_idx = global_row
                col = global_row + k
            else
                vec_idx = global_row + k
                col = vec_idx
            end

            if 1 <= vec_idx <= vec_len && 1 <= col <= n && 1 <= global_row <= m
                local_v_idx = vec_idx - my_v_start + 1
                push!(local_I, local_row_idx)
                push!(local_J, col)
                push!(local_V, v_cpu[local_v_idx])
            end
        end
    end

    # Step 4: Build M^T directly as CSC (swap I↔J), then wrap in lazy transpose for CSR
    AT_local = isempty(local_I) ?
        SparseMatrixCSC(n, local_nrows, ones(Int, local_nrows + 1), Int[], T[]) :
        sparse(local_J, local_I, local_V, n, local_nrows)

    # Build CPU result first
    result_cpu = SparseMatrixMPI_local(transpose(AT_local); comm=comm)

    # Convert nzval to match input array type if GPU
    if AV !== Vector{T}
        result_nzval = similar(first_v.v, length(result_cpu.nzval))
        copyto!(result_nzval, result_cpu.nzval)
        return SparseMatrixMPI{T,Int,typeof(result_nzval)}(
            result_cpu.structural_hash, result_cpu.row_partition, result_cpu.col_partition,
            result_cpu.col_indices, result_cpu.rowptr, result_cpu.colval, result_nzval,
            result_cpu.nrows_local, result_cpu.ncols_compressed, nothing, nothing,
            result_cpu.rowptr_target, result_cpu.colval_target)
    end
    return result_cpu
end

"""
    spdiagm(v::VectorMPI)

Construct a sparse diagonal SparseMatrixMPI with VectorMPI v on the main diagonal.

# Example
```julia
v = VectorMPI([1.0, 2.0, 3.0])
A = spdiagm(v)  # 3×3 diagonal matrix
```
"""
function spdiagm(v::VectorMPI{T,AV}) where {T,AV}
    # Ultra-fast path for main diagonal: reuse cached structure when possible
    n = length(v)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    local_n = length(v.v)

    vec_hash = v.structural_hash
    row_partition = v.partition  # Use same partition as input vector
    col_partition = v.partition  # Square matrix, same column partition

    if haskey(_diag_structure_cache, vec_hash)
        # Reuse cached structure - use new explicit arrays constructor
        cache = _diag_structure_cache[vec_hash]
        nzval = copy(v.v)  # Preserves GPU/CPU type
        # Match input array type
        return SparseMatrixMPI{T,Int,typeof(nzval)}(cache.structural_hash, row_partition, col_partition,
                                       cache.col_indices, cache.colptr, cache.rowval, nzval,
                                       local_n, length(cache.col_indices), nothing, true, cache.colptr, cache.rowval)
    end

    # Build CSR structure directly using explicit arrays
    # rowptr indexes into colval/nzval for each row
    # colval contains LOCAL column indices (1:ncols_compressed)
    # col_indices maps local column index -> global column index
    my_start = v.partition[rank+1]
    rowptr = collect(1:(local_n+1))  # Each row has exactly 1 entry
    colval = collect(1:local_n)  # LOCAL column indices (compressed)
    nzval = copy(v.v)  # Preserves GPU/CPU type

    # col_indices maps local column index -> global column index
    col_indices = collect(my_start:(my_start + local_n - 1))

    # Compute and cache the structure
    diag_hash = compute_structural_hash(row_partition, col_indices, rowptr, colval, comm)
    _diag_structure_cache[vec_hash] = DiagStructureCache(rowptr, colval, col_indices, diag_hash)

    # Diagonal matrices are always symmetric
    # Match input array type
    return SparseMatrixMPI{T,Int,typeof(nzval)}(diag_hash, row_partition, col_partition, col_indices,
                               rowptr, colval, nzval, local_n, length(col_indices), nothing, true, rowptr, colval)
end

"""
    spdiagm(m::Integer, n::Integer, v::VectorMPI)

Construct an m×n sparse diagonal SparseMatrixMPI with VectorMPI v on the main diagonal.

# Example
```julia
v = VectorMPI([1.0, 2.0])
A = spdiagm(4, 4, v)  # 4×4 matrix with [1,2,0,0] on main diagonal
```
"""
function spdiagm(m::Integer, n::Integer, v::VectorMPI)
    return spdiagm(m, n, 0 => v)
end

# ============================================================================
# Mixed Dense-Sparse Operations (Dense on left side)
# ============================================================================

"""
    Base.:*(A::MatrixMPI{T,AM}, B::SparseMatrixMPI{T,Ti,AV}) where {T,AM,Ti,AV}

Compute dense matrix times sparse matrix.
Uses column-by-column approach. Returns a MatrixMPI with the same backend as A.
"""
function Base.:*(A::MatrixMPI{T,AM}, B::SparseMatrixMPI{T,Ti,AV}) where {T,AM,Ti,AV}
    m = size(A, 1)
    n = size(B, 2)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Multiply row by row: result[i, :] = A[i, :] * B
    # Transpose approach: transpose(B) * transpose(A) gives transpose(A * B)
    # We compute column-by-column of the result instead

    columns = Vector{VectorMPI{T}}(undef, n)
    B_t = SparseMatrixMPI(transpose(B))

    for k in 1:n
        # Column k of result = A * (column k of B) = A * B_t'[:, k]
        b_col = B[:, k]
        columns[k] = A * b_col
    end

    # Get partition from first column result
    result_partition = columns[1].partition
    local_m = result_partition[rank+2] - result_partition[rank+1]

    # Build local matrix from column results on CPU
    local_result_cpu = Matrix{T}(undef, local_m, n)
    for k in 1:n
        local_result_cpu[:, k] = Array(columns[k].v)
    end

    # Convert back to original backend if needed
    local_result = A.A isa Matrix ? local_result_cpu : copyto!(similar(A.A, local_m, n), local_result_cpu)

    return MatrixMPI_local(local_result)
end

"""
    Base.:*(At::TransposedMatrixMPI{T,AM}, B::SparseMatrixMPI{T,Ti,AV}) where {T,AM,Ti,AV}

Compute transpose(A) * B where A is dense and B is sparse.
Returns a MatrixMPI with the same backend as A.
"""
function Base.:*(At::TransposedMatrixMPI{T,AM}, B::SparseMatrixMPI{T,Ti,AV}) where {T,AM,Ti,AV}
    A = At.parent
    n = size(B, 2)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Multiply column by column
    columns = Vector{VectorMPI{T}}(undef, n)
    for k in 1:n
        b_col = B[:, k]
        columns[k] = transpose(A) * b_col
    end

    # Get partition from first column result
    result_partition = columns[1].partition
    local_m = result_partition[rank+2] - result_partition[rank+1]

    # Build local matrix from column results on CPU
    local_result_cpu = Matrix{T}(undef, local_m, n)
    for k in 1:n
        local_result_cpu[:, k] = Array(columns[k].v)
    end

    # Convert back to original backend if needed
    local_result = A.A isa Matrix ? local_result_cpu : copyto!(similar(A.A, local_m, n), local_result_cpu)

    return MatrixMPI_local(local_result)
end

# ============================================================================
# UniformScaling Support
# ============================================================================

"""
    IdentityAdditionPlan{T,Ti}

A precomputed plan for adding/subtracting λI to/from a sparse matrix.
Stores the result structure and diagonal indices for efficient execution.

If A already has all diagonal entries, the result has the same structure as A.
Otherwise, the result may have additional diagonal entries.
"""
mutable struct IdentityAdditionPlan{T,Ti}
    # Result matrix structure (colptr and rowval may differ from A if diagonals are added)
    colptr::Vector{Ti}
    rowval::Vector{Ti}

    # Indices for execution:
    # - A_src[k] -> A.nzval index to copy from
    # - dst[k] -> result.nzval index to copy to
    A_src::Vector{Ti}
    dst::Vector{Ti}

    # Diagonal indices in result.nzval (for adding λ)
    diag_indices::Vector{Ti}

    # Whether structure matches A (no new diagonals needed)
    same_structure::Bool

    # Result metadata
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    col_indices::Vector{Int}

    # Cached structural hash
    structural_hash::Blake3Hash
end

"""
    IdentityAdditionPlan(A::SparseMatrixMPI{T,Ti}) where {T,Ti}

Create a plan for computing A + λI.
The plan precomputes the result structure and diagonal indices.
"""
function IdentityAdditionPlan(A::SparseMatrixMPI{T,Ti}) where {T,Ti}
    m, n = size(A)
    if m != n
        throw(DimensionMismatch("matrix must be square to add UniformScaling"))
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1
    local_nrows = my_row_end - my_row_start + 1

    AT = _get_csc(A)  # underlying CSC (ncols x local_nrows)
    col_indices = A.col_indices

    # Check if all diagonals already exist in A
    all_diags_exist = true
    for local_row in 1:local_nrows
        global_row = my_row_start + local_row - 1
        found = false
        for nz_idx in AT.colptr[local_row]:(AT.colptr[local_row+1]-1)
            if col_indices[AT.rowval[nz_idx]] == global_row
                found = true
                break
            end
        end
        if !found && global_row <= n
            all_diags_exist = false
            break
        end
    end

    if all_diags_exist
        # Fast path: result has same structure as A
        A_src = Ti.(1:length(AT.nzval))
        dst = Ti.(1:length(AT.nzval))

        # Find diagonal indices
        diag_indices = Ti[]
        for local_row in 1:local_nrows
            global_row = my_row_start + local_row - 1
            for nz_idx in AT.colptr[local_row]:(AT.colptr[local_row+1]-1)
                if col_indices[AT.rowval[nz_idx]] == global_row
                    push!(diag_indices, nz_idx)
                    break
                end
            end
        end

        # Compute structural hash (same as A since structure unchanged)
        structural_hash = A.structural_hash
        if structural_hash === nothing
            structural_hash = compute_structural_hash(A.row_partition, col_indices, AT, comm)
        end

        return IdentityAdditionPlan{T,Ti}(
            AT.colptr, AT.rowval, A_src, dst, diag_indices, true,
            A.row_partition, A.col_partition, col_indices, structural_hash
        )
    else
        # Slow path: need to add diagonal entries
        # Build new structure with diagonals

        # First pass: count entries per row including new diagonals
        new_col_indices_set = Set(col_indices)
        for local_row in 1:local_nrows
            global_row = my_row_start + local_row - 1
            if global_row <= n
                push!(new_col_indices_set, global_row)
            end
        end
        new_col_indices = sort(collect(new_col_indices_set))

        # Build col_map: old local col -> new local col
        col_map = zeros(Int, length(col_indices))
        for (old_idx, global_col) in enumerate(col_indices)
            col_map[old_idx] = searchsortedfirst(new_col_indices, global_col)
        end

        # Build new structure
        colptr = Vector{Ti}(undef, local_nrows + 1)
        colptr[1] = 1

        for local_row in 1:local_nrows
            global_row = my_row_start + local_row - 1
            diag_new_col = searchsortedfirst(new_col_indices, global_row)

            # Count entries: existing + possibly new diagonal
            count = AT.colptr[local_row+1] - AT.colptr[local_row]
            has_diag = false
            for nz_idx in AT.colptr[local_row]:(AT.colptr[local_row+1]-1)
                if col_indices[AT.rowval[nz_idx]] == global_row
                    has_diag = true
                    break
                end
            end
            if !has_diag && global_row <= n
                count += 1
            end
            colptr[local_row + 1] = colptr[local_row] + count
        end

        nnz_result = colptr[local_nrows + 1] - 1
        rowval = Vector{Ti}(undef, nnz_result)
        A_src = Ti[]
        dst = Ti[]
        diag_indices = Ti[]

        # Second pass: build rowval and index mappings
        for local_row in 1:local_nrows
            global_row = my_row_start + local_row - 1
            diag_new_col = global_row <= n ? searchsortedfirst(new_col_indices, global_row) : 0

            result_idx = colptr[local_row]
            has_diag = false

            # Collect (new_col_idx, old_nz_idx or -1 for new diag)
            entries = Tuple{Int,Int}[]
            for nz_idx in AT.colptr[local_row]:(AT.colptr[local_row+1]-1)
                new_col = col_map[AT.rowval[nz_idx]]
                push!(entries, (new_col, nz_idx))
                if col_indices[AT.rowval[nz_idx]] == global_row
                    has_diag = true
                end
            end
            if !has_diag && global_row <= n
                push!(entries, (diag_new_col, -1))  # -1 marks new diagonal
            end

            # Sort by new column index
            sort!(entries, by=x->x[1])

            for (new_col, old_nz_idx) in entries
                rowval[result_idx] = new_col
                if old_nz_idx > 0
                    push!(A_src, old_nz_idx)
                    push!(dst, result_idx)
                    if col_indices[AT.rowval[old_nz_idx]] == global_row
                        push!(diag_indices, result_idx)
                    end
                else
                    # New diagonal entry (will be set to λ, not copied from A)
                    push!(diag_indices, result_idx)
                end
                result_idx += 1
            end
        end

        # Compute structural hash
        temp_nzval = Vector{T}(undef, nnz_result)
        temp_csc = SparseMatrixCSC(length(new_col_indices), local_nrows, colptr, rowval, temp_nzval)
        structural_hash = compute_structural_hash(A.row_partition, new_col_indices, temp_csc, comm)

        return IdentityAdditionPlan{T,Ti}(
            colptr, rowval, A_src, dst, diag_indices, false,
            A.row_partition, A.col_partition, new_col_indices, structural_hash
        )
    end
end

"""
    _get_identity_addition_plan(A::SparseMatrixMPI{T,Ti}) where {T,Ti}

Get or create a cached identity addition plan for A + λI.
"""
function _get_identity_addition_plan(A::SparseMatrixMPI{T,Ti}) where {T,Ti}
    _ensure_hash(A)
    key = (A.structural_hash, T, Ti)
    if haskey(_identity_addition_plan_cache, key)
        return _identity_addition_plan_cache[key]::IdentityAdditionPlan{T,Ti}
    end
    plan = IdentityAdditionPlan(A)
    _identity_addition_plan_cache[key] = plan
    return plan
end

"""
    Base.:+(A::SparseMatrixMPI{T,Ti}, J::UniformScaling) where {T,Ti}

Add a scalar multiple of the identity matrix to A using plan-based execution.
Returns A + λI where J = λI.
"""
function Base.:+(A::SparseMatrixMPI{T,Ti}, J::UniformScaling) where {T,Ti}
    plan = _get_identity_addition_plan(A)

    λ = J.λ
    RT = promote_type(T, typeof(λ))

    # Allocate result
    nnz_result = plan.colptr[end] - 1
    nzval = Vector{RT}(undef, nnz_result)

    # Ensure A.nzval is on CPU for scalar indexing
    A_nzval_cpu = Array(A.nzval)

    if plan.same_structure
        # Fast path: copy all values, then add λ to diagonals
        @inbounds for k in eachindex(plan.A_src)
            nzval[plan.dst[k]] = A_nzval_cpu[plan.A_src[k]]
        end
        @inbounds for k in plan.diag_indices
            nzval[k] += RT(λ)
        end
    else
        # Slow path: zero first, copy from A, then set diagonals
        fill!(nzval, zero(RT))
        @inbounds for k in eachindex(plan.A_src)
            nzval[plan.dst[k]] = A_nzval_cpu[plan.A_src[k]]
        end
        # Diagonal indices include both existing and new diagonals
        # For existing diagonals, we've already copied the value, so just add λ
        # For new diagonals, value is 0, so adding λ sets it correctly
        @inbounds for k in plan.diag_indices
            nzval[k] += RT(λ)
        end
    end

    # Use explicit CSR arrays for the new struct format
    nrows_local = length(plan.colptr) - 1
    ncols_compressed = length(plan.col_indices)

    # For CPU, rowptr_target and colval_target are the same as rowptr and colval
    return SparseMatrixMPI{RT,Ti,Vector{RT}}(plan.structural_hash, plan.row_partition, plan.col_partition,
        plan.col_indices, plan.colptr, plan.rowval, nzval, nrows_local, ncols_compressed, nothing, nothing, plan.colptr, plan.rowval)
end

"""
    Base.:-(A::SparseMatrixMPI{T,Ti}, J::UniformScaling) where {T,Ti}

Subtract a scalar multiple of the identity matrix from A.
Returns A - λI where J = λI.
"""
function Base.:-(A::SparseMatrixMPI{T,Ti}, J::UniformScaling) where {T,Ti}
    return A + UniformScaling(-J.λ)
end

"""
    Base.:+(J::UniformScaling, A::SparseMatrixMPI{T,Ti}) where {T,Ti}

Add a sparse matrix to a scalar multiple of the identity.
Returns λI + A where J = λI.
"""
function Base.:+(J::UniformScaling, A::SparseMatrixMPI{T,Ti}) where {T,Ti}
    return A + J
end

"""
    Base.:-(J::UniformScaling, A::SparseMatrixMPI{T,Ti}) where {T,Ti}

Subtract a sparse matrix from a scalar multiple of the identity.
Returns λI - A where J = λI.

This is optimized to avoid creating an intermediate -A matrix.
"""
function Base.:-(J::UniformScaling, A::SparseMatrixMPI{T,Ti}) where {T,Ti}
    plan = _get_identity_addition_plan(A)

    λ = J.λ
    RT = promote_type(T, typeof(λ))

    # Allocate result
    nnz_result = plan.colptr[end] - 1
    nzval = Vector{RT}(undef, nnz_result)

    # Ensure A.nzval is on CPU for scalar indexing
    A_nzval_cpu = Array(A.nzval)

    if plan.same_structure
        # Fast path: copy negated values, then add λ to diagonals
        @inbounds for k in eachindex(plan.A_src)
            nzval[plan.dst[k]] = -A_nzval_cpu[plan.A_src[k]]
        end
        @inbounds for k in plan.diag_indices
            nzval[k] += RT(λ)
        end
    else
        # Slow path: zero first, copy negated from A, then add λ to diagonals
        fill!(nzval, zero(RT))
        @inbounds for k in eachindex(plan.A_src)
            nzval[plan.dst[k]] = -A_nzval_cpu[plan.A_src[k]]
        end
        @inbounds for k in plan.diag_indices
            nzval[k] += RT(λ)
        end
    end

    # Use explicit CSR arrays for the new struct format
    nrows_local = length(plan.colptr) - 1
    ncols_compressed = length(plan.col_indices)

    # For CPU, rowptr_target and colval_target are the same as rowptr and colval
    return SparseMatrixMPI{RT,Ti,Vector{RT}}(plan.structural_hash, plan.row_partition, plan.col_partition,
        plan.col_indices, plan.colptr, plan.rowval, nzval, nrows_local, ncols_compressed, nothing, nothing, plan.colptr, plan.rowval)
end

# ============================================================================
# SparseRepartitionPlan: Repartition a SparseMatrixMPI to a new row partition
# ============================================================================

"""
    SparseRepartitionPlan{T,Ti}

Communication plan for repartitioning a SparseMatrixMPI to a new row partition.
The col_partition remains unchanged - only rows are redistributed.

# Fields
## Send-side
- `send_rank_ids::Vector{Int}`: Ranks we send rows to (0-indexed)
- `send_local_row_ranges::Vector{UnitRange{Int}}`: For each rank, range of local rows to send
- `send_nnz_counts::Vector{Int}`: Number of nonzeros to send to each rank
- `send_bufs::Vector{Vector{T}}`: Pre-allocated send value buffers

## Receive-side
- `recv_rank_ids::Vector{Int}`: Ranks we receive rows from (0-indexed)
- `recv_nnz_counts::Vector{Int}`: Number of nonzeros to receive from each rank
- `recv_bufs::Vector{Vector{T}}`: Pre-allocated receive value buffers
- `recv_value_offsets::Vector{Int}`: Offset into result_nzval for each recv rank

## Local data
- `local_src_row_range::UnitRange{Int}`: Local rows (in _get_csc(A) columns) that stay
- `local_value_offset::Int`: Offset into result_nzval for local values
- `local_nnz::Int`: Number of local nonzeros

## Result metadata (EAGER)
- `result_row_partition::Vector{Int}`: Target row partition
- `result_col_partition::Vector{Int}`: Column partition (unchanged)
- `result_col_indices::Vector{Int}`: Union of col_indices from received rows
- `result_AT::SparseMatrixCSC{T,Ti}`: Pre-built sparse structure (values to be filled)
- `result_structural_hash::Blake3Hash`: Pre-computed structural hash
- `compress_map::Vector{Int}`: global_col -> local_col for result
"""
mutable struct SparseRepartitionPlan{T,Ti}
    # Send-side
    send_rank_ids::Vector{Int}
    send_local_row_ranges::Vector{UnitRange{Int}}
    send_nnz_counts::Vector{Int}
    send_bufs::Vector{Vector{T}}
    send_reqs::Vector{MPI.Request}

    # Receive-side
    recv_rank_ids::Vector{Int}
    recv_nnz_counts::Vector{Int}
    recv_bufs::Vector{Vector{T}}
    recv_reqs::Vector{MPI.Request}
    recv_value_offsets::Vector{Int}

    # Local data
    local_src_row_range::UnitRange{Int}
    local_value_offset::Int
    local_nnz::Int

    # Result metadata (EAGER)
    result_row_partition::Vector{Int}
    result_col_partition::Vector{Int}
    result_col_indices::Vector{Int}
    result_AT::SparseMatrixCSC{T,Ti}
    result_structural_hash::Blake3Hash
    compress_map::Vector{Int}
end

"""
    SparseRepartitionPlan(A::SparseMatrixMPI{T,Ti}, p::Vector{Int}) where {T,Ti}

Create a communication plan to repartition `A` to have row partition `p`.
The col_partition remains unchanged.

The plan:
1. Computes row overlaps between source and target partitions
2. Exchanges sparse structure (colptr, rowval) to determine result structure
3. Builds result col_indices, compress_map, and AT structure
4. Computes structural hash eagerly
5. Pre-allocates value buffers
"""
function SparseRepartitionPlan(A::SparseMatrixMPI{T,Ti}, p::Vector{Int}) where {T,Ti}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    AT = _get_csc(A)  # Underlying CSC storage

    # Source partition info
    src_start = A.row_partition[rank+1]
    src_end = A.row_partition[rank+2] - 1

    # Target partition info
    dst_start = p[rank+1]
    dst_end = p[rank+2] - 1
    result_local_nrows = max(0, dst_end - dst_start + 1)

    # ========== PHASE 1: Determine row overlaps ==========

    # For each destination rank r, compute overlap of our rows with rank r's target
    send_row_ranges_map = Dict{Int, UnitRange{Int}}()
    for r in 0:(nranks-1)
        r_start = p[r+1]
        r_end = p[r+2] - 1
        if r_end < r_start
            continue
        end
        overlap_start = max(src_start, r_start)
        overlap_end = min(src_end, r_end)
        if overlap_start <= overlap_end
            # Convert to local row indices (columns in AT)
            local_start = overlap_start - src_start + 1
            local_end = overlap_end - src_start + 1
            send_row_ranges_map[r] = local_start:local_end
        end
    end

    # ========== PHASE 2: Exchange row counts and structure ==========

    # Exchange row counts
    send_row_counts = Int32[haskey(send_row_ranges_map, r) ? length(send_row_ranges_map[r]) : 0 for r in 0:(nranks-1)]
    recv_row_counts = MPI.Alltoall(MPI.UBuffer(send_row_counts, 1), comm)

    # For each rank we send to, compute structure info: [nrows, colptr..., nnz, rowval_globals...]
    struct_send_bufs = Dict{Int, Vector{Int}}()
    struct_send_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if haskey(send_row_ranges_map, r) && r != rank
            row_range = send_row_ranges_map[r]
            nrows = length(row_range)

            # Build colptr for these rows
            colptr = Vector{Int}(undef, nrows + 1)
            colptr[1] = 1
            for (i, local_row) in enumerate(row_range)
                ptr_start = AT.colptr[local_row]
                ptr_end = AT.colptr[local_row+1] - 1
                row_nnz = ptr_end - ptr_start + 1
                colptr[i+1] = colptr[i] + row_nnz
            end
            nnz_to_send = colptr[end] - 1

            # Convert local col indices to global for sending
            rowval_globals = Vector{Int}(undef, nnz_to_send)
            idx = 1
            for local_row in row_range
                for ptr in AT.colptr[local_row]:(AT.colptr[local_row+1]-1)
                    local_col = AT.rowval[ptr]
                    rowval_globals[idx] = A.col_indices[local_col]
                    idx += 1
                end
            end

            # Pack: [nrows, colptr..., nnz, rowval_globals...]
            struct_buf = vcat([nrows], colptr, [nnz_to_send], rowval_globals)
            struct_send_bufs[r] = struct_buf
            req = MPI.Isend(struct_buf, comm; dest=r, tag=94)
            push!(struct_send_reqs, req)
        end
    end

    # Receive structure from other ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    # First exchange structure sizes
    send_struct_sizes = Int32[haskey(struct_send_bufs, r) ? length(struct_send_bufs[r]) : 0 for r in 0:(nranks-1)]
    recv_struct_sizes = MPI.Alltoall(MPI.UBuffer(send_struct_sizes, 1), comm)

    for r in 0:(nranks-1)
        if recv_row_counts[r+1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_struct_sizes[r+1])
            req = MPI.Irecv!(buf, comm; source=r, tag=94)
            push!(struct_recv_reqs, req)
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # ========== PHASE 3: Build result structure ==========

    # Collect all data that will be in our result (local + received)
    # We need to build: result_colptr, result_rowval (global), then compress

    # Data structure to hold rows in order: list of (dst_local_row, colptr_slice, rowval_globals)
    row_data = Vector{Tuple{Int, Vector{Int}, Vector{Int}}}()

    # Local rows that stay on this rank
    local_src_row_range = 1:0
    local_nnz = 0
    if haskey(send_row_ranges_map, rank)
        local_src_row_range = send_row_ranges_map[rank]
        for local_row in local_src_row_range
            global_row = src_start + local_row - 1
            dst_local_row = global_row - dst_start + 1

            ptr_start = AT.colptr[local_row]
            ptr_end = AT.colptr[local_row+1] - 1
            row_nnz = ptr_end - ptr_start + 1

            rowval_globals = [A.col_indices[AT.rowval[ptr]] for ptr in ptr_start:ptr_end]
            push!(row_data, (dst_local_row, [1, 1 + row_nnz], rowval_globals))
            local_nnz += row_nnz
        end
    end

    # Received rows
    recv_rank_ids = Int[]
    recv_nnz_counts = Int[]
    for r in 0:(nranks-1)
        if recv_row_counts[r+1] > 0 && r != rank
            push!(recv_rank_ids, r)
            buf = struct_recv_bufs[r]

            nrows = buf[1]
            colptr = buf[2:nrows+2]
            nnz = buf[nrows+3]
            rowval_globals = buf[nrows+4:end]

            push!(recv_nnz_counts, nnz)

            # Compute global row range for this rank's rows
            r_src_start = A.row_partition[r+1]

            for i in 1:nrows
                # Global row this corresponds to
                # The rows from rank r in order correspond to the overlap
                r_dst_start_global = max(r_src_start, dst_start)
                global_row = r_dst_start_global + (i - 1)
                dst_local_row = global_row - dst_start + 1

                row_ptr_start = colptr[i]
                row_ptr_end = colptr[i+1] - 1
                row_rowvals = rowval_globals[row_ptr_start:row_ptr_end]

                push!(row_data, (dst_local_row, [1, 1 + length(row_rowvals)], row_rowvals))
            end
        end
    end

    # Sort rows by destination local row
    sort!(row_data, by=x -> x[1])

    # Collect all global column indices
    all_global_cols = Set{Int}()
    for (_, _, rowval_globals) in row_data
        for gc in rowval_globals
            push!(all_global_cols, gc)
        end
    end
    result_col_indices = sort(collect(all_global_cols))

    # Build compress_map: global_col -> local_col
    compress_map = zeros(Int, isempty(result_col_indices) ? 0 : maximum(result_col_indices))
    for (local_idx, global_idx) in enumerate(result_col_indices)
        compress_map[global_idx] = local_idx
    end

    # Build result_AT structure
    result_colptr = Vector{Int}(undef, result_local_nrows + 1)
    result_colptr[1] = 1
    result_rowval = Int[]
    result_nzval = Vector{T}()  # Will be filled during execute_plan!

    # Track value offsets for each data source
    local_value_offset = 0
    recv_value_offsets = Int[]
    current_offset = 1

    # First, local data
    for (dst_local_row, colptr_slice, rowval_globals) in row_data
        # Check if this is from local (we'll handle offset tracking separately)
    end

    # Build colptr and rowval from sorted row_data
    current_row = 1
    for (dst_local_row, colptr_slice, rowval_globals) in row_data
        # Fill any empty rows before this one
        while current_row < dst_local_row
            result_colptr[current_row + 1] = result_colptr[current_row]
            current_row += 1
        end

        row_nnz = length(rowval_globals)
        result_colptr[dst_local_row + 1] = result_colptr[dst_local_row] + row_nnz

        # Convert global to local column indices
        for gc in rowval_globals
            push!(result_rowval, compress_map[gc])
        end
        current_row = dst_local_row + 1
    end

    # Fill remaining empty rows
    while current_row <= result_local_nrows
        result_colptr[current_row + 1] = result_colptr[current_row]
        current_row += 1
    end

    total_nnz = result_colptr[end] - 1
    result_nzval = zeros(T, total_nnz)

    result_AT = SparseMatrixCSC(
        length(result_col_indices),
        result_local_nrows,
        result_colptr,
        result_rowval,
        result_nzval
    )

    # ========== PHASE 4: Compute value offsets ==========

    # We need to track where each source's values go in result_nzval
    # Rebuild the offset tracking

    # Track by source: local rows first, then received in recv_rank_ids order
    # Local rows
    if !isempty(local_src_row_range)
        # Find where local rows' values go in result_nzval
        local_value_offset = 0
        for local_row in local_src_row_range
            global_row = src_start + local_row - 1
            dst_local_row = global_row - dst_start + 1
            local_value_offset = result_colptr[dst_local_row]
            break  # First local row gives the offset
        end
    end

    # Received rows - compute offset for each recv rank
    for r in recv_rank_ids
        buf = struct_recv_bufs[r]
        nrows = buf[1]
        r_src_start = A.row_partition[r+1]
        r_dst_start_global = max(r_src_start, dst_start)
        first_global_row = r_dst_start_global
        first_dst_local_row = first_global_row - dst_start + 1
        push!(recv_value_offsets, result_colptr[first_dst_local_row])
    end

    # ========== PHASE 5: Compute structural hash ==========

    result_structural_hash = compute_structural_hash(p, result_col_indices, result_AT, comm)

    # ========== PHASE 6: Build send arrays and pre-allocate buffers ==========

    send_rank_ids = Int[]
    send_local_row_ranges = UnitRange{Int}[]
    send_nnz_counts_arr = Int[]

    for r in 0:(nranks-1)
        if haskey(send_row_ranges_map, r) && r != rank
            push!(send_rank_ids, r)
            push!(send_local_row_ranges, send_row_ranges_map[r])

            # Count nnz for this range
            row_range = send_row_ranges_map[r]
            nnz = 0
            for local_row in row_range
                nnz += AT.colptr[local_row+1] - AT.colptr[local_row]
            end
            push!(send_nnz_counts_arr, nnz)
        end
    end

    send_bufs = [Vector{T}(undef, c) for c in send_nnz_counts_arr]
    recv_bufs = [Vector{T}(undef, c) for c in recv_nnz_counts]
    send_reqs = Vector{MPI.Request}(undef, length(send_rank_ids))
    recv_reqs = Vector{MPI.Request}(undef, length(recv_rank_ids))

    return SparseRepartitionPlan{T,Ti}(
        send_rank_ids, send_local_row_ranges, send_nnz_counts_arr, send_bufs, send_reqs,
        recv_rank_ids, recv_nnz_counts, recv_bufs, recv_reqs, recv_value_offsets,
        local_src_row_range, local_value_offset, local_nnz,
        copy(p), copy(A.col_partition), result_col_indices,
        result_AT, result_structural_hash, compress_map
    )
end

"""
    execute_plan!(plan::SparseRepartitionPlan{T,Ti}, A::SparseMatrixMPI{T,Ti}) where {T,Ti}

Execute a sparse repartition plan to redistribute rows from A to a new partition.
Returns a new SparseMatrixMPI with the target row partition.
"""
function execute_plan!(plan::SparseRepartitionPlan{T,Ti}, A::SparseMatrixMPI{T,Ti}) where {T,Ti}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    src_start = A.row_partition[rank+1]
    dst_start = plan.result_row_partition[rank+1]
    AT = _get_csc(A)

    # Copy result structure (values will be filled in)
    result_nzval = Vector{T}(undef, length(plan.result_AT.nzval))

    # Step 1: Copy local values
    if !isempty(plan.local_src_row_range)
        for local_row in plan.local_src_row_range
            global_row = src_start + local_row - 1
            dst_local_row = global_row - dst_start + 1

            src_ptr_start = AT.colptr[local_row]
            src_ptr_end = AT.colptr[local_row+1] - 1
            dst_ptr_start = plan.result_AT.colptr[dst_local_row]

            for (i, src_ptr) in enumerate(src_ptr_start:src_ptr_end)
                result_nzval[dst_ptr_start + i - 1] = AT.nzval[src_ptr]
            end
        end
    end

    # Step 2: Fill send buffers and send
    @inbounds for i in eachindex(plan.send_rank_ids)
        row_range = plan.send_local_row_ranges[i]
        buf = plan.send_bufs[i]
        buf_idx = 1
        for local_row in row_range
            for ptr in AT.colptr[local_row]:(AT.colptr[local_row+1]-1)
                buf[buf_idx] = AT.nzval[ptr]
                buf_idx += 1
            end
        end
        plan.send_reqs[i] = MPI.Isend(buf, comm; dest=plan.send_rank_ids[i], tag=96)
    end

    # Step 3: Post receives
    @inbounds for i in eachindex(plan.recv_rank_ids)
        plan.recv_reqs[i] = MPI.Irecv!(plan.recv_bufs[i], comm; source=plan.recv_rank_ids[i], tag=96)
    end

    MPI.Waitall(plan.recv_reqs)

    # Step 4: Copy received values into result
    @inbounds for i in eachindex(plan.recv_rank_ids)
        offset = plan.recv_value_offsets[i]
        buf = plan.recv_bufs[i]
        for k in eachindex(buf)
            result_nzval[offset + k - 1] = buf[k]
        end
    end

    MPI.Waitall(plan.send_reqs)

    # Extract CSR components from plan's result_AT structure
    nrows_local = plan.result_AT.n
    ncols_compressed = length(plan.result_col_indices)

    # Copy structure arrays and use them for both CPU and target (since this is CPU-only)
    new_rowptr = copy(plan.result_AT.colptr)
    new_colval = copy(plan.result_AT.rowval)
    return SparseMatrixMPI{T,Ti,Vector{T}}(
        plan.result_structural_hash,
        plan.result_row_partition,
        plan.result_col_partition,
        plan.result_col_indices,
        new_rowptr,
        new_colval,
        result_nzval,
        nrows_local,
        ncols_compressed,
        nothing,  # cached_transpose
        nothing,  # cached_symmetric
        new_rowptr,  # rowptr_target (same as rowptr for CPU)
        new_colval   # colval_target (same as colval for CPU)
    )
end

"""
    get_repartition_plan(A::SparseMatrixMPI{T,Ti}, p::Vector{Int}) where {T,Ti}

Get a memoized SparseRepartitionPlan for repartitioning `A` to row partition `p`.
The plan is cached based on the structural hash of A and the target partition hash.
"""
function get_repartition_plan(A::SparseMatrixMPI{T,Ti}, p::Vector{Int}) where {T,Ti}
    target_hash = compute_partition_hash(p)
    key = (_ensure_hash(A), target_hash, T, Ti)
    if haskey(_repartition_plan_cache, key)
        return _repartition_plan_cache[key]::SparseRepartitionPlan{T,Ti}
    end
    plan = SparseRepartitionPlan(A, p)
    _repartition_plan_cache[key] = plan
    return plan
end

"""
    repartition(A::SparseMatrixMPI{T}, p::Vector{Int}) where T

Redistribute a SparseMatrixMPI to a new row partition `p`.
The col_partition remains unchanged.

The partition `p` must be a valid partition vector of length `nranks + 1` with
`p[1] == 1` and `p[end] == size(A, 1) + 1`.

Returns a new SparseMatrixMPI with the same data but `row_partition == p`.

# Example
```julia
A = SparseMatrixMPI{Float64}(sprand(6, 4, 0.5))  # uniform partition
new_partition = [1, 2, 4, 5, 7]  # 1, 2, 1, 2 rows per rank
A_repart = repartition(A, new_partition)
```
"""
function repartition(A::SparseMatrixMPI{T}, p::Vector{Int}) where T
    # Fast path: partition unchanged
    if A.row_partition == p
        return A
    end

    plan = get_repartition_plan(A, p)
    return execute_plan!(plan, A)
end
