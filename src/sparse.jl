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
    compute_structural_hash(row_partition, col_indices, AT, comm) -> Blake3Hash

Compute a structural hash that is identical across all ranks.

1. Hash local data: row_partition, col_indices, AT.colptr, AT.rowval
2. Allgather all local hashes
3. Hash the gathered hashes to produce a global hash
"""
function compute_structural_hash(row_partition::Vector{Int}, col_indices::Vector{Int},
    AT::SparseMatrixCSC, comm::MPI.Comm)::Blake3Hash
    # Step 1: Compute rank-local hash
    # IMPORTANT: Prefix each vector with its length to disambiguate boundaries.
    # Without length prefixes, different structures could hash to the same value
    # if the concatenated bytes happen to match.
    ctx = Blake3Ctx()
    update!(ctx, reinterpret(UInt8, Int[length(row_partition)]))
    update!(ctx, reinterpret(UInt8, row_partition))
    update!(ctx, reinterpret(UInt8, Int[length(col_indices)]))
    update!(ctx, reinterpret(UInt8, col_indices))
    update!(ctx, reinterpret(UInt8, Int[length(AT.colptr)]))
    update!(ctx, reinterpret(UInt8, AT.colptr))
    update!(ctx, reinterpret(UInt8, Int[length(AT.rowval)]))
    update!(ctx, reinterpret(UInt8, AT.rowval))
    local_hash = digest(ctx)

    # Step 2: Allgather all local hashes
    all_hashes = MPI.Allgather(local_hash, comm)

    # Step 3: Hash them together to produce global hash
    ctx2 = Blake3Ctx()
    update!(ctx2, all_hashes)
    return Blake3Hash(digest(ctx2))
end

"""
    compress_AT(AT::SparseMatrixCSC{T,Int}, col_indices::Vector{Int}) where T

Compress AT from global column indices to local indices 1:length(col_indices).
Returns a new SparseMatrixCSC with m = length(col_indices).

The col_indices array provides the local→global mapping: col_indices[local_idx] = global_col.
"""
function compress_AT(AT::SparseMatrixCSC{T,Int}, col_indices::Vector{Int}) where T
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
function reindex_to_union_cached(AT::SparseMatrixCSC{T,Int}, col_to_union_map::Vector{Int}, union_size::Int) where T
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
function compress_AT_cached(AT::SparseMatrixCSC{T,Int}, compress_map::Vector{Int}, local_size::Int) where T
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
function _rebuild_AT_with_insertions(AT::SparseMatrixCSC{T,Int}, col_indices::Vector{Int},
    insertions::Vector{Tuple{Int,Int,T}},
    row_offset::Int) where T
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
    SparseMatrixMPI{T}

A distributed sparse matrix partitioned by rows across MPI ranks.

# Fields
- `structural_hash::Blake3Hash`: 256-bit Blake3 hash of the structural pattern
- `row_partition::Vector{Int}`: Row partition boundaries, length = nranks + 1
- `col_partition::Vector{Int}`: Column partition boundaries, length = nranks + 1 (placeholder for transpose)
- `col_indices::Vector{Int}`: Global column indices that appear in the local part (local→global mapping)
- `A::SparseMatrixCSR{T,Int}`: Local rows in CSR format for efficient row-wise iteration
- `cached_transpose`: Cached materialized transpose (bidirectionally linked)

# Invariants
- `col_indices`, `row_partition`, and `col_partition` are sorted
- `row_partition[nranks+1]` = total number of rows
- `col_partition[nranks+1]` = total number of columns
- `size(A, 1) == row_partition[rank+1] - row_partition[rank]` (number of local rows)
- `size(A.parent, 1) == length(col_indices)` (compressed column dimension)
- `A.parent.rowval` contains local indices in `1:length(col_indices)`

# Storage Details
The local rows are stored in CSR format (Compressed Sparse Row), which enables efficient
row-wise iteration - essential for a row-partitioned distributed matrix.

In Julia, `SparseMatrixCSR{T,Ti}` is a type alias for `Transpose{T, SparseMatrixCSC{T,Ti}}`.
This type has a dual interpretation:
- **Semantic view**: A lazy transpose of a CSC matrix
- **Storage view**: Row-major (CSR) access to the data

The underlying `A.parent::SparseMatrixCSC` stores the transposed data with:
- `A.parent.m = length(col_indices)` (compressed, not global ncols)
- `A.parent.n` = number of local rows (columns in the parent = rows in CSR)
- `A.parent.colptr` = row pointers for the CSR format
- `A.parent.rowval` = LOCAL column indices (1:length(col_indices))
- `col_indices[local_idx]` maps local→global column indices

This compression avoids "hypersparse" storage where the column dimension would be
the global number of columns even if only a few columns have nonzeros locally.

Access the underlying CSC storage via `A.parent` when needed for low-level operations.
"""
mutable struct SparseMatrixMPI{T}
    structural_hash::OptionalBlake3Hash
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    col_indices::Vector{Int}
    A::SparseMatrixCSR{T,Int}  # Local rows in CSR format (row-major storage)
    cached_transpose::Union{Nothing,SparseMatrixMPI{T}}
end

"""
    SparseMatrixMPI{T}(A::SparseMatrixCSC{T,Int}; comm=MPI.COMM_WORLD, row_partition=..., col_partition=...) where T

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
function SparseMatrixMPI{T}(A::SparseMatrixCSC{T,Int};
    comm::MPI.Comm=MPI.COMM_WORLD,
    row_partition::Vector{Int}=uniform_partition(size(A, 1), MPI.Comm_size(comm)),
    col_partition::Vector{Int}=uniform_partition(size(A, 2), MPI.Comm_size(comm))) where T
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
    SparseMatrixMPI_local(A_local::SparseMatrixCSR{T,Int}; comm=MPI.COMM_WORLD, col_partition=...) where T
    SparseMatrixMPI_local(A_local::Adjoint{T,SparseMatrixCSC{T,Int}}; comm=MPI.COMM_WORLD, col_partition=...) where T

Create a SparseMatrixMPI from a local sparse matrix on each rank.

Unlike `SparseMatrixMPI{T}(A_global)` which takes a global matrix and partitions it,
this constructor takes only the local rows of the matrix that each rank owns.
The row partition is computed by gathering the local row counts from all ranks.

The input `A_local` must be a `SparseMatrixCSR{T,Int}` (or `Adjoint` of `SparseMatrixCSC{T,Int}`) where:
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
function SparseMatrixMPI_local(A_local::SparseMatrixCSR{T,Int};
    comm::MPI.Comm=MPI.COMM_WORLD,
    col_partition::Vector{Int}=uniform_partition(A_local.parent.m, MPI.Comm_size(comm))) where T
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
    A_compressed = transpose(compressed_AT)  # Transpose wrapper for type clarity

    # Structural hash computed lazily on first use via _ensure_hash
    return SparseMatrixMPI{T}(nothing, row_partition, col_partition, col_indices, A_compressed, nothing)
end

# Adjoint version: conjugate values during construction
function SparseMatrixMPI_local(A_local::Adjoint{T,SparseMatrixCSC{T,Int}};
    comm::MPI.Comm=MPI.COMM_WORLD,
    col_partition::Vector{Int}=uniform_partition(A_local.parent.m, MPI.Comm_size(comm))) where T
    # Convert adjoint to transpose with conjugated values
    AT_parent = A_local.parent
    AT_conj = SparseMatrixCSC(AT_parent.m, AT_parent.n, copy(AT_parent.colptr),
        copy(AT_parent.rowval), conj.(AT_parent.nzval))
    return SparseMatrixMPI_local(transpose(AT_conj); comm=comm, col_partition=col_partition)
end

"""
    MatrixPlan{T}

A communication plan for gathering rows from an SparseMatrixMPI.

# Fields
- `rank_ids::Vector{Int}`: Ranks that requested data from us (0-indexed)
- `send_ranges::Vector{Vector{UnitRange{Int}}}`: For each rank, ranges into B.A.parent.nzval to send
- `send_bufs::Vector{Vector{T}}`: Pre-allocated send buffers for each rank
- `send_reqs::Vector{MPI.Request}`: Pre-allocated send request handles
- `recv_rank_ids::Vector{Int}`: Ranks we need to receive data from (0-indexed)
- `recv_bufs::Vector{Vector{T}}`: Pre-allocated receive buffers for each rank
- `recv_reqs::Vector{MPI.Request}`: Pre-allocated receive request handles
- `recv_offsets::Vector{Int}`: Starting offsets into AT.nzval for each recv_rank_ids
- `local_ranges::Vector{Tuple{UnitRange{Int}, Int}}`: (src_range, dst_offset) for local copies
- `AT::SparseMatrixCSC{T,Int}`: Transposed matrix structure for gathered rows (values zeroed)
"""
mutable struct MatrixPlan{T}
    rank_ids::Vector{Int}
    send_ranges::Vector{Vector{UnitRange{Int}}}
    send_bufs::Vector{Vector{T}}
    send_reqs::Vector{MPI.Request}
    recv_rank_ids::Vector{Int}
    recv_bufs::Vector{Vector{T}}
    recv_reqs::Vector{MPI.Request}
    recv_offsets::Vector{Int}
    local_ranges::Vector{Tuple{UnitRange{Int},Int}}
    AT::SparseMatrixCSC{T,Int}
    # Cached hash for product result (computed lazily on first execution)
    product_structural_hash::OptionalBlake3Hash
    product_col_indices::Union{Nothing, Vector{Int}}
    product_row_partition::Union{Nothing, Vector{Int}}
    product_compress_map::Union{Nothing, Vector{Int}}  # global_col -> local_col mapping for compress_AT
end

"""
    MatrixPlan(row_indices::Vector{Int}, B::SparseMatrixMPI{T}) where T

Create a communication plan to gather rows specified by row_indices from B.
Assumes row_indices is sorted.

The plan proceeds in 3 steps:
1. For each row i in row_indices, determine owner. If remote, use isend to request structure.
2. Receive requests from other ranks, add to rank_ids, isend structure responses.
3. Receive structure info, build plan.AT with zeros (sparsity pattern of B[row_indices,:]).
"""
function MatrixPlan(row_indices::Vector{Int}, B::SparseMatrixMPI{T}) where T
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
            start_idx = B.A.parent.colptr[col]
            end_idx = B.A.parent.colptr[col+1] - 1
            col_nnz = end_idx - start_idx + 1
            new_colptr[i+1] = new_colptr[i] + col_nnz
            # B.A.parent.rowval contains LOCAL indices, convert to global using B.col_indices
            for idx in start_idx:end_idx
                push!(rowvals, B.col_indices[B.A.parent.rowval[idx]])
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
            total_nnz += B.A.parent.colptr[local_col+1] - B.A.parent.colptr[local_col]
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
            start_idx = B.A.parent.colptr[local_col]
            end_idx = B.A.parent.colptr[local_col+1] - 1
            col_nnz = end_idx - start_idx + 1

            combined_colptr[out_col+1] = combined_colptr[out_col] + col_nnz
            # B.A.parent.rowval contains LOCAL indices, convert to global using B.col_indices
            for (i, idx) in enumerate(start_idx:end_idx)
                combined_rowval[nnz_idx+i-1] = B.col_indices[B.A.parent.rowval[idx]]
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

    # Prepare send info: for each rank in rank_ids, compute ranges into B.A.parent.nzval
    send_ranges_vec = Vector{Vector{UnitRange{Int}}}()
    send_bufs = Vector{Vector{T}}()

    for r in rank_ids
        requested = rows_requested_by[r]
        ranges = UnitRange{Int}[]
        total_len = 0
        for row in requested
            local_col = row - my_row_start + 1
            start_idx = B.A.parent.colptr[local_col]
            end_idx = B.A.parent.colptr[local_col+1] - 1
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

    return MatrixPlan{T}(
        rank_ids, send_ranges_vec, send_bufs, send_reqs,
        recv_rank_ids, recv_bufs, recv_reqs, recv_offsets_vec,
        local_ranges,
        plan_AT,
        nothing, nothing, nothing, nothing  # product: hash, col_indices, row_partition, compress_map
    )
end

"""
    MatrixPlan(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T

Create a memoized communication plan for A * B.
The plan is cached based on the structural hashes of A and B.
"""
function MatrixPlan(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
    key = (_ensure_hash(A), _ensure_hash(B), T)
    if haskey(_plan_cache, key)
        return _plan_cache[key]::MatrixPlan{T}
    end
    plan = MatrixPlan(A.col_indices, B)
    _plan_cache[key] = plan
    return plan
end

"""
    execute_plan!(plan::MatrixPlan{T}, B::SparseMatrixMPI{T}) where T

Execute a communication plan to gather rows from B into plan.AT.
After execution, plan.AT contains the values from B for the requested rows.
This function is allocation-free (all buffers are pre-allocated in the plan).
"""
function execute_plan!(plan::MatrixPlan{T}, B::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD

    # Step 1: Copy local values into plan.AT
    for (src_range, dst_off) in plan.local_ranges
        plan.AT.nzval[dst_off:dst_off+length(src_range)-1] = view(B.A.parent.nzval, src_range)
    end

    # Step 2: Fill send buffers and send to ranks that requested from us
    for (i, r) in enumerate(plan.rank_ids)
        buf = plan.send_bufs[i]
        offset = 1
        for rng in plan.send_ranges[i]
            n = length(rng)
            buf[offset:offset+n-1] = view(B.A.parent.nzval, rng)
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
    ⊛(A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti}; max_threads=Threads.nthreads()) where {Tv,Ti}

Multithreaded sparse matrix multiplication. Splits B into column blocks
and computes `A * B_block` in parallel using Julia's optimized builtin `*`.

# Threading behavior
- Uses at most `n ÷ 100` threads, where `n = size(B, 2)`, ensuring at least 100 columns per thread
- Falls back to single-threaded `A * B` when `n < 100` or when threading overhead would dominate
- The `max_threads` keyword limits the maximum number of threads used

# Examples
```julia
using SparseArrays
A = sprand(1000, 1000, 0.01)
B = sprand(1000, 500, 0.01)
C = A ⊛ B                    # Use all available threads (up to n÷100)
C = ⊛(A, B; max_threads=2)   # Limit to 2 threads
```
"""
function ⊛(A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti}; max_threads::Int=Threads.nthreads()) where {Tv,Ti}
    n = size(B, 2)
    nthreads = min(max_threads, n ÷ 100)
    if nthreads <= 1
        return A * B
    end

    # Column range for thread t: col_start(t):col_end(t)
    cols_per_thread, remainder = divrem(n, nthreads)
    @inline col_start(t) = 1 + (t - 1) * cols_per_thread + min(t - 1, remainder)
    @inline col_end(t) = t * cols_per_thread + min(t, remainder)

    # Storage for block results
    results = Vector{SparseMatrixCSC{Tv,Ti}}(undef, nthreads)

    # Compute A * B_block in parallel using builtin *
    Threads.@threads :static for t in 1:nthreads
        B_block = B[:, col_start(t):col_end(t)]
        results[t] = A * B_block
    end

    # Concatenate results horizontally
    return hcat(results...)
end

"""
    Base.*(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T

Multiply two distributed sparse matrices A * B.
"""
function Base.:*(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD

    # Get memoized communication plan and execute it
    plan = MatrixPlan(A, B)
    execute_plan!(plan, B)

    # Compute local product using: C^T = B^T * A^T = plan.AT * A.A.parent
    #
    # A.A.parent has shape (n_gathered, local_nrows_A) with local column indices (compressed)
    # plan.AT has shape (ncols_B, n_gathered) where n_gathered = length(A.col_indices)
    #
    # A.A.parent is already compressed with local indices 1:length(A.col_indices),
    # so we can use it directly without reindexing.

    # C^T = B^T * A^T = (plan.AT) * (A.A.parent)
    # plan.AT is (ncols_B, n_gathered), A.A.parent is (n_gathered, local_nrows_A)
    # result is (ncols_B, local_nrows_A) = shape of C.AT
    result_AT = plan.AT ⊛ A.A.parent

    # Use cached col_indices and compress_map if available, otherwise compute and cache
    if plan.product_col_indices !== nothing
        result_col_indices = plan.product_col_indices
        compress_map = plan.product_compress_map
    else
        result_col_indices = isempty(result_AT.rowval) ? Int[] : unique(sort(result_AT.rowval))

        # Build compress_map: compress_map[global_col] = local_col
        if isempty(result_col_indices)
            compress_map = Int[]
        else
            max_col = maximum(result_col_indices)
            compress_map = zeros(Int, max_col)
            for (local_idx, global_idx) in enumerate(result_col_indices)
                compress_map[global_idx] = local_idx
            end
        end

        # Cache for future use with same structural pattern
        plan.product_col_indices = result_col_indices
        plan.product_compress_map = compress_map
        plan.product_row_partition = A.row_partition
    end

    compressed_result_AT = compress_AT_cached(result_AT, compress_map, length(result_col_indices))

    # Use cached structural hash if available, otherwise compute and cache
    # This is important for chained operations like (P' * A) * P where the intermediate
    # result needs a hash for the next multiply's plan lookup
    result_hash = plan.product_structural_hash
    if result_hash === nothing
        # Compute hash for the result structure
        result_hash = compute_structural_hash(A.row_partition, result_col_indices, compressed_result_AT, MPI.COMM_WORLD)
        plan.product_structural_hash = result_hash
    end

    # C = A * B has rows from A and columns from B
    return SparseMatrixMPI{T}(result_hash, A.row_partition, B.col_partition, result_col_indices, transpose(compressed_result_AT),
        nothing)
end

"""
    Base.+(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T

Add two distributed sparse matrices. The result has A's row partition.
"""
function Base.:+(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
    B_repart = repartition(B, A.row_partition)

    if A.col_indices == B_repart.col_indices
        return SparseMatrixMPI{T}(nothing, A.row_partition, A.col_partition,
            A.col_indices, transpose(A.A.parent + B_repart.A.parent), nothing)
    end

    # Merge sorted col_indices and build mappings via linear scan (no Dict)
    union_cols = merge_sorted_unique(A.col_indices, B_repart.col_indices)
    A_map = build_subset_mapping(A.col_indices, union_cols)
    B_map = build_subset_mapping(B_repart.col_indices, union_cols)

    # Add in union space
    result = reindex_to_union_cached(A.A.parent, A_map, length(union_cols)) +
             reindex_to_union_cached(B_repart.A.parent, B_map, length(union_cols))

    return SparseMatrixMPI{T}(nothing, A.row_partition, A.col_partition,
        union_cols, transpose(result), nothing)
end

"""
    Base.-(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T

Subtract two distributed sparse matrices. The result has A's row partition.
"""
function Base.:-(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
    B_repart = repartition(B, A.row_partition)

    if A.col_indices == B_repart.col_indices
        return SparseMatrixMPI{T}(nothing, A.row_partition, A.col_partition,
            A.col_indices, transpose(A.A.parent - B_repart.A.parent), nothing)
    end

    # Merge sorted col_indices and build mappings via linear scan (no Dict)
    union_cols = merge_sorted_unique(A.col_indices, B_repart.col_indices)
    A_map = build_subset_mapping(A.col_indices, union_cols)
    B_map = build_subset_mapping(B_repart.col_indices, union_cols)

    # Subtract in union space
    result = reindex_to_union_cached(A.A.parent, A_map, length(union_cols)) -
             reindex_to_union_cached(B_repart.A.parent, B_map, length(union_cols))

    return SparseMatrixMPI{T}(nothing, A.row_partition, A.col_partition,
        union_cols, transpose(result), nothing)
end

"""
    TransposePlan{T}

A communication plan for computing the transpose of an SparseMatrixMPI.

The transpose of A (with row_partition R and col_partition C) will have:
- row_partition = C (columns of A become rows of A^T)
- col_partition = R (rows of A become columns of A^T)

# Fields
- `rank_ids::Vector{Int}`: Ranks we send data to (0-indexed)
- `send_indices::Vector{Vector{Int}}`: For each rank, indices into A.A.parent.nzval to send
- `send_bufs::Vector{Vector{T}}`: Pre-allocated send buffers
- `send_reqs::Vector{MPI.Request}`: Pre-allocated send request handles
- `recv_rank_ids::Vector{Int}`: Ranks we receive data from (0-indexed)
- `recv_bufs::Vector{Vector{T}}`: Pre-allocated receive buffers
- `recv_reqs::Vector{MPI.Request}`: Pre-allocated receive request handles
- `recv_perm::Vector{Vector{Int}}`: For each recv rank, permutation into AT.nzval
- `local_src_indices::Vector{Int}`: Source indices for local copy
- `local_dst_indices::Vector{Int}`: Destination indices for local copy
- `AT::SparseMatrixCSC{T,Int}`: Transposed matrix structure (values zeroed)
- `row_partition::Vector{Int}`: Row partition for the transposed matrix
- `col_partition::Vector{Int}`: Col partition for the transposed matrix
- `col_indices::Vector{Int}`: Column indices for the transposed matrix
"""
mutable struct TransposePlan{T}
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
    AT::SparseMatrixCSC{T,Int}
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    col_indices::Vector{Int}
    # Precomputed compress_map: compress_map[global_col] = local_col
    compress_map::Vector{Int}
    # Cached structural hash for transpose result (computed lazily on first execution)
    structural_hash::OptionalBlake3Hash
end

"""
    TransposePlan(A::SparseMatrixMPI{T}) where T

Create a communication plan for computing A^T.

The algorithm:
1. For each nonzero A[i,j] (stored as A.A.parent[j, local_i]), determine which rank
   owns row j in A^T (using A.col_partition). Package (i,j) pairs by destination rank.
2. Exchange structure via point-to-point communication.
3. Build the transposed sparse structure and communication buffers.
"""
function TransposePlan(A::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    my_row_start = A.row_partition[rank+1]
    nrows_A = A.row_partition[end] - 1

    # The transpose has swapped partitions
    result_row_partition = A.col_partition
    result_col_partition = A.row_partition

    # Step 1: For each nonzero in A.A.parent, determine destination rank for transpose
    # A.A.parent[j, local_col] corresponds to A[global_row, j] where global_row = my_row_start + local_col - 1
    # In A^T, this is A^T[j, global_row], so it goes to the rank owning row j per col_partition

    # Group nonzeros by destination rank: (global_row, j, src_nzval_idx)
    send_to = [Tuple{Int,Int,Int}[] for _ in 1:nranks]

    for local_col in 1:size(A.A.parent, 2)
        global_row = my_row_start + local_col - 1
        for idx in A.A.parent.colptr[local_col]:(A.A.parent.colptr[local_col+1]-1)
            local_j = A.A.parent.rowval[idx]  # LOCAL column index in A (compressed)
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

    return TransposePlan{T}(
        rank_ids, send_indices_final, send_bufs, send_reqs,
        recv_rank_ids, recv_bufs, recv_reqs, recv_perm,
        local_src_indices, local_dst_indices,
        result_AT, result_row_partition, result_col_partition, result_col_indices,
        compress_map,
        nothing  # structural_hash (computed lazily on first execution)
    )
end

"""
    execute_plan!(plan::TransposePlan{T}, A::SparseMatrixMPI{T}) where T

Execute a transpose plan to compute A^T.
Returns an SparseMatrixMPI representing the transpose.

Note: The returned matrix has its own copy of the sparse data, so the plan
can be safely reused for subsequent transposes.
"""
function execute_plan!(plan::TransposePlan{T}, A::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD

    # Step 1: Copy local values (allocation-free loop)
    local_src = plan.local_src_indices
    local_dst = plan.local_dst_indices
    @inbounds for i in eachindex(local_src, local_dst)
        plan.AT.nzval[local_dst[i]] = A.A.parent.nzval[local_src[i]]
    end

    # Step 2: Fill send buffers and send (allocation-free loops)
    @inbounds for i in eachindex(plan.rank_ids)
        r = plan.rank_ids[i]
        send_idx = plan.send_indices[i]
        buf = plan.send_bufs[i]
        for k in eachindex(send_idx)
            buf[k] = A.A.parent.nzval[send_idx[k]]
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

    return SparseMatrixMPI{T}(plan.structural_hash, plan.row_partition, plan.col_partition,
        plan.col_indices, transpose(compressed_result_AT), nothing)
end

"""
    SparseMatrixMPI{T}(At::Transpose{T, SparseMatrixMPI{T}}) where T

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
function SparseMatrixMPI{T}(At::Transpose{T, SparseMatrixMPI{T}}) where T
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
SparseMatrixMPI(At::Transpose{T, SparseMatrixMPI{T}}) where T = SparseMatrixMPI{T}(At)

# VectorPlan constructor for sparse A * x (adds method to VectorPlan from vectors.jl)

"""
    VectorPlan(A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T

Create a communication plan to gather x[A.col_indices] for matrix-vector multiplication.
"""
function VectorPlan(A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T
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

    send_bufs = [Vector{T}(undef, length(inds)) for inds in send_indices_final]
    recv_bufs = [Vector{T}(undef, send_counts[r+1]) for r in recv_rank_ids]
    send_reqs = Vector{MPI.Request}(undef, length(send_rank_ids))
    recv_reqs = Vector{MPI.Request}(undef, length(recv_rank_ids))
    gathered = Vector{T}(undef, n_gathered)

    return VectorPlan{T}(
        send_rank_ids, send_indices_final, send_bufs, send_reqs,
        recv_rank_ids, recv_bufs, recv_reqs, recv_perm_final,
        local_src_indices, local_dst_indices, gathered,
        nothing, nothing  # result_partition_hash, result_partition (computed lazily)
    )
end

"""
    get_vector_plan(A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T

Get a memoized VectorPlan for A * x.
The plan is cached based on the structural hashes of A and x.
"""
function get_vector_plan(A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T
    key = (_ensure_hash(A), x.structural_hash, T)
    if haskey(_vector_plan_cache, key)
        return _vector_plan_cache[key]::VectorPlan{T}
    end
    plan = VectorPlan(A, x)
    _vector_plan_cache[key] = plan
    return plan
end

# Matrix-vector multiplication

"""
    LinearAlgebra.mul!(y::VectorMPI{T}, A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T

In-place sparse matrix-vector multiplication: y = A * x.

The algorithm:
1. Gather x[A.col_indices] from all ranks using VectorPlan
2. Compute y.v = transpose(A.A.parent) * gathered

A.A.parent is already compressed with local indices 1:length(A.col_indices),
so gathered has length matching A.A.parent.m and can be used directly.
"""
function LinearAlgebra.mul!(y::VectorMPI{T}, A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD

    # Get memoized plan and execute it
    plan = get_vector_plan(A, x)
    gathered = execute_plan!(plan, x)

    # A.A.parent is already compressed with local indices 1:length(A.col_indices)
    # gathered has length length(A.col_indices), matching A.A.parent.m
    # y = A * x => y^T = x^T * A^T => y.v = transpose(A.A.parent) * gathered
    # Use transpose() not ' to avoid conjugation for complex types
    LinearAlgebra.mul!(y.v, transpose(A.A.parent), gathered)
    return y
end

"""
    Base.:*(A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T

Sparse matrix-vector multiplication returning a new VectorMPI.
The result has the same row partition as A.
"""
function Base.:*(A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_rows = A.row_partition[rank+2] - A.row_partition[rank+1]

    # Get the plan and use cached partition hash if available
    plan = get_vector_plan(A, x)
    if plan.result_partition_hash === nothing
        plan.result_partition_hash = compute_partition_hash(A.row_partition)
        plan.result_partition = copy(A.row_partition)
    end

    y = VectorMPI{T}(
        plan.result_partition_hash,
        plan.result_partition,  # Partition is immutable, no need to copy
        Vector{T}(undef, local_rows)
    )

    # Execute the plan directly since we already have it
    gathered = execute_plan!(plan, x)
    LinearAlgebra.mul!(y.v, transpose(A.A.parent), gathered)
    return y
end

"""
    *(vt::Transpose{<:Any, VectorMPI{T}}, A::SparseMatrixMPI{T}) where T

Compute transpose(v) * A as transpose(transpose(A) * v).
Returns a transposed VectorMPI.
"""
function Base.:*(vt::Transpose{<:Any,VectorMPI{T}}, A::SparseMatrixMPI{T}) where T
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

Base.eltype(::SparseMatrixMPI{T}) where T = T
Base.eltype(::Type{SparseMatrixMPI{T}}) where T = T

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
    local_vals = A.A.parent.nzval

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
        # A.A.parent is transposed, so columns of A.A.parent are rows of A
        local_nrows = size(A.A.parent, 2)
        local_max = zero(real(T))
        for col in 1:local_nrows
            row_sum = zero(real(T))
            for idx in A.A.parent.colptr[col]:(A.A.parent.colptr[col+1]-1)
                row_sum += abs(A.A.parent.nzval[idx])
            end
            local_max = max(local_max, row_sum)
        end
        return MPI.Allreduce(local_max, MPI.MAX, comm)

    elseif p == 1
        # Maximum absolute column sum
        # Columns are distributed, need to sum contributions from all ranks
        ncols = A.col_partition[end] - 1

        # Compute local contribution to each column sum
        # A.A.parent.rowval contains LOCAL column indices (compressed)
        # Map local→global using A.col_indices
        local_col_sums = zeros(real(T), ncols)
        col_indices = A.col_indices
        for (idx, local_col) in enumerate(A.A.parent.rowval)
            global_col = col_indices[local_col]
            local_col_sums[global_col] += abs(A.A.parent.nzval[idx])
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
    Base.transpose(A::SparseMatrixMPI{T}) where T

Return a lazy transpose wrapper around A.
"""
Base.transpose(A::SparseMatrixMPI{T}) where T = Transpose(A)

"""
    conj(A::SparseMatrixMPI{T}) where T

Return a new SparseMatrixMPI with conjugated values.
"""
function Base.conj(A::SparseMatrixMPI{T}) where T
    conj_AT = SparseMatrixCSC(
        A.A.parent.m, A.A.parent.n,
        A.A.parent.colptr,  # share structure (immutable)
        A.A.parent.rowval,  # share structure (immutable)
        conj.(A.A.parent.nzval)  # conjugate values
    )
    # Structural hash is the same since structure didn't change
    return SparseMatrixMPI{T}(A.structural_hash, A.row_partition, A.col_partition,
        A.col_indices, transpose(conj_AT), nothing)
end

"""
    Base.adjoint(A::SparseMatrixMPI{T}) where T

Return transpose(conj(A)), i.e., the conjugate transpose.
This is implemented as transpose of a conjugated copy.
"""
Base.adjoint(A::SparseMatrixMPI{T}) where T = transpose(conj(A))

# Scalar multiplication

"""
    *(a::Number, A::SparseMatrixMPI{T}) where T

Scalar times matrix.
"""
function Base.:*(a::Number, A::SparseMatrixMPI{T}) where T
    RT = promote_type(typeof(a), T)
    scaled_AT = SparseMatrixCSC(
        A.A.parent.m, A.A.parent.n,
        A.A.parent.colptr,
        A.A.parent.rowval,
        RT.(a .* A.A.parent.nzval)
    )
    return SparseMatrixMPI{RT}(A.structural_hash, A.row_partition, A.col_partition,
        A.col_indices, transpose(scaled_AT), nothing)
end

"""
    *(A::SparseMatrixMPI{T}, a::Number) where T

Matrix times scalar.
"""
Base.:*(A::SparseMatrixMPI{T}, a::Number) where T = a * A

"""
    -(A::SparseMatrixMPI{T}) where T

Unary negation of a sparse matrix.
"""
Base.:-(A::SparseMatrixMPI{T}) where T = (-1) * A

# Type alias for transpose of SparseMatrixMPI
const TransposedSparseMatrixMPI{T} = Transpose{T,SparseMatrixMPI{T}}

"""
    *(a::Number, At::TransposedSparseMatrixMPI{T}) where T

Scalar times transposed matrix: a * transpose(A) = transpose(a * A).
"""
Base.:*(a::Number, At::TransposedSparseMatrixMPI{T}) where T = transpose(a * At.parent)

"""
    *(At::TransposedSparseMatrixMPI{T}, a::Number) where T

Transposed matrix times scalar: transpose(A) * a = transpose(a * A).
"""
Base.:*(At::TransposedSparseMatrixMPI{T}, a::Number) where T = transpose(a * At.parent)

# Lazy transpose multiplication methods

"""
    *(At::Transpose, Bt::Transpose)

Compute transpose(A) * transpose(B) = transpose(B.parent * A.parent) lazily.
Returns a Transpose wrapper around the product B.parent * A.parent.
"""
function Base.:*(At::TransposedSparseMatrixMPI{T}, Bt::TransposedSparseMatrixMPI{T}) where T
    A = At.parent
    B = Bt.parent
    return transpose(B * A)
end

"""
    *(At::Transpose, B::SparseMatrixMPI)

Compute transpose(A) * B by materializing the transpose of A first.
"""
function Base.:*(At::TransposedSparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
    A = At.parent
    A_transposed = SparseMatrixMPI(transpose(A))
    return A_transposed * B
end

"""
    *(A::SparseMatrixMPI, Bt::Transpose)

Compute A * transpose(B) by materializing the transpose of B first.
"""
function Base.:*(A::SparseMatrixMPI{T}, Bt::TransposedSparseMatrixMPI{T}) where T
    B = Bt.parent
    B_transposed = SparseMatrixMPI(transpose(B))
    return A * B_transposed
end

"""
    Base.:*(At::TransposedSparseMatrixMPI{T}, x::VectorMPI{T}) where T

Compute transpose(A) * x by materializing the transpose of A first.
"""
function Base.:*(At::TransposedSparseMatrixMPI{T}, x::VectorMPI{T}) where T
    A = At.parent
    A_transposed = SparseMatrixMPI(transpose(A))
    return A_transposed * x
end

# ============================================================================
# Mixed Sparse-Dense Operations
# ============================================================================

"""
    Base.:*(A::SparseMatrixMPI{T}, B::MatrixMPI{T}) where T

Compute sparse matrix times dense matrix by column-by-column multiplication.
Returns a MatrixMPI with the same row partition as A.
"""
function Base.:*(A::SparseMatrixMPI{T}, B::MatrixMPI{T}) where T
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

    # Build local matrix from column results
    local_result = Matrix{T}(undef, local_m, n)
    for k in 1:n
        local_result[:, k] = columns[k].v
    end

    return MatrixMPI_local(local_result)
end

"""
    Base.:*(At::TransposedSparseMatrixMPI{T}, B::MatrixMPI{T}) where T

Compute transpose(A) * B by materializing the transpose of A first.
"""
function Base.:*(At::TransposedSparseMatrixMPI{T}, B::MatrixMPI{T}) where T
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
    local_nnz = length(A.A.parent.nzval)
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
    Base.copy(A::SparseMatrixMPI{T}) where T

Create a deep copy of the distributed sparse matrix.
"""
function Base.copy(A::SparseMatrixMPI{T}) where T
    new_AT = SparseMatrixCSC(
        A.A.parent.m, A.A.parent.n,
        copy(A.A.parent.colptr),
        copy(A.A.parent.rowval),
        copy(A.A.parent.nzval)
    )
    return SparseMatrixMPI{T}(
        A.structural_hash,
        copy(A.row_partition),
        copy(A.col_partition),
        copy(A.col_indices),
        transpose(new_AT),
        nothing
    )
end

# ============================================================================
# Extended SparseMatrixCSC API - Element-wise Operations
# ============================================================================

# Helper function for zero-preserving element-wise operations
function _map_nzval(f, A::SparseMatrixMPI{T}) where T
    new_nzval = f.(A.A.parent.nzval)
    RT = eltype(new_nzval)
    new_AT = SparseMatrixCSC(A.A.parent.m, A.A.parent.n, A.A.parent.colptr, A.A.parent.rowval, new_nzval)
    return SparseMatrixMPI{RT}(A.structural_hash, A.row_partition, A.col_partition,
        A.col_indices, transpose(new_AT), nothing)
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
function Base.sum(A::SparseMatrixMPI{T}; dims=nothing) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    if dims === nothing
        # Sum all elements
        local_sum = sum(A.A.parent.nzval; init=zero(T))
        return MPI.Allreduce(local_sum, MPI.SUM, comm)
    elseif dims == 1
        # Sum over rows: result is length-n vector (column sums)
        # A.A.parent.rowval contains LOCAL column indices (compressed)
        # Map local→global using A.col_indices
        local_col_sums = zeros(T, n)
        col_indices = A.col_indices
        for (idx, local_col) in enumerate(A.A.parent.rowval)
            global_col = col_indices[local_col]
            local_col_sums[global_col] += A.A.parent.nzval[idx]
        end
        global_col_sums = MPI.Allreduce(local_col_sums, MPI.SUM, comm)
        return VectorMPI(global_col_sums; comm=comm)
    elseif dims == 2
        # Sum over columns: result is length-m vector (row sums)
        local_nrows = size(A.A.parent, 2)
        local_row_sums = zeros(T, local_nrows)

        for local_col in 1:local_nrows
            for nz_idx in A.A.parent.colptr[local_col]:(A.A.parent.colptr[local_col+1]-1)
                local_row_sums[local_col] += A.A.parent.nzval[nz_idx]
            end
        end

        # Result has A's row partition (partition is immutable, no need to copy)
        hash = compute_partition_hash(A.row_partition)
        return VectorMPI{T}(hash, A.row_partition, local_row_sums)
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
    local_max = isempty(A.A.parent.nzval) ? typemin(real(T)) : maximum(real, A.A.parent.nzval)
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
    local_min = isempty(A.A.parent.nzval) ? typemax(real(T)) : minimum(real, A.A.parent.nzval)
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
    return A.A.parent[local_col_idx, local_row]
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
    dropzeros(A::SparseMatrixMPI{T}) where T

Return a copy of A with explicitly stored zeros removed.
"""
function dropzeros(A::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD

    # Use SparseArrays.dropzeros on local AT
    new_AT = dropzeros(A.A.parent)

    # Recompute col_indices since structure may have changed
    new_col_indices = isempty(new_AT.rowval) ? Int[] : unique(sort(new_AT.rowval))

    # Recompute structural hash since structure changed
    structural_hash = compute_structural_hash(A.row_partition, new_col_indices, new_AT, comm)

    return SparseMatrixMPI{T}(structural_hash, copy(A.row_partition), copy(A.col_partition),
        new_col_indices, transpose(new_AT), nothing)
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
function diag(A::SparseMatrixMPI{T}, k::Integer=0) where T
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
        # Empty diagonal
        partition = ones(Int, nranks + 1)
        hash = compute_partition_hash(partition)
        return VectorMPI{T}(hash, partition, T[])
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
    my_row_start = A.row_partition[rank+1]
    local_diag = Vector{T}(undef, my_diag_len)

    for i in 1:my_diag_len
        d = my_diag_start + i - 1
        local_row = (row_offset + d) - my_row_start + 1
        global_col = col_offset + d
        val = _find_nzval_at_global_col(A, local_row, global_col)
        local_diag[i] = val === nothing ? zero(T) : val
    end

    hash = compute_partition_hash(result_partition)
    return VectorMPI{T}(hash, result_partition, local_diag)
end

"""
    triu(A::SparseMatrixMPI{T}, k::Integer=0) where T

Return the upper triangular part of A, starting from the k-th diagonal.
- k=0: include main diagonal
- k>0: exclude k-1 diagonals below the k-th superdiagonal
- k<0: include |k| subdiagonals
"""
function triu(A::SparseMatrixMPI{T}, k::Integer=0) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    my_row_start = A.row_partition[rank+1]
    col_indices = A.col_indices

    # Build new sparse structure keeping only upper triangular entries
    # Entry (i, j) is kept if j >= i + k, i.e., j - i >= k

    new_colptr = Vector{Int}(undef, size(A.A.parent, 2) + 1)
    new_colptr[1] = 1

    # First pass: count entries per column
    nnz_per_col = zeros(Int, size(A.A.parent, 2))
    for local_col in 1:size(A.A.parent, 2)
        global_row = my_row_start + local_col - 1
        for nz_idx in A.A.parent.colptr[local_col]:(A.A.parent.colptr[local_col+1]-1)
            local_j = A.A.parent.rowval[nz_idx]
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
    new_nzval = Vector{T}(undef, total_nnz)

    # Second pass: fill entries (keep local indices in new_rowval)
    idx = 1
    for local_col in 1:size(A.A.parent, 2)
        global_row = my_row_start + local_col - 1
        for nz_idx in A.A.parent.colptr[local_col]:(A.A.parent.colptr[local_col+1]-1)
            local_j = A.A.parent.rowval[nz_idx]
            j = col_indices[local_j]  # convert to global column index
            if j >= global_row + k
                new_rowval[idx] = local_j  # keep local index
                new_nzval[idx] = A.A.parent.nzval[nz_idx]
                idx += 1
            end
        end
    end

    # new_rowval contains local indices from A. Convert to global col_indices and compress.
    if isempty(new_rowval)
        new_col_indices = Int[]
        compressed_AT = SparseMatrixCSC(0, size(A.A.parent, 2), new_colptr, Int[], T[])
    else
        local_used = unique(sort(new_rowval))
        new_col_indices = col_indices[local_used]  # convert to global
        # Compress: local_used is sorted, use binary search instead of Dict
        compressed_rowval = [searchsortedfirst(local_used, r) for r in new_rowval]
        compressed_AT = SparseMatrixCSC(length(new_col_indices), size(A.A.parent, 2),
            new_colptr, compressed_rowval, new_nzval)
    end

    structural_hash = compute_structural_hash(A.row_partition, new_col_indices, compressed_AT, comm)

    return SparseMatrixMPI{T}(structural_hash, copy(A.row_partition), copy(A.col_partition),
        new_col_indices, transpose(compressed_AT), nothing)
end

"""
    tril(A::SparseMatrixMPI{T}, k::Integer=0) where T

Return the lower triangular part of A, starting from the k-th diagonal.
- k=0: include main diagonal
- k>0: include k superdiagonals
- k<0: exclude |k|-1 diagonals above the |k|-th subdiagonal
"""
function tril(A::SparseMatrixMPI{T}, k::Integer=0) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    my_row_start = A.row_partition[rank+1]
    col_indices = A.col_indices

    # Keep entry (i, j) if j <= i + k

    new_colptr = Vector{Int}(undef, size(A.A.parent, 2) + 1)
    new_colptr[1] = 1

    nnz_per_col = zeros(Int, size(A.A.parent, 2))
    for local_col in 1:size(A.A.parent, 2)
        global_row = my_row_start + local_col - 1
        for nz_idx in A.A.parent.colptr[local_col]:(A.A.parent.colptr[local_col+1]-1)
            local_j = A.A.parent.rowval[nz_idx]
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
    new_nzval = Vector{T}(undef, total_nnz)

    idx = 1
    for local_col in 1:size(A.A.parent, 2)
        global_row = my_row_start + local_col - 1
        for nz_idx in A.A.parent.colptr[local_col]:(A.A.parent.colptr[local_col+1]-1)
            local_j = A.A.parent.rowval[nz_idx]
            j = col_indices[local_j]  # convert to global column index
            if j <= global_row + k
                new_rowval[idx] = local_j  # keep local index
                new_nzval[idx] = A.A.parent.nzval[nz_idx]
                idx += 1
            end
        end
    end

    # new_rowval contains local indices from A. Convert to global col_indices and compress.
    if isempty(new_rowval)
        new_col_indices = Int[]
        compressed_AT = SparseMatrixCSC(0, size(A.A.parent, 2), new_colptr, Int[], T[])
    else
        local_used = unique(sort(new_rowval))
        new_col_indices = col_indices[local_used]  # convert to global
        # Compress: local_used is sorted, use binary search instead of Dict
        compressed_rowval = [searchsortedfirst(local_used, r) for r in new_rowval]
        compressed_AT = SparseMatrixCSC(length(new_col_indices), size(A.A.parent, 2),
            new_colptr, compressed_rowval, new_nzval)
    end

    structural_hash = compute_structural_hash(A.row_partition, new_col_indices, compressed_AT, comm)

    return SparseMatrixMPI{T}(structural_hash, copy(A.row_partition), copy(A.col_partition),
        new_col_indices, transpose(compressed_AT), nothing)
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

    for r in 0:(nranks-1)
        if recv_counts[r+1] > 0 && r != rank
            requested_rows = row_recv_bufs[r]
            triplets = Tuple{Int32,Int32,T}[]

            for global_row in requested_rows
                local_row = global_row - my_row_start + 1
                # A.A.parent has columns = local rows, so iterate column `local_row`
                for nz_idx in A.A.parent.colptr[local_row]:(A.A.parent.colptr[local_row+1]-1)
                    local_col = A.A.parent.rowval[nz_idx]
                    global_col = A.col_indices[local_col]
                    val = A.A.parent.nzval[nz_idx]
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
            for nz_idx in A.A.parent.colptr[local_row]:(A.A.parent.colptr[local_row+1]-1)
                local_col = A.A.parent.rowval[nz_idx]
                global_col = A.col_indices[local_col]
                val = A.A.parent.nzval[nz_idx]
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

    # Determine element type
    T = eltype(first(kv)[2])
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
    repartitioned = Dict{Int, VectorMPI}()
    for (k, v) in kv
        target = _diag_target_partition(row_partition, k, length(v))
        repartitioned[k] = repartition(v, target)
    end

    # Step 3: Build local triplets using vectorized operations
    local_I = Int[]
    local_J = Int[]
    local_V = T[]

    for (k, v) in kv
        vec_len = length(v)
        v_repart = repartitioned[k]
        my_v_start = v_repart.partition[rank+1]
        my_v_end = v_repart.partition[rank+2] - 1
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
            append!(local_V, v_repart.v[v_indices])
        end
    end

    # Step 4: Build M^T directly as CSC (swap I↔J), then wrap in lazy transpose for CSR
    AT_local = isempty(local_I) ?
        SparseMatrixCSC(n, local_nrows, ones(Int, local_nrows + 1), Int[], T[]) :
        sparse(local_J, local_I, local_V, n, local_nrows)

    return SparseMatrixMPI_local(transpose(AT_local); comm=comm)
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

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Determine element type
    T = eltype(first(kv)[2])
    for (_, v) in kv
        T = promote_type(T, eltype(v))
    end

    # Step 1: Compute output row partition
    row_partition = uniform_partition(m, nranks)

    my_row_start = row_partition[rank+1]
    my_row_end = row_partition[rank+2] - 1
    local_nrows = my_row_end - my_row_start + 1

    # Step 2: Repartition each vector to match the rows that need it
    repartitioned = Dict{Int, VectorMPI}()
    for (k, v) in kv
        target = _diag_target_partition(row_partition, k, length(v))
        repartitioned[k] = repartition(v, target)
    end

    # Step 3: Build local triplets
    local_I = Int[]
    local_J = Int[]
    local_V = T[]

    for (k, v) in kv
        vec_len = length(v)
        v_repart = repartitioned[k]
        my_v_start = v_repart.partition[rank+1]

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
                push!(local_V, v_repart.v[local_v_idx])
            end
        end
    end

    # Step 4: Build M^T directly as CSC (swap I↔J), then wrap in lazy transpose for CSR
    AT_local = isempty(local_I) ?
        SparseMatrixCSC(n, local_nrows, ones(Int, local_nrows + 1), Int[], T[]) :
        sparse(local_J, local_I, local_V, n, local_nrows)

    return SparseMatrixMPI_local(transpose(AT_local); comm=comm)
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
function spdiagm(v::VectorMPI{T}) where T
    # Ultra-fast path for main diagonal: build CSC structure directly
    n = length(v)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    my_start = v.partition[rank+1]
    local_n = length(v.v)

    # Build AT (transpose) as CSC directly - no sparse() call needed
    # AT has size (n_cols, local_n_rows): n global columns, local_n local rows
    # Column j of AT corresponds to local row j, which has one entry at global column my_start+j-1
    #
    # IMPORTANT: AT.rowval stores LOCAL/compressed column indices (1, 2, 3, ...)
    # col_indices maps these local indices to global column indices
    colptr = collect(1:(local_n+1))  # Each column has exactly 1 entry
    rowval = collect(1:local_n)  # LOCAL column indices (compressed)
    nzval = copy(v.v)

    AT_local = SparseMatrixCSC(n, local_n, colptr, rowval, nzval)

    # col_indices maps local column index -> global column index
    col_indices = collect(my_start:(my_start + local_n - 1))
    row_partition = v.partition  # Use same partition as input vector
    col_partition = v.partition  # Square matrix, same column partition

    return SparseMatrixMPI{T}(nothing, row_partition, col_partition, col_indices,
                               transpose(AT_local), nothing)
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
    Base.:*(A::MatrixMPI{T}, B::SparseMatrixMPI{T}) where T

Compute dense matrix times sparse matrix.
Uses column-by-column approach with transpose(B)' * transpose(A').
"""
function Base.:*(A::MatrixMPI{T}, B::SparseMatrixMPI{T}) where T
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

    # Build local matrix from column results
    local_result = Matrix{T}(undef, local_m, n)
    for k in 1:n
        local_result[:, k] = columns[k].v
    end

    return MatrixMPI_local(local_result)
end

"""
    Base.:*(At::TransposedMatrixMPI{T}, B::SparseMatrixMPI{T}) where T

Compute transpose(A) * B where A is dense and B is sparse.
"""
function Base.:*(At::TransposedMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
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

    # Build local matrix from column results
    local_result = Matrix{T}(undef, local_m, n)
    for k in 1:n
        local_result[:, k] = columns[k].v
    end

    return MatrixMPI_local(local_result)
end

# ============================================================================
# UniformScaling Support
# ============================================================================

"""
    Base.:+(A::SparseMatrixMPI{T}, J::UniformScaling) where T

Add a scalar multiple of the identity matrix to A.
Returns A + λI where J = λI.

The matrix must be square for this operation.
"""
function Base.:+(A::SparseMatrixMPI{T}, J::UniformScaling) where T
    m, n = size(A)
    if m != n
        throw(DimensionMismatch("matrix must be square to add UniformScaling"))
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    λ = J.λ
    RT = promote_type(T, typeof(λ))

    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1
    local_nrows = my_row_end - my_row_start + 1

    # Gather diagonal elements we need to modify for our local rows
    # Each local row i corresponds to global row my_row_start + i - 1
    # The diagonal element for that row is at column my_row_start + i - 1

    # Build COO format for the result
    local_I = Int[]
    local_J = Int[]
    local_V = RT[]

    AT = A.A.parent  # underlying CSC
    col_indices = A.col_indices

    for local_row in 1:local_nrows
        global_row = my_row_start + local_row - 1
        diag_col = global_row  # Diagonal element

        # Iterate over existing entries in this row
        found_diag = false
        for nz_idx in AT.colptr[local_row]:(AT.colptr[local_row+1]-1)
            local_col_idx = AT.rowval[nz_idx]
            global_col = col_indices[local_col_idx]
            val = AT.nzval[nz_idx]

            if global_col == diag_col
                # This is the diagonal - add λ
                push!(local_I, local_row)
                push!(local_J, global_col)
                push!(local_V, RT(val) + RT(λ))
                found_diag = true
            else
                # Non-diagonal - keep as is
                push!(local_I, local_row)
                push!(local_J, global_col)
                push!(local_V, RT(val))
            end
        end

        # If diagonal wasn't in the sparsity pattern, add it
        if !found_diag && diag_col <= n
            push!(local_I, local_row)
            push!(local_J, diag_col)
            push!(local_V, RT(λ))
        end
    end

    # Build M^T directly as CSC (swap I↔J), then wrap in lazy transpose for CSR
    # This avoids an unnecessary physical transpose operation
    AT_local = isempty(local_I) ?
        SparseMatrixCSC(n, local_nrows, ones(Int, local_nrows + 1), Int[], RT[]) :
        sparse(local_J, local_I, local_V, n, local_nrows)

    return SparseMatrixMPI_local(transpose(AT_local); comm=comm)
end

"""
    Base.:-(A::SparseMatrixMPI{T}, J::UniformScaling) where T

Subtract a scalar multiple of the identity matrix from A.
Returns A - λI where J = λI.
"""
function Base.:-(A::SparseMatrixMPI{T}, J::UniformScaling) where T
    return A + UniformScaling(-J.λ)
end

"""
    Base.:+(J::UniformScaling, A::SparseMatrixMPI{T}) where T

Add a sparse matrix to a scalar multiple of the identity.
Returns λI + A where J = λI.
"""
function Base.:+(J::UniformScaling, A::SparseMatrixMPI{T}) where T
    return A + J
end

"""
    Base.:-(J::UniformScaling, A::SparseMatrixMPI{T}) where T

Subtract a sparse matrix from a scalar multiple of the identity.
Returns λI - A where J = λI.
"""
function Base.:-(J::UniformScaling, A::SparseMatrixMPI{T}) where T
    return (-A) + J
end

# ============================================================================
# SparseRepartitionPlan: Repartition a SparseMatrixMPI to a new row partition
# ============================================================================

"""
    SparseRepartitionPlan{T}

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
- `local_src_row_range::UnitRange{Int}`: Local rows (in A.A.parent columns) that stay
- `local_value_offset::Int`: Offset into result_nzval for local values
- `local_nnz::Int`: Number of local nonzeros

## Result metadata (EAGER)
- `result_row_partition::Vector{Int}`: Target row partition
- `result_col_partition::Vector{Int}`: Column partition (unchanged)
- `result_col_indices::Vector{Int}`: Union of col_indices from received rows
- `result_AT::SparseMatrixCSC{T,Int}`: Pre-built sparse structure (values to be filled)
- `result_structural_hash::Blake3Hash`: Pre-computed structural hash
- `compress_map::Vector{Int}`: global_col -> local_col for result
"""
mutable struct SparseRepartitionPlan{T}
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
    result_AT::SparseMatrixCSC{T,Int}
    result_structural_hash::Blake3Hash
    compress_map::Vector{Int}
end

"""
    SparseRepartitionPlan(A::SparseMatrixMPI{T}, p::Vector{Int}) where T

Create a communication plan to repartition `A` to have row partition `p`.
The col_partition remains unchanged.

The plan:
1. Computes row overlaps between source and target partitions
2. Exchanges sparse structure (colptr, rowval) to determine result structure
3. Builds result col_indices, compress_map, and AT structure
4. Computes structural hash eagerly
5. Pre-allocates value buffers
"""
function SparseRepartitionPlan(A::SparseMatrixMPI{T}, p::Vector{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    AT = A.A.parent  # Underlying CSC storage

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

    return SparseRepartitionPlan{T}(
        send_rank_ids, send_local_row_ranges, send_nnz_counts_arr, send_bufs, send_reqs,
        recv_rank_ids, recv_nnz_counts, recv_bufs, recv_reqs, recv_value_offsets,
        local_src_row_range, local_value_offset, local_nnz,
        copy(p), copy(A.col_partition), result_col_indices,
        result_AT, result_structural_hash, compress_map
    )
end

"""
    execute_plan!(plan::SparseRepartitionPlan{T}, A::SparseMatrixMPI{T}) where T

Execute a sparse repartition plan to redistribute rows from A to a new partition.
Returns a new SparseMatrixMPI with the target row partition.
"""
function execute_plan!(plan::SparseRepartitionPlan{T}, A::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    src_start = A.row_partition[rank+1]
    dst_start = plan.result_row_partition[rank+1]
    AT = A.A.parent

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

    # Build result AT with filled values
    result_AT = SparseMatrixCSC(
        plan.result_AT.m,
        plan.result_AT.n,
        copy(plan.result_AT.colptr),
        copy(plan.result_AT.rowval),
        result_nzval
    )

    return SparseMatrixMPI{T}(
        plan.result_structural_hash,
        plan.result_row_partition,
        plan.result_col_partition,
        plan.result_col_indices,
        transpose(result_AT),
        nothing  # cached_transpose
    )
end

"""
    get_repartition_plan(A::SparseMatrixMPI{T}, p::Vector{Int}) where T

Get a memoized SparseRepartitionPlan for repartitioning `A` to row partition `p`.
The plan is cached based on the structural hash of A and the target partition hash.
"""
function get_repartition_plan(A::SparseMatrixMPI{T}, p::Vector{Int}) where T
    target_hash = compute_partition_hash(p)
    key = (_ensure_hash(A), target_hash, T)
    if haskey(_repartition_plan_cache, key)
        return _repartition_plan_cache[key]::SparseRepartitionPlan{T}
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
