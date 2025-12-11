# SparseMatrixMPI type and sparse matrix operations

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
    ctx = Blake3Ctx()
    update!(ctx, reinterpret(UInt8, row_partition))
    update!(ctx, reinterpret(UInt8, col_indices))
    update!(ctx, reinterpret(UInt8, AT.colptr))
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
    SparseMatrixMPI{T}

A distributed sparse matrix partitioned by rows across MPI ranks.

# Fields
- `structural_hash::Blake3Hash`: 256-bit Blake3 hash of the structural pattern
- `row_partition::Vector{Int}`: Row partition boundaries, length = nranks + 1
- `col_partition::Vector{Int}`: Column partition boundaries, length = nranks + 1 (placeholder for transpose)
- `col_indices::Vector{Int}`: Column indices that appear in the local part
- `A::Transpose{T,SparseMatrixCSC{T,Int}}`: Local rows as a lazy transpose wrapper around CSC storage

# Invariants
- `col_indices`, `row_partition`, and `col_partition` are sorted
- `row_partition[nranks+1]` = total number of rows
- `col_partition[nranks+1]` = total number of columns
- `size(A, 1) == row_partition[rank+1] - row_partition[rank]` (number of local rows)

# Storage Details
The local rows are stored as `A = transpose(A_csc)` where `A_csc::SparseMatrixCSC` has:
- columns corresponding to global column indices (via rowval)
- rows corresponding to local row indices
Access the underlying CSC via `A.parent` when needed for low-level operations.
"""
struct SparseMatrixMPI{T}
    structural_hash::Blake3Hash
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    col_indices::Vector{Int}
    A::Transpose{T,SparseMatrixCSC{T,Int}}
    cached_transpose::Ref{Union{Nothing, SparseMatrixMPI{T}}}
end

"""
    SparseMatrixMPI{T}(A::SparseMatrixCSC{T,Int}) where T

Create an SparseMatrixMPI from a global sparse matrix A, assuming A is identical on all ranks.
The matrix is partitioned by rows across ranks.
"""
function SparseMatrixMPI{T}(A::SparseMatrixCSC{T,Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Compute row partition: roughly equal distribution
    rows_per_rank = div(m, nranks)
    remainder = mod(m, nranks)

    row_partition = Vector{Int}(undef, nranks + 1)
    row_partition[1] = 1
    for r in 1:nranks
        extra = r <= remainder ? 1 : 0
        row_partition[r+1] = row_partition[r] + rows_per_rank + extra
    end

    # Compute col partition: roughly equal distribution (placeholder for transpose)
    cols_per_rank = div(n, nranks)
    col_remainder = mod(n, nranks)

    col_partition = Vector{Int}(undef, nranks + 1)
    col_partition[1] = 1
    for r in 1:nranks
        extra = r <= col_remainder ? 1 : 0
        col_partition[r+1] = col_partition[r] + cols_per_rank + extra
    end

    # Local row range (1-indexed, Julia style)
    row_start = row_partition[rank+1]
    row_end = row_partition[rank+2] - 1
    local_rows = row_start:row_end

    # Extract local rows from A and store as transpose for efficient row-wise access.
    # A[local_rows, :] gives us the local rows as CSC with shape (local_nrows, ncols).
    # We materialize the transpose to get a CSC with shape (ncols, local_nrows) where
    # columns correspond to local rows - this makes row iteration efficient.
    # Then wrap in transpose() so the type documents the relationship:
    #   - A_local.parent is CSC (ncols, local_nrows), columns = local rows
    #   - A_local = transpose(A_local.parent) conceptually represents local rows
    # Note: Use transpose(), not ' (adjoint), to avoid conjugating complex values.
    local_A = A[local_rows, :]
    AT_storage = sparse(transpose(local_A))  # CSC (ncols, local_nrows), columns = local rows
    A_local = transpose(AT_storage)          # Transpose wrapper for type clarity

    # Identify which columns have nonzeros in our local part
    # AT_storage has columns = local rows, rowval contains global column indices
    col_indices = isempty(AT_storage.rowval) ? Int[] : unique(sort(AT_storage.rowval))

    # Compute structural hash (identical across all ranks)
    structural_hash = compute_structural_hash(row_partition, col_indices, AT_storage, comm)

    return SparseMatrixMPI{T}(structural_hash, row_partition, col_partition, col_indices, A_local,
        Ref{Union{Nothing, SparseMatrixMPI{T}}}(nothing))
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
            append!(rowvals, B.A.parent.rowval[start_idx:end_idx])
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
    nrows_AT = size(B.A.parent, 1)
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
            combined_rowval[nnz_idx:nnz_idx+col_nnz-1] = B.A.parent.rowval[start_idx:end_idx]
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
        plan_AT
    )
end

"""
    MatrixPlan(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T

Create a memoized communication plan for A * B.
The plan is cached based on the structural hashes of A and B.
"""
function MatrixPlan(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
    key = (A.structural_hash, B.structural_hash, T)
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
    Base.*(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T

Multiply two distributed sparse matrices A * B.
"""
function Base.:*(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD

    # Get memoized communication plan and execute it
    plan = MatrixPlan(A, B)
    execute_plan!(plan, B)

    # Compute local product using: C^T = B^T * A^T, i.e., result_AT = plan.AT * A_AT_reindexed
    #
    # A.A.parent has shape (ncols_A, local_nrows_A) with rowval containing global column indices
    # plan.AT has shape (ncols_B, n_gathered) where n_gathered = length(A.col_indices)
    #
    # We need to reindex A.A.parent's rowval from global column indices to local indices 1:n_gathered
    # so that plan.AT * A_AT_reindexed gives us the correct result.

    col_indices = A.col_indices
    n_gathered = length(col_indices)
    col_map = Dict(col => i for (i, col) in enumerate(col_indices))

    # Reindex A.A.parent: change rowval from global column indices to local indices
    reindexed_rowval = [col_map[r] for r in A.A.parent.rowval]
    A_AT_reindexed = SparseMatrixCSC(n_gathered, size(A.A.parent, 2), A.A.parent.colptr, reindexed_rowval, A.A.parent.nzval)

    # C^T = B^T * A^T = (plan.AT) * (A_AT_reindexed)
    # plan.AT is (ncols_B, n_gathered), A_AT_reindexed is (n_gathered, local_nrows_A)
    # result is (ncols_B, local_nrows_A) = shape of C.AT
    result_AT = plan.AT * A_AT_reindexed

    # col_indices are the columns of C that have nonzeros in our local rows
    result_col_indices = isempty(result_AT.rowval) ? Int[] : unique(sort(result_AT.rowval))

    # Compute structural hash (identical across all ranks)
    result_hash = compute_structural_hash(A.row_partition, result_col_indices, result_AT, comm)

    # C = A * B has rows from A and columns from B
    return SparseMatrixMPI{T}(result_hash, A.row_partition, B.col_partition, result_col_indices, transpose(result_AT),
        Ref{Union{Nothing, SparseMatrixMPI{T}}}(nothing))
end

# Cache for addition MatrixPlans (keyed by A's row_partition hash and B's structural hash)
const _addition_plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,DataType},Any}()

"""
    get_addition_plan(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T

Get a memoized MatrixPlan for gathering B's rows to match A's row partition.
Used for A + B or A - B operations.
"""
function get_addition_plan(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
    # Cache key: A's structural hash (determines row partition) + B's structural hash
    key = (A.structural_hash, B.structural_hash, T)
    if haskey(_addition_plan_cache, key)
        return _addition_plan_cache[key]::MatrixPlan{T}
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Get the rows we need from B (our local rows in A's partition)
    my_row_start = A.row_partition[rank+1]
    my_row_end = A.row_partition[rank+2] - 1
    row_indices = collect(my_row_start:my_row_end)

    plan = MatrixPlan(row_indices, B)
    _addition_plan_cache[key] = plan
    return plan
end

"""
    Base.+(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T

Add two distributed sparse matrices. The result has A's row partition.
"""
function Base.:+(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD

    # Get plan and gather B's rows
    plan = get_addition_plan(A, B)
    execute_plan!(plan, B)

    # Local addition: (A + B)^T = A^T + B^T, so we can add AT matrices directly
    result_AT = A.A.parent + plan.AT

    # Compute col_indices and hash
    result_col_indices = isempty(result_AT.rowval) ? Int[] : unique(sort(result_AT.rowval))
    structural_hash = compute_structural_hash(A.row_partition, result_col_indices, result_AT, comm)

    return SparseMatrixMPI{T}(structural_hash, A.row_partition, A.col_partition,
        result_col_indices, transpose(result_AT), Ref{Union{Nothing, SparseMatrixMPI{T}}}(nothing))
end

"""
    Base.-(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T

Subtract two distributed sparse matrices. The result has A's row partition.
"""
function Base.:-(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD

    # Get plan and gather B's rows
    plan = get_addition_plan(A, B)
    execute_plan!(plan, B)

    # Local subtraction: (A - B)^T = A^T - B^T, so we can subtract AT matrices directly
    result_AT = A.A.parent - plan.AT

    # Compute col_indices and hash
    result_col_indices = isempty(result_AT.rowval) ? Int[] : unique(sort(result_AT.rowval))
    structural_hash = compute_structural_hash(A.row_partition, result_col_indices, result_AT, comm)

    return SparseMatrixMPI{T}(structural_hash, A.row_partition, A.col_partition,
        result_col_indices, transpose(result_AT), Ref{Union{Nothing, SparseMatrixMPI{T}}}(nothing))
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
            j = A.A.parent.rowval[idx]  # column index in A, becomes row index in A^T
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
    recv_perm = [Int[] for _ in recv_rank_ids]
    local_src_indices = Int[]
    local_dst_indices = Int[]

    recv_rank_to_idx = Dict(r => i for (i, r) in enumerate(recv_rank_ids))

    for (ent_idx, (_, _, src_rank, src_idx)) in enumerate(entries)
        dst_idx = entry_to_nzval_idx[ent_idx]
        if src_rank == rank
            push!(local_src_indices, local_entries_src[src_idx])
            push!(local_dst_indices, dst_idx)
        else
            push!(recv_perm[recv_rank_to_idx[src_rank]], dst_idx)
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

    return TransposePlan{T}(
        rank_ids, send_indices_final, send_bufs, send_reqs,
        recv_rank_ids, recv_bufs, recv_reqs, recv_perm,
        local_src_indices, local_dst_indices,
        result_AT, result_row_partition, result_col_partition, result_col_indices
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
    result_AT = SparseMatrixCSC(
        plan.AT.m, plan.AT.n,
        copy(plan.AT.colptr),
        copy(plan.AT.rowval),
        copy(plan.AT.nzval)
    )

    # Compute structural hash
    structural_hash = compute_structural_hash(plan.row_partition, plan.col_indices, result_AT, comm)

    return SparseMatrixMPI{T}(structural_hash, plan.row_partition, plan.col_partition,
        plan.col_indices, transpose(result_AT), Ref{Union{Nothing, SparseMatrixMPI{T}}}(nothing))
end

"""
    materialize_transpose(A::SparseMatrixMPI{T}) where T

Materialize the transpose of A, using cached result if available.
If the transpose has been computed before, returns the cached result.
Otherwise, computes the transpose via TransposePlan and caches it bidirectionally
(A.cached_transpose[] = Y and Y.cached_transpose[] = A).
"""
function materialize_transpose(A::SparseMatrixMPI{T}) where T
    # Check if already cached
    if A.cached_transpose[] !== nothing
        return A.cached_transpose[]
    end

    # Compute the transpose
    plan = TransposePlan(A)
    Y = execute_plan!(plan, A)

    # Cache bidirectionally
    A.cached_transpose[] = Y
    Y.cached_transpose[] = A

    return Y
end

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
        local_src_indices, local_dst_indices, gathered
    )
end

"""
    get_vector_plan(A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T

Get a memoized VectorPlan for A * x.
The plan is cached based on the structural hashes of A and x.
"""
function get_vector_plan(A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T
    key = (A.structural_hash, x.structural_hash, T)
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
2. Reindex A.A.parent to use local indices
3. Compute y.v = transpose(A_AT_reindexed) * gathered
"""
function LinearAlgebra.mul!(y::VectorMPI{T}, A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD

    # Get memoized plan and execute it
    plan = get_vector_plan(A, x)
    gathered = execute_plan!(plan, x)

    # Reindex A.A.parent for local computation
    col_indices = A.col_indices
    n_gathered = length(col_indices)
    col_map = Dict(col => i for (i, col) in enumerate(col_indices))

    reindexed_rowval = [col_map[r] for r in A.A.parent.rowval]
    A_AT_reindexed = SparseMatrixCSC(n_gathered, size(A.A.parent, 2), A.A.parent.colptr, reindexed_rowval, A.A.parent.nzval)

    # y = A * x => y^T = x^T * A^T => y.v = transpose(A_AT_reindexed) * gathered
    # Use transpose() not ' to avoid conjugation for complex types
    LinearAlgebra.mul!(y.v, transpose(A_AT_reindexed), gathered)
    return y
end

"""
    Base.:*(A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T

Sparse matrix-vector multiplication returning a new VectorMPI.
The result has the same row partition as A.
"""
function Base.:*(A::SparseMatrixMPI{T}, x::VectorMPI{T}) where T
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_rows = A.row_partition[rank + 2] - A.row_partition[rank + 1]
    y = VectorMPI{T}(
        compute_partition_hash(A.row_partition),
        copy(A.row_partition),
        Vector{T}(undef, local_rows)
    )
    return LinearAlgebra.mul!(y, A, x)
end

"""
    *(vt::Transpose{<:Any, VectorMPI{T}}, A::SparseMatrixMPI{T}) where T

Compute transpose(v) * A as transpose(transpose(A) * v).
Returns a transposed VectorMPI.
"""
function Base.:*(vt::Transpose{<:Any, VectorMPI{T}}, A::SparseMatrixMPI{T}) where T
    v = vt.parent
    # transpose(v) * A = transpose(transpose(A) * v)
    A_transposed = materialize_transpose(A)
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
        # A.A.parent.rowval contains column indices of A
        local_col_sums = zeros(real(T), ncols)
        for (idx, col) in enumerate(A.A.parent.rowval)
            local_col_sums[col] += abs(A.A.parent.nzval[idx])
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
        A.col_indices, transpose(conj_AT), Ref{Union{Nothing, SparseMatrixMPI{T}}}(nothing))
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
        A.col_indices, transpose(scaled_AT), Ref{Union{Nothing, SparseMatrixMPI{RT}}}(nothing))
end

"""
    *(A::SparseMatrixMPI{T}, a::Number) where T

Matrix times scalar.
"""
Base.:*(A::SparseMatrixMPI{T}, a::Number) where T = a * A

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
    A_transposed = materialize_transpose(A)
    return A_transposed * B
end

"""
    *(A::SparseMatrixMPI, Bt::Transpose)

Compute A * transpose(B) by materializing the transpose of B first.
"""
function Base.:*(A::SparseMatrixMPI{T}, Bt::TransposedSparseMatrixMPI{T}) where T
    B = Bt.parent
    B_transposed = materialize_transpose(B)
    return A * B_transposed
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
        Ref{Union{Nothing, SparseMatrixMPI{T}}}(nothing)
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
        A.col_indices, transpose(new_AT), Ref{Union{Nothing, SparseMatrixMPI{RT}}}(nothing))
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
        local_col_sums = zeros(T, n)
        for (idx, col) in enumerate(A.A.parent.rowval)
            local_col_sums[col] += A.A.parent.nzval[idx]
        end
        global_col_sums = MPI.Allreduce(local_col_sums, MPI.SUM, comm)
        return VectorMPI(global_col_sums, comm)
    elseif dims == 2
        # Sum over columns: result is length-m vector (row sums)
        local_nrows = size(A.A.parent, 2)
        local_row_sums = zeros(T, local_nrows)

        for local_col in 1:local_nrows
            for nz_idx in A.A.parent.colptr[local_col]:(A.A.parent.colptr[local_col+1]-1)
                local_row_sums[local_col] += A.A.parent.nzval[nz_idx]
            end
        end

        # Result has A's row partition
        hash = compute_partition_hash(A.row_partition)
        return VectorMPI{T}(hash, copy(A.row_partition), local_row_sums)
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
    for local_row in 1:(my_row_end - my_row_start + 1)
        global_row = my_row_start + local_row - 1
        # Diagonal element is at (global_row, global_row) if within bounds
        if global_row <= n
            # Search for column global_row in A.A.parent[:, local_row]
            for nz_idx in A.A.parent.colptr[local_row]:(A.A.parent.colptr[local_row+1]-1)
                if A.A.parent.rowval[nz_idx] == global_row
                    local_trace += A.A.parent.nzval[nz_idx]
                    break
                end
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
        new_col_indices, transpose(new_AT), Ref{Union{Nothing, SparseMatrixMPI{T}}}(nothing))
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

The result is distributed across ranks with an even partition.
"""
function diag(A::SparseMatrixMPI{T}, k::Integer=0) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Compute diagonal length
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

    # Each rank extracts diagonal elements from its rows
    my_row_start = A.row_partition[rank+1]
    my_row_end = A.row_partition[rank+2] - 1

    # Build full diagonal using Allreduce (each rank contributes its portion)
    full_diag = zeros(T, diag_len)
    for d in 1:diag_len
        global_row = row_offset + d
        global_col = col_offset + d

        if global_row >= my_row_start && global_row <= my_row_end
            local_row = global_row - my_row_start + 1
            # Search for column global_col in A.A.parent[:, local_row]
            for nz_idx in A.A.parent.colptr[local_row]:(A.A.parent.colptr[local_row+1]-1)
                if A.A.parent.rowval[nz_idx] == global_col
                    full_diag[d] = A.A.parent.nzval[nz_idx]
                    break
                end
            end
        end
    end

    # Allreduce combines contributions (only one rank has each element)
    global_diag = MPI.Allreduce(full_diag, MPI.SUM, comm)

    # Create VectorMPI from the global diagonal
    return VectorMPI(global_diag, comm)
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

    # Build new sparse structure keeping only upper triangular entries
    # Entry (i, j) is kept if j >= i + k, i.e., j - i >= k

    new_colptr = Vector{Int}(undef, size(A.A.parent, 2) + 1)
    new_colptr[1] = 1

    # First pass: count entries per column
    nnz_per_col = zeros(Int, size(A.A.parent, 2))
    for local_col in 1:size(A.A.parent, 2)
        global_row = my_row_start + local_col - 1
        for nz_idx in A.A.parent.colptr[local_col]:(A.A.parent.colptr[local_col+1]-1)
            j = A.A.parent.rowval[nz_idx]  # column in original A
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

    # Second pass: fill entries
    idx = 1
    for local_col in 1:size(A.A.parent, 2)
        global_row = my_row_start + local_col - 1
        for nz_idx in A.A.parent.colptr[local_col]:(A.A.parent.colptr[local_col+1]-1)
            j = A.A.parent.rowval[nz_idx]
            if j >= global_row + k
                new_rowval[idx] = j
                new_nzval[idx] = A.A.parent.nzval[nz_idx]
                idx += 1
            end
        end
    end

    new_AT = SparseMatrixCSC(A.A.parent.m, A.A.parent.n, new_colptr, new_rowval, new_nzval)
    new_col_indices = isempty(new_rowval) ? Int[] : unique(sort(new_rowval))

    structural_hash = compute_structural_hash(A.row_partition, new_col_indices, new_AT, comm)

    return SparseMatrixMPI{T}(structural_hash, copy(A.row_partition), copy(A.col_partition),
        new_col_indices, transpose(new_AT), Ref{Union{Nothing, SparseMatrixMPI{T}}}(nothing))
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

    # Keep entry (i, j) if j <= i + k

    new_colptr = Vector{Int}(undef, size(A.A.parent, 2) + 1)
    new_colptr[1] = 1

    nnz_per_col = zeros(Int, size(A.A.parent, 2))
    for local_col in 1:size(A.A.parent, 2)
        global_row = my_row_start + local_col - 1
        for nz_idx in A.A.parent.colptr[local_col]:(A.A.parent.colptr[local_col+1]-1)
            j = A.A.parent.rowval[nz_idx]
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
            j = A.A.parent.rowval[nz_idx]
            if j <= global_row + k
                new_rowval[idx] = j
                new_nzval[idx] = A.A.parent.nzval[nz_idx]
                idx += 1
            end
        end
    end

    new_AT = SparseMatrixCSC(A.A.parent.m, A.A.parent.n, new_colptr, new_rowval, new_nzval)
    new_col_indices = isempty(new_rowval) ? Int[] : unique(sort(new_rowval))

    structural_hash = compute_structural_hash(A.row_partition, new_col_indices, new_AT, comm)

    return SparseMatrixMPI{T}(structural_hash, copy(A.row_partition), copy(A.col_partition),
        new_col_indices, transpose(new_AT), Ref{Union{Nothing, SparseMatrixMPI{T}}}(nothing))
end

# ============================================================================
# Extended SparseMatrixCSC API - Diagonal Matrix Construction
# ============================================================================

"""
    _gather_vectormpi(v::VectorMPI{T}) where T

Gather a distributed VectorMPI to a full vector on all ranks.
"""
function _gather_vectormpi(v::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)

    # Get counts for each rank
    counts = Int32[v.partition[r+2] - v.partition[r+1] for r in 0:nranks-1]

    # Use Allgatherv to gather the full vector
    full_v = Vector{T}(undef, length(v))
    MPI.Allgatherv!(v.v, MPI.VBuffer(full_v, counts), comm)

    return full_v
end

"""
    spdiagm(kv::Pair{<:Integer, <:VectorMPI}...)

Construct a sparse diagonal SparseMatrixMPI from pairs of diagonals and VectorMPI vectors.

# Example
```julia
v1 = VectorMPI([1.0, 2.0, 3.0])
v2 = VectorMPI([4.0, 5.0])
A = spdiagm(0 => v1, 1 => v2)  # Main diagonal and first superdiagonal
```
"""
function spdiagm(kv::Pair{<:Integer, <:VectorMPI}...)
    comm = MPI.COMM_WORLD

    # Gather all VectorMPI to full vectors
    gathered_kv = map(kv) do (k, v)
        k => _gather_vectormpi(v)
    end

    # Call standard spdiagm to create global sparse matrix
    A_global = SparseArrays.spdiagm(gathered_kv...)

    # Create SparseMatrixMPI from the global matrix
    return SparseMatrixMPI{eltype(A_global)}(A_global)
end

"""
    spdiagm(m::Integer, n::Integer, kv::Pair{<:Integer, <:VectorMPI}...)

Construct an m×n sparse diagonal SparseMatrixMPI from pairs of diagonals and VectorMPI vectors.

# Example
```julia
v = VectorMPI([1.0, 2.0])
A = spdiagm(4, 4, 0 => v)  # 4×4 matrix with [1,2] on main diagonal
```
"""
function spdiagm(m::Integer, n::Integer, kv::Pair{<:Integer, <:VectorMPI}...)
    comm = MPI.COMM_WORLD

    # Gather all VectorMPI to full vectors
    gathered_kv = map(kv) do (k, v)
        k => _gather_vectormpi(v)
    end

    # Call standard spdiagm to create global sparse matrix
    A_global = SparseArrays.spdiagm(m, n, gathered_kv...)

    # Create SparseMatrixMPI from the global matrix
    return SparseMatrixMPI{eltype(A_global)}(A_global)
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
function spdiagm(v::VectorMPI)
    return spdiagm(0 => v)
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
