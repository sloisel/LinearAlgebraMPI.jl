module LinearAlgebraMPI

using MPI
using Blake3Hash
using SparseArrays
import LinearAlgebra
using LinearAlgebra: Transpose, Adjoint

export SparseMatrixMPI, MatrixPlan, TransposePlan, clear_plan_cache!, execute_plan!

# Type alias for 256-bit Blake3 hash
const Blake3Hash = NTuple{32,UInt8}

# Cache for memoized MatrixPlans
# Key: (A_hash, B_hash, T) - use full 256-bit hashes
const _plan_cache = Dict{Tuple{Blake3Hash, Blake3Hash, DataType}, Any}()

"""
    clear_plan_cache!()

Clear all memoized plan caches.
"""
function clear_plan_cache!()
    empty!(_plan_cache)
    if isdefined(@__MODULE__, :_addition_plan_cache)
        empty!(_addition_plan_cache)
    end
end

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
- `AT::SparseMatrixCSC{T,Int}`: Transpose of local rows (columns in AT correspond to local rows)

# Invariants
- `col_indices`, `row_partition`, and `col_partition` are sorted
- `row_partition[nranks+1]` = total number of rows
- `col_partition[nranks+1]` = total number of columns
- `size(AT, 2) == row_partition[rank+1] - row_partition[rank]`
"""
struct SparseMatrixMPI{T}
    structural_hash::Blake3Hash
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    col_indices::Vector{Int}
    AT::SparseMatrixCSC{T,Int}
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

    # Extract local rows from A. We need AT = transpose of local part.
    # A is CSC, so A[local_rows, :] gives us the local rows.
    # Then we transpose to get AT.
    # Note: Use transpose(), not ' (adjoint), to avoid conjugating complex values.
    local_A = A[local_rows, :]
    AT = sparse(transpose(local_A))

    # Identify which columns have nonzeros in our local part
    # AT has size (n, local_nrows), colptr/rowval describe structure
    col_indices = unique(sort(AT.rowval))

    # Compute structural hash (identical across all ranks)
    structural_hash = compute_structural_hash(row_partition, col_indices, AT, comm)

    return SparseMatrixMPI{T}(structural_hash, row_partition, col_partition, col_indices, AT)
end

"""
    MatrixPlan{T}

A communication plan for gathering rows from an SparseMatrixMPI.

# Fields
- `rank_ids::Vector{Int}`: Ranks that requested data from us (0-indexed)
- `send_ranges::Vector{Vector{UnitRange{Int}}}`: For each rank, ranges into B.AT.nzval to send
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
    local_ranges::Vector{Tuple{UnitRange{Int}, Int}}
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
    rows_requested_by = Dict{Int, Vector{Int}}()  # rank => rows requested

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
    recv_bufs = Dict{Int, Vector{Int}}()
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
    struct_send_bufs = Dict{Int, Vector{Int}}()  # keep buffers alive

    for r in rank_ids
        requested = rows_requested_by[r]
        local_cols = [row - my_row_start + 1 for row in requested]

        ncols = length(local_cols)
        new_colptr = Vector{Int}(undef, ncols + 1)
        new_colptr[1] = 1
        rowvals = Int[]

        for (i, col) in enumerate(local_cols)
            start_idx = B.AT.colptr[col]
            end_idx = B.AT.colptr[col+1] - 1
            col_nnz = end_idx - start_idx + 1
            new_colptr[i+1] = new_colptr[i] + col_nnz
            append!(rowvals, B.AT.rowval[start_idx:end_idx])
        end

        msg = vcat([ncols], new_colptr, [length(rowvals)], rowvals)
        struct_send_bufs[r] = msg
        req = MPI.Isend(msg, comm; dest=r, tag=2)
        push!(struct_send_reqs, req)
    end

    # Step 3: Receive structure info from ranks we requested from
    recv_rank_ids = Int[]  # ranks we receive from (0-indexed)
    struct_recv_bufs = Dict{Int, Vector{Int}}()

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
    nrows_AT = size(B.AT, 1)
    n_total_cols = length(row_indices)

    # First pass: count total nnz
    total_nnz = 0
    for row in row_indices
        owner = searchsortedlast(B.row_partition, row) - 1
        if owner == rank
            local_col = row - my_row_start + 1
            total_nnz += B.AT.colptr[local_col+1] - B.AT.colptr[local_col]
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
    local_ranges = Tuple{UnitRange{Int}, Int}[]

    # Track receive offsets per rank (consecutive values from each rank)
    recv_val_counts = Dict{Int, Int}()

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
            start_idx = B.AT.colptr[local_col]
            end_idx = B.AT.colptr[local_col+1] - 1
            col_nnz = end_idx - start_idx + 1

            combined_colptr[out_col+1] = combined_colptr[out_col] + col_nnz
            combined_rowval[nnz_idx:nnz_idx+col_nnz-1] = B.AT.rowval[start_idx:end_idx]
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

    # Prepare send info: for each rank in rank_ids, compute ranges into B.AT.nzval
    send_ranges_vec = Vector{Vector{UnitRange{Int}}}()
    send_bufs = Vector{Vector{T}}()

    for r in rank_ids
        requested = rows_requested_by[r]
        ranges = UnitRange{Int}[]
        total_len = 0
        for row in requested
            local_col = row - my_row_start + 1
            start_idx = B.AT.colptr[local_col]
            end_idx = B.AT.colptr[local_col+1] - 1
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
        plan.AT.nzval[dst_off:dst_off+length(src_range)-1] = view(B.AT.nzval, src_range)
    end

    # Step 2: Fill send buffers and send to ranks that requested from us
    for (i, r) in enumerate(plan.rank_ids)
        buf = plan.send_bufs[i]
        offset = 1
        for rng in plan.send_ranges[i]
            n = length(rng)
            buf[offset:offset+n-1] = view(B.AT.nzval, rng)
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

    # Compute local product using: C^T = B^T * A^T, i.e., result_AT = plan.AT * A.AT_reindexed
    #
    # A.AT has shape (ncols_A, local_nrows_A) with rowval containing global column indices
    # plan.AT has shape (ncols_B, n_gathered) where n_gathered = length(A.col_indices)
    #
    # We need to reindex A.AT's rowval from global column indices to local indices 1:n_gathered
    # so that plan.AT * A.AT_reindexed gives us the correct result.

    col_indices = A.col_indices
    n_gathered = length(col_indices)
    col_map = Dict(col => i for (i, col) in enumerate(col_indices))

    # Reindex A.AT: change rowval from global column indices to local indices
    reindexed_rowval = [col_map[r] for r in A.AT.rowval]
    A_AT_reindexed = SparseMatrixCSC(n_gathered, size(A.AT, 2), A.AT.colptr, reindexed_rowval, A.AT.nzval)

    # C^T = B^T * A^T = (plan.AT) * (A.AT_reindexed)
    # plan.AT is (ncols_B, n_gathered), A_AT_reindexed is (n_gathered, local_nrows_A)
    # result is (ncols_B, local_nrows_A) = shape of C.AT
    result_AT = plan.AT * A_AT_reindexed

    # col_indices are the columns of C that have nonzeros in our local rows
    result_col_indices = isempty(result_AT.rowval) ? Int[] : unique(sort(result_AT.rowval))

    # Compute structural hash (identical across all ranks)
    result_hash = compute_structural_hash(A.row_partition, result_col_indices, result_AT, comm)

    # C = A * B has rows from A and columns from B
    return SparseMatrixMPI{T}(result_hash, A.row_partition, B.col_partition, result_col_indices, result_AT)
end

# Cache for addition MatrixPlans (keyed by A's row_partition hash and B's structural hash)
const _addition_plan_cache = Dict{Tuple{Blake3Hash, Blake3Hash, DataType}, Any}()

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
    result_AT = A.AT + plan.AT

    # Compute col_indices and hash
    result_col_indices = isempty(result_AT.rowval) ? Int[] : unique(sort(result_AT.rowval))
    structural_hash = compute_structural_hash(A.row_partition, result_col_indices, result_AT, comm)

    return SparseMatrixMPI{T}(structural_hash, A.row_partition, A.col_partition,
                              result_col_indices, result_AT)
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
    result_AT = A.AT - plan.AT

    # Compute col_indices and hash
    result_col_indices = isempty(result_AT.rowval) ? Int[] : unique(sort(result_AT.rowval))
    structural_hash = compute_structural_hash(A.row_partition, result_col_indices, result_AT, comm)

    return SparseMatrixMPI{T}(structural_hash, A.row_partition, A.col_partition,
                              result_col_indices, result_AT)
end

"""
    TransposePlan{T}

A communication plan for computing the transpose of an SparseMatrixMPI.

The transpose of A (with row_partition R and col_partition C) will have:
- row_partition = C (columns of A become rows of A^T)
- col_partition = R (rows of A become columns of A^T)

# Fields
- `rank_ids::Vector{Int}`: Ranks we send data to (0-indexed)
- `send_indices::Vector{Vector{Int}}`: For each rank, indices into A.AT.nzval to send
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
1. For each nonzero A[i,j] (stored as A.AT[j, local_i]), determine which rank
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

    # Step 1: For each nonzero in A.AT, determine destination rank for transpose
    # A.AT[j, local_col] corresponds to A[global_row, j] where global_row = my_row_start + local_col - 1
    # In A^T, this is A^T[j, global_row], so it goes to the rank owning row j per col_partition

    # Group nonzeros by destination rank: (global_row, j, src_nzval_idx)
    send_to = [Tuple{Int,Int,Int}[] for _ in 1:nranks]

    for local_col in 1:size(A.AT, 2)
        global_row = my_row_start + local_col - 1
        for idx in A.AT.colptr[local_col]:(A.AT.colptr[local_col+1]-1)
            j = A.AT.rowval[idx]  # column index in A, becomes row index in A^T
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
    send_indices_map = Dict{Int, Vector{Int}}()
    struct_send_bufs = Dict{Int, Vector{Int}}()
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
    struct_recv_bufs = Dict{Int, Vector{Int}}()
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
    sort!(entries, by = e -> (e[1] - my_AT_row_start + 1, e[2]))

    # Build CSC structure for result.AT
    # result.AT has dimensions (nrows_A, local_ncols)
    colptr = zeros(Int, local_ncols + 1)
    rowval = Vector{Int}(undef, length(entries))
    nzval = zeros(T, length(entries))

    # Count entries per column (store in colptr[col+1] for cumsum)
    for (j, _, _, _) in entries
        local_col = j - my_AT_row_start + 1
        colptr[local_col + 1] += 1
    end

    # Cumsum to convert counts to pointers, starting from 1
    colptr[1] = 1
    for c in 2:(local_ncols + 1)
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
        plan.AT.nzval[local_dst[i]] = A.AT.nzval[local_src[i]]
    end

    # Step 2: Fill send buffers and send (allocation-free loops)
    @inbounds for i in eachindex(plan.rank_ids)
        r = plan.rank_ids[i]
        send_idx = plan.send_indices[i]
        buf = plan.send_bufs[i]
        for k in eachindex(send_idx)
            buf[k] = A.AT.nzval[send_idx[k]]
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
                              plan.col_indices, result_AT)
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
    local_vals = A.AT.nzval

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
        return global_sum^(1/p)
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
        # A.AT is transposed, so columns of A.AT are rows of A
        local_nrows = size(A.AT, 2)
        local_max = zero(real(T))
        for col in 1:local_nrows
            row_sum = zero(real(T))
            for idx in A.AT.colptr[col]:(A.AT.colptr[col+1]-1)
                row_sum += abs(A.AT.nzval[idx])
            end
            local_max = max(local_max, row_sum)
        end
        return MPI.Allreduce(local_max, MPI.MAX, comm)

    elseif p == 1
        # Maximum absolute column sum
        # Columns are distributed, need to sum contributions from all ranks
        ncols = A.col_partition[end] - 1

        # Compute local contribution to each column sum
        # A.AT.rowval contains column indices of A
        local_col_sums = zeros(real(T), ncols)
        for (idx, col) in enumerate(A.AT.rowval)
            local_col_sums[col] += abs(A.AT.nzval[idx])
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
        A.AT.m, A.AT.n,
        A.AT.colptr,  # share structure (immutable)
        A.AT.rowval,  # share structure (immutable)
        conj.(A.AT.nzval)  # conjugate values
    )
    # Structural hash is the same since structure didn't change
    return SparseMatrixMPI{T}(A.structural_hash, A.row_partition, A.col_partition,
                              A.col_indices, conj_AT)
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
        A.AT.m, A.AT.n,
        A.AT.colptr,
        A.AT.rowval,
        RT.(a .* A.AT.nzval)
    )
    return SparseMatrixMPI{RT}(A.structural_hash, A.row_partition, A.col_partition,
                               A.col_indices, scaled_AT)
end

"""
    *(A::SparseMatrixMPI{T}, a::Number) where T

Matrix times scalar.
"""
Base.:*(A::SparseMatrixMPI{T}, a::Number) where T = a * A

# Type alias for transpose of SparseMatrixMPI
const TransposedSparseMatrixMPI{T} = Transpose{T, SparseMatrixMPI{T}}

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
    plan = TransposePlan(A)
    A_transposed = execute_plan!(plan, A)
    return A_transposed * B
end

"""
    *(A::SparseMatrixMPI, Bt::Transpose)

Compute A * transpose(B) by materializing the transpose of B first.
"""
function Base.:*(A::SparseMatrixMPI{T}, Bt::TransposedSparseMatrixMPI{T}) where T
    B = Bt.parent
    plan = TransposePlan(B)
    B_transposed = execute_plan!(plan, B)
    return A * B_transposed
end

end # module LinearAlgebraMPI
