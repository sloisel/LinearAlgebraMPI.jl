# MatrixMPI type and dense matrix operations

"""
    compute_dense_structural_hash(row_partition, col_partition, local_size, comm) -> Blake3Hash

Compute a structural hash for a dense matrix that is identical across all ranks.

1. Hash local data: row_partition, col_partition, local matrix size
2. Allgather all local hashes
3. Hash the gathered hashes to produce a global hash
"""
function compute_dense_structural_hash(row_partition::Vector{Int}, col_partition::Vector{Int},
    local_size::Tuple{Int,Int}, comm::MPI.Comm)::Blake3Hash
    # Step 1: Compute rank-local hash
    ctx = Blake3Ctx()
    update!(ctx, reinterpret(UInt8, row_partition))
    update!(ctx, reinterpret(UInt8, col_partition))
    update!(ctx, reinterpret(UInt8, collect(local_size)))
    local_hash = digest(ctx)

    # Step 2: Allgather all local hashes
    all_hashes = MPI.Allgather(local_hash, comm)

    # Step 3: Hash them together to produce global hash
    ctx2 = Blake3Ctx()
    update!(ctx2, all_hashes)
    return Blake3Hash(digest(ctx2))
end

"""
    MatrixMPI{T}

A distributed dense matrix partitioned by rows across MPI ranks.

# Fields
- `structural_hash::Blake3Hash`: 256-bit Blake3 hash of the structural pattern
- `row_partition::Vector{Int}`: Row partition boundaries, length = nranks + 1
- `col_partition::Vector{Int}`: Column partition boundaries, length = nranks + 1 (for transpose)
- `A::Matrix{T}`: Local rows (NOT transposed), size = (local_nrows, ncols)

# Invariants
- `row_partition` and `col_partition` are sorted
- `row_partition[nranks+1]` = total number of rows + 1
- `col_partition[nranks+1]` = total number of columns + 1
- `size(A, 1) == row_partition[rank+2] - row_partition[rank+1]`
- `size(A, 2) == col_partition[end] - 1`
"""
struct MatrixMPI{T}
    structural_hash::Blake3Hash
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    A::Matrix{T}
end

"""
    MatrixMPI(M::Matrix{T}; row_partition=nothing, col_partition=nothing) where T

Create a MatrixMPI from a global matrix M, assuming M is identical on all ranks.
The matrix is partitioned by rows across ranks.

# Arguments
- `M::Matrix{T}`: The global matrix (must be identical on all ranks)
- `row_partition`: Optional custom row partition (default: even distribution)
- `col_partition`: Optional custom column partition (default: even distribution, for transpose)
"""
function MatrixMPI(M::Matrix{T}; row_partition=nothing, col_partition=nothing) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(M)

    # Compute row partition: roughly equal distribution
    if row_partition === nothing
        rows_per_rank = div(m, nranks)
        remainder = mod(m, nranks)

        row_partition = Vector{Int}(undef, nranks + 1)
        row_partition[1] = 1
        for r in 1:nranks
            extra = r <= remainder ? 1 : 0
            row_partition[r+1] = row_partition[r] + rows_per_rank + extra
        end
    end

    # Compute col partition: roughly equal distribution (for transpose)
    if col_partition === nothing
        cols_per_rank = div(n, nranks)
        col_remainder = mod(n, nranks)

        col_partition = Vector{Int}(undef, nranks + 1)
        col_partition[1] = 1
        for r in 1:nranks
            extra = r <= col_remainder ? 1 : 0
            col_partition[r+1] = col_partition[r] + cols_per_rank + extra
        end
    end

    # Local row range (1-indexed, Julia style)
    row_start = row_partition[rank+1]
    row_end = row_partition[rank+2] - 1
    local_rows = row_start:row_end

    # Extract local rows from M (NOT transposed, unlike SparseMatrixMPI)
    A = M[local_rows, :]

    # Compute structural hash (identical across all ranks)
    structural_hash = compute_dense_structural_hash(row_partition, col_partition, size(A), comm)

    return MatrixMPI{T}(structural_hash, row_partition, col_partition, A)
end

# Dense matrix-vector plan and multiplication

"""
    DenseMatrixVectorPlan{T}

A communication plan for gathering vector elements needed for MatrixMPI * x.

For a dense matrix A with ncols columns, we need all elements x[1:ncols].
The plan gathers these elements from the distributed vector x based on x's partition.

# Fields
- `send_rank_ids::Vector{Int}`: Ranks we send elements to (0-indexed)
- `send_indices::Vector{Vector{Int}}`: For each rank, local indices to send
- `send_bufs::Vector{Vector{T}}`: Pre-allocated send buffers
- `send_reqs::Vector{MPI.Request}`: Pre-allocated send request handles
- `recv_rank_ids::Vector{Int}`: Ranks we receive elements from (0-indexed)
- `recv_bufs::Vector{Vector{T}}`: Pre-allocated receive buffers
- `recv_reqs::Vector{MPI.Request}`: Pre-allocated receive request handles
- `recv_perm::Vector{Vector{Int}}`: For each recv rank, indices into gathered
- `local_src_indices::Vector{Int}`: Source indices for local copy (into x.v)
- `local_dst_indices::Vector{Int}`: Destination indices for local copy (into gathered)
- `gathered::Vector{T}`: Pre-allocated buffer for gathered elements (full x vector)
"""
mutable struct DenseMatrixVectorPlan{T}
    send_rank_ids::Vector{Int}
    send_indices::Vector{Vector{Int}}
    send_bufs::Vector{Vector{T}}
    send_reqs::Vector{MPI.Request}
    recv_rank_ids::Vector{Int}
    recv_bufs::Vector{Vector{T}}
    recv_reqs::Vector{MPI.Request}
    recv_perm::Vector{Vector{Int}}
    local_src_indices::Vector{Int}
    local_dst_indices::Vector{Int}
    gathered::Vector{T}
end

"""
    DenseMatrixVectorPlan(A::MatrixMPI{T}, x::VectorMPI{T}) where T

Create a communication plan to gather all x elements (x[1:ncols]) for dense matrix-vector multiplication.
"""
function DenseMatrixVectorPlan(A::MatrixMPI{T}, x::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # For dense matrix, we need all elements 1:ncols
    ncols = A.col_partition[end] - 1
    col_indices = collect(1:ncols)
    n_gathered = ncols

    my_x_start = x.partition[rank+1]

    # Step 1: Group col_indices by owner rank in x's partition
    needed_from = [Tuple{Int,Int}[] for _ in 1:nranks]
    for (dst_idx, global_idx) in enumerate(col_indices)
        owner = searchsortedlast(x.partition, global_idx) - 1
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
    recv_perm_map = Dict{Int,Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r+1] > 0 && r != rank
            push!(recv_rank_ids, r)
            indices = [t[1] for t in needed_from[r+1]]
            dst_indices = [t[2] for t in needed_from[r+1]]
            recv_perm_map[r] = dst_indices
            struct_send_bufs[r] = indices
            req = MPI.Isend(indices, comm; dest=r, tag=30)
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
            req = MPI.Irecv!(buf, comm; source=r, tag=30)
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

    # Step 6: Handle local elements
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

    return DenseMatrixVectorPlan{T}(
        send_rank_ids, send_indices_final, send_bufs, send_reqs,
        recv_rank_ids, recv_bufs, recv_reqs, recv_perm_final,
        local_src_indices, local_dst_indices, gathered
    )
end

"""
    execute_plan!(plan::DenseMatrixVectorPlan{T}, x::VectorMPI{T}) where T

Execute a dense vector communication plan to gather elements from x.
Returns plan.gathered containing x[1:ncols] for the associated matrix.
"""
function execute_plan!(plan::DenseMatrixVectorPlan{T}, x::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD

    # Step 1: Copy local values
    @inbounds for i in eachindex(plan.local_src_indices, plan.local_dst_indices)
        plan.gathered[plan.local_dst_indices[i]] = x.v[plan.local_src_indices[i]]
    end

    # Step 2: Fill send buffers and send
    @inbounds for i in eachindex(plan.send_rank_ids)
        r = plan.send_rank_ids[i]
        send_idx = plan.send_indices[i]
        buf = plan.send_bufs[i]
        for k in eachindex(send_idx)
            buf[k] = x.v[send_idx[k]]
        end
        plan.send_reqs[i] = MPI.Isend(buf, comm; dest=r, tag=31)
    end

    # Step 3: Receive values
    @inbounds for i in eachindex(plan.recv_rank_ids)
        plan.recv_reqs[i] = MPI.Irecv!(plan.recv_bufs[i], comm; source=plan.recv_rank_ids[i], tag=31)
    end

    MPI.Waitall(plan.recv_reqs)

    # Step 4: Scatter received values into gathered
    @inbounds for i in eachindex(plan.recv_rank_ids)
        perm = plan.recv_perm[i]
        buf = plan.recv_bufs[i]
        for k in eachindex(perm)
            plan.gathered[perm[k]] = buf[k]
        end
    end

    MPI.Waitall(plan.send_reqs)

    return plan.gathered
end

"""
    get_dense_vector_plan(A::MatrixMPI{T}, x::VectorMPI{T}) where T

Get a memoized DenseMatrixVectorPlan for A * x.
The plan is cached based on the structural hashes of A and x.
"""
function get_dense_vector_plan(A::MatrixMPI{T}, x::VectorMPI{T}) where T
    key = (A.structural_hash, x.structural_hash, T)
    if haskey(_dense_vector_plan_cache, key)
        return _dense_vector_plan_cache[key]::DenseMatrixVectorPlan{T}
    end
    plan = DenseMatrixVectorPlan(A, x)
    _dense_vector_plan_cache[key] = plan
    return plan
end

"""
    LinearAlgebra.mul!(y::VectorMPI{T}, A::MatrixMPI{T}, x::VectorMPI{T}) where T

In-place dense matrix-vector multiplication: y = A * x.
"""
function LinearAlgebra.mul!(y::VectorMPI{T}, A::MatrixMPI{T}, x::VectorMPI{T}) where T
    plan = get_dense_vector_plan(A, x)
    gathered = execute_plan!(plan, x)

    # Local computation: y.v = A.A * gathered
    # A.A has shape (local_nrows, ncols), gathered has shape (ncols,)
    LinearAlgebra.mul!(y.v, A.A, gathered)
    return y
end

"""
    Base.:*(A::MatrixMPI{T}, x::VectorMPI{T}) where T

Dense matrix-vector multiplication returning a new VectorMPI.
The result has the same row partition as A.
"""
function Base.:*(A::MatrixMPI{T}, x::VectorMPI{T}) where T
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_rows = A.row_partition[rank + 2] - A.row_partition[rank + 1]
    y = VectorMPI{T}(
        compute_partition_hash(A.row_partition),
        copy(A.row_partition),
        Vector{T}(undef, local_rows)
    )
    return LinearAlgebra.mul!(y, A, x)
end

# Dense transpose plan

"""
    DenseTransposePlan{T}

A communication plan for computing the transpose of a MatrixMPI.

The transpose of A (with row_partition R and col_partition C) will have:
- row_partition = C (columns of A become rows of A^T)
- col_partition = R (rows of A become columns of A^T)

# Fields
- `rank_ids::Vector{Int}`: Ranks we send data to (0-indexed)
- `send_row_ranges::Vector{UnitRange{Int}}`: For each rank, local row range to send
- `send_col_ranges::Vector{UnitRange{Int}}`: For each rank, column range to send
- `send_bufs::Vector{Vector{T}}`: Pre-allocated send buffers
- `send_reqs::Vector{MPI.Request}`: Pre-allocated send request handles
- `recv_rank_ids::Vector{Int}`: Ranks we receive data from (0-indexed)
- `recv_row_ranges::Vector{UnitRange{Int}}`: For each recv rank, row range in result
- `recv_col_ranges::Vector{UnitRange{Int}}`: For each recv rank, column range in result
- `recv_bufs::Vector{Vector{T}}`: Pre-allocated receive buffers
- `recv_reqs::Vector{MPI.Request}`: Pre-allocated receive request handles
- `local_row_range::UnitRange{Int}`: Local rows that stay on this rank
- `local_col_range::UnitRange{Int}`: Local columns that stay on this rank
- `AT::Matrix{T}`: Pre-allocated result matrix
- `row_partition::Vector{Int}`: Row partition for the transposed matrix
- `col_partition::Vector{Int}`: Col partition for the transposed matrix
"""
mutable struct DenseTransposePlan{T}
    rank_ids::Vector{Int}
    send_row_ranges::Vector{UnitRange{Int}}
    send_col_ranges::Vector{UnitRange{Int}}
    send_bufs::Vector{Vector{T}}
    send_reqs::Vector{MPI.Request}
    recv_rank_ids::Vector{Int}
    recv_row_ranges::Vector{UnitRange{Int}}
    recv_col_ranges::Vector{UnitRange{Int}}
    recv_bufs::Vector{Vector{T}}
    recv_reqs::Vector{MPI.Request}
    local_row_range::UnitRange{Int}
    local_col_range::UnitRange{Int}
    AT::Matrix{T}
    row_partition::Vector{Int}
    col_partition::Vector{Int}
end

"""
    DenseTransposePlan(A::MatrixMPI{T}) where T

Create a communication plan for computing A^T.

The algorithm:
1. A^T has row_partition = A.col_partition, col_partition = A.row_partition
2. For each destination rank r (owning rows in A^T = columns in A):
   - Determine which columns of our local A need to go to rank r
   - These become rows r's local A^T
3. Exchange data via point-to-point communication
"""
function DenseTransposePlan(A::MatrixMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # The transpose has swapped partitions
    result_row_partition = A.col_partition
    result_col_partition = A.row_partition

    my_row_start = A.row_partition[rank+1]
    my_row_end = A.row_partition[rank+2] - 1
    my_local_nrows = my_row_end - my_row_start + 1
    ncols = A.col_partition[end] - 1

    # Result dimensions
    result_row_start = result_row_partition[rank+1]
    result_row_end = result_row_partition[rank+2] - 1
    result_local_nrows = result_row_end - result_row_start + 1
    result_ncols = result_col_partition[end] - 1

    # Pre-allocate result matrix
    AT = Matrix{T}(undef, result_local_nrows, result_ncols)

    # Determine send/receive patterns
    # We need to send A[my_rows, cols] where cols go to different destination ranks
    # In A^T, these become A^T[cols, my_rows] on the destination rank

    rank_ids = Int[]
    send_row_ranges = UnitRange{Int}[]
    send_col_ranges = UnitRange{Int}[]
    recv_rank_ids = Int[]
    recv_row_ranges = UnitRange{Int}[]
    recv_col_ranges = UnitRange{Int}[]

    # For each rank, determine which columns of A (= rows of A^T) they own
    for r in 0:(nranks-1)
        dest_row_start = result_row_partition[r+1]
        dest_row_end = result_row_partition[r+2] - 1

        if dest_row_end >= dest_row_start
            # These are columns dest_row_start:dest_row_end in A
            col_start = dest_row_start
            col_end = dest_row_end

            if r != rank
                # Send A[1:local_nrows, col_start:col_end] to rank r
                push!(rank_ids, r)
                push!(send_row_ranges, 1:my_local_nrows)
                # Convert global column indices to local
                local_col_start = col_start
                local_col_end = col_end
                push!(send_col_ranges, local_col_start:local_col_end)
            end
        end

        # Determine what we receive from rank r
        # Rank r owns rows src_row_start:src_row_end in A
        # These become columns src_row_start:src_row_end in A^T
        src_row_start = A.row_partition[r+1]
        src_row_end = A.row_partition[r+2] - 1

        if src_row_end >= src_row_start
            # We receive data for our local rows in A^T (= columns in A that we own)
            if r != rank
                push!(recv_rank_ids, r)
                # Our local rows in A^T
                push!(recv_row_ranges, 1:result_local_nrows)
                # The columns in A^T corresponding to rank r's rows in A
                push!(recv_col_ranges, src_row_start:src_row_end)
            end
        end
    end

    # Determine local copy range (data that stays on this rank)
    # This is the block where both row in A^T (= col in A) and col in A^T (= row in A)
    # are owned by this rank
    local_row_range = 1:0  # empty by default
    local_col_range = 1:0  # empty by default

    # My columns in A that I own in A^T (rows result_row_start:result_row_end)
    my_AT_row_start = result_row_start
    my_AT_row_end = result_row_end

    # My columns in A^T are my rows in A (my_row_start:my_row_end)
    # So local copy: A^T[1:result_local_nrows, my_row_start:my_row_end]
    #              = transpose of A[1:my_local_nrows, my_AT_row_start:my_AT_row_end]
    if my_AT_row_end >= my_AT_row_start && my_row_end >= my_row_start
        local_row_range = 1:result_local_nrows
        local_col_range = my_row_start:my_row_end
    end

    # Allocate send/recv buffers
    send_bufs = Vector{Vector{T}}(undef, length(rank_ids))
    for (i, r) in enumerate(rank_ids)
        nrows_to_send = length(send_row_ranges[i])
        ncols_to_send = length(send_col_ranges[i])
        send_bufs[i] = Vector{T}(undef, nrows_to_send * ncols_to_send)
    end

    recv_bufs = Vector{Vector{T}}(undef, length(recv_rank_ids))
    for (i, r) in enumerate(recv_rank_ids)
        nrows_to_recv = length(recv_row_ranges[i])
        ncols_to_recv = length(recv_col_ranges[i])
        recv_bufs[i] = Vector{T}(undef, nrows_to_recv * ncols_to_recv)
    end

    send_reqs = Vector{MPI.Request}(undef, length(rank_ids))
    recv_reqs = Vector{MPI.Request}(undef, length(recv_rank_ids))

    return DenseTransposePlan{T}(
        rank_ids, send_row_ranges, send_col_ranges, send_bufs, send_reqs,
        recv_rank_ids, recv_row_ranges, recv_col_ranges, recv_bufs, recv_reqs,
        local_row_range, local_col_range,
        AT, result_row_partition, result_col_partition
    )
end

"""
    execute_plan!(plan::DenseTransposePlan{T}, A::MatrixMPI{T}) where T

Execute a dense transpose plan to compute A^T.
Returns a MatrixMPI representing the transpose.
"""
function execute_plan!(plan::DenseTransposePlan{T}, A::MatrixMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    my_row_start = A.row_partition[rank+1]
    result_row_start = plan.row_partition[rank+1]

    # Step 1: Copy local values (transpose the block)
    # A^T[local_row_range, local_col_range] = transpose(A[1:nrows, result_row_start:result_row_end])
    if !isempty(plan.local_row_range) && !isempty(plan.local_col_range)
        local_A_cols = result_row_start:(result_row_start + length(plan.local_row_range) - 1)
        for (dst_col_idx, src_row) in enumerate(plan.local_col_range)
            local_src_row = src_row - my_row_start + 1
            for (dst_row_idx, src_col) in enumerate(local_A_cols)
                plan.AT[dst_row_idx, dst_col_idx + my_row_start - 1] = A.A[local_src_row, src_col]
            end
        end
    end

    # Step 2: Fill send buffers (pack transposed data) and send
    for (i, r) in enumerate(plan.rank_ids)
        row_range = plan.send_row_ranges[i]
        col_range = plan.send_col_ranges[i]
        buf = plan.send_bufs[i]
        buf_idx = 1
        # Pack: for each local row (= column in A^T on destination), pack all columns
        # Order must match unpack: outer loop over source rows, inner loop over columns
        for local_row in row_range
            for c in col_range
                buf[buf_idx] = A.A[local_row, c]
                buf_idx += 1
            end
        end
        plan.send_reqs[i] = MPI.Isend(buf, comm; dest=r, tag=40)
    end

    # Step 3: Receive values
    for (i, r) in enumerate(plan.recv_rank_ids)
        plan.recv_reqs[i] = MPI.Irecv!(plan.recv_bufs[i], comm; source=r, tag=40)
    end

    MPI.Waitall(plan.recv_reqs)

    # Step 4: Unpack received values into AT (transposing in the process)
    for (i, r) in enumerate(plan.recv_rank_ids)
        row_range = plan.recv_row_ranges[i]
        col_range = plan.recv_col_ranges[i]
        buf = plan.recv_bufs[i]
        buf_idx = 1
        # Data was packed in column-major order of A (row-major of A^T)
        # So unpack as: for each source row in A (= dest col in A^T), assign to dest rows
        for src_row in col_range
            dst_col = src_row
            for dst_row in row_range
                plan.AT[dst_row, dst_col] = buf[buf_idx]
                buf_idx += 1
            end
        end
    end

    MPI.Waitall(plan.send_reqs)

    # Create result with a copy of AT
    result_AT = copy(plan.AT)

    # Compute structural hash
    structural_hash = compute_dense_structural_hash(
        plan.row_partition, plan.col_partition, size(result_AT), comm)

    return MatrixMPI{T}(structural_hash, plan.row_partition, plan.col_partition, result_AT)
end

"""
    get_dense_transpose_plan(A::MatrixMPI{T}) where T

Get a memoized DenseTransposePlan for A^T.
The plan is cached based on the structural hash of A and the element type.
"""
function get_dense_transpose_plan(A::MatrixMPI{T}) where T
    key = (A.structural_hash, T)
    if haskey(_dense_transpose_plan_cache, key)
        return _dense_transpose_plan_cache[key]::DenseTransposePlan{T}
    end
    plan = DenseTransposePlan(A)
    _dense_transpose_plan_cache[key] = plan
    return plan
end

# Lazy transpose support for MatrixMPI

"""
    Base.transpose(A::MatrixMPI{T}) where T

Return a lazy transpose wrapper around A.
"""
Base.transpose(A::MatrixMPI{T}) where T = Transpose(A)

"""
    Base.conj(A::MatrixMPI{T}) where T

Return a new MatrixMPI with conjugated values.
"""
function Base.conj(A::MatrixMPI{T}) where T
    return MatrixMPI{T}(A.structural_hash, A.row_partition, A.col_partition, conj.(A.A))
end

"""
    Base.adjoint(A::MatrixMPI{T}) where T

Return transpose(conj(A)), i.e., the conjugate transpose.
"""
Base.adjoint(A::MatrixMPI{T}) where T = transpose(conj(A))

# Type alias for transpose of MatrixMPI
const TransposedMatrixMPI{T} = Transpose{T,MatrixMPI{T}}
const AdjointedMatrixMPI{T} = Adjoint{T,MatrixMPI{T}}

"""
    LinearAlgebra.copy(At::Transpose{T,MatrixMPI{T}}) where T

Materialize a transposed MatrixMPI.
"""
function LinearAlgebra.copy(At::Transpose{T,MatrixMPI{T}}) where T
    A = At.parent
    plan = get_dense_transpose_plan(A)
    return execute_plan!(plan, A)
end

"""
    LinearAlgebra.copy(Ah::Adjoint{T,MatrixMPI{T}}) where T

Materialize an adjointed MatrixMPI.
"""
function LinearAlgebra.copy(Ah::Adjoint{T,MatrixMPI{T}}) where T
    # Adjoint = transpose of conjugate
    A_conj = conj(Ah.parent)
    plan = get_dense_transpose_plan(A_conj)
    return execute_plan!(plan, A_conj)
end

# transpose(A) * x for MatrixMPI

"""
    DenseTransposeVectorPlan{T}

A communication plan for computing transpose(A) * x where A is MatrixMPI.

For transpose(A) * x:
- We need to gather x[1:nrows] according to A's row_partition
- Then compute transpose(A.A) * gathered_x locally
- Result has A.col_partition
"""
mutable struct DenseTransposeVectorPlan{T}
    send_rank_ids::Vector{Int}
    send_indices::Vector{Vector{Int}}
    send_bufs::Vector{Vector{T}}
    send_reqs::Vector{MPI.Request}
    recv_rank_ids::Vector{Int}
    recv_bufs::Vector{Vector{T}}
    recv_reqs::Vector{MPI.Request}
    recv_perm::Vector{Vector{Int}}
    local_src_indices::Vector{Int}
    local_dst_indices::Vector{Int}
    gathered::Vector{T}
end

# Cache for DenseTransposeVectorPlans
const _dense_transpose_vector_plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,DataType},Any}()

"""
    DenseTransposeVectorPlan(A::MatrixMPI{T}, x::VectorMPI{T}) where T

Create a plan to gather x elements according to A's row_partition for transpose(A) * x.
"""
function DenseTransposeVectorPlan(A::MatrixMPI{T}, x::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # For transpose(A) * x, we need x[1:nrows] where nrows = A.row_partition[end] - 1
    nrows = A.row_partition[end] - 1
    col_indices = collect(1:nrows)
    n_gathered = nrows

    my_x_start = x.partition[rank+1]

    # Group col_indices by owner rank in x's partition
    needed_from = [Tuple{Int,Int}[] for _ in 1:nranks]
    for (dst_idx, global_idx) in enumerate(col_indices)
        owner = searchsortedlast(x.partition, global_idx) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(needed_from[owner+1], (global_idx, dst_idx))
    end

    # Exchange counts via Alltoall
    send_counts = [length(needed_from[r+1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send requested indices to each owner rank
    struct_send_bufs = Dict{Int,Vector{Int}}()
    struct_send_reqs = MPI.Request[]
    recv_rank_ids = Int[]
    recv_perm_map = Dict{Int,Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r+1] > 0 && r != rank
            push!(recv_rank_ids, r)
            indices = [t[1] for t in needed_from[r+1]]
            dst_indices = [t[2] for t in needed_from[r+1]]
            recv_perm_map[r] = dst_indices
            struct_send_bufs[r] = indices
            req = MPI.Isend(indices, comm; dest=r, tag=50)
            push!(struct_send_reqs, req)
        end
    end

    # Receive requests from other ranks
    send_rank_ids = Int[]
    struct_recv_bufs = Dict{Int,Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r+1] > 0 && r != rank
            push!(send_rank_ids, r)
            buf = Vector{Int}(undef, recv_counts[r+1])
            req = MPI.Irecv!(buf, comm; source=r, tag=50)
            push!(struct_recv_reqs, req)
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Convert received global indices to local indices for sending
    send_indices_map = Dict{Int,Vector{Int}}()
    for r in send_rank_ids
        global_indices = struct_recv_bufs[r]
        local_indices = [idx - my_x_start + 1 for idx in global_indices]
        send_indices_map[r] = local_indices
    end

    # Handle local elements
    local_src_indices = Int[]
    local_dst_indices = Int[]
    for (global_idx, dst_idx) in needed_from[rank+1]
        local_idx = global_idx - my_x_start + 1
        push!(local_src_indices, local_idx)
        push!(local_dst_indices, dst_idx)
    end

    # Build final arrays and buffers
    sort!(send_rank_ids)
    sort!(recv_rank_ids)

    send_indices_final = [send_indices_map[r] for r in send_rank_ids]
    recv_perm_final = [recv_perm_map[r] for r in recv_rank_ids]

    send_bufs = [Vector{T}(undef, length(inds)) for inds in send_indices_final]
    recv_bufs = [Vector{T}(undef, send_counts[r+1]) for r in recv_rank_ids]
    send_reqs = Vector{MPI.Request}(undef, length(send_rank_ids))
    recv_reqs = Vector{MPI.Request}(undef, length(recv_rank_ids))
    gathered = Vector{T}(undef, n_gathered)

    return DenseTransposeVectorPlan{T}(
        send_rank_ids, send_indices_final, send_bufs, send_reqs,
        recv_rank_ids, recv_bufs, recv_reqs, recv_perm_final,
        local_src_indices, local_dst_indices, gathered
    )
end

"""
    execute_plan!(plan::DenseTransposeVectorPlan{T}, x::VectorMPI{T}) where T

Execute the transpose vector plan to gather x elements.
"""
function execute_plan!(plan::DenseTransposeVectorPlan{T}, x::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD

    # Copy local values
    @inbounds for i in eachindex(plan.local_src_indices, plan.local_dst_indices)
        plan.gathered[plan.local_dst_indices[i]] = x.v[plan.local_src_indices[i]]
    end

    # Fill send buffers and send
    @inbounds for i in eachindex(plan.send_rank_ids)
        r = plan.send_rank_ids[i]
        send_idx = plan.send_indices[i]
        buf = plan.send_bufs[i]
        for k in eachindex(send_idx)
            buf[k] = x.v[send_idx[k]]
        end
        plan.send_reqs[i] = MPI.Isend(buf, comm; dest=r, tag=51)
    end

    # Receive values
    @inbounds for i in eachindex(plan.recv_rank_ids)
        plan.recv_reqs[i] = MPI.Irecv!(plan.recv_bufs[i], comm; source=plan.recv_rank_ids[i], tag=51)
    end

    MPI.Waitall(plan.recv_reqs)

    # Scatter received values into gathered
    @inbounds for i in eachindex(plan.recv_rank_ids)
        perm = plan.recv_perm[i]
        buf = plan.recv_bufs[i]
        for k in eachindex(perm)
            plan.gathered[perm[k]] = buf[k]
        end
    end

    MPI.Waitall(plan.send_reqs)

    return plan.gathered
end

"""
    get_dense_transpose_vector_plan(A::MatrixMPI{T}, x::VectorMPI{T}) where T

Get a memoized DenseTransposeVectorPlan for transpose(A) * x.
"""
function get_dense_transpose_vector_plan(A::MatrixMPI{T}, x::VectorMPI{T}) where T
    key = (A.structural_hash, x.structural_hash, T)
    if haskey(_dense_transpose_vector_plan_cache, key)
        return _dense_transpose_vector_plan_cache[key]::DenseTransposeVectorPlan{T}
    end
    plan = DenseTransposeVectorPlan(A, x)
    _dense_transpose_vector_plan_cache[key] = plan
    return plan
end

"""
    Base.:*(At::Transpose{T,MatrixMPI{T}}, x::VectorMPI{T}) where T

Compute transpose(A) * x without materializing the transpose.
"""
function Base.:*(At::Transpose{T,MatrixMPI{T}}, x::VectorMPI{T}) where T
    A = At.parent
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    plan = get_dense_transpose_vector_plan(A, x)
    gathered = execute_plan!(plan, x)

    # Local computation: transpose(A.A) * local_gathered
    # A.A has shape (local_nrows, ncols), gathered has shape (nrows,)
    # We need gathered[my_row_start:my_row_end] for the local rows
    my_row_start = A.row_partition[rank+1]
    my_row_end = A.row_partition[rank+2] - 1
    local_gathered = @view gathered[my_row_start:my_row_end]

    # transpose(A.A) * local_gathered gives a full vector of size ncols
    # This is only the contribution from our local rows - need to sum across all ranks
    ncols = A.col_partition[end] - 1
    partial_result = transpose(A.A) * local_gathered

    # Allreduce to sum contributions from all ranks
    full_result = MPI.Allreduce(partial_result, MPI.SUM, comm)

    # Extract our portion according to col_partition
    my_col_start = A.col_partition[rank+1]
    my_col_end = A.col_partition[rank+2] - 1
    local_result = full_result[my_col_start:my_col_end]

    # Create result vector
    y = VectorMPI{T}(
        compute_partition_hash(A.col_partition),
        copy(A.col_partition),
        local_result
    )
    return y
end

"""
    Base.:*(Ah::Adjoint{T,MatrixMPI{T}}, x::VectorMPI{T}) where T

Compute adjoint(A) * x = conj(transpose(A)) * x without materializing.
"""
function Base.:*(Ah::Adjoint{T,MatrixMPI{T}}, x::VectorMPI{T}) where T
    A = Ah.parent
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    plan = get_dense_transpose_vector_plan(A, x)
    gathered = execute_plan!(plan, x)

    # Local computation: adjoint(A.A) * local_gathered
    my_row_start = A.row_partition[rank+1]
    my_row_end = A.row_partition[rank+2] - 1
    local_gathered = @view gathered[my_row_start:my_row_end]

    # adjoint(A.A) * local_gathered gives a full vector of size ncols
    # This is only the contribution from our local rows - need to sum across all ranks
    partial_result = adjoint(A.A) * local_gathered

    # Allreduce to sum contributions from all ranks
    full_result = MPI.Allreduce(partial_result, MPI.SUM, comm)

    # Extract our portion according to col_partition
    my_col_start = A.col_partition[rank+1]
    my_col_end = A.col_partition[rank+2] - 1
    local_result = full_result[my_col_start:my_col_end]

    # Create result vector
    y = VectorMPI{T}(
        compute_partition_hash(A.col_partition),
        copy(A.col_partition),
        local_result
    )
    return y
end

# Vector * MatrixMPI operations

"""
    Base.:*(vt::Transpose{<:Any, VectorMPI{T}}, A::MatrixMPI{T}) where T

Compute transpose(v) * A = transpose(transpose(A) * v).
"""
function Base.:*(vt::Transpose{<:Any, VectorMPI{T}}, A::MatrixMPI{T}) where T
    v = vt.parent
    result = transpose(A) * v
    return transpose(result)
end

"""
    Base.:*(vh::Adjoint{<:Any, VectorMPI{T}}, A::MatrixMPI{T}) where T

Compute v' * A = transpose(transpose(A) * conj(v)).
"""
function Base.:*(vh::Adjoint{<:Any, VectorMPI{T}}, A::MatrixMPI{T}) where T
    v = vh.parent
    v_conj = conj(v)
    result = transpose(A) * v_conj
    return transpose(result)
end

# Scalar multiplication for MatrixMPI

"""
    Base.:*(a::Number, A::MatrixMPI{T}) where T

Scalar times dense matrix.
"""
function Base.:*(a::Number, A::MatrixMPI{T}) where T
    RT = promote_type(typeof(a), T)
    return MatrixMPI{RT}(A.structural_hash, A.row_partition, A.col_partition, RT.(a .* A.A))
end

"""
    Base.:*(A::MatrixMPI{T}, a::Number) where T

Dense matrix times scalar.
"""
Base.:*(A::MatrixMPI{T}, a::Number) where T = a * A

"""
    Base.:*(a::Number, At::TransposedMatrixMPI{T}) where T

Scalar times transposed dense matrix.
"""
Base.:*(a::Number, At::TransposedMatrixMPI{T}) where T = transpose(a * At.parent)

"""
    Base.:*(At::TransposedMatrixMPI{T}, a::Number) where T

Transposed dense matrix times scalar.
"""
Base.:*(At::TransposedMatrixMPI{T}, a::Number) where T = transpose(At.parent * a)

# Utility methods for MatrixMPI

"""
    Base.size(A::MatrixMPI)

Return the size of the distributed dense matrix as (nrows, ncols).
"""
function Base.size(A::MatrixMPI)
    nrows = A.row_partition[end] - 1
    ncols = A.col_partition[end] - 1
    return (nrows, ncols)
end

Base.size(A::MatrixMPI, d::Integer) = size(A)[d]

Base.eltype(::MatrixMPI{T}) where T = T
Base.eltype(::Type{MatrixMPI{T}}) where T = T

"""
    LinearAlgebra.norm(A::MatrixMPI{T}, p::Real=2) where T

Compute the p-norm of A treated as a vector of elements.
- `p=2` (default): Frobenius norm
- `p=1`: Sum of absolute values
- `p=Inf`: Maximum absolute value
"""
function LinearAlgebra.norm(A::MatrixMPI{T}, p::Real=2) where T
    comm = MPI.COMM_WORLD
    local_vals = A.A

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
        local_sum = sum(x -> abs(x)^p, local_vals; init=zero(real(T)))
        global_sum = MPI.Allreduce(local_sum, MPI.SUM, comm)
        return global_sum^(1 / p)
    end
end

"""
    LinearAlgebra.opnorm(A::MatrixMPI{T}, p::Real=1) where T

Compute the induced operator norm of dense matrix A.
- `p=1`: Maximum absolute column sum
- `p=Inf`: Maximum absolute row sum
"""
function LinearAlgebra.opnorm(A::MatrixMPI{T}, p::Real=1) where T
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    if p == Inf
        # Maximum absolute row sum
        # Each rank owns some rows
        local_nrows = size(A.A, 1)
        local_max = zero(real(T))
        for i in 1:local_nrows
            row_sum = sum(abs, @view A.A[i, :]; init=zero(real(T)))
            local_max = max(local_max, row_sum)
        end
        return MPI.Allreduce(local_max, MPI.MAX, comm)

    elseif p == 1
        # Maximum absolute column sum
        ncols = size(A.A, 2)
        local_col_sums = zeros(real(T), ncols)
        for j in 1:ncols
            local_col_sums[j] = sum(abs, @view A.A[:, j]; init=zero(real(T)))
        end
        global_col_sums = MPI.Allreduce(local_col_sums, MPI.SUM, comm)
        return maximum(global_col_sums; init=zero(real(T)))

    else
        error("opnorm(A, $p) is not implemented. Use p=1 or p=Inf.")
    end
end
