# VectorMPI type and vector operations

"""
    VectorMPI{T}

A distributed dense vector partitioned across MPI ranks.

# Fields
- `structural_hash::Blake3Hash`: 256-bit Blake3 hash of the partition
- `partition::Vector{Int}`: Partition boundaries, length = nranks + 1
- `v::Vector{T}`: Local vector elements owned by this rank
"""
struct VectorMPI{T}
    structural_hash::Blake3Hash
    partition::Vector{Int}
    v::Vector{T}
end

"""
    VectorMPI(v_global::Vector{T}, comm::MPI.Comm=MPI.COMM_WORLD) where T

Create a VectorMPI from a global vector, partitioning it across MPI ranks.
Assumes v_global is identical on all ranks.
"""
function VectorMPI(v_global::Vector{T}, comm::MPI.Comm=MPI.COMM_WORLD) where T
    nranks = MPI.Comm_size(comm)
    n = length(v_global)

    # Compute partition (same logic as SparseMatrixMPI)
    rows_per_rank = div(n, nranks)
    remainder = mod(n, nranks)

    partition = Vector{Int}(undef, nranks + 1)
    partition[1] = 1
    for r in 1:nranks
        extra = r <= remainder ? 1 : 0
        partition[r+1] = partition[r] + rows_per_rank + extra
    end

    rank = MPI.Comm_rank(comm)
    local_range = partition[rank + 1]:(partition[rank + 2] - 1)
    local_v = v_global[local_range]

    hash = compute_partition_hash(partition)
    return VectorMPI{T}(hash, partition, local_v)
end

"""
    VectorPlan{T}

A communication plan for gathering vector elements needed for A * x.

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
- `gathered::Vector{T}`: Pre-allocated buffer for gathered elements
"""
mutable struct VectorPlan{T}
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
    VectorPlan(target_partition::Vector{Int}, source::VectorMPI{T}) where T

Create a communication plan to gather elements from `source` according to `target_partition`.
This allows binary operations between vectors with different partitions.

After executing, `plan.gathered` contains `source[target_partition[rank+1]:target_partition[rank+2]-1]`.
"""
function VectorPlan(target_partition::Vector{Int}, source::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Indices this rank needs from source (contiguous range)
    my_start = target_partition[rank+1]
    my_end = target_partition[rank+2] - 1
    col_indices = collect(my_start:my_end)
    n_gathered = length(col_indices)

    my_x_start = source.partition[rank+1]

    # Step 1: Group col_indices by owner rank in source's partition
    needed_from = [Tuple{Int,Int}[] for _ in 1:nranks]
    for (dst_idx, global_idx) in enumerate(col_indices)
        owner = searchsortedlast(source.partition, global_idx) - 1
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
    recv_perm_map = Dict{Int,Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r+1] > 0 && r != rank
            push!(recv_rank_ids, r)
            indices = [t[1] for t in needed_from[r+1]]
            dst_indices = [t[2] for t in needed_from[r+1]]
            recv_perm_map[r] = dst_indices
            struct_send_bufs[r] = indices
            req = MPI.Isend(indices, comm; dest=r, tag=22)
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
            req = MPI.Irecv!(buf, comm; source=r, tag=22)
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

    # Step 6: Handle local elements (elements we own in source)
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
    execute_plan!(plan::VectorPlan{T}, x::VectorMPI{T}) where T

Execute a vector communication plan to gather elements from x.
Returns plan.gathered containing x[A.col_indices] for the associated matrix A.
"""
function execute_plan!(plan::VectorPlan{T}, x::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD

    # Step 1: Copy local values (allocation-free loop)
    @inbounds for i in eachindex(plan.local_src_indices, plan.local_dst_indices)
        plan.gathered[plan.local_dst_indices[i]] = x.v[plan.local_src_indices[i]]
    end

    # Step 2: Fill send buffers and send (allocation-free loops)
    @inbounds for i in eachindex(plan.send_rank_ids)
        r = plan.send_rank_ids[i]
        send_idx = plan.send_indices[i]
        buf = plan.send_bufs[i]
        for k in eachindex(send_idx)
            buf[k] = x.v[send_idx[k]]
        end
        plan.send_reqs[i] = MPI.Isend(buf, comm; dest=r, tag=21)
    end

    # Step 3: Receive values
    @inbounds for i in eachindex(plan.recv_rank_ids)
        plan.recv_reqs[i] = MPI.Irecv!(plan.recv_bufs[i], comm; source=plan.recv_rank_ids[i], tag=21)
    end

    MPI.Waitall(plan.recv_reqs)

    # Step 4: Scatter received values into gathered (allocation-free loops)
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
    get_vector_align_plan(target::VectorMPI{T}, source::VectorMPI{T}) where T

Get a memoized VectorPlan for aligning `source` to `target`'s partition.
The plan is cached based on the structural hashes of both vectors.
Uses a separate cache from get_vector_plan to avoid key collisions.
"""
function get_vector_align_plan(target::VectorMPI{T}, source::VectorMPI{T}) where T
    key = (target.structural_hash, source.structural_hash, T)
    if haskey(_vector_align_plan_cache, key)
        return _vector_align_plan_cache[key]::VectorPlan{T}
    end
    plan = VectorPlan(target.partition, source)
    _vector_align_plan_cache[key] = plan
    return plan
end

# Vector operations: conj, transpose, adjoint

"""
    Base.conj(v::VectorMPI{T}) where T

Return a new VectorMPI with conjugated values.
"""
function Base.conj(v::VectorMPI{T}) where T
    return VectorMPI{T}(v.structural_hash, v.partition, conj.(v.v))
end

"""
    Base.transpose(v::VectorMPI{T}) where T

Return a lazy transpose wrapper around v.
"""
Base.transpose(v::VectorMPI{T}) where T = Transpose(v)

"""
    Base.adjoint(v::VectorMPI{T}) where T

Return transpose(conj(v)), i.e., the conjugate transpose.
The conj(v) is materialized.
"""
Base.adjoint(v::VectorMPI{T}) where T = transpose(conj(v))

# Vector norms and reductions

"""
    LinearAlgebra.norm(v::VectorMPI{T}, p::Real=2) where T

Compute the p-norm of the distributed vector v.
- `p=2` (default): Euclidean norm (sqrt of sum of squared absolute values)
- `p=1`: Sum of absolute values
- `p=Inf`: Maximum absolute value
"""
function LinearAlgebra.norm(v::VectorMPI{T}, p::Real=2) where T
    comm = MPI.COMM_WORLD

    if p == 2
        local_sum = sum(abs2, v.v; init=zero(real(T)))
        global_sum = MPI.Allreduce(local_sum, MPI.SUM, comm)
        return sqrt(global_sum)
    elseif p == 1
        local_sum = sum(abs, v.v; init=zero(real(T)))
        return MPI.Allreduce(local_sum, MPI.SUM, comm)
    elseif p == Inf
        local_max = isempty(v.v) ? zero(real(T)) : maximum(abs, v.v)
        return MPI.Allreduce(local_max, MPI.MAX, comm)
    else
        # General p-norm
        local_sum = sum(x -> abs(x)^p, v.v; init=zero(real(T)))
        global_sum = MPI.Allreduce(local_sum, MPI.SUM, comm)
        return global_sum^(1 / p)
    end
end

"""
    Base.maximum(v::VectorMPI{T}) where T

Compute the maximum element of the distributed vector.
"""
function Base.maximum(v::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    local_max = isempty(v.v) ? typemin(real(T)) : maximum(real, v.v)
    return MPI.Allreduce(local_max, MPI.MAX, comm)
end

"""
    Base.minimum(v::VectorMPI{T}) where T

Compute the minimum element of the distributed vector.
"""
function Base.minimum(v::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    local_min = isempty(v.v) ? typemax(real(T)) : minimum(real, v.v)
    return MPI.Allreduce(local_min, MPI.MIN, comm)
end

"""
    Base.sum(v::VectorMPI{T}) where T

Compute the sum of all elements in the distributed vector.
"""
function Base.sum(v::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    local_sum = sum(v.v; init=zero(T))
    return MPI.Allreduce(local_sum, MPI.SUM, comm)
end

"""
    Base.prod(v::VectorMPI{T}) where T

Compute the product of all elements in the distributed vector.
"""
function Base.prod(v::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    local_prod = prod(v.v; init=one(T))
    return MPI.Allreduce(local_prod, MPI.PROD, comm)
end

# Vector addition and subtraction

"""
    Base.:+(u::VectorMPI{T}, v::VectorMPI{T}) where T

Add two distributed vectors. If partitions differ, v is aligned to u's partition.
The result has u's partition.
"""
function Base.:+(u::VectorMPI{T}, v::VectorMPI{T}) where T
    if u.partition == v.partition
        return VectorMPI{T}(u.structural_hash, u.partition, u.v .+ v.v)
    else
        # Align v to u's partition
        plan = get_vector_align_plan(u, v)
        v_aligned = execute_plan!(plan, v)
        return VectorMPI{T}(u.structural_hash, copy(u.partition), u.v .+ v_aligned)
    end
end

"""
    Base.:-(u::VectorMPI{T}, v::VectorMPI{T}) where T

Subtract two distributed vectors. If partitions differ, v is aligned to u's partition.
The result has u's partition.
"""
function Base.:-(u::VectorMPI{T}, v::VectorMPI{T}) where T
    if u.partition == v.partition
        return VectorMPI{T}(u.structural_hash, u.partition, u.v .- v.v)
    else
        # Align v to u's partition
        plan = get_vector_align_plan(u, v)
        v_aligned = execute_plan!(plan, v)
        return VectorMPI{T}(u.structural_hash, copy(u.partition), u.v .- v_aligned)
    end
end

"""
    Base.:-(v::VectorMPI{T}) where T

Negate a distributed vector.
"""
function Base.:-(v::VectorMPI{T}) where T
    return VectorMPI{T}(v.structural_hash, v.partition, .-v.v)
end

# Mixed transpose addition/subtraction
# transpose(u) +/- transpose(v) works, aligning v to u's partition if needed

"""
    Base.:+(ut::Transpose{<:Any, VectorMPI{T}}, vt::Transpose{<:Any, VectorMPI{T}}) where T

Add two transposed vectors. If partitions differ, vt is aligned to ut's partition.
Returns a transposed VectorMPI.
"""
function Base.:+(ut::Transpose{<:Any, VectorMPI{T}}, vt::Transpose{<:Any, VectorMPI{T}}) where T
    return transpose(ut.parent + vt.parent)
end

"""
    Base.:-(ut::Transpose{<:Any, VectorMPI{T}}, vt::Transpose{<:Any, VectorMPI{T}}) where T

Subtract two transposed vectors. If partitions differ, vt is aligned to ut's partition.
Returns a transposed VectorMPI.
"""
function Base.:-(ut::Transpose{<:Any, VectorMPI{T}}, vt::Transpose{<:Any, VectorMPI{T}}) where T
    return transpose(ut.parent - vt.parent)
end

"""
    Base.:-(vt::Transpose{<:Any, VectorMPI{T}}) where T

Negate a transposed vector. Returns a transposed VectorMPI.
"""
function Base.:-(vt::Transpose{<:Any, VectorMPI{T}}) where T
    return transpose(-vt.parent)
end

# Scalar multiplication for VectorMPI

"""
    Base.:*(a::Number, v::VectorMPI{T}) where T

Scalar times vector.
"""
function Base.:*(a::Number, v::VectorMPI{T}) where T
    RT = promote_type(typeof(a), T)
    return VectorMPI{RT}(v.structural_hash, v.partition, RT.(a .* v.v))
end

"""
    Base.:*(v::VectorMPI{T}, a::Number) where T

Vector times scalar.
"""
Base.:*(v::VectorMPI{T}, a::Number) where T = a * v

"""
    Base.:/(v::VectorMPI{T}, a::Number) where T

Vector divided by scalar.
"""
function Base.:/(v::VectorMPI{T}, a::Number) where T
    RT = promote_type(typeof(a), T)
    return VectorMPI{RT}(v.structural_hash, v.partition, RT.(v.v ./ a))
end

# Scalar multiplication for transposed VectorMPI

"""
    Base.:*(a::Number, vt::Transpose{<:Any, VectorMPI{T}}) where T

Scalar times transposed vector.
"""
Base.:*(a::Number, vt::Transpose{<:Any, VectorMPI{T}}) where T = transpose(a * vt.parent)

"""
    Base.:*(vt::Transpose{<:Any, VectorMPI{T}}, a::Number) where T

Transposed vector times scalar.
"""
Base.:*(vt::Transpose{<:Any, VectorMPI{T}}, a::Number) where T = transpose(vt.parent * a)

"""
    Base.:/(vt::Transpose{<:Any, VectorMPI{T}}, a::Number) where T

Transposed vector divided by scalar.
"""
Base.:/(vt::Transpose{<:Any, VectorMPI{T}}, a::Number) where T = transpose(vt.parent / a)

# Vector size and eltype

"""
    Base.length(v::VectorMPI)

Return the total length of the distributed vector.
"""
Base.length(v::VectorMPI) = v.partition[end] - 1

"""
    Base.size(v::VectorMPI)

Return the size of the distributed vector as a tuple.
"""
Base.size(v::VectorMPI) = (length(v),)

Base.size(v::VectorMPI, d::Integer) = d == 1 ? length(v) : 1

Base.eltype(::VectorMPI{T}) where T = T
Base.eltype(::Type{VectorMPI{T}}) where T = T
