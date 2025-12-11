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
    VectorMPI_local(v_local::Vector{T}, comm::MPI.Comm=MPI.COMM_WORLD) where T

Create a VectorMPI from a local vector on each rank.

Unlike `VectorMPI(v_global)` which takes a global vector and partitions it,
this constructor takes only the local portion of the vector that each rank owns.
The partition is computed by gathering the local sizes from all ranks.

# Example
```julia
# Rank 0 has [1.0, 2.0], Rank 1 has [3.0, 4.0, 5.0]
v = VectorMPI_local([1.0, 2.0])  # on rank 0
v = VectorMPI_local([3.0, 4.0, 5.0])  # on rank 1
# Result: distributed vector [1.0, 2.0, 3.0, 4.0, 5.0] with partition [1, 3, 6]
```
"""
function VectorMPI_local(v_local::Vector{T}, comm::MPI.Comm=MPI.COMM_WORLD) where T
    nranks = MPI.Comm_size(comm)

    # Gather local sizes from all ranks
    local_size = Int32(length(v_local))
    all_sizes = MPI.Allgather(local_size, comm)

    # Build partition from sizes
    partition = Vector{Int}(undef, nranks + 1)
    partition[1] = 1
    for r in 1:nranks
        partition[r+1] = partition[r] + all_sizes[r]
    end

    hash = compute_partition_hash(partition)
    return VectorMPI{T}(hash, partition, copy(v_local))
end

"""
    VectorMPI(v_global::Vector{T}, comm::MPI.Comm=MPI.COMM_WORLD) where T

Create a VectorMPI from a global vector, partitioning it across MPI ranks.

The vector is partitioned roughly equally across ranks. Each rank extracts only
its local portion from `v_global`, so:

- **Simple usage**: Pass identical `v_global` to all ranks
- **Efficient usage**: Pass a vector with correct `length(v_global)` on all ranks,
  but only populate the elements that each rank owns (other elements are ignored)

The partition boundaries can be computed as `div(n, nranks)` elements per rank,
with the first `mod(n, nranks)` ranks getting one extra element.
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

# ============================================================================
# Extended VectorMPI API - Element-wise Operations
# ============================================================================

"""
    Base.abs(v::VectorMPI{T}) where T

Return a new VectorMPI with absolute values of all elements.
"""
function Base.abs(v::VectorMPI{T}) where T
    RT = real(T)
    return VectorMPI{RT}(v.structural_hash, v.partition, abs.(v.v))
end

"""
    Base.abs2(v::VectorMPI{T}) where T

Return a new VectorMPI with squared absolute values of all elements.
"""
function Base.abs2(v::VectorMPI{T}) where T
    RT = real(T)
    return VectorMPI{RT}(v.structural_hash, v.partition, abs2.(v.v))
end

"""
    Base.real(v::VectorMPI{T}) where T

Return a new VectorMPI containing the real parts of all elements.
"""
function Base.real(v::VectorMPI{T}) where T
    RT = real(T)
    return VectorMPI{RT}(v.structural_hash, v.partition, real.(v.v))
end

"""
    Base.imag(v::VectorMPI{T}) where T

Return a new VectorMPI containing the imaginary parts of all elements.
"""
function Base.imag(v::VectorMPI{T}) where T
    RT = real(T)
    return VectorMPI{RT}(v.structural_hash, v.partition, imag.(v.v))
end

"""
    Base.copy(v::VectorMPI{T}) where T

Create a deep copy of the distributed vector.
"""
function Base.copy(v::VectorMPI{T}) where T
    return VectorMPI{T}(v.structural_hash, copy(v.partition), copy(v.v))
end

"""
    mean(v::VectorMPI{T}) where T

Compute the mean of all elements in the distributed vector.
"""
function mean(v::VectorMPI{T}) where T
    return sum(v) / length(v)
end

# ============================================================================
# Broadcasting Support for VectorMPI
# ============================================================================

import Base.Broadcast: BroadcastStyle, Broadcasted, DefaultArrayStyle, AbstractArrayStyle
import Base.Broadcast: broadcasted, materialize, instantiate, broadcastable

"""
    VectorMPIStyle <: AbstractArrayStyle{1}

Custom broadcast style for VectorMPI that ensures broadcast operations
return VectorMPI results and handle distributed data correctly.
"""
struct VectorMPIStyle <: AbstractArrayStyle{1} end

# VectorMPI uses VectorMPIStyle
Base.BroadcastStyle(::Type{<:VectorMPI}) = VectorMPIStyle()

# VectorMPI is its own broadcastable representation (don't try to iterate)
Base.Broadcast.broadcastable(v::VectorMPI) = v

# Define axes for VectorMPI (needed for broadcast)
Base.axes(v::VectorMPI) = (Base.OneTo(length(v)),)

# VectorMPIStyle wins over DefaultArrayStyle for scalars and regular arrays
Base.BroadcastStyle(::VectorMPIStyle, ::DefaultArrayStyle{0}) = VectorMPIStyle()
Base.BroadcastStyle(::VectorMPIStyle, ::DefaultArrayStyle{N}) where N = VectorMPIStyle()

# Two VectorMPI => VectorMPIStyle
Base.BroadcastStyle(::VectorMPIStyle, ::VectorMPIStyle) = VectorMPIStyle()

"""
    _find_vectormpi(args...)

Find the first VectorMPI in a tuple of broadcast arguments.
Recursively searches through nested Broadcasted objects.
"""
_find_vectormpi(v::VectorMPI, args...) = v
function _find_vectormpi(bc::Broadcasted, args...)
    # Search in nested Broadcasted
    result = _find_vectormpi(bc.args...)
    if result !== nothing
        return result
    end
    return _find_vectormpi(args...)
end
_find_vectormpi(::Any, args...) = _find_vectormpi(args...)
_find_vectormpi() = nothing

"""
    _find_all_vectormpi(args...)

Find all VectorMPI arguments and return them as a tuple.
"""
_find_all_vectormpi(args::Tuple) = _find_all_vectormpi_impl(args...)
_find_all_vectormpi_impl() = ()
_find_all_vectormpi_impl(v::VectorMPI, args...) = (v, _find_all_vectormpi_impl(args...)...)
_find_all_vectormpi_impl(::Any, args...) = _find_all_vectormpi_impl(args...)

"""
    _check_same_partition(vs::Tuple)

Check that all VectorMPI in the tuple have the same partition.
Returns true if all partitions match, false otherwise.
"""
function _check_same_partition(vs::Tuple)
    isempty(vs) && return true
    first_partition = vs[1].partition
    for v in vs[2:end]
        if v.partition != first_partition
            return false
        end
    end
    return true
end

"""
    _prepare_broadcast_arg(arg, ref_partition, comm)

Prepare a broadcast argument for local computation.
- VectorMPI with same partition: return local vector
- VectorMPI with different partition: align to ref_partition
- Nested Broadcasted: recursively prepare and materialize
- Scalar or other: return as-is
"""
function _prepare_broadcast_arg(v::VectorMPI, ref_partition, comm)
    if v.partition == ref_partition
        return v.v
    else
        # Align to reference partition using existing alignment infrastructure
        aligned = _align_vector(v, ref_partition, comm)
        return aligned
    end
end

# Handle nested Broadcasted objects by recursively preparing their arguments
function _prepare_broadcast_arg(bc::Broadcasted{VectorMPIStyle}, ref_partition, comm)
    # Recursively prepare nested arguments
    prepared_args = map(arg -> _prepare_broadcast_arg(arg, ref_partition, comm), bc.args)
    # Return a new Broadcasted with prepared (local) arguments
    return Broadcasted{Nothing}(bc.f, prepared_args)
end

# Handle Broadcasted with other styles (e.g., scalar operations nested)
function _prepare_broadcast_arg(bc::Broadcasted, ref_partition, comm)
    # Recursively prepare nested arguments
    prepared_args = map(arg -> _prepare_broadcast_arg(arg, ref_partition, comm), bc.args)
    # Return a new Broadcasted with prepared arguments
    return Broadcasted{Nothing}(bc.f, prepared_args)
end

# Handle Base.RefValue (used in literal_pow for things like x.^2)
_prepare_broadcast_arg(r::Base.RefValue, ref_partition, comm) = r

_prepare_broadcast_arg(x, ref_partition, comm) = x

"""
    _get_broadcast_align_plan(target_partition::Vector{Int}, source::VectorMPI{T}) where T

Get or create a cached VectorPlan for aligning `source` to `target_partition`.
Uses point-to-point Isend/Irecv communication for efficiency.
"""
function _get_broadcast_align_plan(target_partition::Vector{Int}, source::VectorMPI{T}) where T
    target_hash = compute_partition_hash(target_partition)
    key = (target_hash, source.structural_hash, T)

    if haskey(_vector_align_plan_cache, key)
        return _vector_align_plan_cache[key]::VectorPlan{T}
    end

    # Create new plan using existing VectorPlan infrastructure
    plan = VectorPlan(target_partition, source)
    _vector_align_plan_cache[key] = plan
    return plan
end

"""
    _align_vector(v::VectorMPI{T}, target_partition, comm) where T

Align a VectorMPI to a target partition, redistributing elements as needed.
Uses cached communication plans with point-to-point Isend/Irecv for efficiency.
Returns the local portion of the aligned vector.
"""
function _align_vector(v::VectorMPI{T}, target_partition, comm) where T
    # Get or create cached plan
    plan = _get_broadcast_align_plan(target_partition, v)

    # Execute the plan (uses Isend/Irecv internally)
    execute_plan!(plan, v)

    # Return a copy of the gathered data (plan.gathered is reused)
    return copy(plan.gathered)
end

"""
    Base.similar(bc::Broadcasted{VectorMPIStyle}, ::Type{ElType}) where ElType

Allocate output array for VectorMPI broadcast.
"""
function Base.similar(bc::Broadcasted{VectorMPIStyle}, ::Type{ElType}) where ElType
    # Find a VectorMPI to get partition info
    v = _find_vectormpi(bc.args...)
    if v === nothing
        error("No VectorMPI found in broadcast arguments")
    end
    # Create output with same partition
    return VectorMPI{ElType}(v.structural_hash, copy(v.partition), Vector{ElType}(undef, length(v.v)))
end

"""
    Base.copyto!(dest::VectorMPI, bc::Broadcasted{VectorMPIStyle})

Execute the broadcast operation and store results in dest.
"""
function Base.copyto!(dest::VectorMPI, bc::Broadcasted{VectorMPIStyle})
    comm = MPI.COMM_WORLD

    # Find all VectorMPI arguments
    all_vmpi = _find_all_vectormpi(bc.args)

    # Use the destination's partition as reference
    ref_partition = dest.partition

    # Prepare all arguments (align VectorMPI to ref_partition, pass others through)
    prepared_args = map(arg -> _prepare_broadcast_arg(arg, ref_partition, comm), bc.args)

    # Perform local broadcast
    local_bc = Broadcasted{Nothing}(bc.f, prepared_args, axes(dest.v))
    copyto!(dest.v, local_bc)

    return dest
end

# Convenience: allow broadcast assignment to existing VectorMPI
function Base.materialize!(dest::VectorMPI, bc::Broadcasted{VectorMPIStyle})
    return copyto!(dest, instantiate(bc))
end
