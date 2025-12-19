# Indexing operations for distributed types
# Communication tags: 40 (index request), 41 (value response)

"""
    _invalidate_cached_transpose!(A::SparseMatrixMPI)

Invalidate the cached transpose of A bidirectionally.
If A has a cached transpose B, then B.cached_transpose is also cleared.
"""
function _invalidate_cached_transpose!(A::SparseMatrixMPI)
    if A.cached_transpose !== nothing
        A.cached_transpose.cached_transpose = nothing
    end
    A.cached_transpose = nothing
end

# ============================================================================
# VectorMPI Indexing
# ============================================================================

"""
    Base.getindex(v::VectorMPI{T}, i::Integer) where T

Get element `v[i]` from a distributed vector.

This is a collective operation - all ranks must call it with the same index.
The owning rank broadcasts the value to all others.

# Example
```julia
v = VectorMPI([1.0, 2.0, 3.0, 4.0])
x = v[2]  # All ranks get 2.0
```
"""
function Base.getindex(v::VectorMPI{T}, i::Integer) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    n = length(v)
    if i < 1 || i > n
        error("VectorMPI index out of bounds: i=$i, length=$n")
    end

    # Find owner using binary search on partition
    owner = searchsortedlast(v.partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Use a 1-element buffer for broadcast (MPI.Bcast! requires arrays)
    buf = Vector{T}(undef, 1)

    # Owner extracts the value into the buffer
    if rank == owner
        local_i = i - v.partition[owner + 1] + 1
        buf[1] = v.v[local_i]
    end

    # Broadcast from owner to all ranks
    MPI.Bcast!(buf, owner, comm)

    return buf[1]
end

"""
    Base.setindex!(v::VectorMPI{T}, val, i::Integer) where T

Set element `v[i] = val` in a distributed vector.

This is a collective operation - all ranks must call it with the same index and value.
Only the owning rank actually modifies the data.

# Example
```julia
v = VectorMPI([1.0, 2.0, 3.0, 4.0])
v[2] = 5.0  # All ranks must call this
```
"""
function Base.setindex!(v::VectorMPI{T}, val, i::Integer) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    n = length(v)
    if i < 1 || i > n
        error("VectorMPI index out of bounds: i=$i, length=$n")
    end

    # Find owner using binary search on partition
    owner = searchsortedlast(v.partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Only owner modifies the value
    if rank == owner
        local_i = i - v.partition[owner + 1] + 1
        v.v[local_i] = convert(T, val)
    end

    return val
end

# ============================================================================
# SparseMatrixMPI Indexing
# ============================================================================

"""
    Base.getindex(A::SparseMatrixMPI{T}, i::Integer, j::Integer) where T

Get element `A[i, j]` from a distributed sparse matrix.

This is a collective operation - all ranks must call it with the same indices.
The owning rank (owner of row i) broadcasts the value to all others.

Returns zero for structural zeros (entries not in the sparsity pattern).

# Example
```julia
A = SparseMatrixMPI{Float64}(sparse([1, 2], [1, 2], [1.0, 2.0], 3, 3))
x = A[1, 1]  # All ranks get 1.0
y = A[1, 2]  # All ranks get 0.0 (structural zero)
```
"""
function Base.getindex(A::SparseMatrixMPI{T}, i::Integer, j::Integer) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    if i < 1 || i > m
        error("SparseMatrixMPI row index out of bounds: i=$i, nrows=$m")
    end
    if j < 1 || j > n
        error("SparseMatrixMPI column index out of bounds: j=$j, ncols=$n")
    end

    # Find owner of row i using binary search on row_partition
    owner = searchsortedlast(A.row_partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Use a 1-element buffer for broadcast (MPI.Bcast! requires arrays)
    buf = Vector{T}(undef, 1)

    # Owner extracts the value into the buffer
    if rank == owner
        local_row = i - A.row_partition[owner + 1] + 1
        AT = A.A.parent  # The underlying CSC storage

        # Find j in col_indices using binary search (col_indices is sorted)
        local_col_idx = searchsortedfirst(A.col_indices, j)

        if local_col_idx <= length(A.col_indices) && A.col_indices[local_col_idx] == j
            # Column j exists in our sparsity pattern - use direct CSC indexing
            # AT[row, col] returns 0 for structural zeros automatically
            buf[1] = AT[local_col_idx, local_row]
        else
            buf[1] = zero(T)  # Column j not in our sparsity pattern at all
        end
    end

    # Broadcast from owner to all ranks
    MPI.Bcast!(buf, owner, comm)

    return buf[1]
end

"""
    Base.setindex!(A::SparseMatrixMPI{T}, val, i::Integer, j::Integer) where T

Set element `A[i, j] = val` in a distributed sparse matrix.

This is a collective operation - all ranks must call it, but each rank may specify
different (i, j, val) values. Only the rank that owns row i will actually apply
the modification; other ranks ignore their (i, j, val) locally.

This means if all ranks call `A[i, j] = val` with the same arguments, the value
will be set correctly. If different ranks specify different (i, j), only the
owners of each respective row will apply their modifications.

If (i, j) is not in the current sparsity pattern, a new structural entry is created.
Setting A[i, j] = 0 creates an explicit zero entry (structural modification).

After structural modifications:
- col_indices may expand if j is a new column for the owning rank
- The internal CSC storage is rebuilt
- structural_hash is recomputed (collective)
- cached_transpose is invalidated
- Plan caches referencing the old hash are cleaned

# Example
```julia
A = SparseMatrixMPI{Float64}(sparse([1, 2], [1, 2], [1.0, 2.0], 3, 3))
A[1, 1] = 5.0  # Modify existing entry (only owner of row 1 applies this)
A[1, 3] = 2.5  # Add new structural entry (only owner of row 1 applies this)
```
"""
function Base.setindex!(A::SparseMatrixMPI{T}, val, i::Integer, j::Integer) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    if i < 1 || i > m
        error("SparseMatrixMPI row index out of bounds: i=$i, nrows=$m")
    end
    if j < 1 || j > n
        error("SparseMatrixMPI column index out of bounds: j=$j, ncols=$n")
    end

    # Find owner of row i
    owner = searchsortedlast(A.row_partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Check if this rank owns the row - if not, we only participate in collective ops
    i_own_row = (rank == owner)

    # Check if column j exists in col_indices (determines if we need structural rebuild)
    needs_col_expansion = false
    if i_own_row
        local_row = i - A.row_partition[owner + 1] + 1
        AT = A.A.parent

        local_col_idx = searchsortedfirst(A.col_indices, j)
        if local_col_idx <= length(A.col_indices) && A.col_indices[local_col_idx] == j
            # Column exists - SparseMatrixCSC handles both existing and new entries
            AT[local_col_idx, local_row] = convert(T, val)
        else
            # New column needed - requires expanding col_indices
            needs_col_expansion = true
        end
    end

    # Check if ANY rank needs column expansion (collective)
    any_expansion_buf = Int32[needs_col_expansion ? 1 : 0]
    MPI.Allreduce!(any_expansion_buf, max, comm)

    if any_expansion_buf[1] == 0
        # No column expansions needed - we're done
        return val
    end

    # Column expansion required by at least one rank - rebuild with new column
    if i_own_row && needs_col_expansion
        row_offset = A.row_partition[rank + 1]
        received_mods = [(i, j, convert(T, val))]
        new_AT, new_col_indices = _rebuild_AT_with_insertions(
            A.A.parent, A.col_indices, received_mods, row_offset
        )

        # Update the struct fields
        A.col_indices = new_col_indices
        A.A = transpose(new_AT)
    end

    # Step 4: Invalidate structural hash (will be recomputed lazily on next use)
    A.structural_hash = nothing

    # Step 5: Invalidate cached transpose bidirectionally
    _invalidate_cached_transpose!(A)

    return val
end

# ============================================================================
# MatrixMPI Indexing
# ============================================================================

"""
    Base.getindex(A::MatrixMPI{T}, i::Integer, j::Integer) where T

Get element `A[i, j]` from a distributed dense matrix.

This is a collective operation - all ranks must call it with the same indices.
The owning rank (owner of row i) broadcasts the value to all others.

# Example
```julia
A = MatrixMPI([1.0 2.0; 3.0 4.0; 5.0 6.0])
x = A[2, 1]  # All ranks get 3.0
```
"""
function Base.getindex(A::MatrixMPI{T}, i::Integer, j::Integer) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    if i < 1 || i > m
        error("MatrixMPI row index out of bounds: i=$i, nrows=$m")
    end
    if j < 1 || j > n
        error("MatrixMPI column index out of bounds: j=$j, ncols=$n")
    end

    # Find owner of row i using binary search on row_partition
    owner = searchsortedlast(A.row_partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Use a 1-element buffer for broadcast (MPI.Bcast! requires arrays)
    buf = Vector{T}(undef, 1)

    # Owner extracts the value into the buffer
    if rank == owner
        local_row = i - A.row_partition[owner + 1] + 1
        buf[1] = A.A[local_row, j]
    end

    # Broadcast from owner to all ranks
    MPI.Bcast!(buf, owner, comm)

    return buf[1]
end

"""
    Base.setindex!(A::MatrixMPI{T}, val, i::Integer, j::Integer) where T

Set element `A[i, j] = val` in a distributed dense matrix.

This is a collective operation - all ranks must call it with the same indices and value.
Only the owning rank (owner of row i) actually modifies the data.

# Example
```julia
A = MatrixMPI([1.0 2.0; 3.0 4.0; 5.0 6.0])
A[2, 1] = 10.0  # All ranks must call this
```
"""
function Base.setindex!(A::MatrixMPI{T}, val, i::Integer, j::Integer) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    if i < 1 || i > m
        error("MatrixMPI row index out of bounds: i=$i, nrows=$m")
    end
    if j < 1 || j > n
        error("MatrixMPI column index out of bounds: j=$j, ncols=$n")
    end

    # Find owner of row i using binary search on row_partition
    owner = searchsortedlast(A.row_partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Only owner modifies the value
    if rank == owner
        local_row = i - A.row_partition[owner + 1] + 1
        A.A[local_row, j] = convert(T, val)
    end

    return val
end

# ============================================================================
# Range Indexing for VectorMPI
# ============================================================================

"""
    _compute_subpartition(partition::Vector{Int}, rng::UnitRange{Int})

Compute a new partition for a subrange of a distributed object.

Given a partition over global indices and a range `rng`, returns a new partition
where each rank's portion is the intersection of its original portion with `rng`,
mapped to local indices starting at 1.

This computation is local (no communication needed) since all ranks have the same partition.
"""
function _compute_subpartition(partition::Vector{Int}, rng::UnitRange{Int})
    nranks = length(partition) - 1
    new_partition = Vector{Int}(undef, nranks + 1)
    new_partition[1] = 1

    for r in 1:nranks
        # Rank r owns global indices partition[r]:(partition[r+1]-1)
        rank_start = partition[r]
        rank_end = partition[r + 1] - 1

        # Intersection with rng
        intersect_start = max(rank_start, first(rng))
        intersect_end = min(rank_end, last(rng))

        if intersect_start <= intersect_end
            count = intersect_end - intersect_start + 1
        else
            count = 0
        end

        new_partition[r + 1] = new_partition[r] + count
    end

    return new_partition
end

"""
    Base.getindex(v::VectorMPI{T}, rng::UnitRange{Int}) where T

Extract a subvector `v[rng]` from a distributed vector, returning a new VectorMPI.

This is a collective operation - all ranks must call it with the same range.
The result has a partition derived from `v.partition` such that each rank
extracts only its local portion (no data communication, only hash computation).

# Example
```julia
v = VectorMPI([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
w = v[3:6]  # Returns VectorMPI with elements [3.0, 4.0, 5.0, 6.0]
```
"""
function Base.getindex(v::VectorMPI{T}, rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    n = length(v)
    if first(rng) < 1 || last(rng) > n
        error("VectorMPI range out of bounds: $rng, length=$n")
    end

    if isempty(rng)
        # Empty range - return empty VectorMPI
        new_partition = ones(Int, MPI.Comm_size(comm) + 1)
        hash = compute_partition_hash(new_partition)
        return VectorMPI{T}(hash, new_partition, T[])
    end

    # Compute new partition (local computation, no communication)
    new_partition = _compute_subpartition(v.partition, rng)

    # Extract local portion
    my_start = v.partition[rank + 1]
    my_end = v.partition[rank + 2] - 1

    # Intersection of my range with rng
    intersect_start = max(my_start, first(rng))
    intersect_end = min(my_end, last(rng))

    if intersect_start <= intersect_end
        # Convert to local indices in v.v
        local_start = intersect_start - my_start + 1
        local_end = intersect_end - my_start + 1
        local_v = v.v[local_start:local_end]
    else
        local_v = T[]
    end

    # Compute hash (requires Allgather for consistency)
    hash = compute_partition_hash(new_partition)

    return VectorMPI{T}(hash, new_partition, local_v)
end

"""
    Base.setindex!(v::VectorMPI{T}, vals, rng::UnitRange{Int}) where T

Set elements `v[rng] = vals` in a distributed vector.

This is a collective operation - all ranks must call it with the same range.
`vals` can be:
- A scalar (broadcast to all positions)
- A VectorMPI with compatible length
- A regular Vector (must have length equal to the range)

Only ranks that own elements in the range modify their local data.

# Example
```julia
v = VectorMPI([1.0, 2.0, 3.0, 4.0])
v[2:3] = 0.0  # Set elements 2 and 3 to zero
v[2:3] = [5.0, 6.0]  # Set elements 2 and 3 to 5.0 and 6.0
```
"""
function Base.setindex!(v::VectorMPI{T}, val::Number, rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    n = length(v)
    if first(rng) < 1 || last(rng) > n
        error("VectorMPI range out of bounds: $rng, length=$n")
    end

    if isempty(rng)
        return val
    end

    # Find intersection with my portion
    my_start = v.partition[rank + 1]
    my_end = v.partition[rank + 2] - 1

    intersect_start = max(my_start, first(rng))
    intersect_end = min(my_end, last(rng))

    if intersect_start <= intersect_end
        # Convert to local indices
        local_start = intersect_start - my_start + 1
        local_end = intersect_end - my_start + 1
        v.v[local_start:local_end] .= convert(T, val)
    end

    return val
end

function Base.setindex!(v::VectorMPI{T}, vals::AbstractVector, rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    n = length(v)
    if first(rng) < 1 || last(rng) > n
        error("VectorMPI range out of bounds: $rng, length=$n")
    end

    if length(vals) != length(rng)
        error("VectorMPI setindex!: length mismatch, got $(length(vals)) values for range of length $(length(rng))")
    end

    if isempty(rng)
        return vals
    end

    # Find intersection with my portion
    my_start = v.partition[rank + 1]
    my_end = v.partition[rank + 2] - 1

    intersect_start = max(my_start, first(rng))
    intersect_end = min(my_end, last(rng))

    if intersect_start <= intersect_end
        # Convert to local indices in v.v
        local_start = intersect_start - my_start + 1
        local_end = intersect_end - my_start + 1
        # Indices into vals
        vals_start = intersect_start - first(rng) + 1
        vals_end = intersect_end - first(rng) + 1
        v.v[local_start:local_end] .= convert.(T, vals[vals_start:vals_end])
    end

    return vals
end

"""
    Base.setindex!(v::VectorMPI{T}, src::VectorMPI, rng::UnitRange{Int}) where T

Set elements `v[rng] = src` where `src` is a distributed VectorMPI.

This is a collective operation - all ranks must call it with the same `rng` and `src`.
Each rank only updates the elements it owns that fall within `rng`.

If the source partition matches the target partition induced by `rng`, a direct
local copy is performed. Otherwise, a communication plan redistributes the source values.

# Example
```julia
v = VectorMPI([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
src = VectorMPI([10.0, 20.0, 30.0])
v[2:4] = src  # Each rank only writes to elements it owns
```
"""
function Base.setindex!(v::VectorMPI{T}, src::VectorMPI, rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    n = length(v)
    if first(rng) < 1 || last(rng) > n
        error("VectorMPI range out of bounds: $rng, length=$n")
    end

    if length(src) != length(rng)
        error("VectorMPI setindex!: length mismatch, got $(length(src)) values for range of length $(length(rng))")
    end

    if isempty(rng)
        return src
    end

    # Compute my owned range intersection with rng
    my_start = v.partition[rank + 1]
    my_end = v.partition[rank + 2] - 1

    intersect_start = max(my_start, first(rng))
    intersect_end = min(my_end, last(rng))

    # Compute the partition of the target range
    target_partition = _compute_subpartition(v.partition, rng)

    # Check if source partition matches target partition
    if src.partition == target_partition
        # Direct local copy - each rank copies its local portion
        if intersect_start <= intersect_end
            local_start = intersect_start - my_start + 1
            local_end = intersect_end - my_start + 1
            v.v[local_start:local_end] .= convert.(T, src.v)
        end
    else
        # Need to align src to target_partition
        plan = VectorPlan(target_partition, src)
        aligned = execute_plan!(plan, src)

        if intersect_start <= intersect_end
            local_start = intersect_start - my_start + 1
            local_end = intersect_end - my_start + 1
            v.v[local_start:local_end] .= convert.(T, aligned)
        end
    end

    return src
end

# ============================================================================
# Range Indexing for MatrixMPI
# ============================================================================

"""
    Base.getindex(A::MatrixMPI{T}, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T

Extract a submatrix `A[row_rng, col_rng]` from a distributed dense matrix, returning a new MatrixMPI.

This is a collective operation - all ranks must call it with the same ranges.
The result has a row partition derived from `A.row_partition` such that each rank
extracts only its local portion (no data communication, only hash computation).

# Example
```julia
A = MatrixMPI(reshape(1.0:12.0, 4, 3))
B = A[2:3, 1:2]  # Returns MatrixMPI submatrix
```
"""
function Base.getindex(A::MatrixMPI{T}, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("MatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("MatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    new_nrows = length(row_rng)
    new_ncols = length(col_rng)

    if isempty(row_rng) || isempty(col_rng)
        # Empty range - return empty MatrixMPI with correct dimensions
        new_row_partition = uniform_partition(new_nrows, nranks)
        new_col_partition = uniform_partition(new_ncols, nranks)
        my_local_rows = new_row_partition[rank + 2] - new_row_partition[rank + 1]
        hash = compute_dense_structural_hash(new_row_partition, new_col_partition, (new_nrows, new_ncols), comm)
        return MatrixMPI{T}(hash, new_row_partition, new_col_partition, Matrix{T}(undef, my_local_rows, new_ncols))
    end

    # Compute new row partition (local computation, no communication)
    new_row_partition = _compute_subpartition(A.row_partition, row_rng)

    # Compute new column partition (standard even distribution for the submatrix column count)
    new_col_partition = uniform_partition(new_ncols, nranks)

    # Extract local portion
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    # Intersection of my row range with row_rng
    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    if intersect_start <= intersect_end
        # Convert to local row indices in A.A
        local_row_start = intersect_start - my_row_start + 1
        local_row_end = intersect_end - my_row_start + 1
        local_A = A.A[local_row_start:local_row_end, col_rng]
    else
        local_A = Matrix{T}(undef, 0, new_ncols)
    end

    # Compute hash (requires Allgather for consistency)
    hash = compute_dense_structural_hash(new_row_partition, new_col_partition, size(local_A), comm)

    return MatrixMPI{T}(hash, new_row_partition, new_col_partition, local_A)
end

# Convenience: A[row_rng, :] - all columns
function Base.getindex(A::MatrixMPI{T}, row_rng::UnitRange{Int}, ::Colon) where T
    return A[row_rng, 1:size(A, 2)]
end

# Convenience: A[:, col_rng] - all rows
function Base.getindex(A::MatrixMPI{T}, ::Colon, col_rng::UnitRange{Int}) where T
    return A[1:size(A, 1), col_rng]
end

# Convenience: A[:, :] - full copy
function Base.getindex(A::MatrixMPI{T}, ::Colon, ::Colon) where T
    return A[1:size(A, 1), 1:size(A, 2)]
end

"""
    Base.getindex(A::MatrixMPI{T}, ::Colon, k::Integer) where T

Extract column k from a distributed dense matrix as a VectorMPI.

This is a collective operation - all ranks must call it.
Each rank extracts its local portion of the column.

# Example
```julia
A = MatrixMPI(reshape(1.0:12.0, 4, 3))
v = A[:, 2]  # Get second column as VectorMPI
```
"""
function Base.getindex(A::MatrixMPI{T}, ::Colon, k::Integer) where T
    m, n = size(A)
    if k < 1 || k > n
        error("MatrixMPI column index out of bounds: k=$k, ncols=$n")
    end
    # Extract local portion of column k
    local_col = A.A[:, k]
    return VectorMPI_local(local_col)
end

"""
    Base.setindex!(A::MatrixMPI{T}, val, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T

Set elements `A[row_rng, col_rng] = val` in a distributed dense matrix.

This is a collective operation - all ranks must call it with the same ranges.
`val` can be a scalar (broadcast to all positions) or a matrix of matching size.

# Example
```julia
A = MatrixMPI(reshape(1.0:12.0, 4, 3))
A[2:3, 1:2] = 0.0  # Set submatrix to zeros
A[2:3, 1:2] = [5.0 6.0; 7.0 8.0]  # Set submatrix to specific values
```
"""
function Base.setindex!(A::MatrixMPI{T}, val::Number, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("MatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("MatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    if isempty(row_rng) || isempty(col_rng)
        return val
    end

    # Find intersection with my row portion
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    if intersect_start <= intersect_end
        local_row_start = intersect_start - my_row_start + 1
        local_row_end = intersect_end - my_row_start + 1
        A.A[local_row_start:local_row_end, col_rng] .= convert(T, val)
    end

    return val
end

function Base.setindex!(A::MatrixMPI{T}, vals::AbstractMatrix, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("MatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("MatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    if size(vals) != (length(row_rng), length(col_rng))
        error("MatrixMPI setindex!: size mismatch, got $(size(vals)) for range of size ($(length(row_rng)), $(length(col_rng)))")
    end

    if isempty(row_rng) || isempty(col_rng)
        return vals
    end

    # Find intersection with my row portion
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    if intersect_start <= intersect_end
        local_row_start = intersect_start - my_row_start + 1
        local_row_end = intersect_end - my_row_start + 1
        # Indices into vals
        vals_row_start = intersect_start - first(row_rng) + 1
        vals_row_end = intersect_end - first(row_rng) + 1
        A.A[local_row_start:local_row_end, col_rng] .= convert.(T, vals[vals_row_start:vals_row_end, :])
    end

    return vals
end

"""
    Base.setindex!(A::MatrixMPI{T}, src::MatrixMPI, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T

Set elements `A[row_rng, col_rng] = src` where `src` is a distributed MatrixMPI.

This is a collective operation - all ranks must call it with the same arguments.
Each rank only updates the rows it owns that fall within `row_rng`.

If the source row partition matches the target partition induced by `row_rng`, a direct
local copy is performed. Otherwise, point-to-point communication redistributes the source rows.

# Example
```julia
A = MatrixMPI(zeros(6, 4))
src = MatrixMPI(ones(3, 2))
A[2:4, 1:2] = src  # Each rank only writes to rows it owns
```
"""
function Base.setindex!(A::MatrixMPI{T}, src::MatrixMPI, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("MatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("MatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    if size(src) != (length(row_rng), length(col_rng))
        error("MatrixMPI setindex!: size mismatch, got $(size(src)) for range of size ($(length(row_rng)), $(length(col_rng)))")
    end

    if isempty(row_rng) || isempty(col_rng)
        return src
    end

    ncols_src = length(col_rng)

    # Compute my owned row intersection with row_rng
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    # Compute the partition of the target range
    target_row_partition = _compute_subpartition(A.row_partition, row_rng)

    # Check if source partition matches target partition
    if src.row_partition == target_row_partition
        # Direct local copy - each rank copies its local portion
        if intersect_start <= intersect_end
            local_row_start = intersect_start - my_row_start + 1
            local_row_end = intersect_end - my_row_start + 1
            A.A[local_row_start:local_row_end, col_rng] .= convert.(T, src.A[:, 1:ncols_src])
        end
    else
        # Partitions don't match - need communication to redistribute rows
        # For each row I need (in target_row_partition), find which rank owns it in src

        # Rows I need in global src indexing (1-based in src)
        my_target_start = target_row_partition[rank + 1]
        my_target_end = target_row_partition[rank + 2] - 1
        num_rows_needed = max(0, my_target_end - my_target_start + 1)

        # Build receive plan: which ranks will send me data
        recv_counts = zeros(Int, nranks)
        recv_row_ranges = Vector{UnitRange{Int}}(undef, nranks)  # global src indices from each rank

        for global_src_row in my_target_start:my_target_end
            src_owner = searchsortedlast(src.row_partition, global_src_row) - 1
            if src_owner >= nranks
                src_owner = nranks - 1
            end
            recv_counts[src_owner + 1] += 1
        end

        # Compute contiguous ranges from each rank
        for r in 0:(nranks-1)
            if recv_counts[r + 1] > 0
                # Find the range of global src indices that rank r owns and that I need
                src_r_start = src.row_partition[r + 1]
                src_r_end = src.row_partition[r + 2] - 1
                range_start = max(my_target_start, src_r_start)
                range_end = min(my_target_end, src_r_end)
                recv_row_ranges[r + 1] = range_start:range_end
            else
                recv_row_ranges[r + 1] = 1:0  # empty
            end
        end

        # Build send plan: which ranks need data from me
        send_counts = zeros(Int, nranks)
        send_row_ranges = Vector{UnitRange{Int}}(undef, nranks)

        my_src_start = src.row_partition[rank + 1]
        my_src_end = src.row_partition[rank + 2] - 1

        for r in 0:(nranks-1)
            # Rank r needs rows in target_row_partition[r+1]:target_row_partition[r+2]-1
            r_target_start = target_row_partition[r + 1]
            r_target_end = target_row_partition[r + 2] - 1

            # Intersection with rows I own in src
            range_start = max(r_target_start, my_src_start)
            range_end = min(r_target_end, my_src_end)

            if range_start <= range_end
                send_counts[r + 1] = range_end - range_start + 1
                send_row_ranges[r + 1] = range_start:range_end
            else
                send_row_ranges[r + 1] = 1:0  # empty
            end
        end

        # Post receives
        recv_reqs = MPI.Request[]
        recv_bufs = Dict{Int, Matrix{T}}()
        for r in 0:(nranks-1)
            if recv_counts[r + 1] > 0 && r != rank
                recv_bufs[r] = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
                push!(recv_reqs, MPI.Irecv!(recv_bufs[r], comm; source=r, tag=60))
            end
        end

        # Send data
        send_reqs = MPI.Request[]
        send_bufs = Dict{Int, Matrix{T}}()
        for r in 0:(nranks-1)
            if send_counts[r + 1] > 0 && r != rank
                rng = send_row_ranges[r + 1]
                local_start = first(rng) - my_src_start + 1
                local_end = last(rng) - my_src_start + 1
                send_bufs[r] = convert.(T, src.A[local_start:local_end, 1:ncols_src])
                push!(send_reqs, MPI.Isend(send_bufs[r], comm; dest=r, tag=60))
            end
        end

        # Wait for receives
        MPI.Waitall(recv_reqs)

        # Assemble the aligned local matrix
        if num_rows_needed > 0
            aligned = Matrix{T}(undef, num_rows_needed, ncols_src)

            for r in 0:(nranks-1)
                if recv_counts[r + 1] > 0
                    rng = recv_row_ranges[r + 1]
                    # Destination indices in aligned
                    dst_start = first(rng) - my_target_start + 1
                    dst_end = last(rng) - my_target_start + 1

                    if r == rank
                        # Local copy from src
                        local_start = first(rng) - my_src_start + 1
                        local_end = last(rng) - my_src_start + 1
                        aligned[dst_start:dst_end, :] .= src.A[local_start:local_end, 1:ncols_src]
                    else
                        # From received buffer
                        aligned[dst_start:dst_end, :] .= recv_bufs[r]
                    end
                end
            end

            # Write to A - only rows I own and that fall in row_rng
            if intersect_start <= intersect_end
                local_row_start = intersect_start - my_row_start + 1
                local_row_end = intersect_end - my_row_start + 1
                A.A[local_row_start:local_row_end, col_rng] .= convert.(T, aligned)
            end
        end

        # Wait for sends
        MPI.Waitall(send_reqs)
    end

    return src
end

# Convenience methods for setindex! with Colon
function Base.setindex!(A::MatrixMPI{T}, val, row_rng::UnitRange{Int}, ::Colon) where T
    return setindex!(A, val, row_rng, 1:size(A, 2))
end

function Base.setindex!(A::MatrixMPI{T}, val, ::Colon, col_rng::UnitRange{Int}) where T
    return setindex!(A, val, 1:size(A, 1), col_rng)
end

# ============================================================================
# Range Indexing for SparseMatrixMPI
# ============================================================================

"""
    Base.getindex(A::SparseMatrixMPI{T}, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T

Extract a submatrix `A[row_rng, col_rng]` from a distributed sparse matrix, returning a new SparseMatrixMPI.

This is a collective operation - all ranks must call it with the same ranges.
The result has a row partition derived from `A.row_partition` such that each rank
extracts only its local portion (no data communication, only hash computation).

# Example
```julia
A = SparseMatrixMPI{Float64}(sprand(10, 10, 0.3))
B = A[3:7, 2:8]  # Returns SparseMatrixMPI submatrix
```
"""
function Base.getindex(A::SparseMatrixMPI{T,Ti}, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where {T,Ti}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("SparseMatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("SparseMatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    new_nrows = length(row_rng)
    new_ncols = length(col_rng)

    if isempty(row_rng) || isempty(col_rng)
        # Empty range - return empty SparseMatrixMPI with correct dimensions
        new_row_partition = uniform_partition(new_nrows, nranks)
        new_col_partition = uniform_partition(new_ncols, nranks)
        my_local_rows = new_row_partition[rank + 2] - new_row_partition[rank + 1]
        # SparseMatrixCSC(ncols, nrows, colptr, rowval, nzval) - transposed storage
        empty_AT = SparseMatrixCSC(new_ncols, my_local_rows, ones(Int, my_local_rows + 1), Int[], T[])
        hash = compute_structural_hash(new_row_partition, Int[], empty_AT, comm)
        return SparseMatrixMPI{T,Ti}(hash, new_row_partition, new_col_partition, Int[],
                                   transpose(empty_AT), nothing, nothing)
    end

    # Compute new row partition (local computation, no communication)
    new_row_partition = _compute_subpartition(A.row_partition, row_rng)

    # Compute new column partition (standard even distribution)
    new_col_partition = uniform_partition(new_ncols, nranks)

    # Extract local portion
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    # Intersection of my row range with row_rng
    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    if intersect_start <= intersect_end
        # Number of local rows in the result
        local_nrows = intersect_end - intersect_start + 1

        # Local row indices (in A.A) that we're extracting
        local_row_start = intersect_start - my_row_start + 1
        local_row_end = intersect_end - my_row_start + 1

        # A.A is transpose(AT) where AT is CSC with columns = local rows
        AT = A.A.parent  # SparseMatrixCSC with columns = local rows

        # Build new sparse structure for the extracted rows
        # AT has columns for each local row, rows indexed by compressed col_indices
        new_colptr = Vector{Int}(undef, local_nrows + 1)
        new_colptr[1] = 1

        # First pass: count entries per extracted row that fall in col_rng
        # Map col_rng to local col_indices
        col_rng_start = first(col_rng)
        col_rng_end = last(col_rng)

        # Find which A.col_indices fall in col_rng
        col_mask = (A.col_indices .>= col_rng_start) .& (A.col_indices .<= col_rng_end)
        new_col_indices = A.col_indices[col_mask] .- (col_rng_start - 1)  # Shift to 1-based for new matrix

        # Build mapping from old local col index to new local col index
        old_to_new_col = zeros(Int, length(A.col_indices))
        new_idx = 1
        for (old_idx, in_range) in enumerate(col_mask)
            if in_range
                old_to_new_col[old_idx] = new_idx
                new_idx += 1
            end
        end

        # Count entries and collect data
        rowval_list = Int[]
        nzval_list = T[]

        for local_row in local_row_start:local_row_end
            col_start = AT.colptr[local_row]
            col_end = AT.colptr[local_row + 1] - 1

            count = 0
            for k in col_start:col_end
                old_col_idx = AT.rowval[k]
                new_col_idx = old_to_new_col[old_col_idx]
                if new_col_idx > 0
                    push!(rowval_list, new_col_idx)
                    push!(nzval_list, AT.nzval[k])
                    count += 1
                end
            end
            new_colptr[local_row - local_row_start + 2] = new_colptr[local_row - local_row_start + 1] + count
        end

        # Sort entries within each column by row index
        for local_row in 1:local_nrows
            start_idx = new_colptr[local_row]
            end_idx = new_colptr[local_row + 1] - 1
            if start_idx <= end_idx
                perm = sortperm(view(rowval_list, start_idx:end_idx))
                rowval_list[start_idx:end_idx] = rowval_list[start_idx:end_idx][perm]
                nzval_list[start_idx:end_idx] = nzval_list[start_idx:end_idx][perm]
            end
        end

        # Recompute col_indices to be just the columns that actually appear
        # rowval_list contains positions into new_col_indices
        if !isempty(rowval_list)
            unique_positions = sort(unique(rowval_list))
            # Map positions to compressed indices: unique_positions is sorted, use binary search
            compressed_rowval = [searchsortedfirst(unique_positions, r) for r in rowval_list]
            # final_col_indices maps compressed index to global column in result
            # new_col_indices contains the shifted global column indices
            final_col_indices = new_col_indices[unique_positions]
        else
            compressed_rowval = Int[]
            final_col_indices = Int[]
        end

        new_AT = SparseMatrixCSC(length(final_col_indices), local_nrows, new_colptr, compressed_rowval, nzval_list)
    else
        # No local rows in range
        local_nrows = 0
        new_AT = SparseMatrixCSC(0, 0, Int[1], Int[], T[])
        final_col_indices = Int[]
    end

    # Compute hash (requires Allgather for consistency)
    hash = compute_structural_hash(new_row_partition, final_col_indices, new_AT, comm)

    return SparseMatrixMPI{T,Ti}(hash, new_row_partition, new_col_partition, final_col_indices,
                               transpose(new_AT), nothing, nothing)
end

# Convenience: A[row_rng, :] - all columns
function Base.getindex(A::SparseMatrixMPI{T}, row_rng::UnitRange{Int}, ::Colon) where T
    return A[row_rng, 1:size(A, 2)]
end

# Convenience: A[:, col_rng] - all rows
function Base.getindex(A::SparseMatrixMPI{T}, ::Colon, col_rng::UnitRange{Int}) where T
    return A[1:size(A, 1), col_rng]
end

# Convenience: A[:, :] - full copy
function Base.getindex(A::SparseMatrixMPI{T}, ::Colon, ::Colon) where T
    return A[1:size(A, 1), 1:size(A, 2)]
end

"""
    Base.getindex(A::SparseMatrixMPI{T}, ::Colon, k::Integer) where T

Extract column k from a distributed sparse matrix as a VectorMPI.

This is a collective operation - all ranks must call it.
Each rank extracts its local portion of the column.

# Example
```julia
A = SparseMatrixMPI{Float64}(sprand(10, 5, 0.3))
v = A[:, 2]  # Get second column as VectorMPI
```
"""
function Base.getindex(A::SparseMatrixMPI{T}, ::Colon, k::Integer) where T
    m, n = size(A)
    if k < 1 || k > n
        error("SparseMatrixMPI column index out of bounds: k=$k, ncols=$n")
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Get local row range
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1
    local_nrows = my_row_end - my_row_start + 1

    # Check if column k is in our col_indices
    local_col_idx = searchsortedfirst(A.col_indices, k)
    has_col = local_col_idx <= length(A.col_indices) && A.col_indices[local_col_idx] == k

    # Build local portion of column
    local_col = zeros(T, local_nrows)

    if has_col && local_nrows > 0
        # A.A.parent is CSC with shape (length(col_indices), local_nrows)
        # Row indices in A.A.parent.rowval are local column indices
        # Column indices in A.A.parent are local row indices
        parent = A.A.parent
        for local_row in 1:local_nrows
            # Iterate over nonzeros in this row (stored as column in parent)
            for idx in parent.colptr[local_row]:(parent.colptr[local_row+1]-1)
                if parent.rowval[idx] == local_col_idx
                    local_col[local_row] = parent.nzval[idx]
                    break
                end
            end
        end
    end

    return VectorMPI_local(local_col)
end

"""
    Base.setindex!(A::SparseMatrixMPI{T}, val::Number, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T

Set elements `A[row_rng, col_rng] = val` in a distributed sparse matrix.

This is a collective operation - all ranks must call it with the same ranges.
Only existing structural nonzeros within the range are modified.
If val is zero, existing entries become explicit zeros (structure is preserved).

# Example
```julia
A = SparseMatrixMPI{Float64}(sprand(10, 10, 0.3))
A[2:4, 3:5] = 0.0  # Set all structural nonzeros in range to zero
A[2:4, 3:5] = 2.0  # Set all structural nonzeros in range to 2.0
```
"""
function Base.setindex!(A::SparseMatrixMPI{T}, val::Number, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("SparseMatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("SparseMatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    if isempty(row_rng) || isempty(col_rng)
        return val
    end

    # Find intersection with my row portion
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    if intersect_start <= intersect_end
        local_row_start = intersect_start - my_row_start + 1
        local_row_end = intersect_end - my_row_start + 1

        AT = A.A.parent
        col_rng_start = first(col_rng)
        col_rng_end = last(col_rng)

        # Iterate over local rows in the range
        for local_row in local_row_start:local_row_end
            col_start = AT.colptr[local_row]
            col_end = AT.colptr[local_row + 1] - 1

            for k in col_start:col_end
                local_col_idx = AT.rowval[k]
                global_col = A.col_indices[local_col_idx]
                if global_col >= col_rng_start && global_col <= col_rng_end
                    AT.nzval[k] = convert(T, val)
                end
            end
        end
    end

    # Invalidate cached transpose bidirectionally (values changed)
    _invalidate_cached_transpose!(A)

    return val
end

"""
    Base.setindex!(A::SparseMatrixMPI{T}, src::SparseMatrixMPI{T}, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T

Set elements `A[row_rng, col_rng] = src` where `src` is a distributed SparseMatrixMPI.

This is a collective operation - all ranks must call it with the same arguments.
Each rank only updates the rows it owns that fall within `row_rng`.

The source sparse matrix values replace the corresponding region in A. This is a
structural modification - new nonzeros from src are added, and the sparsity pattern
of A in the target region is replaced by src's pattern.

After the operation:
- col_indices may expand if src contains columns not in A
- The internal CSC storage is rebuilt for affected rows
- structural_hash is recomputed (collective)
- cached_transpose is invalidated
- Plan caches referencing the old hash are cleaned

# Example
```julia
A = SparseMatrixMPI{Float64}(spzeros(6, 6))
src = SparseMatrixMPI{Float64}(sparse([1, 2], [1, 2], [1.0, 2.0], 3, 3))
A[2:4, 1:3] = src  # Each rank only writes to rows it owns
```
"""
function Base.setindex!(A::SparseMatrixMPI{T}, src::SparseMatrixMPI{T}, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        error("SparseMatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        error("SparseMatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    if size(src) != (length(row_rng), length(col_rng))
        error("SparseMatrixMPI setindex!: size mismatch, got $(size(src)) for range of size ($(length(row_rng)), $(length(col_rng)))")
    end

    if isempty(row_rng) || isempty(col_rng)
        return src
    end

    # Compute my owned row intersection with row_rng
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    intersect_start = max(my_row_start, first(row_rng))
    intersect_end = min(my_row_end, last(row_rng))

    # Compute the partition of the target range (how src rows map to A rows)
    target_row_partition = _compute_subpartition(A.row_partition, row_rng)

    # Column offset for translating src column indices to A column indices
    col_offset = first(col_rng) - 1

    # Rows I need in global src indexing (1-based in src)
    my_target_start = target_row_partition[rank + 1]
    my_target_end = target_row_partition[rank + 2] - 1
    num_rows_needed = max(0, my_target_end - my_target_start + 1)

    # Check if partitions match (fast path)
    partitions_match = (src.row_partition == target_row_partition)

    # Build insertions from local or received src data
    insertions = Vector{Tuple{Int,Int,T}}()

    if partitions_match
        # Fast path: direct local extraction
        if num_rows_needed > 0 && intersect_start <= intersect_end
            src_AT = src.A.parent
            src_my_start = src.row_partition[rank + 1]

            for src_local_row in 1:num_rows_needed
                src_global_row = src_my_start + src_local_row - 1
                A_global_row = first(row_rng) + src_global_row - 1

                # Only process if this A row is owned by us
                if A_global_row >= intersect_start && A_global_row <= intersect_end
                    # Extract entries from src for this row
                    for k in src_AT.colptr[src_local_row]:(src_AT.colptr[src_local_row+1]-1)
                        src_local_col = src_AT.rowval[k]
                        src_global_col = src.col_indices[src_local_col]
                        A_global_col = src_global_col + col_offset
                        if A_global_col >= first(col_rng) && A_global_col <= last(col_rng)
                            push!(insertions, (A_global_row, A_global_col, src_AT.nzval[k]))
                        end
                    end
                end
            end
        end
    else
        # Slow path: need communication
        # Build receive plan
        recv_counts = zeros(Int, nranks)
        for global_src_row in my_target_start:my_target_end
            src_owner = searchsortedlast(src.row_partition, global_src_row) - 1
            if src_owner >= nranks
                src_owner = nranks - 1
            end
            recv_counts[src_owner + 1] += 1
        end

        # Compute contiguous row ranges from each rank
        recv_row_ranges = Vector{UnitRange{Int}}(undef, nranks)
        for r in 0:(nranks-1)
            if recv_counts[r + 1] > 0
                src_r_start = src.row_partition[r + 1]
                src_r_end = src.row_partition[r + 2] - 1
                range_start = max(my_target_start, src_r_start)
                range_end = min(my_target_end, src_r_end)
                recv_row_ranges[r + 1] = range_start:range_end
            else
                recv_row_ranges[r + 1] = 1:0
            end
        end

        # Build send plan
        my_src_start = src.row_partition[rank + 1]
        my_src_end = src.row_partition[rank + 2] - 1

        send_row_ranges = Vector{UnitRange{Int}}(undef, nranks)
        for r in 0:(nranks-1)
            r_target_start = target_row_partition[r + 1]
            r_target_end = target_row_partition[r + 2] - 1
            range_start = max(r_target_start, my_src_start)
            range_end = min(r_target_end, my_src_end)
            if range_start <= range_end
                send_row_ranges[r + 1] = range_start:range_end
            else
                send_row_ranges[r + 1] = 1:0
            end
        end

        # Exchange sparse data: for each row, send (num_entries, col_indices, values)
        # First exchange the number of nonzeros per row
        send_nnz_per_row = Dict{Int, Vector{Int}}()
        send_col_indices = Dict{Int, Vector{Int}}()
        send_values = Dict{Int, Vector{T}}()

        src_AT = src.A.parent

        for r in 0:(nranks-1)
            rng = send_row_ranges[r + 1]
            if !isempty(rng) && r != rank
                nnz_list = Int[]
                cols_list = Int[]
                vals_list = T[]
                for src_global_row in rng
                    src_local_row = src_global_row - my_src_start + 1
                    k_start = src_AT.colptr[src_local_row]
                    k_end = src_AT.colptr[src_local_row + 1] - 1
                    row_nnz = 0
                    for k in k_start:k_end
                        src_local_col = src_AT.rowval[k]
                        src_global_col = src.col_indices[src_local_col]
                        A_global_col = src_global_col + col_offset
                        if A_global_col >= first(col_rng) && A_global_col <= last(col_rng)
                            push!(cols_list, A_global_col)
                            push!(vals_list, src_AT.nzval[k])
                            row_nnz += 1
                        end
                    end
                    push!(nnz_list, row_nnz)
                end
                send_nnz_per_row[r] = nnz_list
                send_col_indices[r] = cols_list
                send_values[r] = vals_list
            end
        end

        # Post receives for nnz counts
        recv_reqs_nnz = MPI.Request[]
        recv_nnz_per_row = Dict{Int, Vector{Int}}()
        for r in 0:(nranks-1)
            if recv_counts[r + 1] > 0 && r != rank
                recv_nnz_per_row[r] = Vector{Int}(undef, recv_counts[r + 1])
                push!(recv_reqs_nnz, MPI.Irecv!(recv_nnz_per_row[r], comm; source=r, tag=70))
            end
        end

        # Send nnz counts
        send_reqs_nnz = MPI.Request[]
        for r in 0:(nranks-1)
            if haskey(send_nnz_per_row, r)
                push!(send_reqs_nnz, MPI.Isend(send_nnz_per_row[r], comm; dest=r, tag=70))
            end
        end

        MPI.Waitall(recv_reqs_nnz)

        # Post receives for col indices and values
        recv_reqs_data = MPI.Request[]
        recv_col_indices = Dict{Int, Vector{Int}}()
        recv_values = Dict{Int, Vector{T}}()
        for r in 0:(nranks-1)
            if recv_counts[r + 1] > 0 && r != rank
                total_nnz = sum(recv_nnz_per_row[r])
                if total_nnz > 0
                    recv_col_indices[r] = Vector{Int}(undef, total_nnz)
                    recv_values[r] = Vector{T}(undef, total_nnz)
                    push!(recv_reqs_data, MPI.Irecv!(recv_col_indices[r], comm; source=r, tag=71))
                    push!(recv_reqs_data, MPI.Irecv!(recv_values[r], comm; source=r, tag=72))
                end
            end
        end

        # Send col indices and values
        send_reqs_data = MPI.Request[]
        for r in 0:(nranks-1)
            if haskey(send_col_indices, r) && !isempty(send_col_indices[r])
                push!(send_reqs_data, MPI.Isend(send_col_indices[r], comm; dest=r, tag=71))
                push!(send_reqs_data, MPI.Isend(send_values[r], comm; dest=r, tag=72))
            end
        end

        MPI.Waitall(send_reqs_nnz)
        MPI.Waitall(recv_reqs_data)

        # Build insertions from received data and local data
        for r in 0:(nranks-1)
            if recv_counts[r + 1] > 0
                rng = recv_row_ranges[r + 1]
                if r == rank
                    # Local data
                    for src_global_row in rng
                        A_global_row = first(row_rng) + src_global_row - 1
                        if A_global_row >= intersect_start && A_global_row <= intersect_end
                            src_local_row = src_global_row - my_src_start + 1
                            for k in src_AT.colptr[src_local_row]:(src_AT.colptr[src_local_row+1]-1)
                                src_local_col = src_AT.rowval[k]
                                src_global_col = src.col_indices[src_local_col]
                                A_global_col = src_global_col + col_offset
                                if A_global_col >= first(col_rng) && A_global_col <= last(col_rng)
                                    push!(insertions, (A_global_row, A_global_col, src_AT.nzval[k]))
                                end
                            end
                        end
                    end
                elseif haskey(recv_nnz_per_row, r)
                    # Received data
                    nnz_list = recv_nnz_per_row[r]
                    if haskey(recv_col_indices, r)
                        cols = recv_col_indices[r]
                        vals = recv_values[r]
                        data_idx = 1
                        for (row_idx, src_global_row) in enumerate(rng)
                            A_global_row = first(row_rng) + src_global_row - 1
                            row_nnz = nnz_list[row_idx]
                            if A_global_row >= intersect_start && A_global_row <= intersect_end
                                for _ in 1:row_nnz
                                    push!(insertions, (A_global_row, cols[data_idx], vals[data_idx]))
                                    data_idx += 1
                                end
                            else
                                data_idx += row_nnz
                            end
                        end
                    end
                end
            end
        end

        MPI.Waitall(send_reqs_data)
    end

    # First, zero out existing entries in the target region (within owned rows)
    # This ensures src's sparsity pattern replaces the old one
    if intersect_start <= intersect_end
        AT = A.A.parent
        local_row_start = intersect_start - my_row_start + 1
        local_row_end = intersect_end - my_row_start + 1
        col_rng_start = first(col_rng)
        col_rng_end = last(col_rng)

        for local_row in local_row_start:local_row_end
            for k in AT.colptr[local_row]:(AT.colptr[local_row+1]-1)
                local_col_idx = AT.rowval[k]
                if local_col_idx <= length(A.col_indices)
                    global_col = A.col_indices[local_col_idx]
                    if global_col >= col_rng_start && global_col <= col_rng_end
                        AT.nzval[k] = zero(T)
                    end
                end
            end
        end
    end

    # Apply insertions using the helper function
    if !isempty(insertions)
        row_offset = A.row_partition[rank + 1]
        new_AT, new_col_indices = _rebuild_AT_with_insertions(
            A.A.parent, A.col_indices, insertions, row_offset
        )
        A.col_indices = new_col_indices
        A.A = transpose(new_AT)
    end

    # Invalidate structural hash (will be recomputed lazily on next use)
    A.structural_hash = nothing

    # Invalidate cached transpose bidirectionally
    _invalidate_cached_transpose!(A)

    return src
end

# Convenience methods for setindex! with Colon
function Base.setindex!(A::SparseMatrixMPI{T}, val, row_rng::UnitRange{Int}, ::Colon) where T
    return setindex!(A, val, row_rng, 1:size(A, 2))
end

function Base.setindex!(A::SparseMatrixMPI{T}, val, ::Colon, col_rng::UnitRange{Int}) where T
    return setindex!(A, val, 1:size(A, 1), col_rng)
end

function Base.setindex!(A::SparseMatrixMPI{T}, val, ::Colon, ::Colon) where T
    return setindex!(A, val, 1:size(A, 1), 1:size(A, 2))
end

# Also add full colon setindex! for MatrixMPI
function Base.setindex!(A::MatrixMPI{T}, val, ::Colon, ::Colon) where T
    return setindex!(A, val, 1:size(A, 1), 1:size(A, 2))
end

# ============================================================================
# VectorMPI Indexing with VectorMPI indices
# ============================================================================

"""
    Base.getindex(v::VectorMPI{T}, idx::VectorMPI{Int}) where T

Extract elements `v[idx]` where `idx` is a distributed VectorMPI of integer indices.

This is a collective operation - all ranks must call it with the same `idx`.
Each rank requests the values at its local indices `idx.v`, which may be owned
by different ranks in `v`. Communication is used to gather the requested values.

The result is a VectorMPI with the same partition as `idx`.

# Example
```julia
v = VectorMPI([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
idx = VectorMPI([3, 1, 5, 2])
result = v[idx]  # Returns VectorMPI with values [3.0, 1.0, 5.0, 2.0]
```
"""
function Base.getindex(v::VectorMPI{T}, idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    n = length(v)

    # Validate indices (local check - each rank checks its own indices)
    for i in idx.v
        if i < 1 || i > n
            error("VectorMPI index out of bounds: $i, length=$n")
        end
    end

    # My local indices into v (global indices)
    local_idx = idx.v
    n_local = length(local_idx)

    # Group local indices by which rank owns them in v
    needed_from = [Tuple{Int,Int}[] for _ in 1:nranks]  # (global_v_idx, local_result_idx)
    for (result_idx, v_global_idx) in enumerate(local_idx)
        owner = searchsortedlast(v.partition, v_global_idx) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(needed_from[owner + 1], (v_global_idx, result_idx))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(needed_from[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send requested indices to each owner rank
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()
    recv_perm_map = Dict{Int, Vector{Int}}()  # Maps rank -> destination indices in result

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in needed_from[r + 1]]
            dst_indices = [t[2] for t in needed_from[r + 1]]
            recv_perm_map[r] = dst_indices
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, MPI.Isend(indices, comm; dest=r, tag=80))
        end
    end

    # Receive index requests from other ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=80))
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Prepare to send values
    my_v_start = v.partition[rank + 1]

    # Post receives for values
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Vector{T}}()
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            recv_bufs[r] = Vector{T}(undef, send_counts[r + 1])
            push!(recv_reqs, MPI.Irecv!(recv_bufs[r], comm; source=r, tag=81))
        end
    end

    # Send values to requesters
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Vector{T}}()
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            global_indices = struct_recv_bufs[r]
            vals = Vector{T}(undef, length(global_indices))
            for (k, g_idx) in enumerate(global_indices)
                local_idx_in_v = g_idx - my_v_start + 1
                vals[k] = v.v[local_idx_in_v]
            end
            send_bufs[r] = vals
            push!(send_reqs, MPI.Isend(vals, comm; dest=r, tag=81))
        end
    end

    MPI.Waitall(recv_reqs)

    # Assemble result
    result_v = Vector{T}(undef, n_local)

    # Fill from local data (indices I own in v)
    for (v_global_idx, result_idx) in needed_from[rank + 1]
        local_idx_in_v = v_global_idx - my_v_start + 1
        result_v[result_idx] = v.v[local_idx_in_v]
    end

    # Fill from received data
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            dst_indices = recv_perm_map[r]
            vals = recv_bufs[r]
            for (k, dst_idx) in enumerate(dst_indices)
                result_v[dst_idx] = vals[k]
            end
        end
    end

    MPI.Waitall(send_reqs)

    # Result has same partition as idx, so same hash
    return VectorMPI{T}(idx.structural_hash, idx.partition, result_v)
end

# ============================================================================
# MatrixMPI Indexing with VectorMPI indices
# ============================================================================

"""
    Base.getindex(A::MatrixMPI{T}, row_idx::VectorMPI{Int}, col_idx::VectorMPI{Int}) where T

Extract a submatrix `A[row_idx, col_idx]` where indices are distributed VectorMPI{Int}.

This is a collective operation - all ranks must call it with the same `row_idx` and `col_idx`.
The result is a MatrixMPI of size `(length(row_idx), length(col_idx))`.

Each rank computes its local portion of the result matrix based on `row_idx`'s partition.
Communication is used to gather row data from ranks that own the requested rows.

# Example
```julia
A = MatrixMPI(reshape(1.0:12.0, 4, 3))
row_idx = VectorMPI([2, 4, 1])
col_idx = VectorMPI([3, 1])
result = A[row_idx, col_idx]  # Returns MatrixMPI submatrix
```
"""
function Base.getindex(A::MatrixMPI{T}, row_idx::VectorMPI{Int}, col_idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Gather col_idx to all ranks (columns are not distributed, all ranks need all column indices)
    col_indices = _gather_vector_to_all(col_idx, comm)

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("MatrixMPI column index out of bounds: $j, ncols=$n")
        end
    end

    ncols_result = length(col_indices)
    nrows_result = length(row_idx)

    # Result partition follows row_idx partition
    result_row_partition = row_idx.partition

    # My local row indices (global indices into A)
    my_row_indices = row_idx.v
    n_local_rows = length(my_row_indices)

    # Validate row indices
    for i in my_row_indices
        if i < 1 || i > m
            error("MatrixMPI row index out of bounds: $i, nrows=$m")
        end
    end

    # Group local row indices by which rank owns them in A
    needed_from = [Tuple{Int,Int}[] for _ in 1:nranks]  # (global_A_row, local_result_row)
    for (result_row, A_global_row) in enumerate(my_row_indices)
        owner = searchsortedlast(A.row_partition, A_global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(needed_from[owner + 1], (A_global_row, result_row))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(needed_from[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send requested row indices to each owner rank
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()
    recv_perm_map = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in needed_from[r + 1]]
            dst_indices = [t[2] for t in needed_from[r + 1]]
            recv_perm_map[r] = dst_indices
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, MPI.Isend(indices, comm; dest=r, tag=82))
        end
    end

    # Receive index requests from other ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=82))
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Prepare to send row data
    my_A_row_start = A.row_partition[rank + 1]

    # Post receives for row data (each row has ncols_result elements)
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Matrix{T}}()
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            recv_bufs[r] = Matrix{T}(undef, send_counts[r + 1], ncols_result)
            push!(recv_reqs, MPI.Irecv!(recv_bufs[r], comm; source=r, tag=83))
        end
    end

    # Send row data to requesters
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Matrix{T}}()
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            global_row_indices = struct_recv_bufs[r]
            nrows_to_send = length(global_row_indices)
            data = Matrix{T}(undef, nrows_to_send, ncols_result)
            for (k, g_row) in enumerate(global_row_indices)
                local_row_in_A = g_row - my_A_row_start + 1
                for (c, g_col) in enumerate(col_indices)
                    data[k, c] = A.A[local_row_in_A, g_col]
                end
            end
            send_bufs[r] = data
            push!(send_reqs, MPI.Isend(data, comm; dest=r, tag=83))
        end
    end

    MPI.Waitall(recv_reqs)

    # Assemble result
    result_A = Matrix{T}(undef, n_local_rows, ncols_result)

    # Fill from local data (rows I own in A)
    for (A_global_row, result_row) in needed_from[rank + 1]
        local_row_in_A = A_global_row - my_A_row_start + 1
        for (c, g_col) in enumerate(col_indices)
            result_A[result_row, c] = A.A[local_row_in_A, g_col]
        end
    end

    # Fill from received data
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            dst_indices = recv_perm_map[r]
            data = recv_bufs[r]
            for (k, dst_row) in enumerate(dst_indices)
                result_A[dst_row, :] .= data[k, :]
            end
        end
    end

    MPI.Waitall(send_reqs)

    # Compute column partition for result
    result_col_partition = uniform_partition(ncols_result, nranks)

    # Compute hash for result
    hash = compute_dense_structural_hash(result_row_partition, result_col_partition, size(result_A), comm)

    return MatrixMPI{T}(hash, result_row_partition, result_col_partition, result_A)
end

# ============================================================================
# SparseMatrixMPI Indexing with VectorMPI indices
# ============================================================================

"""
    Base.getindex(A::SparseMatrixMPI{T}, row_idx::VectorMPI{Int}, col_idx::VectorMPI{Int}) where T

Extract a submatrix `A[row_idx, col_idx]` where indices are distributed VectorMPI{Int}.

This is a collective operation - all ranks must call it with the same `row_idx` and `col_idx`.
The result is a MatrixMPI (dense) of size `(length(row_idx), length(col_idx))`.

Note: The result is dense because arbitrary indexing typically doesn't preserve
useful sparsity patterns. For large sparse matrices with structured indexing,
consider using range-based indexing instead.

# Example
```julia
A = SparseMatrixMPI{Float64}(sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 4))
row_idx = VectorMPI([2, 4, 1])
col_idx = VectorMPI([3, 1])
result = A[row_idx, col_idx]  # Returns MatrixMPI (dense) submatrix
```
"""
function Base.getindex(A::SparseMatrixMPI{T}, row_idx::VectorMPI{Int}, col_idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Gather col_idx to all ranks (columns are not distributed)
    col_indices = _gather_vector_to_all(col_idx, comm)

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("SparseMatrixMPI column index out of bounds: $j, ncols=$n")
        end
    end

    ncols_result = length(col_indices)
    nrows_result = length(row_idx)

    # Build column lookup: global_col -> result_col_position
    col_to_result = Dict{Int, Int}()
    for (pos, col) in enumerate(col_indices)
        col_to_result[col] = pos
    end

    # Result partition follows row_idx partition
    result_row_partition = row_idx.partition

    # My local row indices (global indices into A)
    my_row_indices = row_idx.v
    n_local_rows = length(my_row_indices)

    # Validate row indices
    for i in my_row_indices
        if i < 1 || i > m
            error("SparseMatrixMPI row index out of bounds: $i, nrows=$m")
        end
    end

    # Group local row indices by which rank owns them in A
    needed_from = [Tuple{Int,Int}[] for _ in 1:nranks]
    for (result_row, A_global_row) in enumerate(my_row_indices)
        owner = searchsortedlast(A.row_partition, A_global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(needed_from[owner + 1], (A_global_row, result_row))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(needed_from[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send requested row indices to each owner rank
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()
    recv_perm_map = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in needed_from[r + 1]]
            dst_indices = [t[2] for t in needed_from[r + 1]]
            recv_perm_map[r] = dst_indices
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, MPI.Isend(indices, comm; dest=r, tag=84))
        end
    end

    # Receive index requests from other ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=84))
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Prepare to send row data
    my_A_row_start = A.row_partition[rank + 1]
    AT = A.A.parent

    # Helper function to extract a row from sparse matrix
    function extract_sparse_row(local_row::Int, cols::Vector{Int}, col_lookup::Dict{Int,Int})
        row_data = zeros(T, length(cols))
        for k in AT.colptr[local_row]:(AT.colptr[local_row+1]-1)
            local_col_idx = AT.rowval[k]
            if local_col_idx <= length(A.col_indices)
                global_col = A.col_indices[local_col_idx]
                if haskey(col_lookup, global_col)
                    result_col = col_lookup[global_col]
                    row_data[result_col] = AT.nzval[k]
                end
            end
        end
        return row_data
    end

    # Post receives for row data
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Matrix{T}}()
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            recv_bufs[r] = Matrix{T}(undef, send_counts[r + 1], ncols_result)
            push!(recv_reqs, MPI.Irecv!(recv_bufs[r], comm; source=r, tag=85))
        end
    end

    # Send row data to requesters
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Matrix{T}}()
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            global_row_indices = struct_recv_bufs[r]
            nrows_to_send = length(global_row_indices)
            data = Matrix{T}(undef, nrows_to_send, ncols_result)
            for (k, g_row) in enumerate(global_row_indices)
                local_row_in_A = g_row - my_A_row_start + 1
                data[k, :] .= extract_sparse_row(local_row_in_A, col_indices, col_to_result)
            end
            send_bufs[r] = data
            push!(send_reqs, MPI.Isend(data, comm; dest=r, tag=85))
        end
    end

    MPI.Waitall(recv_reqs)

    # Assemble result
    result_A = zeros(T, n_local_rows, ncols_result)

    # Fill from local data (rows I own in A)
    for (A_global_row, result_row) in needed_from[rank + 1]
        local_row_in_A = A_global_row - my_A_row_start + 1
        result_A[result_row, :] .= extract_sparse_row(local_row_in_A, col_indices, col_to_result)
    end

    # Fill from received data
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            dst_indices = recv_perm_map[r]
            data = recv_bufs[r]
            for (k, dst_row) in enumerate(dst_indices)
                result_A[dst_row, :] .= data[k, :]
            end
        end
    end

    MPI.Waitall(send_reqs)

    # Compute column partition for result
    result_col_partition = uniform_partition(ncols_result, nranks)

    # Compute hash for result
    hash = compute_dense_structural_hash(result_row_partition, result_col_partition, size(result_A), comm)

    return MatrixMPI{T}(hash, result_row_partition, result_col_partition, result_A)
end

# Helper function to gather a VectorMPI to all ranks (generic version for any element type)
function _gather_vector_to_all(v::VectorMPI{T}, comm::MPI.Comm) where T
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Gather counts
    local_count = Int32(length(v.v))
    counts = Vector{Int32}(undef, nranks)
    MPI.Allgather!(Ref(local_count), MPI.UBuffer(counts, 1), comm)

    # Compute displacements
    displs = Vector{Int32}(undef, nranks)
    displs[1] = 0
    for i in 2:nranks
        displs[i] = displs[i-1] + counts[i-1]
    end

    total = sum(counts)
    result = Vector{T}(undef, total)

    # Allgatherv
    MPI.Allgatherv!(v.v, MPI.VBuffer(result, counts, displs), comm)

    return result
end

# ============================================================================
# VectorMPI setindex! with VectorMPI indices
# ============================================================================

"""
    Base.setindex!(v::VectorMPI{T}, src::VectorMPI{T}, idx::VectorMPI{Int}) where T

Set elements `v[idx] = src` where `idx` is a distributed VectorMPI of integer indices
and `src` is a distributed VectorMPI of values.

This is a collective operation - all ranks must call it with the same `idx` and `src`.
The `src` and `idx` must have the same partition (same length and distribution).
Each `src[k]` is assigned to `v[idx[k]]`.

Communication is used to send values from the ranks that own them in `src` to the
ranks that own the destination positions in `v`.

# Example
```julia
v = VectorMPI([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
idx = VectorMPI([3, 1, 5, 2])
src = VectorMPI([30.0, 10.0, 50.0, 20.0])
v[idx] = src  # Sets v[3]=30, v[1]=10, v[5]=50, v[2]=20
```
"""
function Base.setindex!(v::VectorMPI{T}, src::VectorMPI{T}, idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    n = length(v)

    # Validate that src and idx have the same partition
    if src.partition != idx.partition
        error("VectorMPI setindex!: src and idx must have the same partition")
    end

    # Validate indices (local check)
    for i in idx.v
        if i < 1 || i > n
            error("VectorMPI index out of bounds: $i, length=$n")
        end
    end

    # My local indices and values
    local_idx = idx.v
    local_src = src.v
    n_local = length(local_idx)

    # Group (index, value) pairs by which rank owns the destination in v
    send_to = [Tuple{Int,T}[] for _ in 1:nranks]  # (global_v_idx, value)
    for k in 1:n_local
        v_global_idx = local_idx[k]
        value = local_src[k]
        owner = searchsortedlast(v.partition, v_global_idx) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(send_to[owner + 1], (v_global_idx, value))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, MPI.Isend(indices, comm; dest=r, tag=90))
        end
    end

    # Receive indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=90))
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Send values to destination ranks
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Vector{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            values = [t[2] for t in send_to[r + 1]]
            send_bufs[r] = values
            push!(send_reqs, MPI.Isend(values, comm; dest=r, tag=91))
        end
    end

    # Receive values from source ranks
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Vector{T}}()

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{T}(undef, recv_counts[r + 1])
            push!(recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=91))
            recv_bufs[r] = buf
        end
    end

    MPI.Waitall(recv_reqs)

    # Apply local assignments (from my own send_to[rank+1])
    my_v_start = v.partition[rank + 1]
    for (v_global_idx, value) in send_to[rank + 1]
        local_idx_in_v = v_global_idx - my_v_start + 1
        v.v[local_idx_in_v] = value
    end

    # Apply received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            values = recv_bufs[r]
            for (k, v_global_idx) in enumerate(indices)
                local_idx_in_v = v_global_idx - my_v_start + 1
                v.v[local_idx_in_v] = values[k]
            end
        end
    end

    MPI.Waitall(send_reqs)

    return src
end

# ============================================================================
# MatrixMPI setindex! with VectorMPI indices
# ============================================================================

"""
    Base.setindex!(A::MatrixMPI{T}, src::MatrixMPI{T}, row_idx::VectorMPI{Int}, col_idx::VectorMPI{Int}) where T

Set elements `A[row_idx, col_idx] = src` where indices are distributed VectorMPI{Int}
and `src` is a distributed MatrixMPI of values.

This is a collective operation - all ranks must call it with the same arguments.
The `src` must have size `(length(row_idx), length(col_idx))` and its row partition
must match `row_idx`'s partition.

Each `src[i, j]` is assigned to `A[row_idx[i], col_idx[j]]`.

# Example
```julia
A = MatrixMPI(zeros(6, 4))
row_idx = VectorMPI([2, 4, 1])
col_idx = VectorMPI([3, 1])
src = MatrixMPI(ones(3, 2))
A[row_idx, col_idx] = src  # Sets A[2,3]=1, A[2,1]=1, A[4,3]=1, etc.
```
"""
function Base.setindex!(A::MatrixMPI{T}, src::MatrixMPI{T}, row_idx::VectorMPI{Int}, col_idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Gather col_idx to all ranks
    col_indices = _gather_vector_to_all(col_idx, comm)
    ncols_src = length(col_indices)

    # Validate dimensions
    if size(src) != (length(row_idx), ncols_src)
        error("MatrixMPI setindex!: src size $(size(src)) doesn't match index dimensions ($(length(row_idx)), $ncols_src)")
    end

    # Validate that src row partition matches row_idx partition
    if src.row_partition != row_idx.partition
        error("MatrixMPI setindex!: src row partition must match row_idx partition")
    end

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("MatrixMPI column index out of bounds: $j, ncols=$n")
        end
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("MatrixMPI row index out of bounds: $i, nrows=$m")
        end
    end

    # My local row indices and corresponding src rows
    local_row_idx = row_idx.v
    n_local_rows = length(local_row_idx)

    # Group rows by which rank owns the destination in A
    # Each entry: (global_A_row, local_src_row)
    send_to = [Tuple{Int,Int}[] for _ in 1:nranks]
    for local_src_row in 1:n_local_rows
        A_global_row = local_row_idx[local_src_row]
        owner = searchsortedlast(A.row_partition, A_global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(send_to[owner + 1], (A_global_row, local_src_row))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, MPI.Isend(indices, comm; dest=r, tag=92))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=92))
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Send row data to destination ranks
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            nrows_to_send = send_counts[r + 1]
            data = Matrix{T}(undef, nrows_to_send, ncols_src)
            for (k, (_, local_src_row)) in enumerate(send_to[r + 1])
                data[k, :] .= src.A[local_src_row, 1:ncols_src]
            end
            send_bufs[r] = data
            push!(send_reqs, MPI.Isend(data, comm; dest=r, tag=93))
        end
    end

    # Receive row data from source ranks
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
            push!(recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=93))
            recv_bufs[r] = buf
        end
    end

    MPI.Waitall(recv_reqs)

    # Apply local assignments
    my_A_row_start = A.row_partition[rank + 1]
    for (A_global_row, local_src_row) in send_to[rank + 1]
        local_row_in_A = A_global_row - my_A_row_start + 1
        for (c, g_col) in enumerate(col_indices)
            A.A[local_row_in_A, g_col] = src.A[local_src_row, c]
        end
    end

    # Apply received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            data = recv_bufs[r]
            for (k, A_global_row) in enumerate(indices)
                local_row_in_A = A_global_row - my_A_row_start + 1
                for (c, g_col) in enumerate(col_indices)
                    A.A[local_row_in_A, g_col] = data[k, c]
                end
            end
        end
    end

    MPI.Waitall(send_reqs)

    return src
end

# ============================================================================
# SparseMatrixMPI setindex! with VectorMPI indices
# ============================================================================

"""
    Base.setindex!(A::SparseMatrixMPI{T}, src::MatrixMPI{T}, row_idx::VectorMPI{Int}, col_idx::VectorMPI{Int}) where T

Set elements `A[row_idx, col_idx] = src` where indices are distributed VectorMPI{Int}
and `src` is a distributed MatrixMPI of values.

This is a collective operation - all ranks must call it with the same arguments.
The `src` must have size `(length(row_idx), length(col_idx))` and its row partition
must match `row_idx`'s partition.

This is a structural modification - new nonzeros from src are added to A's sparsity
pattern. After the operation, structural_hash is recomputed and caches are invalidated.

# Example
```julia
A = SparseMatrixMPI{Float64}(spzeros(6, 6))
row_idx = VectorMPI([2, 4, 1])
col_idx = VectorMPI([3, 1])
src = MatrixMPI(ones(3, 2))
A[row_idx, col_idx] = src  # Adds nonzeros at specified positions
```
"""
function Base.setindex!(A::SparseMatrixMPI{T}, src::MatrixMPI{T}, row_idx::VectorMPI{Int}, col_idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Gather col_idx to all ranks
    col_indices = _gather_vector_to_all(col_idx, comm)
    ncols_src = length(col_indices)

    # Validate dimensions
    if size(src) != (length(row_idx), ncols_src)
        error("SparseMatrixMPI setindex!: src size $(size(src)) doesn't match index dimensions ($(length(row_idx)), $ncols_src)")
    end

    # Validate that src row partition matches row_idx partition
    if src.row_partition != row_idx.partition
        error("SparseMatrixMPI setindex!: src row partition must match row_idx partition")
    end

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("SparseMatrixMPI column index out of bounds: $j, ncols=$n")
        end
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("SparseMatrixMPI row index out of bounds: $i, nrows=$m")
        end
    end

    # My local row indices and corresponding src rows
    local_row_idx = row_idx.v
    n_local_rows = length(local_row_idx)

    # Group rows by which rank owns the destination in A
    send_to = [Tuple{Int,Int}[] for _ in 1:nranks]
    for local_src_row in 1:n_local_rows
        A_global_row = local_row_idx[local_src_row]
        owner = searchsortedlast(A.row_partition, A_global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(send_to[owner + 1], (A_global_row, local_src_row))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, MPI.Isend(indices, comm; dest=r, tag=94))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=94))
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Send row data to destination ranks
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            nrows_to_send = send_counts[r + 1]
            data = Matrix{T}(undef, nrows_to_send, ncols_src)
            for (k, (_, local_src_row)) in enumerate(send_to[r + 1])
                data[k, :] .= src.A[local_src_row, 1:ncols_src]
            end
            send_bufs[r] = data
            push!(send_reqs, MPI.Isend(data, comm; dest=r, tag=95))
        end
    end

    # Receive row data from source ranks
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
            push!(recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=95))
            recv_bufs[r] = buf
        end
    end

    MPI.Waitall(recv_reqs)

    # Build insertions for structural modification
    insertions = Vector{Tuple{Int,Int,T}}()
    my_A_row_start = A.row_partition[rank + 1]

    # From local assignments
    for (A_global_row, local_src_row) in send_to[rank + 1]
        for (c, g_col) in enumerate(col_indices)
            val = src.A[local_src_row, c]
            push!(insertions, (A_global_row, g_col, val))
        end
    end

    # From received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            data = recv_bufs[r]
            for (k, A_global_row) in enumerate(indices)
                for (c, g_col) in enumerate(col_indices)
                    val = data[k, c]
                    push!(insertions, (A_global_row, g_col, val))
                end
            end
        end
    end

    MPI.Waitall(send_reqs)

    # Apply insertions using the helper function
    if !isempty(insertions)
        row_offset = A.row_partition[rank + 1]
        new_AT, new_col_indices = _rebuild_AT_with_insertions(
            A.A.parent, A.col_indices, insertions, row_offset
        )
        A.col_indices = new_col_indices
        A.A = transpose(new_AT)
    end

    # Invalidate structural hash (will be recomputed lazily on next use)
    A.structural_hash = nothing

    # Invalidate cached transpose bidirectionally
    _invalidate_cached_transpose!(A)

    return src
end

# ============================================================================
# Mixed indexing: VectorMPI with ranges, Colon, and scalars
# ============================================================================

"""
    Base.getindex(A::MatrixMPI{T}, row_idx::VectorMPI{Int}, col_rng::UnitRange{Int}) where T

Get submatrix with rows selected by VectorMPI and columns by range.
Returns a MatrixMPI with row partition matching row_idx.partition.
"""
function Base.getindex(A::MatrixMPI{T}, row_idx::VectorMPI{Int}, col_rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    ncols_result = length(col_rng)

    # Validate column range
    if first(col_rng) < 1 || last(col_rng) > n
        error("MatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("MatrixMPI row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    n_local_result_rows = length(local_row_indices)

    # Group requests by source rank
    requests_to = [Int[] for _ in 1:nranks]
    local_positions = [Int[] for _ in 1:nranks]

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(requests_to[owner + 1], global_row)
        push!(local_positions[owner + 1], pos)
    end

    # Exchange request counts
    send_counts = Int32[length(requests_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send row indices we need
    send_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            push!(send_reqs, MPI.Isend(requests_to[r + 1], comm; dest=r, tag=100))
        end
    end

    # Receive row indices others need from us
    recv_bufs = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=100))
            recv_bufs[r] = buf
        end
    end

    MPI.Waitall(recv_reqs)
    MPI.Waitall(send_reqs)

    # Send requested row data
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Matrix{T}}()
    my_A_start = A.row_partition[rank + 1]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = recv_bufs[r]
            data = Matrix{T}(undef, length(indices), ncols_result)
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                data[k, :] .= A.A[local_row, col_rng]
            end
            data_send_bufs[r] = data
            push!(data_send_reqs, MPI.Isend(data, comm; dest=r, tag=101))
        end
    end

    # Receive row data
    data_recv_bufs = Dict{Int, Matrix{T}}()
    data_recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, send_counts[r + 1], ncols_result)
            push!(data_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=101))
            data_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(data_recv_reqs)

    # Build result matrix
    result_local = Matrix{T}(undef, n_local_result_rows, ncols_result)

    # Handle local rows (from self)
    for (k, global_row) in enumerate(requests_to[rank + 1])
        local_row = global_row - my_A_start + 1
        pos = local_positions[rank + 1][k]
        result_local[pos, :] .= A.A[local_row, col_rng]
    end

    # Handle received rows
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            data = data_recv_bufs[r]
            for (k, pos) in enumerate(local_positions[r + 1])
                result_local[pos, :] .= data[k, :]
            end
        end
    end

    MPI.Waitall(data_send_reqs)

    return MatrixMPI_local(result_local)
end

"""
    Base.getindex(A::MatrixMPI{T}, row_idx::VectorMPI{Int}, ::Colon) where T

Get submatrix with rows selected by VectorMPI and all columns.
"""
function Base.getindex(A::MatrixMPI{T}, row_idx::VectorMPI{Int}, ::Colon) where T
    return A[row_idx, 1:size(A, 2)]
end

"""
    Base.getindex(A::MatrixMPI{T}, row_rng::UnitRange{Int}, col_idx::VectorMPI{Int}) where T

Get submatrix with rows selected by range and columns by VectorMPI.
Returns a MatrixMPI with standard row partition for the given row range size.
"""
function Base.getindex(A::MatrixMPI{T}, row_rng::UnitRange{Int}, col_idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    nrows_result = length(row_rng)

    # Validate row range
    if first(row_rng) < 1 || last(row_rng) > m
        error("MatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end

    # Gather col_idx to all ranks (columns are not distributed)
    col_indices = _gather_vector_to_all(col_idx, comm)
    ncols_result = length(col_indices)

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("MatrixMPI column index out of bounds: $j, ncols=$n")
        end
    end

    # Create result partition for row range
    result_partition = uniform_partition(nrows_result, nranks)

    # Determine which rows I need
    my_result_start = result_partition[rank + 1]
    my_result_end = result_partition[rank + 2] - 1
    n_local_result_rows = my_result_end - my_result_start + 1

    if n_local_result_rows <= 0
        result_local = Matrix{T}(undef, 0, ncols_result)
        return MatrixMPI_local(result_local)
    end

    # Global rows I need from A
    global_rows_needed = [first(row_rng) + my_result_start + i - 2 for i in 1:n_local_result_rows]

    # Group by source rank in A
    requests_to = [Int[] for _ in 1:nranks]
    local_positions = [Int[] for _ in 1:nranks]

    for (pos, global_row) in enumerate(global_rows_needed)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(requests_to[owner + 1], global_row)
        push!(local_positions[owner + 1], pos)
    end

    # Exchange counts
    send_counts = Int32[length(requests_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send row indices
    send_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            push!(send_reqs, MPI.Isend(requests_to[r + 1], comm; dest=r, tag=102))
        end
    end

    # Receive row indices
    recv_bufs = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=102))
            recv_bufs[r] = buf
        end
    end

    MPI.Waitall(recv_reqs)
    MPI.Waitall(send_reqs)

    # Send row data (only the selected columns)
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Matrix{T}}()
    my_A_start = A.row_partition[rank + 1]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = recv_bufs[r]
            data = Matrix{T}(undef, length(indices), ncols_result)
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                for (c, col) in enumerate(col_indices)
                    data[k, c] = A.A[local_row, col]
                end
            end
            data_send_bufs[r] = data
            push!(data_send_reqs, MPI.Isend(data, comm; dest=r, tag=103))
        end
    end

    # Receive row data
    data_recv_bufs = Dict{Int, Matrix{T}}()
    data_recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, send_counts[r + 1], ncols_result)
            push!(data_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=103))
            data_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(data_recv_reqs)

    # Build result
    result_local = Matrix{T}(undef, n_local_result_rows, ncols_result)

    # Local rows
    for (k, global_row) in enumerate(requests_to[rank + 1])
        local_row = global_row - my_A_start + 1
        pos = local_positions[rank + 1][k]
        for (c, col) in enumerate(col_indices)
            result_local[pos, c] = A.A[local_row, col]
        end
    end

    # Received rows
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            data = data_recv_bufs[r]
            for (k, pos) in enumerate(local_positions[r + 1])
                result_local[pos, :] .= data[k, :]
            end
        end
    end

    MPI.Waitall(data_send_reqs)

    return MatrixMPI_local(result_local)
end

"""
    Base.getindex(A::MatrixMPI{T}, ::Colon, col_idx::VectorMPI{Int}) where T

Get submatrix with all rows and columns selected by VectorMPI.
"""
function Base.getindex(A::MatrixMPI{T}, ::Colon, col_idx::VectorMPI{Int}) where T
    return A[1:size(A, 1), col_idx]
end

"""
    Base.getindex(A::MatrixMPI{T}, row_idx::VectorMPI{Int}, j::Int) where T

Get column vector with rows selected by VectorMPI and single column j.
Returns a VectorMPI with partition matching row_idx.partition.
"""
function Base.getindex(A::MatrixMPI{T}, row_idx::VectorMPI{Int}, j::Int) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Validate column
    if j < 1 || j > n
        error("MatrixMPI column index out of bounds: $j, ncols=$n")
    end

    # Validate row indices
    for i in row_idx.v
        if i < 1 || i > m
            error("MatrixMPI row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    n_local = length(local_row_indices)

    # Group by source rank
    requests_to = [Int[] for _ in 1:nranks]
    local_positions = [Int[] for _ in 1:nranks]

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(requests_to[owner + 1], global_row)
        push!(local_positions[owner + 1], pos)
    end

    # Exchange counts
    send_counts = Int32[length(requests_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send row indices
    send_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            push!(send_reqs, MPI.Isend(requests_to[r + 1], comm; dest=r, tag=104))
        end
    end

    # Receive row indices
    recv_bufs = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=104))
            recv_bufs[r] = buf
        end
    end

    MPI.Waitall(recv_reqs)
    MPI.Waitall(send_reqs)

    # Send values
    val_send_reqs = MPI.Request[]
    val_send_bufs = Dict{Int, Vector{T}}()
    my_A_start = A.row_partition[rank + 1]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = recv_bufs[r]
            vals = Vector{T}(undef, length(indices))
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                vals[k] = A.A[local_row, j]
            end
            val_send_bufs[r] = vals
            push!(val_send_reqs, MPI.Isend(vals, comm; dest=r, tag=105))
        end
    end

    # Receive values
    val_recv_bufs = Dict{Int, Vector{T}}()
    val_recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            buf = Vector{T}(undef, send_counts[r + 1])
            push!(val_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=105))
            val_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(val_recv_reqs)

    # Build result
    result_local = Vector{T}(undef, n_local)

    # Local values
    for (k, global_row) in enumerate(requests_to[rank + 1])
        local_row = global_row - my_A_start + 1
        pos = local_positions[rank + 1][k]
        result_local[pos] = A.A[local_row, j]
    end

    # Received values
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            vals = val_recv_bufs[r]
            for (k, pos) in enumerate(local_positions[r + 1])
                result_local[pos] = vals[k]
            end
        end
    end

    MPI.Waitall(val_send_reqs)

    return VectorMPI_local(result_local)
end

"""
    Base.getindex(A::MatrixMPI{T}, i::Int, col_idx::VectorMPI{Int}) where T

Get row vector with single row i and columns selected by VectorMPI.
Returns a VectorMPI with partition matching col_idx.partition.
"""
function Base.getindex(A::MatrixMPI{T}, i::Int, col_idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Validate row
    if i < 1 || i > m
        error("MatrixMPI row index out of bounds: $i, nrows=$m")
    end

    # Gather col_idx to all ranks
    col_indices = _gather_vector_to_all(col_idx, comm)

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("MatrixMPI column index out of bounds: $j, ncols=$n")
        end
    end

    # Determine which rank owns row i
    owner = searchsortedlast(A.row_partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Owner extracts the row and broadcasts
    if rank == owner
        my_A_start = A.row_partition[rank + 1]
        local_row = i - my_A_start + 1
        row_data = [A.A[local_row, j] for j in col_indices]
    else
        row_data = Vector{T}(undef, length(col_indices))
    end

    MPI.Bcast!(row_data, owner, comm)

    # Each rank extracts its local portion based on col_idx.partition
    my_start = col_idx.partition[rank + 1]
    my_end = col_idx.partition[rank + 2] - 1
    n_local = my_end - my_start + 1

    if n_local > 0
        result_local = row_data[my_start:my_end]
    else
        result_local = T[]
    end

    return VectorMPI_local(result_local)
end

# SparseMatrixMPI mixed indexing methods

"""
    Base.getindex(A::SparseMatrixMPI{T}, row_idx::VectorMPI{Int}, col_rng::UnitRange{Int}) where T

Get submatrix with rows selected by VectorMPI and columns by range.
Returns a dense MatrixMPI with row partition matching row_idx.partition.
"""
function Base.getindex(A::SparseMatrixMPI{T}, row_idx::VectorMPI{Int}, col_rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    ncols_result = length(col_rng)

    # Validate column range
    if first(col_rng) < 1 || last(col_rng) > n
        error("SparseMatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    # Validate row indices
    for i in row_idx.v
        if i < 1 || i > m
            error("SparseMatrixMPI row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    n_local_result_rows = length(local_row_indices)

    # Group requests by source rank
    requests_to = [Int[] for _ in 1:nranks]
    local_positions = [Int[] for _ in 1:nranks]

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(requests_to[owner + 1], global_row)
        push!(local_positions[owner + 1], pos)
    end

    # Exchange counts
    send_counts = Int32[length(requests_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send row indices
    send_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            push!(send_reqs, MPI.Isend(requests_to[r + 1], comm; dest=r, tag=106))
        end
    end

    # Receive row indices
    recv_bufs = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=106))
            recv_bufs[r] = buf
        end
    end

    MPI.Waitall(recv_reqs)
    MPI.Waitall(send_reqs)

    # Send row data
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Matrix{T}}()
    my_A_start = A.row_partition[rank + 1]
    local_A = A.A.parent
    col_indices = A.col_indices

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = recv_bufs[r]
            data = zeros(T, length(indices), ncols_result)
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                for (local_j, global_j) in enumerate(col_indices)
                    if global_j in col_rng
                        result_col = global_j - first(col_rng) + 1
                        data[k, result_col] = local_A[local_j, local_row]
                    end
                end
            end
            data_send_bufs[r] = data
            push!(data_send_reqs, MPI.Isend(data, comm; dest=r, tag=107))
        end
    end

    # Receive row data
    data_recv_bufs = Dict{Int, Matrix{T}}()
    data_recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, send_counts[r + 1], ncols_result)
            push!(data_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=107))
            data_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(data_recv_reqs)

    # Build result
    result_local = zeros(T, n_local_result_rows, ncols_result)

    # Local rows
    for (k, global_row) in enumerate(requests_to[rank + 1])
        local_row = global_row - my_A_start + 1
        pos = local_positions[rank + 1][k]
        for (local_j, global_j) in enumerate(col_indices)
            if global_j in col_rng
                result_col = global_j - first(col_rng) + 1
                result_local[pos, result_col] = local_A[local_j, local_row]
            end
        end
    end

    # Received rows
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            data = data_recv_bufs[r]
            for (k, pos) in enumerate(local_positions[r + 1])
                result_local[pos, :] .= data[k, :]
            end
        end
    end

    MPI.Waitall(data_send_reqs)

    return MatrixMPI_local(result_local)
end

"""
    Base.getindex(A::SparseMatrixMPI{T}, row_idx::VectorMPI{Int}, ::Colon) where T

Get submatrix with rows selected by VectorMPI and all columns.
"""
function Base.getindex(A::SparseMatrixMPI{T}, row_idx::VectorMPI{Int}, ::Colon) where T
    return A[row_idx, 1:size(A, 2)]
end

"""
    Base.getindex(A::SparseMatrixMPI{T}, row_rng::UnitRange{Int}, col_idx::VectorMPI{Int}) where T

Get submatrix with rows by range and columns by VectorMPI.
Returns a dense MatrixMPI.
"""
function Base.getindex(A::SparseMatrixMPI{T}, row_rng::UnitRange{Int}, col_idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    nrows_result = length(row_rng)

    # Validate row range
    if first(row_rng) < 1 || last(row_rng) > m
        error("SparseMatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end

    # Gather col_idx to all ranks
    col_indices_result = _gather_vector_to_all(col_idx, comm)
    ncols_result = length(col_indices_result)

    # Validate column indices
    for j in col_indices_result
        if j < 1 || j > n
            error("SparseMatrixMPI column index out of bounds: $j, ncols=$n")
        end
    end

    # Create result partition
    result_partition = uniform_partition(nrows_result, nranks)
    my_result_start = result_partition[rank + 1]
    my_result_end = result_partition[rank + 2] - 1
    n_local_result_rows = my_result_end - my_result_start + 1

    if n_local_result_rows <= 0
        result_local = Matrix{T}(undef, 0, ncols_result)
        return MatrixMPI_local(result_local)
    end

    # Global rows I need
    global_rows_needed = [first(row_rng) + my_result_start + i - 2 for i in 1:n_local_result_rows]

    # Group by source rank
    requests_to = [Int[] for _ in 1:nranks]
    local_positions = [Int[] for _ in 1:nranks]

    for (pos, global_row) in enumerate(global_rows_needed)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(requests_to[owner + 1], global_row)
        push!(local_positions[owner + 1], pos)
    end

    # Exchange counts
    send_counts = Int32[length(requests_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send/receive row indices
    send_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            push!(send_reqs, MPI.Isend(requests_to[r + 1], comm; dest=r, tag=108))
        end
    end

    recv_bufs = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=108))
            recv_bufs[r] = buf
        end
    end

    MPI.Waitall(recv_reqs)
    MPI.Waitall(send_reqs)

    # Send row data
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Matrix{T}}()
    my_A_start = A.row_partition[rank + 1]
    local_A = A.A.parent
    col_indices = A.col_indices

    # Build column index lookup
    col_idx_map = Dict{Int, Int}()
    for (c, j) in enumerate(col_indices_result)
        col_idx_map[j] = c
    end

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = recv_bufs[r]
            data = zeros(T, length(indices), ncols_result)
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                for (local_j, global_j) in enumerate(col_indices)
                    if haskey(col_idx_map, global_j)
                        result_col = col_idx_map[global_j]
                        data[k, result_col] = local_A[local_j, local_row]
                    end
                end
            end
            data_send_bufs[r] = data
            push!(data_send_reqs, MPI.Isend(data, comm; dest=r, tag=109))
        end
    end

    # Receive row data
    data_recv_bufs = Dict{Int, Matrix{T}}()
    data_recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, send_counts[r + 1], ncols_result)
            push!(data_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=109))
            data_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(data_recv_reqs)

    # Build result
    result_local = zeros(T, n_local_result_rows, ncols_result)

    # Local rows
    for (k, global_row) in enumerate(requests_to[rank + 1])
        local_row = global_row - my_A_start + 1
        pos = local_positions[rank + 1][k]
        for (local_j, global_j) in enumerate(col_indices)
            if haskey(col_idx_map, global_j)
                result_col = col_idx_map[global_j]
                result_local[pos, result_col] = local_A[local_j, local_row]
            end
        end
    end

    # Received rows
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            data = data_recv_bufs[r]
            for (k, pos) in enumerate(local_positions[r + 1])
                result_local[pos, :] .= data[k, :]
            end
        end
    end

    MPI.Waitall(data_send_reqs)

    return MatrixMPI_local(result_local)
end

"""
    Base.getindex(A::SparseMatrixMPI{T}, ::Colon, col_idx::VectorMPI{Int}) where T

Get submatrix with all rows and columns by VectorMPI.
"""
function Base.getindex(A::SparseMatrixMPI{T}, ::Colon, col_idx::VectorMPI{Int}) where T
    return A[1:size(A, 1), col_idx]
end

"""
    Base.getindex(A::SparseMatrixMPI{T}, row_idx::VectorMPI{Int}, j::Int) where T

Get column vector with rows by VectorMPI and single column j.
Returns a VectorMPI.
"""
function Base.getindex(A::SparseMatrixMPI{T}, row_idx::VectorMPI{Int}, j::Int) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Validate
    if j < 1 || j > n
        error("SparseMatrixMPI column index out of bounds: $j, ncols=$n")
    end
    for i in row_idx.v
        if i < 1 || i > m
            error("SparseMatrixMPI row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    n_local = length(local_row_indices)

    # Group by source rank
    requests_to = [Int[] for _ in 1:nranks]
    local_positions = [Int[] for _ in 1:nranks]

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(requests_to[owner + 1], global_row)
        push!(local_positions[owner + 1], pos)
    end

    # Exchange counts
    send_counts = Int32[length(requests_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send/receive row indices
    send_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            push!(send_reqs, MPI.Isend(requests_to[r + 1], comm; dest=r, tag=110))
        end
    end

    recv_bufs = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=110))
            recv_bufs[r] = buf
        end
    end

    MPI.Waitall(recv_reqs)
    MPI.Waitall(send_reqs)

    # Send values
    val_send_reqs = MPI.Request[]
    val_send_bufs = Dict{Int, Vector{T}}()
    my_A_start = A.row_partition[rank + 1]
    local_A = A.A.parent
    col_indices = A.col_indices
    local_j_idx = findfirst(==(j), col_indices)

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = recv_bufs[r]
            vals = zeros(T, length(indices))
            if local_j_idx !== nothing
                for (k, global_row) in enumerate(indices)
                    local_row = global_row - my_A_start + 1
                    vals[k] = local_A[local_j_idx, local_row]
                end
            end
            val_send_bufs[r] = vals
            push!(val_send_reqs, MPI.Isend(vals, comm; dest=r, tag=111))
        end
    end

    # Receive values
    val_recv_bufs = Dict{Int, Vector{T}}()
    val_recv_reqs = MPI.Request[]
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            buf = Vector{T}(undef, send_counts[r + 1])
            push!(val_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=111))
            val_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(val_recv_reqs)

    # Build result
    result_local = zeros(T, n_local)

    # Local values
    if local_j_idx !== nothing
        for (k, global_row) in enumerate(requests_to[rank + 1])
            local_row = global_row - my_A_start + 1
            pos = local_positions[rank + 1][k]
            result_local[pos] = local_A[local_j_idx, local_row]
        end
    end

    # Received values
    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            vals = val_recv_bufs[r]
            for (k, pos) in enumerate(local_positions[r + 1])
                result_local[pos] = vals[k]
            end
        end
    end

    MPI.Waitall(val_send_reqs)

    return VectorMPI_local(result_local)
end

"""
    Base.getindex(A::SparseMatrixMPI{T}, i::Int, col_idx::VectorMPI{Int}) where T

Get row vector with single row i and columns by VectorMPI.
Returns a VectorMPI.
"""
function Base.getindex(A::SparseMatrixMPI{T}, i::Int, col_idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Validate
    if i < 1 || i > m
        error("SparseMatrixMPI row index out of bounds: $i, nrows=$m")
    end

    # Gather col_idx
    col_indices_result = _gather_vector_to_all(col_idx, comm)

    for j in col_indices_result
        if j < 1 || j > n
            error("SparseMatrixMPI column index out of bounds: $j, ncols=$n")
        end
    end

    # Determine owner of row i
    owner = searchsortedlast(A.row_partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Owner extracts row and broadcasts
    if rank == owner
        my_A_start = A.row_partition[rank + 1]
        local_row = i - my_A_start + 1
        local_A = A.A.parent
        col_indices = A.col_indices

        row_data = zeros(T, length(col_indices_result))
        # col_indices_result is sorted, use binary search instead of Dict
        for (local_j, global_j) in enumerate(col_indices)
            idx = searchsortedfirst(col_indices_result, global_j)
            if idx <= length(col_indices_result) && col_indices_result[idx] == global_j
                row_data[idx] = local_A[local_j, local_row]
            end
        end
    else
        row_data = Vector{T}(undef, length(col_indices_result))
    end

    MPI.Bcast!(row_data, owner, comm)

    # Extract local portion
    my_start = col_idx.partition[rank + 1]
    my_end = col_idx.partition[rank + 2] - 1
    n_local = my_end - my_start + 1

    if n_local > 0
        result_local = row_data[my_start:my_end]
    else
        result_local = T[]
    end

    return VectorMPI_local(result_local)
end

# ============================================================================
# Mixed setindex!: VectorMPI with ranges, Colon, and scalars
# ============================================================================

"""
    Base.setindex!(A::MatrixMPI{T}, src::MatrixMPI{T}, row_idx::VectorMPI{Int}, col_rng::UnitRange{Int}) where T

Set `A[row_idx, col_rng] = src` where rows are selected by VectorMPI and columns by range.
The `src` must have row partition matching `row_idx.partition` and column count matching `length(col_rng)`.
"""
function Base.setindex!(A::MatrixMPI{T}, src::MatrixMPI{T}, row_idx::VectorMPI{Int}, col_rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    ncols_src = length(col_rng)

    # Validate column range
    if first(col_rng) < 1 || last(col_rng) > n
        error("MatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    # Validate src dimensions
    if size(src, 2) != ncols_src
        error("MatrixMPI setindex!: src columns ($(size(src, 2))) must match range length ($ncols_src)")
    end
    if src.row_partition != row_idx.partition
        error("MatrixMPI setindex!: src row partition must match row_idx partition")
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("MatrixMPI row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    n_local_src_rows = length(local_row_indices)

    # Group data by destination rank
    send_to = [Tuple{Int, Vector{T}}[] for _ in 1:nranks]  # (global_row, row_data)

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        row_data = Vector{T}(src.A[pos, :])
        push!(send_to[owner + 1], (global_row, row_data))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, MPI.Isend(indices, comm; dest=r, tag=110))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=110))
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Send row data to destination ranks
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            n_rows = length(send_to[r + 1])
            data = Matrix{T}(undef, n_rows, ncols_src)
            for (k, (_, row_data)) in enumerate(send_to[r + 1])
                data[k, :] .= row_data
            end
            data_send_bufs[r] = data
            push!(data_send_reqs, MPI.Isend(data, comm; dest=r, tag=111))
        end
    end

    # Receive row data from source ranks
    data_recv_bufs = Dict{Int, Matrix{T}}()
    data_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
            push!(data_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=111))
            data_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(data_recv_reqs)

    # Apply local assignments (from my own send_to[rank+1])
    my_A_start = A.row_partition[rank + 1]
    for (global_row, row_data) in send_to[rank + 1]
        local_row = global_row - my_A_start + 1
        A.A[local_row, col_rng] .= row_data
    end

    # Apply received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            data = data_recv_bufs[r]
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                A.A[local_row, col_rng] .= data[k, :]
            end
        end
    end

    MPI.Waitall(data_send_reqs)

    return src
end

"""
    Base.setindex!(A::MatrixMPI{T}, src::MatrixMPI{T}, row_idx::VectorMPI{Int}, ::Colon) where T

Set `A[row_idx, :] = src` where rows are selected by VectorMPI and all columns.
"""
function Base.setindex!(A::MatrixMPI{T}, src::MatrixMPI{T}, row_idx::VectorMPI{Int}, ::Colon) where T
    A[row_idx, 1:size(A, 2)] = src
    return src
end

"""
    Base.setindex!(A::MatrixMPI{T}, src::MatrixMPI{T}, row_rng::UnitRange{Int}, col_idx::VectorMPI{Int}) where T

Set `A[row_rng, col_idx] = src` where rows are selected by range and columns by VectorMPI.
The `src` must have matching dimensions.
"""
function Base.setindex!(A::MatrixMPI{T}, src::MatrixMPI{T}, row_rng::UnitRange{Int}, col_idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    nrows_src = length(row_rng)

    # Validate row range
    if first(row_rng) < 1 || last(row_rng) > m
        error("MatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end

    # Gather col_idx to all ranks (columns are not distributed)
    col_indices = _gather_vector_to_all(col_idx, comm)
    ncols_src = length(col_indices)

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("MatrixMPI column index out of bounds: $j, ncols=$n")
        end
    end

    # Validate src dimensions
    if size(src, 1) != nrows_src || size(src, 2) != ncols_src
        error("MatrixMPI setindex!: src size ($(size(src))) must match ($nrows_src, $ncols_src)")
    end

    # Compute which rows of src I own vs which rows of A's row_rng I own
    src_partition = src.row_partition
    my_src_start = src_partition[rank + 1]
    my_src_end = src_partition[rank + 2] - 1

    # Map src row indices to global row indices in A
    # src row i corresponds to A row row_rng[i]

    # Group data by destination rank in A
    send_to = [Tuple{Int, Vector{T}}[] for _ in 1:nranks]  # (global_row_in_A, row_data)

    for src_row in my_src_start:my_src_end
        global_row_in_A = row_rng[src_row]
        owner = searchsortedlast(A.row_partition, global_row_in_A) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        local_src_row = src_row - my_src_start + 1
        row_data = Vector{T}(src.A[local_src_row, :])
        push!(send_to[owner + 1], (global_row_in_A, row_data))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, MPI.Isend(indices, comm; dest=r, tag=112))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=112))
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Send row data to destination ranks
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            n_rows = length(send_to[r + 1])
            data = Matrix{T}(undef, n_rows, ncols_src)
            for (k, (_, row_data)) in enumerate(send_to[r + 1])
                data[k, :] .= row_data
            end
            data_send_bufs[r] = data
            push!(data_send_reqs, MPI.Isend(data, comm; dest=r, tag=113))
        end
    end

    # Receive row data from source ranks
    data_recv_bufs = Dict{Int, Matrix{T}}()
    data_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
            push!(data_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=113))
            data_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(data_recv_reqs)

    # Apply local assignments (from my own send_to[rank+1])
    my_A_start = A.row_partition[rank + 1]
    for (global_row_in_A, row_data) in send_to[rank + 1]
        local_row = global_row_in_A - my_A_start + 1
        for (c, global_col) in enumerate(col_indices)
            A.A[local_row, global_col] = row_data[c]
        end
    end

    # Apply received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            data = data_recv_bufs[r]
            for (k, global_row_in_A) in enumerate(indices)
                local_row = global_row_in_A - my_A_start + 1
                for (c, global_col) in enumerate(col_indices)
                    A.A[local_row, global_col] = data[k, c]
                end
            end
        end
    end

    MPI.Waitall(data_send_reqs)

    return src
end

"""
    Base.setindex!(A::MatrixMPI{T}, src::MatrixMPI{T}, ::Colon, col_idx::VectorMPI{Int}) where T

Set `A[:, col_idx] = src` where all rows and columns selected by VectorMPI.
"""
function Base.setindex!(A::MatrixMPI{T}, src::MatrixMPI{T}, ::Colon, col_idx::VectorMPI{Int}) where T
    A[1:size(A, 1), col_idx] = src
    return src
end

"""
    Base.setindex!(A::MatrixMPI{T}, src::VectorMPI{T}, row_idx::VectorMPI{Int}, j::Integer) where T

Set `A[row_idx, j] = src` where rows are selected by VectorMPI and a single column.
The `src` must have partition matching `row_idx.partition`.
"""
function Base.setindex!(A::MatrixMPI{T}, src::VectorMPI{T}, row_idx::VectorMPI{Int}, j::Integer) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Validate column index
    if j < 1 || j > n
        error("MatrixMPI column index out of bounds: $j, ncols=$n")
    end

    # Validate partitions match
    if src.partition != row_idx.partition
        error("MatrixMPI setindex!: src partition must match row_idx partition")
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("MatrixMPI row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    local_src = src.v

    # Group data by destination rank
    send_to = [Tuple{Int, T}[] for _ in 1:nranks]  # (global_row, value)

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(send_to[owner + 1], (global_row, local_src[pos]))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, MPI.Isend(indices, comm; dest=r, tag=114))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=114))
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Send values to destination ranks
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Vector{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            values = [t[2] for t in send_to[r + 1]]
            data_send_bufs[r] = values
            push!(data_send_reqs, MPI.Isend(values, comm; dest=r, tag=115))
        end
    end

    # Receive values from source ranks
    data_recv_bufs = Dict{Int, Vector{T}}()
    data_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{T}(undef, recv_counts[r + 1])
            push!(data_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=115))
            data_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(data_recv_reqs)

    # Apply local assignments (from my own send_to[rank+1])
    my_A_start = A.row_partition[rank + 1]
    for (global_row, value) in send_to[rank + 1]
        local_row = global_row - my_A_start + 1
        A.A[local_row, j] = value
    end

    # Apply received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            values = data_recv_bufs[r]
            for (k, global_row) in enumerate(indices)
                local_row = global_row - my_A_start + 1
                A.A[local_row, j] = values[k]
            end
        end
    end

    MPI.Waitall(data_send_reqs)

    return src
end

"""
    Base.setindex!(A::MatrixMPI{T}, src::VectorMPI{T}, i::Integer, col_idx::VectorMPI{Int}) where T

Set `A[i, col_idx] = src` where a single row and columns selected by VectorMPI.
The `src` must have partition matching `col_idx.partition`.
"""
function Base.setindex!(A::MatrixMPI{T}, src::VectorMPI{T}, i::Integer, col_idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Validate row index
    if i < 1 || i > m
        error("MatrixMPI row index out of bounds: $i, nrows=$m")
    end

    # Validate partitions match
    if src.partition != col_idx.partition
        error("MatrixMPI setindex!: src partition must match col_idx partition")
    end

    # Validate column indices (local check)
    for j in col_idx.v
        if j < 1 || j > n
            error("MatrixMPI column index out of bounds: $j, ncols=$n")
        end
    end

    # Find owner of row i
    owner = searchsortedlast(A.row_partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Gather col_idx and src to owner rank
    col_indices = _gather_vector_to_all(col_idx, comm)
    src_values = _gather_vector_to_all(src, comm)

    # Only owner updates
    if rank == owner
        my_A_start = A.row_partition[rank + 1]
        local_row = i - my_A_start + 1
        for (k, global_col) in enumerate(col_indices)
            A.A[local_row, global_col] = src_values[k]
        end
    end

    return src
end

# ============================================================================
# SparseMatrixMPI mixed setindex!
# ============================================================================

"""
    Base.setindex!(A::SparseMatrixMPI{T}, src::MatrixMPI{T}, row_idx::VectorMPI{Int}, col_rng::UnitRange{Int}) where T

Set `A[row_idx, col_rng] = src` where rows are selected by VectorMPI and columns by range.
The `src` must have row partition matching `row_idx.partition` and column count matching `length(col_rng)`.
"""
function Base.setindex!(A::SparseMatrixMPI{T}, src::MatrixMPI{T}, row_idx::VectorMPI{Int}, col_rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    ncols_src = length(col_rng)

    # Validate column range
    if first(col_rng) < 1 || last(col_rng) > n
        error("SparseMatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    # Validate src dimensions
    if size(src, 2) != ncols_src
        error("SparseMatrixMPI setindex!: src columns ($(size(src, 2))) must match range length ($ncols_src)")
    end
    if src.row_partition != row_idx.partition
        error("SparseMatrixMPI setindex!: src row partition must match row_idx partition")
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("SparseMatrixMPI row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    n_local_src_rows = length(local_row_indices)

    # Group rows by which rank owns the destination in A
    send_to = [Tuple{Int,Int}[] for _ in 1:nranks]
    for local_src_row in 1:n_local_src_rows
        A_global_row = local_row_indices[local_src_row]
        owner = searchsortedlast(A.row_partition, A_global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(send_to[owner + 1], (A_global_row, local_src_row))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, MPI.Isend(indices, comm; dest=r, tag=120))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=120))
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Send row data to destination ranks
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            nrows_to_send = send_counts[r + 1]
            data = Matrix{T}(undef, nrows_to_send, ncols_src)
            for (k, (_, local_src_row)) in enumerate(send_to[r + 1])
                data[k, :] .= src.A[local_src_row, 1:ncols_src]
            end
            send_bufs[r] = data
            push!(send_reqs, MPI.Isend(data, comm; dest=r, tag=121))
        end
    end

    # Receive row data from source ranks
    recv_reqs = MPI.Request[]
    recv_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
            push!(recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=121))
            recv_bufs[r] = buf
        end
    end

    MPI.Waitall(recv_reqs)

    # Build insertions for structural modification
    insertions = Vector{Tuple{Int,Int,T}}()
    col_indices = collect(col_rng)

    # From local assignments
    for (A_global_row, local_src_row) in send_to[rank + 1]
        for (c, g_col) in enumerate(col_indices)
            val = src.A[local_src_row, c]
            push!(insertions, (A_global_row, g_col, val))
        end
    end

    # From received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            data = recv_bufs[r]
            for (k, A_global_row) in enumerate(indices)
                for (c, g_col) in enumerate(col_indices)
                    val = data[k, c]
                    push!(insertions, (A_global_row, g_col, val))
                end
            end
        end
    end

    MPI.Waitall(send_reqs)

    # Apply insertions using the helper function
    if !isempty(insertions)
        row_offset = A.row_partition[rank + 1]
        new_AT, new_col_indices = _rebuild_AT_with_insertions(
            A.A.parent, A.col_indices, insertions, row_offset
        )
        A.col_indices = new_col_indices
        A.A = transpose(new_AT)
    end

    # Invalidate structural hash (will be recomputed lazily on next use)
    A.structural_hash = nothing

    # Invalidate cached transpose bidirectionally
    _invalidate_cached_transpose!(A)

    return src
end

"""
    Base.setindex!(A::SparseMatrixMPI{T}, src::MatrixMPI{T}, row_idx::VectorMPI{Int}, ::Colon) where T

Set `A[row_idx, :] = src` where rows are selected by VectorMPI and all columns.
"""
function Base.setindex!(A::SparseMatrixMPI{T}, src::MatrixMPI{T}, row_idx::VectorMPI{Int}, ::Colon) where T
    A[row_idx, 1:size(A, 2)] = src
    return src
end

"""
    Base.setindex!(A::SparseMatrixMPI{T}, src::MatrixMPI{T}, row_rng::UnitRange{Int}, col_idx::VectorMPI{Int}) where T

Set `A[row_rng, col_idx] = src` where rows are selected by range and columns by VectorMPI.
"""
function Base.setindex!(A::SparseMatrixMPI{T}, src::MatrixMPI{T}, row_rng::UnitRange{Int}, col_idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    nrows_src = length(row_rng)

    # Validate row range
    if first(row_rng) < 1 || last(row_rng) > m
        error("SparseMatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end

    # Gather col_idx to all ranks (columns are not distributed)
    col_indices = _gather_vector_to_all(col_idx, comm)
    ncols_src = length(col_indices)

    # Validate column indices
    for j in col_indices
        if j < 1 || j > n
            error("SparseMatrixMPI column index out of bounds: $j, ncols=$n")
        end
    end

    # Validate src dimensions
    if size(src, 1) != nrows_src || size(src, 2) != ncols_src
        error("SparseMatrixMPI setindex!: src size ($(size(src))) must match ($nrows_src, $ncols_src)")
    end

    # Compute which rows of src I own vs which rows of A's row_rng I own
    src_partition = src.row_partition
    my_src_start = src_partition[rank + 1]
    my_src_end = src_partition[rank + 2] - 1

    # Group data by destination rank in A
    send_to = [Tuple{Int, Int}[] for _ in 1:nranks]  # (global_row_in_A, local_src_row)

    for src_row in my_src_start:my_src_end
        global_row_in_A = row_rng[src_row]
        owner = searchsortedlast(A.row_partition, global_row_in_A) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        local_src_row = src_row - my_src_start + 1
        push!(send_to[owner + 1], (global_row_in_A, local_src_row))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, MPI.Isend(indices, comm; dest=r, tag=122))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=122))
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Send row data to destination ranks
    send_reqs = MPI.Request[]
    send_bufs = Dict{Int, Matrix{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            n_rows = length(send_to[r + 1])
            data = Matrix{T}(undef, n_rows, ncols_src)
            for (k, (_, local_src_row)) in enumerate(send_to[r + 1])
                data[k, :] .= src.A[local_src_row, :]
            end
            send_bufs[r] = data
            push!(send_reqs, MPI.Isend(data, comm; dest=r, tag=123))
        end
    end

    # Receive row data from source ranks
    recv_bufs = Dict{Int, Matrix{T}}()
    recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Matrix{T}(undef, recv_counts[r + 1], ncols_src)
            push!(recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=123))
            recv_bufs[r] = buf
        end
    end

    MPI.Waitall(recv_reqs)

    # Build insertions for structural modification
    insertions = Vector{Tuple{Int,Int,T}}()

    # From local assignments
    for (A_global_row, local_src_row) in send_to[rank + 1]
        for (c, g_col) in enumerate(col_indices)
            val = src.A[local_src_row, c]
            push!(insertions, (A_global_row, g_col, val))
        end
    end

    # From received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            data = recv_bufs[r]
            for (k, A_global_row) in enumerate(indices)
                for (c, g_col) in enumerate(col_indices)
                    val = data[k, c]
                    push!(insertions, (A_global_row, g_col, val))
                end
            end
        end
    end

    MPI.Waitall(send_reqs)

    # Apply insertions using the helper function
    if !isempty(insertions)
        row_offset = A.row_partition[rank + 1]
        new_AT, new_col_indices = _rebuild_AT_with_insertions(
            A.A.parent, A.col_indices, insertions, row_offset
        )
        A.col_indices = new_col_indices
        A.A = transpose(new_AT)
    end

    # Invalidate structural hash (will be recomputed lazily on next use)
    A.structural_hash = nothing

    # Invalidate cached transpose bidirectionally
    _invalidate_cached_transpose!(A)

    return src
end

"""
    Base.setindex!(A::SparseMatrixMPI{T}, src::MatrixMPI{T}, ::Colon, col_idx::VectorMPI{Int}) where T

Set `A[:, col_idx] = src` where all rows and columns selected by VectorMPI.
"""
function Base.setindex!(A::SparseMatrixMPI{T}, src::MatrixMPI{T}, ::Colon, col_idx::VectorMPI{Int}) where T
    A[1:size(A, 1), col_idx] = src
    return src
end

"""
    Base.setindex!(A::SparseMatrixMPI{T}, src::VectorMPI{T}, row_idx::VectorMPI{Int}, j::Integer) where T

Set `A[row_idx, j] = src` where rows are selected by VectorMPI and a single column.
The `src` must have partition matching `row_idx.partition`.
"""
function Base.setindex!(A::SparseMatrixMPI{T}, src::VectorMPI{T}, row_idx::VectorMPI{Int}, j::Integer) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Validate column index
    if j < 1 || j > n
        error("SparseMatrixMPI column index out of bounds: $j, ncols=$n")
    end

    # Validate partitions match
    if src.partition != row_idx.partition
        error("SparseMatrixMPI setindex!: src partition must match row_idx partition")
    end

    # Validate row indices (local check)
    for i in row_idx.v
        if i < 1 || i > m
            error("SparseMatrixMPI row index out of bounds: $i, nrows=$m")
        end
    end

    local_row_indices = row_idx.v
    local_src = src.v

    # Group data by destination rank
    send_to = [Tuple{Int, T}[] for _ in 1:nranks]  # (global_row, value)

    for (pos, global_row) in enumerate(local_row_indices)
        owner = searchsortedlast(A.row_partition, global_row) - 1
        if owner >= nranks
            owner = nranks - 1
        end
        push!(send_to[owner + 1], (global_row, local_src[pos]))
    end

    # Exchange counts via Alltoall
    send_counts = Int32[length(send_to[r + 1]) for r in 0:(nranks-1)]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send row indices to destination ranks
    struct_send_reqs = MPI.Request[]
    struct_send_bufs = Dict{Int, Vector{Int}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            indices = [t[1] for t in send_to[r + 1]]
            struct_send_bufs[r] = indices
            push!(struct_send_reqs, MPI.Isend(indices, comm; dest=r, tag=124))
        end
    end

    # Receive row indices from source ranks
    struct_recv_bufs = Dict{Int, Vector{Int}}()
    struct_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            push!(struct_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=124))
            struct_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(struct_recv_reqs)
    MPI.Waitall(struct_send_reqs)

    # Send values to destination ranks
    data_send_reqs = MPI.Request[]
    data_send_bufs = Dict{Int, Vector{T}}()

    for r in 0:(nranks-1)
        if send_counts[r + 1] > 0 && r != rank
            values = [t[2] for t in send_to[r + 1]]
            data_send_bufs[r] = values
            push!(data_send_reqs, MPI.Isend(values, comm; dest=r, tag=125))
        end
    end

    # Receive values from source ranks
    data_recv_bufs = Dict{Int, Vector{T}}()
    data_recv_reqs = MPI.Request[]

    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{T}(undef, recv_counts[r + 1])
            push!(data_recv_reqs, MPI.Irecv!(buf, comm; source=r, tag=125))
            data_recv_bufs[r] = buf
        end
    end

    MPI.Waitall(data_recv_reqs)

    # Build insertions for structural modification
    insertions = Vector{Tuple{Int,Int,T}}()

    # From local assignments
    for (global_row, value) in send_to[rank + 1]
        push!(insertions, (global_row, j, value))
    end

    # From received assignments
    for r in 0:(nranks-1)
        if recv_counts[r + 1] > 0 && r != rank
            indices = struct_recv_bufs[r]
            values = data_recv_bufs[r]
            for (k, global_row) in enumerate(indices)
                push!(insertions, (global_row, j, values[k]))
            end
        end
    end

    MPI.Waitall(data_send_reqs)

    # Apply insertions using the helper function
    if !isempty(insertions)
        row_offset = A.row_partition[rank + 1]
        new_AT, new_col_indices = _rebuild_AT_with_insertions(
            A.A.parent, A.col_indices, insertions, row_offset
        )
        A.col_indices = new_col_indices
        A.A = transpose(new_AT)
    end

    # Invalidate structural hash (will be recomputed lazily on next use)
    A.structural_hash = nothing

    # Invalidate cached transpose bidirectionally
    _invalidate_cached_transpose!(A)

    return src
end

"""
    Base.setindex!(A::SparseMatrixMPI{T}, src::VectorMPI{T}, i::Integer, col_idx::VectorMPI{Int}) where T

Set `A[i, col_idx] = src` where a single row and columns selected by VectorMPI.
The `src` must have partition matching `col_idx.partition`.
"""
function Base.setindex!(A::SparseMatrixMPI{T}, src::VectorMPI{T}, i::Integer, col_idx::VectorMPI{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Validate row index
    if i < 1 || i > m
        error("SparseMatrixMPI row index out of bounds: $i, nrows=$m")
    end

    # Validate partitions match
    if src.partition != col_idx.partition
        error("SparseMatrixMPI setindex!: src partition must match col_idx partition")
    end

    # Validate column indices (local check)
    for j in col_idx.v
        if j < 1 || j > n
            error("SparseMatrixMPI column index out of bounds: $j, ncols=$n")
        end
    end

    # Find owner of row i
    owner = searchsortedlast(A.row_partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Gather col_idx and src to all ranks (needed because only owner updates)
    col_indices = _gather_vector_to_all(col_idx, comm)
    src_values = _gather_vector_to_all(src, comm)

    # Only owner applies insertions
    if rank == owner
        insertions = Vector{Tuple{Int,Int,T}}()
        for (k, global_col) in enumerate(col_indices)
            push!(insertions, (i, global_col, src_values[k]))
        end

        if !isempty(insertions)
            row_offset = A.row_partition[rank + 1]
            new_AT, new_col_indices = _rebuild_AT_with_insertions(
                A.A.parent, A.col_indices, insertions, row_offset
            )
            A.col_indices = new_col_indices
            A.A = transpose(new_AT)
        end
    end

    # Invalidate structural hash (will be recomputed lazily on next use)
    A.structural_hash = nothing

    # Invalidate cached transpose bidirectionally
    _invalidate_cached_transpose!(A)

    return src
end
