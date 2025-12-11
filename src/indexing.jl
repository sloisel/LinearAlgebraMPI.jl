# Indexing operations for distributed types
# Communication tags: 40 (index request), 41 (value response)

"""
    _mpi_abort_with_stacktrace(msg::String)

Print an error message with stacktrace and abort all MPI processes.
Used for unrecoverable errors like structural modifications.
"""
function _mpi_abort_with_stacktrace(msg::String)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    io = stderr
    println(io, "ERROR on rank $rank: $msg")
    println(io, "Stacktrace:")
    for (i, frame) in enumerate(stacktrace())
        println(io, "  [$i] $frame")
    end
    flush(io)
    MPI.Abort(comm, 1)
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
        _mpi_abort_with_stacktrace("VectorMPI index out of bounds: i=$i, length=$n")
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
        _mpi_abort_with_stacktrace("VectorMPI index out of bounds: i=$i, length=$n")
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

    # Barrier to ensure all ranks stay synchronized
    MPI.Barrier(comm)

    return val
end

# ============================================================================
# SparseMatrixMPI Indexing
# ============================================================================

"""
    _find_csc_entry(AT::SparseMatrixCSC, local_row::Int, local_col::Int)

Find the position of entry (local_row, local_col) in a CSC matrix.
Returns the nzval index if found, or 0 if the entry is structurally zero.

Uses binary search since rowval entries within each column are sorted.
Note: local_row is the row index in the CSC (which corresponds to local column index
in the compressed col_indices), and local_col is the column in CSC (which corresponds
to local row index in the original matrix).
"""
function _find_csc_entry(AT::SparseMatrixCSC, local_row::Int, local_col::Int)
    # Get the range of entries for this column
    col_start = AT.colptr[local_col]
    col_end = AT.colptr[local_col + 1] - 1

    if col_start > col_end
        return 0  # Empty column
    end

    # Binary search for local_row in rowval[col_start:col_end]
    # Note: searchsortedfirst returns first index >= target
    rowval_view = view(AT.rowval, col_start:col_end)
    idx = searchsortedfirst(rowval_view, local_row)

    if idx <= length(rowval_view) && rowval_view[idx] == local_row
        return col_start + idx - 1  # Convert back to absolute index
    end

    return 0  # Entry not found (structural zero)
end

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
        _mpi_abort_with_stacktrace("SparseMatrixMPI row index out of bounds: i=$i, nrows=$m")
    end
    if j < 1 || j > n
        _mpi_abort_with_stacktrace("SparseMatrixMPI column index out of bounds: j=$j, ncols=$n")
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
            # Column j exists in our sparsity pattern
            # In AT: rows are compressed col_indices, columns are local rows
            nzval_idx = _find_csc_entry(AT, local_col_idx, local_row)
            if nzval_idx > 0
                buf[1] = AT.nzval[nzval_idx]
            else
                buf[1] = zero(T)  # Structural zero (column exists but not in this row)
            end
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

This is a collective operation - all ranks must call it with the same indices and value.
Only the owning rank (owner of row i) actually modifies the data.

**Important**: This can only set values at existing structural positions.
Attempting to set a structural zero (create a new nonzero) will abort with an error,
since that would change the matrix structure and invalidate its hash.

# Example
```julia
A = SparseMatrixMPI{Float64}(sparse([1, 2], [1, 2], [1.0, 2.0], 3, 3))
A[1, 1] = 5.0  # OK - modifies existing entry
A[1, 2] = 5.0  # ERROR - would create new nonzero (structural modification)
```
"""
function Base.setindex!(A::SparseMatrixMPI{T}, val, i::Integer, j::Integer) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    if i < 1 || i > m
        _mpi_abort_with_stacktrace("SparseMatrixMPI row index out of bounds: i=$i, nrows=$m")
    end
    if j < 1 || j > n
        _mpi_abort_with_stacktrace("SparseMatrixMPI column index out of bounds: j=$j, ncols=$n")
    end

    # Find owner of row i using binary search on row_partition
    owner = searchsortedlast(A.row_partition, i) - 1
    if owner >= nranks
        owner = nranks - 1
    end

    # Use a 1-element buffer for broadcasting the success status
    success_buf = Int32[1]  # 1 = success, 0 = failure

    # Owner attempts to set the value
    if rank == owner
        local_row = i - A.row_partition[owner + 1] + 1
        AT = A.A.parent  # The underlying CSC storage

        # Find j in col_indices using binary search (col_indices is sorted)
        local_col_idx = searchsortedfirst(A.col_indices, j)

        if local_col_idx <= length(A.col_indices) && A.col_indices[local_col_idx] == j
            # Column j exists in our sparsity pattern
            nzval_idx = _find_csc_entry(AT, local_col_idx, local_row)
            if nzval_idx > 0
                AT.nzval[nzval_idx] = convert(T, val)
                success_buf[1] = Int32(1)  # Success
            else
                success_buf[1] = Int32(0)  # Would need to create new structural entry
            end
        else
            success_buf[1] = Int32(0)  # Column j not in sparsity pattern
        end
    end

    # Broadcast success status from owner
    MPI.Bcast!(success_buf, owner, comm)

    if success_buf[1] == Int32(0)
        _mpi_abort_with_stacktrace(
            "SparseMatrixMPI structural modification not allowed: " *
            "A[$i, $j] = $val would create a new nonzero entry. " *
            "Use a matrix with the required sparsity pattern instead."
        )
    end

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
        _mpi_abort_with_stacktrace("MatrixMPI row index out of bounds: i=$i, nrows=$m")
    end
    if j < 1 || j > n
        _mpi_abort_with_stacktrace("MatrixMPI column index out of bounds: j=$j, ncols=$n")
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
        _mpi_abort_with_stacktrace("MatrixMPI row index out of bounds: i=$i, nrows=$m")
    end
    if j < 1 || j > n
        _mpi_abort_with_stacktrace("MatrixMPI column index out of bounds: j=$j, ncols=$n")
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

    # Barrier to ensure all ranks stay synchronized
    MPI.Barrier(comm)

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
        _mpi_abort_with_stacktrace("VectorMPI range out of bounds: $rng, length=$n")
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
        _mpi_abort_with_stacktrace("VectorMPI range out of bounds: $rng, length=$n")
    end

    if isempty(rng)
        MPI.Barrier(comm)
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

    MPI.Barrier(comm)
    return val
end

function Base.setindex!(v::VectorMPI{T}, vals::AbstractVector, rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    n = length(v)
    if first(rng) < 1 || last(rng) > n
        _mpi_abort_with_stacktrace("VectorMPI range out of bounds: $rng, length=$n")
    end

    if length(vals) != length(rng)
        _mpi_abort_with_stacktrace("VectorMPI setindex!: length mismatch, got $(length(vals)) values for range of length $(length(rng))")
    end

    if isempty(rng)
        MPI.Barrier(comm)
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

    MPI.Barrier(comm)
    return vals
end

function Base.setindex!(v::VectorMPI{T}, src::VectorMPI, rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    n = length(v)
    if first(rng) < 1 || last(rng) > n
        _mpi_abort_with_stacktrace("VectorMPI range out of bounds: $rng, length=$n")
    end

    if length(src) != length(rng)
        _mpi_abort_with_stacktrace("VectorMPI setindex!: length mismatch, got $(length(src)) values for range of length $(length(rng))")
    end

    if isempty(rng)
        MPI.Barrier(comm)
        return src
    end

    # Compute the partition of the target range
    target_partition = _compute_subpartition(v.partition, rng)

    # Check if source partition matches target partition
    if src.partition == target_partition
        # Direct local copy
        my_start = v.partition[rank + 1]
        my_end = v.partition[rank + 2] - 1

        intersect_start = max(my_start, first(rng))
        intersect_end = min(my_end, last(rng))

        if intersect_start <= intersect_end
            local_start = intersect_start - my_start + 1
            local_end = intersect_end - my_start + 1
            v.v[local_start:local_end] .= convert.(T, src.v)
        end
    else
        # Need to align src to target_partition
        plan = VectorPlan(target_partition, src)
        aligned = execute_plan!(plan, src)

        my_start = v.partition[rank + 1]
        my_end = v.partition[rank + 2] - 1

        intersect_start = max(my_start, first(rng))
        intersect_end = min(my_end, last(rng))

        if intersect_start <= intersect_end
            local_start = intersect_start - my_start + 1
            local_end = intersect_end - my_start + 1
            v.v[local_start:local_end] .= convert.(T, aligned)
        end
    end

    MPI.Barrier(comm)
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
        _mpi_abort_with_stacktrace("MatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        _mpi_abort_with_stacktrace("MatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    if isempty(row_rng) || isempty(col_rng)
        # Empty range - return empty MatrixMPI
        new_row_partition = ones(Int, nranks + 1)
        new_col_partition = _compute_partition(0, nranks)
        hash = compute_dense_structural_hash(new_row_partition, new_col_partition, (0, 0), comm)
        return MatrixMPI{T}(hash, new_row_partition, new_col_partition, Matrix{T}(undef, 0, 0))
    end

    # Compute new row partition (local computation, no communication)
    new_row_partition = _compute_subpartition(A.row_partition, row_rng)

    # Compute new column partition (standard even distribution for the submatrix column count)
    new_ncols = length(col_rng)
    new_col_partition = _compute_partition(new_ncols, nranks)

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
        _mpi_abort_with_stacktrace("MatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        _mpi_abort_with_stacktrace("MatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    if isempty(row_rng) || isempty(col_rng)
        MPI.Barrier(comm)
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

    MPI.Barrier(comm)
    return val
end

function Base.setindex!(A::MatrixMPI{T}, vals::AbstractMatrix, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        _mpi_abort_with_stacktrace("MatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        _mpi_abort_with_stacktrace("MatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    if size(vals) != (length(row_rng), length(col_rng))
        _mpi_abort_with_stacktrace("MatrixMPI setindex!: size mismatch, got $(size(vals)) for range of size ($(length(row_rng)), $(length(col_rng)))")
    end

    if isempty(row_rng) || isempty(col_rng)
        MPI.Barrier(comm)
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

    MPI.Barrier(comm)
    return vals
end

function Base.setindex!(A::MatrixMPI{T}, src::MatrixMPI, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        _mpi_abort_with_stacktrace("MatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        _mpi_abort_with_stacktrace("MatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    if size(src) != (length(row_rng), length(col_rng))
        _mpi_abort_with_stacktrace("MatrixMPI setindex!: size mismatch, got $(size(src)) for range of size ($(length(row_rng)), $(length(col_rng)))")
    end

    if isempty(row_rng) || isempty(col_rng)
        MPI.Barrier(comm)
        return src
    end

    # Compute the partition of the target range
    target_row_partition = _compute_subpartition(A.row_partition, row_rng)

    # Check if source partition matches target partition
    if src.row_partition == target_row_partition
        # Direct local copy
        my_row_start = A.row_partition[rank + 1]
        my_row_end = A.row_partition[rank + 2] - 1

        intersect_start = max(my_row_start, first(row_rng))
        intersect_end = min(my_row_end, last(row_rng))

        if intersect_start <= intersect_end
            local_row_start = intersect_start - my_row_start + 1
            local_row_end = intersect_end - my_row_start + 1
            # Need to handle column range mapping in src
            A.A[local_row_start:local_row_end, col_rng] .= convert.(T, src.A[:, 1:length(col_rng)])
        end
    else
        # Partitions don't match - need communication
        _mpi_abort_with_stacktrace("MatrixMPI setindex! with MatrixMPI source: partition mismatch requires communication (not yet implemented)")
    end

    MPI.Barrier(comm)
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
function Base.getindex(A::SparseMatrixMPI{T}, row_rng::UnitRange{Int}, col_rng::UnitRange{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    if first(row_rng) < 1 || last(row_rng) > m
        _mpi_abort_with_stacktrace("SparseMatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        _mpi_abort_with_stacktrace("SparseMatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    new_nrows = length(row_rng)
    new_ncols = length(col_rng)

    if isempty(row_rng) || isempty(col_rng)
        # Empty range - return empty SparseMatrixMPI
        new_row_partition = ones(Int, nranks + 1)
        new_col_partition = _compute_partition(0, nranks)
        empty_AT = SparseMatrixCSC(0, 0, Int[1], Int[], T[])
        hash = compute_structural_hash(new_row_partition, Int[], empty_AT, comm)
        return SparseMatrixMPI{T}(hash, new_row_partition, new_col_partition, Int[],
                                   transpose(empty_AT), Ref{Union{Nothing, SparseMatrixMPI{T}}}(nothing))
    end

    # Compute new row partition (local computation, no communication)
    new_row_partition = _compute_subpartition(A.row_partition, row_rng)

    # Compute new column partition (standard even distribution)
    new_col_partition = _compute_partition(new_ncols, nranks)

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
            # Map positions to compressed 1-based indices
            pos_to_compressed = Dict(p => i for (i, p) in enumerate(unique_positions))
            compressed_rowval = [pos_to_compressed[r] for r in rowval_list]
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

    return SparseMatrixMPI{T}(hash, new_row_partition, new_col_partition, final_col_indices,
                               transpose(new_AT), Ref{Union{Nothing, SparseMatrixMPI{T}}}(nothing))
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
        _mpi_abort_with_stacktrace("SparseMatrixMPI row range out of bounds: $row_rng, nrows=$m")
    end
    if first(col_rng) < 1 || last(col_rng) > n
        _mpi_abort_with_stacktrace("SparseMatrixMPI column range out of bounds: $col_rng, ncols=$n")
    end

    if isempty(row_rng) || isempty(col_rng)
        MPI.Barrier(comm)
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

    MPI.Barrier(comm)
    return val
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
