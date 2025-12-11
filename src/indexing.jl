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
