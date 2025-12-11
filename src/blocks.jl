# ============================================================================
# Block Matrix Primitives - cat, hcat, vcat for distributed matrices
# ============================================================================

# Note: _compute_partition is defined in dense.jl (included before this file)

# ============================================================================
# Core cat implementation for SparseMatrixMPI
# ============================================================================

"""
    Base.cat(As::SparseMatrixMPI...; dims)

Concatenate SparseMatrixMPI matrices.

# Arguments
- `dims::Integer`: Concatenate along dimension `dims` (1=vertical, 2=horizontal)
- `dims::Tuple{Integer,Integer}`: Arrange matrices in a grid of shape `dims=(nrows, ncols)`

# Examples
```julia
cat(A, B; dims=1)           # vcat: stack A on top of B
cat(A, B; dims=2)           # hcat: place A left of B
cat(A, B, C, D; dims=(2,2)) # 2×2 block matrix [A B; C D]
```
"""
function Base.cat(As::SparseMatrixMPI{T}...; dims) where T
    isempty(As) && error("cat requires at least one matrix")
    length(As) == 1 && return copy(As[1])

    # Normalize dims to (nblock_rows, nblock_cols)
    if dims isa Integer
        if dims == 1
            # vcat: n matrices stacked vertically
            block_shape = (length(As), 1)
        elseif dims == 2
            # hcat: n matrices side by side
            block_shape = (1, length(As))
        else
            error("For dims::Integer, only dims=1 (vcat) or dims=2 (hcat) supported")
        end
    elseif dims isa Tuple{Integer, Integer}
        block_shape = dims
    else
        error("dims must be Integer or Tuple{Integer,Integer}")
    end

    nblock_rows, nblock_cols = block_shape
    nblock_rows * nblock_cols == length(As) || error("Number of matrices ($(length(As))) must equal prod(dims) = $(nblock_rows * nblock_cols)")

    # Arrange matrices into grid (row-major order)
    # blocks[i,j] is the matrix at block position (i,j)
    # Use permutedims instead of ' to avoid transposing the matrix elements
    blocks = permutedims(reshape(collect(As), (nblock_cols, nblock_rows)), (2, 1))

    # Validate block sizes: blocks in same row must have same #rows, same column must have same #cols
    block_row_heights = [size(blocks[i, 1], 1) for i in 1:nblock_rows]
    block_col_widths = [size(blocks[1, j], 2) for j in 1:nblock_cols]

    for i in 1:nblock_rows
        for j in 1:nblock_cols
            m, n = size(blocks[i, j])
            m == block_row_heights[i] || error("Block ($i,$j) has $m rows but expected $(block_row_heights[i])")
            n == block_col_widths[j] || error("Block ($i,$j) has $n cols but expected $(block_col_widths[j])")
        end
    end

    # Compute total dimensions
    total_rows = sum(block_row_heights)
    total_cols = sum(block_col_widths)

    # Row and column offsets for each block
    row_offsets = [0; cumsum(block_row_heights[1:end-1])]
    col_offsets = [0; cumsum(block_col_widths[1:end-1])]

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Gather all blocks to global sparse matrix using triplet format
    all_I = Int[]
    all_J = Int[]
    all_V = T[]

    for bi in 1:nblock_rows
        for bj in 1:nblock_cols
            A = blocks[bi, bj]
            row_offset = row_offsets[bi]
            col_offset = col_offsets[bj]

            # Extract local triplets from this block
            my_row_start = A.row_partition[rank + 1]
            col_indices = A.col_indices
            for local_row in 1:size(A.A.parent, 2)
                global_row_in_block = my_row_start + local_row - 1
                global_row = row_offset + global_row_in_block
                for idx in A.A.parent.colptr[local_row]:(A.A.parent.colptr[local_row + 1] - 1)
                    local_col = A.A.parent.rowval[idx]
                    col_in_block = col_indices[local_col]  # map local to global
                    global_col = col_offset + col_in_block
                    push!(all_I, global_row)
                    push!(all_J, global_col)
                    push!(all_V, A.A.parent.nzval[idx])
                end
            end
        end
    end

    # Gather all triplets from all ranks
    local_nnz = Int32(length(all_I))
    all_nnz = MPI.Allgather(local_nnz, comm)

    total_nnz = sum(all_nnz)
    global_I = Vector{Int}(undef, total_nnz)
    global_J = Vector{Int}(undef, total_nnz)
    global_V = Vector{T}(undef, total_nnz)

    MPI.Allgatherv!(all_I, MPI.VBuffer(global_I, all_nnz), comm)
    MPI.Allgatherv!(all_J, MPI.VBuffer(global_J, all_nnz), comm)
    MPI.Allgatherv!(all_V, MPI.VBuffer(global_V, all_nnz), comm)

    # Build global sparse matrix
    global_sparse = sparse(global_I, global_J, global_V, total_rows, total_cols)

    # Create SparseMatrixMPI (constructor computes balanced partition automatically)
    return SparseMatrixMPI{T}(global_sparse)
end

# ============================================================================
# hcat and vcat call cat
# ============================================================================

"""
    Base.hcat(As::SparseMatrixMPI...)

Horizontally concatenate SparseMatrixMPI matrices. Equivalent to `cat(As...; dims=2)`.
"""
Base.hcat(As::SparseMatrixMPI...) = cat(As...; dims=2)

"""
    Base.vcat(As::SparseMatrixMPI...)

Vertically concatenate SparseMatrixMPI matrices. Equivalent to `cat(As...; dims=1)`.
"""
Base.vcat(As::SparseMatrixMPI...) = cat(As...; dims=1)

# ============================================================================
# cat for MatrixMPI (dense)
# ============================================================================

"""
    Base.cat(As::MatrixMPI...; dims)

Concatenate MatrixMPI matrices. Same interface as SparseMatrixMPI version.
"""
function Base.cat(As::MatrixMPI{T}...; dims) where T
    isempty(As) && error("cat requires at least one matrix")
    length(As) == 1 && return copy(As[1])

    # Normalize dims
    if dims isa Integer
        if dims == 1
            block_shape = (length(As), 1)
        elseif dims == 2
            block_shape = (1, length(As))
        else
            error("For dims::Integer, only dims=1 or dims=2 supported")
        end
    elseif dims isa Tuple{Integer, Integer}
        block_shape = dims
    else
        error("dims must be Integer or Tuple{Integer,Integer}")
    end

    nblock_rows, nblock_cols = block_shape
    nblock_rows * nblock_cols == length(As) || error("Number of matrices must equal prod(dims)")

    # Use permutedims instead of ' to avoid transposing the matrix elements
    blocks = permutedims(reshape(collect(As), (nblock_cols, nblock_rows)), (2, 1))

    # Validate sizes
    block_row_heights = [size(blocks[i, 1], 1) for i in 1:nblock_rows]
    block_col_widths = [size(blocks[1, j], 2) for j in 1:nblock_cols]

    for i in 1:nblock_rows, j in 1:nblock_cols
        m, n = size(blocks[i, j])
        m == block_row_heights[i] || error("Block ($i,$j) row size mismatch")
        n == block_col_widths[j] || error("Block ($i,$j) col size mismatch")
    end

    total_rows = sum(block_row_heights)
    total_cols = sum(block_col_widths)
    row_offsets = [0; cumsum(block_row_heights[1:end-1])]
    col_offsets = [0; cumsum(block_col_widths[1:end-1])]

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Gather all blocks to global matrix
    # Each rank contributes its local rows from each block
    my_data = T[]
    my_indices = Int[]  # (global_row, col) pairs encoded

    for bi in 1:nblock_rows
        for bj in 1:nblock_cols
            A = blocks[bi, bj]
            row_offset = row_offsets[bi]
            col_offset = col_offsets[bj]

            my_row_start = A.row_partition[rank + 1]
            my_row_end = A.row_partition[rank + 2] - 1

            for local_row in 1:(my_row_end - my_row_start + 1)
                global_row = row_offset + my_row_start + local_row - 1
                for col in 1:size(A.A, 2)
                    push!(my_indices, global_row)
                    push!(my_indices, col_offset + col)
                    push!(my_data, A.A[local_row, col])
                end
            end
        end
    end

    # Gather counts
    local_count = Int32(length(my_data))
    all_counts = MPI.Allgather(local_count, comm)

    # Gather data
    total_count = sum(all_counts)
    all_data = Vector{T}(undef, total_count)
    MPI.Allgatherv!(my_data, MPI.VBuffer(all_data, all_counts), comm)

    # Gather indices (twice as many as data)
    index_counts = Int32.(all_counts .* 2)
    all_indices = Vector{Int}(undef, sum(index_counts))
    MPI.Allgatherv!(my_indices, MPI.VBuffer(all_indices, index_counts), comm)

    # Build global matrix
    global_matrix = Matrix{T}(undef, total_rows, total_cols)
    fill!(global_matrix, zero(T))

    idx = 1
    for k in 1:length(all_data)
        row = all_indices[2k - 1]
        col = all_indices[2k]
        global_matrix[row, col] = all_data[k]
    end

    # Create MatrixMPI
    return MatrixMPI(global_matrix)
end

Base.hcat(As::MatrixMPI...) = cat(As...; dims=2)
Base.vcat(As::MatrixMPI...) = cat(As...; dims=1)

# ============================================================================
# cat for VectorMPI
# ============================================================================

"""
    Base.cat(vs::VectorMPI...; dims)

Concatenate VectorMPI vectors.
- `dims=1`: vertical concatenation (produces longer VectorMPI)
- `dims=2`: horizontal concatenation (produces MatrixMPI with vectors as columns)
"""
function Base.cat(vs::VectorMPI{T}...; dims) where T
    isempty(vs) && error("cat requires at least one vector")

    if dims isa Integer
        if dims == 1
            return _vcat_vectors(vs...)
        elseif dims == 2
            return _hcat_vectors(vs...)
        else
            error("For VectorMPI, only dims=1 or dims=2 supported")
        end
    elseif dims isa Tuple{Integer, Integer}
        if dims == (1, 1) && length(vs) == 1
            return copy(vs[1])
        elseif dims[1] == 1
            # Single row of vectors -> hcat
            return _hcat_vectors(vs...)
        elseif dims[2] == 1
            # Single column of vectors -> vcat
            return _vcat_vectors(vs...)
        else
            error("VectorMPI cat with dims=$dims not supported (vectors are 1D)")
        end
    else
        error("dims must be Integer or Tuple{Integer,Integer}")
    end
end

function _vcat_vectors(vs::VectorMPI{T}...) where T
    length(vs) == 1 && return copy(vs[1])

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Compute total length and offsets
    lengths = [length(v) for v in vs]
    total_length = sum(lengths)
    offsets = [0; cumsum(lengths[1:end-1])]

    # Gather all contributions with global indices
    my_global_indices = Int[]
    my_values = T[]

    for (vec_idx, v) in enumerate(vs)
        offset = offsets[vec_idx]
        my_start = v.partition[rank + 1]
        my_end = v.partition[rank + 2] - 1
        for (local_idx, global_idx) in enumerate(my_start:my_end)
            push!(my_global_indices, offset + global_idx)
            push!(my_values, v.v[local_idx])
        end
    end

    # Gather
    local_count = Int32(length(my_values))
    all_counts = MPI.Allgather(local_count, comm)

    all_indices = Vector{Int}(undef, sum(all_counts))
    all_values = Vector{T}(undef, sum(all_counts))

    MPI.Allgatherv!(my_global_indices, MPI.VBuffer(all_indices, all_counts), comm)
    MPI.Allgatherv!(my_values, MPI.VBuffer(all_values, all_counts), comm)

    # Build global vector
    global_vector = Vector{T}(undef, total_length)
    for (idx, val) in zip(all_indices, all_values)
        global_vector[idx] = val
    end

    return VectorMPI(global_vector, comm)
end

function _hcat_vectors(vs::VectorMPI{T}...) where T
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)

    # Verify all vectors have same length and partition
    n = length(vs[1])
    row_partition = vs[1].partition
    for (i, v) in enumerate(vs)
        length(v) == n || error("All vectors must have same length for hcat")
        v.partition == row_partition || error("All vectors must have same partition for hcat")
    end

    # Stack local vectors as columns
    local_matrix = hcat([v.v for v in vs]...)
    ncols = length(vs)

    # Compute col partition
    col_partition = _compute_partition(ncols, nranks)

    # Compute structural hash
    structural_hash = compute_dense_structural_hash(row_partition, col_partition, size(local_matrix), comm)

    return MatrixMPI{T}(structural_hash, row_partition, col_partition, local_matrix)
end

Base.hcat(vs::VectorMPI...) = cat(vs...; dims=2)
Base.vcat(vs::VectorMPI...) = cat(vs...; dims=1)

# ============================================================================
# blockdiag for SparseMatrixMPI
# ============================================================================

"""
    blockdiag(As::SparseMatrixMPI...)

Create a block diagonal matrix from the input matrices.

```
blockdiag(A, B, C) = [A 0 0]
                     [0 B 0]
                     [0 0 C]
```

Returns a SparseMatrixMPI.
"""
function blockdiag(As::SparseMatrixMPI{T}...) where T
    isempty(As) && error("blockdiag requires at least one matrix")
    length(As) == 1 && return copy(As[1])

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Compute dimensions and offsets
    row_sizes = [size(A, 1) for A in As]
    col_sizes = [size(A, 2) for A in As]
    total_rows = sum(row_sizes)
    total_cols = sum(col_sizes)
    row_offsets = [0; cumsum(row_sizes[1:end-1])]
    col_offsets = [0; cumsum(col_sizes[1:end-1])]

    # Gather all triplets with diagonal offsets
    all_I = Int[]
    all_J = Int[]
    all_V = T[]

    for (k, A) in enumerate(As)
        row_offset = row_offsets[k]
        col_offset = col_offsets[k]

        my_row_start = A.row_partition[rank + 1]
        col_indices = A.col_indices
        for local_row in 1:size(A.A.parent, 2)
            global_row_in_block = my_row_start + local_row - 1
            global_row = row_offset + global_row_in_block
            for idx in A.A.parent.colptr[local_row]:(A.A.parent.colptr[local_row + 1] - 1)
                local_col = A.A.parent.rowval[idx]
                col_in_block = col_indices[local_col]  # map local to global
                global_col = col_offset + col_in_block
                push!(all_I, global_row)
                push!(all_J, global_col)
                push!(all_V, A.A.parent.nzval[idx])
            end
        end
    end

    # Gather all triplets from all ranks
    local_nnz = Int32(length(all_I))
    all_nnz = MPI.Allgather(local_nnz, comm)

    total_nnz = sum(all_nnz)
    global_I = Vector{Int}(undef, total_nnz)
    global_J = Vector{Int}(undef, total_nnz)
    global_V = Vector{T}(undef, total_nnz)

    MPI.Allgatherv!(all_I, MPI.VBuffer(global_I, all_nnz), comm)
    MPI.Allgatherv!(all_J, MPI.VBuffer(global_J, all_nnz), comm)
    MPI.Allgatherv!(all_V, MPI.VBuffer(global_V, all_nnz), comm)

    # Build global sparse matrix
    global_sparse = sparse(global_I, global_J, global_V, total_rows, total_cols)

    return SparseMatrixMPI{T}(global_sparse)
end
