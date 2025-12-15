module LinearAlgebraMPI

using MPI
using Blake3Hash
using SparseArrays
using MUMPS
import SparseArrays: nnz, issparse, dropzeros, spdiagm, blockdiag
import LinearAlgebra
import LinearAlgebra: tr, diag, triu, tril, Transpose, Adjoint, norm, opnorm, mul!, ldlt, BLAS, issymmetric

export SparseMatrixMPI, MatrixMPI, VectorMPI, clear_plan_cache!, uniform_partition
export ⊛  # Multithreaded sparse matrix multiplication
export VectorMPI_local, MatrixMPI_local, SparseMatrixMPI_local  # Local constructors
export mean  # Our mean function for SparseMatrixMPI and VectorMPI
export io0   # Utility for rank-selective output

# Factorization exports (generic interface, implementation details hidden)
export solve, solve!, finalize!

# Type alias for 256-bit Blake3 hash
const Blake3Hash = NTuple{32,UInt8}
const OptionalBlake3Hash = Union{Nothing, Blake3Hash}

# Cache for memoized MatrixPlans
# Key: (A_hash, B_hash, T) - use full 256-bit hashes
const _plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,DataType},Any}()

# Cache for memoized VectorPlans (for A * x)
const _vector_plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,DataType},Any}()

# Cache for memoized Vector Alignment Plans (for u +/- v with different partitions)
const _vector_align_plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,DataType},Any}()

# Cache for memoized DenseMatrixVectorPlans (for MatrixMPI * VectorMPI)
const _dense_vector_plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,DataType},Any}()

# Cache for memoized DenseTransposePlans (for transpose(MatrixMPI))
const _dense_transpose_plan_cache = Dict{Tuple{Blake3Hash,DataType},Any}()

"""
    clear_plan_cache!()

Clear all memoized plan caches.
"""
function clear_plan_cache!()
    empty!(_plan_cache)
    empty!(_vector_plan_cache)
    empty!(_vector_align_plan_cache)
    empty!(_dense_vector_plan_cache)
    empty!(_dense_transpose_plan_cache)
    if isdefined(@__MODULE__, :_dense_transpose_vector_plan_cache)
        empty!(_dense_transpose_vector_plan_cache)
    end
    if isdefined(@__MODULE__, :_addition_plan_cache)
        empty!(_addition_plan_cache)
    end
end

"""
    compute_partition_hash(partition::Vector{Int}) -> Blake3Hash

Compute a hash of a partition vector. Since partition vectors are identical
across all ranks, no MPI communication is needed.
"""
function compute_partition_hash(partition::Vector{Int})::Blake3Hash
    ctx = Blake3Ctx()
    update!(ctx, reinterpret(UInt8, partition))
    return Blake3Hash(digest(ctx))
end

"""
    uniform_partition(n::Int, nranks::Int) -> Vector{Int}

Compute a balanced partition of `n` elements across `nranks` ranks.
Returns a vector of length `nranks + 1` with 1-indexed partition boundaries.

The first `mod(n, nranks)` ranks get `div(n, nranks) + 1` elements,
the remaining ranks get `div(n, nranks)` elements.

# Example
```julia
partition = uniform_partition(10, 4)  # [1, 4, 7, 9, 11]
# Rank 0: 1:3 (3 elements)
# Rank 1: 4:6 (3 elements)
# Rank 2: 7:8 (2 elements)
# Rank 3: 9:10 (2 elements)
```
"""
function uniform_partition(n::Int, nranks::Int)
    per_rank = div(n, nranks)
    remainder = mod(n, nranks)
    partition = Vector{Int}(undef, nranks + 1)
    partition[1] = 1
    for r in 1:nranks
        extra = r <= remainder ? 1 : 0
        partition[r+1] = partition[r] + per_rank + extra
    end
    return partition
end

# Include the component files (order matters: vectors first, then dense/sparse, then blocks, then indexing)
include("vectors.jl")
include("dense.jl")
include("sparse.jl")
include("blocks.jl")
include("indexing.jl")

# Include MUMPS factorization module
include("mumps_factorization.jl")

# ============================================================================
# Symmetry Check
# ============================================================================

"""
    _compare_rows_distributed(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T

Compare two sparse matrices with potentially different row partitions.
Redistributes B's rows to match A's row partition, then compares locally.
Returns true if all corresponding entries are equal.
"""
function _compare_rows_distributed(A::SparseMatrixMPI{T}, B::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # A's local rows
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1
    my_nrows = my_row_end - my_row_start + 1

    # For each of A's local rows, determine which rank owns that row in B
    # B has row_partition = A.col_partition (since B = transpose(A))
    rows_needed_from = [Int[] for _ in 1:nranks]  # rows_needed_from[r+1] = rows we need from rank r
    for row in my_row_start:my_row_end
        owner = searchsortedlast(B.row_partition, row) - 1
        push!(rows_needed_from[owner + 1], row)
    end

    # Exchange: tell each rank which rows we need from them
    send_counts = Int32[length(rows_needed_from[r + 1]) for r in 0:nranks-1]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

    # Send row requests
    send_reqs = MPI.Request[]
    for r in 0:nranks-1
        if send_counts[r + 1] > 0 && r != rank
            req = MPI.Isend(rows_needed_from[r + 1], comm; dest=r, tag=80)
            push!(send_reqs, req)
        end
    end

    # Receive row requests
    rows_to_send = Dict{Int, Vector{Int}}()
    recv_reqs = MPI.Request[]
    for r in 0:nranks-1
        if recv_counts[r + 1] > 0 && r != rank
            buf = Vector{Int}(undef, recv_counts[r + 1])
            req = MPI.Irecv!(buf, comm; source=r, tag=80)
            push!(recv_reqs, req)
            rows_to_send[r] = buf
        end
    end

    MPI.Waitall(vcat(send_reqs, recv_reqs))

    # Now send the actual row data from B
    # For each row, we send: (num_entries, col_indices..., values...)
    BT = B.A.parent  # underlying CSC (columns = local rows of B)
    B_row_start = B.row_partition[rank + 1]

    # Prepare send buffers: pack row data
    send_data = Vector{Vector{UInt8}}(undef, nranks)
    for r in 0:nranks-1
        rows = get(rows_to_send, r, Int[])
        if isempty(rows) || r == rank
            send_data[r + 1] = UInt8[]
            continue
        end

        # Pack all requested rows into a buffer
        io = IOBuffer()
        for global_row in rows
            local_row = global_row - B_row_start + 1
            ptr_start = BT.colptr[local_row]
            ptr_end = BT.colptr[local_row + 1] - 1
            nnz_row = ptr_end - ptr_start + 1

            write(io, Int32(nnz_row))
            for ptr in ptr_start:ptr_end
                global_col = B.col_indices[BT.rowval[ptr]]
                write(io, Int32(global_col))
            end
            for ptr in ptr_start:ptr_end
                write(io, BT.nzval[ptr])
            end
        end
        send_data[r + 1] = take!(io)
    end

    # Exchange message sizes so we know how much to receive
    send_sizes = Int32[length(send_data[r + 1]) for r in 0:nranks-1]
    recv_sizes = MPI.Alltoall(MPI.UBuffer(send_sizes, 1), comm)

    # Now send and receive row data with known sizes
    send_data_reqs = MPI.Request[]
    for r in 0:nranks-1
        if r != rank && send_sizes[r + 1] > 0
            req = MPI.Isend(send_data[r + 1], comm; dest=r, tag=81)
            push!(send_data_reqs, req)
        end
    end

    recv_data = Vector{Vector{UInt8}}(undef, nranks)
    recv_data_reqs = MPI.Request[]
    for r in 0:nranks-1
        if r != rank && recv_sizes[r + 1] > 0
            recv_data[r + 1] = Vector{UInt8}(undef, recv_sizes[r + 1])
            req = MPI.Irecv!(recv_data[r + 1], comm; source=r, tag=81)
            push!(recv_data_reqs, req)
        else
            recv_data[r + 1] = UInt8[]
        end
    end

    MPI.Waitall(vcat(send_data_reqs, recv_data_reqs))

    # Receive row data and compare with A's local rows
    local_match = true
    AT = A.A.parent
    A_row_start = A.row_partition[rank + 1]

    # First handle rows we own in both A and B (rank == rank case)
    for global_row in rows_needed_from[rank + 1]
        local_row_A = global_row - A_row_start + 1
        local_row_B = global_row - B_row_start + 1

        # Get A's row entries
        ptr_start_A = AT.colptr[local_row_A]
        ptr_end_A = AT.colptr[local_row_A + 1] - 1
        nnz_A = ptr_end_A - ptr_start_A + 1

        # Get B's row entries
        ptr_start_B = BT.colptr[local_row_B]
        ptr_end_B = BT.colptr[local_row_B + 1] - 1
        nnz_B = ptr_end_B - ptr_start_B + 1

        if nnz_A != nnz_B
            local_match = false
            break
        end

        # Compare entries (need to handle potentially different orderings)
        A_entries = Dict{Int, T}()
        for ptr in ptr_start_A:ptr_end_A
            global_col = A.col_indices[AT.rowval[ptr]]
            A_entries[global_col] = AT.nzval[ptr]
        end

        for ptr in ptr_start_B:ptr_end_B
            global_col = B.col_indices[BT.rowval[ptr]]
            if !haskey(A_entries, global_col) || A_entries[global_col] != BT.nzval[ptr]
                local_match = false
                break
            end
        end

        if !local_match
            break
        end
    end

    # Now compare rows received from other ranks
    for r in 0:nranks-1
        if r == rank || isempty(recv_data[r + 1])
            continue
        end

        if !local_match
            # Already know it doesn't match, data already received
            continue
        end

        io = IOBuffer(recv_data[r + 1])
        for global_row in rows_needed_from[r + 1]
            local_row_A = global_row - A_row_start + 1

            # Read B's row data
            nnz_B = read(io, Int32)
            B_cols = [read(io, Int32) for _ in 1:nnz_B]
            B_vals = [read(io, T) for _ in 1:nnz_B]

            # Get A's row entries
            ptr_start_A = AT.colptr[local_row_A]
            ptr_end_A = AT.colptr[local_row_A + 1] - 1
            nnz_A = ptr_end_A - ptr_start_A + 1

            if nnz_A != nnz_B
                local_match = false
                break
            end

            A_entries = Dict{Int, T}()
            for ptr in ptr_start_A:ptr_end_A
                global_col = A.col_indices[AT.rowval[ptr]]
                A_entries[global_col] = AT.nzval[ptr]
            end

            for (col, val) in zip(B_cols, B_vals)
                if !haskey(A_entries, col) || A_entries[col] != val
                    local_match = false
                    break
                end
            end

            if !local_match
                break
            end
        end
    end

    # Allreduce to check if all ranks matched
    global_match = MPI.Allreduce(local_match ? 1 : 0, MPI.BAND, comm)
    return global_match == 1
end

"""
    LinearAlgebra.issymmetric(A::SparseMatrixMPI{T}) where T

Check if A is symmetric by materializing the transpose and comparing rows.
Returns true if A == transpose(A).
"""
function LinearAlgebra.issymmetric(A::SparseMatrixMPI{T}) where T
    m, n = size(A)
    if m != n
        return false
    end

    At = materialize_transpose(A)
    return _compare_rows_distributed(A, At)
end

# ============================================================================
# Direct Solve Interface (A \ b)
# ============================================================================

"""
    Base.:\\(A::SparseMatrixMPI{T}, b::VectorMPI{T}) where T

Solve A*x = b using LDLT if A is symmetric, otherwise LU.
For repeated solves, compute the factorization once with `lu(A)` or `ldlt(A)`.
"""
function Base.:\(A::SparseMatrixMPI{T}, b::VectorMPI{T}) where T
    F = issymmetric(A) ? LinearAlgebra.ldlt(A) : LinearAlgebra.lu(A)
    x = F \ b
    finalize!(F)
    return x
end

"""
    Base.:\\(At::Transpose{T,SparseMatrixMPI{T}}, b::VectorMPI{T}) where T

Solve transpose(A)*x = b using LDLT if transpose(A) is symmetric, otherwise LU.
"""
function Base.:\(At::Transpose{T,SparseMatrixMPI{T}}, b::VectorMPI{T}) where T
    A_t = materialize_transpose(At.parent)
    F = issymmetric(A_t) ? LinearAlgebra.ldlt(A_t) : LinearAlgebra.lu(A_t)
    x = F \ b
    finalize!(F)
    return x
end

# ============================================================================
# Right Division Interface (b / A)
# ============================================================================

# Right division: x * A = b, so x = b * A^(-1) = transpose(transpose(A) \ transpose(b))
# For row vectors: transpose(v) / A solves x * A = transpose(v)

"""
    Base.:/(vt::Transpose{T,VectorMPI{T}}, A::SparseMatrixMPI{T}) where T

Solve x * A = transpose(v), returning x as a transposed VectorMPI.
Equivalent to transpose(transpose(A) \\ v).
"""
function Base.:/(vt::Transpose{T,VectorMPI{T}}, A::SparseMatrixMPI{T}) where T
    v = vt.parent
    x = transpose(A) \ v
    return transpose(x)
end

"""
    Base.:/(vt::Transpose{T,VectorMPI{T}}, At::Transpose{T,SparseMatrixMPI{T}}) where T

Solve x * transpose(A) = transpose(v), returning x as a transposed VectorMPI.
"""
function Base.:/(vt::Transpose{T,VectorMPI{T}}, At::Transpose{T,SparseMatrixMPI{T}}) where T
    v = vt.parent
    A = At.parent
    x = A \ v
    return transpose(x)
end

# ============================================================================
# Lazy Hash Computation
# ============================================================================

"""
    _ensure_hash(A::SparseMatrixMPI{T}) -> Blake3Hash

Ensure that the structural hash is computed. If `A.structural_hash` is `nothing`,
compute it and cache it in the struct. Returns the hash.

Note: This function calls `compute_structural_hash` which uses MPI.Allgather,
so all ranks must call this together.
"""
function _ensure_hash(A::SparseMatrixMPI{T})::Blake3Hash where T
    if A.structural_hash === nothing
        A.structural_hash = compute_structural_hash(A.row_partition, A.col_indices, A.A.parent, MPI.COMM_WORLD)
    end
    return A.structural_hash
end

"""
    _ensure_hash(A::MatrixMPI{T}) -> Blake3Hash

Ensure that the structural hash is computed. If `A.structural_hash` is `nothing`,
compute it and cache it in the struct. Returns the hash.

Note: This function calls `compute_dense_structural_hash` which uses MPI.Allgather,
so all ranks must call this together.
"""
function _ensure_hash(A::MatrixMPI{T})::Blake3Hash where T
    if A.structural_hash === nothing
        A.structural_hash = compute_dense_structural_hash(A.row_partition, A.col_partition, size(A.A), MPI.COMM_WORLD)
    end
    return A.structural_hash
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    io0(io=stdout; r::Set{Int}=Set{Int}([0]), dn=devnull)

Return `io` if the current MPI rank is in set `r`, otherwise return `dn` (default: `devnull`).

This is useful for printing only from specific ranks:
```julia
println(io0(), "Hello from rank 0!")
println(io0(r=Set([0,1])), "Hello from ranks 0 and 1!")
```

With string interpolation:
```julia
println(io0(), "Matrix A = \$A")
```
"""
function io0(io::IO=stdout; r::Set{Int}=Set{Int}([0]), dn::IO=devnull)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    return rank ∈ r ? io : dn
end

# ============================================================================
# Conversion Functions: MPI types -> Native Julia types
# ============================================================================

"""
    Vector(v::VectorMPI{T}) where T

Gather a distributed VectorMPI to a full Vector on all ranks.
Requires MPI communication (Allgatherv).
"""
function Base.Vector(v::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)

    # Compute counts per rank
    counts = Int32[v.partition[r+2] - v.partition[r+1] for r in 0:nranks-1]

    # Use Allgatherv to gather the full vector
    full_v = Vector{T}(undef, length(v))
    MPI.Allgatherv!(v.v, MPI.VBuffer(full_v, counts), comm)

    return full_v
end

"""
    Matrix(A::MatrixMPI{T}) where T

Gather a distributed MatrixMPI to a full Matrix on all ranks.
Requires MPI communication (Allgatherv).
"""
function Base.Matrix(A::MatrixMPI{T}) where T
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)

    m, n = size(A)

    # Compute row counts per rank (each rank's local rows * ncols = elements to gather)
    row_counts = Int32[A.row_partition[r+2] - A.row_partition[r+1] for r in 0:nranks-1]
    element_counts = Int32.(row_counts .* n)

    # Allocate full matrix
    full_M = Matrix{T}(undef, m, n)

    # Flatten local matrix (column-major order)
    local_flat = vec(A.A)

    # Gather all flattened matrices
    full_flat = Vector{T}(undef, m * n)
    MPI.Allgatherv!(local_flat, MPI.VBuffer(full_flat, element_counts), comm)

    # Reconstruct full matrix from gathered data
    # Each rank's data is stored row-by-row in column-major chunks
    offset = 0
    for r in 0:nranks-1
        row_start = A.row_partition[r+1]
        row_end = A.row_partition[r+2] - 1
        local_nrows = row_end - row_start + 1
        if local_nrows > 0
            # Reshape rank r's data into (local_nrows, n) and copy to full matrix
            rank_data = reshape(@view(full_flat[offset+1:offset+local_nrows*n]), local_nrows, n)
            full_M[row_start:row_end, :] = rank_data
            offset += local_nrows * n
        end
    end

    return full_M
end

"""
    SparseArrays.SparseMatrixCSC(A::SparseMatrixMPI{T}) where T

Gather a distributed SparseMatrixMPI to a full SparseMatrixCSC on all ranks.
Requires MPI communication (Allgatherv).
"""
function SparseArrays.SparseMatrixCSC(A::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    m, n = size(A)
    my_row_start = A.row_partition[rank+1]

    # Extract local triplets (I, J, V)
    AT = A.A.parent  # underlying CSC storage
    local_nnz = nnz(AT)

    local_I = Vector{Int}(undef, local_nnz)
    local_J = Vector{Int}(undef, local_nnz)
    local_V = Vector{T}(undef, local_nnz)

    idx = 1
    for col in 1:AT.n  # AT.n = number of local rows
        global_row = my_row_start + col - 1
        for ptr in AT.colptr[col]:(AT.colptr[col+1]-1)
            # AT.rowval contains LOCAL column indices, convert to global
            global_col = A.col_indices[AT.rowval[ptr]]
            local_I[idx] = global_row
            local_J[idx] = global_col
            local_V[idx] = AT.nzval[ptr]
            idx += 1
        end
    end

    # Gather counts
    local_count = Int32(local_nnz)
    all_counts = MPI.Allgather(local_count, comm)
    total_nnz = sum(all_counts)

    # Gather all triplets
    global_I = Vector{Int}(undef, total_nnz)
    global_J = Vector{Int}(undef, total_nnz)
    global_V = Vector{T}(undef, total_nnz)

    MPI.Allgatherv!(local_I, MPI.VBuffer(global_I, all_counts), comm)
    MPI.Allgatherv!(local_J, MPI.VBuffer(global_J, all_counts), comm)
    MPI.Allgatherv!(local_V, MPI.VBuffer(global_V, all_counts), comm)

    # Build global sparse matrix
    return sparse(global_I, global_J, global_V, m, n)
end

# ============================================================================
# Show Methods
# ============================================================================

"""
    Base.show(io::IO, v::VectorMPI)

Display a VectorMPI by gathering it to a full vector and showing that.
"""
function Base.show(io::IO, v::VectorMPI{T}) where T
    full_v = Vector(v)
    print(io, "VectorMPI{$T}(")
    show(io, full_v)
    print(io, ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", v::VectorMPI)

Pretty-print a VectorMPI.
"""
function Base.show(io::IO, ::MIME"text/plain", v::VectorMPI{T}) where T
    full_v = Vector(v)
    println(io, length(v), "-element VectorMPI{$T}:")
    show(io, MIME("text/plain"), full_v)
end

"""
    Base.show(io::IO, A::MatrixMPI)

Display a MatrixMPI by gathering it to a full matrix and showing that.
"""
function Base.show(io::IO, A::MatrixMPI{T}) where T
    full_A = Matrix(A)
    print(io, "MatrixMPI{$T}(")
    show(io, full_A)
    print(io, ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", A::MatrixMPI)

Pretty-print a MatrixMPI.
"""
function Base.show(io::IO, ::MIME"text/plain", A::MatrixMPI{T}) where T
    full_A = Matrix(A)
    m, n = size(A)
    println(io, "$m×$n MatrixMPI{$T}:")
    show(io, MIME("text/plain"), full_A)
end

"""
    Base.show(io::IO, A::SparseMatrixMPI)

Display a SparseMatrixMPI by gathering it to a full SparseMatrixCSC and showing that.
"""
function Base.show(io::IO, A::SparseMatrixMPI{T}) where T
    full_A = SparseMatrixCSC(A)
    print(io, "SparseMatrixMPI{$T}(")
    show(io, full_A)
    print(io, ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", A::SparseMatrixMPI)

Pretty-print a SparseMatrixMPI.
"""
function Base.show(io::IO, ::MIME"text/plain", A::SparseMatrixMPI{T}) where T
    full_A = SparseMatrixCSC(A)
    m, n = size(A)
    println(io, "$m×$n SparseMatrixMPI{$T} with $(nnz(full_A)) stored entries:")
    show(io, MIME("text/plain"), full_A)
end

end # module LinearAlgebraMPI
