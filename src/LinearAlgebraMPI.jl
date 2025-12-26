module LinearAlgebraMPI

using MPI
using Blake3Hash
using SparseArrays
using MUMPS
using StaticArrays
import SparseArrays: nnz, issparse, dropzeros, spdiagm, blockdiag
import LinearAlgebra
import LinearAlgebra: tr, diag, triu, tril, Transpose, Adjoint, norm, opnorm, mul!, ldlt, BLAS, issymmetric, UniformScaling, dot, Symmetric

export SparseMatrixMPI, MatrixMPI, VectorMPI, clear_plan_cache!, uniform_partition, repartition
export VectorMPI_CPU, MatrixMPI_CPU, SparseMatrixMPI_CPU  # Type aliases for CPU-backed types
export SparseMatrixCSR  # Type alias for Transpose{SparseMatrixCSC} (CSR storage format)
export map_rows  # Row-wise map over distributed vectors/matrices
export VectorMPI_local, MatrixMPI_local, SparseMatrixMPI_local  # Local constructors
export mean  # Our mean function for SparseMatrixMPI and VectorMPI
export io0   # Utility for rank-selective output
export get_backend  # Get the KernelAbstractions backend for a distributed array

# GPU conversion functions (extended by Metal/CUDA extensions)
export mtl, cpu

# Factorization exports (generic interface, implementation details hidden)
export solve, solve!, finalize!, clear_mumps_analysis_cache!

# Type alias for 256-bit Blake3 hash
const Blake3Hash = NTuple{32,UInt8}
const OptionalBlake3Hash = Union{Nothing, Blake3Hash}

# ============================================================================
# SparseMatrixCSR Type Alias and Constructors
# ============================================================================

"""
    SparseMatrixCSR{Tv,Ti} = Transpose{Tv, SparseMatrixCSC{Tv,Ti}}

Type alias for CSR (Compressed Sparse Row) storage format.

## The Dual Life of Transpose{SparseMatrixCSC}

In Julia, the type `Transpose{Tv, SparseMatrixCSC{Tv,Ti}}` has two interpretations:

1. **Semantic interpretation**: A lazy transpose wrapper around a CSC matrix.
   When you call `transpose(A)` on a SparseMatrixCSC, you get this wrapper that
   represents A^T without copying data.

2. **Storage interpretation**: CSR (row-major) access to sparse data.
   The underlying CSC stores columns contiguously, but through the transpose wrapper,
   we can iterate efficiently over rows instead of columns.

This alias clarifies intent: use `SparseMatrixCSR` when you want row-major storage
semantics, and `transpose(A)` when you want the mathematical transpose.

## CSR vs CSC Storage

- **CSC (Compressed Sparse Column)**: Julia's native sparse format. Efficient for
  column-wise operations, matrix-vector products with column access.
- **CSR (Compressed Sparse Row)**: Efficient for row-wise operations, matrix-vector
  products with row access, and row-partitioned distributed matrices.

For `SparseMatrixCSR`, the underlying `parent::SparseMatrixCSC` stores the *transposed*
matrix. If `B = SparseMatrixCSR(A)` represents matrix M, then `B.parent` is a CSC
storing M^T. This means:
- `B.parent.colptr` acts as row pointers for M
- `B.parent.rowval` contains column indices for M
- `B.parent.nzval` contains values in row-major order

## Usage Note

Julia will still display this type as `Transpose{Float64, SparseMatrixCSC{...}}`,
not as `SparseMatrixCSR`. The alias improves code clarity but doesn't affect
type printing.
"""
const SparseMatrixCSR{Tv,Ti} = Transpose{Tv, SparseMatrixCSC{Tv,Ti}}

"""
    SparseMatrixCSR(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

Convert a CSC matrix to CSR format representing the **same** matrix.

If A represents matrix M in CSC format, the result represents M in CSR format.
Element access is unchanged: `B[i,j] == A[i,j]`.

Internally, this:
1. Materializes A^T as CSC (physical transpose)
2. Wraps in lazy transpose to get M back, but with row-major storage

# Example
```julia
A_csc = sparse([1,2,2], [1,1,2], [1.0, 2.0, 3.0], 2, 2)
A_csr = SparseMatrixCSR(A_csc)  # Same matrix, CSR storage
A_csr[1,1] == A_csc[1,1]        # true - same elements
```
"""
function SparseMatrixCSR(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    return transpose(SparseMatrixCSC(transpose(A)))
end

"""
    SparseMatrixCSC(A::SparseMatrixCSR{Tv,Ti}) where {Tv,Ti}

Convert a CSR matrix to CSC format representing the **same** matrix.

This physically transposes the underlying storage to produce a CSC matrix.
Element access is unchanged: the result represents the same matrix as the input.
"""
function SparseArrays.SparseMatrixCSC(A::SparseMatrixCSR{Tv,Ti}) where {Tv,Ti}
    # Use sparse() to avoid dispatching back to our method
    return sparse(transpose(A.parent))
end

# Cache for memoized MatrixPlans
# Key: (A_hash, B_hash, T, Ti, AV) - use full 256-bit hashes, includes array type for GPU support
const _plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,DataType,DataType,Type},Any}()

# Cache for memoized VectorPlans (for A * x)
# Key: (A_hash, x_hash, T, AV) - includes array type for GPU support
# Uses Type instead of DataType to support GPU UnionAll types like MtlVector{Float32}
const _vector_plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,Type,Type},Any}()

# Cache for memoized DenseMatrixVectorPlans (for MatrixMPI * VectorMPI)
# Key: (A_hash, x_hash, T, AM, AV) - includes matrix and vector array types for GPU support
# Uses Type instead of DataType to support GPU UnionAll types like MtlMatrix{Float32}
const _dense_vector_plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,Type,Type,Type},Any}()

# Cache for memoized DenseTransposePlans (for transpose(MatrixMPI))
const _dense_transpose_plan_cache = Dict{Tuple{Blake3Hash,Type},Any}()

# Cache for memoized RepartitionPlans (for repartition)
# Key includes (hash_A, target_hash, T, Ti) for sparse and (hash_A, target_hash, T) for others
const _repartition_plan_cache = Dict{Any,Any}()

# Cache for diagonal matrix structure
# Key: vector's structural hash (which encodes its partition)
# Value: DiagStructureCache with colptr, rowval, col_indices, and structural hash
struct DiagStructureCache
    colptr::Vector{Int}
    rowval::Vector{Int}
    col_indices::Vector{Int}
    structural_hash::Blake3Hash
end
const _diag_structure_cache = Dict{Blake3Hash,DiagStructureCache}()

# Cache for memoized AdditionPlans (for A + B and A - B)
# Key: (A_hash, B_hash, T, Ti, AV) - use full 256-bit hashes, includes array type for GPU support
const _addition_plan_cache = Dict{Tuple{Blake3Hash,Blake3Hash,DataType,DataType,Type},Any}()

# Cache for memoized IdentityAdditionPlans (for A + λI)
# Key: (A_hash, T, Ti) - use full 256-bit hash
const _identity_addition_plan_cache = Dict{Tuple{Blake3Hash,DataType,DataType},Any}()

# ============================================================================
# GPU Conversion Functions (stubs - extended by Metal/CUDA extensions)
# ============================================================================

"""
    mtl(v)

Convert a distributed array to Metal GPU storage.
Requires the Metal package to be loaded.
"""
function mtl end

"""
    cpu(v)

Convert a distributed array from GPU to CPU storage.
"""
function cpu end

"""
    clear_plan_cache!()

Clear all memoized plan caches, including the MUMPS analysis cache.
This is a collective operation that must be called on all MPI ranks together.
"""
function clear_plan_cache!()
    empty!(_plan_cache)
    empty!(_vector_plan_cache)
    empty!(_dense_vector_plan_cache)
    empty!(_dense_transpose_plan_cache)
    empty!(_repartition_plan_cache)
    empty!(_diag_structure_cache)
    empty!(_addition_plan_cache)
    empty!(_identity_addition_plan_cache)
    if isdefined(@__MODULE__, :_dense_transpose_vector_plan_cache)
        empty!(_dense_transpose_vector_plan_cache)
    end
    # Also clear MUMPS analysis cache (defined in mumps_factorization.jl)
    if isdefined(@__MODULE__, :clear_mumps_analysis_cache!)
        clear_mumps_analysis_cache!()
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
    BT = _get_csc(B)  # underlying CSC (columns = local rows of B)
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
    AT = _get_csc(A)
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
Returns true if A == transpose(A). Result is cached for subsequent calls.
"""
function LinearAlgebra.issymmetric(A::SparseMatrixMPI{T}) where T
    # Return cached result if available
    if A.cached_symmetric !== nothing
        return A.cached_symmetric
    end

    m, n = size(A)
    if m != n
        A.cached_symmetric = false
        return false
    end

    At = SparseMatrixMPI(transpose(A))
    result = _compare_rows_distributed(A, At)
    A.cached_symmetric = result
    return result
end

# ============================================================================
# Direct Solve Interface (A \ b)
# ============================================================================

"""
    Base.:\\(A::SparseMatrixMPI{T}, b::VectorMPI{T}) where T

Solve A*x = b using LU factorization.
For symmetric matrices, use `Symmetric(A) \\ b` to use the faster LDLT factorization.
For repeated solves, compute the factorization once with `lu(A)` or `ldlt(A)`.
"""
function Base.:\(A::SparseMatrixMPI{T}, b::VectorMPI{T}) where T
    F = LinearAlgebra.lu(A)
    x = F \ b
    finalize!(F)
    return x
end

"""
    Base.:\\(A::Symmetric{T,<:SparseMatrixMPI{T}}, b::VectorMPI{T}) where T

Solve A*x = b for a symmetric matrix using LDLT (no symmetry check needed).
Use `Symmetric(A)` to wrap a known-symmetric matrix and skip the expensive symmetry check.
"""
function Base.:\(A::Symmetric{T,<:SparseMatrixMPI{T}}, b::VectorMPI{T}) where T
    F = LinearAlgebra.ldlt(parent(A))
    x = F \ b
    finalize!(F)
    return x
end

"""
    Base.:\\(At::Transpose{T,<:SparseMatrixMPI{T}}, b::VectorMPI{T}) where T

Solve transpose(A)*x = b using LU factorization.
"""
function Base.:\(At::Transpose{T,<:SparseMatrixMPI{T}}, b::VectorMPI{T}) where T
    A_t = SparseMatrixMPI(At)
    F = LinearAlgebra.lu(A_t)
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
    Base.:/(vt::Transpose{T,VectorMPI{T}}, At::Transpose{T,<:SparseMatrixMPI{T}}) where T

Solve x * transpose(A) = transpose(v), returning x as a transposed VectorMPI.
"""
function Base.:/(vt::Transpose{T,VectorMPI{T}}, At::Transpose{T,<:SparseMatrixMPI{T}}) where T
    v = vt.parent
    A = At.parent
    x = A \ v
    return transpose(x)
end

# ============================================================================
# Lazy Hash Computation
# ============================================================================

"""
    _ensure_hash(A::SparseMatrixMPI) -> Blake3Hash

Ensure that the structural hash is computed. If `A.structural_hash` is `nothing`,
compute it and cache it in the struct. Returns the hash.

Note: This function calls `compute_structural_hash` which uses MPI.Allgather,
so all ranks must call this together.
"""
function _ensure_hash(A::SparseMatrixMPI)::Blake3Hash
    if A.structural_hash === nothing
        A.structural_hash = compute_structural_hash(A.row_partition, A.col_indices, A.rowptr, A.colval, MPI.COMM_WORLD)
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

    # Ensure local data is on CPU for MPI (GPU arrays not supported by MPI.jl)
    v_cpu = _ensure_cpu(v.v)

    # Use Allgatherv to gather the full vector
    full_v = Vector{T}(undef, length(v))
    MPI.Allgatherv!(v_cpu, MPI.VBuffer(full_v, counts), comm)

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
    # Ensure data is on CPU for MPI (GPU arrays not supported by MPI.jl)
    local_flat = _ensure_cpu(vec(A.A))

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
    AT = _get_csc(A)  # underlying CSC storage
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

# ============================================================================
# map_rows: Row-wise map over distributed vectors/matrices
# ============================================================================

"""
    _get_row_partition(A::VectorMPI) -> Vector{Int}
    _get_row_partition(A::MatrixMPI) -> Vector{Int}

Get the row partition from a distributed type.
"""
_get_row_partition(A::VectorMPI) = A.partition
_get_row_partition(A::MatrixMPI) = A.row_partition

"""
    _align_to_partition(A::VectorMPI{T}, p::Vector{Int}) where T
    _align_to_partition(A::MatrixMPI{T}, p::Vector{Int}) where T

Repartition a distributed type to match partition p.
"""
_align_to_partition(A::VectorMPI, p::Vector{Int}) = repartition(A, p)
_align_to_partition(A::MatrixMPI, p::Vector{Int}) = repartition(A, p)

# Helper: Convert local matrix to Vector{SVector} by transposing then reinterpreting
# For column-major storage, transpose gives us rows as contiguous columns
function _matrix_to_svectors(::Val{K}, M::AbstractMatrix{T}) where {K, T}
    # M is (nrows, K) in column-major. transpose(M) is (K, nrows) column-major.
    # Each column of transpose(M) is a row of M, which can be reinterpreted as SVector{K,T}
    MT = copy(transpose(M))  # Materialize to contiguous memory (stays on GPU if M is GPU)
    vec(reinterpret(reshape, SVector{K, T}, MT))
end
_matrix_to_svectors(M::AbstractMatrix{T}) where {T} = _matrix_to_svectors(Val(size(M, 2)), M)

# Helper: Convert Vector{SVector} back to Matrix
function _svectors_to_matrix(v::AbstractVector{SVector{K,T}}) where {K,T}
    # reinterpret as (K, nrows), then transpose to (nrows, K)
    # Use copy(transpose(...)) to preserve GPU array type (not Matrix(...) which forces CPU)
    copy(transpose(reinterpret(reshape, T, v)))
end

# Helper: Convert to SVector representation for map_rows
# Vectors pass through as-is (each element is a "row")
_to_svectors(v::AbstractVector{T}) where {T<:Number} = v
# Matrices get converted to Vector{SVector}
_to_svectors(M::AbstractMatrix{T}) where {T} = _matrix_to_svectors(M)

# Helper: Convert result back based on type
_from_result(v::AbstractVector{SVector{K,T}}) where {K,T} = _svectors_to_matrix(v)
_from_result(v::AbstractVector{T}) where {T<:Number} = v

"""
    map_rows(f, A...)

Apply function `f` to corresponding rows of distributed vectors/matrices.

Each argument in `A...` must be either a `VectorMPI` or `MatrixMPI`. All inputs
are repartitioned to match the partition of the first argument before applying `f`.

This implementation uses GPU-friendly broadcasting: matrices are converted to
Vector{SVector} via transpose+reinterpret, then f is broadcast over all arguments.
This avoids GPU->CPU->GPU round-trips when the underlying arrays are on GPU.

For each row index i, `f` is called with:
- For `VectorMPI`: the scalar element at index i
- For `MatrixMPI`: an SVector containing the i-th row

## Result Type

The result type depends on what `f` returns:

| `f` returns | Result |
|-------------|--------|
| scalar (`Number`) | `VectorMPI` (one element per input row) |
| `SVector{K,T}` | `MatrixMPI` (K columns, one row per input row) |

## Examples

```julia
# Element-wise product of two vectors
u = VectorMPI([1.0, 2.0, 3.0])
v = VectorMPI([4.0, 5.0, 6.0])
w = map_rows((a, b) -> a * b, u, v)  # VectorMPI([4.0, 10.0, 18.0])

# Row norms of a matrix
A = MatrixMPI(randn(5, 3))
norms = map_rows(r -> norm(r), A)  # VectorMPI of row norms

# Return SVector to build a matrix
A = MatrixMPI(randn(3, 2))
result = map_rows(r -> SVector(sum(r), prod(r)), A)  # 3×2 MatrixMPI

# Mixed inputs: matrix rows combined with vector elements
A = MatrixMPI(randn(4, 3))
w = VectorMPI([1.0, 2.0, 3.0, 4.0])
result = map_rows((row, wi) -> sum(row) * wi, A, w)  # VectorMPI
```
"""
function map_rows(f, A...)
    isempty(A) && error("map_rows requires at least one argument")

    # Get target partition from first argument
    target_partition = _get_row_partition(A[1])

    # Align all arguments to target partition
    aligned = map(a -> _align_to_partition(a, target_partition), A)

    # Convert to SVector representation for broadcasting
    # VectorMPI.v passes through, MatrixMPI.A gets transposed and reinterpreted
    local_arrays = map(aligned) do a
        if a isa VectorMPI
            a.v  # Vector{T} - each element is a "row"
        else
            _to_svectors(a.A)  # Vector{SVector{K,T}} - each SVector is a row
        end
    end

    # Broadcast f over all local arrays (GPU-friendly)
    results = f.(local_arrays...)

    # Convert results back to appropriate type
    local_result = _from_result(results)

    # Wrap in MPI type using first argument's partition info
    first_arg = aligned[1]
    row_partition = first_arg isa VectorMPI ? first_arg.partition : first_arg.row_partition
    hash = compute_partition_hash(row_partition)

    if local_result isa AbstractMatrix
        return MatrixMPI(
            hash,
            row_partition,
            [1, size(local_result, 2) + 1],  # Full columns on each rank
            local_result
        )
    else
        return VectorMPI(
            hash,
            row_partition,
            local_result
        )
    end
end


# ============================================================================
# Base.zeros for Distributed Types
# ============================================================================

# Helper to create a zero array with the correct backend type
# Base case: CPU arrays
_zeros_like(::Type{Vector{T}}, dims...) where T = zeros(T, dims...)
_zeros_like(::Type{Matrix{T}}, dims...) where T = zeros(T, dims...)

# For GPU arrays, extensions will define additional methods

"""
    Base.zeros(::Type{VectorMPI{T,AV}}, n::Integer; comm=MPI.COMM_WORLD) where {T,AV}

Create a distributed zero vector of length `n` with element type `T` and storage type `AV`.

The vector is uniformly partitioned across MPI ranks.

# Examples
```julia
# CPU zero vector
v = zeros(VectorMPI{Float64,Vector{Float64}}, 100)

# Using type alias
v = zeros(VectorMPI_CPU{Float64}, 100)

# GPU zero vector (requires Metal.jl loaded)
using Metal
v = zeros(VectorMPI{Float32,MtlVector{Float32}}, 100)
```
"""
function Base.zeros(::Type{VectorMPI{T,AV}}, n::Integer;
                    comm::MPI.Comm=MPI.COMM_WORLD) where {T,AV<:AbstractVector{T}}
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    partition = uniform_partition(n, nranks)
    local_size = partition[rank + 2] - partition[rank + 1]

    local_v = _zeros_like(AV, local_size)
    hash = compute_partition_hash(partition)

    return VectorMPI{T,AV}(hash, partition, local_v)
end

"""
    Base.zeros(::Type{MatrixMPI{T,AM}}, m::Integer, n::Integer; comm=MPI.COMM_WORLD) where {T,AM}

Create a distributed zero matrix of size `m × n` with element type `T` and storage type `AM`.

The matrix is row-partitioned across MPI ranks.

# Examples
```julia
# CPU zero matrix
A = zeros(MatrixMPI{Float64,Matrix{Float64}}, 100, 50)

# Using type alias
A = zeros(MatrixMPI_CPU{Float64}, 100, 50)

# GPU zero matrix (requires Metal.jl loaded)
using Metal
A = zeros(MatrixMPI{Float32,MtlMatrix{Float32}}, 100, 50)
```
"""
function Base.zeros(::Type{MatrixMPI{T,AM}}, m::Integer, n::Integer;
                    comm::MPI.Comm=MPI.COMM_WORLD) where {T,AM<:AbstractMatrix{T}}
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    row_partition = uniform_partition(m, nranks)
    col_partition = uniform_partition(n, nranks)  # Used for transpose operations
    local_nrows = row_partition[rank + 2] - row_partition[rank + 1]

    local_A = _zeros_like(AM, local_nrows, n)
    # Structural hash computed lazily

    return MatrixMPI{T,AM}(nothing, row_partition, col_partition, local_A)
end

"""
    Base.zeros(::Type{SparseMatrixMPI{T,Ti,AV}}, m::Integer, n::Integer; comm=MPI.COMM_WORLD) where {T,Ti,AV}

Create a distributed zero sparse matrix of size `m × n`.

A zero sparse matrix has no nonzero entries, so the resulting matrix has:
- Empty `rowptr` (all ones)
- Empty `colval` and `nzval`

# Examples
```julia
# CPU zero sparse matrix
A = zeros(SparseMatrixMPI{Float64,Int,Vector{Float64}}, 100, 100)

# Using type alias
A = zeros(SparseMatrixMPI_CPU{Float64,Int}, 100, 100)
```
"""
function Base.zeros(::Type{SparseMatrixMPI{T,Ti,AV}}, m::Integer, n::Integer;
                    comm::MPI.Comm=MPI.COMM_WORLD) where {T,Ti<:Integer,AV<:AbstractVector{T}}
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    row_partition = uniform_partition(m, nranks)
    col_partition = uniform_partition(n, nranks)
    local_nrows = row_partition[rank + 2] - row_partition[rank + 1]

    # Empty sparse structure
    rowptr = ones(Ti, local_nrows + 1)  # All rows have 0 entries
    colval = Ti[]
    nzval = _zeros_like(AV, 0)  # Empty but correct type
    col_indices = Int[]  # No columns referenced

    # For CPU, rowptr_target/colval_target are the same as rowptr/colval
    # For GPU, they would be GPU copies (but empty arrays don't matter)
    rowptr_target = rowptr
    colval_target = colval

    return SparseMatrixMPI{T,Ti,AV}(
        nothing,  # Hash computed lazily
        row_partition,
        col_partition,
        col_indices,
        rowptr,
        colval,
        nzval,
        local_nrows,
        0,  # ncols_compressed = 0 (no columns referenced)
        nothing,  # cached_transpose
        true,  # cached_symmetric (zero matrix is symmetric)
        rowptr_target,
        colval_target
    )
end

# ============================================================================
# Precompilation Workload
# ============================================================================

using PrecompileTools

@setup_workload begin
    # Small test data for precompilation (runs with single MPI rank)
    n = 8

    # Sparse matrix (tridiagonal) - Float64
    I_sp = Int[]; J_sp = Int[]; V_sp = Float64[]
    for i in 1:n
        push!(I_sp, i); push!(J_sp, i); push!(V_sp, 4.0)
        if i > 1
            push!(I_sp, i); push!(J_sp, i-1); push!(V_sp, -1.0)
            push!(I_sp, i-1); push!(J_sp, i); push!(V_sp, -1.0)
        end
    end
    A_sparse_f64 = sparse(I_sp, J_sp, V_sp, n, n)

    # Sparse matrix - ComplexF64
    A_sparse_c64 = sparse(I_sp, J_sp, ComplexF64.(V_sp), n, n)

    # Dense matrix - Float64
    A_dense_f64 = Float64[i == j ? 4.0 : (abs(i-j) == 1 ? -1.0 : 0.0) for i in 1:n, j in 1:n]

    # Dense matrix - ComplexF64
    A_dense_c64 = ComplexF64.(A_dense_f64)

    # Vectors
    v_f64 = ones(Float64, n)
    v_c64 = ones(ComplexF64, n)

    # Identity for SPD construction
    I_sparse = sparse(1.0 * LinearAlgebra.I, n, n)

    @compile_workload begin
        # === MPI Jail Escape ===
        # When precompiling under mpiexec, the subprocess inherits MPI environment
        # variables but isn't part of the MPI job. Clean them to allow MPI.Init()
        # to succeed as a fresh single-rank process.
        for k in collect(keys(ENV))
            if startswith(k, "PMI") || startswith(k, "PMIX") || startswith(k, "OMPI_") || startswith(k, "MPI_")
                delete!(ENV, k)
            end
        end

        MPI.Init()

        # === VectorMPI operations (Float64) ===
        v = VectorMPI(v_f64)
        w = VectorMPI(2.0 .* v_f64)
        _ = v + w
        _ = v - w
        _ = 2.0 * v
        _ = v * 2.0
        _ = norm(v)
        _ = dot(v, w)
        _ = conj(v)
        _ = length(v)
        _ = size(v)

        # VectorMPI (ComplexF64)
        vc = VectorMPI(v_c64)
        _ = conj(vc)
        _ = norm(vc)

        # === SparseMatrixMPI operations (Float64) ===
        A = SparseMatrixMPI{Float64}(A_sparse_f64)
        B = SparseMatrixMPI{Float64}(A_sparse_f64)
        _ = A + B
        _ = A - B
        _ = 2.0 * A
        _ = A * v
        _ = A * B
        _ = transpose(A)
        At = SparseMatrixMPI(transpose(A))
        _ = size(A)
        _ = nnz(A)
        _ = norm(A)

        # SparseMatrixMPI (ComplexF64)
        Ac = SparseMatrixMPI{ComplexF64}(A_sparse_c64)
        _ = Ac * vc

        # === MatrixMPI operations (Float64) ===
        D = MatrixMPI(A_dense_f64)
        _ = 2.0 * D
        _ = D * v
        _ = transpose(D)
        Dt = copy(transpose(D))  # Materialize dense transpose
        _ = size(D)
        _ = norm(D)

        # MatrixMPI (ComplexF64)
        Dc = MatrixMPI(A_dense_c64)
        _ = Dc * vc

        # === Mixed operations ===
        _ = A * D  # Sparse * Dense

        # === Indexing ===
        _ = v[1]
        _ = A[1, 1]
        _ = D[1, 1]

        # === Factorization (MUMPS) ===
        # Make symmetric positive definite: A + A^T + 10I
        At_mat = SparseMatrixMPI(transpose(A))
        I_dist = SparseMatrixMPI{Float64}(I_sparse)
        A_spd = A + At_mat + I_dist * 10.0
        F = LinearAlgebra.ldlt(A_spd)
        x = F \ v
        finalize!(F)

        # LU factorization
        F_lu = LinearAlgebra.lu(A)
        x = F_lu \ v
        finalize!(F_lu)

        # === Block operations ===
        _ = cat(v, w; dims=1)
        _ = blockdiag(A, B)

        # === Conversions ===
        _ = Vector(v)
        _ = Matrix(D)
        _ = SparseMatrixCSC(A)

        # Clear caches
        clear_plan_cache!()
    end
end

end # module LinearAlgebraMPI
