module LinearAlgebraMPI

using MPI
using Blake3Hash
using SparseArrays
import SparseArrays: nnz, issparse, dropzeros, spdiagm, blockdiag
import LinearAlgebra
import LinearAlgebra: tr, diag, triu, tril, Transpose, Adjoint, norm, opnorm, mul!

export SparseMatrixMPI, MatrixMPI, VectorMPI, clear_plan_cache!
export VectorMPI_local, MatrixMPI_local, SparseMatrixMPI_local  # Local constructors
export mean  # Our mean function for SparseMatrixMPI and VectorMPI
export io0   # Utility for rank-selective output

# Type alias for 256-bit Blake3 hash
const Blake3Hash = NTuple{32,UInt8}

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

# Include the component files (order matters: vectors first, then dense/sparse, then blocks)
include("vectors.jl")
include("dense.jl")
include("sparse.jl")
include("blocks.jl")

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
