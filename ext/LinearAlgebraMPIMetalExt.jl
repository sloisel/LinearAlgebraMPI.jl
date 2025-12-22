"""
    LinearAlgebraMPIMetalExt

Extension module for Metal GPU support in LinearAlgebraMPI.
Provides constructors and operations for MtlArray-backed distributed arrays.
"""
module LinearAlgebraMPIMetalExt

using LinearAlgebraMPI
using Metal
using Adapt

# Re-export for convenience
const MtlVectorMPI{T} = LinearAlgebraMPI.VectorMPI{T,MtlVector{T}}

"""
    mtl(v::LinearAlgebraMPI.VectorMPI)

Convert a CPU VectorMPI to a Metal GPU VectorMPI.
"""
function LinearAlgebraMPI.mtl(v::LinearAlgebraMPI.VectorMPI{T}) where T
    return adapt(MtlArray, v)
end

"""
    cpu(v::LinearAlgebraMPI.VectorMPI{T,<:MtlVector})

Convert a Metal GPU VectorMPI to a CPU VectorMPI.
"""
function LinearAlgebraMPI.cpu(v::LinearAlgebraMPI.VectorMPI{T,<:MtlVector}) where T
    return adapt(Array, v)
end

# Note: Metal.jl already provides adapt_storage methods for MtlArray
# so we don't need to define them here

# Type alias for Metal SparseMatrixMPI
const MtlSparseMatrixMPI{T,Ti} = LinearAlgebraMPI.SparseMatrixMPI{T,Ti,MtlVector{T}}

"""
    mtl(A::LinearAlgebraMPI.SparseMatrixMPI)

Convert a CPU SparseMatrixMPI to a Metal GPU SparseMatrixMPI.
Only the nonzero values (`nzval`) are moved to GPU. Structural arrays
(`rowptr`, `colval`, partitions) remain on CPU.
"""
function LinearAlgebraMPI.mtl(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,Vector{T}}) where {T,Ti}
    nzval_gpu = MtlVector(A.nzval)
    return LinearAlgebraMPI.SparseMatrixMPI{T,Ti,MtlVector{T}}(
        A.structural_hash,
        A.row_partition,
        A.col_partition,
        A.col_indices,
        A.rowptr,
        A.colval,
        nzval_gpu,
        A.nrows_local,
        A.ncols_compressed,
        nothing,  # Invalidate cached_transpose (would need to convert too)
        A.cached_symmetric
    )
end

"""
    cpu(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,<:MtlVector})

Convert a Metal GPU SparseMatrixMPI to a CPU SparseMatrixMPI.
"""
function LinearAlgebraMPI.cpu(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,<:MtlVector}) where {T,Ti}
    nzval_cpu = Array(A.nzval)
    return LinearAlgebraMPI.SparseMatrixMPI{T,Ti,Vector{T}}(
        A.structural_hash,
        A.row_partition,
        A.col_partition,
        A.col_indices,
        A.rowptr,
        A.colval,
        nzval_cpu,
        A.nrows_local,
        A.ncols_compressed,
        nothing,  # Invalidate cached_transpose
        A.cached_symmetric
    )
end

# Type alias for Metal MatrixMPI
const MtlMatrixMPI{T} = LinearAlgebraMPI.MatrixMPI{T,MtlMatrix{T}}

"""
    mtl(A::LinearAlgebraMPI.MatrixMPI)

Convert a CPU MatrixMPI to a Metal GPU MatrixMPI.
"""
function LinearAlgebraMPI.mtl(A::LinearAlgebraMPI.MatrixMPI{T,Matrix{T}}) where T
    A_gpu = MtlMatrix(A.A)
    return LinearAlgebraMPI.MatrixMPI{T,MtlMatrix{T}}(
        A.structural_hash,
        A.row_partition,
        A.col_partition,
        A_gpu
    )
end

"""
    cpu(A::LinearAlgebraMPI.MatrixMPI{T,<:MtlMatrix})

Convert a Metal GPU MatrixMPI to a CPU MatrixMPI.
"""
function LinearAlgebraMPI.cpu(A::LinearAlgebraMPI.MatrixMPI{T,<:MtlMatrix}) where T
    A_cpu = Array(A.A)
    return LinearAlgebraMPI.MatrixMPI{T,Matrix{T}}(
        A.structural_hash,
        A.row_partition,
        A.col_partition,
        A_cpu
    )
end

end # module
