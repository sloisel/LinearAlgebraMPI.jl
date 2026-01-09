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
function LinearAlgebraMPI.mtl(v::LinearAlgebraMPI.VectorMPI{T,Vector{T}}) where T
    return adapt(MtlArray, v)
end

# No-op for already-GPU vectors
function LinearAlgebraMPI.mtl(v::LinearAlgebraMPI.VectorMPI{T,<:MtlVector}) where T
    return v
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
The `nzval` and target structure arrays are moved to GPU.
The CPU structure arrays (`rowptr`, `colval`, partitions) remain on CPU for MPI.
"""
function LinearAlgebraMPI.mtl(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,Vector{T}}) where {T,Ti}
    nzval_gpu = MtlVector(A.nzval)
    # Convert structure arrays to GPU (used by unified SpMV kernel)
    rowptr_target = MtlVector(A.rowptr)
    colval_target = MtlVector(A.colval)
    # Use typeof() to get the concrete GPU type (e.g., MtlVector{T, PrivateStorage})
    AV = typeof(nzval_gpu)
    return LinearAlgebraMPI.SparseMatrixMPI{T,Ti,AV}(
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
        A.cached_symmetric,
        rowptr_target,
        colval_target
    )
end

# No-op for already-GPU sparse matrices
function LinearAlgebraMPI.mtl(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,<:MtlVector}) where {T,Ti}
    return A
end

"""
    cpu(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,<:MtlVector})

Convert a Metal GPU SparseMatrixMPI to a CPU SparseMatrixMPI.
"""
function LinearAlgebraMPI.cpu(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,<:MtlVector}) where {T,Ti}
    nzval_cpu = Array(A.nzval)
    # For CPU, rowptr_target and colval_target are the same as rowptr and colval
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
        A.cached_symmetric,
        A.rowptr,  # rowptr_target (same as rowptr for CPU)
        A.colval   # colval_target (same as colval for CPU)
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
    # Use typeof() to get the concrete GPU type (e.g., MtlMatrix{T, PrivateStorage})
    AM = typeof(A_gpu)
    return LinearAlgebraMPI.MatrixMPI{T,AM}(
        A.structural_hash,
        A.row_partition,
        A.col_partition,
        A_gpu
    )
end

# No-op for already-GPU matrices
function LinearAlgebraMPI.mtl(A::LinearAlgebraMPI.MatrixMPI{T,<:MtlMatrix}) where T
    return A
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

# ============================================================================
# MUMPS Factorization Support
# ============================================================================

"""
    _array_to_backend(v::Vector{T}, ::Type{<:MtlVector}) where T

Convert a CPU vector to a Metal GPU vector.
Used by MUMPS factorization for round-trip GPU conversion during solve.
"""
function LinearAlgebraMPI._array_to_backend(v::Vector{T}, ::Type{<:MtlVector}) where T
    return MtlVector(v)
end

"""
    _convert_vector_to_backend(v::LinearAlgebraMPI.VectorMPI{T,<:Vector}, ::Type{<:MtlVector}) where T

Convert a CPU VectorMPI to GPU (Metal) backend.
WARNING: This function exists ONLY for MUMPS, which is a CPU-only solver.
MUMPS requires GPU→CPU→GPU cycling. Do NOT use this for general operations.
"""
function LinearAlgebraMPI._convert_vector_to_backend(v::LinearAlgebraMPI.VectorMPI{T,<:Vector}, ::Type{<:MtlVector}) where T
    return LinearAlgebraMPI.mtl(v)
end

# ============================================================================
# Backend Conversion for Distributed Types
# ============================================================================

"""
    _to_same_backend(cpu::VectorMPI{T,Vector{T}}, ::VectorMPI{S,<:MtlVector}) where {T,S}

Convert a CPU VectorMPI to Metal GPU backend to match the template.
"""
function LinearAlgebraMPI._to_same_backend(cpu::LinearAlgebraMPI.VectorMPI{T,Vector{T}}, ::LinearAlgebraMPI.VectorMPI{S,<:MtlVector}) where {T,S}
    return LinearAlgebraMPI.mtl(cpu)
end

"""
    _to_same_backend(cpu::MatrixMPI{T,Matrix{T}}, ::MatrixMPI{S,<:MtlMatrix}) where {T,S}

Convert a CPU MatrixMPI to Metal GPU backend to match the template.
"""
function LinearAlgebraMPI._to_same_backend(cpu::LinearAlgebraMPI.MatrixMPI{T,Matrix{T}}, ::LinearAlgebraMPI.MatrixMPI{S,<:MtlMatrix}) where {T,S}
    return LinearAlgebraMPI.mtl(cpu)
end

"""
    _to_same_backend(cpu::VectorMPI{T,Vector{T}}, ::MatrixMPI{S,<:MtlMatrix}) where {T,S}

Convert a CPU VectorMPI to Metal GPU backend using a MatrixMPI template.
Used by vertex_indices when the input is a MatrixMPI.
"""
function LinearAlgebraMPI._to_same_backend(cpu::LinearAlgebraMPI.VectorMPI{T,Vector{T}}, ::LinearAlgebraMPI.MatrixMPI{S,<:MtlMatrix}) where {T,S}
    return LinearAlgebraMPI.mtl(cpu)
end

# ============================================================================
# Base.zeros Support
# ============================================================================

"""
    _zeros_like(::Type{<:MtlVector{T}}, dims...) where T

Create a zero MtlVector of the specified dimensions.
Used by Base.zeros(VectorMPI{T,MtlVector{T}}, n).
Accepts concrete types like MtlVector{T, PrivateStorage}.
"""
LinearAlgebraMPI._zeros_like(::Type{<:MtlVector{T}}, dims...) where T = Metal.zeros(T, dims...)

"""
    _zeros_like(::Type{<:MtlMatrix{T}}, dims...) where T

Create a zero MtlMatrix of the specified dimensions.
Used by Base.zeros(MatrixMPI{T,MtlMatrix{T}}, m, n).
Accepts concrete types like MtlMatrix{T, PrivateStorage}.
"""
LinearAlgebraMPI._zeros_like(::Type{<:MtlMatrix{T}}, dims...) where T = Metal.zeros(T, dims...)

# ============================================================================
# MatrixPlan Index Array Support
# ============================================================================

"""
    _index_array_type(::Type{<:MtlVector{T}}, ::Type{Ti}) where {T,Ti}

Map MtlVector{T} value array type to MtlVector{Ti} index array type.
Used by MatrixPlan to store symbolic index arrays on GPU.
Accepts concrete types like MtlVector{T, PrivateStorage}.
"""
LinearAlgebraMPI._index_array_type(::Type{<:MtlVector{T}}, ::Type{Ti}) where {T,Ti} = MtlVector{Ti}

"""
    _to_target_backend(v::Vector{Ti}, ::Type{<:MtlVector}) where Ti

Convert a CPU index vector to Metal GPU.
Used by SparseMatrixMPI constructors to create GPU structure arrays.
Accepts concrete types like MtlVector{T, PrivateStorage}.
"""
LinearAlgebraMPI._to_target_backend(v::Vector{Ti}, ::Type{<:MtlVector}) where Ti = MtlVector(v)

# ============================================================================
# GPU map_rows_gpu implementation via Metal kernels
# ============================================================================

using StaticArrays

"""
    _map_rows_gpu_kernel(f, arg1::MtlMatrix, rest::MtlMatrix...)

GPU-accelerated row-wise map for Metal arrays.
Each thread processes one row, applying `f` to the corresponding rows of all input matrices.
Returns a Metal matrix with the same number of rows.
"""
function LinearAlgebraMPI._map_rows_gpu_kernel(f, arg1::MtlMatrix{T}, rest::MtlMatrix...) where T
    n = size(arg1, 1)

    # Get output size by evaluating f on first row (copy to CPU to avoid scalar indexing)
    arg1_row1 = Array(view(arg1, 1:1, :))[1, :]
    first_rows = (SVector{size(arg1,2),T}(arg1_row1...),)
    for m in rest
        m_row1 = Array(view(m, 1:1, :))[1, :]
        first_rows = (first_rows..., SVector{size(m,2),T}(m_row1...))
    end
    sample_out = f(first_rows...)

    if sample_out isa SVector
        out_cols = length(sample_out)
    elseif sample_out isa SMatrix
        out_cols = length(sample_out)  # Flatten matrix output
    else
        out_cols = 1  # Scalar output
    end

    # Allocate output
    output = Metal.zeros(T, n, out_cols)

    # Create kernel
    _map_rows_kernel_dispatch(f, output, arg1, rest...)

    return output
end

"""
Dispatch to appropriate kernel based on number of arguments.
"""
function _map_rows_kernel_dispatch(f, output::MtlMatrix{T}, arg1::MtlMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    out_cols = size(output, 2)

    kernel = @metal launch=false _map_rows_kernel_1arg(f, output, arg1, Val(ncols1), Val(out_cols))
    threads = min(n, 256)
    groups = cld(n, threads)
    kernel(f, output, arg1, Val(ncols1), Val(out_cols); threads=threads, groups=groups)
    Metal.synchronize()
end

function _map_rows_kernel_dispatch(f, output::MtlMatrix{T}, arg1::MtlMatrix{T}, arg2::MtlMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    ncols2 = size(arg2, 2)
    out_cols = size(output, 2)

    kernel = @metal launch=false _map_rows_kernel_2args(f, output, arg1, arg2, Val(ncols1), Val(ncols2), Val(out_cols))
    threads = min(n, 256)
    groups = cld(n, threads)
    kernel(f, output, arg1, arg2, Val(ncols1), Val(ncols2), Val(out_cols); threads=threads, groups=groups)
    Metal.synchronize()
end

function _map_rows_kernel_dispatch(f, output::MtlMatrix{T}, arg1::MtlMatrix{T}, arg2::MtlMatrix{T}, arg3::MtlMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    ncols2 = size(arg2, 2)
    ncols3 = size(arg3, 2)
    out_cols = size(output, 2)

    kernel = @metal launch=false _map_rows_kernel_3args(f, output, arg1, arg2, arg3, Val(ncols1), Val(ncols2), Val(ncols3), Val(out_cols))
    threads = min(n, 256)
    groups = cld(n, threads)
    kernel(f, output, arg1, arg2, arg3, Val(ncols1), Val(ncols2), Val(ncols3), Val(out_cols); threads=threads, groups=groups)
    Metal.synchronize()
end

function _map_rows_kernel_dispatch(f, output::MtlMatrix{T}, arg1::MtlMatrix{T}, arg2::MtlMatrix{T}, arg3::MtlMatrix{T}, arg4::MtlMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    ncols2 = size(arg2, 2)
    ncols3 = size(arg3, 2)
    ncols4 = size(arg4, 2)
    out_cols = size(output, 2)

    kernel = @metal launch=false _map_rows_kernel_4args(f, output, arg1, arg2, arg3, arg4, Val(ncols1), Val(ncols2), Val(ncols3), Val(ncols4), Val(out_cols))
    threads = min(n, 256)
    groups = cld(n, threads)
    kernel(f, output, arg1, arg2, arg3, arg4, Val(ncols1), Val(ncols2), Val(ncols3), Val(ncols4), Val(out_cols); threads=threads, groups=groups)
    Metal.synchronize()
end

function _map_rows_kernel_dispatch(f, output::MtlMatrix{T}, arg1::MtlMatrix{T}, arg2::MtlMatrix{T}, arg3::MtlMatrix{T}, arg4::MtlMatrix{T}, arg5::MtlMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    ncols2 = size(arg2, 2)
    ncols3 = size(arg3, 2)
    ncols4 = size(arg4, 2)
    ncols5 = size(arg5, 2)
    out_cols = size(output, 2)

    kernel = @metal launch=false _map_rows_kernel_5args(f, output, arg1, arg2, arg3, arg4, arg5, Val(ncols1), Val(ncols2), Val(ncols3), Val(ncols4), Val(ncols5), Val(out_cols))
    threads = min(n, 256)
    groups = cld(n, threads)
    kernel(f, output, arg1, arg2, arg3, arg4, arg5, Val(ncols1), Val(ncols2), Val(ncols3), Val(ncols4), Val(ncols5), Val(out_cols); threads=threads, groups=groups)
    Metal.synchronize()
end

# ============================================================================
# Metal kernels
# ============================================================================

function _map_rows_kernel_1arg(f, output, arg1, ::Val{NC1}, ::Val{OCols}) where {NC1, OCols}
    i = thread_position_in_grid_1d()
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        result = f(row1)
        _write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _map_rows_kernel_2args(f, output, arg1, arg2, ::Val{NC1}, ::Val{NC2}, ::Val{OCols}) where {NC1, NC2, OCols}
    i = thread_position_in_grid_1d()
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        result = f(row1, row2)
        _write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _map_rows_kernel_3args(f, output, arg1, arg2, arg3, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{OCols}) where {NC1, NC2, NC3, OCols}
    i = thread_position_in_grid_1d()
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        row3 = SVector{NC3,T}(ntuple(j -> @inbounds(arg3[i,j]), Val(NC3)))
        result = f(row1, row2, row3)
        _write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _map_rows_kernel_4args(f, output, arg1, arg2, arg3, arg4, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{OCols}) where {NC1, NC2, NC3, NC4, OCols}
    i = thread_position_in_grid_1d()
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        row3 = SVector{NC3,T}(ntuple(j -> @inbounds(arg3[i,j]), Val(NC3)))
        row4 = SVector{NC4,T}(ntuple(j -> @inbounds(arg4[i,j]), Val(NC4)))
        result = f(row1, row2, row3, row4)
        _write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _map_rows_kernel_5args(f, output, arg1, arg2, arg3, arg4, arg5, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{NC5}, ::Val{OCols}) where {NC1, NC2, NC3, NC4, NC5, OCols}
    i = thread_position_in_grid_1d()
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        row3 = SVector{NC3,T}(ntuple(j -> @inbounds(arg3[i,j]), Val(NC3)))
        row4 = SVector{NC4,T}(ntuple(j -> @inbounds(arg4[i,j]), Val(NC4)))
        row5 = SVector{NC5,T}(ntuple(j -> @inbounds(arg5[i,j]), Val(NC5)))
        result = f(row1, row2, row3, row4, row5)
        _write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

# Helper to write result (scalar, SVector, or SMatrix) to output row
@inline function _write_result!(output, i, result::Number, ::Val{1})
    @inbounds output[i, 1] = result
    return nothing
end

@inline function _write_result!(output, i, result::SVector{N,T}, ::Val{N}) where {N,T}
    for j in 1:N
        @inbounds output[i, j] = result[j]
    end
    return nothing
end

@inline function _write_result!(output, i, result::SMatrix{M,N,T}, ::Val{MN}) where {M,N,T,MN}
    for j in 1:MN
        @inbounds output[i, j] = result[j]
    end
    return nothing
end

end # module
