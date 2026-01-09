"""
    LinearAlgebraMPICUDAExt

Extension module for CUDA GPU support in LinearAlgebraMPI.
Provides:
- cu()/cpu() conversions for CuArray-backed distributed arrays
- cuDSS sparse direct solver with NCCL backend for multi-GPU factorization

Requires: `using CUDA, NCCL, CUDSS_jll` before loading LinearAlgebraMPI.
"""
module LinearAlgebraMPICUDAExt

using LinearAlgebraMPI
using CUDA
using NCCL
using CUDSS_jll
using Adapt
using MPI
using SparseArrays
using LinearAlgebra

# ============================================================================
# Part 1: Core CUDA Support (cu/cpu conversions, backend helpers)
# ============================================================================

# Type aliases for convenience
const CuVectorMPI{T} = LinearAlgebraMPI.VectorMPI{T,CuVector{T}}
const CuSparseMatrixMPI{T,Ti} = LinearAlgebraMPI.SparseMatrixMPI{T,Ti,CuVector{T}}
const CuMatrixMPI{T} = LinearAlgebraMPI.MatrixMPI{T,CuMatrix{T}}

# ----------------------------------------------------------------------------
# VectorMPI conversions
# ----------------------------------------------------------------------------

"""
    cu(v::LinearAlgebraMPI.VectorMPI)

Convert a CPU VectorMPI to a CUDA GPU VectorMPI.
"""
function LinearAlgebraMPI.cu(v::LinearAlgebraMPI.VectorMPI{T,Vector{T}}) where T
    return adapt(CuArray, v)
end

# No-op for already-GPU vectors
function LinearAlgebraMPI.cu(v::LinearAlgebraMPI.VectorMPI{T,<:CuVector}) where T
    return v
end

"""
    cpu(v::LinearAlgebraMPI.VectorMPI{T,<:CuVector})

Convert a CUDA GPU VectorMPI to a CPU VectorMPI.
"""
function LinearAlgebraMPI.cpu(v::LinearAlgebraMPI.VectorMPI{T,<:CuVector}) where T
    return adapt(Array, v)
end

# ----------------------------------------------------------------------------
# SparseMatrixMPI conversions
# ----------------------------------------------------------------------------

"""
    cu(A::LinearAlgebraMPI.SparseMatrixMPI)

Convert a CPU SparseMatrixMPI to a CUDA GPU SparseMatrixMPI.
The `nzval` and target structure arrays are moved to GPU.
The CPU structure arrays (`rowptr`, `colval`, partitions) remain on CPU for MPI.
"""
function LinearAlgebraMPI.cu(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,Vector{T}}) where {T,Ti}
    nzval_gpu = CuVector(A.nzval)
    # Convert structure arrays to GPU (used by unified SpMV kernel)
    rowptr_target = CuVector(A.rowptr)
    colval_target = CuVector(A.colval)
    # Use typeof() to get the concrete GPU type
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
        nothing,  # Invalidate cached_transpose
        A.cached_symmetric,
        rowptr_target,
        colval_target
    )
end

# No-op for already-GPU sparse matrices
function LinearAlgebraMPI.cu(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,<:CuVector}) where {T,Ti}
    return A
end

"""
    cpu(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,<:CuVector})

Convert a CUDA GPU SparseMatrixMPI to a CPU SparseMatrixMPI.
"""
function LinearAlgebraMPI.cpu(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,<:CuVector}) where {T,Ti}
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
        A.cached_symmetric,
        A.rowptr,  # rowptr_target (same as rowptr for CPU)
        A.colval   # colval_target (same as colval for CPU)
    )
end

# ----------------------------------------------------------------------------
# MatrixMPI conversions
# ----------------------------------------------------------------------------

"""
    cu(A::LinearAlgebraMPI.MatrixMPI)

Convert a CPU MatrixMPI to a CUDA GPU MatrixMPI.
"""
function LinearAlgebraMPI.cu(A::LinearAlgebraMPI.MatrixMPI{T,Matrix{T}}) where T
    A_gpu = CuMatrix(A.A)
    AM = typeof(A_gpu)
    return LinearAlgebraMPI.MatrixMPI{T,AM}(
        A.structural_hash,
        A.row_partition,
        A.col_partition,
        A_gpu
    )
end

# No-op for already-GPU matrices
function LinearAlgebraMPI.cu(A::LinearAlgebraMPI.MatrixMPI{T,<:CuMatrix}) where T
    return A
end

"""
    cpu(A::LinearAlgebraMPI.MatrixMPI{T,<:CuMatrix})

Convert a CUDA GPU MatrixMPI to a CPU MatrixMPI.
"""
function LinearAlgebraMPI.cpu(A::LinearAlgebraMPI.MatrixMPI{T,<:CuMatrix}) where T
    A_cpu = Array(A.A)
    return LinearAlgebraMPI.MatrixMPI{T,Matrix{T}}(
        A.structural_hash,
        A.row_partition,
        A.col_partition,
        A_cpu
    )
end

# ----------------------------------------------------------------------------
# Backend helper functions
# ----------------------------------------------------------------------------

"""
    _zeros_like(::Type{<:CuVector{T}}, dims...) where T

Create a zero CuVector of the specified dimensions.
"""
LinearAlgebraMPI._zeros_like(::Type{<:CuVector{T}}, dims...) where T = CUDA.zeros(T, dims...)

"""
    _zeros_like(::Type{<:CuMatrix{T}}, dims...) where T

Create a zero CuMatrix of the specified dimensions.
"""
LinearAlgebraMPI._zeros_like(::Type{<:CuMatrix{T}}, dims...) where T = CUDA.zeros(T, dims...)

"""
    _index_array_type(::Type{<:CuVector{T}}, ::Type{Ti}) where {T,Ti}

Map CuVector{T} value array type to CuVector{Ti} index array type.
"""
LinearAlgebraMPI._index_array_type(::Type{<:CuVector{T}}, ::Type{Ti}) where {T,Ti} = CuVector{Ti}

"""
    _to_target_backend(v::Vector{Ti}, ::Type{<:CuVector}) where Ti

Convert a CPU index vector to CUDA GPU.
"""
LinearAlgebraMPI._to_target_backend(v::Vector{Ti}, ::Type{<:CuVector}) where Ti = CuVector(v)

"""
    _array_to_backend(v::Vector{T}, ::Type{<:CuVector}) where T

Convert a CPU vector to a CUDA GPU vector.
Used by factorization for round-trip GPU conversion during solve.
"""
function LinearAlgebraMPI._array_to_backend(v::Vector{T}, ::Type{<:CuVector}) where T
    return CuVector(v)
end

"""
    _convert_vector_to_backend(v::LinearAlgebraMPI.VectorMPI{T,<:Vector}, ::Type{<:CuVector}) where T

Convert a CPU VectorMPI to GPU (CUDA) backend.
"""
function LinearAlgebraMPI._convert_vector_to_backend(v::LinearAlgebraMPI.VectorMPI{T,<:Vector}, ::Type{<:CuVector}) where T
    return LinearAlgebraMPI.cu(v)
end

# ============================================================================
# Backend Conversion for Distributed Types
# ============================================================================

"""
    _to_same_backend(cpu::VectorMPI{T,Vector{T}}, ::VectorMPI{S,<:CuVector}) where {T,S}

Convert a CPU VectorMPI to CUDA GPU backend to match the template.
"""
function LinearAlgebraMPI._to_same_backend(cpu::LinearAlgebraMPI.VectorMPI{T,Vector{T}}, ::LinearAlgebraMPI.VectorMPI{S,<:CuVector}) where {T,S}
    return LinearAlgebraMPI.cu(cpu)
end

"""
    _to_same_backend(cpu::MatrixMPI{T,Matrix{T}}, ::MatrixMPI{S,<:CuMatrix}) where {T,S}

Convert a CPU MatrixMPI to CUDA GPU backend to match the template.
"""
function LinearAlgebraMPI._to_same_backend(cpu::LinearAlgebraMPI.MatrixMPI{T,Matrix{T}}, ::LinearAlgebraMPI.MatrixMPI{S,<:CuMatrix}) where {T,S}
    return LinearAlgebraMPI.cu(cpu)
end

"""
    _to_same_backend(cpu::VectorMPI{T,Vector{T}}, ::MatrixMPI{S,<:CuMatrix}) where {T,S}

Convert a CPU VectorMPI to CUDA GPU backend using a MatrixMPI template.
Used by vertex_indices when the input is a MatrixMPI.
"""
function LinearAlgebraMPI._to_same_backend(cpu::LinearAlgebraMPI.VectorMPI{T,Vector{T}}, ::LinearAlgebraMPI.MatrixMPI{S,<:CuMatrix}) where {T,S}
    return LinearAlgebraMPI.cu(cpu)
end

# ============================================================================
# Part 2: cuDSS Constants and ccall Wrappers
# ============================================================================

const cudssHandle_t = Ptr{Cvoid}
const cudssConfig_t = Ptr{Cvoid}
const cudssData_t = Ptr{Cvoid}
const cudssMatrix_t = Ptr{Cvoid}

# Status codes
const CUDSS_STATUS_SUCCESS = UInt32(0)

# Data parameters
const CUDSS_DATA_COMM = UInt32(11)

# Phases (can be OR'd together)
const CUDSS_PHASE_ANALYSIS = Cint(3)  # REORDERING | SYMBOLIC
const CUDSS_PHASE_FACTORIZATION = Cint(4)
const CUDSS_PHASE_SOLVE = Cint(1008)

# Matrix types
const CUDSS_MTYPE_GENERAL = UInt32(0)
const CUDSS_MTYPE_SYMMETRIC = UInt32(1)
const CUDSS_MTYPE_SPD = UInt32(3)

# Matrix view types
const CUDSS_MVIEW_FULL = UInt32(0)
const CUDSS_MVIEW_LOWER = UInt32(1)

# Index base
const CUDSS_BASE_ZERO = UInt32(0)
const CUDSS_BASE_ONE = UInt32(1)

# Layout
const CUDSS_LAYOUT_COL_MAJOR = UInt32(0)

# CUDA data type mapping
_cuda_data_type(::Type{Float32}) = UInt32(0)   # CUDA_R_32F
_cuda_data_type(::Type{Float64}) = UInt32(1)   # CUDA_R_64F
_cuda_data_type(::Type{Int32}) = UInt32(10)    # CUDA_R_32I
_cuda_data_type(::Type{Int64}) = UInt32(24)    # CUDA_R_64I

# Low-level ccall wrappers (no finalizers, explicit error handling)

function _cudss_create(handle_ref::Ref{cudssHandle_t})
    status = @ccall libcudss.cudssCreate(handle_ref::Ptr{cudssHandle_t})::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssCreate failed with status $status")
    return nothing
end

function _cudss_destroy(handle::cudssHandle_t)
    status = @ccall libcudss.cudssDestroy(handle::cudssHandle_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssDestroy failed with status $status")
    return nothing
end

function _cudss_set_comm_layer(handle::cudssHandle_t, lib_path::String)
    status = @ccall libcudss.cudssSetCommLayer(handle::cudssHandle_t,
                                                lib_path::Cstring)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssSetCommLayer failed with status $status")
    return nothing
end

function _cudss_config_create(config_ref::Ref{cudssConfig_t})
    status = @ccall libcudss.cudssConfigCreate(config_ref::Ptr{cudssConfig_t})::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssConfigCreate failed with status $status")
    return nothing
end

function _cudss_config_destroy(config::cudssConfig_t)
    status = @ccall libcudss.cudssConfigDestroy(config::cudssConfig_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssConfigDestroy failed with status $status")
    return nothing
end

function _cudss_data_create(handle::cudssHandle_t, data_ref::Ref{cudssData_t})
    status = @ccall libcudss.cudssDataCreate(handle::cudssHandle_t,
                                              data_ref::Ptr{cudssData_t})::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssDataCreate failed with status $status")
    return nothing
end

function _cudss_data_destroy(handle::cudssHandle_t, data::cudssData_t)
    status = @ccall libcudss.cudssDataDestroy(handle::cudssHandle_t,
                                               data::cudssData_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssDataDestroy failed with status $status")
    return nothing
end

function _cudss_data_set(handle::cudssHandle_t, data::cudssData_t,
                         param::UInt32, value::Ptr{Cvoid}, size::Csize_t)
    status = @ccall libcudss.cudssDataSet(handle::cudssHandle_t,
                                           data::cudssData_t,
                                           param::UInt32,
                                           value::Ptr{Cvoid},
                                           size::Csize_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssDataSet failed with status $status")
    return nothing
end

function _cudss_matrix_create_csr(matrix_ref::Ref{cudssMatrix_t},
                                   nrows::Int64, ncols::Int64, nnz::Int64,
                                   row_offsets::CuPtr{Cvoid}, row_end::CuPtr{Cvoid},
                                   col_indices::CuPtr{Cvoid}, values::CuPtr{Cvoid},
                                   index_type::UInt32, value_type::UInt32,
                                   mtype::UInt32, mview::UInt32, index_base::UInt32)
    status = @ccall libcudss.cudssMatrixCreateCsr(
        matrix_ref::Ptr{cudssMatrix_t},
        nrows::Int64, ncols::Int64, nnz::Int64,
        row_offsets::CuPtr{Cvoid}, row_end::CuPtr{Cvoid},
        col_indices::CuPtr{Cvoid}, values::CuPtr{Cvoid},
        index_type::UInt32, value_type::UInt32,
        mtype::UInt32, mview::UInt32, index_base::UInt32)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssMatrixCreateCsr failed with status $status")
    return nothing
end

function _cudss_matrix_create_dn(matrix_ref::Ref{cudssMatrix_t},
                                  nrows::Int64, ncols::Int64, ld::Int64,
                                  values::CuPtr{Cvoid}, value_type::UInt32, layout::UInt32)
    status = @ccall libcudss.cudssMatrixCreateDn(
        matrix_ref::Ptr{cudssMatrix_t},
        nrows::Int64, ncols::Int64, ld::Int64,
        values::CuPtr{Cvoid}, value_type::UInt32, layout::UInt32)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssMatrixCreateDn failed with status $status")
    return nothing
end

function _cudss_matrix_destroy(matrix::cudssMatrix_t)
    status = @ccall libcudss.cudssMatrixDestroy(matrix::cudssMatrix_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssMatrixDestroy failed with status $status")
    return nothing
end

function _cudss_matrix_set_distribution_row1d(matrix::cudssMatrix_t,
                                               first_row::Int64, last_row::Int64)
    status = @ccall libcudss.cudssMatrixSetDistributionRow1d(
        matrix::cudssMatrix_t, first_row::Int64, last_row::Int64)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssMatrixSetDistributionRow1d failed with status $status")
    return nothing
end

function _cudss_execute(handle::cudssHandle_t, phase::Cint,
                        config::cudssConfig_t, data::cudssData_t,
                        matrix::cudssMatrix_t, solution::cudssMatrix_t, rhs::cudssMatrix_t)
    status = @ccall libcudss.cudssExecute(
        handle::cudssHandle_t, phase::Cint,
        config::cudssConfig_t, data::cudssData_t,
        matrix::cudssMatrix_t, solution::cudssMatrix_t, rhs::cudssMatrix_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssExecute (phase=$phase) failed with status $status")
    return nothing
end

# ============================================================================
# Part 3: NCCL Bootstrap and CuDSSFactorizationMPI Type
# ============================================================================

"""
    _init_nccl_from_mpi(mpi_comm::MPI.Comm) -> NCCL.Communicator

Initialize NCCL communicator using MPI for the initial UniqueID broadcast.
This is the minimal MPI usage - just 128 bytes to bootstrap NCCL.
"""
function _init_nccl_from_mpi(mpi_comm::MPI.Comm)
    rank = MPI.Comm_rank(mpi_comm)
    nranks = MPI.Comm_size(mpi_comm)

    # Rank 0 generates the unique ID
    if rank == 0
        unique_id = NCCL.UniqueID()
        unique_id_bytes = collect(reinterpret(UInt8, [unique_id.internal]))
    else
        unique_id_bytes = zeros(UInt8, 128)
    end

    # Broadcast the unique ID from rank 0 to all ranks
    MPI.Bcast!(unique_id_bytes, 0, mpi_comm)

    # Reconstruct UniqueID on non-root ranks
    if rank != 0
        internal_tuple = NTuple{128, Int8}(reinterpret(Int8, unique_id_bytes))
        unique_id = NCCL.UniqueID(internal_tuple)
    end

    # Create NCCL communicator on each rank
    nccl_comm = NCCL.Communicator(nranks, rank; unique_id=unique_id)

    return nccl_comm
end

"""
    CuDSSFactorizationMPI{T}

Distributed cuDSS factorization result with NCCL backend.
Matches MUMPS factorization API: use with `F \\ b` or `solve(F, b)`.

IMPORTANT: Uses MPI-safe finalization pattern. Resources are queued for
destruction and processed at the next collective operation.
"""
mutable struct CuDSSFactorizationMPI{T}
    id::Int
    handle::cudssHandle_t
    config::cudssConfig_t
    data::cudssData_t
    matrix::cudssMatrix_t
    solution::cudssMatrix_t
    rhs::cudssMatrix_t
    # GPU arrays (must keep references to prevent GC)
    row_offsets::CuVector{Int32}
    col_indices::CuVector{Int32}
    values::CuVector{T}
    x_gpu::CuVector{T}
    b_gpu::CuVector{T}
    # NCCL communicator
    nccl_comm_storage::Vector{Ptr{Nothing}}
    nccl_comm::NCCL.Communicator
    # Metadata
    n::Int  # Global matrix dimension
    local_nrows::Int
    first_row::Int  # 0-based
    last_row::Int   # 0-based, inclusive
    symmetric::Bool
    row_partition::Vector{Int}
    destroyed::Bool
end

# Registry for MPI-safe finalization (same pattern as MUMPS)
const _cudss_count = Ref{Int}(0)
const _cudss_registry = Dict{Int, CuDSSFactorizationMPI}()
const _cudss_destroy_list = Int[]
const _cudss_destroy_list_lock = ReentrantLock()

"""
    _queue_cudss_for_destruction(F::CuDSSFactorizationMPI)

Queue a factorization for synchronized destruction.
Called by Julia finalizer - does NOT call MPI directly.
"""
function _queue_cudss_for_destruction(F::CuDSSFactorizationMPI)
    lock(_cudss_destroy_list_lock) do
        push!(_cudss_destroy_list, F.id)
    end
end

"""
    _process_cudss_finalizers()

Process queued cuDSS destructions. MUST be called collectively on all ranks.
"""
function _process_cudss_finalizers()
    comm = MPI.COMM_WORLD

    # Get local destroy list
    local_list = lock(_cudss_destroy_list_lock) do
        list = copy(_cudss_destroy_list)
        empty!(_cudss_destroy_list)
        list
    end

    # Gather all destruction requests
    all_counts = MPI.Allgather(Int32(length(local_list)), comm)
    total_count = sum(all_counts)

    if total_count == 0
        return
    end

    # Gather all IDs
    all_ids = MPI.Allgatherv(Int32.(local_list), all_counts, comm)

    # Compute dead list (same on all ranks)
    dead_list = sort!(unique(all_ids))

    # Destroy in order
    for id in dead_list
        if haskey(_cudss_registry, id)
            F = _cudss_registry[id]
            delete!(_cudss_registry, id)
            _destroy_cudss!(F)
        end
    end
end

"""
    _destroy_cudss!(F::CuDSSFactorizationMPI)

Actually destroy cuDSS resources. Internal function.
"""
function _destroy_cudss!(F::CuDSSFactorizationMPI)
    F.destroyed && return

    # Destroy in reverse order of creation
    _cudss_matrix_destroy(F.rhs)
    _cudss_matrix_destroy(F.solution)
    _cudss_matrix_destroy(F.matrix)
    _cudss_data_destroy(F.handle, F.data)
    _cudss_config_destroy(F.config)
    _cudss_destroy(F.handle)

    F.destroyed = true
    return nothing
end

"""
    finalize!(F::CuDSSFactorizationMPI)

Explicitly finalize a cuDSS factorization.
This is a collective operation - must be called on all ranks.
"""
function LinearAlgebraMPI.finalize!(F::CuDSSFactorizationMPI)
    haskey(_cudss_registry, F.id) || return F
    delete!(_cudss_registry, F.id)
    _destroy_cudss!(F)
    return F
end

# ============================================================================
# Part 4: lu/ldlt and solve interface
# ============================================================================

"""
    lu(A::SparseMatrixMPI{T,Ti,<:CuVector})

Compute LU factorization of a GPU sparse matrix using cuDSS.
Returns a CuDSSFactorizationMPI that can be used with `F \\ b`.
"""
function LinearAlgebra.lu(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,<:CuVector}) where {T,Ti}
    return _create_cudss_factorization(A, false)
end

"""
    ldlt(A::SparseMatrixMPI{T,Ti,<:CuVector})

Compute LDLT (Cholesky) factorization of a symmetric positive definite GPU sparse matrix.
Returns a CuDSSFactorizationMPI that can be used with `F \\ b`.
"""
function LinearAlgebra.ldlt(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,<:CuVector}) where {T,Ti}
    return _create_cudss_factorization(A, true)
end

"""
Internal function to create cuDSS factorization.
"""
function _create_cudss_factorization(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,AV}, symmetric::Bool) where {T,Ti,AV}
    # Process any pending finalizers (MPI collective)
    _process_cudss_finalizers()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Assign GPU to rank
    num_gpus = length(CUDA.devices())
    gpu_id = mod(rank, num_gpus)
    CUDA.device!(gpu_id)

    # Get matrix dimensions
    n = A.row_partition[end] - 1  # Global dimension (1-indexed partition)
    local_nrows = A.nrows_local
    first_row = A.row_partition[rank + 1] - 1  # Convert to 0-based
    last_row = A.row_partition[rank + 2] - 2   # 0-based, inclusive

    # Convert to CSR format with 0-based indices for cuDSS
    # A.rowptr is 1-based, need 0-based
    row_offsets_cpu = Int32.(A.rowptr .- 1)
    row_offsets = CuVector{Int32}(row_offsets_cpu)

    # A.colval contains LOCAL indices into col_indices
    # Need GLOBAL column indices for cuDSS
    col_indices_global_cpu = Int32.(A.col_indices[A.colval] .- 1)  # 0-based global
    col_indices_gpu = CuVector{Int32}(col_indices_global_cpu)

    # Values (already on GPU)
    values_cpu = LinearAlgebraMPI._ensure_cpu(A.nzval)
    values_gpu = CuVector{T}(values_cpu)
    nnz_local = length(values_gpu)

    # Allocate solution and RHS vectors
    x_gpu = CUDA.zeros(T, local_nrows)
    b_gpu = CUDA.zeros(T, local_nrows)

    # Find NCCL communication layer library
    comm_lib = joinpath(CUDSS_jll.artifact_dir, "lib", "libcudss_commlayer_nccl.so")
    if !isfile(comm_lib)
        error("NCCL communication layer not found at $comm_lib")
    end

    # Initialize NCCL communicator
    nccl_comm = _init_nccl_from_mpi(comm)

    # Initialize cuDSS
    handle_ref = Ref{cudssHandle_t}(C_NULL)
    _cudss_create(handle_ref)
    handle = handle_ref[]

    _cudss_set_comm_layer(handle, comm_lib)

    config_ref = Ref{cudssConfig_t}(C_NULL)
    _cudss_config_create(config_ref)
    config = config_ref[]

    data_ref = Ref{cudssData_t}(C_NULL)
    _cudss_data_create(handle, data_ref)
    data = data_ref[]

    # Set NCCL communicator
    nccl_comm_storage = Vector{Ptr{Nothing}}(undef, 1)
    nccl_comm_storage[1] = Ptr{Nothing}(nccl_comm.handle)
    _cudss_data_set(handle, data, CUDSS_DATA_COMM,
                    Ptr{Cvoid}(pointer(nccl_comm_storage)), Csize_t(sizeof(Ptr{Nothing})))

    # Create sparse matrix wrapper
    mtype = symmetric ? CUDSS_MTYPE_SPD : CUDSS_MTYPE_GENERAL
    matrix_ref = Ref{cudssMatrix_t}(C_NULL)
    _cudss_matrix_create_csr(matrix_ref,
        Int64(n), Int64(n), Int64(nnz_local),
        reinterpret(CuPtr{Cvoid}, pointer(row_offsets)),
        CuPtr{Cvoid}(0),
        reinterpret(CuPtr{Cvoid}, pointer(col_indices_gpu)),
        reinterpret(CuPtr{Cvoid}, pointer(values_gpu)),
        _cuda_data_type(Int32), _cuda_data_type(T),
        mtype, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO)
    matrix = matrix_ref[]
    _cudss_matrix_set_distribution_row1d(matrix, Int64(first_row), Int64(last_row))

    # Create dense wrappers for x and b
    solution_ref = Ref{cudssMatrix_t}(C_NULL)
    _cudss_matrix_create_dn(solution_ref,
        Int64(local_nrows), Int64(1), Int64(local_nrows),
        reinterpret(CuPtr{Cvoid}, pointer(x_gpu)),
        _cuda_data_type(T), CUDSS_LAYOUT_COL_MAJOR)
    solution = solution_ref[]
    _cudss_matrix_set_distribution_row1d(solution, Int64(first_row), Int64(last_row))

    rhs_ref = Ref{cudssMatrix_t}(C_NULL)
    _cudss_matrix_create_dn(rhs_ref,
        Int64(local_nrows), Int64(1), Int64(local_nrows),
        reinterpret(CuPtr{Cvoid}, pointer(b_gpu)),
        _cuda_data_type(T), CUDSS_LAYOUT_COL_MAJOR)
    rhs = rhs_ref[]
    _cudss_matrix_set_distribution_row1d(rhs, Int64(first_row), Int64(last_row))

    MPI.Barrier(comm)

    # Execute analysis and factorization phases
    _cudss_execute(handle, CUDSS_PHASE_ANALYSIS, config, data, matrix, solution, rhs)
    CUDA.synchronize()
    MPI.Barrier(comm)

    _cudss_execute(handle, CUDSS_PHASE_FACTORIZATION, config, data, matrix, solution, rhs)
    CUDA.synchronize()
    MPI.Barrier(comm)

    # Create factorization object
    _cudss_count[] += 1
    id = _cudss_count[]

    F = CuDSSFactorizationMPI{T}(
        id, handle, config, data, matrix, solution, rhs,
        row_offsets, col_indices_gpu, values_gpu, x_gpu, b_gpu,
        nccl_comm_storage, nccl_comm,
        n, local_nrows, first_row, last_row,
        symmetric, copy(A.row_partition), false
    )

    # Register for MPI-safe finalization
    _cudss_registry[id] = F
    finalizer(_queue_cudss_for_destruction, F)

    return F
end

"""
    solve(F::CuDSSFactorizationMPI{T}, b::VectorMPI{T,<:CuArray}) where T

Solve the linear system using the cuDSS factorization.
Input must be a GPU-backed VectorMPI.
"""
function LinearAlgebraMPI.solve(F::CuDSSFactorizationMPI{T}, b::LinearAlgebraMPI.VectorMPI{T,<:CuArray}) where T
    F.destroyed && error("CuDSSFactorizationMPI has been destroyed")

    comm = MPI.COMM_WORLD

    # Copy b directly to RHS buffer (GPU to GPU)
    copyto!(F.b_gpu, b.v)

    # Execute solve phase
    _cudss_execute(F.handle, CUDSS_PHASE_SOLVE, F.config, F.data, F.matrix, F.solution, F.rhs)
    CUDA.synchronize()
    MPI.Barrier(comm)

    # Return GPU vector (copy from internal buffer)
    return LinearAlgebraMPI.VectorMPI(b.structural_hash, b.partition, copy(F.x_gpu))
end

"""
    \\(F::CuDSSFactorizationMPI{T}, b::VectorMPI{T,<:CuArray}) where T

Solve the linear system using backslash notation.
"""
function Base.:\(F::CuDSSFactorizationMPI{T}, b::LinearAlgebraMPI.VectorMPI{T,<:CuArray}) where T
    return LinearAlgebraMPI.solve(F, b)
end

# ============================================================================
# GPU map_rows_gpu implementation via CUDA kernels
# ============================================================================

using StaticArrays

"""
    _map_rows_gpu_kernel(f, arg1::CuMatrix, rest::CuMatrix...)

GPU-accelerated row-wise map for CUDA arrays.
"""
function LinearAlgebraMPI._map_rows_gpu_kernel(f, arg1::CuMatrix{T}, rest::CuMatrix...) where T
    n = size(arg1, 1)

    # Get output size by evaluating f on first row
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
        out_cols = length(sample_out)
    else
        out_cols = 1
    end

    output = CUDA.zeros(T, n, out_cols)
    _cuda_map_rows_kernel_dispatch(f, output, arg1, rest...)

    return output
end

function _cuda_map_rows_kernel_dispatch(f, output::CuMatrix{T}, arg1::CuMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    out_cols = size(output, 2)

    kernel = @cuda launch=false _cuda_map_rows_kernel_1arg(f, output, arg1, Val(ncols1), Val(out_cols))
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel(f, output, arg1, Val(ncols1), Val(out_cols); threads=threads, blocks=blocks)
    CUDA.synchronize()
end

function _cuda_map_rows_kernel_dispatch(f, output::CuMatrix{T}, arg1::CuMatrix{T}, arg2::CuMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    ncols2 = size(arg2, 2)
    out_cols = size(output, 2)

    kernel = @cuda launch=false _cuda_map_rows_kernel_2args(f, output, arg1, arg2, Val(ncols1), Val(ncols2), Val(out_cols))
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel(f, output, arg1, arg2, Val(ncols1), Val(ncols2), Val(out_cols); threads=threads, blocks=blocks)
    CUDA.synchronize()
end

# CUDA kernels
function _cuda_map_rows_kernel_1arg(f, output, arg1, ::Val{NC1}, ::Val{OCols}) where {NC1, OCols}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        result = f(row1)
        _cuda_write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _cuda_map_rows_kernel_2args(f, output, arg1, arg2, ::Val{NC1}, ::Val{NC2}, ::Val{OCols}) where {NC1, NC2, OCols}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        result = f(row1, row2)
        _cuda_write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

@inline function _cuda_write_result!(output, i, result::Number, ::Val{1})
    @inbounds output[i, 1] = result
    return nothing
end

@inline function _cuda_write_result!(output, i, result::SVector{N,T}, ::Val{N}) where {N,T}
    for j in 1:N
        @inbounds output[i, j] = result[j]
    end
    return nothing
end

@inline function _cuda_write_result!(output, i, result::SMatrix{M,N,T}, ::Val{MN}) where {M,N,T,MN}
    for j in 1:MN
        @inbounds output[i, j] = result[j]
    end
    return nothing
end

end # module
