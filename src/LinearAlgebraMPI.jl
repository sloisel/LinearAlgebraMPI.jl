module LinearAlgebraMPI

using MPI
using Blake3Hash
using SparseArrays
import LinearAlgebra
using LinearAlgebra: Transpose, Adjoint

export SparseMatrixMPI, MatrixMPI, VectorMPI, MatrixPlan, TransposePlan, VectorPlan, DenseMatrixVectorPlan, DenseTransposePlan, clear_plan_cache!, execute_plan!

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

# Include the three component files
include("vectors.jl")
include("dense.jl")
include("sparse.jl")

end # module LinearAlgebraMPI
