# Shared test utilities for parameterized testing
# This module provides test configurations for CPU and GPU backends

module TestUtils

using SparseArrays

# Detect Metal availability BEFORE loading LinearAlgebraMPI
# (Metal must be loaded first for GPU detection to work)
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch e
    false
end

if METAL_AVAILABLE
    @info "Metal is available for GPU tests"
end

# Import LinearAlgebraMPI after Metal check
using LinearAlgebraMPI

# Backend configurations: (ScalarType, to_backend_fn, backend_name)
const CPU_CONFIGS = [
    (Float64, identity, "CPU"),
    (ComplexF64, identity, "CPU")
]

const GPU_CONFIGS = if METAL_AVAILABLE
    [
        (Float32, LinearAlgebraMPI.mtl, "Metal")
        # ComplexF32 skipped - Julia's complex ops use Float64 internally, unsupported on Metal
    ]
else
    Tuple{Type, Function, String}[]
end

const ALL_CONFIGS = [CPU_CONFIGS; GPU_CONFIGS]

# For operations that only work on CPU (like MUMPS factorization)
const CPU_ONLY_CONFIGS = CPU_CONFIGS

"""
    tridiagonal_matrix(T, n=8)

Generate a deterministic tridiagonal test matrix of type T.
"""
function tridiagonal_matrix(::Type{T}, n::Int=8) where T
    I = [1:n; 1:n-1; 2:n]
    J = [1:n; 2:n; 1:n-1]
    V = if T <: Complex
        T.([2.0*ones(n); -0.5*ones(n-1); -0.5*ones(n-1)]) .+
        im .* T.([0.1*ones(n); 0.2*ones(n-1); -0.2*ones(n-1)])
    else
        T.([2.0*ones(n); -0.5*ones(n-1); -0.5*ones(n-1)])
    end
    sparse(I, J, V, n, n)
end

"""
    dense_matrix(T, m, n)

Generate a deterministic dense test matrix of type T.
"""
function dense_matrix(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for i in 1:m, j in 1:n
        if T <: Complex
            A[i, j] = T(i + j) + im * T(i - j)
        else
            A[i, j] = T(i + j)
        end
    end
    A
end

"""
    test_vector(T, n=8)

Generate a deterministic test vector of type T.
"""
function test_vector(::Type{T}, n::Int=8) where T
    if T <: Complex
        T.(1:n) .+ im .* T.(n:-1:1)
    else
        T.(collect(1.0:n))
    end
end

"""
    test_vector_pair(T, n=8)

Generate two deterministic test vectors for addition/subtraction tests.
"""
function test_vector_pair(::Type{T}, n::Int=8) where T
    u = test_vector(T, n)
    v = if T <: Complex
        T.(n:-1:1) .+ im .* T.(1:n)
    else
        T.(collect(Float64(n):-1.0:1))
    end
    u, v
end

"""
    tolerance(T)

Return appropriate tolerance for type T.
Float32 tolerance is looser (1e-4) to accommodate accumulated errors
in matrix operations like transpose(A) * B.
"""
tolerance(::Type{Float64}) = 1e-12
tolerance(::Type{ComplexF64}) = 1e-12
tolerance(::Type{Float32}) = 1e-4
tolerance(::Type{ComplexF32}) = 1e-4

"""
    to_cpu(x)

Convert to CPU if on GPU, otherwise return as-is.
Works for both CPU and GPU arrays.
"""
to_cpu(x) = x

# For VectorMPI: return as-is if already CPU, convert if GPU
to_cpu(x::VectorMPI{T, Vector{T}}) where T = x
to_cpu(x::SparseMatrixMPI{T, Ti, Vector{T}}) where {T, Ti} = x
to_cpu(x::MatrixMPI{T, Matrix{T}}) where T = x

# GPU versions (only available when Metal is loaded)
if METAL_AVAILABLE
    to_cpu(x::VectorMPI{T, <:Metal.MtlVector}) where T = LinearAlgebraMPI.cpu(x)
    to_cpu(x::SparseMatrixMPI{T, Ti, <:Metal.MtlVector}) where {T, Ti} = LinearAlgebraMPI.cpu(x)
    to_cpu(x::MatrixMPI{T, <:Metal.MtlMatrix}) where T = LinearAlgebraMPI.cpu(x)
end

"""
    local_values(v::VectorMPI)

Get local values as a CPU array for comparison.
Works for both CPU and GPU vectors.
"""
function local_values(v::VectorMPI{T, Vector{T}}) where T
    return v.v
end

if METAL_AVAILABLE
    function local_values(v::VectorMPI{T, <:Metal.MtlVector}) where T
        return Array(v.v)
    end
end

export METAL_AVAILABLE, CPU_CONFIGS, GPU_CONFIGS, ALL_CONFIGS, CPU_ONLY_CONFIGS
export tridiagonal_matrix, dense_matrix, test_vector, test_vector_pair
export tolerance, to_cpu, local_values

end # module
