# Tests for new operations
# Parameterized over scalar types and backends (CPU and GPU)

# Check Metal availability BEFORE loading MPI
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch e
    false
end

using MPI
MPI.Init()

using LinearAlgebraMPI
using LinearAlgebra
using SparseArrays
using Test
using Random

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

ts = @testset QuietTestSet "New Operations" begin

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)
    is_complex = T <: Complex
    Treal = real(T)

    # Use fixed seed for deterministic random data across all ranks
    Random.seed!(42)

    # Create test matrices and vectors deterministically
    n = 8  # Matrix size
    m = 6  # For non-square matrices

    # Create a test sparse matrix (n x n) with deterministic values
    A_sparse_local_vals = T.([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                              0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    A_sparse_local = sparse(
        [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 1],
        A_sparse_local_vals, n, n
    )
    A_sparse_local = A_sparse_local + transpose(A_sparse_local) + T(2) * I
    A_sparse = to_backend(SparseMatrixMPI{T}(A_sparse_local))

    # Create a test dense matrix (n x m) with deterministic values
    B_dense_local = T.([Float64(i + j*0.1) for i in 1:n, j in 1:m])
    B_dense = to_backend(MatrixMPI(B_dense_local))

    # Create a test dense matrix (m x n)
    C_dense_local = T.([Float64(i*0.1 + j) for i in 1:m, j in 1:n])
    C_dense = to_backend(MatrixMPI(C_dense_local))

    # Create test vectors
    x_local = T.(Float64.(1:n) .+ 0.1)
    x = to_backend(VectorMPI(x_local))

    y_local = T.(Float64.(n:-1:1) .+ 0.1)
    y = to_backend(VectorMPI(y_local))


    println(io0(), "[test] transpose(SparseMatrixMPI) * VectorMPI ($T, $backend_name)")
    result1 = transpose(A_sparse) * x
    expected1 = transpose(A_sparse_local) * x_local
    @test norm(Vector(result1) - expected1) < TOL


    println(io0(), "[test] SparseMatrixMPI * MatrixMPI ($T, $backend_name)")
    result2 = A_sparse * B_dense
    expected2 = A_sparse_local * B_dense_local
    @test norm(Matrix(result2) - expected2) < TOL


    println(io0(), "[test] transpose(SparseMatrixMPI) * MatrixMPI ($T, $backend_name)")
    result3 = transpose(A_sparse) * B_dense
    expected3 = transpose(A_sparse_local) * B_dense_local
    @test norm(Matrix(result3) - expected3) < TOL


    println(io0(), "[test] MatrixMPI * SparseMatrixMPI ($T, $backend_name)")
    # D_sparse is n x m, E_dense is m x n, so E_dense * D_sparse is m x m
    D_sparse_local_vals = T.([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    D_sparse_local = sparse(
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        D_sparse_local_vals, n, m
    )
    D_sparse = to_backend(SparseMatrixMPI{T}(D_sparse_local))

    E_dense_local = T.([Float64(i*0.2 + j*0.3) for i in 1:m, j in 1:n])
    E_dense = to_backend(MatrixMPI(E_dense_local))

    result4 = E_dense * D_sparse
    expected4 = E_dense_local * D_sparse_local
    @test norm(Matrix(result4) - expected4) < TOL


    println(io0(), "[test] transpose(MatrixMPI) * MatrixMPI ($T, $backend_name)")
    # transpose(B_dense) is m x n, B_dense is n x m, so result is m x m
    result5 = transpose(B_dense) * B_dense
    expected5 = transpose(B_dense_local) * B_dense_local
    @test norm(Matrix(result5) - expected5) < TOL


    println(io0(), "[test] transpose(MatrixMPI) * SparseMatrixMPI ($T, $backend_name)")
    # transpose(B_dense) is m x n, A_sparse is n x n
    result6 = transpose(B_dense) * A_sparse
    expected6 = transpose(B_dense_local) * A_sparse_local
    @test norm(Matrix(result6) - expected6) < TOL


    println(io0(), "[test] MatrixMPI column indexing ($T, $backend_name)")
    for k in 1:m
        result7 = B_dense[:, k]
        expected7 = B_dense_local[:, k]
        @test norm(Vector(result7) - expected7) < TOL
    end


    println(io0(), "[test] SparseMatrixMPI column indexing ($T, $backend_name)")
    for k in 1:n
        result8 = A_sparse[:, k]
        expected8 = Vector(A_sparse_local[:, k])
        @test norm(Vector(result8) - expected8) < TOL
    end


    println(io0(), "[test] dot(VectorMPI, VectorMPI) ($T, $backend_name)")
    result9 = dot(x, y)
    expected9 = dot(x_local, y_local)
    @test abs(result9 - expected9) < TOL

    # Self dot product
    result9b = dot(x, x)
    expected9b = dot(x_local, x_local)
    @test abs(result9b - expected9b) < TOL


    println(io0(), "[test] UniformScaling A + λI ($T, $backend_name)")
    λ = T(3.5)
    result10 = A_sparse + λ*I
    expected10 = A_sparse_local + λ*I
    @test norm(SparseMatrixCSC(result10) - expected10, Inf) < TOL


    println(io0(), "[test] UniformScaling A - λI ($T, $backend_name)")
    result11 = A_sparse - λ*I
    expected11 = A_sparse_local - λ*I
    @test norm(SparseMatrixCSC(result11) - expected11, Inf) < TOL


    println(io0(), "[test] UniformScaling λI + A ($T, $backend_name)")
    result12 = λ*I + A_sparse
    expected12 = λ*I + A_sparse_local
    @test norm(SparseMatrixCSC(result12) - expected12, Inf) < TOL


    println(io0(), "[test] UniformScaling λI - A ($T, $backend_name)")
    result13 = λ*I - A_sparse
    expected13 = λ*I - A_sparse_local
    @test norm(SparseMatrixCSC(result13) - expected13, Inf) < TOL

end  # for (T, to_backend, backend_name)

end  # QuietTestSet

# Aggregate counts across ranks
local_counts = [
    get(ts.counts, :pass, 0),
    get(ts.counts, :fail, 0),
    get(ts.counts, :error, 0),
    get(ts.counts, :broken, 0),
    get(ts.counts, :skip, 0),
]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

println(io0(), "Test Summary: New Operations | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
