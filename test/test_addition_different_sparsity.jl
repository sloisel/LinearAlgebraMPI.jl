# Tests for matrix addition with different sparsity patterns
# Parameterized over scalar types and backends (CPU and GPU)

# Check Metal availability BEFORE loading MPI
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch e
    false
end

using Test
using MPI
MPI.Init()

using LinearAlgebraMPI
using LinearAlgebraMPI: SparseMatrixMPI, VectorMPI, io0, clear_plan_cache!
using SparseArrays
using LinearAlgebra

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

ts = @testset QuietTestSet "Addition Different Sparsity" begin

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)

    println(io0(), "[test] Matrix addition with different sparsity patterns ($T, $backend_name)")

    # This test reproduces a bug where adding matrices with different sparsity patterns
    # can fail due to cached MatrixPlan having stale local_ranges

    n = 8

    # Matrix A: tridiagonal pattern
    A_native = spdiagm(n, n,
        -1 => T.(ones(n-1)),
        0 => T.(2*ones(n)),
        1 => T.(ones(n-1))
    )

    # Matrix B: different pattern (only diagonal and one off-diagonal)
    B_native = spdiagm(n, n,
        0 => T.(3*ones(n)),
        2 => T.(0.5*ones(n-2))  # Different off-diagonal than A
    )

    A_mpi = to_backend(SparseMatrixMPI{T}(A_native))
    B_mpi = to_backend(SparseMatrixMPI{T}(B_native))

    # Test A + B
    C_mpi = A_mpi + B_mpi
    C_native = SparseMatrixCSC(C_mpi)
    C_expected = A_native + B_native

    @test norm(C_native - C_expected) < TOL


    println(io0(), "[test] D' * W * D products ($T, $backend_name)")

    # Create D operators similar to fem1d
    dx = spdiagm(n, n, 0 => T.(-ones(n)), 1 => T.(ones(n-1)))
    dx[end, end] = zero(T)  # Boundary
    id = spdiagm(n, n, 0 => T.(ones(n)))  # Identity matrix

    D_dx = to_backend(SparseMatrixMPI{T}(dx))
    D_id = to_backend(SparseMatrixMPI{T}(id))

    # Create a diagonal weight matrix
    w = T.(ones(n) * 0.5)
    W = to_backend(SparseMatrixMPI{T}(spdiagm(n, n, 0 => w)))

    # Compute products with different structure
    M1 = D_id' * W * D_dx
    M2 = D_dx' * W * D_id

    # This addition previously failed with BoundsError due to cached plan issue
    M_sum = M1 + M2
    M_sum_native = SparseMatrixCSC(M_sum)

    # Compute expected result using native Julia
    M1_expected = id' * spdiagm(n, n, 0 => w) * dx
    M2_expected = dx' * spdiagm(n, n, 0 => w) * id
    M_sum_expected = M1_expected + M2_expected

    @test norm(M_sum_native - M_sum_expected) < TOL


    println(io0(), "[test] Hessian-style accumulation ($T, $backend_name)")

    # Recreate matrices for this test
    D_dx2 = to_backend(SparseMatrixMPI{T}(dx))
    D_id2 = to_backend(SparseMatrixMPI{T}(id))
    W2 = to_backend(SparseMatrixMPI{T}(spdiagm(n, n, 0 => w)))

    # Start with one product
    H = D_dx2' * W2 * D_dx2

    # Add another product with different structure
    H = H + D_id2' * W2 * D_id2

    # Add cross terms (this pattern caused the original bug)
    cross1 = D_dx2' * W2 * D_id2
    cross2 = D_id2' * W2 * D_dx2
    cross_sum = cross1 + cross2
    H = H + cross_sum
    H_native_final = SparseMatrixCSC(H)

    # Compute expected
    W_diag = spdiagm(n, n, 0 => w)
    H_expected = dx' * W_diag * dx +
                 id' * W_diag * id +
                 dx' * W_diag * id +
                 id' * W_diag * dx

    @test norm(H_native_final - H_expected) < TOL


    println(io0(), "[test] Exact bug-triggering pattern ($T, $backend_name)")

    # Create fresh diagonal matrices each time
    foo1 = to_backend(SparseMatrixMPI{T}(spdiagm(n, n, 0 => T.(0.3 * ones(n)))))
    foo2 = to_backend(SparseMatrixMPI{T}(spdiagm(n, n, 0 => T.(0.7 * ones(n)))))
    D_dx3 = to_backend(SparseMatrixMPI{T}(dx))

    # Compute products: these have DIFFERENT sparsity patterns
    prod1 = foo1 * D_dx3
    prod2 = D_dx3' * foo2

    # This is where the bug occurred - adding matrices with different structure
    sum_result = prod1 + prod2
    sum_native = SparseMatrixCSC(sum_result)

    # Expected
    foo1_native = spdiagm(n, n, 0 => T.(0.3 * ones(n)))
    foo2_native = spdiagm(n, n, 0 => T.(0.7 * ones(n)))
    sum_expected = foo1_native * dx + dx' * foo2_native

    @test norm(sum_native - sum_expected) < TOL

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

println(io0(), "Test Summary: Addition Different Sparsity | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
