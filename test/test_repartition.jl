# MPI test for repartition
# This file is executed under mpiexec by runtests.jl
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
using SparseArrays
using LinearAlgebra: norm
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

ts = @testset QuietTestSet "Repartition" begin

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)

    println(io0(), "[test] VectorMPI repartition ($T, $backend_name)")

    # Test 1: VectorMPI uniform to non-uniform partition
    n = 12
    v_global = T.(collect(1.0:n))
    v = to_backend(VectorMPI(v_global))
    # Create non-uniform partition
    new_p = LinearAlgebraMPI.uniform_partition(n, nranks)
    # Shift elements: first rank gets fewer, last gets more
    if nranks >= 2
        new_p = [1]
        total = 0
        for r in 0:(nranks-1)
            if r < nranks - 1
                count = div(n, nranks) - 1 + (r < mod(n, nranks) ? 1 : 0)
            else
                count = n - total
            end
            total += count
            push!(new_p, total + 1)
        end
    end

    v_repart = repartition(v, new_p)
    v_repart_global = Vector(v_repart)
    @test norm(v_repart_global - v_global) < TOL
    @test v_repart.partition == new_p


    println(io0(), "[test] VectorMPI same partition fast path ($T, $backend_name)")

    v_same = to_backend(VectorMPI(v_global))
    v2 = repartition(v_same, v_same.partition)
    @test v2 === v_same  # Should be same object


    println(io0(), "[test] VectorMPI plan caching ($T, $backend_name)")

    v_cache = to_backend(VectorMPI(v_global))
    LinearAlgebraMPI.clear_plan_cache!()
    v3_repart = repartition(v_cache, new_p)
    v4_repart = repartition(v_cache, new_p)
    @test v3_repart.structural_hash == v4_repart.structural_hash


    println(io0(), "[test] MatrixMPI repartition ($T, $backend_name)")

    # Test dense matrix repartition
    m, ncols = 8, 4
    M_global = T.(reshape(Float64.(1:m*ncols), m, ncols))
    M = to_backend(MatrixMPI(M_global))

    # Create new partition for rows
    new_row_p = LinearAlgebraMPI.uniform_partition(m, nranks)
    if nranks >= 2
        # Create uneven partition
        new_row_p = [1]
        total = 0
        for r in 0:(nranks-1)
            if r == 0
                count = 1
            else
                count = div(m - 1, nranks - 1) + (r - 1 < mod(m - 1, nranks - 1) ? 1 : 0)
            end
            total += count
            push!(new_row_p, total + 1)
        end
    end

    M_repart = repartition(M, new_row_p)
    M_repart_global = Matrix(M_repart)
    @test norm(M_repart_global - M_global) < TOL
    @test M_repart.row_partition == new_row_p
    @test M_repart.col_partition == M.col_partition  # unchanged


    println(io0(), "[test] MatrixMPI same partition fast path ($T, $backend_name)")

    M_same = to_backend(MatrixMPI(M_global))
    M2 = repartition(M_same, M_same.row_partition)
    @test M2 === M_same


    println(io0(), "[test] SparseMatrixMPI repartition ($T, $backend_name)")

    # Test sparse matrix repartition
    m_sparse, n_sparse = 8, 6
    I_idx = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7, 2, 4, 6, 8]
    J_idx = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 4, 5, 6, 1]
    V_sparse = T.(1:length(I_idx))
    A_global = sparse(I_idx, J_idx, V_sparse, m_sparse, n_sparse)

    A = to_backend(SparseMatrixMPI{T}(A_global))

    # Create new partition (must differ from uniform to test actual repartition)
    # Uniform for 8 rows, 4 ranks is [1, 3, 5, 7, 9] (2 rows each)
    # Use [1, 4, 6, 8, 9] instead (3, 2, 2, 1 rows per rank)
    new_sparse_p = if nranks == 4
        [1, 4, 6, 8, 9]
    elseif nranks >= 2
        # Give first rank one extra row, last rank one fewer
        p = [1]
        base = div(m_sparse, nranks)
        remainder = mod(m_sparse, nranks)
        total = 0
        for r in 0:(nranks-1)
            count = base + (r < remainder ? 1 : 0)
            # Shift: first rank gets +1, last rank gets -1
            if r == 0
                count += 1
            elseif r == nranks - 1
                count -= 1
            end
            total += count
            push!(p, total + 1)
        end
        p
    else
        LinearAlgebraMPI.uniform_partition(m_sparse, nranks)
    end

    A_repart = repartition(A, new_sparse_p)
    A_repart_global = SparseMatrixCSC(A_repart)
    A_global_from_dist = SparseMatrixCSC(A)
    @test norm(A_repart_global - A_global_from_dist, Inf) < TOL
    @test A_repart.row_partition == new_sparse_p
    @test A_repart.col_partition == A.col_partition


    println(io0(), "[test] SparseMatrixMPI nnz preserved ($T, $backend_name)")

    @test nnz(A_repart) == nnz(A)


    println(io0(), "[test] SparseMatrixMPI same partition fast path ($T, $backend_name)")

    A_same = to_backend(SparseMatrixMPI{T}(A_global))
    A2 = repartition(A_same, A_same.row_partition)
    @test A2 === A_same


    println(io0(), "[test] Operations after repartition ($T, $backend_name)")

    # Test that repartitioned matrix can be used in operations
    A_for_ops = to_backend(SparseMatrixMPI{T}(A_global))
    A_repart_ops = repartition(A_for_ops, new_sparse_p)
    x = to_backend(VectorMPI(ones(T, n_sparse)))
    y_orig = A_for_ops * x
    y_repart = A_repart_ops * x
    y_diff = norm(Vector(y_orig) - Vector(y_repart))
    @test y_diff < TOL


    println(io0(), "[test] Repartition plan caching ($T, $backend_name)")

    A_cache = to_backend(SparseMatrixMPI{T}(A_global))
    LinearAlgebraMPI.clear_plan_cache!()
    A3_repart = repartition(A_cache, new_sparse_p)
    A4_repart = repartition(A_cache, new_sparse_p)
    @test A3_repart.structural_hash == A4_repart.structural_hash

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

println(io0(), "Test Summary: Repartition | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
