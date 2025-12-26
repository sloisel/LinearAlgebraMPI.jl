# Tests for local constructors: VectorMPI_local, MatrixMPI_local, SparseMatrixMPI_local
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
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

ts = @testset QuietTestSet "Local Constructors" begin

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)

    println(io0(), "[test] VectorMPI_local basic ($T, $backend_name)")

    # Test VectorMPI_local: each rank provides different-sized local parts
    # Create local vectors that, when concatenated, form [1, 2, 3, ..., 10]
    local_sizes = [div(10, nranks) + (r <= mod(10, nranks) ? 1 : 0) for r in 1:nranks]
    global_start = sum(local_sizes[1:rank]) + 1
    global_end = sum(local_sizes[1:rank+1])
    v_local = T.(collect(global_start:global_end))

    v_mpi = to_backend(VectorMPI_local(v_local))
    v_global = Vector(v_mpi)

    # All ranks should have the same global vector
    @test norm(v_global - T.(1:10)) < TOL
    @test length(v_mpi) == 10


    println(io0(), "[test] VectorMPI_local roundtrip consistency ($T, $backend_name)")

    # Test that VectorMPI_local produces same result as VectorMPI for default partition
    v_original = T.([1.5, -2.3, 3.7, 4.1, -5.9, 6.2, 7.8, -8.4])
    v_from_global = to_backend(VectorMPI(v_original))

    # Extract local part (need CPU array for VectorMPI_local)
    v_local_extract = Array(v_from_global.v)
    v_from_local = to_backend(VectorMPI_local(v_local_extract))
    v_back = Vector(v_from_local)

    @test norm(v_back - v_original) < TOL
    @test v_from_local.partition == v_from_global.partition


    println(io0(), "[test] MatrixMPI_local basic ($T, $backend_name)")

    # Test MatrixMPI_local: each rank provides some rows of a matrix
    # Create a 10x4 matrix distributed across ranks
    m_global = 10
    n_cols = 4
    row_sizes = [div(m_global, nranks) + (r <= mod(m_global, nranks) ? 1 : 0) for r in 1:nranks]
    row_start = sum(row_sizes[1:rank]) + 1
    row_end = sum(row_sizes[1:rank+1])

    # Create deterministic local matrix based on global row indices
    M_local = T.([Float64(i + j*0.1) for i in row_start:row_end, j in 1:n_cols])
    M_mpi = to_backend(MatrixMPI_local(M_local))

    # Verify size
    @test size(M_mpi) == (m_global, n_cols)

    # Gather and verify
    M_gathered = Matrix(M_mpi)
    M_expected = T.([Float64(i + j*0.1) for i in 1:m_global, j in 1:n_cols])
    @test norm(M_gathered - M_expected) < TOL


    println(io0(), "[test] MatrixMPI_local roundtrip consistency ($T, $backend_name)")

    # Test that MatrixMPI_local produces same result as MatrixMPI for default partition
    M_original = T.([1.1 2.2 3.3;
                     4.4 5.5 6.6;
                     7.7 8.8 9.9;
                     10.0 11.1 12.2;
                     13.3 14.4 15.5])
    M_from_global = to_backend(MatrixMPI(M_original))

    # Extract local part (need CPU array for MatrixMPI_local)
    M_local_extract = Array(M_from_global.A)
    M_from_local = to_backend(MatrixMPI_local(M_local_extract))
    M_back = Matrix(M_from_local)

    @test norm(M_back - M_original) < TOL
    @test M_from_local.row_partition == M_from_global.row_partition


    println(io0(), "[test] SparseMatrixMPI_local basic ($T, $backend_name)")

    # Test SparseMatrixMPI_local: each rank provides local rows
    # Create a 12x8 sparse matrix with known structure
    m_sparse = 12
    n_sparse = 8
    sparse_row_sizes = [div(m_sparse, nranks) + (r <= mod(m_sparse, nranks) ? 1 : 0) for r in 1:nranks]
    sparse_row_start = sum(sparse_row_sizes[1:rank]) + 1
    sparse_row_end = sum(sparse_row_sizes[1:rank+1])
    local_sparse_nrows = sparse_row_end - sparse_row_start + 1

    # Build local sparse rows: diagonal + some off-diagonal entries
    I_local = Int[]
    J_local = Int[]
    V_local = T[]

    for local_row in 1:local_sparse_nrows
        global_row = sparse_row_start + local_row - 1
        # Diagonal entry if within bounds
        if global_row <= n_sparse
            push!(I_local, local_row)
            push!(J_local, global_row)
            push!(V_local, T(global_row))
        end
        # Off-diagonal entry
        col_off = mod(global_row, n_sparse) + 1
        push!(I_local, local_row)
        push!(J_local, col_off)
        push!(V_local, T(global_row) * T(0.1))
    end

    # Create local CSC in transpose form (columns = local rows, rowval = global columns)
    local_csc = sparse(J_local, I_local, V_local, n_sparse, local_sparse_nrows)
    local_transpose = transpose(local_csc)

    S_mpi = to_backend(SparseMatrixMPI_local(local_transpose))
    @test size(S_mpi) == (m_sparse, n_sparse)


    println(io0(), "[test] SparseMatrixMPI_local roundtrip consistency ($T, $backend_name)")

    # Test roundtrip: global -> partition -> local -> rebuild
    # Use a deterministic sparse matrix
    I_orig = [1, 2, 3, 4, 5, 6, 7, 8, 1, 5, 9, 10]
    J_orig = [1, 2, 3, 4, 5, 6, 7, 8, 5, 1, 2, 3]
    V_orig = T.([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 12.2])
    S_original = sparse(I_orig, J_orig, V_orig, 12, 10)

    S_from_global = to_backend(SparseMatrixMPI{T}(S_original))

    # Extract local part - using explicit CSR arrays (nzval needs CPU conversion)
    col_indices = S_from_global.col_indices
    nzval_cpu = Array(S_from_global.nzval)

    # Rebuild CSC from explicit arrays with global indices
    AT_uncompressed = SparseMatrixCSC(
        size(S_original, 2),  # original global ncols
        S_from_global.nrows_local,  # number of local rows
        copy(S_from_global.rowptr),  # becomes colptr in CSC
        [col_indices[r] for r in S_from_global.colval],  # map local to global indices
        copy(nzval_cpu)
    )

    # Rebuild from local
    S_from_local = to_backend(SparseMatrixMPI_local(transpose(AT_uncompressed)))
    S_back = SparseMatrixCSC(S_from_local)

    @test norm(S_back - S_original, Inf) < TOL
    @test S_from_local.row_partition == S_from_global.row_partition


    println(io0(), "[test] MatrixMPI_local * VectorMPI_local ($T, $backend_name)")

    # Test that locally constructed matrices work with operations
    # Create compatible matrix and vector
    m_op = 8
    n_op = 6
    op_row_sizes = [div(m_op, nranks) + (r <= mod(m_op, nranks) ? 1 : 0) for r in 1:nranks]
    op_row_start = sum(op_row_sizes[1:rank]) + 1
    op_row_end = sum(op_row_sizes[1:rank+1])

    A_local_op = T.([Float64(i + j*0.1) for i in op_row_start:op_row_end, j in 1:n_op])
    A_mpi_op = to_backend(MatrixMPI_local(A_local_op))

    # Create vector partition that matches columns
    v_op_sizes = [div(n_op, nranks) + (r <= mod(n_op, nranks) ? 1 : 0) for r in 1:nranks]
    v_op_start = sum(v_op_sizes[1:rank]) + 1
    v_op_end = sum(v_op_sizes[1:rank+1])
    v_local_op = T.(collect(v_op_start:v_op_end))
    v_mpi_op = to_backend(VectorMPI_local(v_local_op))

    # Compute A * v
    y_mpi = A_mpi_op * v_mpi_op

    # Verify against expected result
    A_full = Matrix(A_mpi_op)
    v_full = Vector(v_mpi_op)
    y_expected = A_full * v_full
    y_result = Vector(y_mpi)

    @test norm(y_result - y_expected) < TOL


    println(io0(), "[test] SparseMatrixMPI_local * VectorMPI_local ($T, $backend_name)")

    # Test sparse matrix-vector multiplication with local constructors
    I_sp = [1, 2, 3, 4, 5, 6, 1, 3, 5, 7, 9]
    J_sp = [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2]
    V_sp = T.([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    S_sp_global = sparse(I_sp, J_sp, V_sp, 10, 8)

    S_sp_mpi = to_backend(SparseMatrixMPI{T}(S_sp_global))
    v_sp_global = T.(1:8)
    v_sp_mpi = to_backend(VectorMPI(v_sp_global))

    # Compute result
    y_sp = S_sp_mpi * v_sp_mpi
    y_sp_expected = S_sp_global * v_sp_global
    y_sp_result = Vector(y_sp)

    @test norm(y_sp_result - y_sp_expected) < TOL

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

println(io0(), "Test Summary: Local Constructors | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
