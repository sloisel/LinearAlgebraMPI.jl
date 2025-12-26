# Tests for indexing operations (getindex, setindex!)
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

ts = @testset QuietTestSet "Indexing" begin

# Indexing operations use scalar indexing throughout (v.v[i], A.A[i,j], etc.)
# which is not supported on GPU. Use CPU-only configs.
for (T, to_backend, backend_name) in TestUtils.CPU_ONLY_CONFIGS
    TOL = TestUtils.tolerance(T)
    is_complex = T <: Complex

    # Create deterministic test data
    n = 12  # Vector size, divisible by common rank counts

    # VectorMPI test data
    v_global = T.(collect(1.0:Float64(n)))
    v = to_backend(VectorMPI(v_global))


    println(io0(), "[test] VectorMPI getindex ($T, $backend_name)")

    # Test VectorMPI getindex - various indices
    for i in 1:n
        result = v[i]
        @test abs(result - v_global[i]) < TOL
    end


    println(io0(), "[test] VectorMPI setindex! ($T, $backend_name)")

    # Test VectorMPI setindex! - create a fresh vector to modify
    v_modify = to_backend(VectorMPI(T.(collect(1.0:Float64(n)))))

    # Set each element to a new value (all ranks participate)
    for i in 1:n
        v_modify[i] = T(i * 10)
    end

    # Verify all values were set correctly
    for i in 1:n
        result = v_modify[i]
        @test abs(result - T(i * 10)) < TOL
    end


    # SparseMatrixMPI test data
    I_vals = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7]
    J_vals = [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8]
    V_vals = T.([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.1, 0.3, 0.5, 0.7])
    A_global = sparse(I_vals, J_vals, V_vals, n, n)
    A = to_backend(SparseMatrixMPI{T}(A_global))


    println(io0(), "[test] SparseMatrixMPI getindex - existing entries ($T, $backend_name)")

    # Test SparseMatrixMPI getindex - existing entries
    for k in eachindex(I_vals)
        i, j = I_vals[k], J_vals[k]
        result = A[i, j]
        @test abs(result - A_global[i, j]) < TOL
    end


    println(io0(), "[test] SparseMatrixMPI getindex - structural zeros ($T, $backend_name)")

    # Test SparseMatrixMPI getindex - structural zeros
    test_zero_positions = [(1, 3), (1, 4), (2, 1), (2, 4)]
    for (i, j) in test_zero_positions
        result = A[i, j]
        @test abs(result) < TOL
    end


    println(io0(), "[test] SparseMatrixMPI setindex! - modify existing ($T, $backend_name)")

    # Test SparseMatrixMPI setindex! - modify existing entries
    A_modify = to_backend(SparseMatrixMPI{T}(sparse(I_vals, J_vals, copy(V_vals), n, n)))

    # Modify each existing entry
    for k in eachindex(I_vals)
        i, j = I_vals[k], J_vals[k]
        new_val = T(k * 100)
        A_modify[i, j] = new_val
    end

    # Verify all values were modified correctly
    for k in eachindex(I_vals)
        i, j = I_vals[k], J_vals[k]
        result = A_modify[i, j]
        @test abs(result - T(k * 100)) < TOL
    end


    # MatrixMPI test data
    M_global = T.([Float64(i + j * 0.1) for i in 1:n, j in 1:n])
    M = to_backend(MatrixMPI(M_global))


    println(io0(), "[test] MatrixMPI getindex ($T, $backend_name)")

    # Test MatrixMPI getindex - various positions
    for i in 1:n
        for j in [1, div(n, 2), n]
            result = M[i, j]
            @test abs(result - M_global[i, j]) < TOL
        end
    end


    println(io0(), "[test] MatrixMPI setindex! ($T, $backend_name)")

    # Test MatrixMPI setindex!
    M_modify = to_backend(MatrixMPI(copy(M_global)))

    # Modify several entries
    test_positions = [(1, 1), (3, 5), (n, n), (div(n, 2), div(n, 2))]
    for (i, j) in test_positions
        new_val = T(i * 100 + j)
        M_modify[i, j] = new_val
    end

    # Verify all modifications
    for (i, j) in test_positions
        result = M_modify[i, j]
        expected = T(i * 100 + j)
        @test abs(result - expected) < TOL
    end


    println(io0(), "[test] Edge cases - boundary indices ($T, $backend_name)")

    # Test edge cases: first and last elements
    @test abs(v[1] - v_global[1]) < TOL
    @test abs(v[n] - v_global[n]) < TOL

    # MatrixMPI - corners
    @test abs(M[1, 1] - M_global[1, 1]) < TOL
    @test abs(M[1, n] - M_global[1, n]) < TOL
    @test abs(M[n, 1] - M_global[n, 1]) < TOL
    @test abs(M[n, n] - M_global[n, n]) < TOL


    println(io0(), "[test] VectorMPI range getindex ($T, $backend_name)")

    # Test VectorMPI range getindex - extract subvector
    for (rng_start, rng_end) in [(1, 4), (3, 7), (5, n)]
        rng = rng_start:rng_end
        w = v[rng]
        @test length(w) == length(rng)
        # Verify the result
        w_gathered = Vector(w)
        for (local_idx, global_idx) in enumerate(rng)
            @test abs(w_gathered[local_idx] - v_global[global_idx]) < TOL
        end
    end


    println(io0(), "[test] VectorMPI range setindex! scalar ($T, $backend_name)")

    # Test VectorMPI range setindex! with scalar
    v_range_modify = to_backend(VectorMPI(copy(v_global)))
    v_range_modify[3:6] = T(99)
    for i in 1:n
        if 3 <= i <= 6
            @test abs(v_range_modify[i] - T(99)) < TOL
        else
            @test abs(v_range_modify[i] - v_global[i]) < TOL
        end
    end


    println(io0(), "[test] VectorMPI range setindex! vector ($T, $backend_name)")

    # Test VectorMPI range setindex! with a regular vector
    v_range_modify2 = to_backend(VectorMPI(copy(v_global)))
    v_range_modify2[3:6] = T.([100.0, 200.0, 300.0, 400.0])
    for i in 1:n
        if 3 <= i <= 6
            @test abs(v_range_modify2[i] - T((i - 2) * 100)) < TOL
        else
            @test abs(v_range_modify2[i] - v_global[i]) < TOL
        end
    end


    println(io0(), "[test] MatrixMPI column getindex ($T, $backend_name)")

    # Test MatrixMPI column getindex
    for j in [1, div(n, 2), n]
        col = M[:, j]
        col_gathered = Vector(col)
        for i in 1:n
            @test abs(col_gathered[i] - M_global[i, j]) < TOL
        end
    end


    println(io0(), "[test] MatrixMPI row getindex ($T, $backend_name)")

    # Test MatrixMPI row getindex
    for i in [1, div(n, 2), n]
        row = M[i, :]
        row_gathered = Vector(row)
        for j in 1:n
            @test abs(row_gathered[j] - M_global[i, j]) < TOL
        end
    end


    println(io0(), "[test] MatrixMPI submatrix getindex ($T, $backend_name)")

    # Test MatrixMPI submatrix getindex
    sub = M[2:5, 3:7]
    sub_gathered = Matrix(sub)
    for i in 1:4
        for j in 1:5
            @test abs(sub_gathered[i, j] - M_global[i+1, j+2]) < TOL
        end
    end


    println(io0(), "[test] SparseMatrixMPI column getindex ($T, $backend_name)")

    # Test SparseMatrixMPI column getindex
    for j in [1, 2, 4]
        col = A[:, j]
        col_gathered = Vector(col)
        for i in 1:n
            @test abs(col_gathered[i] - A_global[i, j]) < TOL
        end
    end


    println(io0(), "[test] SparseMatrixMPI submatrix getindex ($T, $backend_name)")

    # Test SparseMatrixMPI submatrix getindex
    sub_sp = A[1:4, 1:4]
    sub_sp_csc = SparseMatrixCSC(sub_sp)
    ref_sub = A_global[1:4, 1:4]
    @test norm(sub_sp_csc - ref_sub, Inf) < TOL


    println(io0(), "[test] MatrixMPI column setindex! ($T, $backend_name)")

    # Test MatrixMPI column setindex!
    M_col_modify = to_backend(MatrixMPI(copy(M_global)))
    new_col = T.(Float64.(100:100+n-1))
    M_col_modify[:, 3] = new_col
    result_gathered = Matrix(M_col_modify)
    for i in 1:n
        @test abs(result_gathered[i, 3] - new_col[i]) < TOL
        if 3 != 1
            @test abs(result_gathered[i, 1] - M_global[i, 1]) < TOL  # Unchanged column
        end
    end


    println(io0(), "[test] MatrixMPI submatrix setindex! ($T, $backend_name)")

    # Test MatrixMPI submatrix setindex!
    M_sub_modify = to_backend(MatrixMPI(copy(M_global)))
    new_sub = T.([Float64(1000 + i*10 + j) for i in 1:3, j in 1:3])
    M_sub_modify[2:4, 3:5] = new_sub
    result_sub = Matrix(M_sub_modify)
    for i in 1:3
        for j in 1:3
            @test abs(result_sub[i+1, j+2] - new_sub[i, j]) < TOL
        end
    end
    # Verify unchanged regions
    @test abs(result_sub[1, 1] - M_global[1, 1]) < TOL
    @test abs(result_sub[n, n] - M_global[n, n]) < TOL


    println(io0(), "[test] VectorMPI indexing with VectorMPI indices ($T, $backend_name)")

    # Test VectorMPI getindex with VectorMPI indices
    idx_global = [1, 3, 5, 7, 9]
    idx_mpi = to_backend(VectorMPI(idx_global))
    v_indexed = v[idx_mpi]
    v_indexed_gathered = Vector(v_indexed)
    for (k, global_idx) in enumerate(idx_global)
        @test abs(v_indexed_gathered[k] - v_global[global_idx]) < TOL
    end


    println(io0(), "[test] MatrixMPI indexing with VectorMPI row indices ($T, $backend_name)")

    # Test MatrixMPI getindex with VectorMPI row indices
    row_idx_global = [1, 3, 5]
    row_idx_mpi = to_backend(VectorMPI(row_idx_global))
    M_row_indexed = M[row_idx_mpi, 2:4]
    M_row_indexed_gathered = Matrix(M_row_indexed)
    for (ri, global_row) in enumerate(row_idx_global)
        for cj in 1:3
            @test abs(M_row_indexed_gathered[ri, cj] - M_global[global_row, cj+1]) < TOL
        end
    end


    println(io0(), "[test] SparseMatrixMPI row setindex! ($T, $backend_name)")

    # Test SparseMatrixMPI row setindex!
    A_row_modify = to_backend(SparseMatrixMPI{T}(copy(A_global)))
    new_row_vals = T.([99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 0.0, 0.0, 0.0, 0.0])
    A_row_modify[1, :] = new_row_vals[1:n]
    for j in 1:8
        @test abs(A_row_modify[1, j] - new_row_vals[j]) < TOL
    end


    println(io0(), "[test] VectorMPI arithmetic after indexing ($T, $backend_name)")

    # Test that indexed vectors work in arithmetic
    w1 = v[1:6]
    w2 = v[7:12]
    w1_data = Vector(w1)
    w2_data = Vector(w2)
    @test length(w1_data) == 6
    @test length(w2_data) == 6
    @test abs(sum(w1_data) - sum(v_global[1:6])) < TOL
    @test abs(sum(w2_data) - sum(v_global[7:12])) < TOL


    println(io0(), "[test] MatrixMPI operations after indexing ($T, $backend_name)")

    # Test that indexed matrices work in operations
    sub_M = M[1:6, 1:6]
    sub_v = to_backend(VectorMPI(T.(ones(6))))
    result = sub_M * sub_v
    result_gathered = Vector(result)
    expected = M_global[1:6, 1:6] * ones(T, 6)
    @test norm(result_gathered - expected) < TOL


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

println(io0(), "Test Summary: Indexing | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
