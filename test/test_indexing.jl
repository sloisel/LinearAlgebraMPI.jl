# Tests for indexing operations (getindex, setindex!)
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

const TOL = 1e-12

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Helper function to gather VectorMPI to root (all ranks must call)
function gather_to_root(v::VectorMPI{T}) where T
    local_data = v.v
    local_len = Int32(length(local_data))

    # Gather lengths from all ranks
    all_lens = MPI.Gather(local_len, 0, comm)

    if rank == 0
        # Allocate receive buffer
        total_len = sum(all_lens)
        recv_buf = Vector{T}(undef, total_len)
        counts = Vector{Int32}(all_lens)
        disps = Int32[0; cumsum(counts[1:end-1])]

        # Gather data
        MPI.Gatherv!(local_data, MPI.VBuffer(recv_buf, counts, disps), 0, comm)
        return recv_buf
    else
        MPI.Gatherv!(local_data, nothing, 0, comm)
        return T[]
    end
end

# Helper function to gather MatrixMPI to root
function gather_to_root(A::MatrixMPI{T}) where T
    m, ncols = size(A)
    # Flatten in row-major order for consistent reconstruction
    local_data = vec(permutedims(A.A))  # Row-major flattening
    local_len = Int32(length(local_data))

    all_lens = MPI.Gather(local_len, 0, comm)

    if rank == 0
        total_len = sum(all_lens)
        recv_buf = Vector{T}(undef, total_len)
        counts = Vector{Int32}(all_lens)
        disps = Int32[0; cumsum(counts[1:end-1])]

        MPI.Gatherv!(local_data, MPI.VBuffer(recv_buf, counts, disps), 0, comm)

        # Reconstruct matrix row by row
        result = Matrix{T}(undef, m, ncols)
        offset = 0
        for r in 0:nranks-1
            local_nrows = A.row_partition[r+2] - A.row_partition[r+1]
            for i in 1:local_nrows
                global_row = A.row_partition[r+1] + i - 1
                for j in 1:ncols
                    result[global_row, j] = recv_buf[offset + (i-1)*ncols + j]
                end
            end
            offset += counts[r+1]
        end
        return result
    else
        MPI.Gatherv!(local_data, nothing, 0, comm)
        return Matrix{T}(undef, 0, 0)
    end
end

# Helper function to gather SparseMatrixMPI to root (returns dense matrix)
function gather_to_root(A::SparseMatrixMPI{T}) where T
    m, ncols = size(A)
    # Convert local sparse to dense, send rows
    # Note: local_A has shape (length(col_indices), local_nrows) with compressed columns
    local_nrows = A.row_partition[rank+2] - A.row_partition[rank+1]
    local_dense = zeros(T, local_nrows, ncols)  # Start with zeros
    local_A = A.A.parent  # CSC with shape (length(col_indices), local_nrows)
    col_indices = A.col_indices  # Maps local col index -> global col index

    # Extract values using the sparse matrix interface
    for i in 1:local_nrows
        for local_j in 1:length(col_indices)
            global_j = col_indices[local_j]
            local_dense[i, global_j] = local_A[local_j, i]
        end
    end

    # Flatten in row-major order (transpose then vec, then transpose back conceptually)
    local_data = vec(permutedims(local_dense))  # Row-major flattening
    local_len = Int32(length(local_data))

    all_lens = MPI.Gather(local_len, 0, comm)

    if rank == 0
        total_len = sum(all_lens)
        recv_buf = Vector{T}(undef, total_len)
        counts = Vector{Int32}(all_lens)
        disps = Int32[0; cumsum(counts[1:end-1])]

        MPI.Gatherv!(local_data, MPI.VBuffer(recv_buf, counts, disps), 0, comm)

        result = Matrix{T}(undef, m, ncols)
        offset = 0
        for r in 0:nranks-1
            local_nr = A.row_partition[r+2] - A.row_partition[r+1]
            for i in 1:local_nr
                global_row = A.row_partition[r+1] + i - 1
                for j in 1:ncols
                    result[global_row, j] = recv_buf[offset + (i-1)*ncols + j]
                end
            end
            offset += counts[r+1]
        end
        return result
    else
        MPI.Gatherv!(local_data, nothing, 0, comm)
        return Matrix{T}(undef, 0, 0)
    end
end

# Create deterministic test data (same on all ranks)
n = 12  # Vector size, divisible by common rank counts

# VectorMPI test data
v_global = collect(1.0:Float64(n))
v = VectorMPI(v_global)

# Complex VectorMPI
v_complex_global = ComplexF64[k + (k * 0.5)im for k in 1:n]
v_complex = VectorMPI(v_complex_global)

# SparseMatrixMPI test data - create a matrix with known sparsity pattern
# Deterministic sparse matrix: diagonal + some off-diagonal entries
I_vals = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7]
J_vals = [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8]
V_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.1, 0.3, 0.5, 0.7]
A_global = sparse(I_vals, J_vals, V_vals, n, n)
A = SparseMatrixMPI{Float64}(A_global)

# Complex SparseMatrixMPI
V_complex = ComplexF64[v + v*0.5im for v in V_vals]
A_complex_global = sparse(I_vals, J_vals, V_complex, n, n)
A_complex = SparseMatrixMPI{ComplexF64}(A_complex_global)

# MatrixMPI test data - dense matrix
M_global = Float64[i + j * 0.1 for i in 1:n, j in 1:n]
M = MatrixMPI(M_global)

# Complex MatrixMPI
M_complex_global = ComplexF64[i + j * 0.1 + (i - j) * 0.05im for i in 1:n, j in 1:n]
M_complex = MatrixMPI(M_complex_global)

ts = @testset QuietTestSet "Indexing" begin

if rank == 0
    println("[test] VectorMPI getindex")
    flush(stdout)
end

# Test VectorMPI getindex - various indices
# All these should work regardless of which rank owns the element
for i in 1:n
    result = v[i]
    @test result ≈ v_global[i] atol=TOL
end

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI getindex (complex)")
    flush(stdout)
end

# Test complex VectorMPI getindex
for i in [1, div(n, 2), n]
    result = v_complex[i]
    @test result ≈ v_complex_global[i] atol=TOL
end

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI setindex!")
    flush(stdout)
end

# Test VectorMPI setindex! - create a fresh vector to modify
v_modify_global = collect(1.0:Float64(n))
v_modify = VectorMPI(v_modify_global)

# Set each element to a new value (all ranks participate)
for i in 1:n
    v_modify[i] = Float64(i * 10)
end

# Verify all values were set correctly
for i in 1:n
    result = v_modify[i]
    @test result ≈ Float64(i * 10) atol=TOL
end

MPI.Barrier(comm)

if rank == 0
    println("[test] SparseMatrixMPI getindex - existing entries")
    flush(stdout)
end

# Test SparseMatrixMPI getindex - existing entries (structural nonzeros)
for k in eachindex(I_vals)
    i, j = I_vals[k], J_vals[k]
    result = A[i, j]
    @test result ≈ A_global[i, j] atol=TOL
end

MPI.Barrier(comm)

if rank == 0
    println("[test] SparseMatrixMPI getindex - structural zeros")
    flush(stdout)
end

# Test SparseMatrixMPI getindex - structural zeros
# These positions have no entry in the sparsity pattern
test_zero_positions = [
    (1, 3), (1, 4), (2, 1), (2, 4),
    (3, 1), (4, 2), (5, 1), (6, 2)
]
for (i, j) in test_zero_positions
    result = A[i, j]
    @test result ≈ 0.0 atol=TOL
end

MPI.Barrier(comm)

if rank == 0
    println("[test] SparseMatrixMPI getindex (complex)")
    flush(stdout)
end

# Test complex SparseMatrixMPI getindex
for k in [1, 5, 9]
    i, j = I_vals[k], J_vals[k]
    result = A_complex[i, j]
    @test result ≈ A_complex_global[i, j] atol=TOL
end

MPI.Barrier(comm)

if rank == 0
    println("[test] SparseMatrixMPI setindex! - modify existing entries")
    flush(stdout)
end

# Test SparseMatrixMPI setindex! - modify existing entries
# Create a fresh matrix to modify
A_modify_global = sparse(I_vals, J_vals, copy(V_vals), n, n)
A_modify = SparseMatrixMPI{Float64}(A_modify_global)

# Modify each existing entry
for k in eachindex(I_vals)
    i, j = I_vals[k], J_vals[k]
    new_val = Float64(k * 100)
    A_modify[i, j] = new_val
end

# Verify all values were modified correctly
for k in eachindex(I_vals)
    i, j = I_vals[k], J_vals[k]
    result = A_modify[i, j]
    @test result ≈ Float64(k * 100) atol=TOL
end

MPI.Barrier(comm)

if rank == 0
    println("[test] SparseMatrixMPI setindex! (complex)")
    flush(stdout)
end

# Test complex SparseMatrixMPI setindex!
A_complex_modify_global = sparse(I_vals, J_vals, copy(V_complex), n, n)
A_complex_modify = SparseMatrixMPI{ComplexF64}(A_complex_modify_global)

# Modify a few entries with complex values
test_modify_indices = [1, 5, 9]
for k in test_modify_indices
    i, j = I_vals[k], J_vals[k]
    new_val = ComplexF64(k * 100, k * 50)
    A_complex_modify[i, j] = new_val
end

# Verify
for k in test_modify_indices
    i, j = I_vals[k], J_vals[k]
    result = A_complex_modify[i, j]
    expected = ComplexF64(k * 100, k * 50)
    @test result ≈ expected atol=TOL
end

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI getindex")
    flush(stdout)
end

# Test MatrixMPI getindex - various positions
for i in 1:n
    for j in [1, div(n, 2), n]
        result = M[i, j]
        @test result ≈ M_global[i, j] atol=TOL
    end
end

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI getindex (complex)")
    flush(stdout)
end

# Test complex MatrixMPI getindex
for i in [1, div(n, 2), n]
    for j in [1, div(n, 2), n]
        result = M_complex[i, j]
        @test result ≈ M_complex_global[i, j] atol=TOL
    end
end

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI setindex!")
    flush(stdout)
end

# Test MatrixMPI setindex!
M_modify_global = copy(M_global)
M_modify = MatrixMPI(M_modify_global)

# Modify several entries
test_positions = [(1, 1), (3, 5), (n, n), (div(n, 2), div(n, 2))]
for (i, j) in test_positions
    new_val = Float64(i * 100 + j)
    M_modify[i, j] = new_val
end

# Verify all modifications
for (i, j) in test_positions
    result = M_modify[i, j]
    expected = Float64(i * 100 + j)
    @test result ≈ expected atol=TOL
end

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI setindex! (complex)")
    flush(stdout)
end

# Test complex MatrixMPI setindex!
M_complex_modify_global = copy(M_complex_global)
M_complex_modify = MatrixMPI(M_complex_modify_global)

for (i, j) in test_positions
    new_val = ComplexF64(i * 100 + j, i - j)
    M_complex_modify[i, j] = new_val
end

for (i, j) in test_positions
    result = M_complex_modify[i, j]
    expected = ComplexF64(i * 100 + j, i - j)
    @test result ≈ expected atol=TOL
end

MPI.Barrier(comm)

if rank == 0
    println("[test] Edge cases - boundary indices")
    flush(stdout)
end

# Test edge cases: first and last elements
# VectorMPI
@test v[1] ≈ v_global[1] atol=TOL
@test v[n] ≈ v_global[n] atol=TOL

# SparseMatrixMPI - first and last rows/cols
@test A[1, 1] ≈ A_global[1, 1] atol=TOL
@test A[8, 8] ≈ A_global[8, 8] atol=TOL  # Last nonzero diagonal entry

# MatrixMPI - corners
@test M[1, 1] ≈ M_global[1, 1] atol=TOL
@test M[1, n] ≈ M_global[1, n] atol=TOL
@test M[n, 1] ≈ M_global[n, 1] atol=TOL
@test M[n, n] ≈ M_global[n, n] atol=TOL

MPI.Barrier(comm)

# ============================================================================
# Range Indexing Tests
# ============================================================================

if rank == 0
    println("[test] VectorMPI range getindex")
    flush(stdout)
end

# Test VectorMPI range getindex - extract subvector
for (rng_start, rng_end) in [(1, 4), (3, 7), (5, n), (1, n)]
    rng = rng_start:rng_end
    w = v[rng]
    @test length(w) == length(rng)
    # Gather and verify the result
    for (local_idx, global_idx) in enumerate(rng)
        @test w[local_idx] ≈ v_global[global_idx] atol=TOL
    end
end

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI range setindex! (scalar)")
    flush(stdout)
end

# Test VectorMPI range setindex! with scalar
v_range_modify = VectorMPI(copy(v_global))
v_range_modify[3:6] = 99.0
for i in 1:n
    if 3 <= i <= 6
        @test v_range_modify[i] ≈ 99.0 atol=TOL
    else
        @test v_range_modify[i] ≈ v_global[i] atol=TOL
    end
end

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI range setindex! (vector)")
    flush(stdout)
end

# Test VectorMPI range setindex! with a regular vector
v_range_modify2 = VectorMPI(copy(v_global))
v_range_modify2[3:6] = [100.0, 200.0, 300.0, 400.0]
for i in 1:n
    if 3 <= i <= 6
        @test v_range_modify2[i] ≈ (i - 2) * 100.0 atol=TOL
    else
        @test v_range_modify2[i] ≈ v_global[i] atol=TOL
    end
end

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI range getindex")
    flush(stdout)
end

# Test MatrixMPI range getindex
row_rng = 2:6
col_rng = 3:8
M_sub = M[row_rng, col_rng]
@test size(M_sub) == (length(row_rng), length(col_rng))
for i in 1:length(row_rng)
    for j in 1:length(col_rng)
        @test M_sub[i, j] ≈ M_global[row_rng[i], col_rng[j]] atol=TOL
    end
end

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI range getindex with Colon")
    flush(stdout)
end

# Test MatrixMPI range getindex with Colon
M_rows = M[2:5, :]
@test size(M_rows) == (4, n)
for i in 1:4
    for j in 1:n
        @test M_rows[i, j] ≈ M_global[i+1, j] atol=TOL
    end
end

M_cols = M[:, 3:7]
@test size(M_cols) == (n, 5)
for i in 1:n
    for j in 1:5
        @test M_cols[i, j] ≈ M_global[i, j+2] atol=TOL
    end
end

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI range setindex! (scalar)")
    flush(stdout)
end

# Test MatrixMPI range setindex! with scalar
M_range_modify = MatrixMPI(copy(M_global))
M_range_modify[2:4, 3:5] = 0.0
for i in 1:n
    for j in 1:n
        if 2 <= i <= 4 && 3 <= j <= 5
            @test M_range_modify[i, j] ≈ 0.0 atol=TOL
        else
            @test M_range_modify[i, j] ≈ M_global[i, j] atol=TOL
        end
    end
end

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI range setindex! (matrix)")
    flush(stdout)
end

# Test MatrixMPI range setindex! with a matrix
M_range_modify2 = MatrixMPI(copy(M_global))
new_vals = Float64[10*i + j for i in 1:3, j in 1:4]
M_range_modify2[2:4, 3:6] = new_vals
for i in 2:4
    for j in 3:6
        @test M_range_modify2[i, j] ≈ 10*(i-1) + (j-2) atol=TOL
    end
end

MPI.Barrier(comm)

if rank == 0
    println("[test] SparseMatrixMPI range getindex")
    flush(stdout)
end

# Test SparseMatrixMPI range getindex
row_rng = 2:6
col_rng = 2:7
A_sub = A[row_rng, col_rng]
@test size(A_sub) == (length(row_rng), length(col_rng))

# Verify structural nonzeros are correctly extracted
# Original matrix has entries at (2,2), (3,3), (3,4), (4,4), (5,5), (5,6), (6,6)
# In the submatrix (rows 2:6, cols 2:7), these become:
# (1,1)=2.0, (2,2)=3.0, (2,3)=0.3, (3,3)=4.0, (4,4)=5.0, (4,5)=0.5, (5,5)=6.0
@test A_sub[1, 1] ≈ A_global[2, 2] atol=TOL  # Diagonal entry
@test A_sub[2, 2] ≈ A_global[3, 3] atol=TOL  # Diagonal entry
@test A_sub[3, 3] ≈ A_global[4, 4] atol=TOL  # Diagonal entry

MPI.Barrier(comm)

if rank == 0
    println("[test] SparseMatrixMPI range getindex with Colon")
    flush(stdout)
end

# Test with Colon
A_rows = A[2:5, :]
@test size(A_rows) == (4, n)
A_cols = A[:, 3:8]
@test size(A_cols) == (n, 6)

MPI.Barrier(comm)

if rank == 0
    println("[test] SparseMatrixMPI range setindex! (scalar)")
    flush(stdout)
end

# Test SparseMatrixMPI range setindex! with scalar
# Create a fresh copy for modification
A_modify = SparseMatrixMPI{Float64}(copy(A_global))
# Set all entries in rows 2:4, cols 2:5 to 0.0
A_modify[2:4, 2:5] = 0.0

# Original: A_global[2,2]=2.0, A_global[3,3]=3.0, A_global[3,4]=0.3, A_global[4,4]=4.0
# After: these should all be 0.0
@test A_modify[2, 2] ≈ 0.0 atol=TOL
@test A_modify[3, 3] ≈ 0.0 atol=TOL
@test A_modify[3, 4] ≈ 0.0 atol=TOL
@test A_modify[4, 4] ≈ 0.0 atol=TOL

# Entries outside the range should be unchanged
@test A_modify[1, 1] ≈ A_global[1, 1] atol=TOL
@test A_modify[5, 5] ≈ A_global[5, 5] atol=TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI range getindex preserves partition structure")
    flush(stdout)
end

# Verify that extracting a range and then extracting the same range again gives same partition
w1 = v[3:8]
w2 = v[3:8]
@test w1.partition == w2.partition
@test w1.structural_hash == w2.structural_hash

MPI.Barrier(comm)

if rank == 0
    println("[test] Complex VectorMPI range getindex")
    flush(stdout)
end

# Test complex VectorMPI range getindex
w_complex = v_complex[2:6]
@test length(w_complex) == 5
for (local_idx, global_idx) in enumerate(2:6)
    @test w_complex[local_idx] ≈ v_complex_global[global_idx] atol=TOL
end

MPI.Barrier(comm)

# ============================================================================
# VectorMPI indexing with VectorMPI indices
# ============================================================================

if rank == 0
    println("[test] VectorMPI getindex with VectorMPI indices")
    flush(stdout)
end

# Test v[idx] where idx is VectorMPI{Int}
idx_global = [3, 1, 5, 2, 6, 4]
idx = VectorMPI(idx_global)
result = v[idx]

# Result should have same partition as idx
@test result.partition == idx.partition
@test length(result) == length(idx)

# Check values - use gather to compare on all ranks without collective getindex
result_gathered = gather_to_root(result)
if rank == 0
    for k in 1:length(idx_global)
        @test result_gathered[k] ≈ v_global[idx_global[k]] atol=TOL
    end
end

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI setindex! with VectorMPI indices")
    flush(stdout)
end

# Test v[idx] = src where idx and src are VectorMPI
v_modify = VectorMPI(copy(v_global))
idx_set = VectorMPI([2, 4, 6])
src_values = VectorMPI([20.0, 40.0, 60.0])
v_modify[idx_set] = src_values

# Gather to check values without collective single-element getindex
v_modify_gathered = gather_to_root(v_modify)
if rank == 0
    @test v_modify_gathered[2] ≈ 20.0 atol=TOL
    @test v_modify_gathered[4] ≈ 40.0 atol=TOL
    @test v_modify_gathered[6] ≈ 60.0 atol=TOL
    # Unchanged values
    @test v_modify_gathered[1] ≈ v_global[1] atol=TOL
    @test v_modify_gathered[3] ≈ v_global[3] atol=TOL
    @test v_modify_gathered[5] ≈ v_global[5] atol=TOL
end

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI getindex with VectorMPI indices")
    flush(stdout)
end

# Test A[row_idx, col_idx] for MatrixMPI
A_dense_global = Float64[i + j/10 for i in 1:6, j in 1:4]
A_dense = MatrixMPI(A_dense_global)

row_idx = VectorMPI([2, 5, 1, 4])
col_idx = VectorMPI([3, 1])

result_dense = A_dense[row_idx, col_idx]
@test size(result_dense) == (4, 2)

# Check values using gather: result[i,j] = A[row_idx[i], col_idx[j]]
row_idx_global = [2, 5, 1, 4]
col_idx_global = [3, 1]
result_dense_gathered = gather_to_root(result_dense)
if rank == 0
    for i in 1:4
        for j in 1:2
            @test result_dense_gathered[i, j] ≈ A_dense_global[row_idx_global[i], col_idx_global[j]] atol=TOL
        end
    end
end

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI setindex! with VectorMPI indices")
    flush(stdout)
end

# Test A[row_idx, col_idx] = src for MatrixMPI
A_dense_modify = MatrixMPI(zeros(6, 4))
row_idx_set = VectorMPI([1, 3, 5])
col_idx_set = VectorMPI([2, 4])
src_dense = MatrixMPI(ones(3, 2) * 7.0)

A_dense_modify[row_idx_set, col_idx_set] = src_dense

# Check values using gather
A_dense_modify_gathered = gather_to_root(A_dense_modify)
if rank == 0
    # Check that values were set
    @test A_dense_modify_gathered[1, 2] ≈ 7.0 atol=TOL
    @test A_dense_modify_gathered[1, 4] ≈ 7.0 atol=TOL
    @test A_dense_modify_gathered[3, 2] ≈ 7.0 atol=TOL
    @test A_dense_modify_gathered[3, 4] ≈ 7.0 atol=TOL
    @test A_dense_modify_gathered[5, 2] ≈ 7.0 atol=TOL
    @test A_dense_modify_gathered[5, 4] ≈ 7.0 atol=TOL

    # Check that other values are still zero
    @test A_dense_modify_gathered[1, 1] ≈ 0.0 atol=TOL
    @test A_dense_modify_gathered[2, 2] ≈ 0.0 atol=TOL
end

MPI.Barrier(comm)

if rank == 0
    println("[test] SparseMatrixMPI getindex with VectorMPI indices")
    flush(stdout)
end

# Test A[row_idx, col_idx] for SparseMatrixMPI (returns dense MatrixMPI)
A_sparse_test = SparseMatrixMPI{Float64}(A_global)
row_idx_sparse = VectorMPI([2, 4, 1])
col_idx_sparse = VectorMPI([1, 3, 5])

result_sparse = A_sparse_test[row_idx_sparse, col_idx_sparse]
@test result_sparse isa MatrixMPI
@test size(result_sparse) == (3, 3)

# Check values using gather
row_idx_sparse_global = [2, 4, 1]
col_idx_sparse_global = [1, 3, 5]
result_sparse_gathered = gather_to_root(result_sparse)
if rank == 0
    for i in 1:3
        for j in 1:3
            @test result_sparse_gathered[i, j] ≈ A_global[row_idx_sparse_global[i], col_idx_sparse_global[j]] atol=TOL
        end
    end
end

MPI.Barrier(comm)

if rank == 0
    println("[test] SparseMatrixMPI setindex! with VectorMPI indices")
    flush(stdout)
end

# Test A[row_idx, col_idx] = src for SparseMatrixMPI
A_sparse_modify = SparseMatrixMPI{Float64}(spzeros(6, 6))
row_idx_sparse_set = VectorMPI([1, 3, 5])
col_idx_sparse_set = VectorMPI([2, 4])
src_sparse = MatrixMPI(ones(3, 2) * 9.0)

A_sparse_modify[row_idx_sparse_set, col_idx_sparse_set] = src_sparse

# Check values using gather (convert to dense first)
A_sparse_modify_dense = gather_to_root(A_sparse_modify)
if rank == 0
    # Check that values were set (structural modification)
    @test A_sparse_modify_dense[1, 2] ≈ 9.0 atol=TOL
    @test A_sparse_modify_dense[1, 4] ≈ 9.0 atol=TOL
    @test A_sparse_modify_dense[3, 2] ≈ 9.0 atol=TOL
    @test A_sparse_modify_dense[3, 4] ≈ 9.0 atol=TOL
    @test A_sparse_modify_dense[5, 2] ≈ 9.0 atol=TOL
    @test A_sparse_modify_dense[5, 4] ≈ 9.0 atol=TOL

    # Check that other values are still zero
    @test A_sparse_modify_dense[1, 1] ≈ 0.0 atol=TOL
    @test A_sparse_modify_dense[2, 2] ≈ 0.0 atol=TOL
end

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI indexing with VectorMPI indices (complex)")
    flush(stdout)
end

# Test with ComplexF64
v_complex_modify = VectorMPI(copy(v_complex_global))
idx_complex = VectorMPI([1, 3, 5])
src_complex = VectorMPI([100.0 + 200.0im, 300.0 + 400.0im, 500.0 + 600.0im])
v_complex_modify[idx_complex] = src_complex

# Use gather to check values (single-element indexing is collective)
v_complex_modify_gathered = gather_to_root(v_complex_modify)
if rank == 0
    @test v_complex_modify_gathered[1] ≈ 100.0 + 200.0im atol=TOL
    @test v_complex_modify_gathered[3] ≈ 300.0 + 400.0im atol=TOL
    @test v_complex_modify_gathered[5] ≈ 500.0 + 600.0im atol=TOL
    @test v_complex_modify_gathered[2] ≈ v_complex_global[2] atol=TOL
end

MPI.Barrier(comm)

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

if rank == 0
    println("Test Summary: Indexing | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")
    flush(stdout)
end

MPI.Barrier(comm)
MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
