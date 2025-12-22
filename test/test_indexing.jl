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
    # Note: CSR has shape (nrows_local, ncols_compressed) with compressed columns
    local_nrows = A.row_partition[rank+2] - A.row_partition[rank+1]
    local_dense = zeros(T, local_nrows, ncols)  # Start with zeros
    col_indices = A.col_indices  # Maps local col index -> global col index

    # Extract values using the CSR arrays (rowptr, colval, nzval)
    for i in 1:local_nrows
        for idx in A.rowptr[i]:(A.rowptr[i+1]-1)
            local_j = A.colval[idx]
            global_j = col_indices[local_j]
            local_dense[i, global_j] = A.nzval[idx]
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

println(io0(), "[test] VectorMPI getindex")

# Test VectorMPI getindex - various indices
# All these should work regardless of which rank owns the element
for i in 1:n
    result = v[i]
    @test result ≈ v_global[i] atol=TOL
end


println(io0(), "[test] VectorMPI getindex (complex)")

# Test complex VectorMPI getindex
for i in [1, div(n, 2), n]
    result = v_complex[i]
    @test result ≈ v_complex_global[i] atol=TOL
end


println(io0(), "[test] VectorMPI setindex!")

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


println(io0(), "[test] SparseMatrixMPI getindex - existing entries")

# Test SparseMatrixMPI getindex - existing entries (structural nonzeros)
for k in eachindex(I_vals)
    i, j = I_vals[k], J_vals[k]
    result = A[i, j]
    @test result ≈ A_global[i, j] atol=TOL
end


println(io0(), "[test] SparseMatrixMPI getindex - structural zeros")

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


println(io0(), "[test] SparseMatrixMPI getindex (complex)")

# Test complex SparseMatrixMPI getindex
for k in [1, 5, 9]
    i, j = I_vals[k], J_vals[k]
    result = A_complex[i, j]
    @test result ≈ A_complex_global[i, j] atol=TOL
end


println(io0(), "[test] SparseMatrixMPI setindex! - modify existing entries")

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


println(io0(), "[test] SparseMatrixMPI setindex! (complex)")

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


println(io0(), "[test] MatrixMPI getindex")

# Test MatrixMPI getindex - various positions
for i in 1:n
    for j in [1, div(n, 2), n]
        result = M[i, j]
        @test result ≈ M_global[i, j] atol=TOL
    end
end


println(io0(), "[test] MatrixMPI getindex (complex)")

# Test complex MatrixMPI getindex
for i in [1, div(n, 2), n]
    for j in [1, div(n, 2), n]
        result = M_complex[i, j]
        @test result ≈ M_complex_global[i, j] atol=TOL
    end
end


println(io0(), "[test] MatrixMPI setindex!")

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


println(io0(), "[test] MatrixMPI setindex! (complex)")

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


println(io0(), "[test] Edge cases - boundary indices")

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


# ============================================================================
# Range Indexing Tests
# ============================================================================

println(io0(), "[test] VectorMPI range getindex")

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


println(io0(), "[test] VectorMPI range setindex! (scalar)")

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


println(io0(), "[test] VectorMPI range setindex! (vector)")

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


println(io0(), "[test] MatrixMPI range getindex")

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


println(io0(), "[test] MatrixMPI range getindex with Colon")

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


println(io0(), "[test] MatrixMPI range setindex! (scalar)")

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


println(io0(), "[test] MatrixMPI range setindex! (matrix)")

# Test MatrixMPI range setindex! with a matrix
M_range_modify2 = MatrixMPI(copy(M_global))
new_vals = Float64[10*i + j for i in 1:3, j in 1:4]
M_range_modify2[2:4, 3:6] = new_vals
for i in 2:4
    for j in 3:6
        @test M_range_modify2[i, j] ≈ 10*(i-1) + (j-2) atol=TOL
    end
end


println(io0(), "[test] SparseMatrixMPI range getindex")

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


println(io0(), "[test] SparseMatrixMPI range getindex with Colon")

# Test with Colon
A_rows = A[2:5, :]
@test size(A_rows) == (4, n)
A_cols = A[:, 3:8]
@test size(A_cols) == (n, 6)


println(io0(), "[test] SparseMatrixMPI range setindex! (scalar)")

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


println(io0(), "[test] VectorMPI range getindex preserves partition structure")

# Verify that extracting a range and then extracting the same range again gives same partition
w1 = v[3:8]
w2 = v[3:8]
@test w1.partition == w2.partition
@test w1.structural_hash == w2.structural_hash


println(io0(), "[test] Complex VectorMPI range getindex")

# Test complex VectorMPI range getindex
w_complex = v_complex[2:6]
@test length(w_complex) == 5
for (local_idx, global_idx) in enumerate(2:6)
    @test w_complex[local_idx] ≈ v_complex_global[global_idx] atol=TOL
end


# ============================================================================
# VectorMPI indexing with VectorMPI indices
# ============================================================================

println(io0(), "[test] VectorMPI getindex with VectorMPI indices")

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


println(io0(), "[test] VectorMPI setindex! with VectorMPI indices")

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


println(io0(), "[test] MatrixMPI getindex with VectorMPI indices")

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


println(io0(), "[test] MatrixMPI setindex! with VectorMPI indices")

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


println(io0(), "[test] SparseMatrixMPI getindex with VectorMPI indices")

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


println(io0(), "[test] SparseMatrixMPI setindex! with VectorMPI indices")

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


println(io0(), "[test] VectorMPI indexing with VectorMPI indices (complex)")

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


# ============================================================================
# Cross-rank communication tests
# These tests use indices that span multiple MPI ranks to exercise
# point-to-point communication paths
# ============================================================================

println(io0(), "[test] Cross-rank VectorMPI getindex with VectorMPI indices")

# Create a larger vector to ensure indices span multiple ranks
n_large = 100
v_large_global = Float64[i * 1.5 for i in 1:n_large]
v_large = VectorMPI(v_large_global)

# Create indices that definitely span multiple ranks
# Use indices from beginning, middle and end to ensure cross-rank access
idx_crossrank_global = vcat(1:5, div(n_large,2)-2:div(n_large,2)+2, n_large-4:n_large)
idx_crossrank = VectorMPI(idx_crossrank_global)
result_crossrank = v_large[idx_crossrank]

@test length(result_crossrank) == length(idx_crossrank_global)
result_gathered = gather_to_root(result_crossrank)
if rank == 0
    for k in 1:length(idx_crossrank_global)
        @test result_gathered[k] ≈ v_large_global[idx_crossrank_global[k]] atol=TOL
    end
end


println(io0(), "[test] Cross-rank VectorMPI setindex! with VectorMPI indices")

# Test setindex! with cross-rank indices
v_large_modify = VectorMPI(copy(v_large_global))
idx_set_crossrank = VectorMPI(vcat(1:3, n_large-2:n_large))
src_set_crossrank = VectorMPI(Float64[1000.0 + i for i in 1:6])
v_large_modify[idx_set_crossrank] = src_set_crossrank

v_large_gathered = gather_to_root(v_large_modify)
if rank == 0
    # Check modified values
    @test v_large_gathered[1] ≈ 1001.0 atol=TOL
    @test v_large_gathered[2] ≈ 1002.0 atol=TOL
    @test v_large_gathered[3] ≈ 1003.0 atol=TOL
    @test v_large_gathered[n_large-2] ≈ 1004.0 atol=TOL
    @test v_large_gathered[n_large-1] ≈ 1005.0 atol=TOL
    @test v_large_gathered[n_large] ≈ 1006.0 atol=TOL
    # Check unmodified values
    @test v_large_gathered[4] ≈ v_large_global[4] atol=TOL
    @test v_large_gathered[50] ≈ v_large_global[50] atol=TOL
end


println(io0(), "[test] Cross-rank MatrixMPI getindex with VectorMPI indices")

# Create larger dense matrix
M_large_global = Float64[i + j/100 for i in 1:40, j in 1:10]
M_large = MatrixMPI(M_large_global)

# Cross-rank row indices
row_idx_large = VectorMPI(vcat(1:3, 20:22, 38:40))
col_idx_large = VectorMPI([1, 5, 10])

result_M_large = M_large[row_idx_large, col_idx_large]
@test size(result_M_large) == (9, 3)

result_M_gathered = gather_to_root(result_M_large)
row_idx_arr = vcat(1:3, 20:22, 38:40)
col_idx_arr = [1, 5, 10]
if rank == 0
    for i in 1:9
        for j in 1:3
            @test result_M_gathered[i, j] ≈ M_large_global[row_idx_arr[i], col_idx_arr[j]] atol=TOL
        end
    end
end


println(io0(), "[test] Cross-rank MatrixMPI setindex! with VectorMPI indices")

# Test cross-rank setindex!
M_large_modify = MatrixMPI(zeros(40, 10))
row_idx_set_large = VectorMPI(vcat(1:2, 39:40))
col_idx_set_large = VectorMPI([2, 8])
src_M_large = MatrixMPI(ones(4, 2) * 77.0)

M_large_modify[row_idx_set_large, col_idx_set_large] = src_M_large

M_large_modify_gathered = gather_to_root(M_large_modify)
if rank == 0
    @test M_large_modify_gathered[1, 2] ≈ 77.0 atol=TOL
    @test M_large_modify_gathered[1, 8] ≈ 77.0 atol=TOL
    @test M_large_modify_gathered[2, 2] ≈ 77.0 atol=TOL
    @test M_large_modify_gathered[39, 2] ≈ 77.0 atol=TOL
    @test M_large_modify_gathered[40, 8] ≈ 77.0 atol=TOL
    @test M_large_modify_gathered[20, 5] ≈ 0.0 atol=TOL  # Unchanged
end


println(io0(), "[test] Cross-rank SparseMatrixMPI getindex with VectorMPI indices")

# Create larger sparse matrix with cross-rank structure
I_large = vcat(1:40, 1:20)  # Diagonal + some off-diagonals
J_large = vcat(1:40, 21:40)
V_large = Float64[i + j/100 for (i, j) in zip(I_large, J_large)]
A_large_global = sparse(I_large, J_large, V_large, 40, 40)
A_large = SparseMatrixMPI{Float64}(A_large_global)

row_idx_sparse_large = VectorMPI(vcat(1:3, 38:40))
col_idx_sparse_large = VectorMPI([1, 20, 40])

result_A_large = A_large[row_idx_sparse_large, col_idx_sparse_large]
@test size(result_A_large) == (6, 3)

result_A_gathered = gather_to_root(result_A_large)
row_arr = vcat(1:3, 38:40)
col_arr = [1, 20, 40]
if rank == 0
    for i in 1:6
        for j in 1:3
            @test result_A_gathered[i, j] ≈ A_large_global[row_arr[i], col_arr[j]] atol=TOL
        end
    end
end


println(io0(), "[test] Cross-rank SparseMatrixMPI setindex! with VectorMPI indices")

# Test cross-rank setindex! for SparseMatrixMPI (structural modification)
A_large_modify = SparseMatrixMPI{Float64}(spzeros(40, 40))
row_idx_sparse_set = VectorMPI(vcat(1:2, 39:40))
col_idx_sparse_set = VectorMPI([1, 40])
src_A_large = MatrixMPI(ones(4, 2) * 88.0)

A_large_modify[row_idx_sparse_set, col_idx_sparse_set] = src_A_large

A_large_modify_gathered = gather_to_root(A_large_modify)
if rank == 0
    @test A_large_modify_gathered[1, 1] ≈ 88.0 atol=TOL
    @test A_large_modify_gathered[1, 40] ≈ 88.0 atol=TOL
    @test A_large_modify_gathered[2, 1] ≈ 88.0 atol=TOL
    @test A_large_modify_gathered[39, 1] ≈ 88.0 atol=TOL
    @test A_large_modify_gathered[40, 40] ≈ 88.0 atol=TOL
    @test A_large_modify_gathered[20, 20] ≈ 0.0 atol=TOL  # Unchanged
end


# ============================================================================
# Structural modification tests (SparseMatrixMPI insert new nonzeros)
# ============================================================================

println(io0(), "[test] SparseMatrixMPI structural modification - single element insert")

# Create sparse matrix and insert at a position that has no structural nonzero
I_struct = [1, 2, 3, 4]
J_struct = [1, 2, 3, 4]
V_struct = [1.0, 2.0, 3.0, 4.0]
A_struct_global = sparse(I_struct, J_struct, V_struct, 8, 8)
A_struct = SparseMatrixMPI{Float64}(A_struct_global)

# Insert at (1, 5) which is a structural zero
A_struct[1, 5] = 99.0

# Verify the insertion
@test A_struct[1, 5] ≈ 99.0 atol=TOL
# Original values should be unchanged
@test A_struct[1, 1] ≈ 1.0 atol=TOL
@test A_struct[2, 2] ≈ 2.0 atol=TOL


# ============================================================================
# Cached transpose invalidation tests
# ============================================================================

println(io0(), "[test] Cached transpose bidirectional invalidation")

# Create a matrix and compute its transpose to create cached_transpose
I_cache = [1, 2, 3, 4, 1]
J_cache = [1, 2, 3, 4, 3]
V_cache = [1.0, 2.0, 3.0, 4.0, 0.5]
A_cache_global = sparse(I_cache, J_cache, V_cache, 6, 6)
A_cache = SparseMatrixMPI{Float64}(A_cache_global)

# Compute transpose by multiplying with identity - this triggers transpose materialization
# Use transpose(A) * B to force materialization and cache creation
# Note: sparse(I(6)) returns Bool, need Float64
identity_global = sparse(Float64.(I(6)))
B_cache = SparseMatrixMPI{Float64}(identity_global)
C_cache = transpose(A_cache) * B_cache
# After this, A_cache.cached_transpose should be populated

# Verify cache was created
@test A_cache.cached_transpose !== nothing
AT_cache = A_cache.cached_transpose
@test AT_cache.cached_transpose !== nothing
@test AT_cache.cached_transpose === A_cache  # Bidirectional link

# Now do a structural modification on A - this should invalidate both caches
# (non-structural modifications don't invalidate cache, only structural ones do)
A_cache[2, 5] = 99.0  # Insert at a position that's not in sparsity pattern

# A_cache's cached transpose should now be nothing
@test A_cache.cached_transpose === nothing
# AT_cache's cached_transpose should also have been invalidated through the bidirectional link
@test AT_cache.cached_transpose === nothing


# ============================================================================
# Mixed indexing tests: VectorMPI + range, VectorMPI + Colon, VectorMPI + Int
# ============================================================================

println(io0(), "[test] MatrixMPI getindex with VectorMPI rows and range columns")

row_idx_mix = VectorMPI([2, 5, 8])
M_mix = M[row_idx_mix, 3:7]
@test size(M_mix) == (3, 5)

M_mix_gathered = gather_to_root(M_mix)
if rank == 0
    for (local_i, global_i) in enumerate([2, 5, 8])
        for (local_j, global_j) in enumerate(3:7)
            @test M_mix_gathered[local_i, local_j] ≈ M_global[global_i, global_j] atol=TOL
        end
    end
end


println(io0(), "[test] MatrixMPI getindex with range rows and VectorMPI columns")

col_idx_mix = VectorMPI([1, 4, 7, 10])
M_mix2 = M[2:5, col_idx_mix]
@test size(M_mix2) == (4, 4)

M_mix2_gathered = gather_to_root(M_mix2)
if rank == 0
    for (local_i, global_i) in enumerate(2:5)
        for (local_j, global_j) in enumerate([1, 4, 7, 10])
            @test M_mix2_gathered[local_i, local_j] ≈ M_global[global_i, global_j] atol=TOL
        end
    end
end


println(io0(), "[test] MatrixMPI getindex with VectorMPI rows and Colon columns")

row_idx_colon = VectorMPI([1, 6, n])
M_colon = M[row_idx_colon, :]
@test size(M_colon) == (3, n)

M_colon_gathered = gather_to_root(M_colon)
if rank == 0
    for (local_i, global_i) in enumerate([1, 6, n])
        for j in 1:n
            @test M_colon_gathered[local_i, j] ≈ M_global[global_i, j] atol=TOL
        end
    end
end


println(io0(), "[test] MatrixMPI getindex with Colon rows and VectorMPI columns")

col_idx_colon = VectorMPI([2, 5, 8, 11])
M_colon2 = M[:, col_idx_colon]
@test size(M_colon2) == (n, 4)

M_colon2_gathered = gather_to_root(M_colon2)
if rank == 0
    for i in 1:n
        for (local_j, global_j) in enumerate([2, 5, 8, 11])
            @test M_colon2_gathered[i, local_j] ≈ M_global[i, global_j] atol=TOL
        end
    end
end


println(io0(), "[test] MatrixMPI getindex with VectorMPI rows and Int column")

row_idx_int = VectorMPI([1, 4, 7, 10])
M_int_col = M[row_idx_int, 5]
@test M_int_col isa VectorMPI
@test length(M_int_col) == 4

M_int_col_gathered = gather_to_root(M_int_col)
if rank == 0
    for (local_i, global_i) in enumerate([1, 4, 7, 10])
        @test M_int_col_gathered[local_i] ≈ M_global[global_i, 5] atol=TOL
    end
end


println(io0(), "[test] MatrixMPI getindex with Int row and VectorMPI columns")

col_idx_int = VectorMPI([2, 4, 6, 8])
M_int_row = M[3, col_idx_int]
@test M_int_row isa VectorMPI
@test length(M_int_row) == 4

M_int_row_gathered = gather_to_root(M_int_row)
if rank == 0
    for (local_j, global_j) in enumerate([2, 4, 6, 8])
        @test M_int_row_gathered[local_j] ≈ M_global[3, global_j] atol=TOL
    end
end


println(io0(), "[test] SparseMatrixMPI getindex with VectorMPI rows and range columns")

row_idx_sp_mix = VectorMPI([1, 3, 5, 7])
A_mix = A[row_idx_sp_mix, 2:6]
@test size(A_mix) == (4, 5)

A_mix_gathered = gather_to_root(A_mix)
if rank == 0
    for (local_i, global_i) in enumerate([1, 3, 5, 7])
        for (local_j, global_j) in enumerate(2:6)
            @test A_mix_gathered[local_i, local_j] ≈ A_global[global_i, global_j] atol=TOL
        end
    end
end


println(io0(), "[test] SparseMatrixMPI getindex with range rows and VectorMPI columns")

col_idx_sp_mix = VectorMPI([1, 3, 5, 7])
A_mix2 = A[2:5, col_idx_sp_mix]
@test size(A_mix2) == (4, 4)

A_mix2_gathered = gather_to_root(A_mix2)
if rank == 0
    for (local_i, global_i) in enumerate(2:5)
        for (local_j, global_j) in enumerate([1, 3, 5, 7])
            @test A_mix2_gathered[local_i, local_j] ≈ A_global[global_i, global_j] atol=TOL
        end
    end
end


println(io0(), "[test] SparseMatrixMPI getindex with VectorMPI rows and Colon columns")

row_idx_sp_colon = VectorMPI([2, 4, 6, 8])
A_colon = A[row_idx_sp_colon, :]
@test size(A_colon) == (4, n)

A_colon_gathered = gather_to_root(A_colon)
if rank == 0
    for (local_i, global_i) in enumerate([2, 4, 6, 8])
        for j in 1:n
            @test A_colon_gathered[local_i, j] ≈ A_global[global_i, j] atol=TOL
        end
    end
end


println(io0(), "[test] SparseMatrixMPI getindex with Colon rows and VectorMPI columns")

col_idx_sp_colon = VectorMPI([1, 4, 8])
A_colon2 = A[:, col_idx_sp_colon]
@test size(A_colon2) == (n, 3)

A_colon2_gathered = gather_to_root(A_colon2)
if rank == 0
    for i in 1:n
        for (local_j, global_j) in enumerate([1, 4, 8])
            @test A_colon2_gathered[i, local_j] ≈ A_global[i, global_j] atol=TOL
        end
    end
end


println(io0(), "[test] SparseMatrixMPI getindex with VectorMPI rows and Int column")

row_idx_sp_int = VectorMPI([1, 3, 5, 7])
A_int_col = A[row_idx_sp_int, 3]
@test A_int_col isa VectorMPI
@test length(A_int_col) == 4

A_int_col_gathered = gather_to_root(A_int_col)
if rank == 0
    for (local_i, global_i) in enumerate([1, 3, 5, 7])
        @test A_int_col_gathered[local_i] ≈ A_global[global_i, 3] atol=TOL
    end
end


println(io0(), "[test] SparseMatrixMPI getindex with Int row and VectorMPI columns")

col_idx_sp_int = VectorMPI([1, 2, 3, 4])
A_int_row = A[2, col_idx_sp_int]
@test A_int_row isa VectorMPI
@test length(A_int_row) == 4

A_int_row_gathered = gather_to_root(A_int_row)
if rank == 0
    for (local_j, global_j) in enumerate([1, 2, 3, 4])
        @test A_int_row_gathered[local_j] ≈ A_global[2, global_j] atol=TOL
    end
end


# ============================================================================
# Mixed setindex! tests
# ============================================================================

println(io0(), "[test] MatrixMPI setindex! with VectorMPI rows and range columns")

M_setmix = MatrixMPI(zeros(n, n))
row_idx_setmix = VectorMPI([1, 4, 7])
M_setmix[row_idx_setmix, 2:5] = MatrixMPI(ones(3, 4) * 55.0)

M_setmix_gathered = gather_to_root(M_setmix)
if rank == 0
    @test M_setmix_gathered[1, 2] ≈ 55.0 atol=TOL
    @test M_setmix_gathered[4, 3] ≈ 55.0 atol=TOL
    @test M_setmix_gathered[7, 5] ≈ 55.0 atol=TOL
    @test M_setmix_gathered[2, 2] ≈ 0.0 atol=TOL  # Unchanged
end


println(io0(), "[test] MatrixMPI setindex! with range rows and VectorMPI columns")

M_setmix2 = MatrixMPI(zeros(n, n))
col_idx_setmix = VectorMPI([1, 5, 9])
M_setmix2[2:4, col_idx_setmix] = MatrixMPI(ones(3, 3) * 66.0)

M_setmix2_gathered = gather_to_root(M_setmix2)
if rank == 0
    @test M_setmix2_gathered[2, 1] ≈ 66.0 atol=TOL
    @test M_setmix2_gathered[3, 5] ≈ 66.0 atol=TOL
    @test M_setmix2_gathered[4, 9] ≈ 66.0 atol=TOL
    @test M_setmix2_gathered[5, 5] ≈ 0.0 atol=TOL  # Unchanged
end


println(io0(), "[test] MatrixMPI setindex! with VectorMPI rows and Int column")

M_setint = MatrixMPI(zeros(n, n))
row_idx_setint = VectorMPI([2, 6, 10])
M_setint[row_idx_setint, 4] = VectorMPI([111.0, 222.0, 333.0])

M_setint_gathered = gather_to_root(M_setint)
if rank == 0
    @test M_setint_gathered[2, 4] ≈ 111.0 atol=TOL
    @test M_setint_gathered[6, 4] ≈ 222.0 atol=TOL
    @test M_setint_gathered[10, 4] ≈ 333.0 atol=TOL
end


println(io0(), "[test] MatrixMPI setindex! with Int row and VectorMPI columns")

M_setint2 = MatrixMPI(zeros(n, n))
col_idx_setint = VectorMPI([3, 7, 11])
M_setint2[5, col_idx_setint] = VectorMPI([444.0, 555.0, 666.0])

M_setint2_gathered = gather_to_root(M_setint2)
if rank == 0
    @test M_setint2_gathered[5, 3] ≈ 444.0 atol=TOL
    @test M_setint2_gathered[5, 7] ≈ 555.0 atol=TOL
    @test M_setint2_gathered[5, 11] ≈ 666.0 atol=TOL
end


println(io0(), "[test] SparseMatrixMPI setindex! with VectorMPI rows and range columns")

A_setmix = SparseMatrixMPI{Float64}(spzeros(n, n))
row_idx_sp_setmix = VectorMPI([1, 4, 7])
A_setmix[row_idx_sp_setmix, 2:4] = MatrixMPI(ones(3, 3) * 77.0)

A_setmix_gathered = gather_to_root(A_setmix)
if rank == 0
    @test A_setmix_gathered[1, 2] ≈ 77.0 atol=TOL
    @test A_setmix_gathered[4, 3] ≈ 77.0 atol=TOL
    @test A_setmix_gathered[7, 4] ≈ 77.0 atol=TOL
    @test A_setmix_gathered[2, 2] ≈ 0.0 atol=TOL  # Unchanged
end


println(io0(), "[test] SparseMatrixMPI setindex! with range rows and VectorMPI columns")

A_setmix2 = SparseMatrixMPI{Float64}(spzeros(n, n))
col_idx_sp_setmix = VectorMPI([1, 5, 9])
A_setmix2[2:4, col_idx_sp_setmix] = MatrixMPI(ones(3, 3) * 88.0)

A_setmix2_gathered = gather_to_root(A_setmix2)
if rank == 0
    @test A_setmix2_gathered[2, 1] ≈ 88.0 atol=TOL
    @test A_setmix2_gathered[3, 5] ≈ 88.0 atol=TOL
    @test A_setmix2_gathered[4, 9] ≈ 88.0 atol=TOL
    @test A_setmix2_gathered[5, 5] ≈ 0.0 atol=TOL  # Unchanged
end


println(io0(), "[test] SparseMatrixMPI setindex! with VectorMPI rows and Int column")

A_setint = SparseMatrixMPI{Float64}(spzeros(n, n))
row_idx_sp_setint = VectorMPI([2, 6, 10])
A_setint[row_idx_sp_setint, 4] = VectorMPI([111.0, 222.0, 333.0])

A_setint_gathered = gather_to_root(A_setint)
if rank == 0
    @test A_setint_gathered[2, 4] ≈ 111.0 atol=TOL
    @test A_setint_gathered[6, 4] ≈ 222.0 atol=TOL
    @test A_setint_gathered[10, 4] ≈ 333.0 atol=TOL
end


println(io0(), "[test] SparseMatrixMPI setindex! with Int row and VectorMPI columns")

A_setint2 = SparseMatrixMPI{Float64}(spzeros(n, n))
col_idx_sp_setint = VectorMPI([3, 7, 11])
A_setint2[5, col_idx_sp_setint] = VectorMPI([444.0, 555.0, 666.0])

A_setint2_gathered = gather_to_root(A_setint2)
if rank == 0
    @test A_setint2_gathered[5, 3] ≈ 444.0 atol=TOL
    @test A_setint2_gathered[5, 7] ≈ 555.0 atol=TOL
    @test A_setint2_gathered[5, 11] ≈ 666.0 atol=TOL
end


# ============================================================================
# Range setindex! with SparseMatrixMPI (matrix source)
# ============================================================================

println(io0(), "[test] SparseMatrixMPI range setindex! (matrix)")

A_range_matrix = SparseMatrixMPI{Float64}(copy(A_global))
new_block = Float64[100*i + j for i in 1:3, j in 1:4]
A_range_matrix[2:4, 3:6] = SparseMatrixMPI{Float64}(sparse(new_block))

for i in 2:4
    for j in 3:6
        @test A_range_matrix[i, j] ≈ 100*(i-1) + (j-2) atol=TOL
    end
end


# ============================================================================
# VectorMPI range setindex! with VectorMPI source
# ============================================================================

println(io0(), "[test] VectorMPI range setindex! (VectorMPI source)")

v_vec_src = VectorMPI(copy(v_global))
src_vec = VectorMPI([100.0, 200.0, 300.0, 400.0])
v_vec_src[3:6] = src_vec

for i in 1:n
    if 3 <= i <= 6
        @test v_vec_src[i] ≈ (i - 2) * 100.0 atol=TOL
    else
        @test v_vec_src[i] ≈ v_global[i] atol=TOL
    end
end


# ============================================================================
# MatrixMPI getindex with Colon, Colon (full matrix copy)
# ============================================================================

println(io0(), "[test] MatrixMPI getindex with Colon, Colon")

M_full = M[:, :]
@test size(M_full) == size(M)
@test M_full.row_partition == M.row_partition

# Verify all elements match
for i in 1:n
    for j in 1:n
        @test M_full[i, j] ≈ M_global[i, j] atol=TOL
    end
end


# ============================================================================
# MatrixMPI range setindex! with MatrixMPI source
# ============================================================================

println(io0(), "[test] MatrixMPI range setindex! (MatrixMPI source)")

M_matrix_src = MatrixMPI(zeros(n, n))
src_matrix = MatrixMPI(Float64[10*i + j for i in 1:4, j in 1:5])
M_matrix_src[2:5, 3:7] = src_matrix

for i in 2:5
    for j in 3:7
        @test M_matrix_src[i, j] ≈ 10*(i-1) + (j-2) atol=TOL
    end
end
# Unchanged regions should still be zero
@test M_matrix_src[1, 1] ≈ 0.0 atol=TOL
@test M_matrix_src[6, 8] ≈ 0.0 atol=TOL


# ============================================================================
# SparseMatrixMPI range setindex! with cross-rank SparseMatrixMPI source
# This tests the path: if num_rows_needed > 0 && intersect_start <= intersect_end
# ============================================================================

println(io0(), "[test] SparseMatrixMPI range setindex! (cross-rank sparse source)")

# Create a larger sparse matrix where the source spans multiple ranks
n_crossrank = 40
A_crossrank = SparseMatrixMPI{Float64}(spzeros(n_crossrank, n_crossrank))

# Create source matrix that spans multiple ranks (rows 1:20)
I_src = vcat(1:20, 1:10)  # Diagonal + some off-diagonal
J_src = vcat(1:20, 11:20)
V_src = Float64[i + j/100 for (i, j) in zip(I_src, J_src)]
src_sparse_global = sparse(I_src, J_src, V_src, 20, 20)
src_sparse = SparseMatrixMPI{Float64}(src_sparse_global)

# Set rows 10:29 and cols 5:24 from the source
# This forces cross-rank communication as src spans rows 1:20 and dest spans 10:29
A_crossrank[10:29, 5:24] = src_sparse

# Verify some values
# src[1,1] = 1.01 should go to A[10, 5]
@test A_crossrank[10, 5] ≈ 1.01 atol=TOL
# src[10,10] = 10.10 should go to A[19, 14]
@test A_crossrank[19, 14] ≈ 10.10 atol=TOL
# src[20,20] = 20.20 should go to A[29, 24]
@test A_crossrank[29, 24] ≈ 20.20 atol=TOL
# Position outside source should be zero
@test A_crossrank[1, 1] ≈ 0.0 atol=TOL


println(io0(), "[test] SparseMatrixMPI range setindex! (cross-rank with row intersection)")

# Another test where source and dest row ranges intersect differently on each rank
A_crossrank2 = SparseMatrixMPI{Float64}(spzeros(n_crossrank, n_crossrank))

# Create a source with specific pattern
I_src2 = [1, 2, 3, 4, 5, 1, 2, 3]
J_src2 = [1, 2, 3, 4, 5, 3, 4, 5]
V_src2 = Float64[100*i + j for (i, j) in zip(I_src2, J_src2)]
src_sparse2_global = sparse(I_src2, J_src2, V_src2, 5, 5)
src_sparse2 = SparseMatrixMPI{Float64}(src_sparse2_global)

# Assign to rows 18:22, cols 18:22 - this spans rank boundaries with 4 ranks and n=40
A_crossrank2[18:22, 18:22] = src_sparse2

# Verify diagonal entries
@test A_crossrank2[18, 18] ≈ 101.0 atol=TOL  # src[1,1] = 100*1 + 1 = 101
@test A_crossrank2[19, 19] ≈ 202.0 atol=TOL  # src[2,2]
@test A_crossrank2[20, 20] ≈ 303.0 atol=TOL  # src[3,3]
@test A_crossrank2[21, 21] ≈ 404.0 atol=TOL  # src[4,4]
@test A_crossrank2[22, 22] ≈ 505.0 atol=TOL  # src[5,5]

# Verify off-diagonal entries
@test A_crossrank2[18, 20] ≈ 103.0 atol=TOL  # src[1,3]
@test A_crossrank2[19, 21] ≈ 204.0 atol=TOL  # src[2,4]
@test A_crossrank2[20, 22] ≈ 305.0 atol=TOL  # src[3,5]


# ============================================================================
# Empty range tests for SparseMatrixMPI
# ============================================================================

println(io0(), "[test] SparseMatrixMPI getindex with empty row range")

A_empty_test = SparseMatrixMPI{Float64}(A_global)
A_empty_rows = A_empty_test[1:0, 1:5]
# Should match Julia builtin behavior: (0, 5)
@test size(A_empty_rows) == (0, 5)
@test nnz(SparseMatrixCSC(A_empty_rows)) == 0


println(io0(), "[test] SparseMatrixMPI getindex with empty column range")

A_empty_cols = A_empty_test[1:5, 1:0]
# Should match Julia builtin behavior: (5, 0)
@test size(A_empty_cols) == (5, 0)
@test nnz(SparseMatrixCSC(A_empty_cols)) == 0


println(io0(), "[test] SparseMatrixMPI getindex with both ranges empty")

A_empty_both = A_empty_test[1:0, 1:0]
@test size(A_empty_both) == (0, 0)
@test nnz(SparseMatrixCSC(A_empty_both)) == 0


# Also test MatrixMPI empty ranges
println(io0(), "[test] MatrixMPI getindex with empty row range")

M_empty_test = MatrixMPI(M_global)
M_empty_rows = M_empty_test[1:0, 1:5]
@test size(M_empty_rows) == (0, 5)


println(io0(), "[test] MatrixMPI getindex with empty column range")

M_empty_cols = M_empty_test[1:5, 1:0]
@test size(M_empty_cols) == (5, 0)


println(io0(), "[test] MatrixMPI getindex with both ranges empty")

M_empty_both = M_empty_test[1:0, 1:0]
@test size(M_empty_both) == (0, 0)


# ============================================================================
# SparseMatrixMPI setindex! fast path with matching partitions
# ============================================================================

println(io0(), "[test] SparseMatrixMPI setindex! fast path (matching partitions)")

# Create source and dest with same size so partitions will match
n_match = 20
A_dest_match = SparseMatrixMPI{Float64}(spzeros(n_match, n_match))
I_src_match = [1, 2, 3, 4, 5, 10, 15, 20]
J_src_match = [2, 3, 4, 5, 6, 11, 16, 20]
V_src_match = Float64[10*i + j for (i, j) in zip(I_src_match, J_src_match)]
src_match_global = sparse(I_src_match, J_src_match, V_src_match, n_match, n_match)
src_match = SparseMatrixMPI{Float64}(src_match_global)

# Assign entire matrix (partitions will match exactly)
A_dest_match[1:n_match, 1:n_match] = src_match

# Verify values
@test A_dest_match[1, 2] ≈ 12.0 atol=TOL
@test A_dest_match[5, 6] ≈ 56.0 atol=TOL
@test A_dest_match[10, 11] ≈ 111.0 atol=TOL
@test A_dest_match[20, 20] ≈ 220.0 atol=TOL
@test A_dest_match[1, 1] ≈ 0.0 atol=TOL  # Structural zero


# ============================================================================
# MatrixMPI getindex with VectorMPI rows and Int column (cross-rank)
# ============================================================================

println(io0(), "[test] MatrixMPI getindex with VectorMPI rows and Int column (cross-rank)")

# Create larger matrix to ensure indices span multiple ranks
n_cross = 40
M_cross_global = Float64[i * 10.0 + j for i in 1:n_cross, j in 1:10]
M_cross = MatrixMPI(M_cross_global)

# Create row indices spanning all ranks
row_idx_cross = VectorMPI(vcat(1:3, div(n_cross,2)-1:div(n_cross,2)+1, n_cross-2:n_cross))
result_cross = M_cross[row_idx_cross, 5]

@test result_cross isa VectorMPI
@test length(result_cross) == 9

result_cross_gathered = gather_to_root(result_cross)
expected_rows = vcat(1:3, div(n_cross,2)-1:div(n_cross,2)+1, n_cross-2:n_cross)
if rank == 0
    for (k, row) in enumerate(expected_rows)
        @test result_cross_gathered[k] ≈ M_cross_global[row, 5] atol=TOL
    end
end


# ============================================================================
# MatrixMPI setindex! with VectorMPI rows and range columns (cross-rank send)
# ============================================================================

println(io0(), "[test] MatrixMPI setindex! with VectorMPI rows, range columns (cross-rank)")

M_set_cross = MatrixMPI(zeros(n_cross, 10))
row_idx_set_cross = VectorMPI(vcat(1:2, n_cross-1:n_cross))  # First and last rows
src_matrix_cross = MatrixMPI(Float64[100*i + j for i in 1:4, j in 1:5])

M_set_cross[row_idx_set_cross, 3:7] = src_matrix_cross

M_set_gathered = gather_to_root(M_set_cross)
if rank == 0
    @test M_set_gathered[1, 3] ≈ 101.0 atol=TOL
    @test M_set_gathered[1, 7] ≈ 105.0 atol=TOL
    @test M_set_gathered[2, 3] ≈ 201.0 atol=TOL
    @test M_set_gathered[n_cross-1, 3] ≈ 301.0 atol=TOL
    @test M_set_gathered[n_cross, 7] ≈ 405.0 atol=TOL
    @test M_set_gathered[10, 5] ≈ 0.0 atol=TOL  # Unchanged
end


# ============================================================================
# SparseMatrixMPI setindex! with MatrixMPI source, VectorMPI rows (cross-rank)
# ============================================================================

println(io0(), "[test] SparseMatrixMPI setindex! with MatrixMPI source, VectorMPI rows (cross-rank)")

A_sparse_set_cross = SparseMatrixMPI{Float64}(spzeros(n_cross, 10))
row_idx_sparse_cross = VectorMPI(vcat(1:2, n_cross-1:n_cross))
src_dense_cross = MatrixMPI(Float64[1000*i + j for i in 1:4, j in 1:3])

A_sparse_set_cross[row_idx_sparse_cross, 2:4] = src_dense_cross

A_sparse_gathered = gather_to_root(A_sparse_set_cross)
if rank == 0
    @test A_sparse_gathered[1, 2] ≈ 1001.0 atol=TOL
    @test A_sparse_gathered[1, 4] ≈ 1003.0 atol=TOL
    @test A_sparse_gathered[2, 2] ≈ 2001.0 atol=TOL
    @test A_sparse_gathered[n_cross-1, 2] ≈ 3001.0 atol=TOL
    @test A_sparse_gathered[n_cross, 4] ≈ 4003.0 atol=TOL
    @test A_sparse_gathered[10, 5] ≈ 0.0 atol=TOL  # Unchanged
end


# ============================================================================
# Cross-type assignment (Float64 -> ComplexF64)
# ============================================================================

println(io0(), "[test] VectorMPI cross-type setindex! (Float64 -> ComplexF64)")

# Create a ComplexF64 vector and assign Float64 values to a range
n_cross_type = 8
v_cx_target = VectorMPI(ComplexF64.(1:n_cross_type) .+ im .* ComplexF64.(n_cross_type:-1:1))
v_real_src = VectorMPI(Float64[100.0, 200.0, 300.0])

# Assign Float64 VectorMPI to a range of ComplexF64 VectorMPI
v_cx_target[2:4] = v_real_src

v_cx_target_gathered = gather_to_root(v_cx_target)
if rank == 0
    # Check that assignment worked and types are correct
    @test v_cx_target_gathered[1] ≈ ComplexF64(1.0 + 8.0im) atol=TOL  # Unchanged
    @test v_cx_target_gathered[2] ≈ ComplexF64(100.0 + 0.0im) atol=TOL  # Assigned from real
    @test v_cx_target_gathered[3] ≈ ComplexF64(200.0 + 0.0im) atol=TOL  # Assigned from real
    @test v_cx_target_gathered[4] ≈ ComplexF64(300.0 + 0.0im) atol=TOL  # Assigned from real
    @test v_cx_target_gathered[5] ≈ ComplexF64(5.0 + 4.0im) atol=TOL  # Unchanged
end


println(io0(), "[test] MatrixMPI cross-type setindex! (Float64 -> ComplexF64)")

# Create a ComplexF64 matrix and assign Float64 values to a submatrix
m_cross_type = 6
n_cols_cross = 5
M_cx_target = MatrixMPI(ComplexF64[i + im*j for i in 1:m_cross_type, j in 1:n_cols_cross])
M_real_src = MatrixMPI(Float64[1000.0*i + j for i in 1:2, j in 1:3])

# Assign Float64 MatrixMPI to a range of ComplexF64 MatrixMPI
M_cx_target[2:3, 1:3] = M_real_src

M_cx_target_gathered = gather_to_root(M_cx_target)
if rank == 0
    # Check that assignment worked and types are correct
    @test M_cx_target_gathered[1, 1] ≈ ComplexF64(1.0 + 1.0im) atol=TOL  # Unchanged
    @test M_cx_target_gathered[2, 1] ≈ ComplexF64(1001.0 + 0.0im) atol=TOL  # Assigned from real
    @test M_cx_target_gathered[2, 2] ≈ ComplexF64(1002.0 + 0.0im) atol=TOL  # Assigned from real
    @test M_cx_target_gathered[2, 3] ≈ ComplexF64(1003.0 + 0.0im) atol=TOL  # Assigned from real
    @test M_cx_target_gathered[3, 1] ≈ ComplexF64(2001.0 + 0.0im) atol=TOL  # Assigned from real
    @test M_cx_target_gathered[3, 3] ≈ ComplexF64(2003.0 + 0.0im) atol=TOL  # Assigned from real
    @test M_cx_target_gathered[4, 1] ≈ ComplexF64(4.0 + 1.0im) atol=TOL  # Unchanged
    @test M_cx_target_gathered[2, 4] ≈ ComplexF64(2.0 + 4.0im) atol=TOL  # Unchanged (column not in assignment)
end


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
