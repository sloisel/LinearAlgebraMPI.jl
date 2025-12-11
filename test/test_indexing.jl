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
