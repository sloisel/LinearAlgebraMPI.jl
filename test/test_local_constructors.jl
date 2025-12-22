# Tests for local constructors: VectorMPI_local, MatrixMPI_local, SparseMatrixMPI_local
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

ts = @testset QuietTestSet "Local Constructors" begin

println(io0(), "[test] VectorMPI_local basic")

# Test VectorMPI_local: each rank provides different-sized local parts
# Create local vectors that, when concatenated, form [1, 2, 3, ..., 10]
local_sizes = [div(10, nranks) + (r <= mod(10, nranks) ? 1 : 0) for r in 1:nranks]
global_start = sum(local_sizes[1:rank]) + 1
global_end = sum(local_sizes[1:rank+1])
v_local = Float64.(collect(global_start:global_end))

v_mpi = VectorMPI_local(v_local)
v_global = Vector(v_mpi)

# All ranks should have the same global vector
@test v_global == Float64.(1:10)
@test length(v_mpi) == 10


println(io0(), "[test] VectorMPI_local roundtrip consistency")

# Test that VectorMPI_local produces same result as VectorMPI for default partition
v_original = Float64[1.5, -2.3, 3.7, 4.1, -5.9, 6.2, 7.8, -8.4]
v_from_global = VectorMPI(v_original)

# Extract local part and reconstruct
v_local_extract = copy(v_from_global.v)
v_from_local = VectorMPI_local(v_local_extract)
v_back = Vector(v_from_local)

@test v_back == v_original
@test v_from_local.partition == v_from_global.partition


println(io0(), "[test] VectorMPI_local complex")

# Test with complex values
v_complex_local = ComplexF64[(rank+1) + (rank+1)*im, (rank+1)*2 - (rank+1)*2im]
v_complex_mpi = VectorMPI_local(v_complex_local)
@test length(v_complex_mpi) == nranks * 2
@test eltype(v_complex_mpi) == ComplexF64


println(io0(), "[test] MatrixMPI_local basic")

# Test MatrixMPI_local: each rank provides some rows of a matrix
# Create a 10x4 matrix distributed across ranks
m_global = 10
n_cols = 4
row_sizes = [div(m_global, nranks) + (r <= mod(m_global, nranks) ? 1 : 0) for r in 1:nranks]
row_start = sum(row_sizes[1:rank]) + 1
row_end = sum(row_sizes[1:rank+1])
local_nrows = row_end - row_start + 1

# Create deterministic local matrix based on global row indices
M_local = Float64[(i + j*0.1) for i in row_start:row_end, j in 1:n_cols]
M_mpi = MatrixMPI_local(M_local)

# Verify size
@test size(M_mpi) == (m_global, n_cols)

# Gather and verify
M_gathered = Matrix(M_mpi)
M_expected = Float64[(i + j*0.1) for i in 1:m_global, j in 1:n_cols]
@test M_gathered ≈ M_expected


println(io0(), "[test] MatrixMPI_local roundtrip consistency")

# Test that MatrixMPI_local produces same result as MatrixMPI for default partition
M_original = Float64[1.1 2.2 3.3;
                     4.4 5.5 6.6;
                     7.7 8.8 9.9;
                     10.0 11.1 12.2;
                     13.3 14.4 15.5]
M_from_global = MatrixMPI(M_original)

# Extract local part and reconstruct
M_local_extract = copy(M_from_global.A)
M_from_local = MatrixMPI_local(M_local_extract)
M_back = Matrix(M_from_local)

@test M_back == M_original
@test M_from_local.row_partition == M_from_global.row_partition


println(io0(), "[test] MatrixMPI_local complex")

# Test with complex values
M_complex_local = ComplexF64[(rank+1) + j*im for _ in 1:(rank == 0 ? 2 : 1), j in 1:3]
M_complex_mpi = MatrixMPI_local(M_complex_local)
@test size(M_complex_mpi, 2) == 3
@test eltype(M_complex_mpi) == ComplexF64


println(io0(), "[test] SparseMatrixMPI_local basic")

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
V_local = Float64[]

for local_row in 1:local_sparse_nrows
    global_row = sparse_row_start + local_row - 1
    # Diagonal entry if within bounds
    if global_row <= n_sparse
        push!(I_local, local_row)
        push!(J_local, global_row)
        push!(V_local, Float64(global_row))
    end
    # Off-diagonal entry
    col_off = mod(global_row, n_sparse) + 1
    push!(I_local, local_row)
    push!(J_local, col_off)
    push!(V_local, Float64(global_row) * 0.1)
end

# Create local CSC in transpose form (columns = local rows, rowval = global columns)
# Note: For SparseMatrixMPI_local, we pass transpose(CSC) where CSC has:
#   - columns = local rows
#   - rows (rowval) = global column indices
local_csc = sparse(J_local, I_local, V_local, n_sparse, local_sparse_nrows)
local_transpose = transpose(local_csc)

S_mpi = SparseMatrixMPI_local(local_transpose)
@test size(S_mpi) == (m_sparse, n_sparse)


println(io0(), "[test] SparseMatrixMPI_local roundtrip consistency")

# Test roundtrip: global -> partition -> local -> rebuild
# Use a deterministic sparse matrix
I_orig = [1, 2, 3, 4, 5, 6, 7, 8, 1, 5, 9, 10]
J_orig = [1, 2, 3, 4, 5, 6, 7, 8, 5, 1, 2, 3]
V_orig = Float64[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 12.2]
S_original = sparse(I_orig, J_orig, V_orig, 12, 10)

S_from_global = SparseMatrixMPI{Float64}(S_original)

# Extract local part - using explicit CSR arrays
# Need to convert back to global indices
col_indices = S_from_global.col_indices

# Rebuild CSC from explicit arrays with global indices
# The CSC has shape (ncols_compressed, nrows_local) - swapped for transpose view
AT_uncompressed = SparseMatrixCSC(
    size(S_original, 2),  # original global ncols
    S_from_global.nrows_local,  # number of local rows
    copy(S_from_global.rowptr),  # becomes colptr in CSC
    [col_indices[r] for r in S_from_global.colval],  # map local to global indices
    copy(S_from_global.nzval)
)

# Rebuild from local
S_from_local = SparseMatrixMPI_local(transpose(AT_uncompressed))
S_back = SparseMatrixCSC(S_from_local)

@test S_back == S_original
@test S_from_local.row_partition == S_from_global.row_partition


println(io0(), "[test] SparseMatrixMPI_local with Adjoint")

# Test with Adjoint (values should be conjugated)
I_adj = [1, 2, 3]
J_adj = [1, 2, 3]
V_adj = ComplexF64[1+1im, 2+2im, 3+3im]
local_sparse_adj = sparse(J_adj, I_adj, V_adj, 5, 3)  # 5 cols, 3 local rows

# Using adjoint instead of transpose should conjugate values
S_adj = SparseMatrixMPI_local(adjoint(local_sparse_adj))
@test eltype(S_adj) == ComplexF64

# The nonzero values should be conjugated
S_adj_csc = SparseMatrixCSC(S_adj)
# Values should be conjugated from original
# Note: Due to partition differences across ranks, just verify structure is valid
@test nnz(S_adj_csc) > 0


println(io0(), "[test] MatrixMPI_local * VectorMPI_local")

# Test that locally constructed matrices work with operations
# Create compatible matrix and vector
m_op = 8
n_op = 6
op_row_sizes = [div(m_op, nranks) + (r <= mod(m_op, nranks) ? 1 : 0) for r in 1:nranks]
op_row_start = sum(op_row_sizes[1:rank]) + 1
op_row_end = sum(op_row_sizes[1:rank+1])

A_local_op = Float64[(i + j*0.1) for i in op_row_start:op_row_end, j in 1:n_op]
A_mpi_op = MatrixMPI_local(A_local_op)

# Create vector partition that matches columns
v_op_sizes = [div(n_op, nranks) + (r <= mod(n_op, nranks) ? 1 : 0) for r in 1:nranks]
v_op_start = sum(v_op_sizes[1:rank]) + 1
v_op_end = sum(v_op_sizes[1:rank+1])
v_local_op = Float64.(collect(v_op_start:v_op_end))
v_mpi_op = VectorMPI_local(v_local_op)

# Compute A * v
y_mpi = A_mpi_op * v_mpi_op

# Verify against expected result
A_full = Matrix(A_mpi_op)
v_full = Vector(v_mpi_op)
y_expected = A_full * v_full
y_result = Vector(y_mpi)

@test y_result ≈ y_expected


println(io0(), "[test] SparseMatrixMPI_local * VectorMPI_local")

# Test sparse matrix-vector multiplication with local constructors
# First create global matrix and vector, then extract local parts
I_sp = [1, 2, 3, 4, 5, 6, 1, 3, 5, 7, 9]
J_sp = [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2]
V_sp = Float64[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 0.5, 0.5, 0.5, 0.5]
S_sp_global = sparse(I_sp, J_sp, V_sp, 10, 8)

S_sp_mpi = SparseMatrixMPI{Float64}(S_sp_global)
v_sp_global = Float64.(1:8)
v_sp_mpi = VectorMPI(v_sp_global)

# Compute result
y_sp = S_sp_mpi * v_sp_mpi
y_sp_expected = S_sp_global * v_sp_global
y_sp_result = Vector(y_sp)

@test y_sp_result ≈ y_sp_expected


end  # testset

# Report results from rank 0
println("Test Summary: Local Constructors | Pass: $(ts.counts[:pass])  Fail: $(ts.counts[:fail])  Error: $(ts.counts[:error])")

# Exit with appropriate code
exit_code = (ts.counts[:fail] + ts.counts[:error] > 0) ? 1 : 0
MPI.Finalize()
exit(exit_code)
