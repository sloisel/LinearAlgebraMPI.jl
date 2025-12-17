# MPI test for repartition
# This file is executed under mpiexec by runtests.jl

using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra: norm
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

const TOL = 1e-12

ts = @testset QuietTestSet "Repartition" begin

println(io0(), "[test] VectorMPI repartition")

# Test 1: VectorMPI uniform to non-uniform partition
n = 12
v_global = collect(1.0:n)
v = VectorMPI(v_global)
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

println(io0(), "[test] VectorMPI same partition (fast path)")

v2 = repartition(v, v.partition)
@test v2 === v  # Should be same object

println(io0(), "[test] VectorMPI ComplexF64")

v_c_global = ComplexF64.(1:n) .+ im .* ComplexF64.(n:-1:1)
v_c = VectorMPI(v_c_global)
v_c_repart = repartition(v_c, new_p)
v_c_repart_global = Vector(v_c_repart)
@test norm(v_c_repart_global - v_c_global) < TOL

println(io0(), "[test] VectorMPI plan caching")

LinearAlgebraMPI.clear_plan_cache!()
v3_repart = repartition(v, new_p)
v4_repart = repartition(v, new_p)
@test v3_repart.structural_hash == v4_repart.structural_hash

println(io0(), "[test] MatrixMPI repartition")

# Test dense matrix repartition
m, ncols = 8, 4
M_global = reshape(Float64.(1:m*ncols), m, ncols)
M = MatrixMPI(M_global)

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

println(io0(), "[test] MatrixMPI same partition (fast path)")

M2 = repartition(M, M.row_partition)
@test M2 === M

println(io0(), "[test] MatrixMPI ComplexF64")

M_c_global = ComplexF64.(reshape(1:m*ncols, m, ncols)) .+ im
M_c = MatrixMPI(M_c_global)
M_c_repart = repartition(M_c, new_row_p)
M_c_repart_global = Matrix(M_c_repart)
@test norm(M_c_repart_global - M_c_global) < TOL

println(io0(), "[test] SparseMatrixMPI repartition")

# Test sparse matrix repartition
m_sparse, n_sparse = 8, 6
I_idx = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7, 2, 4, 6, 8]
J_idx = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 4, 5, 6, 1]
V_sparse = Float64.(1:length(I_idx))
A_global = sparse(I_idx, J_idx, V_sparse, m_sparse, n_sparse)

A = SparseMatrixMPI{Float64}(A_global)

# Create new partition
new_sparse_p = LinearAlgebraMPI.uniform_partition(m_sparse, nranks)
if nranks >= 2
    new_sparse_p = [1]
    total = 0
    for r in 0:(nranks-1)
        if r == 0
            count = 2
        else
            count = div(m_sparse - 2, nranks - 1) + (r - 1 < mod(m_sparse - 2, nranks - 1) ? 1 : 0)
        end
        total += count
        push!(new_sparse_p, total + 1)
    end
end

A_repart = repartition(A, new_sparse_p)
A_repart_global = SparseMatrixCSC(A_repart)
A_global_from_dist = SparseMatrixCSC(A)
@test norm(A_repart_global - A_global_from_dist, Inf) < TOL
@test A_repart.row_partition == new_sparse_p
@test A_repart.col_partition == A.col_partition

println(io0(), "[test] SparseMatrixMPI nnz preserved")

@test nnz(A_repart) == nnz(A)

println(io0(), "[test] SparseMatrixMPI same partition (fast path)")

A2 = repartition(A, A.row_partition)
@test A2 === A

println(io0(), "[test] SparseMatrixMPI ComplexF64")

V_sparse_c = ComplexF64.(V_sparse) .+ im
A_c_global = sparse(I_idx, J_idx, V_sparse_c, m_sparse, n_sparse)
A_c = SparseMatrixMPI{ComplexF64}(A_c_global)
A_c_repart = repartition(A_c, new_sparse_p)
A_c_repart_global = SparseMatrixCSC(A_c_repart)
A_c_global_from_dist = SparseMatrixCSC(A_c)
@test norm(A_c_repart_global - A_c_global_from_dist, Inf) < TOL

println(io0(), "[test] Operations after repartition")

# Test that repartitioned matrix can be used in operations
x = VectorMPI(ones(n_sparse))
y_orig = A * x
y_repart = A_repart * x
y_diff = norm(Vector(y_orig) - Vector(y_repart))
@test y_diff < TOL

println(io0(), "[test] Repartition plan caching")

LinearAlgebraMPI.clear_plan_cache!()
A3_repart = repartition(A, new_sparse_p)
A4_repart = repartition(A, new_sparse_p)
@test A3_repart.structural_hash == A4_repart.structural_hash

end  # QuietTestSet

# Aggregate counts across ranks
local_counts = [ts.counts[:pass], ts.counts[:fail], ts.counts[:error], ts.counts[:broken], ts.counts[:skip]]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

println(io0(), "Test Summary: Repartition | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
