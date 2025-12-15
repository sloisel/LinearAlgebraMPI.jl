# MPI test for addition and subtraction
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

const TOL = 1e-12

ts = @testset QuietTestSet "Addition" begin

println(io0(), "[test] Matrix addition")

n = 8
I_A = [1:n; 1:n-1; 2:n]
J_A = [1:n; 2:n; 1:n-1]
V_A = [2.0*ones(Float64, n); -0.5*ones(n-1); -0.5*ones(n-1)]
A = sparse(I_A, J_A, V_A, n, n)

I_B = [1:n; 1:n-1; 2:n]
J_B = [1:n; 2:n; 1:n-1]
V_B = [1.5*ones(Float64, n); 0.25*ones(n-1); 0.25*ones(n-1)]
B = sparse(I_B, J_B, V_B, n, n)

Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)
Cdist = Adist + Bdist
C_ref = A + B
C_ref_dist = SparseMatrixMPI{Float64}(C_ref)
err = norm(Cdist - C_ref_dist, Inf)
@test err < TOL

println(io0(), "[test] Matrix addition with ComplexF64")

V_A_c = ComplexF64.([2.0*ones(n); -0.5*ones(n-1); -0.5*ones(n-1)]) .+
        im .* ComplexF64.([0.1*ones(n); 0.2*ones(n-1); -0.2*ones(n-1)])
A_c = sparse(I_A, J_A, V_A_c, n, n)

V_B_c = ComplexF64.([1.5*ones(n); 0.25*ones(n-1); 0.25*ones(n-1)]) .+
        im .* ComplexF64.([-0.1*ones(n); 0.1*ones(n-1); 0.1*ones(n-1)])
B_c = sparse(I_B, J_B, V_B_c, n, n)

Adist_c = SparseMatrixMPI{ComplexF64}(A_c)
Bdist_c = SparseMatrixMPI{ComplexF64}(B_c)
Cdist_c = Adist_c + Bdist_c
C_ref_c = A_c + B_c
C_ref_dist_c = SparseMatrixMPI{ComplexF64}(C_ref_c)
err_c = norm(Cdist_c - C_ref_dist_c, Inf)
@test err_c < TOL

println(io0(), "[test] Matrix subtraction")

V_A2 = [3.0*ones(Float64, n); -0.7*ones(n-1); -0.7*ones(n-1)]
A2 = sparse(I_A, J_A, V_A2, n, n)
V_B2 = [1.0*ones(Float64, n); 0.3*ones(n-1); 0.3*ones(n-1)]
B2 = sparse(I_B, J_B, V_B2, n, n)

Adist2 = SparseMatrixMPI{Float64}(A2)
Bdist2 = SparseMatrixMPI{Float64}(B2)
Cdist2 = Adist2 - Bdist2
C_ref2 = A2 - B2
C_ref_dist2 = SparseMatrixMPI{Float64}(C_ref2)
err2 = norm(Cdist2 - C_ref_dist2, Inf)
@test err2 < TOL

println(io0(), "[test] Different sparsity patterns")

I_A3 = [1, 1, 2, 3, 4, 5, 6, 7, 8]
J_A3 = [1, 2, 2, 3, 4, 5, 6, 7, 8]
V_A3 = Float64.(1:9)
A3 = sparse(I_A3, J_A3, V_A3, n, n)

I_B3 = [1, 2, 2, 3, 4, 5, 6, 7, 8]
J_B3 = [1, 1, 2, 3, 4, 5, 6, 7, 8]
V_B3 = Float64.(9:-1:1)
B3 = sparse(I_B3, J_B3, V_B3, n, n)

Adist3 = SparseMatrixMPI{Float64}(A3)
Bdist3 = SparseMatrixMPI{Float64}(B3)
Cdist3 = Adist3 + Bdist3
C_ref3 = A3 + B3
C_ref_dist3 = SparseMatrixMPI{Float64}(C_ref3)
err3 = norm(Cdist3 - C_ref_dist3, Inf)
@test err3 < TOL

println(io0(), "[test] Cached addition path")

# Test that repeating the same addition uses the cached plan
Cdist3_repeat = Adist3 + Bdist3
err3_repeat = norm(Cdist3_repeat - C_ref_dist3, Inf)
@test err3_repeat < TOL

println(io0(), "[test] Cached subtraction path")

# Test that repeating the same subtraction uses the cached plan
Ddist = Adist3 - Bdist3
D_ref = A3 - B3
D_ref_dist = SparseMatrixMPI{Float64}(D_ref)
err_sub1 = norm(Ddist - D_ref_dist, Inf)
@test err_sub1 < TOL

Ddist_repeat = Adist3 - Bdist3
err_sub2 = norm(Ddist_repeat - D_ref_dist, Inf)
@test err_sub2 < TOL

end  # QuietTestSet

# Aggregate counts across ranks
local_counts = [ts.counts[:pass], ts.counts[:fail], ts.counts[:error], ts.counts[:broken], ts.counts[:skip]]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

println(io0(), "Test Summary: Addition | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
