# Test for multithreaded sparse matrix multiplication (⊛ operator)
# This file is executed under mpiexec by runtests.jl

using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra: norm
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

const TOL = 1e-12

ts = @testset QuietTestSet "Threaded Sparse Multiplication" begin

println(io0(), "[test] Threaded sparse multiplication (⊛) with Float64")

n = 20
I_A = [1:n; 1:n-1; 2:n]
J_A = [1:n; 2:n; 1:n-1]
V_A = [2.0*ones(Float64, n); -0.5*ones(n-1); -0.5*ones(n-1)]
A = sparse(I_A, J_A, V_A, n, n)

I_B = [1:n; 1:n-1; 2:n]
J_B = [1:n; 2:n; 1:n-1]
V_B = [1.5*ones(Float64, n); 0.25*ones(n-1); 0.25*ones(n-1)]
B = sparse(I_B, J_B, V_B, n, n)

C_threaded = A ⊛ B
C_ref = A * B
@test norm(C_threaded - C_ref, Inf) < TOL

println(io0(), "[test] Threaded sparse multiplication (⊛) with ComplexF64")

V_A_c = ComplexF64.([2.0*ones(n); -0.5*ones(n-1); -0.5*ones(n-1)]) .+
        im .* ComplexF64.([0.1*ones(n); 0.2*ones(n-1); -0.2*ones(n-1)])
A_c = sparse(I_A, J_A, V_A_c, n, n)

V_B_c = ComplexF64.([1.5*ones(n); 0.25*ones(n-1); 0.25*ones(n-1)]) .+
        im .* ComplexF64.([-0.1*ones(n); 0.1*ones(n-1); 0.1*ones(n-1)])
B_c = sparse(I_B, J_B, V_B_c, n, n)

C_threaded_c = A_c ⊛ B_c
C_ref_c = A_c * B_c
@test norm(C_threaded_c - C_ref_c, Inf) < TOL

println(io0(), "[test] Non-square matrices")

m, k, n2 = 15, 10, 8
I_A2 = [1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14]
J_A2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5]
V_A2 = Float64.(1:15) ./ 10
A2 = sparse(I_A2, J_A2, V_A2, m, k)

I_B2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3, 5]
J_B2 = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5]
V_B2 = Float64.(1:13) ./ 5
B2 = sparse(I_B2, J_B2, V_B2, k, n2)

C_threaded2 = A2 ⊛ B2
C_ref2 = A2 * B2
@test norm(C_threaded2 - C_ref2, Inf) < TOL

println(io0(), "[test] Single column result")

A3 = sparse([1, 2, 3], [1, 1, 1], [1.0, 2.0, 3.0], 3, 1)
B3 = sparse([1], [1], [2.0], 1, 1)
C_threaded3 = A3 ⊛ B3
C_ref3 = A3 * B3
@test norm(C_threaded3 - C_ref3, Inf) < TOL

println(io0(), "[test] Empty result")

A4 = sparse(Int[], Int[], Float64[], 5, 3)
B4 = sparse(Int[], Int[], Float64[], 3, 4)
C_threaded4 = A4 ⊛ B4
C_ref4 = A4 * B4
@test nnz(C_threaded4) == 0
@test size(C_threaded4) == size(C_ref4)

println(io0(), "[test] Large matrix (triggers threading, n > 200)")

# 500×500 matrix large enough to trigger threading (n ÷ 100 = 5 threads)
n5 = 500
I_A5 = [1:n5; 1:n5-1; 2:n5]
J_A5 = [1:n5; 2:n5; 1:n5-1]
V_A5 = [2.0*ones(Float64, n5); -0.5*ones(n5-1); -0.5*ones(n5-1)]
A5 = sparse(I_A5, J_A5, V_A5, n5, n5)

C_threaded5 = A5 ⊛ A5
C_ref5 = A5 * A5
@test norm(C_threaded5 - C_ref5, Inf) < TOL

println(io0(), "[test] All tests passed")

end  # QuietTestSet

# Aggregate counts across ranks
comm = MPI.COMM_WORLD
local_counts = [
    get(ts.counts, :pass, 0),
    get(ts.counts, :fail, 0),
    get(ts.counts, :error, 0),
    get(ts.counts, :broken, 0),
    get(ts.counts, :skip, 0),
]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

println(io0(), "Test Summary: Threaded Sparse Multiplication | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
