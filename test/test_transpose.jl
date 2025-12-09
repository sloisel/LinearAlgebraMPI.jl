# MPI test for transpose
# This file is executed under mpiexec by runtests.jl

using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra: norm
using Test

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

const TOL = 1e-12

@testset "Transpose" begin
    # Create a deterministic test matrix (same on all ranks)
    m, n = 10, 8
    I_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3, 5, 7, 9]
    J_idx = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2,   3, 5, 7, 1, 4]
    V = Float64.(1:length(I_idx))
    A = sparse(I_idx, J_idx, V, m, n)

    Adist = SparseMatrixMPI{Float64}(A)

    # Create and execute transpose plan
    plan = TransposePlan(Adist)
    ATdist = execute_plan!(plan, Adist)

    # Reference transpose
    AT_ref = sparse(A')
    AT_ref_dist = SparseMatrixMPI{Float64}(AT_ref)

    err = norm(ATdist - AT_ref_dist, Inf)
    @test err < TOL

    if rank == 0
        println("  ✓ Transpose: error = $err")
    end
end

@testset "Transpose with ComplexF64" begin
    m, n = 10, 8
    I_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3, 5, 7, 9]
    J_idx = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2,   3, 5, 7, 1, 4]
    V = ComplexF64.(1:length(I_idx)) .+ im .* ComplexF64.(length(I_idx):-1:1)
    A = sparse(I_idx, J_idx, V, m, n)

    Adist = SparseMatrixMPI{ComplexF64}(A)

    plan = TransposePlan(Adist)
    ATdist = execute_plan!(plan, Adist)

    AT_ref = sparse(transpose(A))  # transpose, not adjoint
    AT_ref_dist = SparseMatrixMPI{ComplexF64}(AT_ref)

    err = norm(ATdist - AT_ref_dist, Inf)
    @test err < TOL

    if rank == 0
        println("  ✓ Transpose with ComplexF64: error = $err")
    end
end

@testset "Square Matrix Transpose" begin
    n = 8
    I_idx = [1:n; 1:n-1; 2:n]
    J_idx = [1:n; 2:n; 1:n-1]
    V = [2.0*ones(Float64, n); 0.3*ones(n-1); 0.7*ones(n-1)]
    A = sparse(I_idx, J_idx, V, n, n)

    Adist = SparseMatrixMPI{Float64}(A)

    plan = TransposePlan(Adist)
    ATdist = execute_plan!(plan, Adist)

    AT_ref = sparse(A')
    AT_ref_dist = SparseMatrixMPI{Float64}(AT_ref)

    err = norm(ATdist - AT_ref_dist, Inf)
    @test err < TOL

    if rank == 0
        println("  ✓ Square matrix transpose: error = $err")
    end
end

MPI.Finalize()
