# MPI test for matrix multiplication
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

@testset "Matrix Multiplication" begin
    # Create deterministic test matrices (same on all ranks)
    n = 8
    # Matrix A: tridiagonal
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = [2.0*ones(Float64, n); -0.5*ones(n-1); -0.5*ones(n-1)]
    A = sparse(I_A, J_A, V_A, n, n)

    # Matrix B: different tridiagonal pattern
    I_B = [1:n; 1:n-1; 2:n]
    J_B = [1:n; 2:n; 1:n-1]
    V_B = [1.5*ones(Float64, n); 0.25*ones(n-1); 0.25*ones(n-1)]
    B = sparse(I_B, J_B, V_B, n, n)

    # Create distributed matrices
    Adist = SparseMatrixMPI{Float64}(A)
    Bdist = SparseMatrixMPI{Float64}(B)

    # Compute distributed product
    Cdist = Adist * Bdist

    # Reference result
    C_ref = A * B
    C_ref_dist = SparseMatrixMPI{Float64}(C_ref)

    # Compare using norm
    err = norm(Cdist - C_ref_dist, Inf)
    @test err < TOL

    if rank == 0
        println("  ✓ Matrix multiplication: error = $err")
    end
end

@testset "Matrix Multiplication with ComplexF64" begin
    n = 8
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = ComplexF64.([2.0*ones(n); -0.5*ones(n-1); -0.5*ones(n-1)]) .+
          im .* ComplexF64.([0.1*ones(n); 0.2*ones(n-1); -0.2*ones(n-1)])
    A = sparse(I_A, J_A, V_A, n, n)

    I_B = [1:n; 1:n-1; 2:n]
    J_B = [1:n; 2:n; 1:n-1]
    V_B = ComplexF64.([1.5*ones(n); 0.25*ones(n-1); 0.25*ones(n-1)]) .+
          im .* ComplexF64.([-0.1*ones(n); 0.1*ones(n-1); 0.1*ones(n-1)])
    B = sparse(I_B, J_B, V_B, n, n)

    Adist = SparseMatrixMPI{ComplexF64}(A)
    Bdist = SparseMatrixMPI{ComplexF64}(B)

    Cdist = Adist * Bdist
    C_ref = A * B
    C_ref_dist = SparseMatrixMPI{ComplexF64}(C_ref)

    err = norm(Cdist - C_ref_dist, Inf)
    @test err < TOL

    if rank == 0
        println("  ✓ Matrix multiplication with ComplexF64: error = $err")
    end
end

@testset "Non-square Matrix Multiplication" begin
    # A is 6x8, B is 8x10, C should be 6x10
    m, k, n = 6, 8, 10
    I_A = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]
    V_A = Float64.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, k)

    I_B = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
    J_B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    V_B = Float64.(1:length(I_B))
    B = sparse(I_B, J_B, V_B, k, n)

    Adist = SparseMatrixMPI{Float64}(A)
    Bdist = SparseMatrixMPI{Float64}(B)

    Cdist = Adist * Bdist
    C_ref = A * B
    C_ref_dist = SparseMatrixMPI{Float64}(C_ref)

    err = norm(Cdist - C_ref_dist, Inf)
    @test err < TOL

    if rank == 0
        println("  ✓ Non-square matrix multiplication: error = $err")
    end
end

MPI.Finalize()
