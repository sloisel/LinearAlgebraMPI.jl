# MPI test for addition and subtraction
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

@testset "Matrix Addition" begin
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

    if rank == 0
        println("  ✓ Matrix addition: error = $err")
    end
end

@testset "Matrix Addition with ComplexF64" begin
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

    Cdist = Adist + Bdist
    C_ref = A + B
    C_ref_dist = SparseMatrixMPI{ComplexF64}(C_ref)

    err = norm(Cdist - C_ref_dist, Inf)
    @test err < TOL

    if rank == 0
        println("  ✓ Matrix addition with ComplexF64: error = $err")
    end
end

@testset "Matrix Subtraction" begin
    n = 8
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = [3.0*ones(Float64, n); -0.7*ones(n-1); -0.7*ones(n-1)]
    A = sparse(I_A, J_A, V_A, n, n)

    I_B = [1:n; 1:n-1; 2:n]
    J_B = [1:n; 2:n; 1:n-1]
    V_B = [1.0*ones(Float64, n); 0.3*ones(n-1); 0.3*ones(n-1)]
    B = sparse(I_B, J_B, V_B, n, n)

    Adist = SparseMatrixMPI{Float64}(A)
    Bdist = SparseMatrixMPI{Float64}(B)

    Cdist = Adist - Bdist
    C_ref = A - B
    C_ref_dist = SparseMatrixMPI{Float64}(C_ref)

    err = norm(Cdist - C_ref_dist, Inf)
    @test err < TOL

    if rank == 0
        println("  ✓ Matrix subtraction: error = $err")
    end
end

@testset "Different Sparsity Patterns" begin
    n = 8
    # Matrix A: upper triangular entries
    I_A = [1, 1, 2, 3, 4, 5, 6, 7, 8]
    J_A = [1, 2, 2, 3, 4, 5, 6, 7, 8]
    V_A = Float64.(1:9)
    A = sparse(I_A, J_A, V_A, n, n)

    # Matrix B: lower triangular entries
    I_B = [1, 2, 2, 3, 4, 5, 6, 7, 8]
    J_B = [1, 1, 2, 3, 4, 5, 6, 7, 8]
    V_B = Float64.(9:-1:1)
    B = sparse(I_B, J_B, V_B, n, n)

    Adist = SparseMatrixMPI{Float64}(A)
    Bdist = SparseMatrixMPI{Float64}(B)

    Cdist = Adist + Bdist
    C_ref = A + B
    C_ref_dist = SparseMatrixMPI{Float64}(C_ref)

    err = norm(Cdist - C_ref_dist, Inf)
    @test err < TOL

    if rank == 0
        println("  ✓ Different sparsity patterns: error = $err")
    end
end

MPI.Finalize()
