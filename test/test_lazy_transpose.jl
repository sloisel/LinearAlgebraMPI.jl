using Test
using MPI
using SparseArrays
using LinearAlgebra: Transpose, norm, opnorm
using LinearAlgebraMPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Tolerance for comparing results
const TOL = 1e-12

@testset "Lazy Transpose Multiplication" begin
    @testset "transpose(A) * transpose(B) = transpose(B * A)" begin
        # C is 8x6, D is 6x8
        # C' is 6x8, D' is 8x6
        # C' * D' should be 6x6, and equal to (D * C)'
        m, n, p = 8, 6, 6
        I_C = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
        J_C = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        V_C = Float64.(1:length(I_C))
        C = sparse(I_C, J_C, V_C, m, n)

        I_D = [1, 2, 3, 4, 5, 6, 1, 2]
        J_D = [1, 2, 3, 4, 5, 6, 7, 8]
        V_D = Float64.(1:length(I_D))
        D = sparse(I_D, J_D, V_D, p, m)

        Cdist = SparseMatrixMPI{Float64}(C)
        Ddist = SparseMatrixMPI{Float64}(D)

        # Compute transpose(C) * transpose(D) using lazy method
        result_lazy = transpose(Cdist) * transpose(Ddist)

        # Materialize the result
        plan = TransposePlan(result_lazy.parent)
        result_dist = execute_plan!(plan, result_lazy.parent)

        # Reference: (D * C)'
        ref = sparse((D * C)')
        ref_dist = SparseMatrixMPI{Float64}(ref)

        err = norm(result_dist - ref_dist, Inf)
        @test err < TOL

        if rank == 0
            println("  ✓ transpose(A) * transpose(B) = transpose(B * A): error = $err")
        end
    end

    @testset "transpose(A) * B (materialize left)" begin
        # A is 8x6, so A' is 6x8
        # B is 8x10, so A' * B is 6x10
        m, n, p = 8, 6, 10
        I_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7]
        J_A = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
        V_A = Float64.(1:length(I_A))
        A = sparse(I_A, J_A, V_A, m, n)

        I_B = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7]
        J_B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2]
        V_B = Float64.(1:length(I_B))
        B = sparse(I_B, J_B, V_B, m, p)

        Adist = SparseMatrixMPI{Float64}(A)
        Bdist = SparseMatrixMPI{Float64}(B)

        result_dist = transpose(Adist) * Bdist
        ref = sparse(A') * B
        ref_dist = SparseMatrixMPI{Float64}(ref)

        err = norm(result_dist - ref_dist, Inf)
        @test err < TOL

        if rank == 0
            println("  ✓ transpose(A) * B: error = $err")
        end
    end

    @testset "A * transpose(B) (materialize right)" begin
        # A is 8x10, B is 6x10, so B' is 10x6
        # A * B' is 8x6
        m, n, p = 8, 10, 6
        I_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
        J_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        V_A = Float64.(1:length(I_A))
        A = sparse(I_A, J_A, V_A, m, n)

        I_B = [1, 2, 3, 4, 5, 6, 1, 2]
        J_B = [1, 2, 3, 4, 5, 6, 7, 8]
        V_B = Float64.(1:length(I_B))
        B = sparse(I_B, J_B, V_B, p, n)

        Adist = SparseMatrixMPI{Float64}(A)
        Bdist = SparseMatrixMPI{Float64}(B)

        result_dist = Adist * transpose(Bdist)
        ref = A * sparse(B')
        ref_dist = SparseMatrixMPI{Float64}(ref)

        err = norm(result_dist - ref_dist, Inf)
        @test err < TOL

        if rank == 0
            println("  ✓ A * transpose(B): error = $err")
        end
    end
end

@testset "Adjoint (conjugate transpose) with ComplexF64" begin
    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8]
    V_A = ComplexF64.(1:length(I_A)) .+ im .* ComplexF64.(length(I_A):-1:1)
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = SparseMatrixMPI{ComplexF64}(A)

    # A' = conj(A)^T
    Aadj = Adist'
    @test Aadj isa Transpose

    # Materialize and compare
    plan = TransposePlan(Aadj.parent)
    result_dist = execute_plan!(plan, Aadj.parent)
    ref = sparse(A')
    ref_dist = SparseMatrixMPI{ComplexF64}(ref)

    err = norm(result_dist - ref_dist, Inf)
    @test err < TOL

    if rank == 0
        println("  ✓ Adjoint with ComplexF64: error = $err")
    end
end

@testset "conj(A) with ComplexF64" begin
    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8]
    V_A = ComplexF64.(1:length(I_A)) .+ im .* ComplexF64.(length(I_A):-1:1)
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = SparseMatrixMPI{ComplexF64}(A)
    result_dist = conj(Adist)
    ref = conj(A)
    ref_dist = SparseMatrixMPI{ComplexF64}(ref)

    err = norm(result_dist - ref_dist, Inf)
    @test err < TOL

    if rank == 0
        println("  ✓ conj(A) with ComplexF64: error = $err")
    end
end

@testset "Scalar multiplication" begin
    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8]
    V_A = Float64.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = SparseMatrixMPI{Float64}(A)

    # Test a * A
    a = 2.5
    result_dist = a * Adist
    ref_dist = SparseMatrixMPI{Float64}(a * A)
    err1 = norm(result_dist - ref_dist, Inf)
    @test err1 < TOL

    # Test A * a
    result_dist = Adist * a
    err2 = norm(result_dist - ref_dist, Inf)
    @test err2 < TOL

    # Test a * transpose(A)
    At = transpose(Adist)
    result_lazy = a * At
    plan = TransposePlan(result_lazy.parent)
    result_dist = execute_plan!(plan, result_lazy.parent)
    ref = sparse((a * A)')
    ref_dist = SparseMatrixMPI{Float64}(ref)
    err3 = norm(result_dist - ref_dist, Inf)
    @test err3 < TOL

    # Test transpose(A) * a
    result_lazy = At * a
    result_dist = execute_plan!(plan, result_lazy.parent)
    err4 = norm(result_dist - ref_dist, Inf)
    @test err4 < TOL

    if rank == 0
        println("  ✓ Scalar multiplication: errors = $err1, $err2, $err3, $err4")
    end
end

@testset "Scalar multiplication with ComplexF64" begin
    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8]
    V_A = ComplexF64.(1:length(I_A)) .+ im .* ComplexF64.(length(I_A):-1:1)
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = SparseMatrixMPI{ComplexF64}(A)
    a = 2.0 + 3.0im

    # Test a * A
    result_dist = a * Adist
    ref_dist = SparseMatrixMPI{ComplexF64}(a * A)
    err1 = norm(result_dist - ref_dist, Inf)
    @test err1 < TOL

    # Test A * a
    result_dist = Adist * a
    err2 = norm(result_dist - ref_dist, Inf)
    @test err2 < TOL

    # Test a * A' (adjoint)
    # A' = transpose(conj(A)), so a * A' = a * transpose(conj(A)) = transpose(a * conj(A))
    Aadj = Adist'
    result_lazy = a * Aadj
    plan = TransposePlan(result_lazy.parent)
    result_dist = execute_plan!(plan, result_lazy.parent)
    ref = sparse(transpose(a * conj(A)))
    ref_dist = SparseMatrixMPI{ComplexF64}(ref)
    err3 = norm(result_dist - ref_dist, Inf)
    @test err3 < TOL

    if rank == 0
        println("  ✓ Scalar multiplication with ComplexF64: errors = $err1, $err2, $err3")
    end
end

@testset "norm" begin
    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8]
    V_A = Float64.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = SparseMatrixMPI{Float64}(A)

    err1 = abs(norm(Adist) - norm(A))
    err2 = abs(norm(Adist, 1) - norm(A, 1))
    err3 = abs(norm(Adist, Inf) - norm(A, Inf))
    err4 = abs(norm(Adist, 3) - norm(A, 3))

    @test err1 < TOL
    @test err2 < TOL
    @test err3 < TOL
    @test err4 < TOL

    if rank == 0
        println("  ✓ norm: errors = $err1, $err2, $err3, $err4")
    end
end

@testset "norm with ComplexF64" begin
    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8]
    V_A = ComplexF64.(1:length(I_A)) .+ im .* ComplexF64.(length(I_A):-1:1)
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = SparseMatrixMPI{ComplexF64}(A)

    err1 = abs(norm(Adist) - norm(A))
    err2 = abs(norm(Adist, 1) - norm(A, 1))
    err3 = abs(norm(Adist, Inf) - norm(A, Inf))

    @test err1 < TOL
    @test err2 < TOL
    @test err3 < TOL

    if rank == 0
        println("  ✓ norm with ComplexF64: errors = $err1, $err2, $err3")
    end
end

@testset "opnorm" begin
    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3, 2, 4]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
    V_A = Float64.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = SparseMatrixMPI{Float64}(A)

    err1 = abs(opnorm(Adist, 1) - opnorm(A, 1))
    err2 = abs(opnorm(Adist, Inf) - opnorm(A, Inf))

    @test err1 < TOL
    @test err2 < TOL

    if rank == 0
        println("  ✓ opnorm: errors = $err1, $err2")
    end
end

@testset "opnorm with ComplexF64" begin
    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3, 2, 4]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
    V_A = ComplexF64.(1:length(I_A)) .+ im .* ComplexF64.(length(I_A):-1:1)
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = SparseMatrixMPI{ComplexF64}(A)

    err1 = abs(opnorm(Adist, 1) - opnorm(A, 1))
    err2 = abs(opnorm(Adist, Inf) - opnorm(A, Inf))

    @test err1 < TOL
    @test err2 < TOL

    if rank == 0
        println("  ✓ opnorm with ComplexF64: errors = $err1, $err2")
    end
end

MPI.Finalize()
