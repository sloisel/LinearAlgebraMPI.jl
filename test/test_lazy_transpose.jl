# MPI test for lazy transpose operations
# This file is executed under mpiexec by runtests.jl
# Parameterized over scalar types and backends (CPU and GPU)

# Check Metal availability BEFORE loading MPI
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch e
    false
end

using Test
using MPI
using SparseArrays
using LinearAlgebra: Transpose, norm, opnorm
using LinearAlgebraMPI

MPI.Init()

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

comm = MPI.COMM_WORLD

ts = @testset QuietTestSet "Lazy Transpose" begin

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)

    println(io0(), "[test] transpose(A) * transpose(B) = transpose(B * A) ($T, $backend_name)")

    # C is 8x6, D is 6x8
    # C' is 6x8, D' is 8x6
    # C' * D' should be 6x6, and equal to (D * C)'
    m, n, p = 8, 6, 6
    I_C = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
    J_C = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
    V_C = T <: Complex ? T.(1:length(I_C)) .+ im .* T.(length(I_C):-1:1) : T.(1:length(I_C))
    C = sparse(I_C, J_C, V_C, m, n)

    I_D = [1, 2, 3, 4, 5, 6, 1, 2]
    J_D = [1, 2, 3, 4, 5, 6, 7, 8]
    V_D = T <: Complex ? T.(1:length(I_D)) .+ im .* T.(length(I_D):-1:1) : T.(1:length(I_D))
    D = sparse(I_D, J_D, V_D, p, m)

    Cdist = to_backend(SparseMatrixMPI{T}(C))
    Ddist = to_backend(SparseMatrixMPI{T}(D))

    # Compute transpose(C) * transpose(D) using lazy method
    result_lazy = transpose(Cdist) * transpose(Ddist)

    # Materialize the result (internal API)
    plan = LinearAlgebraMPI.TransposePlan(result_lazy.parent)
    result_dist = LinearAlgebraMPI.execute_plan!(plan, result_lazy.parent)

    # Reference: transpose(D * C)
    ref = sparse(transpose(D * C))
    ref_dist = to_backend(SparseMatrixMPI{T}(ref))

    result_dist_cpu = TestUtils.to_cpu(result_dist)
    ref_dist_cpu = TestUtils.to_cpu(ref_dist)
    err = norm(result_dist_cpu - ref_dist_cpu, Inf)
    @test err < TOL


    println(io0(), "[test] transpose(A) * B materialize left ($T, $backend_name)")

    # A is 8x6, so A' is 6x8
    # B is 8x10, so A' * B is 6x10
    m, n, p = 8, 6, 10
    I_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7]
    J_A = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
    V_A = T <: Complex ? T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1) : T.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    I_B = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7]
    J_B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2]
    V_B = T <: Complex ? T.(1:length(I_B)) .+ im .* T.(length(I_B):-1:1) : T.(1:length(I_B))
    B = sparse(I_B, J_B, V_B, m, p)

    Adist = to_backend(SparseMatrixMPI{T}(A))
    Bdist = to_backend(SparseMatrixMPI{T}(B))

    result_dist = transpose(Adist) * Bdist
    ref = sparse(transpose(A)) * B
    ref_dist = to_backend(SparseMatrixMPI{T}(ref))

    result_dist_cpu = TestUtils.to_cpu(result_dist)
    ref_dist_cpu = TestUtils.to_cpu(ref_dist)
    err = norm(result_dist_cpu - ref_dist_cpu, Inf)
    @test err < TOL


    println(io0(), "[test] A * transpose(B) materialize right ($T, $backend_name)")

    # A is 8x10, B is 6x10, so B' is 10x6
    # A * B' is 8x6
    m, n, p = 8, 10, 6
    I_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    V_A = T <: Complex ? T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1) : T.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    I_B = [1, 2, 3, 4, 5, 6, 1, 2]
    J_B = [1, 2, 3, 4, 5, 6, 7, 8]
    V_B = T <: Complex ? T.(1:length(I_B)) .+ im .* T.(length(I_B):-1:1) : T.(1:length(I_B))
    B = sparse(I_B, J_B, V_B, p, n)

    Adist = to_backend(SparseMatrixMPI{T}(A))
    Bdist = to_backend(SparseMatrixMPI{T}(B))

    result_dist = Adist * transpose(Bdist)
    ref = A * sparse(transpose(B))
    ref_dist = to_backend(SparseMatrixMPI{T}(ref))

    result_dist_cpu = TestUtils.to_cpu(result_dist)
    ref_dist_cpu = TestUtils.to_cpu(ref_dist)
    err = norm(result_dist_cpu - ref_dist_cpu, Inf)
    @test err < TOL


    if T <: Complex
        println(io0(), "[test] Adjoint conjugate transpose ($T, $backend_name)")

        m, n = 6, 8
        I_A = [1, 2, 3, 4, 5, 6, 1, 3]
        J_A = [1, 2, 3, 4, 5, 6, 7, 8]
        V_A = T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1)
        A = sparse(I_A, J_A, V_A, m, n)

        Adist = to_backend(SparseMatrixMPI{T}(A))

        # A' = conj(A)^T
        Aadj = Adist'
        @test Aadj isa Transpose

        # Materialize and compare (internal API)
        plan = LinearAlgebraMPI.TransposePlan(Aadj.parent)
        result_dist = LinearAlgebraMPI.execute_plan!(plan, Aadj.parent)
        ref = sparse(A')
        ref_dist = to_backend(SparseMatrixMPI{T}(ref))

        result_dist_cpu = TestUtils.to_cpu(result_dist)
        ref_dist_cpu = TestUtils.to_cpu(ref_dist)
        err = norm(result_dist_cpu - ref_dist_cpu, Inf)
        @test err < TOL


        println(io0(), "[test] conj(A) ($T, $backend_name)")

        m, n = 6, 8
        I_A = [1, 2, 3, 4, 5, 6, 1, 3]
        J_A = [1, 2, 3, 4, 5, 6, 7, 8]
        V_A = T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1)
        A = sparse(I_A, J_A, V_A, m, n)

        Adist = to_backend(SparseMatrixMPI{T}(A))
        result_dist = conj(Adist)
        ref = conj(A)
        ref_dist = to_backend(SparseMatrixMPI{T}(ref))

        result_dist_cpu = TestUtils.to_cpu(result_dist)
        ref_dist_cpu = TestUtils.to_cpu(ref_dist)
        err = norm(result_dist_cpu - ref_dist_cpu, Inf)
        @test err < TOL
    end


    println(io0(), "[test] Scalar multiplication ($T, $backend_name)")

    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8]
    V_A = T <: Complex ? T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1) : T.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = to_backend(SparseMatrixMPI{T}(A))

    # Test a * A
    a = T <: Complex ? T(2.5 + 0.5im) : T(2.5)
    result_dist = a * Adist
    ref_dist = to_backend(SparseMatrixMPI{T}(a * A))
    result_dist_cpu = TestUtils.to_cpu(result_dist)
    ref_dist_cpu = TestUtils.to_cpu(ref_dist)
    err1 = norm(result_dist_cpu - ref_dist_cpu, Inf)
    @test err1 < TOL

    # Test A * a
    result_dist = Adist * a
    result_dist_cpu = TestUtils.to_cpu(result_dist)
    err2 = norm(result_dist_cpu - ref_dist_cpu, Inf)
    @test err2 < TOL

    # Test a * transpose(A) (internal API)
    At = transpose(Adist)
    result_lazy = a * At
    plan = LinearAlgebraMPI.TransposePlan(result_lazy.parent)
    result_dist = LinearAlgebraMPI.execute_plan!(plan, result_lazy.parent)
    ref = sparse(transpose(a * A))
    ref_dist = to_backend(SparseMatrixMPI{T}(ref))
    result_dist_cpu = TestUtils.to_cpu(result_dist)
    ref_dist_cpu = TestUtils.to_cpu(ref_dist)
    err3 = norm(result_dist_cpu - ref_dist_cpu, Inf)
    @test err3 < TOL

    # Test transpose(A) * a
    result_lazy = At * a
    result_dist = LinearAlgebraMPI.execute_plan!(plan, result_lazy.parent)
    result_dist_cpu = TestUtils.to_cpu(result_dist)
    err4 = norm(result_dist_cpu - ref_dist_cpu, Inf)
    @test err4 < TOL


    println(io0(), "[test] norm ($T, $backend_name)")

    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8]
    V_A = T <: Complex ? T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1) : T.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = to_backend(SparseMatrixMPI{T}(A))
    Adist_cpu = TestUtils.to_cpu(Adist)

    err1 = abs(norm(Adist_cpu) - norm(A))
    err2 = abs(norm(Adist_cpu, 1) - norm(A, 1))
    err3 = abs(norm(Adist_cpu, Inf) - norm(A, Inf))
    err4 = abs(norm(Adist_cpu, 3) - norm(A, 3))

    @test err1 < TOL
    @test err2 < TOL
    @test err3 < TOL
    @test err4 < TOL


    println(io0(), "[test] opnorm ($T, $backend_name)")

    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 3, 2, 4]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
    V_A = T <: Complex ? T.(1:length(I_A)) .+ im .* T.(length(I_A):-1:1) : T.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    Adist = to_backend(SparseMatrixMPI{T}(A))
    Adist_cpu = TestUtils.to_cpu(Adist)

    err1 = abs(opnorm(Adist_cpu, 1) - opnorm(A, 1))
    err2 = abs(opnorm(Adist_cpu, Inf) - opnorm(A, Inf))

    @test err1 < TOL
    @test err2 < TOL

end  # for (T, to_backend, backend_name)

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

println(io0(), "Test Summary: Lazy Transpose | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
