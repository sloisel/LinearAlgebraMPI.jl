# MPI test for map_rows
# This file is executed under mpiexec by runtests.jl
# Parameterized over scalar types and backends (CPU and GPU)

# Check Metal availability BEFORE loading MPI
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch e
    false
end

using MPI
MPI.Init()

using LinearAlgebraMPI
using LinearAlgebra
using StaticArrays
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

ts = @testset QuietTestSet "map_rows" begin

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)

    # Skip complex for some tests that use real-only operations
    is_complex = T <: Complex
    Treal = real(T)

    println(io0(), "[test] VectorMPI -> scalar ($T, $backend_name)")

    v = to_backend(VectorMPI(T.([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])))
    result = map_rows(r -> r^2, v)
    expected = T.([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0])
    @test norm(Vector(result) - expected) < TOL


    println(io0(), "[test] Two VectorMPIs -> scalar ($T, $backend_name)")

    u = to_backend(VectorMPI(T.([1.0, 2.0, 3.0, 4.0])))
    v2 = to_backend(VectorMPI(T.([4.0, 3.0, 2.0, 1.0])))
    result2 = map_rows((a, b) -> a * b, u, v2)
    expected2 = T.([4.0, 6.0, 6.0, 4.0])
    @test norm(Vector(result2) - expected2) < TOL


    println(io0(), "[test] MatrixMPI -> scalar row norms ($T, $backend_name)")

    A = to_backend(MatrixMPI(T.([1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0; 1.0 1.0 1.0])))
    result3 = map_rows(r -> norm(r), A)
    expected3 = Treal.([1.0, 2.0, 3.0, sqrt(3.0)])
    @test norm(Vector(result3) - expected3) < TOL


    println(io0(), "[test] f returns SVector -> MatrixMPI ($T, $backend_name)")

    A4 = to_backend(MatrixMPI(T.([1.0 2.0; 3.0 4.0; 5.0 6.0])))
    result4 = map_rows(r -> SVector(T(1), T(2), T(3)), A4)
    expected4 = T.([1 2 3; 1 2 3; 1 2 3])
    @test norm(Matrix(result4) - expected4) < TOL
    @test size(result4) == (3, 3)


    println(io0(), "[test] f returns SVector from row ($T, $backend_name)")

    A5 = to_backend(MatrixMPI(T.([1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0])))
    result5 = map_rows(r -> SVector(sum(r), prod(r)), A5)
    expected5 = T.([3.0 2.0; 7.0 12.0; 11.0 30.0; 15.0 56.0])
    @test norm(Matrix(result5) - expected5) < TOL
    @test size(result5) == (4, 2)


    println(io0(), "[test] MatrixMPI + VectorMPI -> scalar ($T, $backend_name)")

    A6 = to_backend(MatrixMPI(T.([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0; 10.0 11.0 12.0])))
    w = to_backend(VectorMPI(T.([1.0, 2.0, 3.0, 4.0])))
    # Compute weighted row sums
    result6 = map_rows((row, wi) -> sum(row) * wi, A6, w)
    expected6 = T.([6.0 * 1.0, 15.0 * 2.0, 24.0 * 3.0, 33.0 * 4.0])
    @test norm(Vector(result6) - expected6) < TOL


    println(io0(), "[test] Two MatrixMPIs -> scalar ($T, $backend_name)")

    A7 = to_backend(MatrixMPI(T.([1.0 2.0; 3.0 4.0])))
    B7 = to_backend(MatrixMPI(T.([10.0 20.0; 30.0 40.0])))
    result7 = map_rows((a, b) -> dot(a, b), A7, B7)
    # Row 1: [1,2] · [10,20] = 10 + 40 = 50
    # Row 2: [3,4] · [30,40] = 90 + 160 = 250
    expected7 = Treal.([50.0, 250.0])
    @test norm(Vector(result7) - expected7) < TOL


    println(io0(), "[test] Different partitions ($T, $backend_name)")

    u8 = to_backend(VectorMPI(T.([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])))
    v8 = to_backend(VectorMPI(T.([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])))
    result8 = map_rows((a, b) -> a + b, u8, v8)
    expected8 = T.([11.0, 22.0, 33.0, 44.0, 55.0, 66.0])
    @test norm(Vector(result8) - expected8) < TOL


    if is_complex
        println(io0(), "[test] Complex numbers ($T, $backend_name)")

        v9 = to_backend(VectorMPI(T[1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im]))
        result9 = map_rows(r -> abs2(r), v9)
        expected9 = Treal.([5.0, 25.0, 61.0, 113.0])
        @test norm(Vector(result9) - expected9) < TOL


        println(io0(), "[test] Complex matrix -> SVector ($T, $backend_name)")

        A10 = to_backend(MatrixMPI(T[1.0+1.0im 2.0-1.0im; 3.0+2.0im 4.0-2.0im]))
        result10 = map_rows(r -> SVector(real(r[1]), imag(r[2])), A10)
        expected10 = Treal.([1.0 -1.0; 3.0 -2.0])
        @test norm(Matrix(result10) - expected10) < TOL
    end


    println(io0(), "[test] Identity SVector transform ($T, $backend_name)")

    A11 = to_backend(MatrixMPI(T.([1.0 2.0 3.0; 4.0 5.0 6.0])))
    result11 = map_rows(r -> r, A11)  # r is already SVector
    @test norm(Matrix(result11) - Matrix(A11)) < TOL


    println(io0(), "[test] Row max ($T, $backend_name)")

    if !is_complex  # maximum only works on real types
        A12 = to_backend(MatrixMPI(T.([1.0 5.0 3.0; 7.0 2.0 4.0; 3.0 3.0 9.0])))
        result12 = map_rows(r -> maximum(r), A12)
        expected12 = T.([5.0, 7.0, 9.0])
        @test norm(Vector(result12) - expected12) < TOL
    end

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

println(io0(), "Test Summary: map_rows | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
