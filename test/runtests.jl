using Test
using MPI

# Force precompilation of test dependencies
try
    @eval using LinearAlgebraMPI
    @eval using SparseArrays
    @eval using LinearAlgebra
    println("Precompilation complete for test environment")
    flush(stdout)
catch err
    @warn "Precompile step hit an error; tests may still proceed" err
end

# Helper to run a test file under mpiexec with a fixed project and check exit status
function run_mpi_test(test_file::AbstractString; nprocs::Integer=4, expect_success::Bool=true)
    # Allow overriding mpiexec via environment variable (useful for CI with system MPI)
    mpiexec_cmd = get(ENV, "MPIEXEC_PATH", nothing)
    if mpiexec_cmd === nothing
        mpiexec_cmd = MPI.mpiexec()
    else
        mpiexec_cmd = Cmd([mpiexec_cmd])
    end
    # Use the active test environment project (which has already been precompiled above)
    # This ensures LocalPreferences.toml is honored and avoids recompilation under mpiexec
    test_proj = Base.active_project()
    cmd = `$mpiexec_cmd -n $nprocs $(Base.julia_cmd()) --project=$test_proj $test_file`
    proc = run(ignorestatus(cmd))
    ok = success(proc)
    if ok != expect_success
        @info "MPI test exit status mismatch" test_file = test_file ok = ok expect_success = expect_success exitcode = proc.exitcode cmd = cmd active_proj = test_proj
    end
    @test ok == expect_success
end

@testset "LinearAlgebraMPI Tests" begin
    @testset "MPI Matrix Multiplication" begin
        run_mpi_test(joinpath(@__DIR__, "test_matrix_multiplication.jl"); nprocs=4, expect_success=true)
    end
    @testset "MPI Transpose" begin
        run_mpi_test(joinpath(@__DIR__, "test_transpose.jl"); nprocs=4, expect_success=true)
    end
    @testset "MPI Addition" begin
        run_mpi_test(joinpath(@__DIR__, "test_addition.jl"); nprocs=4, expect_success=true)
    end
    @testset "MPI Lazy Transpose" begin
        run_mpi_test(joinpath(@__DIR__, "test_lazy_transpose.jl"); nprocs=4, expect_success=true)
    end
    @testset "MPI Vector Multiplication" begin
        run_mpi_test(joinpath(@__DIR__, "test_vector_multiplication.jl"); nprocs=4, expect_success=true)
    end
    @testset "MPI Dense Matrix" begin
        run_mpi_test(joinpath(@__DIR__, "test_dense_matrix.jl"); nprocs=4, expect_success=true)
    end
end
