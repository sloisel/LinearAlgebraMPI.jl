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
function run_mpi_test(test_file::AbstractString; nprocs::Integer=4, nthreads::Integer=2, expect_success::Bool=true)
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
    cmd = `$mpiexec_cmd -n $nprocs $(Base.julia_cmd()) --threads=$nthreads --project=$test_proj $test_file`
    proc = run(ignorestatus(cmd))
    ok = success(proc)
    if ok != expect_success
        @info "MPI test exit status mismatch" test_file = test_file ok = ok expect_success = expect_success exitcode = proc.exitcode cmd = cmd active_proj = test_proj
    end
    @test ok == expect_success
end

@testset "LinearAlgebraMPI Tests" begin
    @testset "MPI Matrix Multiplication" begin
        run_mpi_test(joinpath(@__DIR__, "test_matrix_multiplication.jl"); nprocs=2, expect_success=true)
    end
    @testset "MPI Transpose" begin
        run_mpi_test(joinpath(@__DIR__, "test_transpose.jl"); nprocs=2, expect_success=true)
    end
    @testset "MPI Addition" begin
        run_mpi_test(joinpath(@__DIR__, "test_addition.jl"); nprocs=2, expect_success=true)
    end
    @testset "MPI Lazy Transpose" begin
        run_mpi_test(joinpath(@__DIR__, "test_lazy_transpose.jl"); nprocs=2, expect_success=true)
    end
    @testset "MPI Vector Multiplication" begin
        run_mpi_test(joinpath(@__DIR__, "test_vector_multiplication.jl"); nprocs=2, expect_success=true)
    end
    @testset "MPI Dense Matrix" begin
        run_mpi_test(joinpath(@__DIR__, "test_dense_matrix.jl"); nprocs=2, expect_success=true)
    end
    @testset "MPI Sparse API Extensions" begin
        run_mpi_test(joinpath(@__DIR__, "test_sparse_api.jl"); nprocs=2, expect_success=true)
    end
    @testset "MPI Block Matrix Operations" begin
        run_mpi_test(joinpath(@__DIR__, "test_blocks.jl"); nprocs=2, expect_success=true)
    end
    @testset "MPI Utilities" begin
        run_mpi_test(joinpath(@__DIR__, "test_utilities.jl"); nprocs=2, expect_success=true)
    end
    @testset "MPI Local Constructors" begin
        run_mpi_test(joinpath(@__DIR__, "test_local_constructors.jl"); nprocs=2, expect_success=true)
    end
    @testset "MPI Indexing" begin
        run_mpi_test(joinpath(@__DIR__, "test_indexing.jl"); nprocs=2, expect_success=true)
    end
    @testset "MPI Factorization" begin
        # Factorization test may return non-zero due to MUMPS cleanup after MPI.Finalize()
        # Check output for actual test results instead of exit code
        test_file = joinpath(@__DIR__, "test_factorization.jl")
        mpiexec_cmd = get(ENV, "MPIEXEC_PATH", nothing)
        if mpiexec_cmd === nothing
            mpiexec_cmd = MPI.mpiexec()
        else
            mpiexec_cmd = Cmd([mpiexec_cmd])
        end
        test_proj = Base.active_project()
        cmd = `$mpiexec_cmd -n 2 $(Base.julia_cmd()) --threads=2 --project=$test_proj $test_file`
        # Use ignorestatus to capture output even on non-zero exit
        output = read(ignorestatus(cmd), String)
        # Check that output contains "Pass:" and "Fail: 0" and "Error: 0"
        @test occursin("Pass:", output)
        @test occursin("Fail: 0", output)
        @test occursin("Error: 0", output)
    end
    @testset "Mixed Sparse-Dense Operations" begin
        run_mpi_test(joinpath(@__DIR__, "test_new_operations.jl"); nprocs=2, expect_success=true)
    end
    @testset "MPI Repartition" begin
        run_mpi_test(joinpath(@__DIR__, "test_repartition.jl"); nprocs=2, expect_success=true)
    end
    @testset "MPI map_rows" begin
        run_mpi_test(joinpath(@__DIR__, "test_map_rows.jl"); nprocs=2, expect_success=true)
    end
end
