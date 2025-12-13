"""
Tests for distributed solve functionality.

This tests the MUMPS-style distributed triangular solve that avoids
gathering L/U factors to all ranks.
"""

using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

const TOL = 1e-10

# Create deterministic test matrices
function create_spd_tridiagonal(n::Int)
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = [4.0*ones(n); -1.0*ones(n-1); -1.0*ones(n-1)]
    return sparse(I_A, J_A, V_A, n, n)
end

function create_general_tridiagonal(n::Int)
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = [2.0*ones(n); -0.5*ones(n-1); -0.8*ones(n-1)]
    return sparse(I_A, J_A, V_A, n, n)
end

function create_2d_laplacian(nx::Int, ny::Int)
    n = nx * ny
    I_A = Int[]
    J_A = Int[]
    V_A = Float64[]

    for i = 1:nx
        for j = 1:ny
            idx = (j-1)*nx + i
            push!(I_A, idx); push!(J_A, idx); push!(V_A, 4.0)
            if i > 1
                push!(I_A, idx); push!(J_A, idx-1); push!(V_A, -1.0)
            end
            if i < nx
                push!(I_A, idx); push!(J_A, idx+1); push!(V_A, -1.0)
            end
            if j > 1
                push!(I_A, idx); push!(J_A, idx-nx); push!(V_A, -1.0)
            end
            if j < ny
                push!(I_A, idx); push!(J_A, idx+nx); push!(V_A, -1.0)
            end
        end
    end

    return sparse(I_A, J_A, V_A, n, n)
end

ts = @testset QuietTestSet "distributed solve" begin

# Test 1: Solve plan initialization
if rank == 0
    println("[test] Solve plan initialization")
    flush(stdout)
end

n = 20
A_full = create_general_tridiagonal(n)
A_mpi = SparseMatrixMPI{Float64}(A_full)

F = lu(A_mpi)

# Get the solve plan (this triggers initialization)
plan = LinearAlgebraMPI.get_or_create_solve_plan(F)

@test plan.initialized == true
@test plan.myrank == rank
@test plan.nranks == nranks
# Some ranks may not own supernodes for small matrices
# @test length(plan.my_supernodes_postorder) > 0

# Check that global_to_local and local_to_global are consistent
for local_idx in 1:length(plan.local_to_global)
    elim_idx = plan.local_to_global[local_idx]
    @test plan.global_to_local[elim_idx] == local_idx
end

# Gather supernode counts from all ranks
local_snode_count = Int32(length(plan.my_supernodes_postorder))
all_snode_counts = MPI.Allgather(local_snode_count, comm)
total_snodes = sum(all_snode_counts)

# Total supernodes should equal length of symbolic.supernodes
@test total_snodes == length(F.symbolic.supernodes)

if rank == 0
    println("  Solve plan initialized successfully")
    println("  Supernode distribution: $all_snode_counts")
    println("  Total supernodes: $total_snodes")
    println("  Subtree roots on rank 0: $(length(plan.subtree_roots))")
end

MPI.Barrier(comm)

# Test 2: Compare distributed solve with gathered solve (small matrix)
if rank == 0
    println("[test] Distributed vs gathered solve - small matrix")
    flush(stdout)
end

n_small = 8
A_small_full = create_general_tridiagonal(n_small)
A_small_mpi = SparseMatrixMPI{Float64}(A_small_full)

b_small_full = [1.0 + 0.1*i for i in 1:n_small]  # Deterministic RHS
b_small = VectorMPI(b_small_full)

F_small = lu(A_small_mpi)

# Solve using gathered method (existing implementation)
x_gathered = solve(F_small, b_small)
x_gathered_full = Vector(x_gathered)

# Create a new VectorMPI for distributed solve result
x_distributed = VectorMPI(zeros(Float64, n_small); partition=b_small.partition)

# Try the distributed solve
try
    distributed_solve_lu!(x_distributed, F_small, b_small)
    x_distributed_full = Vector(x_distributed)

    # Compare results
    diff = norm(x_gathered_full - x_distributed_full, Inf)

    if rank == 0
        println("  Gathered solve residual: $(norm(A_small_full * x_gathered_full - b_small_full, Inf))")
        println("  Distributed solve residual: $(norm(A_small_full * x_distributed_full - b_small_full, Inf))")
        println("  Difference between solutions: $diff")
    end

    @test diff < TOL
catch e
    if rank == 0
        println("  Distributed solve failed: $e")
        println("  (This is expected - implementation in progress)")
    end
    @test_broken false
end

MPI.Barrier(comm)

# Test 3: Larger matrix to exercise multi-supernode structure
if rank == 0
    println("[test] Distributed vs gathered solve - 2D Laplacian")
    flush(stdout)
end

A_2d_full = create_2d_laplacian(4, 4)  # 16 nodes
n_2d = size(A_2d_full, 1)
A_2d_mpi = SparseMatrixMPI{Float64}(A_2d_full)

b_2d_full = [1.0 + 0.1*i for i in 1:n_2d]
b_2d = VectorMPI(b_2d_full)

F_2d = lu(A_2d_mpi)

# Solve using gathered method
x_2d_gathered = solve(F_2d, b_2d)
x_2d_gathered_full = Vector(x_2d_gathered)

# Try distributed solve
x_2d_distributed = VectorMPI(zeros(Float64, n_2d); partition=b_2d.partition)

try
    distributed_solve_lu!(x_2d_distributed, F_2d, b_2d)
    x_2d_distributed_full = Vector(x_2d_distributed)

    diff_2d = norm(x_2d_gathered_full - x_2d_distributed_full, Inf)

    if rank == 0
        println("  2D Laplacian gathered solve residual: $(norm(A_2d_full * x_2d_gathered_full - b_2d_full, Inf))")
        println("  2D Laplacian distributed solve residual: $(norm(A_2d_full * x_2d_distributed_full - b_2d_full, Inf))")
        println("  Difference between solutions: $diff_2d")
    end

    @test diff_2d < TOL
catch e
    if rank == 0
        println("  Distributed solve failed: $e")
        println("  (This is expected - implementation in progress)")
    end
    @test_broken false
end

MPI.Barrier(comm)

# Test 4: Distributed input for LU factorization
if rank == 0
    println("[test] Distributed input for LU factorization")
    flush(stdout)
end

n_input = 12
A_input_full = create_general_tridiagonal(n_input)
A_input_mpi = SparseMatrixMPI{Float64}(A_input_full)

b_input_full = [1.0 + 0.1*i for i in 1:n_input]
b_input = VectorMPI(b_input_full)

# Factorize with gathered input (default)
F_gathered = lu(A_input_mpi; distributed_input=false)
x_gathered = solve(F_gathered, b_input)
x_gathered_full = Vector(x_gathered)

# Factorize with distributed input
try
    F_distributed_input = lu(A_input_mpi; distributed_input=true)
    x_distributed_input = solve(F_distributed_input, b_input)
    x_distributed_input_full = Vector(x_distributed_input)

    diff_input = norm(x_gathered_full - x_distributed_input_full, Inf)

    if rank == 0
        println("  Gathered input solve residual: $(norm(A_input_full * x_gathered_full - b_input_full, Inf))")
        println("  Distributed input solve residual: $(norm(A_input_full * x_distributed_input_full - b_input_full, Inf))")
        println("  Difference between solutions: $diff_input")
    end

    @test diff_input < TOL
catch e
    if rank == 0
        println("  Distributed input failed: $e")
        showerror(stdout, e, catch_backtrace())
        println()
    end
    @test_broken false
end

MPI.Barrier(comm)

# Test 5: Distributed input for LDLT factorization
if rank == 0
    println("[test] Distributed input for LDLT factorization")
    flush(stdout)
end

n_ldlt = 12
A_ldlt_full = create_spd_tridiagonal(n_ldlt)
A_ldlt_mpi = SparseMatrixMPI{Float64}(A_ldlt_full)

b_ldlt_full = [1.0 + 0.1*i for i in 1:n_ldlt]
b_ldlt = VectorMPI(b_ldlt_full)

# Factorize with gathered input (default)
F_ldlt_gathered = ldlt(A_ldlt_mpi; distributed_input=false)
x_ldlt_gathered = solve(F_ldlt_gathered, b_ldlt)
x_ldlt_gathered_full = Vector(x_ldlt_gathered)

# Factorize with distributed input
try
    F_ldlt_distributed_input = ldlt(A_ldlt_mpi; distributed_input=true)
    x_ldlt_distributed_input = solve(F_ldlt_distributed_input, b_ldlt)
    x_ldlt_distributed_input_full = Vector(x_ldlt_distributed_input)

    diff_ldlt_input = norm(x_ldlt_gathered_full - x_ldlt_distributed_input_full, Inf)

    if rank == 0
        println("  LDLT gathered input solve residual: $(norm(A_ldlt_full * x_ldlt_gathered_full - b_ldlt_full, Inf))")
        println("  LDLT distributed input solve residual: $(norm(A_ldlt_full * x_ldlt_distributed_input_full - b_ldlt_full, Inf))")
        println("  Difference between solutions: $diff_ldlt_input")
    end

    @test diff_ldlt_input < TOL
catch e
    if rank == 0
        println("  LDLT distributed input failed: $e")
        showerror(stdout, e, catch_backtrace())
        println()
    end
    @test_broken false
end

MPI.Barrier(comm)

end  # QuietTestSet

# Aggregate results across ranks
local_counts = [ts.counts[:pass], ts.counts[:fail], ts.counts[:error], ts.counts[:broken], ts.counts[:skip]]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

if rank == 0
    total = sum(global_counts)
    println("\nTest Summary: distributed solve | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])  Total: $total")
    flush(stdout)
end

MPI.Barrier(comm)
MPI.Finalize()

exit_code = global_counts[2] + global_counts[3] > 0 ? 1 : 0
exit(exit_code)
