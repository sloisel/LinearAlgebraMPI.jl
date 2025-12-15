"""
Tests for distributed LU and LDLT factorization.
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
    # Symmetric positive definite tridiagonal matrix
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = [4.0*ones(n); -1.0*ones(n-1); -1.0*ones(n-1)]
    return sparse(I_A, J_A, V_A, n, n)
end

function create_general_tridiagonal(n::Int)
    # General (unsymmetric) tridiagonal matrix
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = [2.0*ones(n); -0.5*ones(n-1); -0.8*ones(n-1)]
    return sparse(I_A, J_A, V_A, n, n)
end

function create_symmetric_indefinite(n::Int)
    # Symmetric indefinite matrix
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    # Alternating signs on diagonal
    diag_vals = [(-1.0)^i * 2.0 for i in 1:n]
    V_A = [diag_vals; -1.0*ones(n-1); -1.0*ones(n-1)]
    return sparse(I_A, J_A, V_A, n, n)
end

function create_2d_laplacian(nx::Int, ny::Int)
    # 2D Laplacian on nx x ny grid
    n = nx * ny
    I_A = Int[]
    J_A = Int[]
    V_A = Float64[]

    for i = 1:nx
        for j = 1:ny
            idx = (j-1)*nx + i
            # Diagonal
            push!(I_A, idx)
            push!(J_A, idx)
            push!(V_A, 4.0)
            # Left neighbor
            if i > 1
                push!(I_A, idx)
                push!(J_A, idx-1)
                push!(V_A, -1.0)
            end
            # Right neighbor
            if i < nx
                push!(I_A, idx)
                push!(J_A, idx+1)
                push!(V_A, -1.0)
            end
            # Bottom neighbor
            if j > 1
                push!(I_A, idx)
                push!(J_A, idx-nx)
                push!(V_A, -1.0)
            end
            # Top neighbor
            if j < ny
                push!(I_A, idx)
                push!(J_A, idx+nx)
                push!(V_A, -1.0)
            end
        end
    end

    return sparse(I_A, J_A, V_A, n, n)
end

function create_complex_symmetric(n::Int)
    # Complex symmetric (NOT Hermitian) matrix
    # A = A^T but A != A' (adjoint)
    I_A = Int[]
    J_A = Int[]
    V_A = ComplexF64[]

    # Complex diagonal
    for i = 1:n
        push!(I_A, i)
        push!(J_A, i)
        push!(V_A, 3.0 + 1.0im)  # Complex diagonal
    end

    # Complex symmetric off-diagonal (not Hermitian: A[i,j] = A[j,i], not conj)
    for i = 1:n-1
        val = -0.5 + 0.2im
        push!(I_A, i+1)
        push!(J_A, i)
        push!(V_A, val)
        push!(I_A, i)
        push!(J_A, i+1)
        push!(V_A, val)  # Same value, not conjugate
    end

    return sparse(I_A, J_A, V_A, n, n)
end

ts = @testset QuietTestSet "Distributed Factorization Tests" begin

# Test 1: LU factorization of small matrix
println(io0(), "[test] LU factorization - small matrix")

n = 8
A_full = create_general_tridiagonal(n)
A = SparseMatrixMPI{Float64}(A_full)

F = lu(A)
@test size(F) == (n, n)

b_full = ones(n)
b = VectorMPI(b_full)
x = F \ b

x_full = Vector(x)
residual = A_full * x_full - b_full
err = norm(residual, Inf)

println(io0(), "  LU solve residual: $err")
@test err < TOL


# Test 2: LDLT factorization of SPD matrix
println(io0(), "[test] LDLT factorization - SPD matrix")

n = 10
A_full = create_spd_tridiagonal(n)
A = SparseMatrixMPI{Float64}(A_full)

F = ldlt(A)
@test size(F) == (n, n)

b_full = ones(n)
b = VectorMPI(b_full)
x = F \ b

x_full = Vector(x)
residual = A_full * x_full - b_full
err = norm(residual, Inf)

println(io0(), "  LDLT solve residual (SPD): $err")
@test err < TOL


# Test 3: LDLT with symmetric indefinite matrix
println(io0(), "[test] LDLT factorization - indefinite matrix")

n = 8
A_full = create_symmetric_indefinite(n)
A = SparseMatrixMPI{Float64}(A_full)

F = ldlt(A)

b_full = collect(1.0:n)
b = VectorMPI(b_full)
x = solve(F, b)

x_full = Vector(x)
residual = A_full * x_full - b_full
err = norm(residual, Inf)

println(io0(), "  LDLT solve residual (indefinite): $err")
@test err < TOL


# Test 4: Factorization reuse (multiple solves with same factorization)
println(io0(), "[test] Factorization reuse")

n = 8
A_full = create_spd_tridiagonal(n)
A = SparseMatrixMPI{Float64}(A_full)
F = ldlt(A)

b1_full = ones(n)
b1 = VectorMPI(b1_full)
x1 = solve(F, b1)

b2_full = collect(1.0:n)
b2 = VectorMPI(b2_full)
x2 = solve(F, b2)

x1_full = Vector(x1)
x2_full = Vector(x2)

err1 = norm(A_full * x1_full - b1_full, Inf)
err2 = norm(A_full * x2_full - b2_full, Inf)

println(io0(), "  Residual 1: $err1")
println(io0(), "  Residual 2: $err2")

@test err1 < TOL
@test err2 < TOL


# Test 5: Complex-valued matrix (LU)
println(io0(), "[test] LU factorization - complex")

n = 6
A_full_real = create_general_tridiagonal(n)
A_full = Complex{Float64}.(A_full_real) + im * spdiagm(0 => 0.1*ones(n))
A = SparseMatrixMPI{ComplexF64}(A_full)

F = lu(A)

b_full = ones(ComplexF64, n)
b = VectorMPI(b_full)
x = solve(F, b)

x_full = Vector(x)
residual = A_full * x_full - b_full
err = norm(residual, Inf)

println(io0(), "  LU solve residual (complex): $err")
@test err < TOL


# Test 6: Transpose solve - transpose(A) \ b
println(io0(), "[test] Transpose solve")

n = 8
A_full = create_general_tridiagonal(n)
A = SparseMatrixMPI{Float64}(A_full)
b_full = ones(n)
b = VectorMPI(b_full)

x_t = transpose(A) \ b

x_t_full = Vector(x_t)
residual_t = transpose(A_full) * x_t_full - b_full
err_t = norm(residual_t, Inf)

println(io0(), "  Transpose solve residual: $err_t")
@test err_t < TOL


# Test 7: Adjoint solve - A' \ b
println(io0(), "[test] Adjoint solve")

x_a = A' \ b

x_a_full = Vector(x_a)
residual_a = A_full' * x_a_full - b_full
err_a = norm(residual_a, Inf)

println(io0(), "  Adjoint solve residual: $err_a")
@test err_a < TOL


# Test 8: Right division - transpose(v) / A
println(io0(), "[test] Right division - transpose(v) / A")

# transpose(v) / A solves x * A = transpose(v)
x_rd = transpose(b) / A

# Verify: x * A should equal transpose(b)
x_rd_parent = x_rd.parent
x_rd_full = Vector(x_rd_parent)
residual_rd = x_rd_full' * A_full - b_full'
err_rd = norm(residual_rd, Inf)

println(io0(), "  Right division residual: $err_rd")
@test err_rd < TOL


# Test 9: Right division - transpose(v) / transpose(A)
println(io0(), "[test] Right division - transpose(v) / transpose(A)")

x_rdt = transpose(b) / transpose(A)
x_rdt_full = Vector(x_rdt.parent)
residual_rdt = x_rdt_full' * transpose(A_full) - b_full'
err_rdt = norm(residual_rdt, Inf)

println(io0(), "  Right division (transpose) residual: $err_rdt")
@test err_rdt < TOL


# Test 10: 2D Laplacian (larger problem)
println(io0(), "[test] LDLT factorization - 2D Laplacian")

A_2d_full = create_2d_laplacian(6, 6)  # 36-element grid
A_2d = SparseMatrixMPI{Float64}(A_2d_full)

F_2d = ldlt(A_2d)

b_2d_full = ones(36)
b_2d = VectorMPI(b_2d_full)
x_2d = solve(F_2d, b_2d)

x_2d_full = Vector(x_2d)
residual_2d = A_2d_full * x_2d_full - b_2d_full
err_2d = norm(residual_2d, Inf)

println(io0(), "  2D Laplacian LDLT residual: $err_2d")
@test err_2d < TOL


# Test 11: LU with 2D Laplacian
println(io0(), "[test] LU factorization - 2D Laplacian")

A_2d_lu_full = create_2d_laplacian(5, 5)  # 25-element grid
A_2d_lu = SparseMatrixMPI{Float64}(A_2d_lu_full)

F_2d_lu = lu(A_2d_lu)

b_2d_lu_full = ones(25)
b_2d_lu = VectorMPI(b_2d_lu_full)
x_2d_lu = solve(F_2d_lu, b_2d_lu)

x_2d_lu_full = Vector(x_2d_lu)
residual_2d_lu = A_2d_lu_full * x_2d_lu_full - b_2d_lu_full
err_2d_lu = norm(residual_2d_lu, Inf)

println(io0(), "  2D Laplacian LU residual: $err_2d_lu")
@test err_2d_lu < TOL


# Test 12: Complex symmetric LDLT
println(io0(), "[test] LDLT factorization - complex symmetric")

n = 6
A_cx_full = create_complex_symmetric(n)
A_cx = SparseMatrixMPI{ComplexF64}(A_cx_full)

F_cx = ldlt(A_cx)

b_cx_full = ones(ComplexF64, n)
b_cx = VectorMPI(b_cx_full)
x_cx = solve(F_cx, b_cx)

x_cx_full = Vector(x_cx)
residual_cx = A_cx_full * x_cx_full - b_cx_full
err_cx = norm(residual_cx, Inf)

println(io0(), "  Complex symmetric LDLT residual: $err_cx")
@test err_cx < TOL


# Test 13: Block diagonal matrix (multiple disconnected components)
println(io0(), "[test] Block diagonal matrix")

block_size = 10
n_multi = 2 * block_size
A_multi = spzeros(n_multi, n_multi)
for b_idx in 0:1
    offset = b_idx * block_size
    for i in 1:block_size
        A_multi[offset + i, offset + i] = 4.0
        if i > 1
            A_multi[offset + i, offset + i - 1] = -1.0
            A_multi[offset + i - 1, offset + i] = -1.0
        end
    end
end
A_multi_mpi = SparseMatrixMPI{Float64}(A_multi)

F_multi = ldlt(A_multi_mpi)

b_multi_full = ones(n_multi)
b_multi = VectorMPI(b_multi_full)
x_multi = solve(F_multi, b_multi)
x_multi_full = Vector(x_multi)
err_multi = norm(A_multi * x_multi_full - b_multi_full, Inf)

println(io0(), "  Block diagonal LDLT residual: $err_multi")
@test err_multi < TOL


# Test 14: Larger problem size
println(io0(), "[test] Larger problem size (100x100 grid)")

A_large_full = create_2d_laplacian(10, 10)  # 100 DOF
A_large = SparseMatrixMPI{Float64}(A_large_full)

F_large = ldlt(A_large)

b_large_full = ones(100)
b_large = VectorMPI(b_large_full)
x_large = solve(F_large, b_large)

x_large_full = Vector(x_large)
residual_large = A_large_full * x_large_full - b_large_full
err_large = norm(residual_large, Inf)

println(io0(), "  100 DOF LDLT residual: $err_large")
@test err_large < TOL


# Test 15: solve! (in-place solve)
println(io0(), "[test] solve! (in-place)")

n = 8
A_full = create_spd_tridiagonal(n)
A = SparseMatrixMPI{Float64}(A_full)
F = ldlt(A)

b_full = ones(n)
b = VectorMPI(b_full)
x = VectorMPI(zeros(n))

solve!(x, F, b)

x_full = Vector(x)
err = norm(A_full * x_full - b_full, Inf)

println(io0(), "  solve! residual: $err")
@test err < TOL


# Test 16: issymmetric with asymmetric partitions (exercises cross-rank row comparison)
println(io0(), "[test] issymmetric with asymmetric partitions - symmetric matrix")

# Use size that guarantees different partitions with 4 ranks
# n=12: uniform gives [1,4,7,10,13], we use [1,3,6,9,13] for columns
n_asym = 12
A_sym_full_asym = create_spd_tridiagonal(n_asym)
row_part = LinearAlgebraMPI.uniform_partition(n_asym, nranks)
# Create a different valid partition: sizes 2,3,3,4 instead of 3,3,3,3
col_part = if nranks == 4
    [1, 3, 6, 9, 13]
else
    # For other rank counts, just offset by 1 where possible
    rp = copy(row_part)
    for i in 2:length(rp)-1
        if rp[i] + 1 < rp[i+1]
            rp[i] += 1
            break
        end
    end
    rp
end

A_asym = SparseMatrixMPI{Float64}(A_sym_full_asym; row_partition=row_part, col_partition=col_part)
@test issymmetric(A_asym) == true
println(io0(), "  Symmetric matrix with asymmetric partitions: passed")


# Test 17: issymmetric with asymmetric partitions - non-symmetric matrix
println(io0(), "[test] issymmetric with asymmetric partitions - non-symmetric matrix")

A_nonsym_full_asym = create_general_tridiagonal(n_asym)  # Has -0.5 above diagonal, -0.8 below
A_nonsym_asym = SparseMatrixMPI{Float64}(A_nonsym_full_asym; row_partition=row_part, col_partition=col_part)
@test issymmetric(A_nonsym_asym) == false
println(io0(), "  Non-symmetric matrix with asymmetric partitions: passed")

end  # QuietTestSet

# Aggregate results across ranks
local_counts = [ts.counts[:pass], ts.counts[:fail], ts.counts[:error], ts.counts[:broken], ts.counts[:skip]]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

total = sum(global_counts)
println(io0(), "\nTest Summary: distributed factorization | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])  Total: $total")

# MPI.Finalize() is called automatically by MPI.jl's atexit hook on clean exit
# Note: Exit code is determined by runtests.jl checking the output for Pass/Fail counts
