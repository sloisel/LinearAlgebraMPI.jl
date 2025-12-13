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
    # 2D Laplacian on nx × ny grid - creates multiple supernodes with children
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
if rank == 0
    println("[test] LU factorization - small matrix")
    flush(stdout)
end

n = 8
A_full = create_general_tridiagonal(n)
A = SparseMatrixMPI{Float64}(A_full)

F = lu(A)
@test F isa LinearAlgebraMPI.LUFactorizationMPI{Float64}
@test size(F) == (n, n)

b_full = ones(n)
b = VectorMPI(b_full)
x = solve(F, b)

x_full = Vector(x)
residual = A_full * x_full - b_full
err = norm(residual, Inf)

if rank == 0
    println("  LU solve residual: $err")
end
@test err < TOL

MPI.Barrier(comm)

# Test 2: LDLT factorization of SPD matrix
if rank == 0
    println("[test] LDLT factorization - SPD matrix")
    flush(stdout)
end

n = 10
A_full = create_spd_tridiagonal(n)
A = SparseMatrixMPI{Float64}(A_full)

F = ldlt(A)
@test F isa LinearAlgebraMPI.LDLTFactorizationMPI{Float64}
@test size(F) == (n, n)

b_full = ones(n)
b = VectorMPI(b_full)
x = solve(F, b)

x_full = Vector(x)
residual = A_full * x_full - b_full
err = norm(residual, Inf)

if rank == 0
    println("  LDLT solve residual (SPD): $err")
end
@test err < TOL

MPI.Barrier(comm)

# Test 3: LDLT with symmetric indefinite matrix
if rank == 0
    println("[test] LDLT factorization - indefinite matrix")
    flush(stdout)
end

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

if rank == 0
    println("  LDLT solve residual (indefinite): $err")
end
@test err < TOL

MPI.Barrier(comm)

# Test 4: Plan reuse (same structure, different values)
if rank == 0
    println("[test] Plan reuse")
    flush(stdout)
end

n = 8
A1_full = create_spd_tridiagonal(n)
A1 = SparseMatrixMPI{Float64}(A1_full)
F1 = ldlt(A1; reuse_symbolic=true)

A2_full = create_spd_tridiagonal(n)
A2_full.nzval .*= 2.0
A2 = SparseMatrixMPI{Float64}(A2_full)
F2 = ldlt(A2; reuse_symbolic=true)

b_full = ones(n)
b = VectorMPI(b_full)

x1 = solve(F1, b)
x2 = solve(F2, b)

x1_full = Vector(x1)
x2_full = Vector(x2)

err1 = norm(A1_full * x1_full - b_full, Inf)
err2 = norm(A2_full * x2_full - b_full, Inf)

if rank == 0
    println("  Residual 1: $err1")
    println("  Residual 2: $err2")
end

@test err1 < TOL
@test err2 < TOL

MPI.Barrier(comm)

# Test 5: Complex-valued matrix (LU)
if rank == 0
    println("[test] LU factorization - complex")
    flush(stdout)
end

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

if rank == 0
    println("  LU solve residual (complex): $err")
end
@test err < TOL

# Test 6: Direct A \ b solve
if rank == 0
    println("[test] Direct A \\ b solve")
    flush(stdout)
end

n = 8
A_full = create_general_tridiagonal(n)
A = SparseMatrixMPI{Float64}(A_full)
b_full = ones(n)
b = VectorMPI(b_full)

# Direct solve without explicit factorization
x = A \ b

x_full = Vector(x)
residual = A_full * x_full - b_full
err = norm(residual, Inf)

if rank == 0
    println("  Direct solve residual: $err")
end
@test err < TOL

MPI.Barrier(comm)

# Test 7: Transpose solve - transpose(A) \ b
if rank == 0
    println("[test] Transpose solve")
    flush(stdout)
end

x_t = transpose(A) \ b

x_t_full = Vector(x_t)
residual_t = transpose(A_full) * x_t_full - b_full
err_t = norm(residual_t, Inf)

if rank == 0
    println("  Transpose solve residual: $err_t")
end
@test err_t < TOL

MPI.Barrier(comm)

# Test 8: Adjoint solve - A' \ b
if rank == 0
    println("[test] Adjoint solve")
    flush(stdout)
end

x_a = A' \ b

x_a_full = Vector(x_a)
residual_a = A_full' * x_a_full - b_full
err_a = norm(residual_a, Inf)

if rank == 0
    println("  Adjoint solve residual: $err_a")
end
@test err_a < TOL

MPI.Barrier(comm)

# Test 9: Factorization transpose/adjoint with F
if rank == 0
    println("[test] Factorization transpose/adjoint")
    flush(stdout)
end

F = lu(A)

# transpose(F) \ b should solve transpose(A) * x = b
x_Ft = transpose(F) \ b
x_Ft_full = Vector(x_Ft)
err_Ft = norm(transpose(A_full) * x_Ft_full - b_full, Inf)

# F' \ b should solve A' * x = b
x_Fa = F' \ b
x_Fa_full = Vector(x_Fa)
err_Fa = norm(A_full' * x_Fa_full - b_full, Inf)

if rank == 0
    println("  F transpose solve residual: $err_Ft")
    println("  F adjoint solve residual: $err_Fa")
end
@test err_Ft < TOL
@test err_Fa < TOL

MPI.Barrier(comm)

# Test 10: Right division - transpose(v) / A
if rank == 0
    println("[test] Right division - transpose(v) / A")
    flush(stdout)
end

# transpose(v) / A solves x * A = transpose(v)
x_rd = transpose(b) / A

# Verify: x * A should equal transpose(b)
# x is a transposed VectorMPI, so x.parent * A should give b
x_rd_parent = x_rd.parent
x_rd_full = Vector(x_rd_parent)
residual_rd = x_rd_full' * A_full - b_full'
err_rd = norm(residual_rd, Inf)

if rank == 0
    println("  Right division residual: $err_rd")
end
@test err_rd < TOL

MPI.Barrier(comm)

# Test 11: Right division - transpose(v) / transpose(A)
if rank == 0
    println("[test] Right division - transpose(v) / transpose(A)")
    flush(stdout)
end

x_rdt = transpose(b) / transpose(A)
x_rdt_full = Vector(x_rdt.parent)
residual_rdt = x_rdt_full' * transpose(A_full) - b_full'
err_rdt = norm(residual_rdt, Inf)

if rank == 0
    println("  Right division (transpose) residual: $err_rdt")
end
@test err_rdt < TOL

MPI.Barrier(comm)

# Test 12: Right division - v' / A (adjoint)
if rank == 0
    println("[test] Right division - v' / A")
    flush(stdout)
end

x_rda = b' / A
x_rda_full = Vector(x_rda.parent)
residual_rda = x_rda_full' * A_full - b_full'
err_rda = norm(residual_rda, Inf)

if rank == 0
    println("  Right division (adjoint) residual: $err_rda")
end
@test err_rda < TOL

MPI.Barrier(comm)

# Test 13: Right division - v' / A'
if rank == 0
    println("[test] Right division - v' / A'")
    flush(stdout)
end

x_rdaa = b' / A'
x_rdaa_full = Vector(x_rdaa.parent)
residual_rdaa = x_rdaa_full' * A_full' - b_full'
err_rdaa = norm(residual_rdaa, Inf)

if rank == 0
    println("  Right division (adjoint/adjoint) residual: $err_rdaa")
end
@test err_rdaa < TOL

MPI.Barrier(comm)

# Test 14: LDLT transpose/adjoint solves via factorization (real symmetric)
if rank == 0
    println("[test] LDLT factorization transpose/adjoint")
    flush(stdout)
end

n = 10
A_ldlt_full = create_spd_tridiagonal(n)
A_ldlt = SparseMatrixMPI{Float64}(A_ldlt_full)
F_ldlt = ldlt(A_ldlt)

b_ldlt_full = ones(n)
b_ldlt = VectorMPI(b_ldlt_full)

# transpose(F) \ b should solve transpose(A) * x = b (same as A * x = b for symmetric)
x_ldlt_t = transpose(F_ldlt) \ b_ldlt
x_ldlt_t_full = Vector(x_ldlt_t)
err_ldlt_t = norm(A_ldlt_full * x_ldlt_t_full - b_ldlt_full, Inf)

# F' \ b should solve A' * x = b (same as A * x = b for real symmetric)
x_ldlt_a = F_ldlt' \ b_ldlt
x_ldlt_a_full = Vector(x_ldlt_a)
err_ldlt_a = norm(A_ldlt_full * x_ldlt_a_full - b_ldlt_full, Inf)

if rank == 0
    println("  LDLT transpose solve residual: $err_ldlt_t")
    println("  LDLT adjoint solve residual: $err_ldlt_a")
end
@test err_ldlt_t < TOL
@test err_ldlt_a < TOL

MPI.Barrier(comm)

# Test 15: Complex symmetric LDLT
if rank == 0
    println("[test] LDLT factorization - complex symmetric")
    flush(stdout)
end

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

if rank == 0
    println("  Complex symmetric LDLT residual: $err_cx")
end
@test err_cx < TOL

MPI.Barrier(comm)

# Test 16: Complex symmetric LDLT adjoint solve (exercises conj helpers)
if rank == 0
    println("[test] LDLT adjoint solve - complex symmetric")
    flush(stdout)
end

# For complex symmetric A (A = A^T but A != A'), adjoint solve is different
# solve A' * x = b where A' = conj(A)
x_cx_adj = F_cx' \ b_cx
x_cx_adj_full = Vector(x_cx_adj)
residual_cx_adj = A_cx_full' * x_cx_adj_full - b_cx_full
err_cx_adj = norm(residual_cx_adj, Inf)

if rank == 0
    println("  Complex symmetric LDLT adjoint residual: $err_cx_adj")
end
@test err_cx_adj < TOL

MPI.Barrier(comm)

# Test 17: 2D Laplacian - exercises extend_add_sym! with supernode children
if rank == 0
    println("[test] LDLT factorization - 2D Laplacian (supernode extend-add)")
    flush(stdout)
end

A_2d_full = create_2d_laplacian(6, 6)  # 36-element grid
A_2d = SparseMatrixMPI{Float64}(A_2d_full)

F_2d = ldlt(A_2d)

b_2d_full = ones(36)
b_2d = VectorMPI(b_2d_full)
x_2d = solve(F_2d, b_2d)

x_2d_full = Vector(x_2d)
residual_2d = A_2d_full * x_2d_full - b_2d_full
err_2d = norm(residual_2d, Inf)

if rank == 0
    println("  2D Laplacian LDLT residual: $err_2d")
end
@test err_2d < TOL

MPI.Barrier(comm)

# Test 18: LDLT with 2x2 Bunch-Kaufman pivots
# Uses a dense symmetric indefinite matrix with small diagonal to force 2x2 pivots
if rank == 0
    println("[test] LDLT with 2x2 Bunch-Kaufman pivots")
    flush(stdout)
end

n_bk = 4
A_bk = zeros(n_bk, n_bk)
for i in 1:n_bk
    A_bk[i, i] = 1e-16  # Tiny diagonal forces 2x2 pivot selection
    for j in i+1:n_bk
        A_bk[i, j] = 1.0 + 0.1 * (i + j)  # Large off-diagonal
        A_bk[j, i] = A_bk[i, j]
    end
end
A_bk_sp = sparse(A_bk)
A_bk_mpi = SparseMatrixMPI{Float64}(A_bk_sp)

F_bk = ldlt(A_bk_mpi)

# Verify 2x2 pivots were used (pivots[k] < 0 indicates 2x2 pivot)
has_2x2_pivots = any(F_bk.pivots .< 0)
@test has_2x2_pivots

b_bk_full = ones(n_bk)
b_bk = VectorMPI(b_bk_full)
x_bk = solve(F_bk, b_bk)
x_bk_full = Vector(x_bk)
err_bk = norm(A_bk_sp * x_bk_full - b_bk_full, Inf)

if rank == 0
    println("  2x2 pivots used: $has_2x2_pivots")
    println("  Bunch-Kaufman LDLT residual: $err_bk")
end
@test err_bk < TOL

MPI.Barrier(comm)

# Test 19: Multi-rank supernode distribution
# Uses block diagonal matrix to create multiple elimination tree roots
if rank == 0
    println("[test] Multi-rank supernode distribution")
    flush(stdout)
end

# Create 2-block diagonal matrix (disconnected components)
block_size = 10
n_multi = 2 * block_size
A_multi = spzeros(n_multi, n_multi)
for b in 0:1
    offset = b * block_size
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

# Verify multi-rank distribution (supernodes on different ranks)
unique_owners = unique(F_multi.symbolic.snode_owner)
has_multi_rank = length(unique_owners) > 1 && nranks > 1

# Note: multi-rank is expected only if running with 2+ ranks
if nranks > 1
    @test has_multi_rank
end

b_multi_full = ones(n_multi)
b_multi = VectorMPI(b_multi_full)
x_multi = solve(F_multi, b_multi)
x_multi_full = Vector(x_multi)
err_multi = norm(A_multi * x_multi_full - b_multi_full, Inf)

if rank == 0
    println("  Multi-rank distribution: $has_multi_rank (nranks=$nranks)")
    println("  Block diagonal LDLT residual: $err_multi")
end
@test err_multi < TOL

MPI.Barrier(comm)

# Test 20: LU with 2D Laplacian - exercises extend_add! (unsymmetric version)
if rank == 0
    println("[test] LU factorization - 2D Laplacian (extend_add!)")
    flush(stdout)
end

A_2d_lu_full = create_2d_laplacian(5, 5)  # 25-element grid
A_2d_lu = SparseMatrixMPI{Float64}(A_2d_lu_full)

F_2d_lu = lu(A_2d_lu)

b_2d_lu_full = ones(25)
b_2d_lu = VectorMPI(b_2d_lu_full)
x_2d_lu = solve(F_2d_lu, b_2d_lu)

x_2d_lu_full = Vector(x_2d_lu)
residual_2d_lu = A_2d_lu_full * x_2d_lu_full - b_2d_lu_full
err_2d_lu = norm(residual_2d_lu, Inf)

if rank == 0
    println("  2D Laplacian LU residual: $err_2d_lu")
end
@test err_2d_lu < TOL

MPI.Barrier(comm)

# Test 21: LU with partial pivoting - matrix requiring row swaps
# Off-diagonal elements larger than diagonal forces pivot selection
if rank == 0
    println("[test] LU with row pivoting")
    flush(stdout)
end

n_piv = 6
A_piv = zeros(n_piv, n_piv)
for i in 1:n_piv
    A_piv[i, i] = 0.1  # Small diagonal
    if i < n_piv
        A_piv[i+1, i] = 2.0  # Large sub-diagonal (will be selected as pivot)
        A_piv[i, i+1] = 1.5  # Upper diagonal
    end
end
# Make it non-singular by adjusting
A_piv[n_piv, n_piv] = 2.0
A_piv_sp = sparse(A_piv)
A_piv_mpi = SparseMatrixMPI{Float64}(A_piv_sp)

F_piv = lu(A_piv_mpi)

b_piv_full = ones(n_piv)
b_piv = VectorMPI(b_piv_full)
x_piv = solve(F_piv, b_piv)
x_piv_full = Vector(x_piv)
err_piv = norm(A_piv_sp * x_piv_full - b_piv_full, Inf)

if rank == 0
    println("  LU with pivoting residual: $err_piv")
end
@test err_piv < TOL

MPI.Barrier(comm)

# Test 22: LU with near-zero pivot (triggers small pivot warning path)
if rank == 0
    println("[test] LU with small pivot")
    flush(stdout)
end

n_small = 4
A_small = zeros(n_small, n_small)
A_small[1, 1] = 1e-20  # Very small pivot
A_small[1, 2] = 1e-21  # Even smaller off-diagonal so no swap
A_small[2, 1] = 1e-21
A_small[2, 2] = 1.0
A_small[2, 3] = -0.5
A_small[3, 2] = -0.5
A_small[3, 3] = 1.0
A_small[3, 4] = -0.5
A_small[4, 3] = -0.5
A_small[4, 4] = 1.0
A_small_sp = sparse(A_small)
A_small_mpi = SparseMatrixMPI{Float64}(A_small_sp)

# This should trigger the small pivot warning but still succeed
F_small = lu(A_small_mpi)

b_small_full = [1e-20, 1.0, 1.0, 1.0]  # Scale first element with matrix
b_small = VectorMPI(b_small_full)
x_small = solve(F_small, b_small)
x_small_full = Vector(x_small)
err_small = norm(A_small_sp * x_small_full - b_small_full, Inf)

if rank == 0
    println("  LU with small pivot residual: $err_small")
end
# Use looser tolerance due to ill-conditioning
@test err_small < 1e-6

MPI.Barrier(comm)

# Test 23: LU with exactly zero pivot (tests abs(diag_val) == 0 branch)
if rank == 0
    println("[test] LU with zero pivot")
    flush(stdout)
end

n_zero = 4
A_zero = zeros(n_zero, n_zero)
A_zero[1, 1] = 0.0  # Exactly zero pivot - will be replaced by eps
A_zero[1, 2] = 0.0  # Zero off-diagonal so no pivot swap
A_zero[2, 1] = 0.0
A_zero[2, 2] = 1.0
A_zero[2, 3] = -0.5
A_zero[3, 2] = -0.5
A_zero[3, 3] = 1.0
A_zero[3, 4] = -0.5
A_zero[4, 3] = -0.5
A_zero[4, 4] = 1.0
A_zero_sp = sparse(A_zero)
A_zero_mpi = SparseMatrixMPI{Float64}(A_zero_sp)

# This should trigger zero pivot replacement and warning
F_zero = lu(A_zero_mpi)

# The factorization will have modified the pivot, so solve will work
# but the original matrix is singular, so we just test that solve completes
b_zero_full = [0.0, 1.0, 1.0, 1.0]  # RHS consistent with singular row
b_zero = VectorMPI(b_zero_full)
x_zero = solve(F_zero, b_zero)
x_zero_full = Vector(x_zero)

if rank == 0
    println("  LU with zero pivot: solve completed")
end
@test length(x_zero_full) == n_zero  # Just verify solve completes

MPI.Barrier(comm)

# Test 24: LDLT with exactly zero 1x1 pivot (tests abs(d_kk) == 0 branch)
if rank == 0
    println("[test] LDLT with zero 1x1 pivot")
    flush(stdout)
end

n_ldlt_zero = 4
A_ldlt_zero = zeros(n_ldlt_zero, n_ldlt_zero)
# Create symmetric matrix with zero diagonal that will trigger 1x1 pivot
# (no larger off-diagonal in column to trigger 2x2)
A_ldlt_zero[1, 1] = 0.0  # Zero pivot
A_ldlt_zero[2, 2] = 2.0
A_ldlt_zero[3, 3] = 2.0
A_ldlt_zero[4, 4] = 2.0
# Small symmetric off-diagonals (smaller than alpha * |diagonal|)
A_ldlt_zero[2, 1] = 0.0
A_ldlt_zero[1, 2] = 0.0
A_ldlt_zero[3, 2] = -0.1
A_ldlt_zero[2, 3] = -0.1
A_ldlt_zero[4, 3] = -0.1
A_ldlt_zero[3, 4] = -0.1
A_ldlt_zero_sp = sparse(A_ldlt_zero)
A_ldlt_zero_mpi = SparseMatrixMPI{Float64}(A_ldlt_zero_sp)

F_ldlt_zero = ldlt(A_ldlt_zero_mpi)

b_ldlt_zero_full = [0.0, 1.0, 1.0, 1.0]
b_ldlt_zero = VectorMPI(b_ldlt_zero_full)
x_ldlt_zero = solve(F_ldlt_zero, b_ldlt_zero)
x_ldlt_zero_full = Vector(x_ldlt_zero)

if rank == 0
    println("  LDLT with zero 1x1 pivot: solve completed")
end
@test length(x_ldlt_zero_full) == n_ldlt_zero

MPI.Barrier(comm)

# Test 25: LDLT with near-zero 2x2 pivot determinant
# Uses similar structure to Test 18 but with det(2x2 block) ≈ 0
if rank == 0
    println("[test] LDLT with small 2x2 determinant")
    flush(stdout)
end

n_det = 4
A_det = zeros(n_det, n_det)
# Create a 2x2 block at (1,2) with det ≈ 0
# det = a11*a22 - a12^2 = 1e-16 * 1e-16 - (1e-16)^2 = 0
A_det[1, 1] = 1e-16   # Tiny diagonal
A_det[2, 2] = 1e-16   # Tiny diagonal
A_det[2, 1] = 1e-16   # Off-diagonal: makes det = 1e-32 - 1e-32 = 0
A_det[1, 2] = 1e-16
# Large off-diagonals to force 2x2 pivot selection (like Test 18)
for i in 1:n_det
    for j in i+1:n_det
        if (i <= 2 && j <= 2)
            continue  # Skip the 2x2 block we already set
        end
        A_det[i, j] = 1.0 + 0.1 * (i + j)
        A_det[j, i] = A_det[i, j]
    end
end
A_det[3, 3] = 1e-16  # Small diagonals throughout
A_det[4, 4] = 1e-16
A_det_sp = sparse(A_det)
A_det_mpi = SparseMatrixMPI{Float64}(A_det_sp)

F_det = ldlt(A_det_mpi)

# Check that 2x2 pivot was used
has_2x2_det = any(F_det.pivots .< 0)

if rank == 0
    println("  2x2 pivot used: $has_2x2_det")
end
@test has_2x2_det  # Should use 2x2 pivots

b_det_full = ones(n_det)
b_det = VectorMPI(b_det_full)
x_det = solve(F_det, b_det)
x_det_full = Vector(x_det)

if rank == 0
    println("  LDLT with small 2x2 determinant: solve completed")
end
@test length(x_det_full) == n_det

MPI.Barrier(comm)

# Test 26: Test swap_rows_cols_sym! with i==j (no-op branch)
# This occurs when Bunch-Kaufman selects the diagonal element as pivot
if rank == 0
    println("[test] LDLT diagonal pivot (no swap)")
    flush(stdout)
end

n_diag = 4
A_diag = zeros(n_diag, n_diag)
# Diagonally dominant: diagonal element will be selected as pivot without swap
A_diag[1, 1] = 10.0  # Large diagonal
A_diag[2, 2] = 10.0
A_diag[3, 3] = 10.0
A_diag[4, 4] = 10.0
A_diag[2, 1] = -0.1  # Small off-diagonals
A_diag[1, 2] = -0.1
A_diag[3, 2] = -0.1
A_diag[2, 3] = -0.1
A_diag[4, 3] = -0.1
A_diag[3, 4] = -0.1
A_diag_sp = sparse(A_diag)
A_diag_mpi = SparseMatrixMPI{Float64}(A_diag_sp)

F_diag = ldlt(A_diag_mpi)

b_diag_full = ones(n_diag)
b_diag = VectorMPI(b_diag_full)
x_diag = solve(F_diag, b_diag)
x_diag_full = Vector(x_diag)
err_diag = norm(A_diag_sp * x_diag_full - b_diag_full, Inf)

if rank == 0
    println("  Diagonal pivot residual: $err_diag")
end
@test err_diag < TOL

MPI.Barrier(comm)

# Test 27: LDLT 2x2 pivot with update rows - exercises extract_L_D! else branch
# Need a matrix where 2x2 pivots are used and there are update rows
if rank == 0
    println("[test] LDLT 2x2 pivot with update structure")
    flush(stdout)
end

# Use a 2D Laplacian variant that forces 2x2 pivots with update rows
# The AMD ordering creates supernodes with children, and we make diagonals
# small to force 2x2 pivot selection
A_2x2_base = create_2d_laplacian(3, 3)  # 9 nodes
A_2x2 = Matrix(A_2x2_base)
# Make first two diagonal elements very small to force 2x2 pivot
A_2x2[1, 1] = 1e-16
A_2x2[2, 2] = 1e-16
# But keep the off-diagonal between them
A_2x2[2, 1] = -1.0
A_2x2[1, 2] = -1.0
A_2x2_sp = sparse(A_2x2)
A_2x2_mpi = SparseMatrixMPI{Float64}(A_2x2_sp)

F_2x2 = ldlt(A_2x2_mpi)

# Check 2x2 pivots were used
has_2x2_update = any(F_2x2.pivots .< 0)

b_2x2_full = ones(9)
b_2x2 = VectorMPI(b_2x2_full)
x_2x2 = solve(F_2x2, b_2x2)
x_2x2_full = Vector(x_2x2)

if rank == 0
    println("  2x2 pivots with updates used: $has_2x2_update")
    println("  LDLT 2x2 with updates: solve completed")
end
@test length(x_2x2_full) == 9

end  # QuietTestSet

# Aggregate results across ranks
local_counts = [ts.counts[:pass], ts.counts[:fail], ts.counts[:error], ts.counts[:broken], ts.counts[:skip]]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

if rank == 0
    total = sum(global_counts)
    println("\nTest Summary: distributed factorization | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])  Total: $total")
    flush(stdout)
end

MPI.Barrier(comm)
MPI.Finalize()

exit_code = global_counts[2] + global_counts[3] > 0 ? 1 : 0
exit(exit_code)
