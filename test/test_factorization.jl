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
