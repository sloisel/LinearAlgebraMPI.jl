"""
Tests for distributed LU and LDLT factorization.
Parameterized over all scalar types and backends.
MUMPS internally converts to Float64/ComplexF64 but supports any input type/backend.
"""

# Check Metal availability BEFORE loading MPI (for consistency with other tests)
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch e
    false
end

using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Create deterministic test matrices (parameterized by type)
function create_spd_tridiagonal(::Type{T}, n::Int) where T
    # Symmetric positive definite tridiagonal matrix
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = T.([4.0*ones(n); -1.0*ones(n-1); -1.0*ones(n-1)])
    return sparse(I_A, J_A, V_A, n, n)
end

function create_general_tridiagonal(::Type{T}, n::Int) where T
    # General (unsymmetric) tridiagonal matrix
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = T.([2.0*ones(n); -0.5*ones(n-1); -0.8*ones(n-1)])
    return sparse(I_A, J_A, V_A, n, n)
end

function create_symmetric_indefinite(::Type{T}, n::Int) where T
    # Symmetric indefinite matrix
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    # Alternating signs on diagonal
    diag_vals = [(-1.0)^i * 2.0 for i in 1:n]
    V_A = T.([diag_vals; -1.0*ones(n-1); -1.0*ones(n-1)])
    return sparse(I_A, J_A, V_A, n, n)
end

function create_2d_laplacian(::Type{T}, nx::Int, ny::Int) where T
    # 2D Laplacian on nx x ny grid
    n = nx * ny
    I_A = Int[]
    J_A = Int[]
    V_A = T[]

    for i = 1:nx
        for j = 1:ny
            idx = (j-1)*nx + i
            # Diagonal
            push!(I_A, idx)
            push!(J_A, idx)
            push!(V_A, T(4.0))
            # Left neighbor
            if i > 1
                push!(I_A, idx)
                push!(J_A, idx-1)
                push!(V_A, T(-1.0))
            end
            # Right neighbor
            if i < nx
                push!(I_A, idx)
                push!(J_A, idx+1)
                push!(V_A, T(-1.0))
            end
            # Bottom neighbor
            if j > 1
                push!(I_A, idx)
                push!(J_A, idx-nx)
                push!(V_A, T(-1.0))
            end
            # Top neighbor
            if j < ny
                push!(I_A, idx)
                push!(J_A, idx+nx)
                push!(V_A, T(-1.0))
            end
        end
    end

    return sparse(I_A, J_A, V_A, n, n)
end

function create_complex_symmetric(::Type{T}, n::Int) where T
    # Complex symmetric (NOT Hermitian) matrix
    # A = A^T but A != A' (adjoint)
    @assert T <: Complex "create_complex_symmetric requires complex type"

    I_A = Int[]
    J_A = Int[]
    V_A = T[]

    # Complex diagonal
    for i = 1:n
        push!(I_A, i)
        push!(J_A, i)
        push!(V_A, T(3.0 + 1.0im))  # Complex diagonal
    end

    # Complex symmetric off-diagonal (not Hermitian: A[i,j] = A[j,i], not conj)
    for i = 1:n-1
        val = T(-0.5 + 0.2im)
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

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)
    RT = real(T)  # Real type for creating real-valued test data
    VT, ST, MT = TestUtils.expected_types(T, to_backend)
    VT_real, ST_real, MT_real = TestUtils.expected_types(RT, to_backend)

    # Skip tridiagonal tests for CUDA - cuDSS MGMN has a bug with narrow-band matrices
    # See bug/ folder for minimal reproducer
    if backend_name == "CUDA"
        println(io0(), "[skip] Tridiagonal tests skipped for CUDA (cuDSS MGMN bug)")
    else

    println(io0(), "[test] LU factorization - small matrix ($T, $backend_name)")

    n = 8
    A_full = create_general_tridiagonal(T, n)
    A_cpu = SparseMatrixMPI{T}(A_full)
    A = assert_type(to_backend(A_cpu), ST)

    F = lu(A)
    @test size(F) == (n, n)
    @test eltype(F) == T  # Factorization preserves original type

    b_full = ones(T, n)
    b = assert_type(to_backend(VectorMPI(b_full)), VT)
    x = assert_type(F \ b, VT)

    # Convert GPU result back to CPU for comparison
    x_cpu = TestUtils.to_cpu(x)
    x_full = Vector(x_cpu)
    residual = A_full * x_full - b_full
    err = assert_uniform(norm(residual, Inf), name="lu_residual")

    println(io0(), "  LU solve residual: $err")
    @test err < TOL


    println(io0(), "[test] LDLT factorization - SPD matrix ($T, $backend_name)")

    n = 10
    A_full = create_spd_tridiagonal(RT, n)  # SPD matrices are real
    A_cpu = SparseMatrixMPI{RT}(A_full)
    A = assert_type(to_backend(A_cpu), ST_real)

    F = ldlt(A)
    @test size(F) == (n, n)

    b_full = ones(RT, n)
    b = assert_type(to_backend(VectorMPI(b_full)), VT_real)
    x = assert_type(F \ b, VT_real)

    x_cpu = TestUtils.to_cpu(x)
    x_full = Vector(x_cpu)
    residual = A_full * x_full - b_full
    err = norm(residual, Inf)

    println(io0(), "  LDLT solve residual (SPD): $err")
    @test err < TOL


    println(io0(), "[test] LDLT factorization - indefinite matrix ($T, $backend_name)")

    n = 8
    A_full = create_symmetric_indefinite(RT, n)  # Symmetric indefinite is real
    A_cpu = SparseMatrixMPI{RT}(A_full)
    A = assert_type(to_backend(A_cpu), ST_real)

    F = ldlt(A)

    b_full = RT.(1:n)
    b = assert_type(to_backend(VectorMPI(b_full)), VT_real)
    x = assert_type(solve(F, b), VT_real)

    x_cpu = TestUtils.to_cpu(x)
    x_full = Vector(x_cpu)
    residual = A_full * x_full - b_full
    err = norm(residual, Inf)

    println(io0(), "  LDLT solve residual (indefinite): $err")
    @test err < TOL


    println(io0(), "[test] Factorization reuse ($T, $backend_name)")

    n = 8
    A_full = create_spd_tridiagonal(RT, n)
    A_cpu = SparseMatrixMPI{RT}(A_full)
    A = assert_type(to_backend(A_cpu), ST_real)
    F = ldlt(A)

    b1_full = ones(RT, n)
    b1 = assert_type(to_backend(VectorMPI(b1_full)), VT_real)
    x1 = assert_type(solve(F, b1), VT_real)

    b2_full = RT.(1:n)
    b2 = assert_type(to_backend(VectorMPI(b2_full)), VT_real)
    x2 = assert_type(solve(F, b2), VT_real)

    x1_cpu = TestUtils.to_cpu(x1)
    x2_cpu = TestUtils.to_cpu(x2)
    x1_full = Vector(x1_cpu)
    x2_full = Vector(x2_cpu)

    err1 = norm(A_full * x1_full - b1_full, Inf)
    err2 = norm(A_full * x2_full - b2_full, Inf)

    println(io0(), "  Residual 1: $err1")
    println(io0(), "  Residual 2: $err2")

    @test err1 < TOL
    @test err2 < TOL

    end  # if backend_name != "CUDA"


    # Transpose/adjoint solve tests only for CPU (requires transpose materialization)
    if backend_name == "CPU"
        println(io0(), "[test] Transpose solve ($T, $backend_name)")

        n = 8
        A_full = create_general_tridiagonal(T, n)
        A = assert_type(SparseMatrixMPI{T}(A_full), ST)
        b_full = ones(T, n)
        b = assert_type(VectorMPI(b_full), VT)

        x_t = assert_type(transpose(A) \ b, VT)

        x_t_full = Vector(x_t)
        residual_t = transpose(A_full) * x_t_full - b_full
        err_t = norm(residual_t, Inf)

        println(io0(), "  Transpose solve residual: $err_t")
        @test err_t < TOL


        println(io0(), "[test] Adjoint solve ($T, $backend_name)")

        x_a = assert_type(A' \ b, VT)

        x_a_full = Vector(x_a)
        residual_a = A_full' * x_a_full - b_full
        err_a = norm(residual_a, Inf)

        println(io0(), "  Adjoint solve residual: $err_a")
        @test err_a < TOL


        println(io0(), "[test] Right division - transpose(v) / A ($T, $backend_name)")

        # transpose(v) / A solves x * A = transpose(v)
        x_rd = transpose(b) / A

        # Verify: x * A should equal transpose(b)
        x_rd_parent = x_rd.parent
        x_rd_full = Vector(x_rd_parent)
        residual_rd = x_rd_full' * A_full - b_full'
        err_rd = norm(residual_rd, Inf)

        println(io0(), "  Right division residual: $err_rd")
        @test err_rd < TOL


        println(io0(), "[test] Right division - transpose(v) / transpose(A) ($T, $backend_name)")

        x_rdt = transpose(b) / transpose(A)
        x_rdt_full = Vector(x_rdt.parent)
        residual_rdt = x_rdt_full' * transpose(A_full) - b_full'
        err_rdt = norm(residual_rdt, Inf)

        println(io0(), "  Right division (transpose) residual: $err_rdt")
        @test err_rdt < TOL
    end


    println(io0(), "[test] LDLT factorization - 2D Laplacian ($T, $backend_name)")

    A_2d_full = create_2d_laplacian(RT, 6, 6)  # 36-element grid
    A_2d_cpu = SparseMatrixMPI{RT}(A_2d_full)
    A_2d = assert_type(to_backend(A_2d_cpu), ST_real)

    F_2d = ldlt(A_2d)

    b_2d_full = ones(RT, 36)
    b_2d = assert_type(to_backend(VectorMPI(b_2d_full)), VT_real)
    x_2d = assert_type(solve(F_2d, b_2d), VT_real)

    x_2d_cpu = TestUtils.to_cpu(x_2d)
    x_2d_full = Vector(x_2d_cpu)
    residual_2d = A_2d_full * x_2d_full - b_2d_full
    err_2d = norm(residual_2d, Inf)

    println(io0(), "  2D Laplacian LDLT residual: $err_2d")
    @test err_2d < TOL


    println(io0(), "[test] LU factorization - 2D Laplacian ($T, $backend_name)")

    A_2d_lu_full = create_2d_laplacian(T, 5, 5)  # 25-element grid
    A_2d_lu_cpu = SparseMatrixMPI{T}(A_2d_lu_full)
    A_2d_lu = assert_type(to_backend(A_2d_lu_cpu), ST)

    F_2d_lu = lu(A_2d_lu)

    b_2d_lu_full = ones(T, 25)
    b_2d_lu = assert_type(to_backend(VectorMPI(b_2d_lu_full)), VT)
    x_2d_lu = assert_type(solve(F_2d_lu, b_2d_lu), VT)

    x_2d_lu_cpu = TestUtils.to_cpu(x_2d_lu)
    x_2d_lu_full = Vector(x_2d_lu_cpu)
    residual_2d_lu = A_2d_lu_full * x_2d_lu_full - b_2d_lu_full
    err_2d_lu = norm(residual_2d_lu, Inf)

    println(io0(), "  2D Laplacian LU residual: $err_2d_lu")
    @test err_2d_lu < TOL


    # Skip block diagonal test for CUDA - uses tridiagonal blocks (cuDSS MGMN bug)
    if backend_name != "CUDA"
    println(io0(), "[test] Block diagonal matrix ($T, $backend_name)")

    block_size = 10
    n_multi = 2 * block_size
    A_multi = spzeros(RT, n_multi, n_multi)
    for b_idx in 0:1
        offset = b_idx * block_size
        for i in 1:block_size
            A_multi[offset + i, offset + i] = RT(4.0)
            if i > 1
                A_multi[offset + i, offset + i - 1] = RT(-1.0)
                A_multi[offset + i - 1, offset + i] = RT(-1.0)
            end
        end
    end
    A_multi_cpu = SparseMatrixMPI{RT}(A_multi)
    A_multi_mpi = assert_type(to_backend(A_multi_cpu), ST_real)

    F_multi = ldlt(A_multi_mpi)

    b_multi_full = ones(RT, n_multi)
    b_multi = assert_type(to_backend(VectorMPI(b_multi_full)), VT_real)
    x_multi = assert_type(solve(F_multi, b_multi), VT_real)
    x_multi_cpu = TestUtils.to_cpu(x_multi)
    x_multi_full = Vector(x_multi_cpu)
    err_multi = norm(A_multi * x_multi_full - b_multi_full, Inf)

    println(io0(), "  Block diagonal LDLT residual: $err_multi")
    @test err_multi < TOL
    end  # if backend_name != "CUDA"


    println(io0(), "[test] Larger problem size (100x100 grid) ($T, $backend_name)")

    A_large_full = create_2d_laplacian(RT, 10, 10)  # 100 DOF
    A_large_cpu = SparseMatrixMPI{RT}(A_large_full)
    A_large = assert_type(to_backend(A_large_cpu), ST_real)

    F_large = ldlt(A_large)

    b_large_full = ones(RT, 100)
    b_large = assert_type(to_backend(VectorMPI(b_large_full)), VT_real)
    x_large = assert_type(solve(F_large, b_large), VT_real)

    x_large_cpu = TestUtils.to_cpu(x_large)
    x_large_full = Vector(x_large_cpu)
    residual_large = A_large_full * x_large_full - b_large_full
    err_large = norm(residual_large, Inf)

    println(io0(), "  100 DOF LDLT residual: $err_large")
    @test err_large < TOL


    # Skip solve! test for CUDA - uses tridiagonal (cuDSS MGMN bug)
    if backend_name != "CUDA"
    println(io0(), "[test] solve! (in-place) ($T, $backend_name)")

    n = 8
    A_full = create_spd_tridiagonal(RT, n)
    A_cpu = SparseMatrixMPI{RT}(A_full)
    A = assert_type(to_backend(A_cpu), ST_real)
    F = ldlt(A)

    b_full = ones(RT, n)
    b = assert_type(to_backend(VectorMPI(b_full)), VT_real)
    x = assert_type(to_backend(VectorMPI(zeros(RT, n))), VT_real)

    solve!(x, F, b)

    x_cpu = TestUtils.to_cpu(x)
    x_full = Vector(x_cpu)
    err = norm(A_full * x_full - b_full, Inf)

    println(io0(), "  solve! residual: $err")
    @test err < TOL
    end  # if backend_name != "CUDA"


    # issymmetric tests only on CPU (doesn't make sense for GPU arrays)
    if backend_name == "CPU"
        println(io0(), "[test] issymmetric with asymmetric partitions - symmetric matrix ($T, $backend_name)")

        # Use size that guarantees different partitions with 4 ranks
        n_asym = 12
        A_sym_full_asym = create_spd_tridiagonal(RT, n_asym)
        row_part = LinearAlgebraMPI.uniform_partition(n_asym, nranks)
        # Create a different valid partition
        col_part = if nranks == 4
            [1, 3, 6, 9, 13]
        else
            rp = copy(row_part)
            for i in 2:length(rp)-1
                if rp[i] + 1 < rp[i+1]
                    rp[i] += 1
                    break
                end
            end
            rp
        end

        A_asym = SparseMatrixMPI{RT}(A_sym_full_asym; row_partition=row_part, col_partition=col_part)
        @test issymmetric(A_asym) == true
        println(io0(), "  Symmetric matrix with asymmetric partitions: passed")


        println(io0(), "[test] issymmetric with asymmetric partitions - non-symmetric matrix ($T, $backend_name)")

        A_nonsym_full_asym = create_general_tridiagonal(RT, n_asym)
        A_nonsym_asym = SparseMatrixMPI{RT}(A_nonsym_full_asym; row_partition=row_part, col_partition=col_part)
        @test issymmetric(A_nonsym_asym) == false
        println(io0(), "  Non-symmetric matrix with asymmetric partitions: passed")
    end


    # Complex-specific tests
    if T <: Complex
        println(io0(), "[test] LU factorization - complex matrix ($T, $backend_name)")

        n = 6
        A_full_real = create_general_tridiagonal(RT, n)
        A_full = Complex{RT}.(A_full_real) + im * spdiagm(0 => RT(0.1)*ones(RT, n))
        A_cpu = SparseMatrixMPI{T}(A_full)
        A = assert_type(to_backend(A_cpu), ST)

        F = lu(A)

        b_full = ones(T, n)
        b = assert_type(to_backend(VectorMPI(b_full)), VT)
        x = assert_type(solve(F, b), VT)

        x_cpu = TestUtils.to_cpu(x)
        x_full = Vector(x_cpu)
        residual = A_full * x_full - b_full
        err = norm(residual, Inf)

        println(io0(), "  LU solve residual (complex): $err")
        @test err < TOL


        println(io0(), "[test] LDLT factorization - complex symmetric ($T, $backend_name)")

        n = 6
        A_cx_full = create_complex_symmetric(T, n)
        A_cx_cpu = SparseMatrixMPI{T}(A_cx_full)
        A_cx = assert_type(to_backend(A_cx_cpu), ST)

        F_cx = ldlt(A_cx)

        b_cx_full = ones(T, n)
        b_cx = assert_type(to_backend(VectorMPI(b_cx_full)), VT)
        x_cx = assert_type(solve(F_cx, b_cx), VT)

        x_cx_cpu = TestUtils.to_cpu(x_cx)
        x_cx_full = Vector(x_cx_cpu)
        residual_cx = A_cx_full * x_cx_full - b_cx_full
        err_cx = norm(residual_cx, Inf)

        println(io0(), "  Complex symmetric LDLT residual: $err_cx")
        @test err_cx < TOL
    end

end  # for (T, to_backend, backend_name)

end  # QuietTestSet

# Aggregate results across ranks
local_counts = [
    get(ts.counts, :pass, 0),
    get(ts.counts, :fail, 0),
    get(ts.counts, :error, 0),
    get(ts.counts, :broken, 0),
    get(ts.counts, :skip, 0),
]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

println(io0(), "Test Summary: distributed factorization | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
