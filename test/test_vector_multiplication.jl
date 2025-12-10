# MPI test for matrix-vector multiplication
# This file is executed under mpiexec by runtests.jl

using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra
using Test

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

const TOL = 1e-12

@testset "Matrix-Vector Multiplication" begin
    # Create deterministic test matrix (same on all ranks)
    n = 8
    # Matrix A: tridiagonal
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = [2.0*ones(Float64, n); -0.5*ones(n-1); -0.5*ones(n-1)]
    A = sparse(I_A, J_A, V_A, n, n)

    # Vector x: simple sequence
    x_global = collect(1.0:n)

    # Create distributed versions
    Adist = SparseMatrixMPI{Float64}(A)
    xdist = VectorMPI(x_global)

    # Compute distributed product
    ydist = Adist * xdist

    # Reference result
    y_ref = A * x_global

    # Check local portion matches
    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    err = maximum(abs.(ydist.v .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ Matrix-vector multiplication: error = $err")
    end
end

@testset "Matrix-Vector Multiplication In-Place" begin
    n = 8
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = [2.0*ones(Float64, n); -0.5*ones(n-1); -0.5*ones(n-1)]
    A = sparse(I_A, J_A, V_A, n, n)

    x_global = collect(1.0:n)

    Adist = SparseMatrixMPI{Float64}(A)
    xdist = VectorMPI(x_global)

    # Create output vector with matching partition
    ydist = VectorMPI(zeros(n))

    # In-place multiplication
    LinearAlgebra.mul!(ydist, Adist, xdist)

    # Reference result
    y_ref = A * x_global

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    err = maximum(abs.(ydist.v .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ Matrix-vector multiplication (in-place): error = $err")
    end
end

@testset "Matrix-Vector Multiplication with ComplexF64" begin
    n = 8
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = ComplexF64.([2.0*ones(n); -0.5*ones(n-1); -0.5*ones(n-1)]) .+
          im .* ComplexF64.([0.1*ones(n); 0.2*ones(n-1); -0.2*ones(n-1)])
    A = sparse(I_A, J_A, V_A, n, n)

    x_global = ComplexF64.(1:n) .+ im .* ComplexF64.(n:-1:1)

    Adist = SparseMatrixMPI{ComplexF64}(A)
    xdist = VectorMPI(x_global)

    ydist = Adist * xdist
    y_ref = A * x_global

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    err = maximum(abs.(ydist.v .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ Matrix-vector multiplication with ComplexF64: error = $err")
    end
end

@testset "Non-square Matrix-Vector Multiplication" begin
    # A is 6x8, x is length 8, y should be length 6
    m, n = 6, 8
    I_A = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
    J_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]
    V_A = Float64.(1:length(I_A))
    A = sparse(I_A, J_A, V_A, m, n)

    x_global = collect(1.0:n)

    Adist = SparseMatrixMPI{Float64}(A)
    xdist = VectorMPI(x_global)

    ydist = Adist * xdist
    y_ref = A * x_global

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    err = maximum(abs.(ydist.v .- local_ref); init=0.0)
    @test err < TOL

    if rank == 0
        println("  ✓ Non-square matrix-vector multiplication: error = $err")
    end
end

@testset "Vector Transpose and Adjoint" begin
    n = 8
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = ComplexF64.([2.0*ones(n); -0.5*ones(n-1); -0.5*ones(n-1)]) .+
          im .* ComplexF64.([0.1*ones(n); 0.2*ones(n-1); -0.2*ones(n-1)])
    A = sparse(I_A, J_A, V_A, n, n)

    x_global = ComplexF64.(1:n) .+ im .* ComplexF64.(n:-1:1)

    Adist = SparseMatrixMPI{ComplexF64}(A)
    xdist = VectorMPI(x_global)

    # Test conj(v)
    xconj = conj(xdist)
    xconj_ref = conj.(x_global)
    my_start = xdist.partition[rank+1]
    my_end = xdist.partition[rank+2] - 1
    err_conj = maximum(abs.(xconj.v .- xconj_ref[my_start:my_end]))
    @test err_conj < TOL

    # Test transpose(v) * A = transpose(transpose(A) * v)
    # This returns a transposed vector
    yt = transpose(xdist) * Adist
    y_ref = transpose(transpose(A) * x_global)

    # yt is Transpose{T, VectorMPI{T}}, so yt.parent.v contains the local values
    # The result should have A's col_partition
    my_col_start = Adist.col_partition[rank+1]
    my_col_end = Adist.col_partition[rank+2] - 1
    local_ref = collect(y_ref)[my_col_start:my_col_end]
    err_transpose = maximum(abs.(yt.parent.v .- local_ref))
    @test err_transpose < TOL

    # Test adjoint: v' * A = transpose(conj(v)) * A = transpose(transpose(A) * conj(v))
    yt_adj = xdist' * Adist
    y_adj_ref = x_global' * A  # This is a row vector

    local_adj_ref = collect(y_adj_ref)[my_col_start:my_col_end]
    err_adjoint = maximum(abs.(yt_adj.parent.v .- local_adj_ref))
    @test err_adjoint < TOL

    if rank == 0
        println("  ✓ Vector transpose and adjoint: errors = $err_conj, $err_transpose, $err_adjoint")
    end
end

@testset "Vector Norms" begin
    n = 10
    x_global = collect(1.0:n)
    xdist = VectorMPI(x_global)

    # 2-norm
    norm2 = norm(xdist)
    norm2_ref = norm(x_global)
    @test abs(norm2 - norm2_ref) < TOL

    # 1-norm
    norm1 = norm(xdist, 1)
    norm1_ref = norm(x_global, 1)
    @test abs(norm1 - norm1_ref) < TOL

    # Inf-norm
    norminf = norm(xdist, Inf)
    norminf_ref = norm(x_global, Inf)
    @test abs(norminf - norminf_ref) < TOL

    # 3-norm (general p)
    norm3 = norm(xdist, 3)
    norm3_ref = norm(x_global, 3)
    @test abs(norm3 - norm3_ref) < TOL

    # Complex vector norms
    z_global = ComplexF64.(1:n) .+ im .* ComplexF64.(n:-1:1)
    zdist = VectorMPI(z_global)

    cnorm2 = norm(zdist)
    cnorm2_ref = norm(z_global)
    @test abs(cnorm2 - cnorm2_ref) < TOL

    if rank == 0
        println("  ✓ Vector norms: 2-norm=$norm2, 1-norm=$norm1, Inf-norm=$norminf")
    end
end

@testset "Vector Reductions" begin
    n = 8
    x_global = collect(1.0:n)
    xdist = VectorMPI(x_global)

    # sum
    s = sum(xdist)
    s_ref = sum(x_global)
    @test abs(s - s_ref) < TOL

    # prod
    p = prod(xdist)
    p_ref = prod(x_global)
    @test abs(p - p_ref) < TOL

    # maximum
    mx = maximum(xdist)
    mx_ref = maximum(x_global)
    @test abs(mx - mx_ref) < TOL

    # minimum
    mn = minimum(xdist)
    mn_ref = minimum(x_global)
    @test abs(mn - mn_ref) < TOL

    if rank == 0
        println("  ✓ Vector reductions: sum=$s, prod=$p, max=$mx, min=$mn")
    end
end

@testset "Vector Addition and Subtraction" begin
    n = 8
    u_global = collect(1.0:n)
    v_global = collect(n:-1.0:1)

    udist = VectorMPI(u_global)
    vdist = VectorMPI(v_global)

    my_start = udist.partition[rank+1]
    my_end = udist.partition[rank+2] - 1

    # u + v
    wdist = udist + vdist
    w_ref = u_global + v_global
    err_add = maximum(abs.(wdist.v .- w_ref[my_start:my_end]))
    @test err_add < TOL

    # u - v
    wdist = udist - vdist
    w_ref = u_global - v_global
    err_sub = maximum(abs.(wdist.v .- w_ref[my_start:my_end]))
    @test err_sub < TOL

    # -v
    wdist = -vdist
    w_ref = -v_global
    err_neg = maximum(abs.(wdist.v .- w_ref[my_start:my_end]))
    @test err_neg < TOL

    if rank == 0
        println("  ✓ Vector addition/subtraction: add=$err_add, sub=$err_sub, neg=$err_neg")
    end
end

@testset "Vector Operations with Different Partitions" begin
    # Clear cache to avoid interference from previous tests
    LinearAlgebraMPI.clear_plan_cache!()

    n = 12
    u_global = collect(1.0:n)
    v_global = collect(n:-1.0:1)

    # Create u with default partition
    udist = VectorMPI(u_global)

    # Create v with a different (custom) partition
    # Default partition for 4 ranks on 12 elements: [1, 4, 7, 10, 13]
    # Use a shifted partition: [1, 3, 6, 9, 13] (different sizes)
    custom_partition = [1, 3, 6, 9, 13]
    v_hash = LinearAlgebraMPI.compute_partition_hash(custom_partition)
    local_v_range = custom_partition[rank+1]:(custom_partition[rank+2]-1)
    vdist = VectorMPI{Float64}(v_hash, custom_partition, v_global[local_v_range])

    # Verify partitions are different
    @test udist.partition != vdist.partition

    # u + v (result should have u's partition)
    wdist = udist + vdist
    w_ref = u_global + v_global
    my_start = udist.partition[rank+1]
    my_end = udist.partition[rank+2] - 1
    err_add = maximum(abs.(wdist.v .- w_ref[my_start:my_end]))
    @test err_add < TOL
    @test wdist.partition == udist.partition

    # u - v
    wdist = udist - vdist
    w_ref = u_global - v_global
    err_sub = maximum(abs.(wdist.v .- w_ref[my_start:my_end]))
    @test err_sub < TOL

    # v + u (result should have v's partition)
    wdist2 = vdist + udist
    w_ref2 = v_global + u_global
    my_v_start = vdist.partition[rank+1]
    my_v_end = vdist.partition[rank+2] - 1
    err_add2 = maximum(abs.(wdist2.v .- w_ref2[my_v_start:my_v_end]))
    @test err_add2 < TOL
    @test wdist2.partition == vdist.partition

    # transpose(u) + transpose(v) with different partitions
    wt = transpose(udist) + transpose(vdist)
    w_add_ref = u_global + v_global  # Reset reference for addition
    err_tadd = maximum(abs.(wt.parent.v .- w_add_ref[my_start:my_end]))
    @test err_tadd < TOL

    if rank == 0
        println("  ✓ Different partition ops: add=$err_add, sub=$err_sub, reverse_add=$err_add2, transpose_add=$err_tadd")
    end
end

@testset "Transposed Vector Addition and Subtraction" begin
    n = 8
    u_global = collect(1.0:n)
    v_global = collect(n:-1.0:1)

    udist = VectorMPI(u_global)
    vdist = VectorMPI(v_global)

    my_start = udist.partition[rank+1]
    my_end = udist.partition[rank+2] - 1

    # transpose(u) + transpose(v)
    wt = transpose(udist) + transpose(vdist)
    w_ref = u_global + v_global
    err_add = maximum(abs.(wt.parent.v .- w_ref[my_start:my_end]))
    @test err_add < TOL

    # transpose(u) - transpose(v)
    wt = transpose(udist) - transpose(vdist)
    w_ref = u_global - v_global
    err_sub = maximum(abs.(wt.parent.v .- w_ref[my_start:my_end]))
    @test err_sub < TOL

    # -transpose(v)
    wt = -transpose(vdist)
    w_ref = -v_global
    err_neg = maximum(abs.(wt.parent.v .- w_ref[my_start:my_end]))
    @test err_neg < TOL

    if rank == 0
        println("  ✓ Transposed vector add/sub: add=$err_add, sub=$err_sub, neg=$err_neg")
    end
end

@testset "Scalar Multiplication" begin
    n = 8
    v_global = collect(1.0:n)
    vdist = VectorMPI(v_global)
    a = 3.5

    my_start = vdist.partition[rank+1]
    my_end = vdist.partition[rank+2] - 1

    # a * v
    wdist = a * vdist
    w_ref = a * v_global
    err_av = maximum(abs.(wdist.v .- w_ref[my_start:my_end]))
    @test err_av < TOL

    # v * a
    wdist = vdist * a
    err_va = maximum(abs.(wdist.v .- w_ref[my_start:my_end]))
    @test err_va < TOL

    # v / a
    wdist = vdist / a
    w_ref = v_global / a
    err_div = maximum(abs.(wdist.v .- w_ref[my_start:my_end]))
    @test err_div < TOL

    # a * transpose(v)
    wt = a * transpose(vdist)
    w_ref = a * v_global
    err_avt = maximum(abs.(wt.parent.v .- w_ref[my_start:my_end]))
    @test err_avt < TOL

    # transpose(v) * a
    wt = transpose(vdist) * a
    err_vta = maximum(abs.(wt.parent.v .- w_ref[my_start:my_end]))
    @test err_vta < TOL

    # transpose(v) / a
    wt = transpose(vdist) / a
    w_ref = v_global / a
    err_vtdiv = maximum(abs.(wt.parent.v .- w_ref[my_start:my_end]))
    @test err_vtdiv < TOL

    if rank == 0
        println("  ✓ Scalar multiplication: a*v=$err_av, v*a=$err_va, v/a=$err_div")
        println("                           a*transpose(v)=$err_avt, transpose(v)*a=$err_vta, transpose(v)/a=$err_vtdiv")
    end
end

@testset "Vector Size and Eltype" begin
    n = 8
    v_global = collect(1.0:n)
    vdist = VectorMPI(v_global)

    @test length(vdist) == n
    @test size(vdist) == (n,)
    @test size(vdist, 1) == n
    @test eltype(vdist) == Float64
    @test eltype(VectorMPI{ComplexF64}) == ComplexF64

    if rank == 0
        println("  ✓ Vector size/eltype: length=$n, size=$(size(vdist)), eltype=$(eltype(vdist))")
    end
end

MPI.Finalize()
