# MPI test for dense matrix (MatrixMPI) operations
# This file is executed under mpiexec by runtests.jl

using MPI
MPI.Init()

using LinearAlgebraMPI
using LinearAlgebra
using Test

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

const TOL = 1e-12

@testset "MatrixMPI Construction" begin
    m, n = 8, 6
    # Deterministic matrix
    A = Float64.([i + j for i in 1:m, j in 1:n])

    Adist = MatrixMPI(A)

    @test size(Adist) == (m, n)
    @test size(Adist, 1) == m
    @test size(Adist, 2) == n
    @test eltype(Adist) == Float64

    # Check local rows
    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = A[my_start:my_end, :]
    err = maximum(abs.(Adist.A .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ MatrixMPI construction: size=$(size(Adist)), error=$err")
    end
end

@testset "MatrixMPI Vector Multiplication" begin
    m, n = 8, 6
    A = Float64.([i + j for i in 1:m, j in 1:n])
    x_global = collect(1.0:n)

    Adist = MatrixMPI(A)
    xdist = VectorMPI(x_global)

    ydist = Adist * xdist
    y_ref = A * x_global

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    err = maximum(abs.(ydist.v .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ MatrixMPI * VectorMPI: error = $err")
    end
end

@testset "MatrixMPI Vector Multiplication In-Place" begin
    m, n = 8, 6
    A = Float64.([i + j for i in 1:m, j in 1:n])
    x_global = collect(1.0:n)

    Adist = MatrixMPI(A)
    xdist = VectorMPI(x_global)
    ydist = VectorMPI(zeros(m))

    LinearAlgebra.mul!(ydist, Adist, xdist)
    y_ref = A * x_global

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    err = maximum(abs.(ydist.v .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ MatrixMPI mul! (in-place): error = $err")
    end
end

@testset "MatrixMPI with ComplexF64" begin
    m, n = 8, 6
    A = ComplexF64.([i + j for i in 1:m, j in 1:n]) .+ im .* ComplexF64.([i - j for i in 1:m, j in 1:n])
    x_global = ComplexF64.(1:n) .+ im .* ComplexF64.(n:-1:1)

    Adist = MatrixMPI(A)
    xdist = VectorMPI(x_global)

    ydist = Adist * xdist
    y_ref = A * x_global

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    err = maximum(abs.(ydist.v .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ MatrixMPI with ComplexF64: error = $err")
    end
end

@testset "MatrixMPI Transpose * Vector" begin
    m, n = 8, 6
    A = Float64.([i + j for i in 1:m, j in 1:n])
    x_global = collect(1.0:m)

    Adist = MatrixMPI(A)
    xdist = VectorMPI(x_global)

    # transpose(A) * x
    ydist = transpose(Adist) * xdist
    y_ref = transpose(A) * x_global

    my_col_start = Adist.col_partition[rank+1]
    my_col_end = Adist.col_partition[rank+2] - 1
    local_ref = y_ref[my_col_start:my_col_end]

    err = maximum(abs.(ydist.v .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ transpose(MatrixMPI) * VectorMPI: error = $err")
    end
end

@testset "MatrixMPI Adjoint * Vector" begin
    m, n = 8, 6
    A = ComplexF64.([i + j for i in 1:m, j in 1:n]) .+ im .* ComplexF64.([i - j for i in 1:m, j in 1:n])
    x_global = ComplexF64.(1:m) .+ im .* ComplexF64.(m:-1:1)

    Adist = MatrixMPI(A)
    xdist = VectorMPI(x_global)

    # adjoint(A) * x
    ydist = adjoint(Adist) * xdist
    y_ref = adjoint(A) * x_global

    my_col_start = Adist.col_partition[rank+1]
    my_col_end = Adist.col_partition[rank+2] - 1
    local_ref = y_ref[my_col_start:my_col_end]

    err = maximum(abs.(ydist.v .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ adjoint(MatrixMPI) * VectorMPI: error = $err")
    end
end

@testset "Vector * MatrixMPI" begin
    m, n = 8, 6
    A = Float64.([i + j for i in 1:m, j in 1:n])
    x_global = collect(1.0:m)

    Adist = MatrixMPI(A)
    xdist = VectorMPI(x_global)

    # transpose(v) * A
    yt = transpose(xdist) * Adist
    y_ref = transpose(x_global) * A

    my_col_start = Adist.col_partition[rank+1]
    my_col_end = Adist.col_partition[rank+2] - 1
    local_ref = collect(y_ref)[my_col_start:my_col_end]

    err = maximum(abs.(yt.parent.v .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ transpose(VectorMPI) * MatrixMPI: error = $err")
    end
end

@testset "Vector Adjoint * MatrixMPI" begin
    m, n = 8, 6
    A = ComplexF64.([i + j for i in 1:m, j in 1:n]) .+ im .* ComplexF64.([i - j for i in 1:m, j in 1:n])
    x_global = ComplexF64.(1:m) .+ im .* ComplexF64.(m:-1:1)

    Adist = MatrixMPI(A)
    xdist = VectorMPI(x_global)

    # v' * A
    yt = xdist' * Adist
    y_ref = x_global' * A

    my_col_start = Adist.col_partition[rank+1]
    my_col_end = Adist.col_partition[rank+2] - 1
    local_ref = collect(y_ref)[my_col_start:my_col_end]

    err = maximum(abs.(yt.parent.v .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ VectorMPI' * MatrixMPI: error = $err")
    end
end

@testset "MatrixMPI Transpose Materialization" begin
    m, n = 8, 6
    A = Float64.([i + j for i in 1:m, j in 1:n])

    Adist = MatrixMPI(A)
    At_dist = copy(transpose(Adist))

    At_ref = transpose(A)
    @test size(At_dist) == (n, m)

    my_start = At_dist.row_partition[rank+1]
    my_end = At_dist.row_partition[rank+2] - 1
    local_ref = At_ref[my_start:my_end, :]

    err = maximum(abs.(At_dist.A .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ MatrixMPI transpose materialization: error = $err")
    end
end

@testset "MatrixMPI Adjoint Materialization" begin
    m, n = 8, 6
    A = ComplexF64.([i + j for i in 1:m, j in 1:n]) .+ im .* ComplexF64.([i - j for i in 1:m, j in 1:n])

    Adist = MatrixMPI(A)
    Ah_dist = copy(adjoint(Adist))

    Ah_ref = adjoint(A)
    @test size(Ah_dist) == (n, m)

    my_start = Ah_dist.row_partition[rank+1]
    my_end = Ah_dist.row_partition[rank+2] - 1
    local_ref = Ah_ref[my_start:my_end, :]

    err = maximum(abs.(Ah_dist.A .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ MatrixMPI adjoint materialization: error = $err")
    end
end

@testset "MatrixMPI Scalar Multiplication" begin
    m, n = 8, 6
    A = Float64.([i + j for i in 1:m, j in 1:n])
    a = 3.5

    Adist = MatrixMPI(A)

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1

    # a * A
    Bdist = a * Adist
    B_ref = a * A
    err_aA = maximum(abs.(Bdist.A .- B_ref[my_start:my_end, :]))
    @test err_aA < TOL

    # A * a
    Bdist = Adist * a
    err_Aa = maximum(abs.(Bdist.A .- B_ref[my_start:my_end, :]))
    @test err_Aa < TOL

    # a * transpose(A)
    Ct = a * transpose(Adist)
    @test isa(Ct, Transpose)

    # transpose(A) * a
    Ct = transpose(Adist) * a
    @test isa(Ct, Transpose)

    if rank == 0
        println("  ✓ MatrixMPI scalar multiplication: a*A=$err_aA, A*a=$err_Aa")
    end
end

@testset "MatrixMPI Conj" begin
    m, n = 8, 6
    A = ComplexF64.([i + j for i in 1:m, j in 1:n]) .+ im .* ComplexF64.([i - j for i in 1:m, j in 1:n])

    Adist = MatrixMPI(A)
    Aconj_dist = conj(Adist)

    Aconj_ref = conj.(A)
    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = Aconj_ref[my_start:my_end, :]

    err = maximum(abs.(Aconj_dist.A .- local_ref))
    @test err < TOL

    if rank == 0
        println("  ✓ MatrixMPI conj: error = $err")
    end
end

@testset "MatrixMPI Norms" begin
    m, n = 8, 6
    A = Float64.([i + j for i in 1:m, j in 1:n])

    Adist = MatrixMPI(A)

    # Frobenius norm (2-norm)
    norm2 = norm(Adist)
    norm2_ref = norm(A)
    @test abs(norm2 - norm2_ref) < TOL

    # 1-norm (element-wise)
    norm1 = norm(Adist, 1)
    norm1_ref = norm(A, 1)
    @test abs(norm1 - norm1_ref) < TOL

    # Inf-norm (element-wise)
    norminf = norm(Adist, Inf)
    norminf_ref = norm(A, Inf)
    @test abs(norminf - norminf_ref) < TOL

    if rank == 0
        println("  ✓ MatrixMPI norms: 2-norm=$norm2, 1-norm=$norm1, Inf-norm=$norminf")
    end
end

@testset "MatrixMPI Operator Norms" begin
    m, n = 8, 6
    A = Float64.([i + j for i in 1:m, j in 1:n])

    Adist = MatrixMPI(A)

    # 1-norm (max column sum)
    opnorm1 = opnorm(Adist, 1)
    opnorm1_ref = opnorm(A, 1)
    @test abs(opnorm1 - opnorm1_ref) < TOL

    # Inf-norm (max row sum)
    opnorminf = opnorm(Adist, Inf)
    opnorminf_ref = opnorm(A, Inf)
    @test abs(opnorminf - opnorminf_ref) < TOL

    if rank == 0
        println("  ✓ MatrixMPI operator norms: 1-norm=$opnorm1, Inf-norm=$opnorminf")
    end
end

@testset "Square MatrixMPI Operations" begin
    n = 8
    A = Float64.([i + j for i in 1:n, j in 1:n])
    x_global = collect(1.0:n)

    Adist = MatrixMPI(A)
    xdist = VectorMPI(x_global)

    # A * x
    ydist = Adist * xdist
    y_ref = A * x_global

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    err = maximum(abs.(ydist.v .- local_ref))
    @test err < TOL

    # transpose(A) * x (same partition since square)
    ydist_t = transpose(Adist) * xdist
    y_ref_t = transpose(A) * x_global

    err_t = maximum(abs.(ydist_t.v .- y_ref_t[my_start:my_end]))
    @test err_t < TOL

    if rank == 0
        println("  ✓ Square MatrixMPI: A*x error=$err, transpose(A)*x error=$err_t")
    end
end

MPI.Finalize()
