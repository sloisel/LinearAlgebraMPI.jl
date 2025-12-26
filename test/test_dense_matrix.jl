# MPI test for dense matrix (MatrixMPI) operations
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
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Helper function to gather a MatrixMPI back to a global matrix for testing
function gather_matrix(A::MatrixMPI{T}) where T
    m = A.row_partition[end] - 1
    n = size(A.A, 2)
    counts = Int32[A.row_partition[r+1] - A.row_partition[r] for r in 1:nranks]

    # Gather each column
    result = Matrix{T}(undef, m, n)
    for j in 1:n
        col_data = Array(A.A[:, j])  # Array() handles GPU arrays
        full_col = Vector{T}(undef, m)
        MPI.Allgatherv!(col_data, MPI.VBuffer(full_col, counts), comm)
        result[:, j] = full_col
    end
    return result
end

ts = @testset QuietTestSet "Dense Matrix" begin

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)

    println(io0(), "[test] MatrixMPI construction ($T, $backend_name)")

    m, n = 8, 6
    A = TestUtils.dense_matrix(T, m, n)

    Adist = to_backend(MatrixMPI(A))

    @test size(Adist) == (m, n)
    @test size(Adist, 1) == m
    @test size(Adist, 2) == n
    @test eltype(Adist) == T

    # Check local rows
    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = A[my_start:my_end, :]
    local_A = Array(Adist.A)
    err = maximum(abs.(local_A .- local_ref))
    @test err < TOL


    println(io0(), "[test] MatrixMPI * VectorMPI ($T, $backend_name)")

    m, n = 8, 6
    A = TestUtils.dense_matrix(T, m, n)
    x_global = TestUtils.test_vector(T, n)

    Adist = to_backend(MatrixMPI(A))
    xdist = to_backend(VectorMPI(x_global))

    ydist = Adist * xdist
    y_ref = A * x_global

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    local_y = TestUtils.local_values(ydist)
    err = maximum(abs.(local_y .- local_ref))
    @test err < TOL


    println(io0(), "[test] MatrixMPI mul! in-place ($T, $backend_name)")

    m, n = 8, 6
    A = TestUtils.dense_matrix(T, m, n)
    x_global = TestUtils.test_vector(T, n)

    Adist = to_backend(MatrixMPI(A))
    xdist = to_backend(VectorMPI(x_global))
    ydist = to_backend(VectorMPI(zeros(T, m)))

    LinearAlgebra.mul!(ydist, Adist, xdist)
    y_ref = A * x_global

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    local_y = TestUtils.local_values(ydist)
    err = maximum(abs.(local_y .- local_ref))
    @test err < TOL


    println(io0(), "[test] transpose(MatrixMPI) * VectorMPI ($T, $backend_name)")

    m, n = 8, 6
    A = TestUtils.dense_matrix(T, m, n)
    x_global = TestUtils.test_vector(T, m)

    Adist = to_backend(MatrixMPI(A))
    xdist = to_backend(VectorMPI(x_global))

    # transpose(A) * x
    ydist = transpose(Adist) * xdist
    y_ref = transpose(A) * x_global

    my_col_start = Adist.col_partition[rank+1]
    my_col_end = Adist.col_partition[rank+2] - 1
    local_ref = y_ref[my_col_start:my_col_end]

    local_y = TestUtils.local_values(ydist)
    err = maximum(abs.(local_y .- local_ref))
    @test err < TOL


    # Adjoint tests only meaningful for complex
    if T <: Complex
        println(io0(), "[test] adjoint(MatrixMPI) * VectorMPI ($T, $backend_name)")

        m, n = 8, 6
        A = TestUtils.dense_matrix(T, m, n)
        x_global = TestUtils.test_vector(T, m)

        Adist = to_backend(MatrixMPI(A))
        xdist = to_backend(VectorMPI(x_global))

        # adjoint(A) * x
        ydist = adjoint(Adist) * xdist
        y_ref = adjoint(A) * x_global

        my_col_start = Adist.col_partition[rank+1]
        my_col_end = Adist.col_partition[rank+2] - 1
        local_ref = y_ref[my_col_start:my_col_end]

        local_y = TestUtils.local_values(ydist)
        err = maximum(abs.(local_y .- local_ref))
        @test err < TOL
    end


    println(io0(), "[test] transpose(VectorMPI) * MatrixMPI ($T, $backend_name)")

    m, n = 8, 6
    A = TestUtils.dense_matrix(T, m, n)
    x_global = TestUtils.test_vector(T, m)

    Adist = to_backend(MatrixMPI(A))
    xdist = to_backend(VectorMPI(x_global))

    # transpose(v) * A
    yt = transpose(xdist) * Adist
    y_ref = transpose(x_global) * A

    my_col_start = Adist.col_partition[rank+1]
    my_col_end = Adist.col_partition[rank+2] - 1
    local_ref = collect(y_ref)[my_col_start:my_col_end]

    local_yt = TestUtils.local_values(yt.parent)
    err = maximum(abs.(local_yt .- local_ref))
    @test err < TOL


    println(io0(), "[test] VectorMPI' * MatrixMPI ($T, $backend_name)")

    m, n = 8, 6
    A = TestUtils.dense_matrix(T, m, n)
    x_global = TestUtils.test_vector(T, m)

    Adist = to_backend(MatrixMPI(A))
    xdist = to_backend(VectorMPI(x_global))

    # v' * A
    yt = xdist' * Adist
    y_ref = x_global' * A

    my_col_start = Adist.col_partition[rank+1]
    my_col_end = Adist.col_partition[rank+2] - 1
    local_ref = collect(y_ref)[my_col_start:my_col_end]

    local_yt = TestUtils.local_values(yt.parent)
    err = maximum(abs.(local_yt .- local_ref))
    @test err < TOL


    println(io0(), "[test] MatrixMPI transpose materialization ($T, $backend_name)")

    m, n = 8, 6
    A = TestUtils.dense_matrix(T, m, n)

    Adist = to_backend(MatrixMPI(A))
    At_dist = copy(transpose(Adist))

    At_ref = transpose(A)
    @test size(At_dist) == (n, m)

    my_start = At_dist.row_partition[rank+1]
    my_end = At_dist.row_partition[rank+2] - 1
    local_ref = At_ref[my_start:my_end, :]

    local_At = Array(At_dist.A)
    err = maximum(abs.(local_At .- local_ref))
    @test err < TOL


    if T <: Complex
        println(io0(), "[test] MatrixMPI adjoint materialization ($T, $backend_name)")

        m, n = 8, 6
        A = TestUtils.dense_matrix(T, m, n)

        Adist = to_backend(MatrixMPI(A))
        Ah_dist = copy(adjoint(Adist))

        Ah_ref = adjoint(A)
        @test size(Ah_dist) == (n, m)

        my_start = Ah_dist.row_partition[rank+1]
        my_end = Ah_dist.row_partition[rank+2] - 1
        local_ref = Ah_ref[my_start:my_end, :]

        local_Ah = Array(Ah_dist.A)
        err = maximum(abs.(local_Ah .- local_ref))
        @test err < TOL
    end


    println(io0(), "[test] MatrixMPI scalar multiplication ($T, $backend_name)")

    m, n = 8, 6
    A = TestUtils.dense_matrix(T, m, n)
    a = T <: Complex ? T(3.5 + 0.5im) : T(3.5)

    Adist = to_backend(MatrixMPI(A))

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1

    # a * A
    Bdist = a * Adist
    B_ref = a * A
    local_B = Array(Bdist.A)
    err_aA = maximum(abs.(local_B .- B_ref[my_start:my_end, :]))
    @test err_aA < TOL

    # A * a
    Bdist = Adist * a
    local_B = Array(Bdist.A)
    err_Aa = maximum(abs.(local_B .- B_ref[my_start:my_end, :]))
    @test err_Aa < TOL

    # a * transpose(A)
    Ct = a * transpose(Adist)
    @test isa(Ct, Transpose)

    # transpose(A) * a
    Ct = transpose(Adist) * a
    @test isa(Ct, Transpose)


    if T <: Complex
        println(io0(), "[test] MatrixMPI conj ($T, $backend_name)")

        m, n = 8, 6
        A = TestUtils.dense_matrix(T, m, n)

        Adist = to_backend(MatrixMPI(A))
        Aconj_dist = conj(Adist)

        Aconj_ref = conj.(A)
        my_start = Adist.row_partition[rank+1]
        my_end = Adist.row_partition[rank+2] - 1
        local_ref = Aconj_ref[my_start:my_end, :]

        local_Aconj = Array(Aconj_dist.A)
        err = maximum(abs.(local_Aconj .- local_ref))
        @test err < TOL
    end


    println(io0(), "[test] MatrixMPI norms ($T, $backend_name)")

    m, n = 8, 6
    A = TestUtils.dense_matrix(real(T), m, n)  # Use real type for norm tests

    Adist = to_backend(MatrixMPI(A))
    Adist_cpu = TestUtils.to_cpu(Adist)

    # Frobenius norm (2-norm)
    norm2 = norm(Adist_cpu)
    norm2_ref = norm(A)
    @test abs(norm2 - norm2_ref) < TOL

    # 1-norm (element-wise)
    norm1 = norm(Adist_cpu, 1)
    norm1_ref = norm(A, 1)
    @test abs(norm1 - norm1_ref) < TOL

    # Inf-norm (element-wise)
    norminf = norm(Adist_cpu, Inf)
    norminf_ref = norm(A, Inf)
    @test abs(norminf - norminf_ref) < TOL


    println(io0(), "[test] MatrixMPI operator norms ($T, $backend_name)")

    m, n = 8, 6
    A = TestUtils.dense_matrix(real(T), m, n)

    Adist = to_backend(MatrixMPI(A))
    Adist_cpu = TestUtils.to_cpu(Adist)

    # 1-norm (max column sum)
    opnorm1 = opnorm(Adist_cpu, 1)
    opnorm1_ref = opnorm(A, 1)
    @test abs(opnorm1 - opnorm1_ref) < TOL

    # Inf-norm (max row sum)
    opnorminf = opnorm(Adist_cpu, Inf)
    opnorminf_ref = opnorm(A, Inf)
    @test abs(opnorminf - opnorminf_ref) < TOL


    println(io0(), "[test] Square MatrixMPI operations ($T, $backend_name)")

    n = 8
    A = TestUtils.dense_matrix(T, n, n)
    x_global = TestUtils.test_vector(T, n)

    Adist = to_backend(MatrixMPI(A))
    xdist = to_backend(VectorMPI(x_global))

    # A * x
    ydist = Adist * xdist
    y_ref = A * x_global

    my_start = Adist.row_partition[rank+1]
    my_end = Adist.row_partition[rank+2] - 1
    local_ref = y_ref[my_start:my_end]

    local_y = TestUtils.local_values(ydist)
    err = maximum(abs.(local_y .- local_ref))
    @test err < TOL

    # transpose(A) * x (same partition since square)
    ydist_t = transpose(Adist) * xdist
    y_ref_t = transpose(A) * x_global

    local_yt = TestUtils.local_values(ydist_t)
    err_t = maximum(abs.(local_yt .- y_ref_t[my_start:my_end]))
    @test err_t < TOL

end  # for (T, to_backend, backend_name)


# mapslices tests
for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)

    println(io0(), "[test] mapslices dims=2 row-wise ($T, $backend_name)")

    m, n = 8, 5
    A = TestUtils.dense_matrix(real(T), m, n)

    Adist = MatrixMPI(A)

    # Function that transforms each row: 5 elements -> 3 elements
    f_row = x -> [norm(x), maximum(x), sum(x)]

    Bdist = mapslices(f_row, Adist; dims=2)
    B_ref = mapslices(f_row, A; dims=2)

    @test size(Bdist) == size(B_ref)

    gathered = gather_matrix(Bdist)
    err = maximum(abs.(gathered .- B_ref))
    @test err < TOL


    println(io0(), "[test] mapslices dims=1 column-wise ($T, $backend_name)")

    m, n = 8, 5
    A = TestUtils.dense_matrix(real(T), m, n)

    Adist = MatrixMPI(A)

    # Function that transforms each column: 8 elements -> 2 elements
    f_col = x -> [norm(x), maximum(x)]

    Bdist = mapslices(f_col, Adist; dims=1)
    B_ref = mapslices(f_col, A; dims=1)

    @test size(Bdist) == size(B_ref)

    gathered = gather_matrix(Bdist)
    err = maximum(abs.(gathered .- B_ref))
    @test err < TOL


    println(io0(), "[test] mapslices dims=2 preserves row partition ($T, $backend_name)")

    m, n = 8, 5
    A = TestUtils.dense_matrix(real(T), m, n)

    Adist = MatrixMPI(A)

    f_partition = x -> [norm(x), maximum(x)]
    Bdist = mapslices(f_partition, Adist; dims=2)

    # dims=2 preserves row partition
    @test Bdist.row_partition == Adist.row_partition

end  # mapslices tests

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

println(io0(), "Test Summary: Dense Matrix | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
