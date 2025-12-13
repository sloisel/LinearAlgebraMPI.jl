# MPI test for dense matrix (MatrixMPI) operations
# This file is executed under mpiexec by runtests.jl

using MPI
MPI.Init()

using LinearAlgebraMPI
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

const TOL = 1e-12

# Helper function to gather a MatrixMPI back to a global matrix for testing
function gather_matrix(A::MatrixMPI{T}) where T
    m = A.row_partition[end] - 1
    n = size(A.A, 2)
    counts = Int32[A.row_partition[r+1] - A.row_partition[r] for r in 1:nranks]

    # Gather each column
    result = Matrix{T}(undef, m, n)
    for j in 1:n
        col_data = A.A[:, j]
        full_col = Vector{T}(undef, m)
        MPI.Allgatherv!(col_data, MPI.VBuffer(full_col, counts), comm)
        result[:, j] = full_col
    end
    return result
end

ts = @testset QuietTestSet "Dense Matrix" begin

if rank == 0
    println("[test] MatrixMPI construction")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI * VectorMPI")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI mul! (in-place)")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI with ComplexF64")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] transpose(MatrixMPI) * VectorMPI")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] adjoint(MatrixMPI) * VectorMPI")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] transpose(VectorMPI) * MatrixMPI")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI' * MatrixMPI (Float64)")
    flush(stdout)
end

m, n = 8, 6
A = Float64.([i + j for i in 1:m, j in 1:n])
x_global = collect(1.0:m)

Adist = MatrixMPI(A)
xdist = VectorMPI(x_global)

# v' * A (Float64)
yt = xdist' * Adist
y_ref = x_global' * A

my_col_start = Adist.col_partition[rank+1]
my_col_end = Adist.col_partition[rank+2] - 1
local_ref = collect(y_ref)[my_col_start:my_col_end]

err = maximum(abs.(yt.parent.v .- local_ref))
@test err < TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI' * MatrixMPI (ComplexF64)")
    flush(stdout)
end

m, n = 8, 6
A = ComplexF64.([i + j for i in 1:m, j in 1:n]) .+ im .* ComplexF64.([i - j for i in 1:m, j in 1:n])
x_global = ComplexF64.(1:m) .+ im .* ComplexF64.(m:-1:1)

Adist = MatrixMPI(A)
xdist = VectorMPI(x_global)

# v' * A (ComplexF64)
yt = xdist' * Adist
y_ref = x_global' * A

my_col_start = Adist.col_partition[rank+1]
my_col_end = Adist.col_partition[rank+2] - 1
local_ref = collect(y_ref)[my_col_start:my_col_end]

err = maximum(abs.(yt.parent.v .- local_ref))
@test err < TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI transpose materialization")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI adjoint materialization")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI scalar multiplication")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI conj")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI norms")
    flush(stdout)
end

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

# Non-integer p-norm (p = 1.5)
norm15 = norm(Adist, 1.5)
norm15_ref = norm(A, 1.5)
@test abs(norm15 - norm15_ref) < TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI operator norms")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] Square MatrixMPI operations")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] mapslices dims=2 (row-wise)")
    flush(stdout)
end

m, n = 8, 5
A = Float64.([i + 0.1*j for i in 1:m, j in 1:n])

Adist = MatrixMPI(A)

# Function that transforms each row: 5 elements -> 3 elements
f_row = x -> [norm(x), maximum(x), sum(x)]

Bdist = mapslices(f_row, Adist; dims=2)
B_ref = mapslices(f_row, A; dims=2)

@test size(Bdist) == size(B_ref)

gathered = gather_matrix(Bdist)
err = maximum(abs.(gathered .- B_ref))
@test err < TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] mapslices dims=1 (column-wise)")
    flush(stdout)
end

m, n = 8, 5
A = Float64.([i + 0.1*j for i in 1:m, j in 1:n])

Adist = MatrixMPI(A)

# Function that transforms each column: 8 elements -> 2 elements
f_col = x -> [norm(x), maximum(x)]

Bdist = mapslices(f_col, Adist; dims=1)
B_ref = mapslices(f_col, A; dims=1)

@test size(Bdist) == size(B_ref)

gathered = gather_matrix(Bdist)
err = maximum(abs.(gathered .- B_ref))
@test err < TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] mapslices dims=2 preserves row partition")
    flush(stdout)
end

m, n = 8, 5
A = Float64.([i + 0.1*j for i in 1:m, j in 1:n])

Adist = MatrixMPI(A)

f_partition = x -> [norm(x), maximum(x)]
Bdist = mapslices(f_partition, Adist; dims=2)

# dims=2 preserves row partition
@test Bdist.row_partition == Adist.row_partition

MPI.Barrier(comm)

if rank == 0
    println("[test] mapslices with ComplexF64")
    flush(stdout)
end

m, n = 8, 5
A = ComplexF64.([i + 0.1*j for i in 1:m, j in 1:n]) .+ im .* ComplexF64.([i - j for i in 1:m, j in 1:n])

Adist = MatrixMPI(A)

# Function that returns real values
f_complex = x -> [norm(x), abs(maximum(real.(x)))]

Bdist = mapslices(f_complex, Adist; dims=2)
B_ref = mapslices(f_complex, A; dims=2)

@test size(Bdist) == size(B_ref)

gathered = gather_matrix(Bdist)
err = maximum(abs.(gathered .- B_ref))
@test err < TOL

MPI.Barrier(comm)

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

if rank == 0
    println("Test Summary: Dense Matrix | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")
    flush(stdout)
end

MPI.Barrier(comm)
MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
