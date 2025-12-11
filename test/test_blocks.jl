using MPI
using LinearAlgebraMPI
using SparseArrays
using Test
using Random

MPI.Init()

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Use fixed seed for reproducibility across ranks
Random.seed!(42)

# Helper to gather SparseMatrixMPI back to global SparseMatrixCSC for comparison
function gather_sparse(Adist::SparseMatrixMPI{T}) where T
    m, n = size(Adist)

    # Collect local triplets
    my_I = Int[]
    my_J = Int[]
    my_V = T[]

    my_row_start = Adist.row_partition[rank + 1]
    col_indices = Adist.col_indices
    for local_row in 1:size(Adist.A.parent, 2)
        global_row = my_row_start + local_row - 1
        for idx in Adist.A.parent.colptr[local_row]:(Adist.A.parent.colptr[local_row + 1] - 1)
            push!(my_I, global_row)
            local_col = Adist.A.parent.rowval[idx]
            push!(my_J, col_indices[local_col])  # map local to global
            push!(my_V, Adist.A.parent.nzval[idx])
        end
    end

    # Gather counts
    local_nnz = Int32(length(my_I))
    all_nnz = MPI.Allgather(local_nnz, comm)

    # Gather triplets
    total_nnz = sum(all_nnz)
    all_I = Vector{Int}(undef, total_nnz)
    all_J = Vector{Int}(undef, total_nnz)
    all_V = Vector{T}(undef, total_nnz)

    MPI.Allgatherv!(my_I, MPI.VBuffer(all_I, all_nnz), comm)
    MPI.Allgatherv!(my_J, MPI.VBuffer(all_J, all_nnz), comm)
    MPI.Allgatherv!(my_V, MPI.VBuffer(all_V, all_nnz), comm)

    return sparse(all_I, all_J, all_V, m, n)
end

# Helper to gather VectorMPI back to global Vector
function gather_vector(v::VectorMPI{T}) where T
    n = length(v)
    my_start = v.partition[rank + 1]
    my_end = v.partition[rank + 2] - 1
    local_count = my_end - my_start + 1

    all_counts = MPI.Allgather(Int32(local_count), comm)
    full_v = Vector{T}(undef, sum(all_counts))
    MPI.Allgatherv!(v.v, MPI.VBuffer(full_v, all_counts), comm)

    return full_v
end

# Generate random sparse matrices with given dimensions and density
function make_random_sparse(m, n, density=0.3)
    return sprand(m, n, density)
end

ts = @testset QuietTestSet "Block Matrices" begin

if rank == 0
    println("[test] cat dims=1 (vcat)")
    flush(stdout)
end

# Create random matrices to stack vertically (same number of columns)
A = make_random_sparse(8, 10)
B = make_random_sparse(6, 10)
C = make_random_sparse(4, 10)

# Reference: Julia's cat
ref = cat(A, B, C; dims=1)

# MPI version
Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)
Cdist = SparseMatrixMPI{Float64}(C)

result_dist = cat(Adist, Bdist, Cdist; dims=1)
result = gather_sparse(result_dist)

@test result == ref

MPI.Barrier(comm)

if rank == 0
    println("[test] cat dims=2 (hcat)")
    flush(stdout)
end

# Create random matrices to stack horizontally (same number of rows)
A = make_random_sparse(10, 8)
B = make_random_sparse(10, 6)
C = make_random_sparse(10, 4)

# Reference
ref = cat(A, B, C; dims=2)

# MPI version
Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)
Cdist = SparseMatrixMPI{Float64}(C)

result_dist = cat(Adist, Bdist, Cdist; dims=2)
result = gather_sparse(result_dist)

@test result == ref

MPI.Barrier(comm)

if rank == 0
    println("[test] cat dims=(2,2) block matrix")
    flush(stdout)
end

# Create 4 matrices for 2x2 block
# Block layout: [A B; C D]
# A and C must have same #cols, B and D must have same #cols
# A and B must have same #rows, C and D must have same #rows
A = make_random_sparse(8, 6)
B = make_random_sparse(8, 5)
C = make_random_sparse(7, 6)
D = make_random_sparse(7, 5)

# Reference: build block matrix manually
ref = [A B; C D]

# MPI version (row-major order: A, B, C, D)
Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)
Cdist = SparseMatrixMPI{Float64}(C)
Ddist = SparseMatrixMPI{Float64}(D)

result_dist = cat(Adist, Bdist, Cdist, Ddist; dims=(2, 2))
result = gather_sparse(result_dist)

@test result == ref

MPI.Barrier(comm)

if rank == 0
    println("[test] cat dims=(3,2) block matrix")
    flush(stdout)
end

# 3 rows, 2 columns of blocks
# [A B]
# [C D]
# [E F]
A = make_random_sparse(5, 7)
B = make_random_sparse(5, 4)
C = make_random_sparse(6, 7)
D = make_random_sparse(6, 4)
E = make_random_sparse(4, 7)
F = make_random_sparse(4, 4)

# Reference
ref = [A B; C D; E F]

# MPI version
Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)
Cdist = SparseMatrixMPI{Float64}(C)
Ddist = SparseMatrixMPI{Float64}(D)
Edist = SparseMatrixMPI{Float64}(E)
Fdist = SparseMatrixMPI{Float64}(F)

result_dist = cat(Adist, Bdist, Cdist, Ddist, Edist, Fdist; dims=(3, 2))
result = gather_sparse(result_dist)

@test result == ref

MPI.Barrier(comm)

if rank == 0
    println("[test] cat dims=(2,3) block matrix")
    flush(stdout)
end

# 2 rows, 3 columns of blocks
# [A B C]
# [D E F]
A = make_random_sparse(6, 5)
B = make_random_sparse(6, 4)
C = make_random_sparse(6, 3)
D = make_random_sparse(5, 5)
E = make_random_sparse(5, 4)
F = make_random_sparse(5, 3)

# Reference
ref = [A B C; D E F]

# MPI version
Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)
Cdist = SparseMatrixMPI{Float64}(C)
Ddist = SparseMatrixMPI{Float64}(D)
Edist = SparseMatrixMPI{Float64}(E)
Fdist = SparseMatrixMPI{Float64}(F)

result_dist = cat(Adist, Bdist, Cdist, Ddist, Edist, Fdist; dims=(2, 3))
result = gather_sparse(result_dist)

@test result == ref

MPI.Barrier(comm)

if rank == 0
    println("[test] cat dims=(1,n) same as hcat")
    flush(stdout)
end

A = make_random_sparse(10, 5)
B = make_random_sparse(10, 6)
C = make_random_sparse(10, 4)

Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)
Cdist = SparseMatrixMPI{Float64}(C)

# dims=(1,3) should be same as hcat
result1 = gather_sparse(cat(Adist, Bdist, Cdist; dims=(1, 3)))
result2 = gather_sparse(hcat(Adist, Bdist, Cdist))
ref = hcat(A, B, C)

@test result1 == ref
@test result2 == ref

MPI.Barrier(comm)

if rank == 0
    println("[test] cat dims=(n,1) same as vcat")
    flush(stdout)
end

A = make_random_sparse(5, 10)
B = make_random_sparse(6, 10)
C = make_random_sparse(4, 10)

Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)
Cdist = SparseMatrixMPI{Float64}(C)

# dims=(3,1) should be same as vcat
result1 = gather_sparse(cat(Adist, Bdist, Cdist; dims=(3, 1)))
result2 = gather_sparse(vcat(Adist, Bdist, Cdist))
ref = vcat(A, B, C)

@test result1 == ref
@test result2 == ref

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI vcat")
    flush(stdout)
end

v1 = rand(10)
v2 = rand(8)
v3 = rand(12)

ref = vcat(v1, v2, v3)

v1dist = VectorMPI(v1)
v2dist = VectorMPI(v2)
v3dist = VectorMPI(v3)

result_dist = vcat(v1dist, v2dist, v3dist)
result = gather_vector(result_dist)

@test result == ref

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI hcat")
    flush(stdout)
end

# Create vectors with same length
v1 = rand(10)
v2 = rand(10)
v3 = rand(10)

ref = hcat(v1, v2, v3)

v1dist = VectorMPI(v1)
v2dist = VectorMPI(v2)
v3dist = VectorMPI(v3)

result_dist = hcat(v1dist, v2dist, v3dist)

# Result should be MatrixMPI - verify local data is correct
expected_local = hcat(v1dist.v, v2dist.v, v3dist.v)
@test result_dist.A == expected_local
@test size(result_dist) == (10, 3)

MPI.Barrier(comm)

if rank == 0
    println("[test] ComplexF64 cat")
    flush(stdout)
end

A = sprand(ComplexF64, 8, 6, 0.3)
B = sprand(ComplexF64, 8, 5, 0.3)
C = sprand(ComplexF64, 7, 6, 0.3)
D = sprand(ComplexF64, 7, 5, 0.3)

ref = [A B; C D]

Adist = SparseMatrixMPI{ComplexF64}(A)
Bdist = SparseMatrixMPI{ComplexF64}(B)
Cdist = SparseMatrixMPI{ComplexF64}(C)
Ddist = SparseMatrixMPI{ComplexF64}(D)

result_dist = cat(Adist, Bdist, Cdist, Ddist; dims=(2, 2))
result = gather_sparse(result_dist)

@test result == ref

MPI.Barrier(comm)

if rank == 0
    println("[test] blockdiag")
    flush(stdout)
end

A = make_random_sparse(8, 6)
B = make_random_sparse(5, 7)
C = make_random_sparse(4, 3)

ref = blockdiag(A, B, C)

Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)
Cdist = SparseMatrixMPI{Float64}(C)

result_dist = blockdiag(Adist, Bdist, Cdist)
result = gather_sparse(result_dist)

@test result == ref
@test size(result) == (8 + 5 + 4, 6 + 7 + 3)

MPI.Barrier(comm)

if rank == 0
    println("[test] blockdiag ComplexF64")
    flush(stdout)
end

A = sprand(ComplexF64, 6, 5, 0.3)
B = sprand(ComplexF64, 4, 8, 0.3)

ref = blockdiag(A, B)

Adist = SparseMatrixMPI{ComplexF64}(A)
Bdist = SparseMatrixMPI{ComplexF64}(B)

result_dist = blockdiag(Adist, Bdist)
result = gather_sparse(result_dist)

@test result == ref

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
    println("Test Summary: Block Matrices | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")
    flush(stdout)
end

MPI.Barrier(comm)
MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
