# MPI test for block matrix operations (cat, blockdiag)
# This file is executed under mpiexec by runtests.jl
# Parameterized over scalar types and backends (CPU/GPU)

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
using SparseArrays
using LinearAlgebra: norm
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Helper to create deterministic sparse test matrices
function make_sparse(::Type{T}, m, n, pattern::Symbol=:diagonal) where T
    if pattern == :diagonal
        # Simple diagonal pattern
        k = min(m, n)
        I = collect(1:k)
        J = collect(1:k)
        V = T <: Complex ? T.(1:k) .+ im .* T.(k:-1:1) : T.(1:k)
        return sparse(I, J, V, m, n)
    elseif pattern == :tridiagonal
        I = Int[]
        J = Int[]
        V = T[]
        for i in 1:min(m, n)
            push!(I, i); push!(J, i)
            push!(V, T <: Complex ? T(2) + im * T(0.1) : T(2))
            if i > 1 && i <= n
                push!(I, i); push!(J, i-1)
                push!(V, T <: Complex ? T(-0.5) + im * T(0.2) : T(-0.5))
            end
            if i < m && i < n
                push!(I, i); push!(J, i+1)
                push!(V, T <: Complex ? T(-0.5) - im * T(0.2) : T(-0.5))
            end
        end
        return sparse(I, J, V, m, n)
    elseif pattern == :scattered
        # Scattered non-zeros for more interesting sparsity
        I = Int[]
        J = Int[]
        V = T[]
        for i in 1:m
            for j in 1:n
                if (i + j) % 3 == 0
                    push!(I, i); push!(J, j)
                    push!(V, T <: Complex ? T(i + j) + im * T(i - j) : T(i + j))
                end
            end
        end
        return isempty(I) ? spzeros(T, m, n) : sparse(I, J, V, m, n)
    end
end

# Helper to create deterministic dense test matrices
function make_dense(::Type{T}, m, n) where T
    A = zeros(T, m, n)
    for i in 1:m, j in 1:n
        if T <: Complex
            A[i, j] = T(i + j) + im * T(i - j)
        else
            A[i, j] = T(i + j)
        end
    end
    A
end

# Helper to create deterministic test vectors
function make_vector(::Type{T}, n) where T
    if T <: Complex
        T.(1:n) .+ im .* T.(n:-1:1)
    else
        T.(1:n)
    end
end

ts = @testset QuietTestSet "Block Matrices" begin

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)

    println(io0(), "[test] cat dims=1 (vcat) ($T, $backend_name)")

    # Create sparse matrices to stack vertically (same number of columns)
    A = make_sparse(T, 8, 10, :scattered)
    B = make_sparse(T, 6, 10, :diagonal)
    C = make_sparse(T, 4, 10, :tridiagonal)

    # Reference: Julia's cat
    ref = cat(A, B, C; dims=1)

    # MPI version
    Adist = to_backend(SparseMatrixMPI{T}(A))
    Bdist = to_backend(SparseMatrixMPI{T}(B))
    Cdist = to_backend(SparseMatrixMPI{T}(C))

    result_dist = cat(Adist, Bdist, Cdist; dims=1)
    result = SparseMatrixCSC(result_dist)

    @test norm(result - ref, Inf) < TOL


    println(io0(), "[test] cat dims=2 (hcat) ($T, $backend_name)")

    # Create sparse matrices to stack horizontally (same number of rows)
    A = make_sparse(T, 10, 8, :scattered)
    B = make_sparse(T, 10, 6, :diagonal)
    C = make_sparse(T, 10, 4, :tridiagonal)

    # Reference
    ref = cat(A, B, C; dims=2)

    # MPI version
    Adist = to_backend(SparseMatrixMPI{T}(A))
    Bdist = to_backend(SparseMatrixMPI{T}(B))
    Cdist = to_backend(SparseMatrixMPI{T}(C))

    result_dist = cat(Adist, Bdist, Cdist; dims=2)
    result = SparseMatrixCSC(result_dist)

    @test norm(result - ref, Inf) < TOL


    println(io0(), "[test] cat dims=(2,2) block matrix ($T, $backend_name)")

    # Create 4 matrices for 2x2 block
    # Block layout: [A B; C D]
    A = make_sparse(T, 8, 6, :scattered)
    B = make_sparse(T, 8, 5, :diagonal)
    C = make_sparse(T, 7, 6, :tridiagonal)
    D = make_sparse(T, 7, 5, :scattered)

    # Reference: build block matrix manually
    ref = [A B; C D]

    # MPI version (row-major order: A, B, C, D)
    Adist = to_backend(SparseMatrixMPI{T}(A))
    Bdist = to_backend(SparseMatrixMPI{T}(B))
    Cdist = to_backend(SparseMatrixMPI{T}(C))
    Ddist = to_backend(SparseMatrixMPI{T}(D))

    result_dist = cat(Adist, Bdist, Cdist, Ddist; dims=(2, 2))
    result = SparseMatrixCSC(result_dist)

    @test norm(result - ref, Inf) < TOL


    println(io0(), "[test] cat dims=(3,2) block matrix ($T, $backend_name)")

    # 3 rows, 2 columns of blocks
    # [A B]
    # [C D]
    # [E F]
    A = make_sparse(T, 5, 7, :scattered)
    B = make_sparse(T, 5, 4, :diagonal)
    C = make_sparse(T, 6, 7, :tridiagonal)
    D = make_sparse(T, 6, 4, :scattered)
    E = make_sparse(T, 4, 7, :diagonal)
    F = make_sparse(T, 4, 4, :tridiagonal)

    # Reference
    ref = [A B; C D; E F]

    # MPI version
    Adist = to_backend(SparseMatrixMPI{T}(A))
    Bdist = to_backend(SparseMatrixMPI{T}(B))
    Cdist = to_backend(SparseMatrixMPI{T}(C))
    Ddist = to_backend(SparseMatrixMPI{T}(D))
    Edist = to_backend(SparseMatrixMPI{T}(E))
    Fdist = to_backend(SparseMatrixMPI{T}(F))

    result_dist = cat(Adist, Bdist, Cdist, Ddist, Edist, Fdist; dims=(3, 2))
    result = SparseMatrixCSC(result_dist)

    @test norm(result - ref, Inf) < TOL


    println(io0(), "[test] cat dims=(2,3) block matrix ($T, $backend_name)")

    # 2 rows, 3 columns of blocks
    # [A B C]
    # [D E F]
    A = make_sparse(T, 6, 5, :scattered)
    B = make_sparse(T, 6, 4, :diagonal)
    C = make_sparse(T, 6, 3, :tridiagonal)
    D = make_sparse(T, 5, 5, :diagonal)
    E = make_sparse(T, 5, 4, :scattered)
    F = make_sparse(T, 5, 3, :diagonal)

    # Reference
    ref = [A B C; D E F]

    # MPI version
    Adist = to_backend(SparseMatrixMPI{T}(A))
    Bdist = to_backend(SparseMatrixMPI{T}(B))
    Cdist = to_backend(SparseMatrixMPI{T}(C))
    Ddist = to_backend(SparseMatrixMPI{T}(D))
    Edist = to_backend(SparseMatrixMPI{T}(E))
    Fdist = to_backend(SparseMatrixMPI{T}(F))

    result_dist = cat(Adist, Bdist, Cdist, Ddist, Edist, Fdist; dims=(2, 3))
    result = SparseMatrixCSC(result_dist)

    @test norm(result - ref, Inf) < TOL


    println(io0(), "[test] VectorMPI vcat ($T, $backend_name)")

    v1 = make_vector(T, 10)
    v2 = make_vector(T, 8) .* T(2)
    v3 = make_vector(T, 12) .* T(3)

    ref = vcat(v1, v2, v3)

    v1dist = to_backend(VectorMPI(v1))
    v2dist = to_backend(VectorMPI(v2))
    v3dist = to_backend(VectorMPI(v3))

    result_dist = vcat(v1dist, v2dist, v3dist)
    result = Vector(result_dist)

    @test norm(result - ref, Inf) < TOL


    println(io0(), "[test] VectorMPI hcat ($T, $backend_name)")

    # Create vectors with same length
    v1 = make_vector(T, 10)
    v2 = make_vector(T, 10) .* T(2)
    v3 = make_vector(T, 10) .* T(3)

    ref = hcat(v1, v2, v3)

    v1dist = to_backend(VectorMPI(v1))
    v2dist = to_backend(VectorMPI(v2))
    v3dist = to_backend(VectorMPI(v3))

    result_dist = hcat(v1dist, v2dist, v3dist)

    # Result should be MatrixMPI
    @test size(result_dist) == (10, 3)
    result_full = Matrix(result_dist)
    @test norm(result_full - ref, Inf) < TOL


    println(io0(), "[test] blockdiag ($T, $backend_name)")

    A = make_sparse(T, 8, 6, :scattered)
    B = make_sparse(T, 5, 7, :diagonal)
    C = make_sparse(T, 4, 3, :tridiagonal)

    ref = blockdiag(A, B, C)

    Adist = to_backend(SparseMatrixMPI{T}(A))
    Bdist = to_backend(SparseMatrixMPI{T}(B))
    Cdist = to_backend(SparseMatrixMPI{T}(C))

    result_dist = blockdiag(Adist, Bdist, Cdist)
    result = SparseMatrixCSC(result_dist)

    @test norm(result - ref, Inf) < TOL
    @test size(result) == (8 + 5 + 4, 6 + 7 + 3)


    println(io0(), "[test] MatrixMPI vcat ($T, $backend_name)")

    # Create dense matrices to stack vertically
    A_dense = make_dense(T, 8, 10)
    B_dense = make_dense(T, 6, 10) .* T(2)
    C_dense = make_dense(T, 4, 10) .* T(3)

    ref = vcat(A_dense, B_dense, C_dense)

    Adist = to_backend(MatrixMPI(A_dense))
    Bdist = to_backend(MatrixMPI(B_dense))
    Cdist = to_backend(MatrixMPI(C_dense))

    result_dist = vcat(Adist, Bdist, Cdist)

    @test size(result_dist) == size(ref)
    @test size(result_dist, 2) == 10  # columns preserved
    result_full = Matrix(result_dist)
    @test norm(result_full - ref, Inf) < TOL


    println(io0(), "[test] MatrixMPI hcat ($T, $backend_name)")

    # Create dense matrices to stack horizontally
    A_dense = make_dense(T, 10, 8)
    B_dense = make_dense(T, 10, 6) .* T(2)
    C_dense = make_dense(T, 10, 4) .* T(3)

    ref = hcat(A_dense, B_dense, C_dense)

    Adist = to_backend(MatrixMPI(A_dense))
    Bdist = to_backend(MatrixMPI(B_dense))
    Cdist = to_backend(MatrixMPI(C_dense))

    result_dist = hcat(Adist, Bdist, Cdist)

    @test size(result_dist) == size(ref)
    result_full = Matrix(result_dist)
    @test norm(result_full - ref, Inf) < TOL


    println(io0(), "[test] MatrixMPI cat dims=(2,2) ($T, $backend_name)")

    # Create 4 matrices for 2x2 block [A B; C D]
    A_dense = make_dense(T, 8, 6)
    B_dense = make_dense(T, 8, 5) .* T(2)
    C_dense = make_dense(T, 7, 6) .* T(3)
    D_dense = make_dense(T, 7, 5) .* T(4)

    ref = [A_dense B_dense; C_dense D_dense]

    Adist = to_backend(MatrixMPI(A_dense))
    Bdist = to_backend(MatrixMPI(B_dense))
    Cdist = to_backend(MatrixMPI(C_dense))
    Ddist = to_backend(MatrixMPI(D_dense))

    result_dist = cat(Adist, Bdist, Cdist, Ddist; dims=(2, 2))

    @test size(result_dist) == size(ref)
    result_full = Matrix(result_dist)
    @test norm(result_full - ref, Inf) < TOL


    println(io0(), "[test] VectorMPI cat with tuple dims ($T, $backend_name)")

    # Test dims=(n,1) same as vcat
    v1 = make_vector(T, 10)
    v2 = make_vector(T, 8) .* T(2)
    v3 = make_vector(T, 12) .* T(3)

    ref = vcat(v1, v2, v3)

    v1dist = to_backend(VectorMPI(v1))
    v2dist = to_backend(VectorMPI(v2))
    v3dist = to_backend(VectorMPI(v3))

    result_dist = cat(v1dist, v2dist, v3dist; dims=(3, 1))
    result = Vector(result_dist)

    @test norm(result - ref, Inf) < TOL

    # Test dims=(1,n) same as hcat
    v1 = make_vector(T, 10)
    v2 = make_vector(T, 10) .* T(2)
    v3 = make_vector(T, 10) .* T(3)

    v1dist = to_backend(VectorMPI(v1))
    v2dist = to_backend(VectorMPI(v2))
    v3dist = to_backend(VectorMPI(v3))

    result_dist = cat(v1dist, v2dist, v3dist; dims=(1, 3))

    @test size(result_dist) == (10, 3)

    # Test dims=(1,1) with single vector
    v_single = make_vector(T, 15)
    v_single_dist = to_backend(VectorMPI(v_single))
    result_single = cat(v_single_dist; dims=(1, 1))
    result_gathered = Vector(result_single)

    @test norm(result_gathered - v_single, Inf) < TOL

end  # for configs

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

println("Test Summary: Block Matrices | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
