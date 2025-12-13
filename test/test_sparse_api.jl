# Tests for extended SparseMatrixMPI API
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

const TOL = 1e-12

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Create deterministic test matrices (same on all ranks)
n = 20
A_global = sparse([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                  [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                  n, n)
ref_nnz = nnz(A_global)
ref_sum = sum(A_global)
ref_abs_sum = sum(abs.(A_global))

Adist = SparseMatrixMPI{Float64}(A_global)

# Complex test matrix
A_complex = sparse([1, 2, 3, 4, 5, 1, 2, 3],
                   [1, 2, 3, 4, 5, 6, 7, 8],
                   ComplexF64[1+2im, 3-1im, 2+1im, -1+3im, 4-2im, 1-1im, 2+2im, 3+1im],
                   n, n)
Adist_complex = SparseMatrixMPI{ComplexF64}(A_complex)

ts = @testset QuietTestSet "Sparse API" begin

if rank == 0
    println("[test] Structural queries")
    flush(stdout)
end

# Test structural queries - compute values BEFORE @test to avoid MPI issues
dist_nnz = nnz(Adist)
dist_issparse = issparse(Adist)
@test dist_nnz == ref_nnz
@test dist_issparse == true

MPI.Barrier(comm)

if rank == 0
    println("[test] Copy")
    flush(stdout)
end

# Test copy
B = copy(Adist)
b_nnz = nnz(B)
b_size = size(B)
@test b_nnz == ref_nnz
@test b_size == size(A_global)

MPI.Barrier(comm)

if rank == 0
    println("[test] Element-wise operations")
    flush(stdout)
end

# Test element-wise operations
B = abs(Adist)
b_sum = sum(B)
@test b_sum ≈ ref_abs_sum atol=TOL

B2 = abs2(Adist)
b2_sum = sum(B2)
@test b2_sum ≈ sum(abs2.(A_global)) atol=TOL

B_real = real(Adist_complex)
br_sum = sum(B_real)
@test br_sum ≈ sum(real.(A_complex)) atol=TOL

B_imag = imag(Adist_complex)
bi_sum = sum(B_imag)
@test bi_sum ≈ sum(imag.(A_complex)) atol=TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] Reductions")
    flush(stdout)
end

# Test reductions
dist_sum = sum(Adist)
dist_max = maximum(Adist)
dist_min = minimum(Adist)
dist_mean = mean(Adist)
dist_tr = tr(Adist)
@test dist_sum ≈ ref_sum atol=TOL
@test dist_max ≈ maximum(A_global.nzval) atol=TOL
@test dist_min ≈ minimum(A_global.nzval) atol=TOL
@test dist_mean ≈ ref_sum / (n * n) atol=TOL
@test dist_tr ≈ tr(A_global) atol=TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] Sum with dims")
    flush(stdout)
end

# Test sum with dims
col_sums = sum(Adist; dims=1)
ref_col_sums = vec(sum(A_global; dims=1))
counts1 = Int32[col_sums.partition[r+2] - col_sums.partition[r+1] for r in 0:nranks-1]
full_col_sums = similar(col_sums.v, sum(counts1))
MPI.Allgatherv!(col_sums.v, MPI.VBuffer(full_col_sums, counts1), comm)
err1 = norm(full_col_sums - ref_col_sums)
@test err1 < TOL

row_sums = sum(Adist; dims=2)
ref_row_sums = vec(sum(A_global; dims=2))
counts2 = Int32[row_sums.partition[r+2] - row_sums.partition[r+1] for r in 0:nranks-1]
full_row_sums = similar(row_sums.v, sum(counts2))
MPI.Allgatherv!(row_sums.v, MPI.VBuffer(full_row_sums, counts2), comm)
err2 = norm(full_row_sums - ref_row_sums)
@test err2 < TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] Dropzeros")
    flush(stdout)
end

# Test dropzeros
A_with_zeros = copy(A_global)
A_with_zeros[1, 1] = 0.0
ref_nnz_zeros = nnz(A_with_zeros)
Adist_zeros = SparseMatrixMPI{Float64}(A_with_zeros)
B = dropzeros(Adist_zeros)
b_nnz = nnz(B)
@test b_nnz <= ref_nnz_zeros

MPI.Barrier(comm)

if rank == 0
    println("[test] Diagonal extraction")
    flush(stdout)
end

# Test diag
d = diag(Adist)
ref_d = diag(A_global)
counts_d = Int32[d.partition[r+2] - d.partition[r+1] for r in 0:nranks-1]
full_d = similar(d.v, sum(counts_d))
MPI.Allgatherv!(d.v, MPI.VBuffer(full_d, counts_d), comm)
err1 = norm(full_d - ref_d)
@test err1 < TOL

d1 = diag(Adist, 1)
ref_d1 = diag(A_global, 1)
counts_d1 = Int32[d1.partition[r+2] - d1.partition[r+1] for r in 0:nranks-1]
full_d1 = similar(d1.v, sum(counts_d1))
MPI.Allgatherv!(d1.v, MPI.VBuffer(full_d1, counts_d1), comm)
err2 = norm(full_d1 - ref_d1)
@test err2 < TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] Triangular parts")
    flush(stdout)
end

# Test triu and tril
U = triu(Adist)
ref_U = triu(A_global)
u_nnz = nnz(U)
@test u_nnz == nnz(ref_U)

L = tril(Adist)
ref_L = tril(A_global)
l_nnz = nnz(L)
@test l_nnz == nnz(ref_L)

U1 = triu(Adist, 1)
ref_U1 = triu(A_global, 1)
u1_nnz = nnz(U1)
@test u1_nnz == nnz(ref_U1)

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI extensions")
    flush(stdout)
end

# Test VectorMPI extensions
v_global = collect(1.0:Float64(n))
v = VectorMPI(v_global)

abs_sum = sum(abs(v))
@test abs_sum ≈ sum(abs.(v_global)) atol=TOL

abs2_sum = sum(abs2(v))
@test abs2_sum ≈ sum(abs2.(v_global)) atol=TOL

v_mean = mean(v)
@test v_mean ≈ sum(v_global) / n atol=TOL

v_copy = copy(v)
vcopy_sum = sum(v_copy)
v_sum = sum(v)
@test vcopy_sum ≈ v_sum atol=TOL

v_complex_global = vcat(ComplexF64[1+2im, 3-1im, 2+1im, -1+3im], zeros(ComplexF64, n - 4))
v_complex = VectorMPI(v_complex_global)
vr_sum = sum(real(v_complex))
@test vr_sum ≈ sum(real.(v_complex_global)) atol=TOL
vi_sum = sum(imag(v_complex))
@test vi_sum ≈ sum(imag.(v_complex_global)) atol=TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] spdiagm")
    flush(stdout)
end

# Test spdiagm
# Test main diagonal only
v_global = collect(1.0:5.0)
v = VectorMPI(v_global)
A = spdiagm(v)
ref_A = spdiagm(v_global)
a_nnz = nnz(A)
a_sum = sum(A)
a_tr = tr(A)
@test a_nnz == nnz(ref_A)
@test a_sum ≈ sum(ref_A) atol=TOL
@test a_tr ≈ tr(ref_A) atol=TOL

# Test multiple diagonals
v1_global = collect(1.0:4.0)
v2_global = collect(10.0:12.0)
v1 = VectorMPI(v1_global)
v2 = VectorMPI(v2_global)
B = spdiagm(0 => v1, 1 => v2)
ref_B = spdiagm(0 => v1_global, 1 => v2_global)
b_nnz = nnz(B)
b_sum = sum(B)
@test b_nnz == nnz(ref_B)
@test b_sum ≈ sum(ref_B) atol=TOL

# Test with explicit size
C = spdiagm(6, 6, 0 => v1)
ref_C = spdiagm(6, 6, v1_global)
c_nnz = nnz(C)
c_size = size(C)
@test c_nnz == nnz(ref_C)
@test c_size == (6, 6)

# Test subdiagonal
D = spdiagm(-1 => v2)
ref_D = spdiagm(-1 => v2_global)
d_nnz = nnz(D)
d_sum = sum(D)
@test d_nnz == nnz(ref_D)
@test d_sum ≈ sum(ref_D) atol=TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI broadcasting")
    flush(stdout)
end

# Test VectorMPI broadcasting
v_global = collect(1.0:10.0)
w_global = collect(11.0:20.0)
v = VectorMPI(v_global)
w = VectorMPI(w_global)

# Test element-wise addition
vw_add = v .+ w
vw_add_sum = sum(vw_add)
@test vw_add_sum ≈ sum(v_global .+ w_global) atol=TOL

# Test element-wise multiplication
vw_mul = v .* w
vw_mul_sum = sum(vw_mul)
@test vw_mul_sum ≈ sum(v_global .* w_global) atol=TOL

# Test broadcast with scalar
v_scaled = v .* 2.0
v_scaled_sum = sum(v_scaled)
@test v_scaled_sum ≈ sum(v_global .* 2.0) atol=TOL

v_plus_scalar = v .+ 100.0
v_plus_scalar_sum = sum(v_plus_scalar)
@test v_plus_scalar_sum ≈ sum(v_global .+ 100.0) atol=TOL

# Test function broadcasting
v_sin = sin.(v)
v_sin_sum = sum(v_sin)
@test v_sin_sum ≈ sum(sin.(v_global)) atol=TOL

v_exp = exp.(v)
v_exp_sum = sum(v_exp)
@test v_exp_sum ≈ sum(exp.(v_global)) atol=TOL

# Test compound broadcast expression
compound = v .* 2.0 .+ w .^ 2
compound_sum = sum(compound)
@test compound_sum ≈ sum(v_global .* 2.0 .+ w_global .^ 2) atol=TOL

# Test in-place broadcast assignment (materialize!)
dest = VectorMPI(zeros(10))
dest .= v .+ w
dest_sum = sum(dest)
@test dest_sum ≈ sum(v_global .+ w_global) atol=TOL

# Test in-place compound broadcast (materialize!)
dest2 = VectorMPI(zeros(10))
dest2 .= v .* 2.0 .+ w .^ 2
dest2_sum = sum(dest2)
@test dest2_sum ≈ sum(v_global .* 2.0 .+ w_global .^ 2) atol=TOL

# Test in-place with scalar (materialize!)
dest3 = VectorMPI(zeros(10))
dest3 .= v .* 3.0 .+ 10.0
dest3_sum = sum(dest3)
@test dest3_sum ≈ sum(v_global .* 3.0 .+ 10.0) atol=TOL

# Test broadcast with different element types
v_int_global = collect(1:10)
v_int = VectorMPI(Float64.(v_int_global))
v_sqrt = sqrt.(v_int)
v_sqrt_sum = sum(v_sqrt)
@test v_sqrt_sum ≈ sum(sqrt.(Float64.(v_int_global))) atol=TOL

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI broadcasting with different partitions")
    flush(stdout)
end

# Test broadcasting with different partitions (requires MPI communication)
n_part = 12
v_global_part = collect(1.0:Float64(n_part))
w_global_part = collect(101.0:Float64(100+n_part))

# Create vectors with different partitions
# v uses default even partition
v_part = VectorMPI(v_global_part)

# w uses a custom uneven partition that adapts to nranks
# Create a partition where elements are unevenly distributed
function make_uneven_partition(n::Int, nranks::Int)
    # Distribute n elements unevenly: give rank 0 one extra element
    partition = Vector{Int}(undef, nranks + 1)
    partition[1] = 1
    base = div(n, nranks)
    remainder = mod(n, nranks)
    for r in 1:nranks
        # Give first `remainder` ranks one extra, but shift pattern
        extra = (r <= remainder) ? 1 : 0
        # Add an extra offset to make it different from default
        if r == 1 && nranks > 1
            extra += 1  # Give rank 0 an extra element
        elseif r == nranks && nranks > 1
            extra -= 1  # Take one from last rank
        end
        partition[r+1] = partition[r] + base + extra
    end
    # Ensure last partition boundary is correct
    partition[end] = n + 1
    return partition
end

custom_partition = make_uneven_partition(n_part, nranks)
w_custom_local_start = custom_partition[rank+1]
w_custom_local_end = custom_partition[rank+2] - 1
w_local_part = w_global_part[w_custom_local_start:w_custom_local_end]
w_hash = LinearAlgebraMPI.compute_partition_hash(custom_partition)
w_part = VectorMPI{Float64}(w_hash, custom_partition, w_local_part)

# Verify w has correct global values
w_sum_part = sum(w_part)
@test w_sum_part ≈ sum(w_global_part) atol=TOL

# Test broadcasting between vectors with different partitions
# This should trigger _align_vector with MPI.Alltoallv!
vw_add_part = v_part .+ w_part
vw_add_sum_part = sum(vw_add_part)
@test vw_add_sum_part ≈ sum(v_global_part .+ w_global_part) atol=TOL

vw_mul_part = v_part .* w_part
vw_mul_sum_part = sum(vw_mul_part)
@test vw_mul_sum_part ≈ sum(v_global_part .* w_global_part) atol=TOL

# Compound expression with different partitions
compound_part = v_part .* 2.0 .+ w_part
compound_sum_part = sum(compound_part)
@test compound_sum_part ≈ sum(v_global_part .* 2.0 .+ w_global_part) atol=TOL

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
    println("Test Summary: Sparse API | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")
    flush(stdout)
end

MPI.Barrier(comm)
MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
