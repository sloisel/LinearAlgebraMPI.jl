# Tests for extended SparseMatrixMPI API
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

ts = @testset QuietTestSet "Sparse API" begin

for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)

    println(io0(), "[test] Structural queries ($T, $backend_name)")

    # Create deterministic test matrix
    n = 20
    A_global = sparse([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                      T.([1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
                      n, n)
    ref_nnz = nnz(A_global)
    Adist = to_backend(SparseMatrixMPI{T}(A_global))

    dist_nnz = nnz(Adist)
    dist_issparse = issparse(Adist)
    @test dist_nnz == ref_nnz
    @test dist_issparse == true


    println(io0(), "[test] Copy ($T, $backend_name)")

    B = copy(Adist)
    b_nnz = nnz(B)
    b_size = size(B)
    @test b_nnz == ref_nnz
    @test b_size == size(A_global)

    # Recreate Adist for subsequent tests (copy may have different structure)
    Adist = to_backend(SparseMatrixMPI{T}(A_global))


    println(io0(), "[test] Element-wise operations ($T, $backend_name)")

    ref_abs_sum = sum(abs.(A_global))
    B = abs(Adist)
    b_sum = sum(B)
    @test b_sum ≈ ref_abs_sum atol=TOL

    B2 = abs2(Adist)
    b2_sum = sum(B2)
    @test b2_sum ≈ sum(abs2.(A_global)) atol=TOL

    # floor/ceil/round only for real types
    if !(T <: Complex)
        B_floor = floor(Adist)
        bf_sum = sum(B_floor)
        @test bf_sum ≈ sum(floor.(A_global.nzval)) atol=TOL

        B_ceil = ceil(Adist)
        bc_sum = sum(B_ceil)
        @test bc_sum ≈ sum(ceil.(A_global.nzval)) atol=TOL

        B_round = round(Adist)
        br_sum = sum(B_round)
        @test br_sum ≈ sum(round.(A_global.nzval)) atol=TOL
    end

    B_map = map(x -> x^2 + one(T), Adist)
    bm_sum = sum(B_map)
    @test bm_sum ≈ sum(x -> x^2 + one(T), A_global.nzval) atol=TOL


    println(io0(), "[test] Reductions ($T, $backend_name)")

    ref_sum = sum(A_global)
    dist_sum = sum(Adist)
    dist_mean = mean(Adist)
    dist_tr = tr(Adist)
    @test dist_sum ≈ ref_sum atol=TOL
    @test dist_mean ≈ ref_sum / (n * n) atol=TOL
    @test dist_tr ≈ tr(A_global) atol=TOL

    # max/min only meaningful for real types
    if !(T <: Complex)
        dist_max = maximum(Adist)
        dist_min = minimum(Adist)
        @test dist_max ≈ maximum(A_global.nzval) atol=TOL
        @test dist_min ≈ minimum(A_global.nzval) atol=TOL
    end


    println(io0(), "[test] Sum with dims ($T, $backend_name)")

    col_sums = sum(Adist; dims=1)
    ref_col_sums = vec(sum(A_global; dims=1))
    # Gather results via Vector() which handles CPU staging
    full_col_sums = Vector(col_sums)
    err1 = norm(full_col_sums - ref_col_sums)
    @test err1 < TOL

    row_sums = sum(Adist; dims=2)
    ref_row_sums = vec(sum(A_global; dims=2))
    full_row_sums = Vector(row_sums)
    err2 = norm(full_row_sums - ref_row_sums)
    @test err2 < TOL


    println(io0(), "[test] Dropzeros ($T, $backend_name)")

    A_with_zeros = copy(A_global)
    A_with_zeros[1, 1] = zero(T)
    ref_nnz_zeros = nnz(A_with_zeros)
    Adist_zeros = to_backend(SparseMatrixMPI{T}(A_with_zeros))
    B = dropzeros(Adist_zeros)
    b_nnz = nnz(B)
    @test b_nnz <= ref_nnz_zeros


    println(io0(), "[test] Diagonal extraction ($T, $backend_name)")

    d = diag(Adist)
    ref_d = diag(A_global)
    full_d = Vector(d)
    err1 = norm(full_d - ref_d)
    @test err1 < TOL

    d1 = diag(Adist, 1)
    ref_d1 = diag(A_global, 1)
    full_d1 = Vector(d1)
    err2 = norm(full_d1 - ref_d1)
    @test err2 < TOL

    dm1 = diag(Adist, -1)
    ref_dm1 = diag(A_global, -1)
    full_dm1 = Vector(dm1)
    err3 = norm(full_dm1 - ref_dm1)
    @test err3 < TOL

    d_empty = diag(Adist, n + 5)
    @test length(d_empty) == 0

    d_empty2 = diag(Adist, -(n + 5))
    @test length(d_empty2) == 0


    println(io0(), "[test] Triangular parts ($T, $backend_name)")

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


    println(io0(), "[test] VectorMPI extensions ($T, $backend_name)")

    v_global = T.(collect(1.0:Float64(n)))
    v = to_backend(VectorMPI(v_global))

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


    println(io0(), "[test] spdiagm ($T, $backend_name)")

    v_global = T.(collect(1.0:5.0))
    v_spd = to_backend(VectorMPI(v_global))
    A_spd = spdiagm(v_spd)
    ref_A = spdiagm(v_global)
    a_nnz = nnz(A_spd)
    a_sum = sum(A_spd)
    a_tr = tr(A_spd)
    @test a_nnz == nnz(ref_A)
    @test a_sum ≈ sum(ref_A) atol=TOL
    @test a_tr ≈ tr(ref_A) atol=TOL

    v1_global = T.(collect(1.0:4.0))
    v2_global = T.(collect(10.0:12.0))
    v1 = to_backend(VectorMPI(v1_global))
    v2 = to_backend(VectorMPI(v2_global))
    B_spd = spdiagm(0 => v1, 1 => v2)
    ref_B = spdiagm(0 => v1_global, 1 => v2_global)
    b_nnz = nnz(B_spd)
    b_sum = sum(B_spd)
    @test b_nnz == nnz(ref_B)
    @test b_sum ≈ sum(ref_B) atol=TOL

    C_spd = spdiagm(6, 6, 0 => v1)
    ref_C = spdiagm(6, 6, v1_global)
    c_nnz = nnz(C_spd)
    c_size = size(C_spd)
    @test c_nnz == nnz(ref_C)
    @test c_size == (6, 6)

    D_spd = spdiagm(-1 => v2)
    ref_D = spdiagm(-1 => v2_global)
    d_nnz = nnz(D_spd)
    d_sum = sum(D_spd)
    @test d_nnz == nnz(ref_D)
    @test d_sum ≈ sum(ref_D) atol=TOL


    println(io0(), "[test] VectorMPI broadcasting ($T, $backend_name)")

    v_global = T.(collect(1.0:10.0))
    w_global = T.(collect(11.0:20.0))
    v = to_backend(VectorMPI(v_global))
    w = to_backend(VectorMPI(w_global))

    vw_add = v .+ w
    vw_add_sum = sum(vw_add)
    @test vw_add_sum ≈ sum(v_global .+ w_global) atol=TOL

    vw_mul = v .* w
    vw_mul_sum = sum(vw_mul)
    @test vw_mul_sum ≈ sum(v_global .* w_global) atol=TOL

    v_scaled = v .* T(2.0)
    v_scaled_sum = sum(v_scaled)
    @test v_scaled_sum ≈ sum(v_global .* T(2.0)) atol=TOL

    v_plus_scalar = v .+ T(100.0)
    v_plus_scalar_sum = sum(v_plus_scalar)
    @test v_plus_scalar_sum ≈ sum(v_global .+ T(100.0)) atol=TOL

    # Function broadcasting (only for real types - sin/exp may have issues with complex on GPU)
    if !(T <: Complex)
        v_sin = sin.(v)
        v_sin_sum = sum(v_sin)
        @test v_sin_sum ≈ sum(sin.(v_global)) atol=TOL

        v_exp = exp.(v)
        v_exp_sum = sum(v_exp)
        @test v_exp_sum ≈ sum(exp.(v_global)) atol=TOL
    end

    compound = v .* T(2.0) .+ w .^ 2
    compound_sum = sum(compound)
    @test compound_sum ≈ sum(v_global .* T(2.0) .+ w_global .^ 2) atol=TOL

    dest = to_backend(VectorMPI(zeros(T, 10)))
    dest .= v .+ w
    dest_sum = sum(dest)
    @test dest_sum ≈ sum(v_global .+ w_global) atol=TOL

    dest2 = to_backend(VectorMPI(zeros(T, 10)))
    dest2 .= v .* T(2.0) .+ w .^ 2
    dest2_sum = sum(dest2)
    @test dest2_sum ≈ sum(v_global .* T(2.0) .+ w_global .^ 2) atol=TOL

    dest3 = to_backend(VectorMPI(zeros(T, 10)))
    dest3 .= v .* T(3.0) .+ T(10.0)
    dest3_sum = sum(dest3)
    @test dest3_sum ≈ sum(v_global .* T(3.0) .+ T(10.0)) atol=TOL

    if !(T <: Complex)
        v_int = to_backend(VectorMPI(T.(collect(1:10))))
        v_sqrt = sqrt.(v_int)
        v_sqrt_sum = sum(v_sqrt)
        @test v_sqrt_sum ≈ sum(sqrt.(T.(collect(1:10)))) atol=TOL
    end


    println(io0(), "[test] VectorMPI broadcasting with different partitions ($T, $backend_name)")

    n_part = 12
    v_global_part = T.(collect(1.0:Float64(n_part)))
    w_global_part = T.(collect(101.0:Float64(100+n_part)))

    v_part = to_backend(VectorMPI(v_global_part))

    function make_uneven_partition(n::Int, nranks::Int)
        partition = Vector{Int}(undef, nranks + 1)
        partition[1] = 1
        base = div(n, nranks)
        remainder = mod(n, nranks)
        for r in 1:nranks
            extra = (r <= remainder) ? 1 : 0
            if r == 1 && nranks > 1
                extra += 1
            elseif r == nranks && nranks > 1
                extra -= 1
            end
            partition[r+1] = partition[r] + base + extra
        end
        partition[end] = n + 1
        return partition
    end

    custom_partition = make_uneven_partition(n_part, nranks)
    w_custom_local_start = custom_partition[rank+1]
    w_custom_local_end = custom_partition[rank+2] - 1
    w_local_part = w_global_part[w_custom_local_start:w_custom_local_end]
    w_hash = LinearAlgebraMPI.compute_partition_hash(custom_partition)
    # Create VectorMPI with custom partition (stays on CPU, to_backend not applied here
    # since the VectorMPI constructor signature with hash requires specific type)
    w_part = to_backend(VectorMPI{T}(w_hash, custom_partition, w_local_part))

    w_sum_part = sum(w_part)
    @test w_sum_part ≈ sum(w_global_part) atol=TOL

    vw_add_part = v_part .+ w_part
    vw_add_sum_part = sum(vw_add_part)
    @test vw_add_sum_part ≈ sum(v_global_part .+ w_global_part) atol=TOL

    vw_mul_part = v_part .* w_part
    vw_mul_sum_part = sum(vw_mul_part)
    @test vw_mul_sum_part ≈ sum(v_global_part .* w_global_part) atol=TOL

    compound_part = v_part .* T(2.0) .+ w_part
    compound_sum_part = sum(compound_part)
    @test compound_sum_part ≈ sum(v_global_part .* T(2.0) .+ w_global_part) atol=TOL

end  # for (T, to_backend, backend_name)


# Complex-specific tests (run on CPU only since they're just ComplexF64)
println(io0(), "[test] Complex element-wise operations (ComplexF64)")

T_cpx = ComplexF64
TOL_cpx = TestUtils.tolerance(T_cpx)
n_cpx = 20

A_complex = sparse([1, 2, 3, 4, 5, 1, 2, 3],
                   [1, 2, 3, 4, 5, 6, 7, 8],
                   T_cpx[1+2im, 3-1im, 2+1im, -1+3im, 4-2im, 1-1im, 2+2im, 3+1im],
                   n_cpx, n_cpx)
Adist_complex = SparseMatrixMPI{T_cpx}(A_complex)

B_real = real(Adist_complex)
br_sum = sum(B_real)
@test br_sum ≈ sum(real.(A_complex)) atol=TOL_cpx

B_imag = imag(Adist_complex)
bi_sum = sum(B_imag)
@test bi_sum ≈ sum(imag.(A_complex)) atol=TOL_cpx

v_complex_global = vcat(T_cpx[1+2im, 3-1im, 2+1im, -1+3im], zeros(T_cpx, n_cpx - 4))
v_complex = VectorMPI(v_complex_global)
vr_sum = sum(real(v_complex))
@test vr_sum ≈ sum(real.(v_complex_global)) atol=TOL_cpx
vi_sum = sum(imag(v_complex))
@test vi_sum ≈ sum(imag.(v_complex_global)) atol=TOL_cpx


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

println(io0(), "Test Summary: Sparse API | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
