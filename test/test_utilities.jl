# Tests for utility functions: io0, show methods, and MPI->native conversions
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

ts = @testset QuietTestSet "Utilities" begin

# io0 tests don't need parameterization - they test I/O behavior
println(io0(), "[test] io0 rank selection")

# Test io0 - capture output to buffer
io_buf = IOBuffer()
print(io0(io_buf), "test")
output = String(take!(io_buf))
if rank == 0
    @test output == "test"
else
    @test output == ""
end

# Test io0 with custom rank set
io_buf = IOBuffer()
print(io0(io_buf; r=Set([1, 2])), "hello")
output = String(take!(io_buf))
if rank in [1, 2]
    @test output == "hello"
else
    @test output == ""
end

# Test io0 with devnull override
io_buf = IOBuffer()
dn_buf = IOBuffer()
result_io = io0(io_buf; r=Set([0]), dn=dn_buf)
if rank == 0
    @test result_io === io_buf
else
    @test result_io === dn_buf
end


# Parameterized tests for type conversions
for (T, to_backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)

    println(io0(), "[test] Vector conversion roundtrip ($T, $backend_name)")

    # Test Vector conversion: native -> MPI -> native (bit-for-bit)
    v_original = T.([1.5, -2.3, 3.7, 4.1, -5.9, 6.2, 7.8, -8.4, 9.0, 10.1])
    v_mpi = to_backend(VectorMPI(v_original))
    v_back = Vector(v_mpi)
    @test norm(v_back - v_original) < TOL
    @test eltype(v_back) == T
    @test length(v_back) == length(v_original)


    println(io0(), "[test] Matrix conversion roundtrip ($T, $backend_name)")

    # Test Matrix conversion: native -> MPI -> native (bit-for-bit)
    M_original = T.([1.1 2.2 3.3 4.4;
                     5.5 6.6 7.7 8.8;
                     9.9 10.0 11.1 12.2;
                     13.3 14.4 15.5 16.6;
                     17.7 18.8 19.9 20.0;
                     21.1 22.2 23.3 24.4])
    M_mpi = to_backend(MatrixMPI(M_original))
    M_back = Matrix(M_mpi)
    @test norm(M_back - M_original) < TOL
    @test eltype(M_back) == T
    @test size(M_back) == size(M_original)


    println(io0(), "[test] SparseMatrixCSC conversion roundtrip ($T, $backend_name)")

    # Test SparseMatrixCSC conversion: native -> MPI -> native (bit-for-bit)
    # Create a nontrivial sparse matrix with various patterns
    I_sp = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 9, 10, 5, 6, 11, 12, 8, 9, 10]
    J_sp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 12, 13, 5, 6, 14, 15, 16]
    V_sp = T.([1.1, -2.2, 3.3, -4.4, 5.5, -6.6, 7.7, -8.8, 9.9, -10.0,
               11.1, -12.2, 13.3, -14.4, 15.5, -16.6, 17.7, -18.8, 19.9, -20.0])
    S_original = sparse(I_sp, J_sp, V_sp, 15, 20)

    S_mpi = to_backend(SparseMatrixMPI{T}(S_original))
    S_back = SparseMatrixCSC(S_mpi)

    @test norm(S_back - S_original, Inf) < TOL
    @test nnz(S_back) == nnz(S_original)
    @test size(S_back) == size(S_original)
    @test eltype(S_back) == T

end  # for (T, to_backend, backend_name)


# Show method tests (CPU only, display behavior)
println(io0(), "[test] VectorMPI show methods")

# Test show methods for VectorMPI
v_test = VectorMPI(Float64[1.0, 2.0, 3.0, 4.0])
io = IOBuffer()
show(io, v_test)
s = String(take!(io))
@test occursin("VectorMPI", s)
@test occursin("Float64", s)

io = IOBuffer()
show(io, MIME("text/plain"), v_test)
s = String(take!(io))
@test occursin("VectorMPI", s)
@test occursin("4", s)  # length

# Test string interpolation
s_interp = "$v_test"
@test occursin("VectorMPI", s_interp)


println(io0(), "[test] MatrixMPI show methods")

# Test show methods for MatrixMPI
M_test = MatrixMPI(Float64[1.0 2.0; 3.0 4.0; 5.0 6.0])
io = IOBuffer()
show(io, M_test)
s = String(take!(io))
@test occursin("MatrixMPI", s)
@test occursin("Float64", s)

io = IOBuffer()
show(io, MIME("text/plain"), M_test)
s = String(take!(io))
@test occursin("MatrixMPI", s)

# Test string interpolation
s_interp = "$M_test"
@test occursin("MatrixMPI", s_interp)


println(io0(), "[test] SparseMatrixMPI show methods")

# Test show methods for SparseMatrixMPI
S_test = SparseMatrixMPI{Float64}(sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 5, 5))
io = IOBuffer()
show(io, S_test)
s = String(take!(io))
@test occursin("SparseMatrixMPI", s)
@test occursin("Float64", s)

io = IOBuffer()
show(io, MIME("text/plain"), S_test)
s = String(take!(io))
@test occursin("SparseMatrixMPI", s)
@test occursin("stored entries", s)

# Test string interpolation
s_interp = "$S_test"
@test occursin("SparseMatrixMPI", s_interp)


println(io0(), "[test] io0 with show/print integration")

# Test that io0 works with println and MPI types
io_buf = IOBuffer()
println(io0(io_buf), v_test)
output = String(take!(io_buf))
if rank == 0
    @test occursin("VectorMPI", output)
else
    @test output == ""
end

io_buf = IOBuffer()
println(io0(io_buf), "Value: ", M_test)
output = String(take!(io_buf))
if rank == 0
    @test occursin("MatrixMPI", output)
    @test occursin("Value:", output)
else
    @test output == ""
end


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

println(io0(), "Test Summary: Utilities | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")

MPI.Finalize()

if global_counts[2] > 0 || global_counts[3] > 0
    exit(1)
end
