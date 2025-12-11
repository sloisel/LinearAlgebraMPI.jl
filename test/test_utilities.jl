# Tests for utility functions: io0, show methods, and MPI->native conversions
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

ts = @testset QuietTestSet "Utilities" begin

if rank == 0
    println("[test] io0 rank selection")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] Vector conversion roundtrip")
    flush(stdout)
end

# Test Vector conversion: native -> MPI -> native (bit-for-bit)
v_original = Float64[1.5, -2.3, 3.7, 4.1, -5.9, 6.2, 7.8, -8.4, 9.0, 10.1]
v_mpi = VectorMPI(v_original)
v_back = Vector(v_mpi)
@test v_back === v_original || v_back == v_original  # bit-for-bit or equal
@test eltype(v_back) == eltype(v_original)
@test length(v_back) == length(v_original)

# Complex vector
v_complex = ComplexF64[1+2im, 3-4im, 5+6im, 7-8im, 9+10im]
v_mpi_c = VectorMPI(v_complex)
v_back_c = Vector(v_mpi_c)
@test v_back_c == v_complex
@test eltype(v_back_c) == ComplexF64

MPI.Barrier(comm)

if rank == 0
    println("[test] Matrix conversion roundtrip")
    flush(stdout)
end

# Test Matrix conversion: native -> MPI -> native (bit-for-bit)
M_original = Float64[1.1 2.2 3.3 4.4;
                     5.5 6.6 7.7 8.8;
                     9.9 10.0 11.1 12.2;
                     13.3 14.4 15.5 16.6;
                     17.7 18.8 19.9 20.0;
                     21.1 22.2 23.3 24.4]
M_mpi = MatrixMPI(M_original)
M_back = Matrix(M_mpi)
@test M_back == M_original
@test eltype(M_back) == eltype(M_original)
@test size(M_back) == size(M_original)

# Complex matrix
M_complex = ComplexF64[1+1im 2+2im 3+3im;
                       4-4im 5-5im 6-6im;
                       7+7im 8+8im 9+9im;
                       10-10im 11-11im 12-12im]
M_mpi_c = MatrixMPI(M_complex)
M_back_c = Matrix(M_mpi_c)
@test M_back_c == M_complex
@test eltype(M_back_c) == ComplexF64

MPI.Barrier(comm)

if rank == 0
    println("[test] SparseMatrixCSC conversion roundtrip")
    flush(stdout)
end

# Test SparseMatrixCSC conversion: native -> MPI -> native (bit-for-bit)
# Create a nontrivial sparse matrix with various patterns
I = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 9, 10, 5, 6, 11, 12, 8, 9, 10]
J = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 12, 13, 5, 6, 14, 15, 16]
V = Float64[1.1, -2.2, 3.3, -4.4, 5.5, -6.6, 7.7, -8.8, 9.9, -10.0,
            11.1, -12.2, 13.3, -14.4, 15.5, -16.6, 17.7, -18.8, 19.9, -20.0]
S_original = sparse(I, J, V, 15, 20)

S_mpi = SparseMatrixMPI{Float64}(S_original)
S_back = SparseMatrixCSC(S_mpi)

@test S_back == S_original
@test nnz(S_back) == nnz(S_original)
@test size(S_back) == size(S_original)
@test eltype(S_back) == eltype(S_original)

# Verify structure is identical
@test S_back.colptr == S_original.colptr
@test S_back.rowval == S_original.rowval
@test S_back.nzval == S_original.nzval

# Complex sparse matrix
I_c = [1, 2, 3, 4, 5, 1, 3, 5, 7, 9]
J_c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
V_c = ComplexF64[1+1im, 2-2im, 3+3im, 4-4im, 5+5im, 6-6im, 7+7im, 8-8im, 9+9im, 10-10im]
S_complex = sparse(I_c, J_c, V_c, 12, 12)

S_mpi_c = SparseMatrixMPI{ComplexF64}(S_complex)
S_back_c = SparseMatrixCSC(S_mpi_c)
@test S_back_c == S_complex
@test eltype(S_back_c) == ComplexF64

MPI.Barrier(comm)

if rank == 0
    println("[test] VectorMPI show methods")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] MatrixMPI show methods")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] SparseMatrixMPI show methods")
    flush(stdout)
end

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

MPI.Barrier(comm)

if rank == 0
    println("[test] io0 with show/print integration")
    flush(stdout)
end

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

MPI.Barrier(comm)

end  # testset

# Report results from rank 0
if rank == 0
    println("Test Summary: Utilities | Pass: $(ts.counts[:pass])  Fail: $(ts.counts[:fail])  Error: $(ts.counts[:error])")
    flush(stdout)
end

# Exit with appropriate code
exit_code = (ts.counts[:fail] + ts.counts[:error] > 0) ? 1 : 0
MPI.Barrier(comm)
MPI.Finalize()
exit(exit_code)
