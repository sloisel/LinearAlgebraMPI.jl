# GPU tests for LinearAlgebraMPI
# Tests Metal GPU support for VectorMPI operations

using Test

# Check if Metal is available BEFORE loading MPI
# (Metal must be loaded first for GPU detection to work)
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch e
    @info "Metal not available: $e"
    false
end

using MPI

# Initialize MPI if needed
if !MPI.Initialized()
    MPI.Init()
end

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra

if METAL_AVAILABLE
    @info "Metal is available, running GPU tests"

    @testset "Metal VectorMPI" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        nranks = MPI.Comm_size(MPI.COMM_WORLD)

        # Test 1: Basic VectorMPI conversion CPU <-> GPU
        @testset "CPU-GPU conversion" begin
            n = 100
            v_cpu = VectorMPI(Float32.(collect(1.0:n)))

            # Convert to GPU
            v_gpu = LinearAlgebraMPI.mtl(v_cpu)
            @test v_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Convert back to CPU
            v_cpu2 = LinearAlgebraMPI.cpu(v_gpu)
            @test v_cpu2 isa VectorMPI{Float32,Vector{Float32}}

            # Values should match
            @test v_cpu.v == v_cpu2.v
        end

        # Test 2: VectorMPI_local with GPU arrays
        @testset "VectorMPI_local with GPU" begin
            local_size = 10 + rank * 5  # Different size per rank
            local_data = MtlVector(Float32.(collect(1.0:local_size)))

            v_gpu = VectorMPI_local(local_data)
            @test v_gpu isa VectorMPI{Float32,<:Metal.MtlVector}
            @test length(v_gpu.v) == local_size
        end

        # Test 3: Vector addition on GPU
        @testset "Vector addition on GPU" begin
            n = 50
            u_cpu = VectorMPI(Float32.(rand(n)))
            v_cpu = VectorMPI(Float32.(rand(n)))

            u_gpu = LinearAlgebraMPI.mtl(u_cpu)
            v_gpu = LinearAlgebraMPI.mtl(v_cpu)

            # Add on GPU
            w_gpu = u_gpu + v_gpu
            @test w_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Compare with CPU result
            w_cpu = u_cpu + v_cpu
            @test Array(w_gpu.v) ≈ w_cpu.v
        end

        # Test 4: Scalar multiplication on GPU
        @testset "Scalar multiplication on GPU" begin
            n = 50
            v_cpu = VectorMPI(Float32.(rand(n)))
            v_gpu = LinearAlgebraMPI.mtl(v_cpu)

            # Scalar multiply on GPU
            w_gpu = 2.5f0 * v_gpu
            @test w_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Compare with CPU
            w_cpu = 2.5f0 * v_cpu
            @test Array(w_gpu.v) ≈ w_cpu.v
        end

        # Test 5: Vector dot product with GPU
        @testset "Vector dot product" begin
            n = 50
            x_cpu = VectorMPI(Float32.(rand(n)))
            y_cpu = VectorMPI(Float32.(rand(n)))

            x_gpu = LinearAlgebraMPI.mtl(x_cpu)
            y_gpu = LinearAlgebraMPI.mtl(y_cpu)

            # Dot product (reduction goes through MPI, needs CPU)
            # For now, convert back to CPU for dot
            d_cpu = dot(x_cpu, y_cpu)
            d_gpu = dot(LinearAlgebraMPI.cpu(x_gpu), LinearAlgebraMPI.cpu(y_gpu))
            @test d_cpu ≈ d_gpu
        end

        # Note: Mixed CPU-matrix/GPU-vector operations are not supported.
        # Use mtl(A) to move sparse matrix to GPU, then A_gpu * x_gpu.
        # See "Metal SparseMatrixMPI" testset for all-GPU sparse operations.

        # Test 6: Broadcasting on GPU
        @testset "Broadcasting on GPU" begin
            n = 50
            v_cpu = VectorMPI(Float32.(rand(n)))
            v_gpu = LinearAlgebraMPI.mtl(v_cpu)

            # Element-wise operations
            w_gpu = abs.(v_gpu)
            @test Array(w_gpu.v) ≈ abs.(v_cpu.v)

            # Note: Complex broadcasting like v .+ 1 may not work yet
            # since broadcasting returns VectorMPI with local v from broadcast
        end

        # Test 8: Dense MatrixMPI * VectorMPI with GPU vector
        @testset "Dense A*x with GPU vector" begin
            m, n = 20, 15
            # Create dense matrix (stays on CPU)
            A_full = Float32.(rand(m, n))
            A = MatrixMPI(A_full)

            # Create CPU and GPU vectors
            x_cpu = VectorMPI(Float32.(rand(n)))
            x_gpu = LinearAlgebraMPI.mtl(x_cpu)

            # Multiply with GPU vector
            y_gpu = A * x_gpu
            @test y_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Compare with CPU result
            y_cpu = A * x_cpu
            @test Array(y_gpu.v) ≈ y_cpu.v atol=1e-5
        end

        # Test 9: Dense transpose(A) * x with GPU vector
        @testset "Dense transpose(A)*x with GPU vector" begin
            m, n = 20, 15
            A_full = Float32.(rand(m, n))
            A = MatrixMPI(A_full)

            # Create CPU and GPU vectors
            x_cpu = VectorMPI(Float32.(rand(m)))
            x_gpu = LinearAlgebraMPI.mtl(x_cpu)

            # Multiply with GPU vector
            y_gpu = transpose(A) * x_gpu
            @test y_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Compare with CPU result
            y_cpu = transpose(A) * x_cpu
            @test Array(y_gpu.v) ≈ y_cpu.v atol=1e-5
        end

        # Test 10: VectorMPI subtraction on GPU
        @testset "Vector subtraction on GPU" begin
            n = 50
            u_cpu = VectorMPI(Float32.(rand(n)))
            v_cpu = VectorMPI(Float32.(rand(n)))

            u_gpu = LinearAlgebraMPI.mtl(u_cpu)
            v_gpu = LinearAlgebraMPI.mtl(v_cpu)

            # Subtract on GPU
            w_gpu = u_gpu - v_gpu
            @test w_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Compare with CPU result
            w_cpu = u_cpu - v_cpu
            @test Array(w_gpu.v) ≈ w_cpu.v
        end

        # Test 11: VectorMPI norm on GPU
        @testset "Vector norm on GPU" begin
            n = 50
            v_cpu = VectorMPI(Float32.(rand(n)))
            v_gpu = LinearAlgebraMPI.mtl(v_cpu)

            # Norm (reduction goes through MPI)
            n_cpu = norm(v_cpu)
            n_gpu = norm(LinearAlgebraMPI.cpu(v_gpu))
            @test n_cpu ≈ n_gpu
        end

        println(io0(), "=== All Metal VectorMPI GPU tests passed! ===")
    end

    @testset "Metal SparseMatrixMPI" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        nranks = MPI.Comm_size(MPI.COMM_WORLD)

        # Test 1: Basic SparseMatrixMPI conversion CPU <-> GPU
        @testset "CPU-GPU conversion" begin
            n = 50
            A_full = Float32.(sprand(n, n, 0.2)) + Float32(1.0)*I
            A_cpu = SparseMatrixMPI{Float32}(A_full)

            # Convert to GPU (only nzval moves to GPU)
            A_gpu = LinearAlgebraMPI.mtl(A_cpu)
            @test A_gpu isa SparseMatrixMPI{Float32,Int,<:Metal.MtlVector}
            @test A_gpu.rowptr isa Vector  # Structure stays on CPU
            @test A_gpu.colval isa Vector
            @test A_gpu.nzval isa Metal.MtlVector

            # Convert back to CPU
            A_cpu2 = LinearAlgebraMPI.cpu(A_gpu)
            @test A_cpu2 isa SparseMatrixMPI{Float32,Int,Vector{Float32}}
            @test A_cpu.nzval == A_cpu2.nzval
        end

        # Test 2: Scalar multiplication on GPU sparse matrix
        @testset "Scalar multiplication on GPU" begin
            n = 50
            A_full = Float32.(sprand(n, n, 0.2))
            A_cpu = SparseMatrixMPI{Float32}(A_full)
            A_gpu = LinearAlgebraMPI.mtl(A_cpu)

            # Scalar multiply on GPU
            B_gpu = 2.5f0 * A_gpu
            @test B_gpu isa SparseMatrixMPI{Float32,Int,<:Metal.MtlVector}

            # Compare with CPU
            B_cpu = 2.5f0 * A_cpu
            @test Array(B_gpu.nzval) ≈ B_cpu.nzval
        end

        # Test 3: Element-wise abs on GPU sparse matrix
        @testset "Element-wise abs on GPU" begin
            n = 50
            A_full = Float32.(sprand(n, n, 0.2)) .- Float32(0.5)
            A_cpu = SparseMatrixMPI{Float32}(A_full)
            A_gpu = LinearAlgebraMPI.mtl(A_cpu)

            # abs on GPU
            B_gpu = abs(A_gpu)
            @test B_gpu isa SparseMatrixMPI{Float32,Int,<:Metal.MtlVector}

            # Compare with CPU
            B_cpu = abs(A_cpu)
            @test Array(B_gpu.nzval) ≈ B_cpu.nzval
        end

        # Test 4: GPU sparse matrix * GPU vector
        @testset "GPU A * GPU x" begin
            n = 30
            A_full = Float32.(sprand(n, n, 0.3)) + Float32(1.0)*I
            A_cpu = SparseMatrixMPI{Float32}(A_full)
            A_gpu = LinearAlgebraMPI.mtl(A_cpu)

            x_cpu = VectorMPI(Float32.(rand(n)))
            x_gpu = LinearAlgebraMPI.mtl(x_cpu)

            # Multiply GPU * GPU
            y_gpu = A_gpu * x_gpu
            @test y_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Compare with CPU result
            y_cpu = A_cpu * x_cpu
            @test Array(y_gpu.v) ≈ y_cpu.v atol=1e-5
        end

        # Test 5: Complex sparse matrix on GPU
        @testset "Complex sparse on GPU" begin
            n = 30
            A_full = ComplexF32.(sprand(n, n, 0.2)) .+ ComplexF32(0.5 + 0.3im)
            A_cpu = SparseMatrixMPI{ComplexF32}(A_full)
            A_gpu = LinearAlgebraMPI.mtl(A_cpu)

            # conj on GPU
            B_gpu = conj(A_gpu)
            @test B_gpu isa SparseMatrixMPI{ComplexF32,Int,<:Metal.MtlVector}

            # Compare with CPU
            B_cpu = conj(A_cpu)
            @test Array(B_gpu.nzval) ≈ B_cpu.nzval
        end

        # Test 6: GPU sparse matrix * GPU sparse matrix (A*B)
        @testset "GPU A * GPU B" begin
            m, k, n = 20, 25, 15
            A_full = Float32.(sprand(m, k, 0.3))
            B_full = Float32.(sprand(k, n, 0.3))
            A_cpu = SparseMatrixMPI{Float32}(A_full)
            B_cpu = SparseMatrixMPI{Float32}(B_full)
            A_gpu = LinearAlgebraMPI.mtl(A_cpu)
            B_gpu = LinearAlgebraMPI.mtl(B_cpu)

            # Multiply GPU * GPU
            C_gpu = A_gpu * B_gpu
            @test C_gpu isa SparseMatrixMPI{Float32,Int,<:Metal.MtlVector}

            # Compare with CPU result
            C_cpu = A_cpu * B_cpu
            @test Array(C_gpu.nzval) ≈ C_cpu.nzval atol=1e-5
        end

        # Test 7: Complex GPU sparse matrix * GPU sparse matrix
        @testset "Complex GPU A * GPU B" begin
            m, k, n = 15, 20, 12
            A_full = ComplexF32.(sprand(m, k, 0.2)) .+ ComplexF32(0.1 + 0.1im)
            B_full = ComplexF32.(sprand(k, n, 0.2)) .+ ComplexF32(0.1 - 0.1im)
            A_cpu = SparseMatrixMPI{ComplexF32}(A_full)
            B_cpu = SparseMatrixMPI{ComplexF32}(B_full)
            A_gpu = LinearAlgebraMPI.mtl(A_cpu)
            B_gpu = LinearAlgebraMPI.mtl(B_cpu)

            # Multiply GPU * GPU
            C_gpu = A_gpu * B_gpu
            @test C_gpu isa SparseMatrixMPI{ComplexF32,Int,<:Metal.MtlVector}

            # Compare with CPU result
            C_cpu = A_cpu * B_cpu
            @test Array(C_gpu.nzval) ≈ C_cpu.nzval atol=1e-4
        end

        # Test 8: GPU sparse matrix addition
        @testset "GPU A + GPU B" begin
            n = 30
            A_full = Float32.(sprand(n, n, 0.3))
            B_full = Float32.(sprand(n, n, 0.3))
            A_cpu = SparseMatrixMPI{Float32}(A_full)
            B_cpu = SparseMatrixMPI{Float32}(B_full)
            A_gpu = LinearAlgebraMPI.mtl(A_cpu)
            B_gpu = LinearAlgebraMPI.mtl(B_cpu)

            # Add GPU + GPU
            C_gpu = A_gpu + B_gpu
            @test C_gpu isa SparseMatrixMPI{Float32,Int,<:Metal.MtlVector}

            # Compare with CPU result
            C_cpu = A_cpu + B_cpu
            @test Array(C_gpu.nzval) ≈ C_cpu.nzval atol=1e-5
        end

        # Test 9: GPU sparse matrix subtraction
        @testset "GPU A - GPU B" begin
            n = 30
            A_full = Float32.(sprand(n, n, 0.3))
            B_full = Float32.(sprand(n, n, 0.3))
            A_cpu = SparseMatrixMPI{Float32}(A_full)
            B_cpu = SparseMatrixMPI{Float32}(B_full)
            A_gpu = LinearAlgebraMPI.mtl(A_cpu)
            B_gpu = LinearAlgebraMPI.mtl(B_cpu)

            # Subtract GPU - GPU
            C_gpu = A_gpu - B_gpu
            @test C_gpu isa SparseMatrixMPI{Float32,Int,<:Metal.MtlVector}

            # Compare with CPU result
            C_cpu = A_cpu - B_cpu
            @test Array(C_gpu.nzval) ≈ C_cpu.nzval atol=1e-5
        end

        # Test 10: GPU sparse matrix transpose
        @testset "GPU transpose(A)" begin
            m, n = 25, 30
            A_full = Float32.(sprand(m, n, 0.3))
            A_cpu = SparseMatrixMPI{Float32}(A_full)
            A_gpu = LinearAlgebraMPI.mtl(A_cpu)

            # Materialize transpose on GPU
            At_gpu = SparseMatrixMPI(transpose(A_gpu))
            @test At_gpu isa SparseMatrixMPI{Float32,Int,<:Metal.MtlVector}

            # Compare with CPU result
            At_cpu = SparseMatrixMPI(transpose(A_cpu))
            @test Array(At_gpu.nzval) ≈ At_cpu.nzval atol=1e-5
        end

        println(io0(), "=== All Metal SparseMatrixMPI GPU tests passed! ===")
    end

    @testset "Metal MatrixMPI" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        nranks = MPI.Comm_size(MPI.COMM_WORLD)

        # Test 1: Basic MatrixMPI conversion CPU <-> GPU
        @testset "CPU-GPU conversion" begin
            m, n = 40, 30
            A_full = Float32.(rand(m, n))
            A_cpu = MatrixMPI(A_full)

            # Convert to GPU
            A_gpu = LinearAlgebraMPI.mtl(A_cpu)
            @test A_gpu isa MatrixMPI{Float32,<:Metal.MtlMatrix}

            # Convert back to CPU
            A_cpu2 = LinearAlgebraMPI.cpu(A_gpu)
            @test A_cpu2 isa MatrixMPI{Float32,Matrix{Float32}}
            @test A_cpu.A == A_cpu2.A
        end

        # Test 2: GPU MatrixMPI * GPU VectorMPI
        @testset "GPU A * GPU x" begin
            m, n = 40, 30
            A_full = Float32.(rand(m, n))
            A_cpu = MatrixMPI(A_full)
            A_gpu = LinearAlgebraMPI.mtl(A_cpu)

            x_cpu = VectorMPI(Float32.(rand(n)))
            x_gpu = LinearAlgebraMPI.mtl(x_cpu)

            # Multiply GPU * GPU
            y_gpu = A_gpu * x_gpu
            @test y_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Compare with CPU result
            y_cpu = A_cpu * x_cpu
            @test Array(y_gpu.v) ≈ y_cpu.v atol=1e-5
        end

        # Test 3: Scalar multiplication on GPU dense matrix
        @testset "Scalar multiplication on GPU" begin
            m, n = 30, 20
            A_full = Float32.(rand(m, n))
            A_cpu = MatrixMPI(A_full)
            A_gpu = LinearAlgebraMPI.mtl(A_cpu)

            # Scalar multiply on GPU
            B_gpu = 3.0f0 * A_gpu
            @test B_gpu isa MatrixMPI{Float32,<:Metal.MtlMatrix}

            # Compare with CPU
            B_cpu = 3.0f0 * A_cpu
            @test Array(B_gpu.A) ≈ B_cpu.A
        end

        println(io0(), "=== All Metal MatrixMPI GPU tests passed! ===")
    end

    println(io0(), "=== All Metal GPU tests passed! ===")
else
    @info "Metal not available, skipping GPU tests"

    @testset "GPU tests skipped" begin
        @test_skip "Metal not available"
    end
end
