# Examples

This page provides detailed examples of using LinearAlgebraMPI.jl for various distributed sparse matrix operations.

## Matrix Multiplication

### Square Matrices

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Create a tridiagonal matrix (same on all ranks)
n = 100
I = [1:n; 1:n-1; 2:n]
J = [1:n; 2:n; 1:n-1]
V = [2.0*ones(n); -0.5*ones(n-1); -0.5*ones(n-1)]
A = sparse(I, J, V, n, n)

# Create another tridiagonal matrix
V2 = [1.5*ones(n); 0.25*ones(n-1); 0.25*ones(n-1)]
B = sparse(I, J, V2, n, n)

# Distribute matrices
Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)

# Multiply
Cdist = Adist * Bdist

# Verify against reference
C_ref = A * B
C_ref_dist = SparseMatrixMPI{Float64}(C_ref)
err = norm(Cdist - C_ref_dist, Inf)

if rank == 0
    println("Multiplication error: $err")
end

MPI.Finalize()
```

### Non-Square Matrices

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra

rank = MPI.Comm_rank(MPI.COMM_WORLD)

# A is 6x8, B is 8x10, result is 6x10
m, k, n = 6, 8, 10

I_A = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
J_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]
V_A = Float64.(1:length(I_A))
A = sparse(I_A, J_A, V_A, m, k)

I_B = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
J_B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
V_B = Float64.(1:length(I_B))
B = sparse(I_B, J_B, V_B, k, n)

Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)

Cdist = Adist * Bdist

@assert size(Cdist) == (m, n)

if rank == 0
    println("Result size: $(size(Cdist))")
end

MPI.Finalize()
```

## Complex Matrices

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra

rank = MPI.Comm_rank(MPI.COMM_WORLD)

n = 8
I = [1:n; 1:n-1; 2:n]
J = [1:n; 2:n; 1:n-1]

# Complex values
V_A = ComplexF64.([2.0*ones(n); -0.5*ones(n-1); -0.5*ones(n-1)]) .+
      im .* ComplexF64.([0.1*ones(n); 0.2*ones(n-1); -0.2*ones(n-1)])
A = sparse(I, J, V_A, n, n)

V_B = ComplexF64.([1.5*ones(n); 0.25*ones(n-1); 0.25*ones(n-1)]) .+
      im .* ComplexF64.([-0.1*ones(n); 0.1*ones(n-1); 0.1*ones(n-1)])
B = sparse(I, J, V_B, n, n)

Adist = SparseMatrixMPI{ComplexF64}(A)
Bdist = SparseMatrixMPI{ComplexF64}(B)

# Multiplication
Cdist = Adist * Bdist

# Conjugate
Aconj = conj(Adist)

# Adjoint (conjugate transpose) - returns lazy wrapper
Aadj = Adist'

# Materialize adjoint
plan = TransposePlan(Aadj.parent)
Aadj_mat = execute_plan!(plan, Aadj.parent)

if rank == 0
    println("Complex matrix operations completed")
end

MPI.Finalize()
```

## Addition and Subtraction

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra

rank = MPI.Comm_rank(MPI.COMM_WORLD)

n = 8

# Matrices with different sparsity patterns
# Matrix A: upper triangular entries
I_A = [1, 1, 2, 3, 4, 5, 6, 7, 8]
J_A = [1, 2, 2, 3, 4, 5, 6, 7, 8]
V_A = Float64.(1:9)
A = sparse(I_A, J_A, V_A, n, n)

# Matrix B: lower triangular entries
I_B = [1, 2, 2, 3, 4, 5, 6, 7, 8]
J_B = [1, 1, 2, 3, 4, 5, 6, 7, 8]
V_B = Float64.(9:-1:1)
B = sparse(I_B, J_B, V_B, n, n)

Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)

# Addition - handles different sparsity patterns
Cdist = Adist + Bdist

# Subtraction
Ddist = Adist - Bdist

# Verify
C_ref_dist = SparseMatrixMPI{Float64}(A + B)
err = norm(Cdist - C_ref_dist, Inf)

if rank == 0
    println("Addition error: $err")
end

MPI.Finalize()
```

## Transpose Operations

### Lazy Transpose

The lazy transpose creates a wrapper without actually transposing the data:

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra

rank = MPI.Comm_rank(MPI.COMM_WORLD)

m, n = 8, 6
I_C = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
J_C = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
V_C = Float64.(1:length(I_C))
C = sparse(I_C, J_C, V_C, m, n)

I_D = [1, 2, 3, 4, 5, 6, 1, 2]
J_D = [1, 2, 3, 4, 5, 6, 7, 8]
V_D = Float64.(1:length(I_D))
D = sparse(I_D, J_D, V_D, n, m)

Cdist = SparseMatrixMPI{Float64}(C)
Ddist = SparseMatrixMPI{Float64}(D)

# transpose(C) * transpose(D) = transpose(D * C)
# This is computed lazily
result_lazy = transpose(Cdist) * transpose(Ddist)

# The result is wrapped in Transpose
# Materialize to get the actual matrix
plan = TransposePlan(result_lazy.parent)
result_dist = execute_plan!(plan, result_lazy.parent)

# Verify
ref = sparse((D * C)')
ref_dist = SparseMatrixMPI{Float64}(ref)
err = norm(result_dist - ref_dist, Inf)

if rank == 0
    println("Lazy transpose multiplication error: $err")
end

MPI.Finalize()
```

### Eager Transpose

When you need the transposed data immediately:

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays

rank = MPI.Comm_rank(MPI.COMM_WORLD)

m, n = 8, 6
I = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7]
J = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
V = Float64.(1:length(I))
A = sparse(I, J, V, m, n)

Adist = SparseMatrixMPI{Float64}(A)

# Create transpose plan
plan = TransposePlan(Adist)

# Execute to get transposed matrix
At_dist = execute_plan!(plan, Adist)

# At_dist is now an SparseMatrixMPI representing A^T
@assert size(At_dist) == (n, m)

if rank == 0
    println("Transpose size: $(size(At_dist))")
end

MPI.Finalize()
```

### Mixed Transpose Multiplication

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra

rank = MPI.Comm_rank(MPI.COMM_WORLD)

# A is 8x6, so A' is 6x8
# B is 8x10, so A' * B is 6x10
m, n, p = 8, 6, 10

I_A = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7]
J_A = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
V_A = Float64.(1:length(I_A))
A = sparse(I_A, J_A, V_A, m, n)

I_B = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3, 5, 7]
J_B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2]
V_B = Float64.(1:length(I_B))
B = sparse(I_B, J_B, V_B, m, p)

Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)

# transpose(A) * B - A is automatically materialized as transpose
result_dist = transpose(Adist) * Bdist

# Verify
ref = sparse(A') * B
ref_dist = SparseMatrixMPI{Float64}(ref)
err = norm(result_dist - ref_dist, Inf)

if rank == 0
    println("transpose(A) * B error: $err")
end

MPI.Finalize()
```

## Scalar Multiplication

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra

rank = MPI.Comm_rank(MPI.COMM_WORLD)

m, n = 6, 8
I = [1, 2, 3, 4, 5, 6, 1, 3]
J = [1, 2, 3, 4, 5, 6, 7, 8]
V = Float64.(1:length(I))
A = sparse(I, J, V, m, n)

Adist = SparseMatrixMPI{Float64}(A)

# Scalar times matrix
a = 2.5
result1 = a * Adist
result2 = Adist * a  # Equivalent

# Scalar times lazy transpose
At = transpose(Adist)
result3 = a * At  # Returns transpose(a * A)

# Verify
ref_dist = SparseMatrixMPI{Float64}(a * A)
err1 = norm(result1 - ref_dist, Inf)
err2 = norm(result2 - ref_dist, Inf)

if rank == 0
    println("Scalar multiplication errors: $err1, $err2")
end

MPI.Finalize()
```

## Computing Norms

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra

rank = MPI.Comm_rank(MPI.COMM_WORLD)

m, n = 6, 8
I = [1, 2, 3, 4, 5, 6, 1, 3, 2, 4]
J = [1, 2, 3, 4, 5, 6, 7, 8, 1, 3]
V = Float64.(1:length(I))
A = sparse(I, J, V, m, n)

Adist = SparseMatrixMPI{Float64}(A)

# Element-wise norms (treating matrix as vector)
frob_norm = norm(Adist)        # Frobenius (2-norm)
one_norm = norm(Adist, 1)      # Sum of absolute values
inf_norm = norm(Adist, Inf)    # Max absolute value
p_norm = norm(Adist, 3)        # General p-norm

# Operator norms
op_1 = opnorm(Adist, 1)        # Max absolute column sum
op_inf = opnorm(Adist, Inf)    # Max absolute row sum

if rank == 0
    println("Frobenius norm: $frob_norm")
    println("1-norm: $one_norm")
    println("Inf-norm: $inf_norm")
    println("3-norm: $p_norm")
    println("Operator 1-norm: $op_1")
    println("Operator Inf-norm: $op_inf")
end

MPI.Finalize()
```

## Iterative Methods Example

Here's an example of using LinearAlgebraMPI.jl for power iteration to find the dominant eigenvalue:

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Create a symmetric positive definite matrix
n = 100
I = [1:n; 1:n-1; 2:n]
J = [1:n; 2:n; 1:n-1]
V = [4.0*ones(n); -ones(n-1); -ones(n-1)]
A = sparse(I, J, V, n, n)

Adist = SparseMatrixMPI{Float64}(A)

# For power iteration, we need matrix-vector products
# Currently LinearAlgebraMPI focuses on matrix-matrix products
# This example shows how to use A*A for related computations

# Compute A^2
A2dist = Adist * Adist

# Compute the Frobenius norm of A^2
norm_A2 = norm(A2dist)

if rank == 0
    println("||A^2||_F = $norm_A2")
    # For SPD matrices, this relates to the eigenvalues
end

MPI.Finalize()
```

## Plan Caching and Management

```julia
using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays

rank = MPI.Comm_rank(MPI.COMM_WORLD)

n = 100
A = spdiagm(0 => 2.0*ones(n), 1 => -ones(n-1), -1 => -ones(n-1))
B = spdiagm(0 => 1.5*ones(n), 1 => 0.5*ones(n-1), -1 => 0.5*ones(n-1))

Adist = SparseMatrixMPI{Float64}(A)
Bdist = SparseMatrixMPI{Float64}(B)

# First multiplication - creates and caches the plan
C1 = Adist * Bdist

# Second multiplication - reuses cached plan (faster)
C2 = Adist * Bdist

# Third multiplication - still uses cached plan
C3 = Adist * Bdist

# Clear caches when done to free memory
clear_plan_cache!()

if rank == 0
    println("Cached multiplication completed")
end

MPI.Finalize()
```
