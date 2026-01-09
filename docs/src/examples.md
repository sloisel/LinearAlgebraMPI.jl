# Examples

This page provides detailed examples of using LinearAlgebraMPI.jl for various distributed sparse matrix operations.

## Matrix Multiplication

### Square Matrices

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays
using LinearAlgebra

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

println(io0(), "Multiplication error: $err")

```

### Non-Square Matrices

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays
using LinearAlgebra

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

println(io0(), "Result size: $(size(Cdist))")

```

## Complex Matrices

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays
using LinearAlgebra

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

# Using adjoint in multiplication (materializes automatically)
result = Aadj * Bdist

println(io0(), "Complex matrix operations completed")

```

## Addition and Subtraction

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays
using LinearAlgebra

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

println(io0(), "Addition error: $err")

```

## Transpose Operations

### Lazy Transpose

The `transpose` function creates a lazy wrapper without transposing the data. This is efficient because the actual transpose is only computed when needed:

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays
using LinearAlgebra

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
# This is computed efficiently without explicitly transposing
result = transpose(Cdist) * transpose(Ddist)

println(io0(), "Lazy transpose multiplication completed")

```

### Transpose in Multiplication

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays
using LinearAlgebra

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

println(io0(), "transpose(A) * B error: $err")

```

## Scalar Multiplication

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays
using LinearAlgebra

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

println(io0(), "Scalar multiplication errors: $err1, $err2")

```

## Computing Norms

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays
using LinearAlgebra

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

println(io0(), "Frobenius norm: $frob_norm")
println(io0(), "1-norm: $one_norm")
println(io0(), "Inf-norm: $inf_norm")
println(io0(), "3-norm: $p_norm")
println(io0(), "Operator 1-norm: $op_1")
println(io0(), "Operator Inf-norm: $op_inf")

```

## Iterative Methods Example

Here's an example of using LinearAlgebraMPI.jl for power iteration to find the dominant eigenvalue:

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays
using LinearAlgebra

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

println(io0(), "||A^2||_F = $norm_A2")
# For SPD matrices, this relates to the eigenvalues

```

## Solving Linear Systems

LinearAlgebraMPI provides distributed sparse direct solvers using the multifrontal method.

### LDLT Factorization (Symmetric Matrices)

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays
using LinearAlgebra

# Create a symmetric positive definite tridiagonal matrix
n = 100
I = [1:n; 1:n-1; 2:n]
J = [1:n; 2:n; 1:n-1]
V = [4.0*ones(n); -ones(n-1); -ones(n-1)]
A = sparse(I, J, V, n, n)

# Distribute the matrix
Adist = SparseMatrixMPI{Float64}(A)

# Compute LDLT factorization
F = ldlt(Adist)

# Create right-hand side
b = VectorMPI(ones(n))

# Solve Ax = b
x = solve(F, b)

# Or use backslash syntax
x = F \ b

# Verify solution
x_full = Vector(x)
residual = norm(A * x_full - ones(n), Inf)

println(io0(), "LDLT solve residual: $residual")

```

### LU Factorization (General Matrices)

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays
using LinearAlgebra

# Create a general (non-symmetric) tridiagonal matrix
n = 100
I = [1:n; 1:n-1; 2:n]
J = [1:n; 2:n; 1:n-1]
V = [2.0*ones(n); -0.5*ones(n-1); -0.8*ones(n-1)]  # Non-symmetric
A = sparse(I, J, V, n, n)

# Distribute and factorize
Adist = SparseMatrixMPI{Float64}(A)
F = lu(Adist)

# Solve
b = VectorMPI(ones(n))
x = solve(F, b)

# Verify
x_full = Vector(x)
residual = norm(A * x_full - ones(n), Inf)

println(io0(), "LU solve residual: $residual")

```

### Symmetric Indefinite Matrices

LDLT uses Bunch-Kaufman pivoting to handle symmetric indefinite matrices:

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays
using LinearAlgebra

# Symmetric indefinite matrix (alternating signs on diagonal)
n = 50
I = [1:n; 1:n-1; 2:n]
J = [1:n; 2:n; 1:n-1]
diag_vals = [(-1.0)^i * 2.0 for i in 1:n]  # Alternating signs
V = [diag_vals; -ones(n-1); -ones(n-1)]
A = sparse(I, J, V, n, n)

Adist = SparseMatrixMPI{Float64}(A)
F = ldlt(Adist)

b = VectorMPI(collect(1.0:n))
x = solve(F, b)

x_full = Vector(x)
residual = norm(A * x_full - collect(1.0:n), Inf)

println(io0(), "Indefinite LDLT residual: $residual")

```

### Reusing Symbolic Factorization

For sequences of matrices with the same sparsity pattern, the symbolic factorization is cached and reused:

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays
using LinearAlgebra

n = 100
I = [1:n; 1:n-1; 2:n]
J = [1:n; 2:n; 1:n-1]

# First matrix
V1 = [4.0*ones(n); -ones(n-1); -ones(n-1)]
A1 = sparse(I, J, V1, n, n)
A1dist = SparseMatrixMPI{Float64}(A1)

# First factorization - computes symbolic phase
F1 = ldlt(A1dist; reuse_symbolic=true)

# Second matrix - same structure, different values
V2 = [8.0*ones(n); -2.0*ones(n-1); -2.0*ones(n-1)]
A2 = sparse(I, J, V2, n, n)
A2dist = SparseMatrixMPI{Float64}(A2)

# Second factorization - reuses cached symbolic phase (faster)
F2 = ldlt(A2dist; reuse_symbolic=true)

# Both factorizations work
b = VectorMPI(ones(n))
x1 = solve(F1, b)
x2 = solve(F2, b)

x1_full = Vector(x1)
x2_full = Vector(x2)
println(io0(), "F1 residual: ", norm(A1 * x1_full - ones(n), Inf))
println(io0(), "F2 residual: ", norm(A2 * x2_full - ones(n), Inf))

```

## Plan Caching and Management

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays

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
clear_plan_cache!()  # Clears all caches including factorization

# Or clear specific caches:
# clear_symbolic_cache!()           # Symbolic factorizations only
# clear_factorization_plan_cache!() # Factorization plans only

println(io0(), "Cached multiplication completed")

```

## Dense Matrix Operations with mapslices

The `mapslices` function applies a function to each row or column of a distributed dense matrix. This is useful for computing row-wise or column-wise statistics.

### Row-wise Operations (dims=2)

Row-wise operations are local - no MPI communication is needed since rows are already distributed:

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using LinearAlgebra

# Create a deterministic dense matrix (same on all ranks)
m, n = 100, 10
A_global = Float64.([i + 0.1*j for i in 1:m, j in 1:n])

# Distribute
Adist = MatrixMPI(A_global)

# Compute row statistics: for each row, compute [norm, max, sum]
# This transforms 100×10 matrix to 100×3 matrix
row_stats = mapslices(x -> [norm(x), maximum(x), sum(x)], Adist; dims=2)

println(io0(), "Row statistics shape: $(size(row_stats))")  # (100, 3)

```

### Column-wise Operations (dims=1)

Column-wise operations require MPI communication to gather each full column:

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using LinearAlgebra

# Create a deterministic dense matrix
m, n = 100, 10
A_global = Float64.([i + 0.1*j for i in 1:m, j in 1:n])

Adist = MatrixMPI(A_global)

# Compute column statistics: for each column, compute [norm, max]
# This transforms 100×10 matrix to 2×10 matrix
col_stats = mapslices(x -> [norm(x), maximum(x)], Adist; dims=1)

println(io0(), "Column statistics shape: $(size(col_stats))")  # (2, 10)

```

### Use Case: Replacing vcat(f.(eachrow(A))...)

The standard Julia pattern `vcat(f.(eachrow(A))...)` doesn't work with distributed matrices because the type information is lost after broadcasting. Use `mapslices` instead:

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using LinearAlgebra

# Standard Julia pattern (for comparison):
# A = randn(5, 2)
# f(x) = transpose([norm(x), maximum(x)])
# B = vcat(f.(eachrow(A))...)

# MPI-compatible equivalent:
A_global = Float64.([i + 0.1*j for i in 1:100, j in 1:10])
Adist = MatrixMPI(A_global)

# Use mapslices with dims=2 to apply function to each row
# The function returns a vector, which becomes a row in the result
g(x) = [norm(x), maximum(x)]
Bdist = mapslices(g, Adist; dims=2)

println(io0(), "Result: $(size(Bdist))")  # (100, 2)

```
