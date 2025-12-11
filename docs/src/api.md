# API Reference

This page documents the public API of LinearAlgebraMPI.jl.

## Types

### SparseMatrixMPI

```@docs
SparseMatrixMPI
```

### MatrixMPI

```@docs
MatrixMPI
```

### VectorMPI

```@docs
VectorMPI
```

## Sparse Matrix Operations

### Arithmetic

```julia
A * B          # Matrix multiplication
A + B          # Addition
A - B          # Subtraction
a * A          # Scalar multiplication
A * a          # Scalar multiplication
```

### Transpose and Adjoint

```julia
transpose(A)   # Lazy transpose
conj(A)        # Conjugate (new matrix)
A'             # Adjoint (conjugate transpose, lazy)
```

### Matrix-Vector Multiplication

```julia
y = A * x      # Returns VectorMPI
mul!(y, A, x)  # In-place version
```

### Vector-Matrix Multiplication

```julia
transpose(v) * A   # Row vector times matrix
v' * A             # Conjugate row vector times matrix
```

### Norms

```julia
norm(A)        # Frobenius norm (default)
norm(A, 1)     # Sum of absolute values
norm(A, Inf)   # Maximum absolute value
norm(A, p)     # General p-norm

opnorm(A, 1)   # Maximum absolute column sum
opnorm(A, Inf) # Maximum absolute row sum
```

### Properties

```julia
size(A)        # Global dimensions (m, n)
size(A, d)     # Size along dimension d
eltype(A)      # Element type
nnz(A)         # Number of nonzeros
issparse(A)    # Returns true
```

### Reductions

```julia
sum(A)         # Sum of all elements
sum(A; dims=1) # Column sums (returns VectorMPI)
sum(A; dims=2) # Row sums (returns VectorMPI)
maximum(A)     # Maximum of stored values
minimum(A)     # Minimum of stored values
tr(A)          # Trace (sum of diagonal)
```

```@docs
mean
```

### Element-wise Operations

```julia
abs(A)         # Absolute value
abs2(A)        # Squared absolute value
real(A)        # Real part
imag(A)        # Imaginary part
floor(A)       # Floor
ceil(A)        # Ceiling
round(A)       # Round
```

### Utilities

```julia
copy(A)        # Deep copy
dropzeros(A)   # Remove stored zeros
diag(A)        # Main diagonal (returns VectorMPI)
diag(A, k)     # k-th diagonal
triu(A)        # Upper triangular
triu(A, k)     # Upper triangular from k-th diagonal
tril(A)        # Lower triangular
tril(A, k)     # Lower triangular from k-th diagonal
```

### Block Operations

```julia
cat(A, B, C; dims=1)       # Vertical concatenation
cat(A, B, C; dims=2)       # Horizontal concatenation
cat(A, B, C, D; dims=(2,2)) # 2x2 block matrix [A B; C D]
vcat(A, B, C)              # Vertical concatenation
hcat(A, B, C)              # Horizontal concatenation
blockdiag(A, B, C)         # Block diagonal matrix
```

### Diagonal Matrix Construction

```julia
spdiagm(v)                 # Diagonal matrix from VectorMPI
spdiagm(m, n, v)           # m x n diagonal matrix
spdiagm(k => v)            # k-th diagonal
spdiagm(0 => v, 1 => w)    # Multiple diagonals
```

## Dense Matrix Operations

### Arithmetic

```julia
A * x          # Matrix-vector multiplication (returns VectorMPI)
transpose(A)   # Lazy transpose
conj(A)        # Conjugate
A'             # Adjoint
a * A          # Scalar multiplication
```

### mapslices

Apply a function to rows or columns of a distributed dense matrix.

```julia
mapslices(f, A; dims=2)   # Apply f to each row (local, no MPI)
mapslices(f, A; dims=1)   # Apply f to each column (requires MPI)
```

**Example:**
```julia
using LinearAlgebra

A = MatrixMPI(randn(100, 10))

# Compute row statistics: norm, max, sum for each row
# Transforms 100x10 to 100x3
B = mapslices(x -> [norm(x), maximum(x), sum(x)], A; dims=2)
```

### Block Operations

```julia
cat(A, B; dims=1)          # Vertical concatenation
cat(A, B; dims=2)          # Horizontal concatenation
vcat(A, B)                 # Vertical concatenation
hcat(A, B)                 # Horizontal concatenation
```

### Norms

```julia
norm(A)        # Frobenius norm
norm(A, p)     # General p-norm
opnorm(A, 1)   # Maximum absolute column sum
opnorm(A, Inf) # Maximum absolute row sum
```

## Vector Operations

### Arithmetic

```julia
u + v          # Addition (auto-aligns partitions)
u - v          # Subtraction
-v             # Negation
a * v          # Scalar multiplication
v * a          # Scalar multiplication
v / a          # Scalar division
```

### Transpose and Adjoint

```julia
transpose(v)   # Lazy transpose (row vector)
conj(v)        # Conjugate
v'             # Adjoint
```

### Norms

```julia
norm(v)        # 2-norm (default)
norm(v, 1)     # 1-norm
norm(v, Inf)   # Infinity norm
norm(v, p)     # General p-norm
```

### Reductions

```julia
sum(v)         # Sum of elements
prod(v)        # Product of elements
maximum(v)     # Maximum element
minimum(v)     # Minimum element
mean(v)        # Mean of elements
```

### Element-wise Operations

```julia
abs(v)         # Absolute value
abs2(v)        # Squared absolute value
real(v)        # Real part
imag(v)        # Imaginary part
copy(v)        # Deep copy
```

### Broadcasting

VectorMPI supports broadcasting for element-wise operations:

```julia
v .+ w         # Element-wise addition
v .* w         # Element-wise multiplication
sin.(v)        # Apply function element-wise
v .* 2.0 .+ w  # Compound expressions
```

### Block Operations

```julia
vcat(u, v, w)  # Concatenate vectors (returns VectorMPI)
hcat(u, v, w)  # Stack as columns (returns MatrixMPI)
```

### Properties

```julia
length(v)      # Global length
size(v)        # Returns (length,)
eltype(v)      # Element type
```

## Utility Functions

### Rank-Selective Output

```@docs
io0
```

### Gathering Distributed Data

Convert distributed MPI types to standard Julia types (gathers data to all ranks):

```julia
Vector(v::VectorMPI)              # Gather to Vector
Matrix(A::MatrixMPI)              # Gather to Matrix
SparseMatrixCSC(A::SparseMatrixMPI) # Gather to SparseMatrixCSC
```

These conversions enable `show` and string interpolation:

```julia
println(io0(), "Result: $v")      # Works with VectorMPI
println(io0(), "Matrix: $A")      # Works with MatrixMPI/SparseMatrixMPI
```

## Cache Management

```@docs
clear_plan_cache!
```

## Full API Index

```@index
```
