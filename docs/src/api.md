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
sum(A)         # Sum of all stored elements
sum(A; dims=1) # Column sums (returns VectorMPI) - SparseMatrixMPI only
sum(A; dims=2) # Row sums (returns VectorMPI) - SparseMatrixMPI only
maximum(A)     # Maximum of stored values
minimum(A)     # Minimum of stored values
tr(A)          # Trace (sum of diagonal) - SparseMatrixMPI only
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

# Create deterministic test matrix (same on all ranks)
A_global = Float64.([i + 0.1*j for i in 1:100, j in 1:10])
A = MatrixMPI(A_global)

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

## Indexing

All distributed types support element access and assignment. These are collective operations - all MPI ranks must call them with the same arguments.

### VectorMPI Indexing

```julia
v[i]           # Get element (collective)
v[i] = x       # Set element (collective)
v[1:10]        # Range indexing (returns VectorMPI)
v[1:10] = x    # Range assignment (scalar or vector)
v[idx]         # VectorMPI{Int} indexing (returns VectorMPI)
v[idx] = src   # VectorMPI{Int} assignment (src::VectorMPI)
```

### MatrixMPI Indexing

```julia
# Single element
A[i, j]        # Get element
A[i, j] = x    # Set element

# Range indexing (returns MatrixMPI)
A[1:3, 2:5]    # Submatrix by ranges
A[1:3, :]      # Row range, all columns
A[:, 2:5]      # All rows, column range

# VectorMPI indices (returns MatrixMPI)
A[row_idx, col_idx]  # Both indices are VectorMPI{Int}

# Mixed indexing (returns MatrixMPI or VectorMPI)
A[row_idx, 1:5]      # VectorMPI rows, range columns
A[row_idx, :]        # VectorMPI rows, all columns
A[1:5, col_idx]      # Range rows, VectorMPI columns
A[:, col_idx]        # All rows, VectorMPI columns
A[row_idx, j]        # VectorMPI rows, single column (returns VectorMPI)
A[i, col_idx]        # Single row, VectorMPI columns (returns VectorMPI)
```

### SparseMatrixMPI Indexing

```julia
# Single element
A[i, j]        # Get element (returns 0 for structural zeros)
A[i, j] = x    # Set element (modifies structure if needed)

# Range indexing (returns SparseMatrixMPI)
A[1:3, 2:5]    # Submatrix by ranges
A[1:3, :]      # Row range, all columns
A[:, 2:5]      # All rows, column range

# VectorMPI indices (returns SparseMatrixMPI)
A[row_idx, col_idx]  # Both indices are VectorMPI{Int}

# Mixed indexing (returns SparseMatrixMPI or VectorMPI)
A[row_idx, 1:5]      # VectorMPI rows, range columns
A[row_idx, :]        # VectorMPI rows, all columns
A[1:5, col_idx]      # Range rows, VectorMPI columns
A[:, col_idx]        # All rows, VectorMPI columns
A[row_idx, j]        # VectorMPI rows, single column (returns VectorMPI)
A[i, col_idx]        # Single row, VectorMPI columns (returns VectorMPI)
```

### setindex! Source Types

For `setindex!` operations, the source type depends on the indexing pattern:

| Index Pattern | Source Type |
|--------------|-------------|
| `A[i, j] = x` | Scalar |
| `A[range, range] = x` | Scalar, Matrix, or distributed matrix |
| `A[VectorMPI, VectorMPI] = src` | MatrixMPI (matching partitions) |
| `A[VectorMPI, range] = src` | MatrixMPI |
| `A[range, VectorMPI] = src` | MatrixMPI |
| `A[VectorMPI, j] = src` | VectorMPI |
| `A[i, VectorMPI] = src` | VectorMPI |

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

### Local Constructors

Create distributed types from local data (each rank provides only its portion):

```@docs
VectorMPI_local
MatrixMPI_local
SparseMatrixMPI_local
```

## Cache Management

```@docs
clear_plan_cache!
```

## Full API Index

```@index
```
