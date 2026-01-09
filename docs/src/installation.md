# Installation

## Prerequisites

### MPI

LinearAlgebraMPI.jl requires an MPI implementation. When you install the package, Julia automatically provides `MPI.jl` with `MPI_jll` (bundled MPI implementation).

For HPC environments, you may want to configure MPI.jl to use your system's MPI installation. See the [MPI.jl documentation](https://juliaparallel.org/MPI.jl/stable/configuration/) for details.

### MUMPS

The package uses MUMPS for sparse direct solves. MUMPS is typically available through your system's package manager or HPC module system.

## Package Installation

### From GitHub

```julia
using Pkg
Pkg.add(url="https://github.com/sloisel/LinearAlgebraMPI.jl")
```

### Development Installation

```bash
git clone https://github.com/sloisel/LinearAlgebraMPI.jl
cd LinearAlgebraMPI.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Verification

Test your installation with MPI:

```bash
cd LinearAlgebraMPI.jl
mpiexec -n 2 julia --project test/runtests.jl
```

## Initialization Pattern

!!! tip "Initialization Pattern"
    Initialize MPI first, then load the package:

```julia
# CORRECT
using MPI
using LinearAlgebraMPI
MPI.Init()
# Now you can use the package
```

## Running MPI Programs

Create a script file (e.g., `my_program.jl`):

```julia
using MPI
using LinearAlgebraMPI
MPI.Init()
using SparseArrays

# Create distributed matrix
A = SparseMatrixMPI{Float64}(sprandn(100, 100, 0.1) + 10I)
b = VectorMPI(randn(100))

# Solve
x = A \ b

println(io0(), "Solution computed!")
```

Run with MPI:

```bash
mpiexec -n 4 julia --project my_program.jl
```

## Troubleshooting

### MPI Issues

If you see MPI-related errors, try rebuilding MPI.jl:

```julia
using Pkg; Pkg.build("MPI")
```

### MUMPS Issues

If MUMPS fails to load, ensure it's properly installed on your system.

## Next Steps

Once installed, proceed to the [User Guide](@ref) to learn how to use the package.
