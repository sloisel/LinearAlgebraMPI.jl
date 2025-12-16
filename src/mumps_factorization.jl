"""
MUMPS-based distributed sparse factorization.

Uses MUMPS with distributed matrix input (ICNTL(18)=3) for efficient
parallel direct solve of sparse linear systems.
"""

using MPI
using SparseArrays
using MUMPS
using MUMPS: Mumps, set_icntl!, MUMPS_INT, MUMPS_INT8, suppress_printing!
import MUMPS: invoke_mumps_unsafe!

# ============================================================================
# MUMPS Automatic Finalization Management
# ============================================================================
#
# MUMPS cleanup requires synchronized MPI calls across all ranks, but Julia's
# GC runs asynchronously on each rank. This system handles automatic cleanup:
#
# 1. Each MUMPS factorization gets a unique integer ID (_mumps_count)
# 2. Objects are registered in _mumps_registry by ID
# 3. Julia's GC finalizer queues the ID to _destroy_list (no MPI calls)
# 4. When creating a new factorization, _process_finalizers() is called:
#    - All ranks broadcast their _destroy_list
#    - Lists are merged, sorted, uniqued
#    - Objects are finalized in deterministic order across all ranks
#
# This ensures synchronized cleanup without blocking in finalizers.

# Global counter for unique MUMPS object IDs
const _mumps_count = Ref{Int}(0)

# Registry mapping ID -> MUMPSFactorizationMPI (prevents GC until removed from registry)
const _mumps_registry = Dict{Int, Any}()

# List of MUMPS IDs queued for destruction by this rank's GC
const _destroy_list = Int[]

# Lock for thread-safe access to _destroy_list (finalizers may run from GC thread)
const _destroy_list_lock = ReentrantLock()

# ============================================================================
# MUMPS Factorization Type
# ============================================================================

"""
    MUMPSFactorizationMPI{T}

Distributed MUMPS factorization result. Can be reused for multiple solves.

Factorization objects are automatically cleaned up when garbage collected,
with synchronized finalization across MPI ranks. Manual `finalize!(F)` is
still available for explicit control (must be called on all ranks together).
"""
mutable struct MUMPSFactorizationMPI{T}
    id::Int  # Unique ID for finalization tracking
    mumps::Any  # Mumps{T,R} where R is the real type (Float64 for both real and complex)
    irn_loc::Vector{MUMPS_INT}
    jcn_loc::Vector{MUMPS_INT}
    a_loc::Vector{T}
    n::Int
    symmetric::Bool
    row_partition::Vector{Int}
    rhs_buffer::Vector{T}
end

Base.size(F::MUMPSFactorizationMPI) = (F.n, F.n)
Base.eltype(::MUMPSFactorizationMPI{T}) where T = T

# ============================================================================
# Automatic Finalization Functions
# ============================================================================

"""
    _queue_for_destruction(F::MUMPSFactorizationMPI)

Julia finalizer callback. Queues the factorization ID for later synchronized
destruction. Does NOT call MPI (unsafe from GC thread).
"""
function _queue_for_destruction(F::MUMPSFactorizationMPI)
    lock(_destroy_list_lock) do
        push!(_destroy_list, F.id)
    end
    return nothing
end

"""
    _process_finalizers()

Process pending MUMPS finalizations in a synchronized manner across all ranks.
This is a **collective operation** - all ranks must call it together.

Called automatically when creating new factorizations. Gathers pending
destruction requests from all ranks, merges them, and finalizes in
deterministic order.
"""
function _process_finalizers()
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)

    # Thread-safe: detach current destroy list, replace with empty
    local_list = lock(_destroy_list_lock) do
        list = copy(_destroy_list)
        empty!(_destroy_list)
        list
    end

    # Allgather counts of how many IDs each rank has
    local_count = Int32(length(local_list))
    all_counts = MPI.Allgather(local_count, comm)

    # Allgatherv to collect all IDs from all ranks
    total_count = sum(all_counts)
    if total_count == 0
        return  # Nothing to finalize
    end

    all_ids = Vector{Int}(undef, total_count)
    MPI.Allgatherv!(local_list, MPI.VBuffer(all_ids, all_counts), comm)

    # Sort and unique to get deterministic order across all ranks
    dead_list = sort!(unique(all_ids))

    # Finalize each in order (check registry to avoid double-finalize)
    for id in dead_list
        if haskey(_mumps_registry, id)
            F = _mumps_registry[id]
            delete!(_mumps_registry, id)
            # Actually finalize the MUMPS object
            F.mumps._finalized = false
            MUMPS.finalize!(F.mumps)
        end
    end
end

# ============================================================================
# Extract COO from SparseMatrixMPI
# ============================================================================

"""
    extract_local_coo(A::SparseMatrixMPI{T}; symmetric::Bool=false)

Extract local COO entries from a distributed sparse matrix for MUMPS input.
Returns (irn_loc, jcn_loc, a_loc) with 1-based global indices.

For symmetric matrices, only lower triangular entries (row >= col) are extracted.
"""
function extract_local_coo(A::SparseMatrixMPI{T}; symmetric::Bool=false) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    row_start = A.row_partition[rank + 1]

    irn_loc = MUMPS_INT[]
    jcn_loc = MUMPS_INT[]
    a_loc = T[]

    # A.A.parent is the underlying CSC storage
    # Columns in A.A.parent correspond to local rows
    # A.col_indices maps local column indices to global
    AT = A.A.parent

    for local_row in 1:AT.n  # AT.n = number of local rows
        global_row = row_start + local_row - 1

        for ptr in AT.colptr[local_row]:(AT.colptr[local_row + 1] - 1)
            local_col_idx = AT.rowval[ptr]
            global_col = A.col_indices[local_col_idx]
            val = AT.nzval[ptr]

            # For symmetric matrices, only include lower triangular (row >= col)
            if !symmetric || global_row >= global_col
                push!(irn_loc, MUMPS_INT(global_row))
                push!(jcn_loc, MUMPS_INT(global_col))
                push!(a_loc, val)
            end
        end
    end

    return irn_loc, jcn_loc, a_loc
end

# ============================================================================
# Create MUMPS Factorization
# ============================================================================

"""
    _create_mumps_factorization(A::SparseMatrixMPI{T}, symmetric::Bool) where T

Create and compute a MUMPS factorization of the distributed matrix A.
"""
function _create_mumps_factorization(A::SparseMatrixMPI{T}, symmetric::Bool) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Process any pending finalizations first (collective operation)
    _process_finalizers()

    # Assign unique ID for this factorization
    id = _mumps_count[]
    _mumps_count[] += 1

    m, n = size(A)
    @assert m == n "Matrix must be square for factorization"

    # Extract local COO entries
    irn_loc, jcn_loc, a_loc = extract_local_coo(A; symmetric=symmetric)
    nz_loc = length(a_loc)

    # Create MUMPS instance
    # sym=0: unsymmetric, sym=1: SPD, sym=2: general symmetric
    mumps_sym = symmetric ? MUMPS.mumps_definite : MUMPS.mumps_unsymmetric
    mumps = Mumps{T}(mumps_sym, MUMPS.default_icntl, MUMPS.default_cntl64)
    mumps._finalized = true  # Disable MUMPS GC finalizer to avoid post-MPI crash

    # Suppress all MUMPS output
    suppress_printing!(mumps)

    # Configure MUMPS for distributed input (displaylevel=0 suppresses verbose messages)
    set_icntl!(mumps, 5, 0; displaylevel=0)    # Assembled matrix format
    set_icntl!(mumps, 14, 50; displaylevel=0)  # Memory relaxation (50%)
    set_icntl!(mumps, 18, 3; displaylevel=0)   # Distributed matrix input
    set_icntl!(mumps, 20, 0; displaylevel=0)   # Dense RHS
    set_icntl!(mumps, 21, 0; displaylevel=0)   # Centralized solution on host

    # Set matrix dimension
    mumps.n = MUMPS_INT(n)

    # Set distributed matrix data
    mumps.nz_loc = MUMPS_INT(nz_loc)
    mumps.nnz_loc = MUMPS_INT8(nz_loc)
    mumps.irn_loc = pointer(irn_loc)
    mumps.jcn_loc = pointer(jcn_loc)
    mumps.a_loc = pointer(a_loc)

    # Analysis phase (job = 1)
    mumps.job = MUMPS_INT(1)
    invoke_mumps_unsafe!(mumps)
    _check_mumps_error(mumps, "analysis")

    # Factorization phase (job = 2)
    mumps.job = MUMPS_INT(2)
    invoke_mumps_unsafe!(mumps)
    _check_mumps_error(mumps, "factorization")

    # Pre-allocate RHS buffer on rank 0
    rhs_buffer = rank == 0 ? zeros(T, n) : T[]

    # Create factorization object with ID
    F = MUMPSFactorizationMPI{T}(
        id, mumps, irn_loc, jcn_loc, a_loc,
        n, symmetric, copy(A.row_partition), rhs_buffer
    )

    # Register in global registry (prevents GC until removed)
    _mumps_registry[id] = F

    # Attach Julia finalizer to queue for synchronized destruction
    finalizer(_queue_for_destruction, F)

    return F
end

"""
    _check_mumps_error(mumps::Mumps, phase::String)

Check for MUMPS errors and throw descriptive exception if found.
"""
function _check_mumps_error(mumps::Mumps, phase::String)
    if mumps.infog[1] < 0
        error("MUMPS $phase error: INFOG(1) = $(mumps.infog[1]), INFOG(2) = $(mumps.infog[2])")
    end
end

# ============================================================================
# LinearAlgebra Interface
# ============================================================================

"""
    LinearAlgebra.lu(A::SparseMatrixMPI{T}) where T

Compute LU factorization of a distributed sparse matrix using MUMPS.
Returns a `MUMPSFactorizationMPI` for use with `\\` or `solve`.
Factorization is automatically cleaned up when garbage collected.
"""
function LinearAlgebra.lu(A::SparseMatrixMPI{T}) where T
    return _create_mumps_factorization(A, false)
end

"""
    LinearAlgebra.ldlt(A::SparseMatrixMPI{T}) where T

Compute LDLT factorization of a distributed symmetric sparse matrix using MUMPS.
The matrix must be symmetric; only the lower triangular part is used.
Returns a `MUMPSFactorizationMPI` for use with `\\` or `solve`.
Factorization is automatically cleaned up when garbage collected.
"""
function LinearAlgebra.ldlt(A::SparseMatrixMPI{T}) where T
    return _create_mumps_factorization(A, true)
end

# ============================================================================
# Solve Interface
# ============================================================================

"""
    solve(F::MUMPSFactorizationMPI{T}, b::VectorMPI{T}) where T

Solve the linear system A*x = b using the precomputed MUMPS factorization.
"""
function solve(F::MUMPSFactorizationMPI{T}, b::VectorMPI{T}) where T
    x = VectorMPI(zeros(T, F.n); partition=b.partition)
    solve!(x, F, b)
    return x
end

"""
    solve!(x::VectorMPI{T}, F::MUMPSFactorizationMPI{T}, b::VectorMPI{T}) where T

Solve A*x = b in-place using MUMPS factorization.
"""
function solve!(x::VectorMPI{T}, F::MUMPSFactorizationMPI{T}, b::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Gather RHS to rank 0
    counts = Int32[b.partition[r+2] - b.partition[r+1] for r in 0:nranks-1]

    if rank == 0
        MPI.Gatherv!(b.v, MPI.VBuffer(F.rhs_buffer, counts), comm; root=0)

        # Set RHS in MUMPS
        F.mumps.nrhs = MUMPS_INT(1)
        F.mumps.lrhs = MUMPS_INT(F.n)
        F.mumps.rhs = pointer(F.rhs_buffer)
    else
        MPI.Gatherv!(b.v, nothing, comm; root=0)
    end

    # Solve phase (job = 3)
    F.mumps.job = MUMPS_INT(3)
    invoke_mumps_unsafe!(F.mumps)
    _check_mumps_error(F.mumps, "solve")

    # Scatter solution from rank 0
    # Result is in F.rhs_buffer on rank 0
    if rank == 0
        MPI.Scatterv!(MPI.VBuffer(F.rhs_buffer, counts), x.v, comm; root=0)
    else
        MPI.Scatterv!(nothing, x.v, comm; root=0)
    end

    return x
end

"""
    Base.:\\(F::MUMPSFactorizationMPI{T}, b::VectorMPI{T}) where T

Solve A*x = b using the backslash operator.
"""
function Base.:\(F::MUMPSFactorizationMPI{T}, b::VectorMPI{T}) where T
    return solve(F, b)
end

# ============================================================================
# Cleanup Interface
# ============================================================================

"""
    finalize!(F::MUMPSFactorizationMPI)

Manually release MUMPS resources. This is a **collective operation** - all
ranks must call it together for immediate cleanup.

If the factorization has already been cleaned up (by automatic finalization
or a previous manual call), this is a no-op but all ranks must still call it.
"""
function finalize!(F::MUMPSFactorizationMPI)
    # Check if already finalized (removed from registry)
    if !haskey(_mumps_registry, F.id)
        return F  # Already finalized, no-op
    end

    # Remove from registry
    delete!(_mumps_registry, F.id)

    # Actually finalize the MUMPS object
    F.mumps._finalized = false  # Re-enable MUMPS finalization
    MUMPS.finalize!(F.mumps)
    return F
end

