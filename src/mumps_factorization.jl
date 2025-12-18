"""
MUMPS-based distributed sparse factorization.

Uses MUMPS with distributed matrix input (ICNTL(18)=3) for efficient
parallel direct solve of sparse linear systems.

Analysis caching: The symbolic analysis phase (ordering, symbolic factorization)
depends only on sparsity structure, not numerical values. We cache the analyzed
MUMPS object by structural hash, so subsequent factorizations with the same
structure skip the expensive analysis phase.
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
# MUMPS Analysis Cache
# ============================================================================
#
# The symbolic analysis phase (job=1) depends only on sparsity structure, not
# numerical values. We cache the analyzed MUMPS object by structural hash,
# allowing subsequent factorizations to skip analysis and only do numeric
# factorization (job=2).
#
# The cache stores MUMPSAnalysisPlan objects, which contain:
# - A pre-analyzed MUMPS object (ready for job=2)
# - The COO index arrays (structure is fixed)
# - Metadata for validation

"""
    MUMPSAnalysisPlan{T}

Cached MUMPS symbolic analysis for a given sparsity structure.
Stores a pre-analyzed MUMPS object that can be reused for numeric
factorization with different values but the same structure.
"""
mutable struct MUMPSAnalysisPlan{T}
    mumps::Any  # Mumps{T,R} after analysis (job=1)
    irn_loc::Vector{MUMPS_INT}  # Row indices (structure, immutable)
    jcn_loc::Vector{MUMPS_INT}  # Column indices (structure, immutable)
    a_loc::Vector{T}  # Value array (updated for each factorization)
    n::Int
    symmetric::Bool
    row_partition::Vector{Int}
    structural_hash::NTuple{32,UInt8}
end

# Cache mapping (structural_hash, symmetric, element_type) -> MUMPSAnalysisPlan
const _mumps_analysis_cache = Dict{Tuple{NTuple{32,UInt8}, Bool, DataType}, Any}()

"""
    clear_mumps_analysis_cache!()

Clear the MUMPS analysis cache. This is a collective operation that must
be called on all MPI ranks together.
"""
function clear_mumps_analysis_cache!()
    for (key, plan) in _mumps_analysis_cache
        # Finalize the cached MUMPS objects
        plan.mumps._finalized = false
        MUMPS.finalize!(plan.mumps)
    end
    empty!(_mumps_analysis_cache)
end

# ============================================================================
# MUMPS Factorization Type
# ============================================================================

"""
    MUMPSFactorizationMPI{T}

Distributed MUMPS factorization result. Can be reused for multiple solves.

Note: The MUMPS object is shared with the analysis cache. The factorization
does not own the MUMPS object and should not finalize it directly.
"""
mutable struct MUMPSFactorizationMPI{T}
    id::Int  # Unique ID for finalization tracking
    mumps::Any  # Mumps{T,R} - shared with cache, do not finalize
    irn_loc::Vector{MUMPS_INT}
    jcn_loc::Vector{MUMPS_INT}
    a_loc::Vector{T}
    n::Int
    symmetric::Bool
    row_partition::Vector{Int}
    rhs_buffer::Vector{T}
    owns_mumps::Bool  # Whether this factorization owns the MUMPS object
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
            # Only finalize if we own the MUMPS object (not shared with cache)
            if F.owns_mumps
                F.mumps._finalized = false
                MUMPS.finalize!(F.mumps)
            end
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
    _get_or_create_analysis_plan(A::SparseMatrixMPI{T}, symmetric::Bool) where T

Get a cached analysis plan or create a new one. Returns the plan with
values updated from matrix A.
"""
function _get_or_create_analysis_plan(A::SparseMatrixMPI{T}, symmetric::Bool) where T
    comm = MPI.COMM_WORLD

    # Ensure structural hash is computed
    structural_hash = _ensure_hash(A)
    cache_key = (structural_hash, symmetric, T)

    if haskey(_mumps_analysis_cache, cache_key)
        # Cache hit: reuse existing analysis
        plan = _mumps_analysis_cache[cache_key]::MUMPSAnalysisPlan{T}

        # Update values from matrix A (structure is already correct)
        _update_values!(plan, A, symmetric)

        return plan, true  # true = cache hit
    end

    # Cache miss: create new analysis
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
    set_icntl!(mumps, 7, 5; displaylevel=0)    # METIS ordering (better fill-in)

    # Enable OpenMP threading in MUMPS
    # ICNTL(16) = number of OpenMP threads (0 = use OMP_NUM_THREADS)
    omp_threads = parse(Int, get(ENV, "OMP_NUM_THREADS", "1"))
    set_icntl!(mumps, 16, omp_threads; displaylevel=0)

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

    # Create and cache the analysis plan
    plan = MUMPSAnalysisPlan{T}(
        mumps, irn_loc, jcn_loc, a_loc,
        n, symmetric, copy(A.row_partition), structural_hash
    )
    _mumps_analysis_cache[cache_key] = plan

    return plan, false  # false = cache miss
end

"""
    _update_values!(plan::MUMPSAnalysisPlan{T}, A::SparseMatrixMPI{T}, symmetric::Bool) where T

Update the values in a cached analysis plan from a new matrix with the same structure.
"""
function _update_values!(plan::MUMPSAnalysisPlan{T}, A::SparseMatrixMPI{T}, symmetric::Bool) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    row_start = A.row_partition[rank + 1]
    AT = A.A.parent

    # Update values in-place (structure must match exactly)
    idx = 1
    for local_row in 1:AT.n
        global_row = row_start + local_row - 1
        for ptr in AT.colptr[local_row]:(AT.colptr[local_row + 1] - 1)
            local_col_idx = AT.rowval[ptr]
            global_col = A.col_indices[local_col_idx]

            if !symmetric || global_row >= global_col
                plan.a_loc[idx] = AT.nzval[ptr]
                idx += 1
            end
        end
    end
end

"""
    _create_mumps_factorization(A::SparseMatrixMPI{T}, symmetric::Bool) where T

Create and compute a MUMPS factorization of the distributed matrix A.
Uses cached symbolic analysis when available for the same sparsity structure.
"""
function _create_mumps_factorization(A::SparseMatrixMPI{T}, symmetric::Bool) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Process any pending finalizations first (collective operation)
    _process_finalizers()

    # Assign unique ID for this factorization
    id = _mumps_count[]
    _mumps_count[] += 1

    # Get or create analysis plan (may be cached)
    plan, cache_hit = _get_or_create_analysis_plan(A, symmetric)

    # Update value pointer (values may have been updated)
    plan.mumps.a_loc = pointer(plan.a_loc)

    # Factorization phase (job = 2)
    plan.mumps.job = MUMPS_INT(2)
    invoke_mumps_unsafe!(plan.mumps)
    _check_mumps_error(plan.mumps, "factorization")

    # Pre-allocate RHS buffer on rank 0
    rhs_buffer = rank == 0 ? zeros(T, plan.n) : T[]

    # Create factorization object with ID
    # Note: We copy the value array since the plan's array is reused.
    # The MUMPS object is shared with the cache (owns_mumps=false).
    F = MUMPSFactorizationMPI{T}(
        id, plan.mumps, plan.irn_loc, plan.jcn_loc, copy(plan.a_loc),
        plan.n, symmetric, copy(plan.row_partition), rhs_buffer, false
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
"""
function LinearAlgebra.lu(A::SparseMatrixMPI{T}) where T
    return _create_mumps_factorization(A, false)
end

"""
    LinearAlgebra.ldlt(A::SparseMatrixMPI{T}) where T

Compute LDLT factorization of a distributed symmetric sparse matrix using MUMPS.
The matrix must be symmetric; only the lower triangular part is used.
Returns a `MUMPSFactorizationMPI` for use with `\\` or `solve`.
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

Release MUMPS resources. Must be called on all ranks together.

Note: If the MUMPS object is shared with the analysis cache (owns_mumps=false),
this only removes the factorization from the registry. The MUMPS object itself
is finalized when `clear_mumps_analysis_cache!()` is called.
"""
function finalize!(F::MUMPSFactorizationMPI)
    # Check if already finalized (removed from registry)
    if !haskey(_mumps_registry, F.id)
        return F  # Already finalized, no-op
    end

    # Remove from registry
    delete!(_mumps_registry, F.id)

    # Only finalize the MUMPS object if we own it (not shared with cache)
    if F.owns_mumps
        F.mumps._finalized = false  # Re-enable MUMPS finalization
        MUMPS.finalize!(F.mumps)
    end

    return F
end

