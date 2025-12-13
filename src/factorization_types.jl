"""
Core data structures for distributed multifrontal LU and LDLT factorization.
"""

using SparseArrays

# ============================================================================
# Basic Types (ported from dev/src/types.jl)
# ============================================================================

"""
    Supernode

Represents a supernode in the elimination tree: a set of contiguous columns
with nearly identical sparsity structure that can be factored together.
"""
struct Supernode
    cols::UnitRange{Int}     # Column indices in this supernode
end

Base.length(s::Supernode) = length(s.cols)
Base.first(s::Supernode) = first(s.cols)
Base.last(s::Supernode) = last(s.cols)

"""
    FrontalInfo

Information about a frontal matrix computed during symbolic factorization.
"""
struct FrontalInfo
    snode_idx::Int               # Supernode index
    row_indices::Vector{Int}     # All row indices in the frontal matrix
    nfs::Int                     # Number of fully summed rows (= supernode size)
end

"""
    FrontalMatrix{T}

Dense storage for a frontal matrix during numerical factorization.

Structure:
    [L11  0 ] [U11  U12]
    [L21  I ] [0    S22]

where:
- L11, U11 are the factors for fully summed columns
- L21, U12 are the factors below/right of the diagonal block
- S22 is the Schur complement (update matrix) to be passed to parent
"""
mutable struct FrontalMatrix{T}
    F::Matrix{T}                # Dense frontal matrix storage
    row_indices::Vector{Int}    # Global row indices (may be swapped during pivoting)
    col_indices::Vector{Int}    # Global column indices (NOT swapped during pivoting)
    nfs::Int                    # Number of fully summed rows/cols
    pivots::Vector{Int}         # Row pivot indices (for partial pivoting)
end

function FrontalMatrix{T}(row_indices::Vector{Int}, col_indices::Vector{Int}, nfs::Int) where T
    nrows = length(row_indices)
    ncols = length(col_indices)
    F = zeros(T, nrows, ncols)
    pivots = collect(1:nfs)  # Initialize as identity
    # col_indices should be a COPY that won't be modified during pivoting
    return FrontalMatrix{T}(F, copy(row_indices), copy(col_indices), nfs, pivots)
end

# ============================================================================
# Symbolic Factorization (MPI-specific)
# ============================================================================

"""
    SymbolicFactorization

Result of symbolic factorization phase. Computed once per sparsity pattern and
cached for reuse. Contains the elimination tree structure and MPI rank assignments.

The symbolic factorization is identical across all MPI ranks.
"""
struct SymbolicFactorization
    perm::Vector{Int}                    # Fill-reducing permutation (AMD)
    invperm::Vector{Int}                 # Inverse fill-reducing permutation
    supernodes::Vector{Supernode}        # Supernode definitions
    snode_parent::Vector{Int}            # Supernodal elimination tree parent pointers
    snode_children::Vector{Vector{Int}}  # Children lists for each supernode
    snode_postorder::Vector{Int}         # Postorder traversal of supernodal tree
    frontal_info::Vector{FrontalInfo}    # Row structure for each frontal matrix
    snode_owner::Vector{Int}             # Primary owning rank for each supernode (0-indexed)
    col_to_snode::Vector{Int}            # Column to supernode mapping
    elim_to_global::Vector{Int}          # Elimination order: elim_to_global[k] = global col at step k
    global_to_elim::Vector{Int}          # Inverse: global_to_elim[col] = elimination step
    structural_hash::Blake3Hash          # For caching
    n::Int                               # Matrix dimension
end

"""
    get_children(sidx::Int, symbolic::SymbolicFactorization) -> Vector{Int}

Get the children of supernode `sidx` in the elimination tree.
"""
function get_children(sidx::Int, symbolic::SymbolicFactorization)
    return symbolic.snode_children[sidx]
end

# ============================================================================
# Distributed LU Factorization
# ============================================================================

"""
    LUFactorizationMPI{T}

Distributed LU factorization result.

The factorization satisfies: P_row * Ap = L * U
where:
- Ap is A symmetrically permuted by perm (the fill-reducing permutation)
- P_row is the row permutation from partial pivoting
- L is unit lower triangular (in elimination order)
- U is upper triangular (in elimination order)

Each MPI rank stores the L and U columns/rows for supernodes it owns.
L and U are stored with GLOBAL indices (1 to n referencing Ap rows/cols).

## Solve sequence
1. Apply fill-reducing perm: work = b[perm]
2. Apply row pivot perm: work2[k] = work[row_perm[k]]
3. Forward solve (in elim order): process columns in elim_to_global order
4. Backward solve (in reverse elim order): process columns in reverse elim_to_global order
5. Apply inverse row pivot perm
6. Apply inverse fill-reducing perm
"""
struct LUFactorizationMPI{T}
    symbolic::SymbolicFactorization
    L_local::SparseMatrixCSC{T,Int}      # Local L columns (global indices)
    U_local::SparseMatrixCSC{T,Int}      # Local U rows (global indices)
    row_perm::Vector{Int}                # Global row permutation from partial pivoting
    inv_row_perm::Vector{Int}            # Inverse row permutation
end

Base.size(lu::LUFactorizationMPI) = (lu.symbolic.n, lu.symbolic.n)
Base.eltype(::LUFactorizationMPI{T}) where T = T

# ============================================================================
# Distributed LDLT Factorization
# ============================================================================

"""
    LDLTFactorizationMPI{T}

Distributed LDLT factorization result for symmetric matrices.

The factorization satisfies: Ap = P' * L * D * Lᵀ * P
where:
- Ap is A symmetrically permuted by perm (the fill-reducing permutation)
- P is the symmetric permutation from Bunch-Kaufman pivoting
- L is unit lower triangular (in elimination order)
- D is block diagonal (1×1 and 2×2 blocks for indefinite matrices)
- Lᵀ is the transpose (NOT adjoint) of L

## Complex number handling

Uses transpose (Lᵀ), not adjoint (L*). This is correct for:
- Real symmetric matrices (A = Aᵀ)
- Complex symmetric matrices (A = Aᵀ, rare but used in some physics applications)

NOT correct for complex Hermitian matrices (A = A*). Use LU factorization instead.

## Pivoting information

- pivots[k] > 0: 1×1 pivot at elimination step k
- pivots[k] < 0: 2×2 pivot starting at elimination step k (paired with k+1)

## Solve sequence
1. Apply fill-reducing perm: work = b[perm]
2. Apply symmetric pivot perm: work2 = work[sym_perm]
3. Forward solve with L
4. Diagonal solve with D (handling 2×2 blocks)
5. Backward solve with Lᵀ
6. Apply inverse symmetric perm
7. Apply inverse fill-reducing perm
"""
struct LDLTFactorizationMPI{T}
    symbolic::SymbolicFactorization
    L_local::SparseMatrixCSC{T,Int}      # Local L columns (global indices)
    D_local::SparseMatrixCSC{T,Int}      # Local D diagonal blocks
    pivots::Vector{Int}                  # Pivot type info: >0 for 1×1, <0 for 2×2 start
    sym_perm::Vector{Int}                # Global symmetric permutation from Bunch-Kaufman
    inv_sym_perm::Vector{Int}            # Inverse symmetric permutation
end

Base.size(ldlt::LDLTFactorizationMPI) = (ldlt.symbolic.n, ldlt.symbolic.n)
Base.eltype(::LDLTFactorizationMPI{T}) where T = T

# ============================================================================
# Distributed Solve Plan
# ============================================================================

"""
    SolvePlan{T}

Communication plan for distributed triangular solves.

With subtree-to-rank assignment, cross-rank communication only occurs at
subtree boundaries. This plan precomputes:
1. Which supernodes send/receive across rank boundaries
2. Buffer sizes and pre-allocated storage
3. Index mappings for RHS/solution vectors

## Forward solve (L y = b)

Process supernodes in postorder (leaves → root):
- Receive update contributions from children on other ranks
- Perform local forward substitution
- Send update contribution to parent on another rank

## Backward solve (U x = y or L^T x = y)

Process supernodes in reverse postorder (root → leaves):
- Receive solution from parent on another rank
- Perform local backward substitution
- Send solution components to children on other ranks

## Communication pattern

For each supernode s where parent is on a different rank:
- Forward: send update contribution (size = number of update rows in parent)
- Backward: receive solution (size = number of update rows shared with parent)
"""
mutable struct SolvePlan{T}
    # Reference to symbolic factorization
    symbolic::SymbolicFactorization

    # MPI info
    myrank::Int
    nranks::Int

    # Supernodes that communicate across rank boundaries
    # (supernodes whose parent is on a different rank = subtree roots)
    subtree_roots::Vector{Int}

    # For each subtree root: the update rows that need to be communicated
    # subtree_root_update_rows[sidx] = row indices (global) that contribute to parent
    subtree_root_update_rows::Dict{Int, Vector{Int}}

    # Forward solve communication (send to parent, receive from children)
    forward_send_to::Dict{Int, Int}           # sidx -> rank to send to
    forward_recv_from::Dict{Int, Int}         # sidx -> rank to receive from

    # Backward solve communication (receive from parent, send to children)
    backward_send_to::Dict{Int, Vector{Int}}  # sidx -> ranks to send to
    backward_recv_from::Dict{Int, Int}        # sidx -> rank to receive from

    # Pre-allocated communication buffers
    send_buffers::Dict{Tuple{Int,Int}, Vector{T}}  # (sidx, dest_rank) -> buffer
    recv_buffers::Dict{Tuple{Int,Int}, Vector{T}}  # (sidx, src_rank) -> buffer

    # Local vector storage: maps global indices to local positions
    # For the solve, we work in elimination order
    local_snode_indices::Dict{Int, UnitRange{Int}}  # sidx -> range in local work vector

    # Vector of supernodes owned by this rank (in postorder)
    my_supernodes_postorder::Vector{Int}

    # Work vector for intermediate values (sized to number of columns this rank owns)
    work_vector::Vector{T}

    # Mapping: global elimination index -> local work index (or 0 if not local)
    global_to_local::Vector{Int}

    # Mapping: local work index -> global elimination index
    local_to_global::Vector{Int}

    # Flag indicating if plan has been initialized
    initialized::Bool
end

"""
    SolvePlan{T}(symbolic::SymbolicFactorization) where T

Create an uninitialized solve plan. Call `initialize_solve_plan!` to set it up.
"""
function SolvePlan{T}(symbolic::SymbolicFactorization) where T
    return SolvePlan{T}(
        symbolic,
        0, 0,  # myrank, nranks - set during init
        Int[],
        Dict{Int, Vector{Int}}(),
        Dict{Int, Int}(),
        Dict{Int, Int}(),
        Dict{Int, Vector{Int}}(),
        Dict{Int, Int}(),
        Dict{Tuple{Int,Int}, Vector{T}}(),
        Dict{Tuple{Int,Int}, Vector{T}}(),
        Dict{Int, UnitRange{Int}}(),
        Int[],
        T[],
        Int[],
        Int[],
        false
    )
end

# ============================================================================
# Cache for Symbolic Factorizations
# ============================================================================

"""
Global cache for symbolic factorizations, keyed by structural hash.
"""
const _symbolic_cache = Dict{Blake3Hash, SymbolicFactorization}()

"""
    clear_symbolic_cache!()

Clear the symbolic factorization cache.
"""
function clear_symbolic_cache!()
    empty!(_symbolic_cache)
end

# ============================================================================
# Distributed Factorization Input Plan
# ============================================================================

"""
    FactorizationInputPlan{T}

Communication plan for distributing matrix entries to ranks that need them
for frontal matrix initialization.

With distributed input, each rank only stores its portion of the input matrix A.
Before factorization, ranks exchange entries needed for their supernodes' frontal
matrices via point-to-point communication.

## Communication pattern

For each supernode this rank owns:
1. Determine which entries of A[perm, perm] are needed (from symbolic factorization)
2. Map back to original A indices: need A[perm[i], perm[j]]
3. Request entries from ranks that own those rows
4. Receive values and store for frontal matrix initialization

## Fields

- `symbolic`: Reference to symbolic factorization
- `myrank`, `nranks`: MPI info
- `needed_entries`: Dict mapping supernode idx -> list of (perm_i, perm_j) needed
- `send_to_ranks`: Which ranks to send entries to
- `recv_from_ranks`: Which ranks to receive entries from
- `send_indices`: For each destination rank -> list of (row, col) pairs to send
- `recv_indices`: For each source rank -> list of (perm_i, perm_j) pairs to receive
- `send_buffers`, `recv_buffers`: Pre-allocated communication buffers
- `initialized`: Flag indicating if plan has been set up
"""
mutable struct FactorizationInputPlan{T}
    # Reference to symbolic factorization
    symbolic::SymbolicFactorization

    # MPI info
    myrank::Int
    nranks::Int

    # Row partition of input matrix A
    row_partition::Vector{Int}

    # For each supernode this rank owns: entries needed (in permuted coordinates)
    # needed_entries[sidx] = [(perm_row, perm_col), ...]
    needed_entries::Dict{Int, Vector{Tuple{Int,Int}}}

    # Ranks we send entries to (these ranks own supernodes needing our A rows)
    send_to_ranks::Vector{Int}

    # Ranks we receive entries from (these ranks own A rows we need)
    recv_from_ranks::Vector{Int}

    # For each destination rank: which (orig_row, orig_col) pairs to send
    # send_indices[dest_rank] = [(row, col), ...]
    send_indices::Dict{Int, Vector{Tuple{Int,Int}}}

    # For each source rank: which (perm_row, perm_col) pairs we receive
    # recv_indices[src_rank] = [(perm_i, perm_j), ...]
    recv_indices::Dict{Int, Vector{Tuple{Int,Int}}}

    # Pre-allocated send/receive buffers
    send_buffers::Dict{Int, Vector{T}}
    recv_buffers::Dict{Int, Vector{T}}

    # Received values storage: maps (perm_i, perm_j) -> value
    received_values::Dict{Tuple{Int,Int}, T}

    # Flag indicating if plan has been initialized
    initialized::Bool
end

"""
    FactorizationInputPlan{T}(symbolic::SymbolicFactorization) where T

Create an uninitialized factorization input plan.
"""
function FactorizationInputPlan{T}(symbolic::SymbolicFactorization) where T
    return FactorizationInputPlan{T}(
        symbolic,
        0, 0,  # myrank, nranks
        Int[],  # row_partition
        Dict{Int, Vector{Tuple{Int,Int}}}(),
        Int[],
        Int[],
        Dict{Int, Vector{Tuple{Int,Int}}}(),
        Dict{Int, Vector{Tuple{Int,Int}}}(),
        Dict{Int, Vector{T}}(),
        Dict{Int, Vector{T}}(),
        Dict{Tuple{Int,Int}, T}(),
        false
    )
end
