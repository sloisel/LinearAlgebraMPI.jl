"""
Numerical LDLT factorization using the distributed multifrontal method.

For symmetric matrices, uses Bunch-Kaufman pivoting for numerical stability
with indefinite matrices.

Main algorithm:
1. Process supernodes in postorder (leaves to root)
2. For each supernode owned by this rank:
   a. Initialize frontal matrix from distributed sparse entries
   b. Extend-add contributions from children (all local since subtrees are on same rank)
   c. Partial factorization with Bunch-Kaufman pivoting
   d. Extract L and D factors to local storage
   e. Store update matrix for parent supernode
"""

using MPI
using SparseArrays

# MPI tags for LDLT factorization communication
const LDLT_UPDATE_TAG = 200
const LDLT_ROWS_TAG = 201
const LDLT_SIZE_TAG = 202

"""
    ldlt(A::SparseMatrixMPI{T}; reuse_symbolic=true) -> LDLTFactorizationMPI{T}

Compute LDLT factorization of a distributed symmetric sparse matrix.

Uses the multifrontal method with:
- AMD fill-reducing ordering
- Bunch-Kaufman pivoting for numerical stability with indefinite matrices
- MUMPS-style subtree-to-rank mapping
- MUMPS-style distributed matrix input (each rank only provides its local portion)

The factorization computes P' * L * D * L^T * P which equals A symmetrically permuted by perm.

Note: Uses transpose (L^T), not adjoint (L*). Correct for real symmetric
and complex symmetric matrices, but NOT for complex Hermitian matrices.

If `reuse_symbolic=true`, caches and reuses symbolic factorization for matrices
with the same sparsity pattern.
"""
function LinearAlgebra.ldlt(A::SparseMatrixMPI{T}; reuse_symbolic::Bool=true) where T
    # Get or compute symbolic factorization (symmetric variant)
    symbolic = reuse_symbolic ?
        get_symbolic_factorization(A; symmetric=true) :
        compute_symbolic_factorization(A; symmetric=true)

    # Perform numerical factorization
    return numerical_factorization_ldlt(A, symbolic)
end

"""
    numerical_factorization_ldlt(A, symbolic) -> LDLTFactorizationMPI{T}

Perform the numerical LDLT factorization with Bunch-Kaufman pivoting.

Uses MUMPS-style distributed matrix input where each rank only provides its
local portion of the matrix. Supports cross-rank communication for extend-add
operations when supernodes are split across ranks.
"""
function numerical_factorization_ldlt(A::SparseMatrixMPI{T},
                                      symbolic::SymbolicFactorization) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    n = symbolic.n
    nsupernodes = length(symbolic.supernodes)

    # Use distributed input plan
    input_plan = get_or_create_input_plan(symbolic, A)
    execute_input_plan!(input_plan, A)

    # Storage for update matrices (keyed by supernode index)
    updates = Dict{Int, Matrix{T}}()
    update_rows = Dict{Int, Vector{Int}}()

    # Accumulators for L and D in COO format (each rank stores its portion)
    L_I = Int[]
    L_J = Int[]
    L_V = T[]
    D_I = Int[]
    D_J = Int[]
    D_V = T[]

    # Track pivot information and symmetric permutation
    pivots_local = zeros(Int, n)
    sym_perm_local = zeros(Int, n)

    # Process supernodes in postorder
    for sidx in symbolic.snode_postorder
        snode = symbolic.supernodes[sidx]
        info = symbolic.frontal_info[sidx]
        snode_owner = symbolic.snode_owner[sidx]

        # Get base elimination step before any pivoting
        base_elim = symbolic.global_to_elim[first(snode.cols)]

        if rank == snode_owner
            # This rank owns this supernode

            # 1. Initialize frontal matrix (symmetric) from distributed input
            F = initialize_frontal_sym_distributed(input_plan, A, snode, info)

            # 2. Extend-add update matrices from children (may involve cross-rank communication)
            for child_sidx in symbolic.snode_children[sidx]
                child_owner = symbolic.snode_owner[child_sidx]

                if child_owner == rank
                    # Local child - use stored update
                    if haskey(updates, child_sidx)
                        extend_add_sym!(F, updates[child_sidx], update_rows[child_sidx])
                        delete!(updates, child_sidx)
                        delete!(update_rows, child_sidx)
                    end
                else
                    # Cross-rank child - receive update matrix
                    # First receive size
                    size_buf = Vector{Int}(undef, 1)
                    MPI.Recv!(size_buf, comm; source=child_owner, tag=LDLT_SIZE_TAG + child_sidx)
                    nrows_update = size_buf[1]

                    if nrows_update > 0
                        # Receive update matrix and indices
                        update_matrix = Matrix{T}(undef, nrows_update, nrows_update)
                        child_rows = Vector{Int}(undef, nrows_update)

                        MPI.Recv!(update_matrix, comm; source=child_owner, tag=LDLT_UPDATE_TAG + child_sidx)
                        MPI.Recv!(child_rows, comm; source=child_owner, tag=LDLT_ROWS_TAG + child_sidx)

                        extend_add_sym!(F, update_matrix, child_rows)
                    end
                end
            end

            # 3. Partial factorization with Bunch-Kaufman pivoting
            D_local, pivot_info = partial_factor_ldlt!(F)

            # 4. Record symmetric permutation and pivot info
            for k = 1:info.nfs
                elim_pos = base_elim + k - 1
                sym_perm_local[elim_pos] = F.row_indices[k]
                pivots_local[elim_pos] = pivot_info[k]
            end

            # 5. Extract L and D entries
            extract_L_D!(F, snode, symbolic.global_to_elim, L_I, L_J, L_V, D_I, D_J, D_V, D_local, pivot_info)

            # 6. Handle update matrix for parent
            parent_sidx = symbolic.snode_parent[sidx]
            if parent_sidx != 0
                parent_owner = symbolic.snode_owner[parent_sidx]
                nfs = info.nfs
                nrows = length(info.row_indices)
                if nrows > nfs
                    update_matrix = copy(F.F[nfs+1:nrows, nfs+1:nrows])
                    rows = copy(F.row_indices[nfs+1:nrows])

                    if parent_owner == rank
                        # Local parent - store update
                        updates[sidx] = update_matrix
                        update_rows[sidx] = rows
                    else
                        # Cross-rank parent - send update matrix
                        nrows_update = length(rows)
                        MPI.Send([nrows_update], parent_owner, LDLT_SIZE_TAG + sidx, comm)
                        MPI.Send(update_matrix, parent_owner, LDLT_UPDATE_TAG + sidx, comm)
                        MPI.Send(rows, parent_owner, LDLT_ROWS_TAG + sidx, comm)
                    end
                else
                    # No update matrix (fully factored)
                    if parent_owner != rank
                        # Still need to send empty size to parent
                        MPI.Send([0], parent_owner, LDLT_SIZE_TAG + sidx, comm)
                    end
                end
            end
        end

        # Synchronize between supernodes
        MPI.Barrier(comm)
    end

    # Build local sparse L and D
    L_local = sparse(L_I, L_J, L_V, n, n)
    D_local = sparse(D_I, D_J, D_V, n, n)

    # Gather global symmetric permutation and pivots via Allreduce
    # Use MAX for sym_perm (values are positive indices or 0)
    # Use SUM for pivots (values are 1 or -1, with only one rank setting each position)
    sym_perm = MPI.Allreduce(sym_perm_local, MPI.MAX, comm)
    pivots = MPI.Allreduce(pivots_local, MPI.SUM, comm)

    # Build inverse symmetric permutation
    inv_sym_perm = zeros(Int, n)
    for k = 1:n
        if sym_perm[k] != 0
            inv_sym_perm[sym_perm[k]] = k
        end
    end

    return LDLTFactorizationMPI{T}(symbolic, L_local, D_local, pivots, sym_perm, inv_sym_perm)
end
