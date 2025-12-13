"""
Numerical LDLT factorization using the distributed multifrontal method.

For symmetric matrices, uses Bunch-Kaufman pivoting for numerical stability
with indefinite matrices.

Main algorithm:
1. Process supernodes in postorder (leaves to root)
2. For each supernode owned by this rank:
   a. Initialize frontal matrix from gathered sparse entries
   b. Extend-add contributions from children (all local since subtrees are on same rank)
   c. Partial factorization with Bunch-Kaufman pivoting
   d. Extract L and D factors to local storage
   e. Store update matrix for parent supernode
"""

using MPI
using SparseArrays

"""
    ldlt(A::SparseMatrixMPI{T}; reuse_symbolic=true, distributed_input=false) -> LDLTFactorizationMPI{T}

Compute LDLT factorization of a distributed symmetric sparse matrix.

Uses the multifrontal method with:
- AMD fill-reducing ordering
- Bunch-Kaufman pivoting for numerical stability with indefinite matrices
- MUMPS-style subtree-to-rank mapping

The factorization computes P' * L * D * L^T * P which equals A symmetrically permuted by perm.

Note: Uses transpose (L^T), not adjoint (L*). Correct for real symmetric
and complex symmetric matrices, but NOT for complex Hermitian matrices.

If `reuse_symbolic=true`, caches and reuses symbolic factorization for matrices
with the same sparsity pattern.

If `distributed_input=true`, uses MUMPS-style distributed matrix input where each
rank only provides its local portion of the matrix, avoiding the O(nnz) gather.
"""
function LinearAlgebra.ldlt(A::SparseMatrixMPI{T}; reuse_symbolic::Bool=true, distributed_input::Bool=false) where T
    # Get or compute symbolic factorization (symmetric variant)
    symbolic = reuse_symbolic ?
        get_symbolic_factorization(A; symmetric=true) :
        compute_symbolic_factorization(A; symmetric=true)

    # Perform numerical factorization
    return numerical_factorization_ldlt(A, symbolic; distributed_input=distributed_input)
end

"""
    numerical_factorization_ldlt(A, symbolic; distributed_input=false) -> LDLTFactorizationMPI{T}

Perform the numerical LDLT factorization with Bunch-Kaufman pivoting.

If `distributed_input=true`, uses distributed matrix input where each rank only
provides its local portion of the matrix.

Note: The current supernode assignment places complete subtrees on single ranks,
so all parent-child communication is local (no MPI communication during factorization).
"""
function numerical_factorization_ldlt(A::SparseMatrixMPI{T},
                                      symbolic::SymbolicFactorization;
                                      distributed_input::Bool=false) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    n = symbolic.n
    nsupernodes = length(symbolic.supernodes)

    # Get matrix input (either gathered or distributed)
    local input_plan::Union{Nothing, FactorizationInputPlan{T}} = nothing
    local Ap::Union{Nothing, SparseMatrixCSC{T,Int}} = nothing

    if distributed_input
        # Use distributed input plan
        input_plan = get_or_create_input_plan(symbolic, A)
        execute_input_plan!(input_plan, A)
    else
        # Gather the permuted matrix on all ranks for local access
        A_full = SparseMatrixCSC(A)
        Ap = A_full[symbolic.perm, symbolic.perm]
    end

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

            # 1. Initialize frontal matrix (symmetric)
            if distributed_input
                F = initialize_frontal_sym_distributed(input_plan, A, snode, info)
            else
                F = initialize_frontal_sym(Ap, snode, info)
            end

            # 2. Extend-add update matrices from children
            # Note: All children are on the same rank (subtrees assigned to single ranks)
            for child_sidx in symbolic.snode_children[sidx]
                if haskey(updates, child_sidx)
                    extend_add_sym!(F, updates[child_sidx], update_rows[child_sidx])
                    delete!(updates, child_sidx)
                    delete!(update_rows, child_sidx)
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

            # 6. Store update matrix for parent (parent is on same rank)
            parent_sidx = symbolic.snode_parent[sidx]
            if parent_sidx != 0
                nfs = info.nfs
                nrows = length(info.row_indices)
                if nrows > nfs
                    updates[sidx] = copy(F.F[nfs+1:nrows, nfs+1:nrows])
                    update_rows[sidx] = copy(F.row_indices[nfs+1:nrows])
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

"""
    gather_L_D(ldlt::LDLTFactorizationMPI{T}) -> (SparseMatrixCSC{T,Int}, SparseMatrixCSC{T,Int})

Gather the distributed L and D factors to full matrices on all ranks.
Useful for debugging and verification.
"""
function gather_L_D(ldlt::LDLTFactorizationMPI{T}) where T
    comm = MPI.COMM_WORLD
    n = ldlt.symbolic.n

    # Gather L
    L_I_local = Int[]
    L_J_local = Int[]
    L_V_local = T[]

    for j in 1:n
        for idx in nzrange(ldlt.L_local, j)
            push!(L_I_local, rowvals(ldlt.L_local)[idx])
            push!(L_J_local, j)
            push!(L_V_local, nonzeros(ldlt.L_local)[idx])
        end
    end

    L_count = Int32(length(L_I_local))
    L_counts = MPI.Allgather(L_count, comm)
    L_total = sum(L_counts)

    L_I_global = Vector{Int}(undef, L_total)
    L_J_global = Vector{Int}(undef, L_total)
    L_V_global = Vector{T}(undef, L_total)

    MPI.Allgatherv!(L_I_local, MPI.VBuffer(L_I_global, L_counts), comm)
    MPI.Allgatherv!(L_J_local, MPI.VBuffer(L_J_global, L_counts), comm)
    MPI.Allgatherv!(L_V_local, MPI.VBuffer(L_V_global, L_counts), comm)

    L_full = sparse(L_I_global, L_J_global, L_V_global, n, n)

    # Gather D
    D_I_local = Int[]
    D_J_local = Int[]
    D_V_local = T[]

    for j in 1:n
        for idx in nzrange(ldlt.D_local, j)
            push!(D_I_local, rowvals(ldlt.D_local)[idx])
            push!(D_J_local, j)
            push!(D_V_local, nonzeros(ldlt.D_local)[idx])
        end
    end

    D_count = Int32(length(D_I_local))
    D_counts = MPI.Allgather(D_count, comm)
    D_total = sum(D_counts)

    D_I_global = Vector{Int}(undef, D_total)
    D_J_global = Vector{Int}(undef, D_total)
    D_V_global = Vector{T}(undef, D_total)

    MPI.Allgatherv!(D_I_local, MPI.VBuffer(D_I_global, D_counts), comm)
    MPI.Allgatherv!(D_J_local, MPI.VBuffer(D_J_global, D_counts), comm)
    MPI.Allgatherv!(D_V_local, MPI.VBuffer(D_V_global, D_counts), comm)

    D_full = sparse(D_I_global, D_J_global, D_V_global, n, n)

    return L_full, D_full
end
