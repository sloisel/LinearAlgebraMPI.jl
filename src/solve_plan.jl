"""
Distributed triangular solve using MUMPS-style tree-based communication.

The key insight is that with subtree-to-rank assignment, cross-rank communication
only occurs at subtree boundaries. This module provides:
1. SolvePlan initialization from symbolic factorization
2. Distributed forward solve (L y = b)
3. Distributed backward solve (U x = y)

Communication pattern:
- Forward solve: process leaves to root, send update contributions upward
- Backward solve: process root to leaves, send solution components downward
"""

using MPI
using SparseArrays

# ============================================================================
# Solve Plan Initialization
# ============================================================================

"""
    initialize_solve_plan!(plan::SolvePlan{T}) where T

Initialize the solve plan with MPI rank information and communication patterns.
Must be called after MPI is initialized.
"""
function initialize_solve_plan!(plan::SolvePlan{T}) where T
    comm = MPI.COMM_WORLD
    plan.myrank = MPI.Comm_rank(comm)
    plan.nranks = MPI.Comm_size(comm)

    symbolic = plan.symbolic
    n = symbolic.n
    nsupernodes = length(symbolic.supernodes)

    # Find supernodes owned by this rank (in postorder)
    plan.my_supernodes_postorder = Int[]
    for sidx in symbolic.snode_postorder
        if symbolic.snode_owner[sidx] == plan.myrank
            push!(plan.my_supernodes_postorder, sidx)
        end
    end

    # Find subtree roots: supernodes whose parent is on a different rank
    plan.subtree_roots = Int[]
    for sidx in 1:nsupernodes
        if symbolic.snode_owner[sidx] == plan.myrank
            parent = symbolic.snode_parent[sidx]
            if parent != 0 && symbolic.snode_owner[parent] != plan.myrank
                push!(plan.subtree_roots, sidx)
            end
        end
    end

    # Build global-to-local and local-to-global mappings
    # Count how many columns this rank owns
    local_count = 0
    for sidx in plan.my_supernodes_postorder
        snode = symbolic.supernodes[sidx]
        local_count += length(snode.cols)
    end

    plan.global_to_local = zeros(Int, n)
    plan.local_to_global = zeros(Int, local_count)

    local_idx = 0
    for sidx in plan.my_supernodes_postorder
        snode = symbolic.supernodes[sidx]
        start_idx = local_idx + 1
        for col in snode.cols
            local_idx += 1
            elim_idx = symbolic.global_to_elim[col]
            plan.global_to_local[elim_idx] = local_idx
            plan.local_to_global[local_idx] = elim_idx
        end
        plan.local_snode_indices[sidx] = start_idx:local_idx
    end

    # Allocate work vector
    plan.work_vector = zeros(T, local_count)

    # Compute subtree_local_columns: for each subtree root, the local columns in its subtree
    # First, find the subtree root for each supernode (the first ancestor that is a subtree root)
    subtree_root_set = Set(plan.subtree_roots)
    snode_to_subtree_root = Dict{Int, Int}()

    for sidx in plan.my_supernodes_postorder
        # Walk up the tree until we find a subtree root or the tree root
        current = sidx
        while true
            if current in subtree_root_set
                snode_to_subtree_root[sidx] = current
                break
            end
            parent = symbolic.snode_parent[current]
            if parent == 0 || symbolic.snode_owner[parent] != plan.myrank
                # Reached tree root or different rank without finding subtree root
                # This supernode is in a subtree that goes all the way to the tree root
                # (its subtree root is the topmost supernode on this rank that has parent on different rank)
                snode_to_subtree_root[sidx] = current
                break
            end
            current = parent
        end
    end

    # Build the mapping from subtree root to local columns
    for sidx in plan.subtree_roots
        plan.subtree_local_columns[sidx] = Int[]
    end

    for sidx in plan.my_supernodes_postorder
        subtree_root = snode_to_subtree_root[sidx]
        if subtree_root in subtree_root_set
            local_range = plan.local_snode_indices[sidx]
            append!(plan.subtree_local_columns[subtree_root], collect(local_range))
        end
    end

    # Build communication patterns for subtree roots
    for sidx in plan.subtree_roots
        parent = symbolic.snode_parent[sidx]
        parent_owner = symbolic.snode_owner[parent]

        # Determine update rows: rows in this supernode's frontal that also
        # appear in parent's frontal (these are the rows that receive contributions)
        info = symbolic.frontal_info[sidx]
        nfs = info.nfs
        update_rows = info.row_indices[nfs+1:end]

        # Convert to elimination indices
        update_rows_elim = [symbolic.global_to_elim[r] for r in update_rows]

        plan.subtree_root_update_rows[sidx] = update_rows_elim
        plan.forward_send_to[sidx] = parent_owner

        # Allocate send buffer for forward solve
        buffer_size = length(update_rows_elim)
        plan.send_buffers[(sidx, parent_owner)] = zeros(T, buffer_size)
    end

    # For backward solve, we receive from parents
    for sidx in plan.subtree_roots
        parent = symbolic.snode_parent[sidx]
        parent_owner = symbolic.snode_owner[parent]

        # Same update rows, receive from parent
        update_rows_elim = plan.subtree_root_update_rows[sidx]
        plan.backward_recv_from[sidx] = parent_owner

        # Allocate recv buffer for backward solve
        buffer_size = length(update_rows_elim)
        plan.recv_buffers[(sidx, parent_owner)] = zeros(T, buffer_size)
    end

    # For forward solve, we receive from children on other ranks
    # For backward solve, we send to children on other ranks
    for sidx in plan.my_supernodes_postorder
        children = symbolic.snode_children[sidx]
        for child in children
            child_owner = symbolic.snode_owner[child]
            if child_owner != plan.myrank
                # This child is a subtree root on another rank
                # We receive from it in forward solve
                plan.forward_recv_from[child] = child_owner

                # In backward solve, we send to it
                if !haskey(plan.backward_send_to, sidx)
                    plan.backward_send_to[sidx] = Int[]
                end
                push!(plan.backward_send_to[sidx], child_owner)

                # Allocate buffers - the child will have computed update_rows_elim
                # We need to know the buffer size, which is the number of update rows
                # that map into our frontal matrix
                child_info = symbolic.frontal_info[child]
                child_nfs = child_info.nfs
                child_update_rows = child_info.row_indices[child_nfs+1:end]
                child_update_rows_elim = [symbolic.global_to_elim[r] for r in child_update_rows]

                buffer_size = length(child_update_rows_elim)
                plan.recv_buffers[(child, child_owner)] = zeros(T, buffer_size)
                plan.send_buffers[(child, child_owner)] = zeros(T, buffer_size)
            end
        end
    end

    plan.initialized = true
    return plan
end

"""
    get_or_create_solve_plan(F::Union{LUFactorizationMPI{T}, LDLTFactorizationMPI{T}}) where T

Get or create a solve plan for the given factorization.
The plan is cached based on the symbolic factorization hash.
"""
const _solve_plan_cache = Dict{Tuple{Blake3Hash, DataType}, Any}()

function get_or_create_solve_plan(F::LUFactorizationMPI{T}) where T
    hash = F.symbolic.structural_hash
    key = (hash, T)
    if haskey(_solve_plan_cache, key)
        return _solve_plan_cache[key]::SolvePlan{T}
    end
    plan = SolvePlan{T}(F.symbolic)
    initialize_solve_plan!(plan)
    _solve_plan_cache[key] = plan
    return plan
end

function get_or_create_solve_plan(F::LDLTFactorizationMPI{T}) where T
    hash = F.symbolic.structural_hash
    key = (hash, T)
    if haskey(_solve_plan_cache, key)
        return _solve_plan_cache[key]::SolvePlan{T}
    end
    plan = SolvePlan{T}(F.symbolic)
    initialize_solve_plan!(plan)
    _solve_plan_cache[key] = plan
    return plan
end

"""
    clear_solve_plan_cache!()

Clear the solve plan cache.
"""
function clear_solve_plan_cache!()
    empty!(_solve_plan_cache)
end

# ============================================================================
# Distributed Forward Solve (L y = b)
# ============================================================================

"""
    distributed_forward_solve_lu!(y_local::Vector{T},
                                   F::LUFactorizationMPI{T},
                                   b::VectorMPI{T},
                                   plan::SolvePlan{T}) where T

Distributed forward solve: L y = c where c is the permuted RHS.

y_local contains the solution values for elimination indices owned by this rank.
The solution is in elimination order.
"""
function distributed_forward_solve_lu!(y_local::Vector{T},
                                        F::LUFactorizationMPI{T},
                                        b_permuted_elim::Vector{T},
                                        plan::SolvePlan{T}) where T
    comm = MPI.COMM_WORLD
    symbolic = plan.symbolic
    L_local = F.L_local

    # Initialize y_local from b_permuted_elim for columns owned by this rank
    for local_idx in 1:length(y_local)
        elim_idx = plan.local_to_global[local_idx]
        y_local[local_idx] = b_permuted_elim[elim_idx]
    end

    # Process supernodes in postorder (leaves to root)
    for sidx in plan.my_supernodes_postorder
        snode = symbolic.supernodes[sidx]
        info = symbolic.frontal_info[sidx]
        nfs = info.nfs

        # Receive contributions from children on other ranks
        children = symbolic.snode_children[sidx]
        for child in children
            child_owner = symbolic.snode_owner[child]
            if child_owner != plan.myrank
                # Receive contribution from child
                recv_buf = plan.recv_buffers[(child, child_owner)]
                MPI.Recv!(recv_buf, comm; source=child_owner, tag=child)

                # Subtract contribution from y_local for the update rows
                # (contribution = L[row,col] * y[col], and forward solve does y[row] -= L[row,col] * y[col])
                child_info = symbolic.frontal_info[child]
                child_nfs = child_info.nfs
                child_update_rows = child_info.row_indices[child_nfs+1:end]

                for (buf_idx, global_row) in enumerate(child_update_rows)
                    elim_row = symbolic.global_to_elim[global_row]
                    local_row = plan.global_to_local[elim_row]
                    if local_row > 0
                        y_local[local_row] -= recv_buf[buf_idx]
                    end
                end
            end
        end

        # Local forward substitution within this supernode
        local_range = plan.local_snode_indices[sidx]
        for local_col in local_range
            elim_col = plan.local_to_global[local_col]

            yk = y_local[local_col]
            if yk != zero(T)
                # Apply L column to rows below
                for idx in nzrange(L_local, elim_col)
                    elim_row = rowvals(L_local)[idx]
                    if elim_row != elim_col  # Skip diagonal (unit)
                        local_row = plan.global_to_local[elim_row]
                        if local_row > 0
                            # This row is owned by this rank
                            y_local[local_row] -= nonzeros(L_local)[idx] * yk
                        end
                    end
                end
            end
        end

        # If this is a subtree root, send contribution to parent
        parent = symbolic.snode_parent[sidx]
        if parent != 0 && symbolic.snode_owner[parent] != plan.myrank
            parent_owner = symbolic.snode_owner[parent]
            send_buf = plan.send_buffers[(sidx, parent_owner)]

            # Prepare contribution: sum over ALL columns in THIS SUBTREE
            # Each update row needs contributions from all columns in the subtree
            # that have nonzeros in that row
            subtree_columns = plan.subtree_local_columns[sidx]
            update_rows_elim = plan.subtree_root_update_rows[sidx]
            for (buf_idx, elim_row) in enumerate(update_rows_elim)
                contribution = zero(T)
                # Iterate through columns belonging to this subtree only
                # Only add contribution when elim_row > elim_col (strictly below diagonal in L)
                for local_col in subtree_columns
                    elim_col = plan.local_to_global[local_col]
                    if elim_row > elim_col  # Only below diagonal
                        yk = y_local[local_col]

                        # Find L[elim_row, elim_col]
                        for idx in nzrange(L_local, elim_col)
                            if rowvals(L_local)[idx] == elim_row
                                contribution += nonzeros(L_local)[idx] * yk
                                break
                            end
                        end
                    end
                end
                send_buf[buf_idx] = contribution
                # Note: We do NOT apply contributions locally here.
                # All contributions to local rows were already applied during
                # local forward substitution in each supernode.
            end

            # Send to parent (parent will apply for rows it owns)
            MPI.Send(send_buf, parent_owner, sidx, comm)
        end
    end

    return y_local
end

# ============================================================================
# Distributed Backward Solve (U x = y)
# ============================================================================

"""
    distributed_backward_solve_lu!(x_local::Vector{T},
                                    F::LUFactorizationMPI{T},
                                    y_local::Vector{T},
                                    plan::SolvePlan{T}) where T

Distributed backward solve: U x = y.

x_local contains the solution values for elimination indices owned by this rank.
"""
function distributed_backward_solve_lu!(x_local::Vector{T},
                                         F::LUFactorizationMPI{T},
                                         y_local::Vector{T},
                                         plan::SolvePlan{T}) where T
    comm = MPI.COMM_WORLD
    symbolic = plan.symbolic
    U_local = F.U_local

    # Initialize x_local from y_local
    copy!(x_local, y_local)
    n = symbolic.n

    # Build column owner mapping
    elim_col_to_owner = zeros(Int, n)
    for sidx in 1:length(symbolic.supernodes)
        snode = symbolic.supernodes[sidx]
        owner = symbolic.snode_owner[sidx]
        for col in snode.cols
            elim_col = symbolic.global_to_elim[col]
            elim_col_to_owner[elim_col] = owner
        end
    end

    # Two-pass approach with Allreduce to handle cross-rank dependencies

    # First pass: each rank does local backward sub for its columns only
    # (not applying cross-rank U entries yet)
    for sidx in reverse(plan.my_supernodes_postorder)
        snode = symbolic.supernodes[sidx]
        info = symbolic.frontal_info[sidx]
        nfs = info.nfs

        # Local backward substitution within this supernode (reverse order)
        local_range = plan.local_snode_indices[sidx]
        for local_col in reverse(collect(local_range))
            elim_col = plan.local_to_global[local_col]

            # First, we need to account for contributions from columns > elim_col
            # that are owned by OTHER ranks. We'll handle this after Allreduce.
            # For now, just compute partial result.

            # Divide by diagonal
            diag_val = zero(T)
            for idx in nzrange(U_local, elim_col)
                if rowvals(U_local)[idx] == elim_col
                    diag_val = nonzeros(U_local)[idx]
                    break
                end
            end
            if diag_val == zero(T)
                error("Zero diagonal in U at elimination step $elim_col")
            end
            x_local[local_col] /= diag_val

            xk = x_local[local_col]

            # Update rows above (only for columns we own)
            for idx in nzrange(U_local, elim_col)
                elim_row = rowvals(U_local)[idx]
                if elim_row != elim_col
                    local_row = plan.global_to_local[elim_row]
                    if local_row > 0
                        # Only apply if the update column is owned by us
                        x_local[local_row] -= nonzeros(U_local)[idx] * xk
                    end
                end
            end
        end
    end

    # Gather all x values
    x_global = zeros(T, n)
    for local_idx in 1:length(x_local)
        elim_idx = plan.local_to_global[local_idx]
        x_global[elim_idx] = x_local[local_idx]
    end
    x_global = MPI.Allreduce(x_global, +, comm)

    # Now re-compute x values by applying contributions from other ranks' columns
    # We need to re-do the backward solve with proper ordering
    # Process columns in decreasing order, applying cross-rank contributions

    # Reset x_local to y_local
    copy!(x_local, y_local)

    # Process ALL columns in decreasing order, using x_global for cross-rank columns
    for elim_col in n:-1:1
        local_col = plan.global_to_local[elim_col]
        col_owner = elim_col_to_owner[elim_col]

        if col_owner == plan.myrank
            # We own this column, compute x[elim_col]
            # First apply contributions from columns > elim_col that we own
            # (already done via y_local updates during first pass... but we reset)
            # Actually, let's do it fresh here

            # Apply contributions from ALL columns > elim_col
            for elim_col2 in elim_col+1:n
                col2_owner = elim_col_to_owner[elim_col2]
                if col2_owner == plan.myrank
                    # Use local x value
                    local_col2 = plan.global_to_local[elim_col2]
                    x_col2 = x_local[local_col2]
                else
                    # Use global x value from other rank
                    x_col2 = x_global[elim_col2]
                end

                # Find U[elim_col, elim_col2]
                for idx in nzrange(U_local, elim_col2)
                    if rowvals(U_local)[idx] == elim_col
                        x_local[local_col] -= nonzeros(U_local)[idx] * x_col2
                        break
                    end
                end
            end

            # Divide by diagonal
            diag_val = zero(T)
            for idx in nzrange(U_local, elim_col)
                if rowvals(U_local)[idx] == elim_col
                    diag_val = nonzeros(U_local)[idx]
                    break
                end
            end
            x_local[local_col] /= diag_val
        end
    end

    return x_local
end

# ============================================================================
# High-level Distributed Solve Interface
# ============================================================================

"""
    distributed_solve_lu!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T

Solve A*x = b using distributed LU factorization without gathering factors.

Steps:
1. Gather b to get components needed for this rank's supernodes
2. Apply permutations to get b in elimination order with pivoting
3. Distributed forward solve: L y = permuted_b
4. Distributed backward solve: U z = y
5. Apply inverse permutations
6. Distribute result to x
"""
function distributed_solve_lu!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n = F.symbolic.n

    # Get or create solve plan
    plan = get_or_create_solve_plan(F)

    # For now, gather b to all ranks (will optimize later)
    # TODO: Only gather elements needed for this rank's supernodes
    b_full = Vector(b)

    # Apply AMD permutation: bp[i] = b[perm[i]]
    bp = Vector{T}(undef, n)
    for i = 1:n
        bp[i] = b_full[F.symbolic.perm[i]]
    end

    # Reorder to elimination order: b_elim[k] = bp[elim_to_global[k]]
    b_elim = Vector{T}(undef, n)
    for k = 1:n
        b_elim[k] = bp[F.symbolic.elim_to_global[k]]
    end

    # Apply pivot permutation
    c = Vector{T}(undef, n)
    for k = 1:n
        src_elim = F.symbolic.global_to_elim[F.row_perm[k]]
        c[k] = b_elim[src_elim]
    end

    # Allocate local vectors
    local_size = length(plan.local_to_global)
    y_local = zeros(T, local_size)
    x_local = zeros(T, local_size)

    # Distributed forward solve
    distributed_forward_solve_lu!(y_local, F, c, plan)

    # Distributed backward solve
    distributed_backward_solve_lu!(x_local, F, y_local, plan)

    # Gather solution from all ranks
    # Each rank has x values for its supernodes in elimination order
    # We need to gather to construct the full solution

    # Gather counts and displacements
    local_count = Int32(local_size)
    all_counts = MPI.Allgather(local_count, comm)

    # Gather local indices and values
    local_indices = plan.local_to_global
    z_global = zeros(T, n)

    # Use Allgatherv to gather all x_local values and their indices
    total_count = sum(all_counts)
    all_x = Vector{T}(undef, total_count)
    all_indices_int32 = Vector{Int32}(undef, total_count)

    MPI.Allgatherv!(x_local, MPI.VBuffer(all_x, all_counts), comm)
    MPI.Allgatherv!(Int32.(local_indices), MPI.VBuffer(all_indices_int32, all_counts), comm)

    # Convert back to Int
    all_indices = Int.(all_indices_int32)

    # Fill in z_global (in elimination order)
    offset = 0
    for r = 0:plan.nranks-1
        count = all_counts[r+1]
        for i = 1:count
            elim_idx = all_indices[offset + i]
            z_global[elim_idx] = all_x[offset + i]
        end
        offset += count
    end

    # Apply inverse permutations
    # Reorder from elimination to global (AMD) order
    zp = Vector{T}(undef, n)
    for k = 1:n
        zp[F.symbolic.elim_to_global[k]] = z_global[k]
    end

    # Apply inverse AMD permutation
    result = Vector{T}(undef, n)
    for i = 1:n
        result[F.symbolic.perm[i]] = zp[i]
    end

    # Distribute result back to VectorMPI
    my_start = x.partition[rank + 1]
    my_end = x.partition[rank + 2] - 1
    for i = my_start:my_end
        x.v[i - my_start + 1] = result[i]
    end

    return x
end

# ============================================================================
# Distributed LDLT Solve
# ============================================================================

"""
    distributed_forward_solve_ldlt!(y_local::Vector{T},
                                     F::LDLTFactorizationMPI{T},
                                     b_permuted::Vector{T},
                                     plan::SolvePlan{T}) where T

Distributed forward solve for LDLT: L y = b where L is unit lower triangular.

Uses tree-based communication (same as LU forward solve) - contributions are
sent through parent-child relationships in the elimination tree.
"""
function distributed_forward_solve_ldlt!(y_local::Vector{T},
                                          F::LDLTFactorizationMPI{T},
                                          b_permuted::Vector{T},
                                          plan::SolvePlan{T}) where T
    comm = MPI.COMM_WORLD
    symbolic = plan.symbolic
    L_local = F.L_local

    # Initialize y_local from b_permuted for columns owned by this rank
    for local_idx in 1:length(y_local)
        elim_idx = plan.local_to_global[local_idx]
        y_local[local_idx] = b_permuted[elim_idx]
    end

    # Process supernodes in postorder (leaves to root)
    for sidx in plan.my_supernodes_postorder
        snode = symbolic.supernodes[sidx]
        info = symbolic.frontal_info[sidx]

        # Receive contributions from children on other ranks
        children = symbolic.snode_children[sidx]
        for child in children
            child_owner = symbolic.snode_owner[child]
            if child_owner != plan.myrank
                recv_buf = plan.recv_buffers[(child, child_owner)]
                MPI.Recv!(recv_buf, comm; source=child_owner, tag=child)

                # Subtract contribution from y_local for the update rows
                child_info = symbolic.frontal_info[child]
                child_nfs = child_info.nfs
                child_update_rows = child_info.row_indices[child_nfs+1:end]

                for (buf_idx, global_row) in enumerate(child_update_rows)
                    elim_row = symbolic.global_to_elim[global_row]
                    local_row = plan.global_to_local[elim_row]
                    if local_row > 0
                        y_local[local_row] -= recv_buf[buf_idx]
                    end
                end
            end
        end

        # Local forward substitution within this supernode
        local_range = plan.local_snode_indices[sidx]
        for local_col in local_range
            elim_col = plan.local_to_global[local_col]

            yk = y_local[local_col]
            if yk != zero(T)
                # Apply L column to rows below (L has unit diagonal, skip it)
                for idx in nzrange(L_local, elim_col)
                    elim_row = rowvals(L_local)[idx]
                    if elim_row != elim_col
                        local_row = plan.global_to_local[elim_row]
                        if local_row > 0
                            y_local[local_row] -= nonzeros(L_local)[idx] * yk
                        end
                    end
                end
            end
        end

        # If this is a subtree root, send contribution to parent
        parent = symbolic.snode_parent[sidx]
        if parent != 0 && symbolic.snode_owner[parent] != plan.myrank
            parent_owner = symbolic.snode_owner[parent]
            send_buf = plan.send_buffers[(sidx, parent_owner)]

            # Use subtree_local_columns to compute contributions from all columns in this subtree
            subtree_columns = get(plan.subtree_local_columns, sidx, local_range)
            update_rows_elim = plan.subtree_root_update_rows[sidx]
            for (buf_idx, elim_row) in enumerate(update_rows_elim)
                contribution = zero(T)
                for local_col in subtree_columns
                    elim_col = plan.local_to_global[local_col]
                    if elim_row > elim_col  # Only below diagonal
                        yk = y_local[local_col]

                        for idx in nzrange(L_local, elim_col)
                            if rowvals(L_local)[idx] == elim_row
                                contribution += nonzeros(L_local)[idx] * yk
                                break
                            end
                        end
                    end
                end
                send_buf[buf_idx] = contribution
            end

            MPI.Send(send_buf, parent_owner, sidx, comm)
        end
    end

    return y_local
end

"""
    distributed_diagonal_solve_ldlt!(z_local::Vector{T},
                                      F::LDLTFactorizationMPI{T},
                                      y_local::Vector{T},
                                      plan::SolvePlan{T}) where T

Distributed diagonal solve for LDLT: D z = y where D is block diagonal.

Each rank processes its own supernodes' D blocks locally.
Handles both 1×1 and 2×2 Bunch-Kaufman pivot blocks.
"""
function distributed_diagonal_solve_ldlt!(z_local::Vector{T},
                                           F::LDLTFactorizationMPI{T},
                                           y_local::Vector{T},
                                           plan::SolvePlan{T}) where T
    symbolic = plan.symbolic
    D_local = F.D_local
    pivots = F.pivots

    copy!(z_local, y_local)

    # Process each column owned by this rank
    local_idx = 1
    while local_idx <= length(z_local)
        elim_idx = plan.local_to_global[local_idx]

        if pivots[elim_idx] >= 0
            # 1×1 pivot
            d_kk = zero(T)
            for idx in nzrange(D_local, elim_idx)
                if rowvals(D_local)[idx] == elim_idx
                    d_kk = nonzeros(D_local)[idx]
                    break
                end
            end
            if abs(d_kk) < eps(real(T))
                d_kk = eps(real(T))
            end
            z_local[local_idx] /= d_kk
            local_idx += 1
        else
            # 2×2 pivot at (elim_idx, elim_idx+1)
            # Check if next column is also owned by this rank
            if local_idx + 1 <= length(z_local)
                elim_idx_next = plan.local_to_global[local_idx + 1]
                if elim_idx_next == elim_idx + 1
                    # Both columns of 2×2 block on this rank
                    d_kk = zero(T)
                    d_k1k = zero(T)
                    d_k1k1 = zero(T)

                    for idx in nzrange(D_local, elim_idx)
                        row = rowvals(D_local)[idx]
                        if row == elim_idx
                            d_kk = nonzeros(D_local)[idx]
                        elseif row == elim_idx + 1
                            d_k1k = nonzeros(D_local)[idx]
                        end
                    end

                    for idx in nzrange(D_local, elim_idx + 1)
                        row = rowvals(D_local)[idx]
                        if row == elim_idx + 1
                            d_k1k1 = nonzeros(D_local)[idx]
                        end
                    end

                    det = d_kk * d_k1k1 - d_k1k * d_k1k
                    if abs(det) < eps(real(T))
                        det = sign(det) * max(abs(det), eps(real(T)))
                    end

                    z_k = z_local[local_idx]
                    z_k1 = z_local[local_idx + 1]

                    z_local[local_idx] = (d_k1k1 * z_k - d_k1k * z_k1) / det
                    z_local[local_idx + 1] = (-d_k1k * z_k + d_kk * z_k1) / det

                    local_idx += 2
                else
                    # Edge case: 2×2 block split across ranks (shouldn't happen with subtree assignment)
                    @warn "2×2 pivot block split across ranks at elim_idx=$elim_idx"
                    local_idx += 1
                end
            else
                local_idx += 1
            end
        end
    end

    return z_local
end

"""
    distributed_backward_solve_ldlt!(x_local::Vector{T},
                                      F::LDLTFactorizationMPI{T},
                                      z_local::Vector{T},
                                      plan::SolvePlan{T}) where T

Distributed backward solve for LDLT: L^T x = z (transpose, not adjoint).

For L^T x = z, row i gives: x[i] = z[i] - sum(L[j,i] * x[j] for j > i)
Process columns in decreasing order.
"""
function distributed_backward_solve_ldlt!(x_local::Vector{T},
                                           F::LDLTFactorizationMPI{T},
                                           z_local::Vector{T},
                                           plan::SolvePlan{T}) where T
    comm = MPI.COMM_WORLD
    symbolic = plan.symbolic
    L_local = F.L_local
    n = symbolic.n

    copy!(x_local, z_local)

    # Build column owner mapping
    elim_col_to_owner = zeros(Int, n)
    for sidx in 1:length(symbolic.supernodes)
        snode = symbolic.supernodes[sidx]
        owner = symbolic.snode_owner[sidx]
        for col in snode.cols
            elim_col = symbolic.global_to_elim[col]
            elim_col_to_owner[elim_col] = owner
        end
    end

    # Two-pass approach with Allreduce to handle cross-rank dependencies

    # First pass: each rank does local backward sub for its columns only
    for sidx in reverse(plan.my_supernodes_postorder)
        local_range = plan.local_snode_indices[sidx]
        for local_col in reverse(collect(local_range))
            elim_col = plan.local_to_global[local_col]

            # Apply L^T contributions from local rows only
            for idx in nzrange(L_local, elim_col)
                elim_row = rowvals(L_local)[idx]
                if elim_row > elim_col
                    local_row = plan.global_to_local[elim_row]
                    if local_row > 0
                        x_local[local_col] -= nonzeros(L_local)[idx] * x_local[local_row]
                    end
                end
            end
        end
    end

    # Gather all x values via Allreduce
    x_global = zeros(T, n)
    for local_idx in 1:length(x_local)
        elim_idx = plan.local_to_global[local_idx]
        x_global[elim_idx] = x_local[local_idx]
    end
    x_global = MPI.Allreduce(x_global, +, comm)

    # Reset x_local to z_local
    copy!(x_local, z_local)

    # Second pass: process all columns in decreasing order
    for elim_col in n:-1:1
        local_col = plan.global_to_local[elim_col]
        col_owner = elim_col_to_owner[elim_col]

        if col_owner == plan.myrank
            # Apply contributions from ALL columns > elim_col
            for elim_col2 in elim_col+1:n
                col2_owner = elim_col_to_owner[elim_col2]
                if col2_owner == plan.myrank
                    local_col2 = plan.global_to_local[elim_col2]
                    x_col2 = x_local[local_col2]
                else
                    x_col2 = x_global[elim_col2]
                end

                # Find L[elim_col2, elim_col] (L is stored in CSC by column elim_col)
                # For L^T solve: x[elim_col] -= L[elim_col2, elim_col] * x[elim_col2]
                for idx in nzrange(L_local, elim_col)
                    if rowvals(L_local)[idx] == elim_col2
                        x_local[local_col] -= nonzeros(L_local)[idx] * x_col2
                        break
                    end
                end
            end
            # Unit diagonal - no division needed
        end
    end

    return x_local
end

"""
    distributed_solve_ldlt!(x::VectorMPI{T}, F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T

Solve A*x = b using distributed LDLT factorization without gathering factors.

Steps:
1. Gather b components needed for this rank's supernodes
2. Apply fill-reducing and symmetric pivot permutations
3. Distributed forward solve: L y = permuted_b
4. Distributed diagonal solve: D z = y
5. Distributed backward solve: L^T w = z
6. Apply inverse permutations
7. Distribute result to x
"""
function distributed_solve_ldlt!(x::VectorMPI{T}, F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n = F.symbolic.n

    # Get or create solve plan
    plan = get_or_create_solve_plan(F)

    # Gather b to all ranks (TODO: optimize to only gather needed elements)
    b_full = Vector(b)

    # Step 1: Apply fill-reducing permutation
    work = Vector{T}(undef, n)
    for i = 1:n
        work[i] = b_full[F.symbolic.perm[i]]
    end

    # Step 2: Apply symmetric pivot permutation
    work2 = Vector{T}(undef, n)
    for k = 1:n
        work2[k] = work[F.sym_perm[k]]
    end

    # Allocate local vectors
    local_size = length(plan.local_to_global)
    y_local = zeros(T, local_size)
    z_local = zeros(T, local_size)
    w_local = zeros(T, local_size)

    # Distributed forward solve: L y = work2
    distributed_forward_solve_ldlt!(y_local, F, work2, plan)

    # Distributed diagonal solve: D z = y
    distributed_diagonal_solve_ldlt!(z_local, F, y_local, plan)

    # Distributed backward solve: L^T w = z
    distributed_backward_solve_ldlt!(w_local, F, z_local, plan)

    # Gather solution from all ranks
    local_count = Int32(local_size)
    all_counts = MPI.Allgather(local_count, comm)

    local_indices = plan.local_to_global
    total_count = sum(all_counts)
    all_w = Vector{T}(undef, total_count)
    all_indices_int32 = Vector{Int32}(undef, total_count)

    MPI.Allgatherv!(w_local, MPI.VBuffer(all_w, all_counts), comm)
    MPI.Allgatherv!(Int32.(local_indices), MPI.VBuffer(all_indices_int32, all_counts), comm)

    all_indices = Int.(all_indices_int32)

    # Fill in w_global (solution in elimination order after L^T solve)
    w_global = zeros(T, n)
    offset = 0
    for r = 0:plan.nranks-1
        count = all_counts[r+1]
        for i = 1:count
            elim_idx = all_indices[offset + i]
            w_global[elim_idx] = all_w[offset + i]
        end
        offset += count
    end

    # Step 6: Apply inverse symmetric permutation
    for k = 1:n
        work[F.sym_perm[k]] = w_global[k]
    end

    # Step 7: Apply inverse fill-reducing permutation
    result = Vector{T}(undef, n)
    for i = 1:n
        result[i] = work[F.symbolic.invperm[i]]
    end

    # Distribute result back to VectorMPI
    my_start = x.partition[rank + 1]
    my_end = x.partition[rank + 2] - 1
    for i = my_start:my_end
        x.v[i - my_start + 1] = result[i]
    end

    return x
end
