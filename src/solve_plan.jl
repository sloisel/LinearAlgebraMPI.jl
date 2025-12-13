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
                MPI.Recv!(recv_buf, child_owner, child, comm)

                # Add contribution to y_local for the update rows
                child_info = symbolic.frontal_info[child]
                child_nfs = child_info.nfs
                child_update_rows = child_info.row_indices[child_nfs+1:end]

                for (buf_idx, global_row) in enumerate(child_update_rows)
                    elim_row = symbolic.global_to_elim[global_row]
                    local_row = plan.global_to_local[elim_row]
                    if local_row > 0
                        y_local[local_row] += recv_buf[buf_idx]
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

            # Prepare contribution: y values for update rows
            update_rows_elim = plan.subtree_root_update_rows[sidx]
            for (buf_idx, elim_row) in enumerate(update_rows_elim)
                # The update contribution is stored in the L column
                # We need to gather the partial sums for these rows
                # But wait - these rows might not be owned by this rank!
                # Actually, for forward solve, we're sending the y[update_rows] values
                # which have been accumulated so far

                # The update rows in the frontal matrix become contributions to
                # the parent's frontal matrix via extend-add
                # In the solve, this means we need to send:
                #   sum over k in this supernode of: L[update_row, k] * y[k]

                # Actually, the accumulated y values for update rows need to be
                # computed and sent. But update rows may not be owned by this rank...

                # Let me reconsider: In forward solve, after processing a supernode,
                # the update rows have accumulated contributions from this supernode.
                # These need to be sent to the parent.

                # The complication: update rows may belong to supernodes on other ranks.
                # For now, let's compute the contribution explicitly.
                contribution = zero(T)
                for local_col in local_range
                    elim_col = plan.local_to_global[local_col]
                    yk = y_local[local_col]

                    # Find L[elim_row, elim_col]
                    for idx in nzrange(L_local, elim_col)
                        if rowvals(L_local)[idx] == elim_row
                            contribution += nonzeros(L_local)[idx] * yk
                            break
                        end
                    end
                end
                send_buf[buf_idx] = contribution
            end

            # Send to parent
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

    # Process supernodes in reverse postorder (root to leaves)
    for sidx in reverse(plan.my_supernodes_postorder)
        snode = symbolic.supernodes[sidx]
        info = symbolic.frontal_info[sidx]
        nfs = info.nfs

        # If this is a subtree root, receive solution from parent
        parent = symbolic.snode_parent[sidx]
        if parent != 0 && symbolic.snode_owner[parent] != plan.myrank
            parent_owner = symbolic.snode_owner[parent]
            recv_buf = plan.recv_buffers[(sidx, parent_owner)]

            MPI.Recv!(recv_buf, parent_owner, sidx + 1000000, comm)  # Use different tag

            # Apply received solution to update local x values
            update_rows_elim = plan.subtree_root_update_rows[sidx]
            local_range = plan.local_snode_indices[sidx]

            for (buf_idx, elim_row) in enumerate(update_rows_elim)
                x_update_row = recv_buf[buf_idx]

                # Subtract U[k, elim_row] * x[elim_row] from x[k] for k in this supernode
                for local_col in local_range
                    elim_col = plan.local_to_global[local_col]

                    # Find U[elim_col, elim_row]
                    for idx in nzrange(U_local, elim_row)
                        if rowvals(U_local)[idx] == elim_col
                            x_local[local_col] -= nonzeros(U_local)[idx] * x_update_row
                            break
                        end
                    end
                end
            end
        end

        # Local backward substitution within this supernode (reverse order)
        local_range = plan.local_snode_indices[sidx]
        for local_col in reverse(collect(local_range))
            elim_col = plan.local_to_global[local_col]

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

            # Update rows above
            for idx in nzrange(U_local, elim_col)
                elim_row = rowvals(U_local)[idx]
                if elim_row != elim_col
                    local_row = plan.global_to_local[elim_row]
                    if local_row > 0
                        x_local[local_row] -= nonzeros(U_local)[idx] * xk
                    end
                end
            end
        end

        # If we have children on other ranks, send solution to them
        children = symbolic.snode_children[sidx]
        for child in children
            child_owner = symbolic.snode_owner[child]
            if child_owner != plan.myrank
                send_buf = plan.send_buffers[(child, child_owner)]

                # Send x values for update rows that the child needs
                child_info = symbolic.frontal_info[child]
                child_nfs = child_info.nfs
                child_update_rows = child_info.row_indices[child_nfs+1:end]

                for (buf_idx, global_row) in enumerate(child_update_rows)
                    elim_row = symbolic.global_to_elim[global_row]
                    local_row = plan.global_to_local[elim_row]
                    if local_row > 0
                        send_buf[buf_idx] = x_local[local_row]
                    else
                        send_buf[buf_idx] = zero(T)  # Not owned by this rank
                    end
                end

                MPI.Send(send_buf, child_owner, child + 1000000, comm)
            end
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

Similar to LU forward solve but L comes from LDLT factorization.
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
                MPI.Recv!(recv_buf, child_owner, child, comm)

                child_info = symbolic.frontal_info[child]
                child_nfs = child_info.nfs
                child_update_rows = child_info.row_indices[child_nfs+1:end]

                for (buf_idx, global_row) in enumerate(child_update_rows)
                    elim_row = symbolic.global_to_elim[global_row]
                    local_row = plan.global_to_local[elim_row]
                    if local_row > 0
                        y_local[local_row] += recv_buf[buf_idx]
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

            update_rows_elim = plan.subtree_root_update_rows[sidx]
            for (buf_idx, elim_row) in enumerate(update_rows_elim)
                contribution = zero(T)
                for local_col in local_range
                    elim_col = plan.local_to_global[local_col]
                    yk = y_local[local_col]

                    for idx in nzrange(L_local, elim_col)
                        if rowvals(L_local)[idx] == elim_row
                            contribution += nonzeros(L_local)[idx] * yk
                            break
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

Process supernodes in reverse postorder (root to leaves).
"""
function distributed_backward_solve_ldlt!(x_local::Vector{T},
                                           F::LDLTFactorizationMPI{T},
                                           z_local::Vector{T},
                                           plan::SolvePlan{T}) where T
    comm = MPI.COMM_WORLD
    symbolic = plan.symbolic
    L_local = F.L_local

    copy!(x_local, z_local)

    # Process supernodes in reverse postorder (root to leaves)
    for sidx in reverse(plan.my_supernodes_postorder)
        snode = symbolic.supernodes[sidx]
        info = symbolic.frontal_info[sidx]

        # If this is a subtree root, receive from parent
        parent = symbolic.snode_parent[sidx]
        if parent != 0 && symbolic.snode_owner[parent] != plan.myrank
            parent_owner = symbolic.snode_owner[parent]
            recv_buf = plan.recv_buffers[(sidx, parent_owner)]

            MPI.Recv!(recv_buf, parent_owner, sidx + 1000000, comm)

            # Apply received values: x[k] -= L[update_row, k]^T * x[update_row]
            update_rows_elim = plan.subtree_root_update_rows[sidx]
            local_range = plan.local_snode_indices[sidx]

            for (buf_idx, elim_row) in enumerate(update_rows_elim)
                x_update_row = recv_buf[buf_idx]

                # For L^T, entry L[elim_row, elim_col] becomes L^T[elim_col, elim_row]
                for local_col in local_range
                    elim_col = plan.local_to_global[local_col]

                    for idx in nzrange(L_local, elim_col)
                        if rowvals(L_local)[idx] == elim_row
                            x_local[local_col] -= nonzeros(L_local)[idx] * x_update_row
                            break
                        end
                    end
                end
            end
        end

        # Local backward substitution with L^T (reverse column order)
        local_range = plan.local_snode_indices[sidx]
        for local_col in reverse(collect(local_range))
            elim_col = plan.local_to_global[local_col]

            # For L^T[elim_col, :], we need entries from column elim_col of L
            # L^T[elim_col, j] = L[j, elim_col] for j > elim_col (below diagonal)
            for idx in nzrange(L_local, elim_col)
                elim_row = rowvals(L_local)[idx]
                if elim_row > elim_col  # Below diagonal in L = right of diagonal in L^T
                    local_row = plan.global_to_local[elim_row]
                    if local_row > 0
                        x_local[local_col] -= nonzeros(L_local)[idx] * x_local[local_row]
                    end
                end
            end
            # Unit diagonal - no division needed
        end

        # Send to children on other ranks
        children = symbolic.snode_children[sidx]
        for child in children
            child_owner = symbolic.snode_owner[child]
            if child_owner != plan.myrank
                send_buf = plan.send_buffers[(child, child_owner)]

                child_info = symbolic.frontal_info[child]
                child_nfs = child_info.nfs
                child_update_rows = child_info.row_indices[child_nfs+1:end]

                for (buf_idx, global_row) in enumerate(child_update_rows)
                    elim_row = symbolic.global_to_elim[global_row]
                    local_row = plan.global_to_local[elim_row]
                    if local_row > 0
                        send_buf[buf_idx] = x_local[local_row]
                    else
                        send_buf[buf_idx] = zero(T)
                    end
                end

                MPI.Send(send_buf, child_owner, child + 1000000, comm)
            end
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
