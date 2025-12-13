"""
Distributed matrix input for factorization.

Instead of gathering the entire matrix to all ranks, this module enables
each rank to request only the matrix entries needed for its supernodes'
frontal matrices.

Communication pattern:
1. Each rank computes which entries of A[perm, perm] it needs
2. Convert to original A coordinates: A[perm[i], perm[j]]
3. Exchange entry requests via Alltoall
4. Look up requested entries in local A portion
5. Exchange entry values via point-to-point
"""

using MPI
using SparseArrays

# ============================================================================
# Input Plan Initialization
# ============================================================================

"""
    initialize_input_plan!(plan::FactorizationInputPlan{T},
                           A::SparseMatrixMPI{T}) where T

Initialize the factorization input plan.

Computes:
1. Which matrix entries each supernode needs (from symbolic factorization)
2. Which ranks own those entries (based on A's row partition)
3. Communication patterns for exchanging entries
"""
function initialize_input_plan!(plan::FactorizationInputPlan{T},
                                 A::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD
    plan.myrank = MPI.Comm_rank(comm)
    plan.nranks = MPI.Comm_size(comm)
    plan.row_partition = copy(A.row_partition)

    symbolic = plan.symbolic
    perm = symbolic.perm
    n = symbolic.n

    # Step 1: For each supernode this rank owns, determine needed entries
    # An entry Ap[i,j] is needed if both i and j are in row_indices
    # and (i is in supernode cols OR j is in supernode cols)

    all_needed = Set{Tuple{Int,Int}}()  # All (orig_row, orig_col) pairs needed

    for sidx in 1:length(symbolic.supernodes)
        if symbolic.snode_owner[sidx] == plan.myrank
            info = symbolic.frontal_info[sidx]
            row_indices = info.row_indices  # Permuted coordinates
            nfs = info.nfs

            entries_for_snode = Tuple{Int,Int}[]

            # For frontal matrix: need Ap[row_indices, row_indices]
            # restricted to entries where row or col is in supernode
            for (local_col, perm_col) in enumerate(row_indices)
                for (local_row, perm_row) in enumerate(row_indices)
                    # Only need entry if local_col <= nfs OR local_row <= nfs
                    if local_col <= nfs || local_row <= nfs
                        push!(entries_for_snode, (perm_row, perm_col))
                        # Original A coordinates: A[perm[perm_row], perm[perm_col]]
                        orig_row = perm[perm_row]
                        orig_col = perm[perm_col]
                        push!(all_needed, (orig_row, orig_col))
                    end
                end
            end

            plan.needed_entries[sidx] = entries_for_snode
        end
    end

    # Step 2: Group needed entries by owning rank
    # Owner is determined by row index in original A
    needs_from_rank = Dict{Int, Vector{Tuple{Int,Int}}}()
    for r in 0:plan.nranks-1
        needs_from_rank[r] = Tuple{Int,Int}[]
    end

    for (orig_row, orig_col) in all_needed
        owner = get_row_owner(orig_row, plan.row_partition)
        push!(needs_from_rank[owner], (orig_row, orig_col))
    end

    # Step 3: Exchange request counts via Alltoall
    send_counts = Int32[length(needs_from_rank[r]) for r in 0:plan.nranks-1]
    recv_counts = Vector{Int32}(undef, plan.nranks)
    MPI.Alltoall!(send_counts, recv_counts, 1, comm)

    # Step 4: Exchange index requests via Alltoallv
    # Pack (row, col) pairs as Int32 pairs
    total_send = sum(send_counts)
    total_recv = sum(recv_counts)

    send_data = Vector{Int32}(undef, 2 * total_send)
    idx = 1
    for r in 0:plan.nranks-1
        for (row, col) in needs_from_rank[r]
            send_data[idx] = Int32(row)
            send_data[idx + 1] = Int32(col)
            idx += 2
        end
    end

    recv_data = Vector{Int32}(undef, 2 * total_recv)
    send_counts_2 = Int32.(2 .* send_counts)
    recv_counts_2 = Int32.(2 .* recv_counts)

    MPI.Alltoallv!(MPI.VBuffer(send_data, send_counts_2), MPI.VBuffer(recv_data, recv_counts_2), comm)

    # Step 5: Build send_indices from received requests
    offset = 0
    for r in 0:plan.nranks-1
        count = recv_counts[r + 1]
        if count > 0
            indices = Tuple{Int,Int}[]
            for i in 1:count
                row = Int(recv_data[offset + 2*(i-1) + 1])
                col = Int(recv_data[offset + 2*(i-1) + 2])
                push!(indices, (row, col))
            end
            plan.send_indices[r] = indices
            if !(r in plan.send_to_ranks)
                push!(plan.send_to_ranks, r)
            end
        end
        offset += 2 * count
    end

    # Step 6: Build recv_indices (map back to permuted coordinates)
    inv_perm = symbolic.invperm
    for r in 0:plan.nranks-1
        if length(needs_from_rank[r]) > 0
            # Convert back to permuted coordinates for lookup
            recv_idx = Tuple{Int,Int}[]
            for (orig_row, orig_col) in needs_from_rank[r]
                perm_row = inv_perm[orig_row]
                perm_col = inv_perm[orig_col]
                push!(recv_idx, (perm_row, perm_col))
            end
            plan.recv_indices[r] = recv_idx
            if !(r in plan.recv_from_ranks)
                push!(plan.recv_from_ranks, r)
            end
        end
    end

    # Step 7: Allocate buffers
    for (r, indices) in plan.send_indices
        plan.send_buffers[r] = zeros(T, length(indices))
    end
    for (r, indices) in plan.recv_indices
        plan.recv_buffers[r] = zeros(T, length(indices))
    end

    plan.initialized = true
    return plan
end

"""
    get_row_owner(row::Int, partition::Vector{Int}) -> Int

Get the rank that owns a given row based on the partition.
"""
function get_row_owner(row::Int, partition::Vector{Int})
    nranks = length(partition) - 1
    for r in 0:nranks-1
        if partition[r+1] <= row < partition[r+2]
            return r
        end
    end
    return nranks - 1  # Last rank
end

# ============================================================================
# Execute Input Plan
# ============================================================================

"""
    execute_input_plan!(plan::FactorizationInputPlan{T},
                        A::SparseMatrixMPI{T}) where T

Execute the input plan: exchange matrix entries between ranks.

After this call, `plan.received_values` contains all entries needed
for this rank's supernodes' frontal matrices.
"""
function execute_input_plan!(plan::FactorizationInputPlan{T},
                              A::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD

    # Clear received values
    empty!(plan.received_values)

    # Fill send buffers with values from local A
    AT = A.A.parent  # CSC storage: columns = local rows
    my_row_start = plan.row_partition[plan.myrank + 1]

    for (dest_rank, indices) in plan.send_indices
        buf = plan.send_buffers[dest_rank]
        for (i, (row, col)) in enumerate(indices)
            # Look up A[row, col] in local storage
            # AT has columns = local rows, rows = A.col_indices
            local_row_idx = row - my_row_start + 1

            # Find column 'col' in the local storage
            val = zero(T)
            if 1 <= local_row_idx <= size(AT, 2)
                for ptr in nzrange(AT, local_row_idx)
                    if A.col_indices[rowvals(AT)[ptr]] == col
                        val = nonzeros(AT)[ptr]
                        break
                    end
                end
            end
            buf[i] = val
        end
    end

    # Exchange values via point-to-point communication
    requests = MPI.Request[]

    # Post receives
    for src_rank in plan.recv_from_ranks
        if haskey(plan.recv_buffers, src_rank)
            req = MPI.Irecv!(plan.recv_buffers[src_rank], src_rank, 100 + src_rank, comm)
            push!(requests, req)
        end
    end

    # Post sends
    for dest_rank in plan.send_to_ranks
        if haskey(plan.send_buffers, dest_rank)
            req = MPI.Isend(plan.send_buffers[dest_rank], dest_rank, 100 + plan.myrank, comm)
            push!(requests, req)
        end
    end

    # Wait for all communication to complete
    MPI.Waitall(requests)

    # Store received values
    for (src_rank, indices) in plan.recv_indices
        buf = plan.recv_buffers[src_rank]
        for (i, (perm_row, perm_col)) in enumerate(indices)
            plan.received_values[(perm_row, perm_col)] = buf[i]
        end
    end

    return plan
end

# ============================================================================
# Helper: Get Entry Value
# ============================================================================

"""
    get_permuted_entry(plan::FactorizationInputPlan{T},
                       A::SparseMatrixMPI{T},
                       perm_row::Int, perm_col::Int) where T

Get the value of Ap[perm_row, perm_col] = A[perm[perm_row], perm[perm_col]].

First checks received_values, then falls back to local A if this rank owns it.
"""
function get_permuted_entry(plan::FactorizationInputPlan{T},
                            A::SparseMatrixMPI{T},
                            perm_row::Int, perm_col::Int) where T
    # Check received values first
    if haskey(plan.received_values, (perm_row, perm_col))
        return plan.received_values[(perm_row, perm_col)]
    end

    # Check if we own this entry locally
    perm = plan.symbolic.perm
    orig_row = perm[perm_row]
    orig_col = perm[perm_col]

    my_row_start = plan.row_partition[plan.myrank + 1]
    my_row_end = plan.row_partition[plan.myrank + 2] - 1

    if my_row_start <= orig_row <= my_row_end
        # Look up in local A
        AT = A.A.parent
        local_row_idx = orig_row - my_row_start + 1

        for ptr in nzrange(AT, local_row_idx)
            if A.col_indices[rowvals(AT)[ptr]] == orig_col
                return nonzeros(AT)[ptr]
            end
        end
    end

    return zero(T)
end

# ============================================================================
# Initialize Frontal Matrix from Distributed Input
# ============================================================================

"""
    initialize_frontal_distributed(plan::FactorizationInputPlan{T},
                                    A::SparseMatrixMPI{T},
                                    snode::Supernode,
                                    info::FrontalInfo) where T

Initialize a frontal matrix using distributed matrix input.

Uses the input plan's received values instead of gathering the full matrix.
"""
function initialize_frontal_distributed(plan::FactorizationInputPlan{T},
                                        A::SparseMatrixMPI{T},
                                        snode::Supernode,
                                        info::FrontalInfo) where T
    row_indices = info.row_indices
    nfs = info.nfs
    nrows = length(row_indices)

    F = FrontalMatrix{T}(copy(row_indices), copy(row_indices), nfs)

    # Fill frontal matrix from distributed input
    for (local_col, perm_col) in enumerate(row_indices)
        for (local_row, perm_row) in enumerate(row_indices)
            # Only fill if local_col <= nfs OR local_row <= nfs
            if local_col <= nfs || local_row <= nfs
                val = get_permuted_entry(plan, A, perm_row, perm_col)
                F.F[local_row, local_col] = val
            end
        end
    end

    return F
end

"""
    initialize_frontal_sym_distributed(plan::FactorizationInputPlan{T},
                                        A::SparseMatrixMPI{T},
                                        snode::Supernode,
                                        info::FrontalInfo) where T

Initialize a symmetric frontal matrix using distributed matrix input.
"""
function initialize_frontal_sym_distributed(plan::FactorizationInputPlan{T},
                                            A::SparseMatrixMPI{T},
                                            snode::Supernode,
                                            info::FrontalInfo) where T
    row_indices = info.row_indices
    nfs = info.nfs
    nrows = length(row_indices)

    F = FrontalMatrix{T}(copy(row_indices), copy(row_indices), nfs)

    # Fill frontal matrix from distributed input (symmetric)
    for (local_col, perm_col) in enumerate(row_indices)
        for (local_row, perm_row) in enumerate(row_indices)
            if local_col <= nfs || local_row <= nfs
                val = get_permuted_entry(plan, A, perm_row, perm_col)
                F.F[local_row, local_col] = val
                if local_row != local_col
                    F.F[local_col, local_row] = val  # Symmetric
                end
            end
        end
    end

    return F
end

# ============================================================================
# Cache for Input Plans
# ============================================================================

const _input_plan_cache = Dict{Tuple{Blake3Hash, Blake3Hash, DataType}, Any}()

"""
    get_or_create_input_plan(symbolic::SymbolicFactorization,
                              A::SparseMatrixMPI{T}) where T

Get or create a factorization input plan for the given symbolic factorization
and matrix structure.
"""
function get_or_create_input_plan(symbolic::SymbolicFactorization,
                                   A::SparseMatrixMPI{T}) where T
    sym_hash = symbolic.structural_hash
    a_hash = A.structural_hash === nothing ? compute_structural_hash(A.row_partition, A.col_indices, A.A.parent, MPI.COMM_WORLD) : A.structural_hash
    key = (sym_hash, a_hash, T)

    if haskey(_input_plan_cache, key)
        return _input_plan_cache[key]::FactorizationInputPlan{T}
    end

    plan = FactorizationInputPlan{T}(symbolic)
    initialize_input_plan!(plan, A)
    _input_plan_cache[key] = plan
    return plan
end

"""
    clear_input_plan_cache!()

Clear the factorization input plan cache.
"""
function clear_input_plan_cache!()
    empty!(_input_plan_cache)
end
