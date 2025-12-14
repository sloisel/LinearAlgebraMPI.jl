"""
Utility functions for multifrontal factorization.

These functions operate on local frontal matrices and handle:
- Extend-add operations for assembling child contributions
- Partial factorization with pivoting
- Extraction of L/U and L/D factors

Note: Frontal matrix initialization is done via distributed input plans
in factorization_input.jl (initialize_frontal_distributed, initialize_frontal_sym_distributed).
"""

# Bunch-Kaufman parameter: alpha = (1 + sqrt(17)) / 8
const BK_ALPHA = (1.0 + sqrt(17.0)) / 8.0

# ============================================================================
# LU Factorization Utilities
# ============================================================================

"""
    extend_add!(F, update, child_rows, child_cols)

Assemble (extend-add) an update matrix from a child into frontal matrix F.

For LU factorization with partial pivoting, child_rows and child_cols may differ
because row pivoting permutes row_indices but not col_indices.
"""
function extend_add!(F::FrontalMatrix{T},
                     update::Matrix{T},
                     child_rows::Vector{Int},
                     child_cols::Vector{Int}) where T
    # Create mappings from global indices to parent local positions
    parent_row_map = Dict{Int, Int}()
    for (local_idx, global_idx) in enumerate(F.row_indices)
        parent_row_map[global_idx] = local_idx
    end

    parent_col_map = Dict{Int, Int}()
    for (local_idx, global_idx) in enumerate(F.col_indices)
        parent_col_map[global_idx] = local_idx
    end

    # Scatter-add
    nchild_rows = size(update, 1)
    nchild_cols = size(update, 2)
    for cj = 1:nchild_cols
        global_col = child_cols[cj]
        if !haskey(parent_col_map, global_col)
            continue
        end
        pj = parent_col_map[global_col]

        for ci = 1:nchild_rows
            global_row = child_rows[ci]
            if !haskey(parent_row_map, global_row)
                continue
            end
            pi = parent_row_map[global_row]

            F.F[pi, pj] += update[ci, cj]
        end
    end
end

"""
    partial_factor!(F) -> pivots

Factor the fully summed part of the frontal matrix with partial pivoting.

IMPORTANT: In multifrontal context, pivoting is restricted to the fully summed block
(rows 1:nfs) to avoid complex interactions with update rows that belong to
ancestor supernodes. This matches the approach used in LDLT factorization.

Returns the pivot indices for each fully summed column.
"""
function partial_factor!(F::FrontalMatrix{T}) where T
    nfs = F.nfs
    nrows = size(F.F, 1)
    ncols = size(F.F, 2)
    pivots = collect(1:nfs)

    for k = 1:nfs
        # Partial pivoting: find max in column k, restricted to rows k:nfs
        # (not k:nrows, to avoid swapping with update rows)
        piv_row = k
        piv_val = abs(F.F[k, k])
        for i = k+1:nfs
            if abs(F.F[i, k]) > piv_val
                piv_row = i
                piv_val = abs(F.F[i, k])
            end
        end

        pivots[k] = piv_row

        # Swap rows if needed
        if piv_row != k
            for j = 1:ncols  # Swap entire row
                F.F[k, j], F.F[piv_row, j] = F.F[piv_row, j], F.F[k, j]
            end
            F.row_indices[k], F.row_indices[piv_row] =
                F.row_indices[piv_row], F.row_indices[k]
        end

        # Check for numerical singularity
        diag_val = F.F[k, k]
        if abs(diag_val) < eps(real(T)) * 1000
            @warn "Small pivot at column $k: |pivot| = $(abs(diag_val))"
            if abs(diag_val) == 0
                diag_val = eps(real(T))
                F.F[k, k] = diag_val
            end
        end

        # Scale column k below diagonal (compute L column)
        for i = k+1:nrows
            F.F[i, k] /= diag_val
        end

        # Rank-1 updates with deferred Schur complement:
        # 1. Update fully summed rows (k+1:nfs) - full width for correct U values
        # 2. Update rows for L columns (nfs+1:nrows, k+1:nfs)
        # 3. Defer Schur complement (nfs+1:nrows, nfs+1:ncols) for batched gemm
        if k < nfs
            # Fully summed rows: need full width for correct U row values
            BLAS.ger!(-one(T), view(F.F, k+1:nfs, k), view(F.F, k, k+1:ncols), view(F.F, k+1:nfs, k+1:ncols))
            # Update rows: only L columns (defer Schur complement)
            if nfs < nrows
                BLAS.ger!(-one(T), view(F.F, nfs+1:nrows, k), view(F.F, k, k+1:nfs), view(F.F, nfs+1:nrows, k+1:nfs))
            end
        end
    end

    # Batched Schur complement update using BLAS Level-3
    # F[nfs+1:nrows, nfs+1:ncols] -= L[nfs+1:nrows, 1:nfs] * U[1:nfs, nfs+1:ncols]
    if nfs < nrows && nfs < ncols
        BLAS.gemm!('N', 'N', -one(T), view(F.F, nfs+1:nrows, 1:nfs), view(F.F, 1:nfs, nfs+1:ncols), one(T), view(F.F, nfs+1:nrows, nfs+1:ncols))
    end

    F.pivots = pivots
    return pivots
end

"""
    extract_LU!(F, snode, global_to_elim, L_I, L_J, L_V, U_I, U_J, U_V)

Extract L and U entries from the factored frontal matrix using ELIMINATION STEP indices.

For L and U to be triangular in the elimination step index space:
- Columns use elimination step: elim_col = global_to_elim[global_col]
- Supernode rows (positions 1 to nfs after pivoting) use: elim_row = base_elim + local_row - 1
  because they become the pivot rows at those elimination steps
- Update rows (positions > nfs) use: elim_row = global_to_elim[global_row]
  but only create L entries if elim_row > elim_col (row not yet eliminated)
"""
function extract_LU!(F::FrontalMatrix{T},
                     snode::Supernode,
                     global_to_elim::Vector{Int},
                     L_I::Vector{Int}, L_J::Vector{Int}, L_V::Vector{T},
                     U_I::Vector{Int}, U_J::Vector{Int}, U_V::Vector{T}) where T
    nfs = F.nfs
    nrows = size(F.F, 1)

    # Base elimination step for this supernode
    base_elim = global_to_elim[F.col_indices[1]]

    for local_col = 1:nfs
        # Column elimination step
        elim_col = base_elim + local_col - 1

        # L column: entries at and below diagonal in LOCAL coordinates
        for local_row = local_col:nrows
            global_row = F.row_indices[local_row]

            # Determine row elimination step based on position
            if local_row <= nfs
                # Supernode row: this row becomes the pivot at this elimination step
                elim_row = base_elim + local_row - 1
            else
                # Update row: will be eliminated when its column is eliminated
                elim_row = global_to_elim[global_row]
                # Only create L entry if row not yet eliminated
                if elim_row <= elim_col
                    continue  # Skip: row was already eliminated
                end
            end

            if local_row == local_col
                # Diagonal: L gets 1 (unit diagonal)
                push!(L_I, elim_row)
                push!(L_J, elim_col)
                push!(L_V, one(T))
            else
                # Below diagonal in local coords
                val = F.F[local_row, local_col]
                if abs(val) > eps(real(T)) * 100
                    push!(L_I, elim_row)
                    push!(L_J, elim_col)
                    push!(L_V, val)
                end
            end
        end

        # U row: entries at and to the right of diagonal in ELIMINATION ORDER
        # The pivot row for this column is at position local_col
        # We need to extract U entries for ALL columns where elim_col >= elim_row,
        # including update columns beyond this supernode
        elim_row_for_U = base_elim + local_col - 1
        ncols = size(F.F, 2)

        for local_col2 = local_col:ncols
            global_col2 = F.col_indices[local_col2]
            elim_col2 = global_to_elim[global_col2]
            val = F.F[local_col, local_col2]

            # Only extract if elim_col2 >= elim_row_for_U (upper triangular in elim order)
            if elim_col2 >= elim_row_for_U
                if abs(val) > eps(real(T)) * 100 || elim_col2 == elim_row_for_U
                    push!(U_I, elim_row_for_U)
                    push!(U_J, elim_col2)
                    push!(U_V, val)
                end
            end
        end
    end
end

# ============================================================================
# LDLT Factorization Utilities (Symmetric with Bunch-Kaufman pivoting)
# ============================================================================

"""
    extend_add_sym!(F, update, child_rows)

Assemble (extend-add) an update matrix from a child into frontal matrix F.
The update is symmetric, so we only need to add to lower triangle and mirror.
"""
function extend_add_sym!(F::FrontalMatrix{T},
                         update::Matrix{T},
                         child_rows::Vector{Int}) where T
    # Create mapping from child rows to parent rows
    parent_row_map = Dict{Int, Int}()
    for (local_idx, global_idx) in enumerate(F.row_indices)
        parent_row_map[global_idx] = local_idx
    end

    # Scatter-add (symmetric)
    nchild = size(update, 1)
    for cj = 1:nchild
        global_col = child_rows[cj]
        if !haskey(parent_row_map, global_col)
            continue
        end
        pj = parent_row_map[global_col]

        for ci = cj:nchild  # Only lower triangle
            global_row = child_rows[ci]
            if !haskey(parent_row_map, global_row)
                continue
            end
            pi = parent_row_map[global_row]

            F.F[pi, pj] += update[ci, cj]
            if pi != pj
                F.F[pj, pi] += update[ci, cj]  # Symmetric
            end
        end
    end
end

"""
    partial_factor_ldlt!(F) -> (D_local, pivot_info)

Factor the fully summed part of the frontal matrix using LDLT with Bunch-Kaufman pivoting.

Returns:
- D_local: Vector of diagonal values (length nfs, may contain 2×2 block entries)
- pivot_info: Vector indicating pivot type (>0 for 1×1, <0 for 2×2 block start)
"""
function partial_factor_ldlt!(F::FrontalMatrix{T}) where T
    nfs = F.nfs
    nrows = size(F.F, 1)

    # D storage: for 1×1 pivot at k, D_local[k] = d_kk
    # For 2×2 pivot at (k,k+1), D_local stores the 2×2 block entries
    D_local = Vector{Any}(undef, nfs)
    pivot_info = zeros(Int, nfs)

    k = 1
    while k <= nfs
        # Bunch-Kaufman pivot selection
        pivot_type, pivot_row = bk_pivot_selection(F.F, k, nfs, nrows)

        if pivot_type == 1
            # 1×1 pivot at position k
            pivot_info[k] = 1

            # Swap rows/cols if needed
            if pivot_row != k
                swap_rows_cols_sym!(F, k, pivot_row, nrows)
            end

            d_kk = F.F[k, k]
            D_local[k] = d_kk

            # Check for numerical issues
            if abs(d_kk) < eps(real(T)) * 1000
                @warn "Small pivot at column $k: |pivot| = $(abs(d_kk))"
                if abs(d_kk) == 0
                    d_kk = eps(real(T))
                    D_local[k] = d_kk
                    F.F[k, k] = d_kk
                end
            end

            # Scale column k below diagonal (compute L column)
            for i = k+1:nrows
                F.F[i, k] /= d_kk
            end

            # Symmetric Schur complement update with batched deferred update:
            # 1. Update fully summed block (k+1:nfs) - full update for pivot selection
            # 2. Update L columns for update rows (nfs+1:nrows, k+1:nfs)
            # 3. Defer Schur complement (nfs+1:nrows, nfs+1:nrows) for batched gemm
            if k < nfs
                # Update fully summed block (lower triangle)
                BLAS.syr!('L', -d_kk, view(F.F, k+1:nfs, k), view(F.F, k+1:nfs, k+1:nfs))
                # Mirror to upper for subsequent iterations within fully summed block
                for j = k+1:nfs
                    for i = j+1:nfs
                        F.F[j, i] = F.F[i, j]
                    end
                end
                # Update L columns for update rows (lower triangle only, cols k+1:nfs)
                if nfs < nrows
                    for j = k+1:nfs
                        ljk = F.F[j, k]
                        ljk_dk = ljk * d_kk
                        for i = nfs+1:nrows
                            F.F[i, j] -= F.F[i, k] * ljk_dk
                        end
                    end
                end
            end
            # DON'T update F[nfs+1:nrows, nfs+1:nrows] - batched at the end

            k += 1

        else
            # 2×2 pivot at positions (k, k+1)
            pivot_info[k] = -1
            pivot_info[k+1] = -1

            # Swap row/col pivot_row with k+1 if needed
            if pivot_row != k + 1
                swap_rows_cols_sym!(F, k + 1, pivot_row, nrows)
            end

            # Extract 2×2 diagonal block
            d_kk = F.F[k, k]
            d_k1k = F.F[k+1, k]
            d_k1k1 = F.F[k+1, k+1]

            D_local[k] = (d_kk, d_k1k, d_k1k1)  # Store as tuple
            D_local[k+1] = nothing  # Marker for second element of 2×2 block

            # Compute inverse of 2×2 block: [d_kk d_k1k; d_k1k d_k1k1]^(-1)
            det = d_kk * d_k1k1 - d_k1k * d_k1k
            if abs(det) < eps(real(T)) * 1000
                @warn "Small 2×2 pivot determinant at columns $k:$(k+1): |det| = $(abs(det))"
                det = sign(det) * max(abs(det), eps(real(T)))
            end

            inv_d_kk = d_k1k1 / det
            inv_d_k1k = -d_k1k / det
            inv_d_k1k1 = d_kk / det

            # Compute L block: L[i, k:k+1] = F[i, k:k+1] * inv(D_block) for i > k+1
            for i = k+2:nrows
                f_ik = F.F[i, k]
                f_ik1 = F.F[i, k+1]

                F.F[i, k] = f_ik * inv_d_kk + f_ik1 * inv_d_k1k
                F.F[i, k+1] = f_ik * inv_d_k1k + f_ik1 * inv_d_k1k1
            end

            # Symmetric Schur complement update with batched deferred update
            if k + 2 <= nfs
                # Update fully summed block
                for j = k+2:nfs
                    ljk = F.F[j, k]
                    ljk1 = F.F[j, k+1]
                    dl_j0 = d_kk * ljk + d_k1k * ljk1
                    dl_j1 = d_k1k * ljk + d_k1k1 * ljk1
                    for i = j:nfs
                        lik = F.F[i, k]
                        lik1 = F.F[i, k+1]
                        F.F[i, j] -= lik * dl_j0 + lik1 * dl_j1
                    end
                end
                # Mirror to upper for subsequent iterations
                for j = k+2:nfs
                    for i = j+1:nfs
                        F.F[j, i] = F.F[i, j]
                    end
                end
                # Update L columns for update rows
                if nfs < nrows
                    for j = k+2:nfs
                        ljk = F.F[j, k]
                        ljk1 = F.F[j, k+1]
                        dl_j0 = d_kk * ljk + d_k1k * ljk1
                        dl_j1 = d_k1k * ljk + d_k1k1 * ljk1
                        for i = nfs+1:nrows
                            lik = F.F[i, k]
                            lik1 = F.F[i, k+1]
                            F.F[i, j] -= lik * dl_j0 + lik1 * dl_j1
                        end
                    end
                end
            end
            # DON'T update F[nfs+1:nrows, nfs+1:nrows] - batched at the end

            k += 2
        end
    end

    # Batched Schur complement update: S -= L * D * L'
    # where L = F[nfs+1:nrows, 1:nfs], D = block diagonal from D_local
    if nfs < nrows
        m = nrows - nfs
        # Compute LD = L * D (column by column, accounting for D structure)
        LD = Matrix{T}(undef, m, nfs)
        j = 1
        while j <= nfs
            if pivot_info[j] > 0
                # 1×1 pivot: LD[:, j] = L[:, j] * D[j]
                d_j = T(D_local[j])
                for i = 1:m
                    LD[i, j] = F.F[nfs + i, j] * d_j
                end
                j += 1
            else
                # 2×2 pivot: LD[:, j:j+1] = L[:, j:j+1] * D_block
                d_kk, d_k1k, d_k1k1 = D_local[j]
                for i = 1:m
                    l_ij = F.F[nfs + i, j]
                    l_ij1 = F.F[nfs + i, j + 1]
                    LD[i, j] = l_ij * d_kk + l_ij1 * d_k1k
                    LD[i, j + 1] = l_ij * d_k1k + l_ij1 * d_k1k1
                end
                j += 2
            end
        end
        # S -= LD * L' using BLAS Level-3
        L = view(F.F, nfs+1:nrows, 1:nfs)
        S = view(F.F, nfs+1:nrows, nfs+1:nrows)
        BLAS.gemm!('N', 'T', -one(T), LD, L, one(T), S)
        # Mirror lower to upper
        for j = 1:m
            for i = j+1:m
                S[j, i] = S[i, j]
            end
        end
    end

    return D_local, pivot_info
end

"""
    bk_pivot_selection(F, k, nfs, nrows) -> (pivot_type, pivot_row)

Bunch-Kaufman pivot selection for symmetric indefinite factorization.

IMPORTANT: In multifrontal context, we restrict pivoting to the fully summed block
(rows/cols 1:nfs) to avoid complex interactions with update rows that belong to
ancestor supernodes. This is sometimes called "restricted" or "local" pivoting.

Returns:
- pivot_type: 1 for 1×1 pivot, 2 for 2×2 pivot
- pivot_row: Row to swap with k (for 1×1) or k+1 (for 2×2)
"""
function bk_pivot_selection(F::Matrix{T}, k::Int, nfs::Int, nrows::Int) where T
    # alpha is always real (it's a threshold for comparing magnitudes)
    alpha = real(T)(BK_ALPHA)

    # For multifrontal, restrict search to fully summed block (1:nfs)
    # This avoids swapping with update rows that belong to ancestor supernodes
    search_limit = nfs

    # Find lambda = max|a_ik| for i in (k+1):nfs (only fully summed rows)
    lambda = zero(real(T))
    r = k
    for i = k+1:search_limit
        val = abs(F[i, k])
        if val > lambda
            lambda = val
            r = i
        end
    end

    a_kk = abs(F[k, k])

    # Case 1: diagonal element is large enough
    if a_kk >= alpha * lambda
        return (1, k)  # 1×1 pivot, no swap needed
    end

    # If lambda is zero, use 1×1 pivot (matrix might be singular)
    if lambda == zero(real(T))
        return (1, k)
    end

    # Find sigma = max|a_ir| for i != r within fully summed block
    sigma = zero(real(T))
    for i = k:search_limit
        if i != r
            # Since matrix is symmetric, use lower triangle
            if i > r
                val = abs(F[i, r])
            else
                val = abs(F[r, i])
            end
            if val > sigma
                sigma = val
            end
        end
    end

    a_rr = abs(F[r, r])

    # Case 2: use 1×1 pivot at k
    if a_kk * sigma >= alpha * lambda^2
        return (1, k)
    end

    # Case 3: use 1×1 pivot at r (swap r to k)
    if a_rr >= alpha * sigma
        return (1, r)
    end

    # Case 4: use 2×2 pivot at (k, r)
    # Need to swap r to position k+1
    # Note: r is guaranteed to be <= nfs due to search_limit
    return (2, r)
end

"""
    swap_rows_cols_sym!(F, i, j, nrows)

Swap rows i and j, and columns i and j (symmetric swap) in the frontal matrix.
Also updates row_indices tracking.
"""
function swap_rows_cols_sym!(F::FrontalMatrix{T}, i::Int, j::Int, nrows::Int) where T
    if i == j
        return
    end

    # Swap rows
    for col = 1:nrows
        F.F[i, col], F.F[j, col] = F.F[j, col], F.F[i, col]
    end

    # Swap columns (for symmetric matrix)
    for row = 1:nrows
        F.F[row, i], F.F[row, j] = F.F[row, j], F.F[row, i]
    end

    # Update row_indices tracking
    F.row_indices[i], F.row_indices[j] = F.row_indices[j], F.row_indices[i]

    # col_indices should also be updated for symmetric case
    F.col_indices[i], F.col_indices[j] = F.col_indices[j], F.col_indices[i]
end

"""
    extract_L_D!(F, snode, global_to_elim, L_I, L_J, L_V, D_I, D_J, D_V, D_local, pivot_info)

Extract L and D entries from the factored frontal matrix.
"""
function extract_L_D!(F::FrontalMatrix{T},
                      snode::Supernode,
                      global_to_elim::Vector{Int},
                      L_I::Vector{Int}, L_J::Vector{Int}, L_V::Vector{T},
                      D_I::Vector{Int}, D_J::Vector{Int}, D_V::Vector{T},
                      D_local::Vector{Any}, pivot_info::Vector{Int}) where T
    nfs = F.nfs
    nrows = size(F.F, 1)

    # Use the first column of the supernode (stable, not affected by pivoting)
    base_elim = global_to_elim[first(snode.cols)]

    k = 1
    while k <= nfs
        elim_col = base_elim + k - 1

        if pivot_info[k] > 0
            # 1×1 pivot
            # D diagonal entry
            push!(D_I, elim_col)
            push!(D_J, elim_col)
            push!(D_V, T(D_local[k]))

            # L column: unit diagonal
            push!(L_I, elim_col)
            push!(L_J, elim_col)
            push!(L_V, one(T))

            # L column: entries below diagonal
            for local_row = k+1:nrows
                global_row = F.row_indices[local_row]

                if local_row <= nfs
                    elim_row = base_elim + local_row - 1
                else
                    elim_row = global_to_elim[global_row]
                    if elim_row <= elim_col
                        continue
                    end
                end

                val = F.F[local_row, k]
                if abs(val) > eps(real(T)) * 100
                    push!(L_I, elim_row)
                    push!(L_J, elim_col)
                    push!(L_V, val)
                end
            end

            k += 1

        else
            # 2×2 pivot
            elim_col2 = base_elim + k

            # D 2×2 block entries
            d_kk, d_k1k, d_k1k1 = D_local[k]

            push!(D_I, elim_col)
            push!(D_J, elim_col)
            push!(D_V, T(d_kk))

            push!(D_I, elim_col2)
            push!(D_J, elim_col)
            push!(D_V, T(d_k1k))

            push!(D_I, elim_col)
            push!(D_J, elim_col2)
            push!(D_V, T(d_k1k))

            push!(D_I, elim_col2)
            push!(D_J, elim_col2)
            push!(D_V, T(d_k1k1))

            # L columns: unit diagonal for both columns
            push!(L_I, elim_col)
            push!(L_J, elim_col)
            push!(L_V, one(T))

            push!(L_I, elim_col2)
            push!(L_J, elim_col2)
            push!(L_V, one(T))

            # L entries below the 2×2 block
            for local_row = k+2:nrows
                global_row = F.row_indices[local_row]

                if local_row <= nfs
                    elim_row = base_elim + local_row - 1
                else
                    elim_row = global_to_elim[global_row]
                    if elim_row <= elim_col2
                        continue
                    end
                end

                val1 = F.F[local_row, k]
                val2 = F.F[local_row, k+1]

                if abs(val1) > eps(real(T)) * 100
                    push!(L_I, elim_row)
                    push!(L_J, elim_col)
                    push!(L_V, val1)
                end

                if abs(val2) > eps(real(T)) * 100
                    push!(L_I, elim_row)
                    push!(L_J, elim_col2)
                    push!(L_V, val2)
                end
            end

            k += 2
        end
    end
end
