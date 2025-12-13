"""
Distributed solve routines for LU factorization.

The factorization gives: L * U = Pr_elim * Ap_elim where:
- Ap = P' * A * P is the AMD-reordered matrix (P is fill-reducing permutation)
- Ap_elim = Ap reordered to elimination order via elim_to_global
- Pr_elim is the row permutation from pivoting, in elimination order indices
- L and U are stored in elimination order indices (1 to n)

To solve A * x = b:
1. Apply AMD permutation: bp = P' * b = b[perm]
2. Reorder to elimination order: b_elim[k] = bp[elim_to_global[k]]
3. Apply pivot permutation: c[k] = b_elim[row_perm_elim[k]] where row_perm_elim maps elim indices
4. Forward solve: L * y = c (both in elimination order)
5. Backward solve: U * z = y (both in elimination order)
6. Reorder from elimination to global: zp[elim_to_global[k]] = z[k]
7. Apply inverse AMD permutation: x = P * zp = zp[invperm]
"""

using MPI
using SparseArrays

"""
    solve(F::LUFactorizationMPI{T}, b::VectorMPI{T}) -> VectorMPI{T}

Solve the linear system A*x = b using the precomputed LU factorization.
"""
function solve(F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    x = VectorMPI(zeros(T, F.symbolic.n); partition=b.partition)
    solve!(x, F, b)
    return x
end

"""
    solve!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T}; distributed::Bool=true)

Solve A*x = b in-place using LU factorization.

By default uses distributed solve (MUMPS-style) that keeps factors distributed
and only communicates at subtree boundaries. Set `distributed=false` to use the
gathered solve which collects L/U to all ranks (useful for debugging).
"""
function solve!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T}; distributed::Bool=true) where T
    if distributed
        distributed_solve_lu!(x, F, b)
    else
        solve_gathered!(x, F, b)
    end
    return x
end

"""
    solve_gathered!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T})

Solve A*x = b in-place using gathered L/U factors (all ranks have full factors).

This is the original sequential solve. Useful for debugging and verification.

L and U are stored in elimination order. The solve works entirely in elimination order:
1. Transform RHS to elimination order with pivoting
2. Triangular solves in elimination order
3. Transform solution back to original order
"""
function solve_gathered!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n = F.symbolic.n

    # Gather b to all ranks for the solve
    b_full = Vector(b)

    # Step 1: Apply AMD permutation: bp[i] = b[perm[i]]
    bp = Vector{T}(undef, n)
    for i = 1:n
        bp[i] = b_full[F.symbolic.perm[i]]
    end

    # Step 2: Reorder to elimination order: b_elim[k] = bp[elim_to_global[k]]
    b_elim = Vector{T}(undef, n)
    for k = 1:n
        b_elim[k] = bp[F.symbolic.elim_to_global[k]]
    end

    # Step 3: Apply pivot permutation in elimination order
    # row_perm[k] is the global row that became pivot at elimination step k
    # We need to map this to elimination order
    c = Vector{T}(undef, n)
    for k = 1:n
        # row_perm[k] is in global (AMD) space, convert to elimination space
        src_elim = F.symbolic.global_to_elim[F.row_perm[k]]
        c[k] = b_elim[src_elim]
    end

    # Gather L and U to all ranks for the solve
    L_full, U_full = gather_L_U(F)

    # Step 4: Forward solve in elimination order: L * y = c
    # L is indexed by elimination order, so we iterate k=1..n
    y = copy(c)
    for k = 1:n
        yk = y[k]
        if yk != zero(T)
            for idx in nzrange(L_full, k)
                row = rowvals(L_full)[idx]
                if row != k
                    y[row] -= nonzeros(L_full)[idx] * yk
                end
            end
        end
    end

    # Step 5: Backward solve in elimination order: U * z = y
    z = copy(y)
    for k = n:-1:1
        # Find diagonal
        diag_val = zero(T)
        for idx in nzrange(U_full, k)
            if rowvals(U_full)[idx] == k
                diag_val = nonzeros(U_full)[idx]
                break
            end
        end
        if diag_val == zero(T)
            error("Zero diagonal in U at elimination step $k")
        end
        z[k] /= diag_val
        zk = z[k]
        for idx in nzrange(U_full, k)
            row = rowvals(U_full)[idx]
            if row != k
                z[row] -= nonzeros(U_full)[idx] * zk
            end
        end
    end

    # Step 6: Reorder from elimination to global (AMD) order
    zp = Vector{T}(undef, n)
    for k = 1:n
        zp[F.symbolic.elim_to_global[k]] = z[k]
    end

    # Step 7: Apply inverse AMD permutation
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

"""
    Base.:\\(F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T

Solve A*x = b using the backslash operator.
"""
function Base.:\(F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    return solve(F, b)
end

# ============================================================================
# Transpose and Adjoint Solves
# ============================================================================

"""
    solve_transpose(F::LUFactorizationMPI{T}, b::VectorMPI{T}) -> VectorMPI{T}

Solve transpose(A)*x = b using the precomputed LU factorization of A.
"""
function solve_transpose(F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    x = VectorMPI(zeros(T, F.symbolic.n); partition=b.partition)
    solve_transpose!(x, F, b)
    return x
end

"""
    solve_transpose!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T})

Solve transpose(A)*x = b in-place using LU factorization of A.

For transpose solve, we solve U' * L' * Pr * x_elim = b_elim.
This means: forward solve U' y = b_elim, backward solve L' z = y,
then apply inverse pivot permutation to z to get x_elim.
"""
function solve_transpose!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n = F.symbolic.n

    b_full = Vector(b)

    # Step 1: Apply AMD permutation: bp[i] = b[perm[i]]
    bp = Vector{T}(undef, n)
    for i = 1:n
        bp[i] = b_full[F.symbolic.perm[i]]
    end

    # Step 2: Reorder to elimination order (NO pivot permutation for transpose solve)
    b_elim = Vector{T}(undef, n)
    for k = 1:n
        b_elim[k] = bp[F.symbolic.elim_to_global[k]]
    end

    # Gather L and U
    L_full, U_full = gather_L_U(F)

    # Step 3: Forward solve with U' (lower triangular in elim order): U' y = b_elim
    y = copy(b_elim)
    for k = 1:n
        # Divide by diagonal
        diag_val = zero(T)
        for idx in nzrange(U_full, k)
            if rowvals(U_full)[idx] == k
                diag_val = nonzeros(U_full)[idx]
                break
            end
        end
        if diag_val == zero(T)
            error("Zero diagonal in U at elimination step $k")
        end
        y[k] /= diag_val
        yk = y[k]
        # Update: for U', entry U[k,j] affects y[j] for j > k
        for j = k+1:n
            for idx in nzrange(U_full, j)
                if rowvals(U_full)[idx] == k
                    y[j] -= nonzeros(U_full)[idx] * yk
                    break
                end
            end
        end
    end

    # Step 4: Backward solve with L' (upper triangular, unit diagonal): L' z = y
    z = copy(y)
    for k = n:-1:1
        zk = z[k]
        # Update: for L', entry L[k,j] affects z[j] for j < k
        for j = 1:k-1
            for idx in nzrange(L_full, j)
                if rowvals(L_full)[idx] == k
                    z[j] -= nonzeros(L_full)[idx] * zk
                    break
                end
            end
        end
    end

    # Step 5: Apply INVERSE pivot permutation to solution
    # Forward pivot was: c[k] = b_elim[global_to_elim[row_perm[k]]]
    # Inverse: x_elim[global_to_elim[row_perm[k]]] = z[k]
    x_elim = Vector{T}(undef, n)
    for k = 1:n
        dest = F.symbolic.global_to_elim[F.row_perm[k]]
        x_elim[dest] = z[k]
    end

    # Step 6: Reorder from elimination to global (AMD) order
    xp = Vector{T}(undef, n)
    for k = 1:n
        xp[F.symbolic.elim_to_global[k]] = x_elim[k]
    end

    # Step 7: Apply inverse AMD permutation
    result = Vector{T}(undef, n)
    for i = 1:n
        result[F.symbolic.perm[i]] = xp[i]
    end

    # Distribute result
    my_start = x.partition[rank + 1]
    my_end = x.partition[rank + 2] - 1
    for i = my_start:my_end
        x.v[i - my_start + 1] = result[i]
    end

    return x
end

"""
    solve_adjoint(F::LUFactorizationMPI{T}, b::VectorMPI{T}) -> VectorMPI{T}

Solve A'*x = b (adjoint/conjugate transpose) using the precomputed LU factorization of A.
"""
function solve_adjoint(F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    x = VectorMPI(zeros(T, F.symbolic.n); partition=b.partition)
    solve_adjoint!(x, F, b)
    return x
end

"""
    solve_adjoint!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T})

Solve A'*x = b (adjoint/conjugate transpose) in-place using LU factorization of A.

Same as transpose solve but with conjugation of L and U values.
"""
function solve_adjoint!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n = F.symbolic.n

    b_full = Vector(b)

    # Step 1: Apply AMD permutation: bp[i] = b[perm[i]]
    bp = Vector{T}(undef, n)
    for i = 1:n
        bp[i] = b_full[F.symbolic.perm[i]]
    end

    # Step 2: Reorder to elimination order (NO pivot permutation for adjoint solve)
    b_elim = Vector{T}(undef, n)
    for k = 1:n
        b_elim[k] = bp[F.symbolic.elim_to_global[k]]
    end

    # Gather L and U
    L_full, U_full = gather_L_U(F)

    # Step 3: Forward solve with U' (lower triangular in elim order, conjugated): U' y = b_elim
    y = copy(b_elim)
    for k = 1:n
        # Divide by conjugate of diagonal
        diag_val = zero(T)
        for idx in nzrange(U_full, k)
            if rowvals(U_full)[idx] == k
                diag_val = conj(nonzeros(U_full)[idx])
                break
            end
        end
        if diag_val == zero(T)
            error("Zero diagonal in U at elimination step $k")
        end
        y[k] /= diag_val
        yk = y[k]
        # Update: for U', entry conj(U[k,j]) affects y[j] for j > k
        for j = k+1:n
            for idx in nzrange(U_full, j)
                if rowvals(U_full)[idx] == k
                    y[j] -= conj(nonzeros(U_full)[idx]) * yk
                    break
                end
            end
        end
    end

    # Step 4: Backward solve with L' (upper triangular, unit diagonal, conjugated): L' z = y
    z = copy(y)
    for k = n:-1:1
        zk = z[k]
        # Update: for L', entry conj(L[k,j]) affects z[j] for j < k
        for j = 1:k-1
            for idx in nzrange(L_full, j)
                if rowvals(L_full)[idx] == k
                    z[j] -= conj(nonzeros(L_full)[idx]) * zk
                    break
                end
            end
        end
    end

    # Step 5: Apply INVERSE pivot permutation to solution
    x_elim = Vector{T}(undef, n)
    for k = 1:n
        dest = F.symbolic.global_to_elim[F.row_perm[k]]
        x_elim[dest] = z[k]
    end

    # Step 6: Reorder from elimination to global (AMD) order
    xp = Vector{T}(undef, n)
    for k = 1:n
        xp[F.symbolic.elim_to_global[k]] = x_elim[k]
    end

    # Step 7: Apply inverse AMD permutation
    result = Vector{T}(undef, n)
    for i = 1:n
        result[F.symbolic.perm[i]] = xp[i]
    end

    # Distribute result
    my_start = x.partition[rank + 1]
    my_end = x.partition[rank + 2] - 1
    for i = my_start:my_end
        x.v[i - my_start + 1] = result[i]
    end

    return x
end

# Wrapper types for transpose/adjoint of factorizations
struct TransposeLU{T}
    parent::LUFactorizationMPI{T}
end

struct AdjointLU{T}
    parent::LUFactorizationMPI{T}
end

Base.transpose(F::LUFactorizationMPI{T}) where T = TransposeLU{T}(F)
Base.adjoint(F::LUFactorizationMPI{T}) where T = AdjointLU{T}(F)

function Base.:\(Ft::TransposeLU{T}, b::VectorMPI{T}) where T
    return solve_transpose(Ft.parent, b)
end

function Base.:\(Fa::AdjointLU{T}, b::VectorMPI{T}) where T
    return solve_adjoint(Fa.parent, b)
end

