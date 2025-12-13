"""
Distributed solve routines for LDLT factorization with Bunch-Kaufman pivoting.

Given L * D * L^T * x = b where:
- L is unit lower triangular (distributed)
- D is block diagonal with 1×1 and 2×2 blocks

Solve by:
1. Apply fill-reducing permutation
2. Apply symmetric pivot permutation
3. Forward solve: L * y = b
4. Diagonal solve: D * z = y (handling 2×2 blocks)
5. Backward solve: L^T * x = z
6. Apply inverse permutations
"""

using MPI
using SparseArrays

"""
    solve(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) -> VectorMPI{T}

Solve the linear system A*x = b using the precomputed LDLT factorization.
"""
function solve(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T
    x = VectorMPI(zeros(T, F.symbolic.n); partition=b.partition)
    solve!(x, F, b)
    return x
end

"""
    solve!(x::VectorMPI{T}, F::LDLTFactorizationMPI{T}, b::VectorMPI{T}; distributed::Bool=true)

Solve A*x = b in-place using LDLT factorization.

By default uses distributed solve (MUMPS-style) that keeps factors distributed
and only communicates at subtree boundaries. Set `distributed=false` to use the
gathered solve which collects L/D to all ranks (useful for debugging).
"""
function solve!(x::VectorMPI{T}, F::LDLTFactorizationMPI{T}, b::VectorMPI{T}; distributed::Bool=true) where T
    if distributed
        distributed_solve_ldlt!(x, F, b)
    else
        solve_gathered!(x, F, b)
    end
    return x
end

"""
    solve_gathered!(x::VectorMPI{T}, F::LDLTFactorizationMPI{T}, b::VectorMPI{T})

Solve A*x = b in-place using gathered L/D factors (all ranks have full factors).

This is the original sequential solve. Useful for debugging and verification.
"""
function solve_gathered!(x::VectorMPI{T}, F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n = F.symbolic.n

    # Gather b to all ranks
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

    # Gather L and D to all ranks
    L_full, D_full = gather_L_D(F)

    # Step 3: Forward solve: L * y = work2
    forward_solve_ldlt!(work2, L_full)

    # Step 4: Diagonal solve: D * z = y
    diagonal_solve_bk!(work2, D_full, F.pivots)

    # Step 5: Backward solve: L^T * w = z
    backward_solve_lt!(work2, L_full)

    # Step 6: Apply inverse symmetric permutation
    for k = 1:n
        work[F.sym_perm[k]] = work2[k]
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

"""
    Base.:\\(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T

Solve A*x = b using the backslash operator.
"""
function Base.:\(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T
    return solve(F, b)
end

# ============================================================================
# Transpose and Adjoint Solves for LDLT
# ============================================================================

"""
    solve_transpose(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) -> VectorMPI{T}

Solve transpose(A)*x = b. For symmetric matrices, transpose(A) = A,
so this is equivalent to solve(F, b).
"""
function solve_transpose(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T
    return solve(F, b)
end

"""
    solve_adjoint(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) -> VectorMPI{T}

Solve A'*x = b (adjoint/conjugate transpose).
For real symmetric matrices, A' = A, so this is equivalent to solve(F, b).
For complex symmetric matrices, A' = conj(A), so we conjugate during the solve.
"""
function solve_adjoint(F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T
    if T <: Real
        return solve(F, b)
    else
        # For complex symmetric: A' = conj(A)
        # Solve conj(A) x = b by conjugating L and D values during solve
        x = VectorMPI(zeros(T, F.symbolic.n); partition=b.partition)
        solve_adjoint!(x, F, b)
        return x
    end
end

"""
    solve_adjoint!(x::VectorMPI{T}, F::LDLTFactorizationMPI{T}, b::VectorMPI{T})

Solve A'*x = b in-place for complex symmetric matrices.
"""
function solve_adjoint!(x::VectorMPI{T}, F::LDLTFactorizationMPI{T}, b::VectorMPI{T}) where T
    if T <: Real
        return solve!(x, F, b)
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n = F.symbolic.n

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

    # Gather L and D
    L_full, D_full = gather_L_D(F)

    # Step 3: Forward solve with conj(L)
    forward_solve_ldlt_conj!(work2, L_full)

    # Step 4: Diagonal solve with conj(D)
    diagonal_solve_bk_conj!(work2, D_full, F.pivots)

    # Step 5: Backward solve with conj(L)^T = conj(L^T)
    backward_solve_lt_conj!(work2, L_full)

    # Step 6: Apply inverse symmetric permutation
    for k = 1:n
        work[F.sym_perm[k]] = work2[k]
    end

    # Step 7: Apply inverse fill-reducing permutation
    result = Vector{T}(undef, n)
    for i = 1:n
        result[i] = work[F.symbolic.invperm[i]]
    end

    # Distribute result
    my_start = x.partition[rank + 1]
    my_end = x.partition[rank + 2] - 1
    for i = my_start:my_end
        x.v[i - my_start + 1] = result[i]
    end

    return x
end

# Wrapper types for transpose/adjoint of LDLT factorizations
struct TransposeLDLT{T}
    parent::LDLTFactorizationMPI{T}
end

struct AdjointLDLT{T}
    parent::LDLTFactorizationMPI{T}
end

Base.transpose(F::LDLTFactorizationMPI{T}) where T = TransposeLDLT{T}(F)
Base.adjoint(F::LDLTFactorizationMPI{T}) where T = AdjointLDLT{T}(F)

function Base.:\(Ft::TransposeLDLT{T}, b::VectorMPI{T}) where T
    return solve_transpose(Ft.parent, b)
end

function Base.:\(Fa::AdjointLDLT{T}, b::VectorMPI{T}) where T
    return solve_adjoint(Fa.parent, b)
end

# ============================================================================
# Local Solve Routines (used after gathering)
# ============================================================================

"""
    forward_solve_ldlt!(x, L)

Solve L * y = x in-place, overwriting x with y.
L is unit lower triangular.
"""
function forward_solve_ldlt!(x::AbstractVector{T}, L::SparseMatrixCSC{T}) where T
    n = length(x)

    for j = 1:n
        xj = x[j]
        if xj != zero(T)
            for i in nzrange(L, j)
                row = rowvals(L)[i]
                if row > j  # Below diagonal only
                    x[row] -= nonzeros(L)[i] * xj
                end
            end
        end
    end

    return x
end

"""
    backward_solve_lt!(x, L)

Solve L^T * y = x in-place (L transpose, not adjoint).
Processes columns in reverse order (n to 1).

For unit lower triangular L, L^T is unit upper triangular.
"""
function backward_solve_lt!(x::AbstractVector{T}, L::SparseMatrixCSC{T}) where T
    n = length(x)

    # Process in reverse order
    for j = n:-1:1
        # For column j of L (which is row j of L^T), update x[j]
        for i in nzrange(L, j)
            row = rowvals(L)[i]
            if row > j  # Below diagonal in L = above diagonal in L^T
                x[j] -= nonzeros(L)[i] * x[row]
            end
        end
        # No division needed since L has unit diagonal
    end

    return x
end

"""
    diagonal_solve_bk!(x, D, pivots)

Solve D * y = x in-place, where D is block diagonal with 1×1 and 2×2 blocks.

pivots[k] > 0: 1×1 pivot at k
pivots[k] < 0: 2×2 pivot starting at k (paired with k+1)
"""
function diagonal_solve_bk!(x::AbstractVector{T}, D::SparseMatrixCSC{T}, pivots::Vector{Int}) where T
    n = length(x)

    k = 1
    while k <= n
        if pivots[k] >= 0
            # 1×1 pivot: x[k] = x[k] / D[k,k]
            d_kk = get_D_entry(D, k, k)
            if abs(d_kk) < eps(real(T))
                @warn "Near-zero diagonal in D at position $k"
                d_kk = eps(real(T))
            end
            x[k] /= d_kk
            k += 1
        else
            # 2×2 pivot at (k, k+1)
            d_kk = get_D_entry(D, k, k)
            d_k1k = get_D_entry(D, k+1, k)
            d_k1k1 = get_D_entry(D, k+1, k+1)

            det = d_kk * d_k1k1 - d_k1k * d_k1k
            if abs(det) < eps(real(T))
                @warn "Near-singular 2×2 block in D at positions $k:$(k+1)"
                det = sign(det) * max(abs(det), eps(real(T)))
            end

            x_k = x[k]
            x_k1 = x[k+1]

            # Solve 2×2 system
            x[k] = (d_k1k1 * x_k - d_k1k * x_k1) / det
            x[k+1] = (-d_k1k * x_k + d_kk * x_k1) / det

            k += 2
        end
    end

    return x
end

"""
    get_D_entry(D, i, j)

Get entry D[i,j] from the sparse block diagonal matrix D.
"""
function get_D_entry(D::SparseMatrixCSC{T}, i::Int, j::Int) where T
    for idx in nzrange(D, j)
        if rowvals(D)[idx] == i
            return nonzeros(D)[idx]
        end
    end
    return zero(T)
end

# ============================================================================
# Conjugate Solve Helpers for Complex Symmetric LDLT
# ============================================================================

"""
    forward_solve_ldlt_conj!(x, L)

Solve conj(L) * y = x in-place.
"""
function forward_solve_ldlt_conj!(x::AbstractVector{T}, L::SparseMatrixCSC{T}) where T
    n = length(x)

    for j = 1:n
        xj = x[j]
        if xj != zero(T)
            for i in nzrange(L, j)
                row = rowvals(L)[i]
                if row > j
                    x[row] -= conj(nonzeros(L)[i]) * xj
                end
            end
        end
    end

    return x
end

"""
    backward_solve_lt_conj!(x, L)

Solve conj(L)^T * y = x in-place.
"""
function backward_solve_lt_conj!(x::AbstractVector{T}, L::SparseMatrixCSC{T}) where T
    n = length(x)

    for j = n:-1:1
        for i in nzrange(L, j)
            row = rowvals(L)[i]
            if row > j
                x[j] -= conj(nonzeros(L)[i]) * x[row]
            end
        end
    end

    return x
end

"""
    diagonal_solve_bk_conj!(x, D, pivots)

Solve conj(D) * y = x in-place for complex symmetric matrices.
"""
function diagonal_solve_bk_conj!(x::AbstractVector{T}, D::SparseMatrixCSC{T}, pivots::Vector{Int}) where T
    n = length(x)

    k = 1
    while k <= n
        if pivots[k] >= 0
            # 1×1 pivot
            d_kk = conj(get_D_entry(D, k, k))
            if abs(d_kk) < eps(real(T))
                d_kk = eps(real(T))
            end
            x[k] /= d_kk
            k += 1
        else
            # 2×2 pivot
            d_kk = conj(get_D_entry(D, k, k))
            d_k1k = conj(get_D_entry(D, k+1, k))
            d_k1k1 = conj(get_D_entry(D, k+1, k+1))

            det = d_kk * d_k1k1 - d_k1k * d_k1k
            if abs(det) < eps(real(T))
                det = sign(det) * max(abs(det), eps(real(T)))
            end

            x_k = x[k]
            x_k1 = x[k+1]

            x[k] = (d_k1k1 * x_k - d_k1k * x_k1) / det
            x[k+1] = (-d_k1k * x_k + d_kk * x_k1) / det

            k += 2
        end
    end

    return x
end
