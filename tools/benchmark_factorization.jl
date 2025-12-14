#!/usr/bin/env julia
#
# Benchmark factorization performance (LU and LDLT) for distributed sparse matrices.
#
# Usage: mpiexec -n 4 julia --project=. --threads=2 tools/benchmark_factorization.jl
#
# Results are saved to tools/benchmark_factorization_results.txt for comparison.

using MPI
MPI.Init()

using SparseArrays
using LinearAlgebra
using Printf
using Statistics
using Dates

# Add the package
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using LinearAlgebraMPI

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

"""
    laplacian_2d_sparse(n)

Create a 2D Laplacian matrix for an approximately sqrt(n) × sqrt(n) grid.
Returns a SparseMatrixCSC.
"""
function laplacian_2d_sparse(n)
    grid_size = max(1, round(Int, sqrt(n)))
    e = ones(grid_size)
    L1D = spdiagm(-1 => -e[1:end-1], 0 => 2*e, 1 => -e[1:end-1])
    I_g = sparse(I, grid_size, grid_size)
    L2D = kron(I_g, L1D) + kron(L1D, I_g)
    return L2D
end

"""
    benchmark_factorization(A_mpi, factorize_func, n_warmup, n_runs)

Benchmark a factorization function, returning median time in seconds.
All ranks participate but only rank 0 reports.
"""
function benchmark_factorization(A_mpi, factorize_func, n_warmup, n_runs)
    # Warmup
    for _ in 1:n_warmup
        F = factorize_func(A_mpi)
    end
    MPI.Barrier(comm)

    times = Float64[]
    for _ in 1:n_runs
        GC.gc()
        MPI.Barrier(comm)
        t_start = MPI.Wtime()
        F = factorize_func(A_mpi)
        MPI.Barrier(comm)
        t_end = MPI.Wtime()
        push!(times, t_end - t_start)
    end

    return median(times)
end

"""
    run_benchmarks()

Run factorization benchmarks for various sizes.
"""
function run_benchmarks()
    sizes = [10_000]

    if rank == 0
        println("Factorization Benchmark")
        println("=" ^ 60)
        println("MPI ranks: $nranks")
        println("Julia threads: $(Threads.nthreads())")
        println("BLAS threads: $(BLAS.get_num_threads())")
        println("BLAS vendor: $(BLAS.get_config())")
        println()
    end

    # Results structure
    results = Dict{String, Any}(
        "config" => Dict(
            "nranks" => nranks,
            "julia_threads" => Threads.nthreads(),
            "blas_threads" => BLAS.get_num_threads(),
            "date" => Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
        ),
        "results" => Dict{Int, Dict{String, Float64}}()
    )

    for n in sizes
        if rank == 0
            println("=" ^ 60)
            println("Benchmarking n = $n")
        end

        # Create local sparse matrix
        A_local = laplacian_2d_sparse(n)
        actual_n = size(A_local, 1)

        if rank == 0
            println("  Actual matrix size: $actual_n × $actual_n, nnz = $(nnz(A_local))")
        end

        # Create distributed matrix (all ranks have same A_local)
        A_mpi = SparseMatrixMPI{Float64}(A_local)

        # Determine iterations based on size (fewer runs for faster benchmarking)
        if actual_n <= 200
            n_warmup, n_runs = 2, 3
        else
            n_warmup, n_runs = 1, 2
        end

        results["results"][n] = Dict{String, Float64}()

        # LU factorization
        if rank == 0
            println("  LU factorization ...")
        end
        time_lu = benchmark_factorization(A_mpi, lu, n_warmup, n_runs)
        results["results"][n]["lu"] = time_lu
        if rank == 0
            println("    LU: $(@sprintf("%.4f", time_lu)) s")
        end

        # LDLT factorization (matrix is symmetric)
        if rank == 0
            println("  LDLT factorization ...")
        end
        time_ldlt = benchmark_factorization(A_mpi, ldlt, n_warmup, n_runs)
        results["results"][n]["ldlt"] = time_ldlt
        if rank == 0
            println("    LDLT: $(@sprintf("%.4f", time_ldlt)) s")
            println()
        end
    end

    return results, sizes
end

"""
    save_results(results, sizes, filename)

Save benchmark results to text file (rank 0 only).
"""
function save_results(results, sizes, filename)
    if rank == 0
        open(filename, "w") do f
            println(f, "# Factorization Benchmark Results")
            println(f, "# Date: $(results["config"]["date"])")
            println(f, "# MPI ranks: $(results["config"]["nranks"])")
            println(f, "# Julia threads: $(results["config"]["julia_threads"])")
            println(f, "# BLAS threads: $(results["config"]["blas_threads"])")
            println(f, "#")
            println(f, "# n,lu_time,ldlt_time")
            for n in sizes
                lu_time = results["results"][n]["lu"]
                ldlt_time = results["results"][n]["ldlt"]
                println(f, "$n,$lu_time,$ldlt_time")
            end
        end
        println("Results saved to: $filename")
    end
end

"""
    generate_html_report(results, sizes, filename)

Generate an HTML report with benchmark results (rank 0 only).
"""
function generate_html_report(results, sizes, filename)
    if rank != 0
        return
    end

    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Factorization Benchmark</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .info {
            background: #e8f4fd;
            border-left: 4px solid #4a90d9;
            padding: 15px;
            margin: 20px 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px 15px;
            text-align: right;
        }
        th { background: #4a90d9; color: white; font-weight: 600; }
        tr:nth-child(even) { background: #f9f9f9; }
        tr:hover { background: #f0f7ff; }
        td:first-child, th:first-child { text-align: left; font-weight: 600; }
        .time { font-family: 'Monaco', 'Menlo', monospace; font-size: 0.9em; }
        .footnote {
            font-size: 0.85em;
            color: #666;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <h1>Factorization Benchmark</h1>
    <h2>LU and LDLT Performance</h2>

    <div class="info">
        <strong>Test Configuration:</strong><br>
        Matrix type: 2D Laplacian (5-point stencil)<br>
        MPI ranks: $(results["config"]["nranks"])<br>
        Julia threads: $(results["config"]["julia_threads"])<br>
        BLAS threads: $(results["config"]["blas_threads"])<br>
        Date: $(results["config"]["date"])
    </div>

    <h2>Results</h2>
    <table>
        <tr>
            <th>Matrix Size</th>
            <th>nnz</th>
            <th>LU Time</th>
            <th>LDLT Time</th>
        </tr>
"""

    for n in sizes
        A = laplacian_2d_sparse(n)
        actual_n = size(A, 1)
        nnz_A = nnz(A)

        time_lu = results["results"][n]["lu"]
        time_ldlt = results["results"][n]["ldlt"]

        html *= """        <tr>
            <td>$actual_n × $actual_n</td>
            <td>$(format_number(nnz_A))</td>
            <td class="time">$(format_time(time_lu))</td>
            <td class="time">$(format_time(time_ldlt))</td>
        </tr>
"""
    end

    html *= """    </table>

    <div class="footnote">
        <strong>Notes:</strong>
        <ul>
            <li>Times are median of multiple runs</li>
            <li>All MPI ranks participate in factorization</li>
            <li>BLAS operations use multithreading internally</li>
        </ul>
    </div>
</body>
</html>
"""

    open(filename, "w") do f
        write(f, html)
    end
    println("Report written to: $filename")
end

function format_time(t)
    if t >= 1.0
        return @sprintf("%.2f s", t)
    elseif t >= 0.001
        return @sprintf("%.2f ms", t * 1000)
    else
        return @sprintf("%.2f μs", t * 1_000_000)
    end
end

function format_number(n)
    if n >= 1_000_000
        return @sprintf("%.1fM", n / 1_000_000)
    elseif n >= 1_000
        return @sprintf("%.1fK", n / 1_000)
    else
        return string(n)
    end
end

# Run benchmarks
if rank == 0
    println("Start time: $(Dates.now())")
    println()
end

results, sizes = run_benchmarks()

# Save results
results_path = joinpath(@__DIR__, "benchmark_factorization_results.txt")
save_results(results, sizes, results_path)

# Generate HTML report
html_path = joinpath(@__DIR__, "benchmark_factorization_results.html")
generate_html_report(results, sizes, html_path)

if rank == 0
    println()
    println("=" ^ 60)
    println("Benchmark complete!")
    println("Open $html_path to view results")
end

MPI.Finalize()
