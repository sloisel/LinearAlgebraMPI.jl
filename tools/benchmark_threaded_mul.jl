#!/usr/bin/env julia
#
# Benchmark comparing sparse matrix multiplication methods:
#   1. Standard Julia (*)
#   2. Our threaded block-parallel (⊛)
#
# Usage: julia --project=. --threads=8 tools/benchmark_threaded_mul.jl

using SparseArrays
using LinearAlgebra
using Printf
using Statistics
using Dates

# Add the package
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using LinearAlgebraMPI: ⊛

"""
    laplacian_2d(n)

Create a 2D Laplacian matrix for an approximately sqrt(n) × sqrt(n) grid.
"""
function laplacian_2d(n)
    grid_size = max(1, round(Int, sqrt(n)))
    e = ones(grid_size)
    L1D = spdiagm(-1 => -e[1:end-1], 0 => 2*e, 1 => -e[1:end-1])
    I_g = sparse(I, grid_size, grid_size)
    L2D = kron(I_g, L1D) + kron(L1D, I_g)
    return L2D
end

"""
    benchmark_multiply(A, B, func, n_warmup, n_runs)

Benchmark a multiplication function, returning median time in seconds.
"""
function benchmark_multiply(A, B, func, n_warmup, n_runs)
    for _ in 1:n_warmup
        C = func(A, B)
    end

    times = Float64[]
    for _ in 1:n_runs
        GC.gc()
        t = @elapsed begin
            C = func(A, B)
        end
        push!(times, t)
    end

    return median(times)
end

"""
    run_benchmarks()

Run benchmarks for various sizes and methods.
"""
function run_benchmarks()
    sizes = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]

    max_threads = Threads.nthreads()
    println("Julia threads available: $max_threads")
    println()

    # Results structure
    results = Dict{Int, Dict{String, Float64}}()

    for n in sizes
        println("=" ^ 60)
        println("Benchmarking n = $n")

        A = laplacian_2d(n)
        actual_n = size(A, 1)
        println("  Actual matrix size: $actual_n × $actual_n, nnz = $(nnz(A))")

        # Determine iterations based on size
        if actual_n <= 100
            n_warmup, n_runs = 10, 20
        elseif actual_n <= 1000
            n_warmup, n_runs = 3, 10
        elseif actual_n <= 10000
            n_warmup, n_runs = 2, 5
        else
            n_warmup, n_runs = 1, 3
        end

        results[n] = Dict{String, Float64}()

        # 1. Standard Julia multiplication
        println("  Standard (*) ...")
        time_std = benchmark_multiply(A, A, *, n_warmup, n_runs)
        results[n]["std"] = time_std
        println("    Standard (*): $(@sprintf("%.6f", time_std)) s")

        # 2. Our threaded block-parallel (⊛)
        println("  Threaded (⊛) ...")
        time_threaded = benchmark_multiply(A, A, ⊛, n_warmup, n_runs)
        results[n]["threaded"] = time_threaded
        speedup = time_std / time_threaded
        println("    Threaded (⊛): $(@sprintf("%.6f", time_threaded)) s (speedup: $(@sprintf("%.2f", speedup))×)")
        println()
    end

    return results, sizes
end

"""
    generate_html_report(results, sizes, filename)

Generate an HTML report with benchmark results.
"""
function generate_html_report(results, sizes, filename)
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Sparse Matrix Multiplication Benchmark</title>
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
        .speedup-good { color: #2e7d32; font-weight: bold; }
        .speedup-bad { color: #c62828; font-weight: bold; }
        .speedup-neutral { color: #666; }
        .time { font-family: 'Monaco', 'Menlo', monospace; font-size: 0.9em; }
        .best { background: #c8e6c9 !important; }
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
    <h1>Sparse Matrix Multiplication Benchmark</h1>
    <h2>Standard (*) vs Threaded (⊛)</h2>

    <div class="info">
        <strong>Test Configuration:</strong><br>
        Matrix type: 2D Laplacian (5-point stencil)<br>
        Julia threads: $(Threads.nthreads())<br>
        Date: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    </div>

    <h2>Results</h2>
    <table>
        <tr>
            <th>Matrix Size</th>
            <th>nnz</th>
            <th>Standard (*)</th>
            <th>Threaded (⊛)</th>
            <th>Speedup</th>
        </tr>
"""

    for n in sizes
        A = laplacian_2d(n)
        actual_n = size(A, 1)
        nnz_A = nnz(A)

        time_std = results[n]["std"]
        time_threaded = results[n]["threaded"]
        speedup = time_std / time_threaded

        best_time = min(time_std, time_threaded)
        css_speedup = speedup >= 1.5 ? "speedup-good" : (speedup < 0.8 ? "speedup-bad" : "speedup-neutral")

        html *= """        <tr>
            <td>$actual_n × $actual_n</td>
            <td>$(format_number(nnz_A))</td>
            <td class="time$(time_std == best_time ? " best" : "")">$(format_time(time_std))</td>
            <td class="time$(time_threaded == best_time ? " best" : "")">$(format_time(time_threaded))</td>
            <td class="$css_speedup">$(@sprintf("%.2f", speedup))×</td>
        </tr>
"""
    end

    html *= """    </table>

    <div class="footnote">
        <strong>Notes:</strong>
        <ul>
            <li>Times are median of multiple runs</li>
            <li>Green cells indicate best time for that matrix size</li>
            <li>Speedup > 1.0 means threaded version is faster</li>
            <li>Green speedup: ≥1.5×, Red: &lt;0.8×</li>
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
println("Sparse Matrix Multiplication Benchmark")
println("=" ^ 60)
println("Methods: Standard (*) vs Threaded (⊛)")
println("Start time: $(Dates.now())")
println()

results, sizes = run_benchmarks()

# Generate HTML report
report_path = joinpath(@__DIR__, "benchmark_results.html")
generate_html_report(results, sizes, report_path)

println()
println("=" ^ 60)
println("Benchmark complete!")
println("Open $report_path to view results")
