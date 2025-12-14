#!/usr/bin/env julia
#
# Benchmark thread scaling for sparse matrix multiplication
# Uses BenchmarkTools for accurate timing
#
# Usage: julia --project=. --threads=8 tools/benchmark_scaling.jl

using SparseArrays
using LinearAlgebra
using Printf
using Dates
using BenchmarkTools

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using LinearAlgebraMPI: ⊛

const SIZES = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
const THREAD_COUNTS = [1, 2, 4, 8]

function laplacian_2d(n)
    grid_size = max(1, round(Int, sqrt(n)))
    e = ones(grid_size)
    L1D = spdiagm(-1 => -e[1:end-1], 0 => 2*e, 1 => -e[1:end-1])
    I_g = sparse(I, grid_size, grid_size)
    return kron(I_g, L1D) + kron(L1D, I_g)
end

function generate_html_report(results, sizes, thread_counts, filename)
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Sparse Matrix Multiplication - Thread Scaling</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
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
            text-align: center;
        }
        th { background: #4a90d9; color: white; font-weight: 600; }
        tr:nth-child(even) { background: #f9f9f9; }
        tr:hover { background: #f0f7ff; }
        td:first-child, th:first-child { text-align: left; font-weight: 600; }
        .speedup-good { color: #2e7d32; font-weight: bold; }
        .speedup-bad { color: #c62828; }
        .speedup-neutral { color: #666; }
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
    <h1>Threaded Sparse Multiplication - Scaling</h1>

    <div class="info">
        <strong>Test Configuration:</strong><br>
        Matrix type: 2D Laplacian (5-point stencil)<br>
        Operation: A × A using ⊛ operator<br>
        Julia threads: $(Threads.nthreads())<br>
        Benchmarking: BenchmarkTools.jl<br>
        Date: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    </div>

    <h2>Speedup vs Single-Threaded Standard (*)</h2>
    <table>
        <tr>
            <th>Matrix Size</th>
"""

    for nt in thread_counts
        html *= "            <th>$nt thread$(nt > 1 ? "s" : "")</th>\n"
    end
    html *= "        </tr>\n"

    for n in sizes
        actual_n = results[n]["actual_n"]
        html *= "        <tr>\n"
        html *= "            <td>$(format_size(actual_n))</td>\n"

        baseline = results[n]["std"]  # Single-threaded standard as baseline

        for nt in thread_counts
            time_threaded = results[n]["threaded_$nt"]
            speedup = baseline / time_threaded

            css = if speedup >= 1.5
                "speedup-good"
            elseif speedup < 0.8
                "speedup-bad"
            else
                "speedup-neutral"
            end

            html *= "            <td class=\"$css\">$(@sprintf("%.2f", speedup))×</td>\n"
        end
        html *= "        </tr>\n"
    end

    html *= """    </table>

    <div class="footnote">
        <strong>Notes:</strong>
        <ul>
            <li>Speedup = (single-threaded * time) / (threaded ⊛ time with N threads)</li>
            <li>Green: ≥1.5× speedup, Red: &lt;0.8× (slower than baseline)</li>
            <li>Times measured using BenchmarkTools.jl @belapsed</li>
        </ul>
    </div>
</body>
</html>
"""

    open(filename, "w") do f
        write(f, html)
    end
end

function format_size(n)
    n = Int(n)
    if n >= 1_000_000
        return @sprintf("%.0fM × %.0fM", n/1e6, n/1e6)
    elseif n >= 1_000
        return @sprintf("%.0fK × %.0fK", n/1e3, n/1e3)
    else
        return "$n × $n"
    end
end

# Main
println("Sparse Matrix Multiplication - Thread Scaling Benchmark")
println("=" ^ 60)
println("Julia threads: $(Threads.nthreads())")
println("Start time: $(Dates.now())")
println()

results = Dict{Int, Dict{String, Float64}}()

for n in SIZES
    println("=" ^ 60)
    println("Problem size n = $n")

    A = laplacian_2d(n)
    actual_n = size(A, 1)
    println("  Matrix: $actual_n × $actual_n, nnz = $(nnz(A))")

    results[n] = Dict{String, Float64}()
    results[n]["actual_n"] = Float64(actual_n)

    # Benchmark standard single-threaded *
    print("  Standard (*): ")
    flush(stdout)
    time_std = @belapsed $A * $A
    results[n]["std"] = time_std
    println(@sprintf("%.6f s", time_std))

    # Benchmark threaded ⊛ with varying thread counts
    for nt in THREAD_COUNTS
        print("  ⊛ ($nt threads): ")
        flush(stdout)
        time_threaded = @belapsed ⊛($A, $A; max_threads=$nt)
        results[n]["threaded_$nt"] = time_threaded
        speedup = time_std / time_threaded
        println(@sprintf("%.6f s  (%.2f×)", time_threaded, speedup))
    end
    println()
end

# Generate report
report_path = joinpath(@__DIR__, "benchmark_scaling.html")
generate_html_report(results, SIZES, THREAD_COUNTS, report_path)

println("=" ^ 60)
println("Report written to: $report_path")
