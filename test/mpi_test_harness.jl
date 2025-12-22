module MPITestHarness

using Test

mutable struct QuietTestSet <: Test.AbstractTestSet
    description::String
    counts::Dict{Symbol,Int}
end

QuietTestSet(description::AbstractString="") = QuietTestSet(String(description), Dict(:pass=>0, :fail=>0, :error=>0, :broken=>0, :skip=>0))

function Test.record(ts::QuietTestSet, res)
    c = ts.counts
    if res isa Test.Pass
        c[:pass] += 1
    elseif res isa Test.Fail
        c[:fail] += 1
    elseif res isa Test.Error
        c[:error] += 1
        println("TEST ERROR: ", res)
    elseif res isa Test.Broken
        c[:broken] += 1
    elseif res isa Test.Skip
        c[:skip] += 1
    end
    return res
end

# Internal: absorb results from arbitrary nodes (results or nested testsets)
function _absorb_counts!(counts::Dict{Symbol,Int}, node)
    if node isa QuietTestSet
        for (k, v) in node.counts
            counts[k] = get(counts, k, 0) + v
        end
    elseif node isa Test.Pass
        counts[:pass] = get(counts, :pass, 0) + 1
    elseif node isa Test.Fail
        counts[:fail] = get(counts, :fail, 0) + 1
    elseif node isa Test.Error
        counts[:error] = get(counts, :error, 0) + 1
    elseif node isa Test.Broken
        counts[:broken] = get(counts, :broken, 0) + 1
    elseif node isa Test.Skip
        counts[:skip] = get(counts, :skip, 0) + 1
    elseif node isa Test.AbstractTestSet
        # Try to descend into arbitrary testset implementations that store `results`
        if hasproperty(node, :results)
            for x in getproperty(node, :results)
                _absorb_counts!(counts, x)
            end
        end
    end
    return nothing
end

function Test.record(ts::QuietTestSet, child::Test.AbstractTestSet)
    _absorb_counts!(ts.counts, child)
    return child
end

Test.finish(ts::QuietTestSet) = ts

end # module
