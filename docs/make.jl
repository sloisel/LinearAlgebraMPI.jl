using Documenter
using LinearAlgebraMPI

makedocs(
    sitename = "LinearAlgebraMPI.jl",
    modules = [LinearAlgebraMPI],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
        "Internals" => "internals.md",
    ],
    checkdocs = :exports,
    remotes = nothing,  # Disable source links (configure when publishing)
)

deploydocs(
    repo = "github.com/sloisel/LinearAlgebraMPI.jl.git",
    devbranch = "main",
)
