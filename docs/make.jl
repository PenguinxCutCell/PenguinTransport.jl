using Documenter
using PenguinTransport

makedocs(
    modules = [PenguinTransport],
    authors = "PenguinxCutCell contributors",
    sitename = "PenguinTransport.jl",
    format = Documenter.HTML(
        canonical = "https://PenguinxCutCell.github.io/PenguinTransport.jl",
        repolink = "https://github.com/PenguinxCutCell/PenguinTransport.jl",
        collapselevel = 2,
    ),
    doctest = false,
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Examples" => "examples.md",
        "Algorithms" => "algorithms.md",
        "Transport Models" => "transport.md",
    ],
    pagesonly = true,
    warnonly = false,
    remotes = nothing,
)

if get(ENV, "CI", "") == "true"
    deploydocs(
        repo = "github.com/PenguinxCutCell/PenguinTransport.jl",
        push_preview = true,
    )
end
