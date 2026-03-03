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
    pages = [
        "Home" => "index.md",
        "Docstrings" => "docstrings.md",
        "API" => "api.md",
        "Reference" => "reference.md",
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
