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
        "Equations" => "equations.md",
        "Boundary Conditions" => "boundary-conditions.md",
        "Numerics" => "numerics.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
        "Validation" => "validation.md",
        "Design Notes" => "design.md",
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
