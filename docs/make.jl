using StructuralClosure
using Documenter

DocMeta.setdocmeta!(
    StructuralClosure,
    :DocTestSetup,
    :(using StructuralClosure);
    recursive = true,
)

makedocs(;
    modules = [StructuralClosure],
    authors = "Syver Døving Agdestein <syverda@gmail.com> and contributors",
    sitename = "StructuralClosure.jl",
    format = Documenter.HTML(;
        canonical = "https://agdestein.github.io/StructuralClosure.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/agdestein/StructuralClosure.jl", devbranch = "main")
