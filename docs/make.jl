using Documenter, Odrpack

makedocs(
    format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"),
    modules=[Odrpack],
    sitename="Odrpack.jl",
    authors="Hugo Vale",
    doctest=true,
    checkdocs=:none,
    pages=[
        "Home" => "index.md",
        #"Examples" => "examples.md",
        "API" => "api.md"
    ]
)

deploydocs(
    repo="github.com/HugoMVale/Odrpack.jl",
    push_preview=true
)