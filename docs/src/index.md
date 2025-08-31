# Odrpack.jl

[Odrpack.jl](https://github.com/HugoMVale/Odrpack.jl) is Julia package that provides bindings for the well-known weighted orthogonal distance regression (ODR) solver [odrpack95](https://github.com/HugoMVale/odrpack95). 

Orthogonal distance regression, also known as [errors-in-variables regression](https://en.wikipedia.org/wiki/Errors-in-variables_models), is designed primarily for instances when both the explanatory and response variables have significant errors. 

![Deming regression; special case of ODR.](https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Total_least_squares.svg/220px-Total_least_squares.svg.png)

!!! note

    This package is still in early alpha stage, and APIs can change any time in the future. Discussions and potential use cases are extremely welcome!


## Installation

`Odrpack.jl` can be installed using the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add Odrpack
```

## Usage

A basic fit can be performed by passing the model function, data, and initial parameter estimates to [`odr_fit`](@ref).

```@example usage
using Odrpack

function f!(x::Vector{Float64}, beta::Vector{Float64}, y::Vector{Float64})
    y .= beta[1] .* exp.(beta[2] .* x)
    return nothing
end

xdata = [0.982, 1.998, 4.978, 6.01]
ydata = [2.7, 7.4, 148.0, 403.0]

beta0 = [2.0, 0.5]
bounds = ([0.0, 0.0], [10.0, 0.9])

sol = odr_fit(
    f!,
    xdata,
    ydata,
    beta0,
    bounds=bounds,
    # rptfile="test_output.txt",
    # report=:short
)

println("Optimized β    : " , round.(sol.beta, sigdigits=5))
println("Optimized δ    : ", round.(sol.delta, sigdigits=5))
println("Optimized x + δ: ", round.(sol.xplusd, sigdigits=5))
```

