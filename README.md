# Odrpack.jl

[![Test](https://github.com/HugoMVale/Odrpack.jl/actions/workflows/test.yml/badge.svg)](https://github.com/HugoMVale/Odrpack.jl/actions)
[![Coverage](https://coveralls.io/repos/github/HugoMVale/Odrpack.jl/badge.svg?branch=main)](https://coveralls.io/github/HugoMVale/Odrpack.jl?branch=main)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/HugoMVale/Odrpack.jl)

## Description

This Julia package provides bindings for the well-known weighted orthogonal distance regression
(ODR) solver [odrpack95]. 

ODR, also known as [errors-in-variables regression], is designed primarily for instances when both
the explanatory and response variables have significant errors. 

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Total_least_squares.svg/220px-Total_least_squares.svg.png" width="250" alt="Deming regression; special case of ODR." style="margin-right: 10px;">
</p>

[errors-in-variables regression]: https://en.wikipedia.org/wiki/Errors-in-variables_models
[odrpack95]: https://github.com/HugoMVale/odrpack95


## Installation

You can install the package in the usual way:

```julia
using Pkg
Pkg.add("Odrpack")
```

## Documentation and Usage

The following example demonstrates a simple use of the package. For more comprehensive examples and explanations, please refer to the [examples](./examples) notebooks.

```julia
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
    # report=:long
)

println("Optimized β    :", sol.beta)
println("Optimized δ    :", sol.delta)
println("Optimized x + δ:", sol.xplusd)
```

```sh
Optimized β    :[1.633376, 0.9]
Optimized δ    :[-0.368861, -0.312730, 0.029286, 0.110314]
Optimized x + δ:[0.613138, 1.685269, 5.007286, 6.120314]
```

