# Implicit orthogonal regression

Estimate the parameters of an ellipse from a set of coordinates.

```math
\begin{align*}
f(\bm{X}, \bm{\beta}) & = \frac{\left[(x-x_0)\cos\theta + (y-y_0)\sin\theta\right]^2}{a^2} \\
& + \frac{\left[(y-y_0)\cos\theta -(x-x_0)\sin\theta\right]^2}{b^2} - 1 = 0
\end{align*}
```

```math
\bm{X} = (x,y)
```

```math
\bm{\beta} = (x_0, y_0, a, b, \theta)
```


```@example implicit_model
using Odrpack
using Plots
```

First, we define the observed data and the model function. 

```@example implicit_model
# Each row represents a point (x, y) of the ellipse
Xdata = [
    0.50 -0.12
    1.20 -0.60
    1.60 -1.00
    1.86 -1.40
    2.12 -2.54
    2.36 -3.36
    2.44 -4.00
    2.36 -4.75
    2.06 -5.25
    1.74 -5.64
    1.34 -5.97
    0.90 -6.32
    -0.28 -6.44
    -0.78 -6.44
    -1.36 -6.41
    -1.90 -6.25
    -2.50 -5.88
    -2.88 -5.50
    -3.18 -5.24
    -3.44 -4.86
]

# Ydata is not used, but is required
Ydata = zeros(size(Xdata, 1), 1)
nothing; # hide
```

```@example implicit_model
function f!(X::Matrix{Float64}, beta::Vector{Float64}, Y::Matrix{Float64})
    x0, y0, a, b, θ = beta
    x = X[:, 1]
    y = X[:, 2]
    Y .=  ((x .- x0) * cos(θ) .+ (y .- y0) * sin(θ)).^2 / a^2 .+ 
          ((y .- y0) * cos(θ) .- (x .- x0) * sin(θ)).^2 / b^2 .- 1
    return nothing
end
```

Then, we define a plausible initial guess ``\bm{\beta_0}`` for the model parameters, as well as the corresponding bounds.

```@example implicit_model
beta0 =[0.0, 0.0, 1.0, 1.0, 0.0]

lower = [-1e2, -1e2, 0e0, 0e0, -π/2]
upper = [+1e2, +1e2, 1e2, 1e2, +π/2]
nothing; # hide
```

Here, we expect the measurement error to be the same across both ``\bm{X}`` coordinates, so a special weighting scheme is unnecessary.

```@example implicit_model
weight_x = 1.0
nothing; # hide
```

We can now launch the regression! As the problem is implicit, we set `task=:implicitODR`. If you want to see a brief computation report, set `report=:short`.

```@example implicit_model
sol = odr_fit(f!, Xdata, Ydata, beta0, bounds=(lower, upper), weight_x=weight_x,
    task=:implicitODR, report=:none);
```

The result is packed in a [`OdrResult`](@ref) struct. Let's check the solution convergence and the estimated model parameters.

```@example implicit_model
sol.stopreason
```

```@example implicit_model
sol.beta
```

All fine! Let's plot the solution.

```@example implicit_model
# Plot observed data
scatter(Xdata[:, 1], Xdata[:, 2], label="Data", legend=false)

# Plot fitted ellipse
x0, y0, a, b, θ = sol.beta
t = range(0, 2π; length=200)
ellipse_x = x0 .+ a .* cos.(t) .* cos(θ) .- b .* sin.(t) .* sin(θ)
ellipse_y = y0 .+ a .* cos.(t) .* sin(θ) .+ b .* sin.(t) .* cos(θ)
plot!(ellipse_x, ellipse_y, color=:orange, linewidth=2, label="Fit")
xlabel!("x")
ylabel!("y")
savefig("implicit-model-plot.svg"); nothing # hide
```

![](implicit-model-plot.svg)
