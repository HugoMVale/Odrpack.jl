# Explicit orthogonal regression

Estimate the parameters of a scalar non-linear model from experimental data.

```math
y = f(x, \bm{\beta}) =  \frac{\beta_1 x^2 + x (1-x)}{\beta_1 x^2 + 2 x (1-x) + \beta_2 (1-x)^2}
```


```@example explicit_model
using Odrpack
using Plots
```

First, we define the experimental data and the model function. 

```@example explicit_model
xdata = [0.100, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800]
ydata = [0.059, 0.243, 0.364, 0.486, 0.583, 0.721, 0.824]
nothing; # hide
```

```@example explicit_model
function f!(x::Vector{Float64}, beta::Vector{Float64}, y::Vector{Float64})
    b1, b2 = beta
    y .= (b1 .* x.^2 .+ x .* (1 .- x)) ./ (b1 .* x.^2 .+ 2 .* x .* (1 .- x) .+ b2 .* (1 .- x).^2)
    return nothing
end
```

Then, we define an initial guess for the model parameters `beta` and, optionally, also the corresponding bounds.

```@example explicit_model
beta0 = [1.0, 1.0]
bounds = ([0.0, 0.0], [2.0, 2.0])
nothing; # hide
```

Lastly, we define the weights for `x` and `y` based on a suitable rationale, such as the estimated uncertainty of each variable.

```@example explicit_model
sigma_x = 0.01
sigma_y = 0.05

weight_x = 1/sigma_x^2
weight_y = 1/sigma_y^2
nothing; # hide
```

We can now launch the regression! If you want to see a brief computation report, set `report=:short`.

```@example explicit_model
sol = odr_fit(f!, xdata, ydata, beta0; bounds=bounds, weight_x=weight_x, weight_y=weight_y)
```

The result is packed in a [`OdrResult`](@ref) struct. Let's check the solution convergence and the estimated model parameters.

```@example explicit_model
sol.stopreason
```

```@example explicit_model
sol.beta
```

All fine! Let's plot the solution.

```@example explicit_model
# Plot experimental data
scatter(xdata, ydata, label="Data", legend=:topleft)

# Plot fitted model
xm = collect(range(0.0, 1.0; length=100))
ym = zeros(length(xm))
f!(xm, sol.beta, ym)
plot!(xm, ym, label="Fitted model")
xlabel!("x")
ylabel!("y")
savefig("explicit-model-plot.svg"); nothing # hide
```

![](explicit-model-plot.svg)
