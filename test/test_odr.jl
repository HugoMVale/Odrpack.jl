using Test
using Odrpack

module TestCases

export case1, case2, case3

using Random

const SEED = 1234567890
const rng = Random.MersenneTwister(SEED)

function add_noise!(x::AbstractArray, noise::Real)
    x .*= (1 .+ noise * (2 * rand(rng, size(x)...) .- 1))
    return nothing
end

function create_case1()
    # m=1, q=1
    function f!(x::Vector{Float64}, beta::Vector{Float64}, y::Vector{Float64})
        y .= beta[1] .+ beta[2] .* x .+ beta[3] .* x .^ 2 .+ beta[4] .* x .^ 3
        return nothing
    end

    beta_star = [1.0, -2.0, 0.1, -0.1]
    x = collect(range(-10.0, 10.0, length=21))
    y = zeros(size(x, 1))
    f!(x, beta_star, y)

    add_noise!(x, 5e-2)
    add_noise!(y, 10e-2)

    return (f=f!, xdata=x, ydata=y, beta0=zeros(length(beta_star)))
end

function create_case2()
    # m=2, q=1
    function f!(x::Matrix{Float64}, beta::Vector{Float64}, y::Vector{Float64})
        y .= (beta[1] * x[:, 1]) .^ 3 .+ x[:, 2] .^ beta[2]
        return nothing
    end

    beta_star = [2.0, 2.0]
    x1 = collect(range(-10.0, 10.0, length=41))
    xdata = hcat(x1, (10 .+ x1 ./ 2))
    ydata = zeros(size(xdata, 1))
    f!(xdata, beta_star, ydata)

    add_noise!(xdata, 5e-2)
    add_noise!(ydata, 10e-2)

    return (f=f!, xdata=xdata, ydata=ydata, beta0=ones(length(beta_star)))
end

function create_case3()
    # m=3, q=2
    function f!(x::Matrix{Float64}, beta::Vector{Float64}, y::Matrix{Float64})
        y[:, 1] .= (beta[1] .* x[:, 1]) .^ 3 .+ x[:, 2] .^ beta[2] .+ exp.(x[:, 3] ./ 2)
        y[:, 2] .= (beta[3] .* x[:, 1]) .^ 2 .+ x[:, 2] .^ beta[2]
        return nothing
    end

    beta_star = [1.0, 2.0, 3.0]
    x1 = collect(range(-1.0, stop=1.0, length=31))
    xdata = hcat(x1, exp.(x1), (x1 .^ 2))
    ydata = zeros(size(xdata, 1), 2)
    f!(xdata, beta_star, ydata)

    add_noise!(xdata, 5e-2)
    add_noise!(ydata, 10e-2)

    return (f=f!, xdata=xdata, ydata=ydata, beta0=[5.0, 5.0, 5.0])
end

const case1 = create_case1()
const case2 = create_case2()
const case3 = create_case3()

end # Cases

using .TestCases

@testset "base-cases" begin

    # Case 1
    sol1 = Odrpack.odr_fit(case1...)
    @test sol1.success
    @test sol1.info == 1

    # Case 2
    sol2 = Odrpack.odr_fit(case2...)
    @test sol2.success
    @test sol2.info == 1

    # Case 3
    sol3 = Odrpack.odr_fit(case3...)
    @test sol3.success
    @test sol3.info == 1

    # x and y don't have the same first dimension
    @test_throws ArgumentError begin
        Odrpack.odr_fit(case1.f, ones(length(case1.xdata) + 1), case1.ydata, case1.beta0)
    end

    # invalid task
    @test_throws ArgumentError begin
        Odrpack.odr_fit(case1..., task="invalid")
    end

end # "base-cases"

@testset "beta-related" begin

    # base case1
    sol1 = Odrpack.odr_fit(case1...)
    @test sol1.success

    # fix some parameters
    sol = Odrpack.odr_fit(case1..., fix_beta=[true, false, false, true])
    @test sol.beta[[1, 4]] ≈ [0.0, 0.0]

    # fix all parameters
    sol = Odrpack.odr_fit(case1..., fix_beta=trues(length(case1.beta0)))
    @test sol.beta ≈ zeros(length(sol.beta))

    # user-defined step_beta
    sol = Odrpack.odr_fit(case1..., step_beta=1e-5 * ones(length(case1.beta0)))
    @test sol.success
    @test all(isapprox.(sol.beta, sol1.beta, rtol=1e-5))

    # user-defined scale_beta
    sol = Odrpack.odr_fit(case1..., scale_beta=[2.0, 2.0, 20.0, 20.0])
    @test sol.success
    @test all(isapprox.(sol.beta, sol1.beta, rtol=1e-5))

    # lower >= beta0
    @test_throws ArgumentError begin
        lower = copy(case1.beta0)
        lower[2:end] .-= 1.0
        Odrpack.odr_fit(case1..., bounds=(lower, nothing))
    end

    # upper <= beta0
    @test_throws ArgumentError begin
        upper = copy(case1.beta0)
        upper[2:end] .+= 1.0
        Odrpack.odr_fit(case1..., bounds=(nothing, upper))
    end

    # invalid lower shape
    @test_throws ArgumentError begin
        lower = -1e99 * ones(length(case1.beta0) + 1)
        Odrpack.odr_fit(case1..., bounds=(lower, nothing))
    end

    # invalid upper shape
    @test_throws ArgumentError begin
        upper = 1e99 * ones(length(case1.beta0) + 1)
        Odrpack.odr_fit(case1..., bounds=(nothing, upper))
    end

    # invalid fix_beta shape
    @test_throws ArgumentError begin
        Odrpack.odr_fit(case1..., fix_beta=[true, false, true])
    end

    # invalid step_beta shape
    @test_throws ArgumentError begin
        Odrpack.odr_fit(case1..., step_beta=[1e-5, 1e-5])
    end

    # invalid scale_beta shape
    @test_throws ArgumentError begin
        Odrpack.odr_fit(case1..., scale_beta=[1.0, 2.0])
    end

end # "beta-related"

@testset "delta0-related" begin

    # base case1
    sol1 = Odrpack.odr_fit(case1...)
    @test sol1.success

    # user-defined delta0=0
    sol = Odrpack.odr_fit(case1..., delta0=zeros(size(case1.xdata)))
    @test sol.success
    @test sol.delta ≈ sol1.delta

    # user-defined delta0 != 0
    sol = Odrpack.odr_fit(case1..., delta0=ones(size(case1.xdata)))
    @test sol.success

    # fix some x
    fix_x = falses(size(case1.xdata))
    fix = [4, 8]
    fix_x[fix] .= true
    sol = Odrpack.odr_fit(case1..., fix_x=fix_x)
    @test sol.delta[fix] ≈ zeros(length(fix))

    # fix some x, broadcast (m,)
    fix_x = falses(size(case3.xdata, 2))
    fix = [2]
    fix_x[fix] .= true
    sol = Odrpack.odr_fit(case3..., fix_x=fix_x)
    @test sol.delta[:, fix] ≈ zeros(size(sol.delta, 1))

    # fix some x, broadcast (n,)
    fix_x = falses(size(case3.xdata, 1))
    fix = [2, 7, 13]
    fix_x[fix] .= true
    sol = Odrpack.odr_fit(case3..., fix_x=fix_x)
    @test sol.delta[fix, :] ≈ zeros(length(fix), size(sol.delta, 2))

    # fix all x (n,)
    fix_x = trues(size(case1.xdata))
    sol = Odrpack.odr_fit(case1..., fix_x=fix_x)
    @test sol.delta ≈ zeros(size(sol.delta))

    # fix all x (n, m)
    fix_x = trues(size(case3.xdata))
    sol = Odrpack.odr_fit(case3..., fix_x=fix_x)

    # user-defined step_delta
    sol3 = Odrpack.odr_fit(case3...)
    @test sol3.success
    for shape in (size(case3.xdata), size(case3.xdata, 1), size(case3.xdata, 2))
        step_delta = fill(1e-5, shape)
        sol = Odrpack.odr_fit(case3..., step_delta=step_delta)
        @test sol.success
        @test all(isapprox.(sol.delta, sol3.delta, atol=1e-4))
    end

    # user-defined scale_delta
    sol3 = Odrpack.odr_fit(case3...)
    @test sol3.success
    for shape in (size(case3.xdata),
        size(case3.xdata, 1),
        size(case3.xdata, 2))
        scale_delta = fill(10.0, shape)
        sol = Odrpack.odr_fit(case3..., scale_delta=scale_delta)
        @test sol.success
        @test all(isapprox.(sol.delta, sol3.delta, atol=1e-4))
    end

    # invalid fix_x shape
    @test_throws ArgumentError begin
        fix_x = [true, false, true]
        Odrpack.odr_fit(case1..., fix_x=fix_x)
    end

    # invalid step_delta shape
    @test_throws ArgumentError begin
        step_delta = [1e-4, 1.0]
        Odrpack.odr_fit(case3..., step_delta=step_delta)
    end

    # invalid scale_delta shape
    @test_throws ArgumentError begin
        scale_delta = [1.0, 1.0, 1.0, 1.0]
        Odrpack.odr_fit(case3..., scale_delta=scale_delta)
    end

    # invalid delta0 shape
    @test_throws ArgumentError begin
        delta0 = zeros(size(case1.ydata))
        Odrpack.odr_fit(case3..., delta0=delta0)
    end

end # "delta0-related"

@testset "weight_x" begin

    # weight_x scalar
    sol = Odrpack.odr_fit(case1..., weight_x=1e100)
    @test all(isapprox.(sol.delta, zeros(size(sol.delta)), atol=1e-90))

    # weight_x scalar
    sol = odr_fit(case1..., weight_x=1e100)
    @test all(isapprox.(sol.delta, zeros(size(sol.delta)), atol=1e-90))

    # weight_x (n,) and m=1
    weight_x = ones(size(case1.xdata))
    fix = [5, 8]
    weight_x[fix] .= 1e100
    sol = odr_fit(case1..., weight_x=weight_x)
    @test all(isapprox.(sol.delta[fix], zeros(size(sol.delta[fix])), atol=1e-90))

    # weight_x (n, m)
    weight_x = ones(size(case3.xdata))
    fix = [5, 14]
    weight_x[fix, :] .= 1e100
    sol = odr_fit(case3..., weight_x=weight_x)
    sol1 = deepcopy(sol)
    @test all(isapprox.(sol.delta[fix, :], zeros(size(sol.delta[fix, :])), atol=1e-90))

    # weight_x (n, 1, m)
    weight_x = reshape(weight_x, size(weight_x, 1), 1, size(weight_x, 2))
    sol = odr_fit(case3..., weight_x=weight_x)
    @test all(isapprox.(sol.delta, sol1.delta, atol=1e-10))
    @test all(isapprox.(sol.eps, sol1.eps, atol=1e-10))

    # weight_x (m,)
    weight_x = ones(size(case3.xdata, 2))
    fix = [2]
    weight_x[fix] .= 1e100
    sol = odr_fit(case3..., weight_x=weight_x)
    sol1 = deepcopy(sol)
    @test all(isapprox.(sol.delta[:, fix], zeros(size(sol.delta[:, fix])), atol=1e-90))

    # weight_x (1, 1, m)
    weight_x = reshape(weight_x, 1, 1, size(weight_x, 1))
    sol = odr_fit(case3..., weight_x=weight_x)
    @test all(isapprox.(sol.delta, sol1.delta, atol=1e-10))
    @test all(isapprox.(sol.eps, sol1.eps, atol=1e-10))

    # weight_x (m, m)
    m = size(case3.xdata, 2)
    weight_x = [i == j ? 1.0 : 0.0 for i in 1:m, j in 1:m]
    fix = [2]
    weight_x[fix, fix] .= 1e100
    sol = odr_fit(case3..., weight_x=weight_x)
    sol1 = deepcopy(sol)
    @test all(isapprox.(sol.delta[:, fix], zeros(size(sol.delta[:, fix])), atol=1e-90))

    # weight_x (1, m, m)
    weight_x = reshape(weight_x, 1, size(weight_x, 1), size(weight_x, 2))
    sol = odr_fit(case3..., weight_x=weight_x)
    @test all(isapprox.(sol.delta, sol1.delta, atol=1e-10))
    @test all(isapprox.(sol.eps, sol1.eps, atol=1e-10))

    # weight_x (n, m, m)
    weight_x = repeat(weight_x, size(case3.xdata, 1), 1, 1)
    sol = odr_fit(case3..., weight_x=weight_x)
    @test all(isapprox.(sol.delta, sol1.delta, atol=1e-10))
    @test all(isapprox.(sol.eps, sol1.eps, atol=1e-10))

    # weight_x has invalid shape
    weight_x = ones(1, 1, 1)
    @test_throws ArgumentError odr_fit(case3..., weight_x=weight_x)

end # "weight_x"