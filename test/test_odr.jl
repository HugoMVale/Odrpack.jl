using Test
using Odrpack

module TestCases

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

case1 = create_case1()
case2 = create_case2()
case3 = create_case3()

end # Cases

@testset "base-cases" begin
    using .TestCases


    # Case 1
    sol1 = Odrpack.odr_fit(TestCases.case1...)
    @test sol1.success
    @test sol1.info == 1

    # Case 2
    sol2 = Odrpack.odr_fit(TestCases.case2...)
    @test sol2.success
    @test sol2.info == 1

    # Case 3
    sol3 = Odrpack.odr_fit(TestCases.case3...)
    @test sol3.success
    @test sol3.info == 1

end # "base-cases"

@testset "delta0-related" begin
    using .TestCases

    # user-defined delta0
    sol = Odrpack.odr_fit(TestCases.case1..., delta0=ones(size(TestCases.case1.xdata)))
    @test sol.success

    # fix some x
    fix_x = falses(size(TestCases.case1.xdata))
    fix = [4, 8]
    fix_x[fix] .= true
    sol = Odrpack.odr_fit(TestCases.case1..., fix_x=fix_x)
    @test sol.delta[fix] ≈ zeros(length(fix))

    # fix some x, broadcast (m,)
    fix_x = falses(size(TestCases.case3.xdata, 2))
    fix = [2]
    fix_x[fix] .= true
    sol = Odrpack.odr_fit(TestCases.case3..., fix_x=fix_x)
    @test sol.delta[:, fix] ≈ zeros(size(sol.delta, 1))

    # fix some x, broadcast (n,)
    fix_x = falses(size(TestCases.case3.xdata, 1))
    fix = [2, 7, 13]
    fix_x[fix] .= true
    sol = Odrpack.odr_fit(TestCases.case3..., fix_x=fix_x)
    @test sol.delta[fix, :] ≈ zeros(length(fix), size(sol.delta, 2))

    # fix all x (n,)
    fix_x = trues(size(TestCases.case1.xdata))
    sol = Odrpack.odr_fit(TestCases.case1..., fix_x=fix_x)
    @test sol.delta ≈ zeros(size(sol.delta))

    # fix all x (n,m)
    fix_x = trues(size(TestCases.case3.xdata))
    sol = Odrpack.odr_fit(TestCases.case3..., fix_x=fix_x)

    # user step_delta
    sol3 = Odrpack.odr_fit(TestCases.case3...)
    @test sol3.success
    for shape in (size(TestCases.case3.xdata),
        size(TestCases.case3.xdata, 1),
        size(TestCases.case3.xdata, 2))
        step_delta = fill(1e-5, shape)
        sol = Odrpack.odr_fit(TestCases.case3..., step_delta=step_delta)
        @test sol.success
        @test all(isapprox.(sol.delta, sol3.delta, atol=1e-4))
    end

    # user scale_delta
    sol3 = Odrpack.odr_fit(TestCases.case3...)
    @test sol3.success
    for shape in (size(TestCases.case3.xdata),
        size(TestCases.case3.xdata, 1),
        size(TestCases.case3.xdata, 2))
        scale_delta = fill(10.0, shape)
        sol = Odrpack.odr_fit(TestCases.case3..., scale_delta=scale_delta)
        @test sol.success
        @test all(isapprox.(sol.delta, sol3.delta, atol=1e-4))
    end

    # invalid inputs
    @test_throws ArgumentError begin
        fix_x = [true, false, true]  # invalid shape
        Odrpack.odr_fit(TestCases.case1..., fix_x=fix_x)
    end

    @test_throws ArgumentError begin
        step_delta = [1e-4, 1.0]  # invalid shape
        Odrpack.odr_fit(TestCases.case3..., step_delta=step_delta)
    end

    @test_throws ArgumentError begin
        scale_delta = [1.0, 1.0, 1.0, 1.0]  # invalid shape
        Odrpack.odr_fit(TestCases.case3..., scale_delta=scale_delta)
    end

    @test_throws ArgumentError begin
        delta0 = zeros(size(TestCases.case1.ydata))  # invalid shape
        Odrpack.odr_fit(TestCases.case3..., delta0=delta0)
    end

end # "delta0-related"