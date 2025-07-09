using Test
using Odrpack
using Random

const SEED = 1234567890
const rng = Random.MersenneTwister(SEED)

@testset "odr_fit" begin

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

        return Dict(
            "f!" => f!,
            "xdata" => x,
            "ydata" => y,
            "beta0" => zeros(length(beta_star))
        )
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

        return Dict(
            "f!" => f!,
            "xdata" => xdata,
            "ydata" => ydata,
            "beta0" => ones(length(beta_star))
        )
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

        return Dict(
            "f!" => f!,
            "xdata" => xdata,
            "ydata" => ydata,
            "beta0" => [5.0, 5.0, 5.0]
        )
    end

    # Case 1
    case1 = create_case1()
    sol1 = Odrpack.odr_fit(case1["f!"], case1["xdata"], case1["ydata"], case1["beta0"])
    @test sol1.success
    @test sol1.info == 1

    # Case 2
    case2 = create_case2()
    sol2 = Odrpack.odr_fit(case2["f!"], case2["xdata"], case2["ydata"], case2["beta0"])
    @test sol2.success
    @test sol2.info == 1

    # Case 3
    case3 = create_case3()
    sol3 = Odrpack.odr_fit(case3["f!"], case3["xdata"], case3["ydata"], case3["beta0"])
    @test sol3.success
    @test sol3.info == 1

end