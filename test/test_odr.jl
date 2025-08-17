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
        y .= beta[1] .+ beta[2] * x .+ beta[3] * x .^ 2 .+ beta[4] * x .^ 3
        return nothing
    end

    beta_star = [1.0, -2.0, 0.1, -0.1]
    xdata = collect(range(-10.0, 10.0, length=21))
    ydata = zeros(size(xdata, 1))
    f!(xdata, beta_star, ydata)

    add_noise!(xdata, 5e-2)
    add_noise!(ydata, 10e-2)

    return (f=f!, xdata=xdata, ydata=ydata, beta0=zeros(length(beta_star)))
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
        y[:, 1] .= (beta[1] * x[:, 1]) .^ 3 .+ x[:, 2] .^ beta[2] .+ exp.(x[:, 3] ./ 2)
        y[:, 2] .= (beta[3] * x[:, 1]) .^ 2 .+ x[:, 2] .^ beta[2]
        return nothing
    end

    beta_star = [1.0, 2.0, 3.0]
    x1 = collect(range(-1.0, 1.0, length=31))
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
    sol1 = odr_fit(case1...)
    @test sol1.success
    @test sol1.info == 1

    # Case 2
    sol2 = odr_fit(case2...)
    @test sol2.success
    @test sol2.info == 1

    # Case 3
    sol3 = odr_fit(case3...)
    @test sol3.success
    @test sol3.info == 1

    # x and y don't have the same first dimension
    @test_throws ArgumentError begin
        odr_fit(case1.f, ones(length(case1.xdata) + 1), case1.ydata, case1.beta0)
    end

    # invalid task
    @test_throws ArgumentError begin
        odr_fit(case1..., task="invalid")
    end

end # "base-cases"

@testset "beta-related" begin

    # base case1
    sol1 = odr_fit(case1...)
    @test sol1.success

    # fix some parameters
    sol = odr_fit(case1..., fix_beta=[true, false, false, true])
    @test sol.beta[[1, 4]] ≈ [0.0, 0.0]

    # fix all parameters
    sol = odr_fit(case1..., fix_beta=trues(length(case1.beta0)))
    @test sol.beta ≈ zeros(length(sol.beta))

    # user-defined step_beta
    sol = odr_fit(case1..., step_beta=1e-5 * ones(length(case1.beta0)))
    @test sol.success
    @test all(isapprox.(sol.beta, sol1.beta, rtol=1e-5))

    # user-defined scale_beta
    sol = odr_fit(case1..., scale_beta=[2.0, 2.0, 20.0, 20.0])
    @test sol.success
    @test all(isapprox.(sol.beta, sol1.beta, rtol=1e-5))

    # lower >= beta0
    @test_throws ArgumentError begin
        lower = copy(case1.beta0)
        lower[2:end] .-= 1.0
        odr_fit(case1..., bounds=(lower, nothing))
    end

    # upper <= beta0
    @test_throws ArgumentError begin
        upper = copy(case1.beta0)
        upper[2:end] .+= 1.0
        odr_fit(case1..., bounds=(nothing, upper))
    end

    # invalid lower shape
    @test_throws ArgumentError begin
        lower = -1e99 * ones(length(case1.beta0) + 1)
        odr_fit(case1..., bounds=(lower, nothing))
    end

    # invalid upper shape
    @test_throws ArgumentError begin
        upper = 1e99 * ones(length(case1.beta0) + 1)
        odr_fit(case1..., bounds=(nothing, upper))
    end

    # invalid fix_beta shape
    @test_throws ArgumentError begin
        odr_fit(case1..., fix_beta=[true, false, true])
    end

    # invalid step_beta shape
    @test_throws ArgumentError begin
        odr_fit(case1..., step_beta=[1e-5, 1e-5])
    end

    # invalid scale_beta shape
    @test_throws ArgumentError begin
        odr_fit(case1..., scale_beta=[1.0, 2.0])
    end

end # "beta-related"

@testset "delta0-related" begin

    # base case1
    sol1 = odr_fit(case1...)
    @test sol1.success

    # user-defined delta0=0
    sol = odr_fit(case1..., delta0=zeros(size(case1.xdata)))
    @test sol.success
    @test sol.delta ≈ sol1.delta

    # user-defined delta0!=0
    sol = odr_fit(case1..., delta0=ones(size(case1.xdata)))
    @test sol.success

    # fix some x
    fix_x = falses(size(case1.xdata))
    fix = [4, 8]
    fix_x[fix] .= true
    sol = odr_fit(case1..., fix_x=fix_x)
    @test sol.delta[fix] ≈ zeros(length(fix))

    # fix some x, broadcast (m,)
    fix_x = falses(size(case3.xdata, 2))
    fix = [2]
    fix_x[fix] .= true
    sol = odr_fit(case3..., fix_x=fix_x)
    @test sol.delta[:, fix] ≈ zeros(size(sol.delta[:, fix]))

    # fix some x, broadcast (n,)
    fix_x = falses(size(case3.xdata, 1))
    fix = [2, 7, 13]
    fix_x[fix] .= true
    sol = odr_fit(case3..., fix_x=fix_x)
    @test sol.delta[fix, :] ≈ zeros(size(sol.delta[fix, :]))

    # fix all x (n,)
    fix_x = trues(size(case1.xdata))
    sol = odr_fit(case1..., fix_x=fix_x)
    @test sol.delta ≈ zeros(size(sol.delta))

    # fix all x (n, m)
    fix_x = trues(size(case3.xdata))
    sol = odr_fit(case3..., fix_x=fix_x)
    @test sol.delta ≈ zeros(size(sol.delta))

    # user-defined step_delta
    sol3 = odr_fit(case3...)
    @test sol3.success
    for shape in (size(case3.xdata), size(case3.xdata, 1), size(case3.xdata, 2))
        step_delta = fill(1e-5, shape)
        sol = odr_fit(case3..., step_delta=step_delta)
        @test sol.success
        @test all(isapprox.(sol.delta, sol3.delta, atol=1e-4))
    end

    # user-defined scale_delta
    sol3 = odr_fit(case3...)
    @test sol3.success
    for shape in (size(case3.xdata), size(case3.xdata, 1), size(case3.xdata, 2))
        scale_delta = fill(10.0, shape)
        sol = odr_fit(case3..., scale_delta=scale_delta)
        @test sol.success
        @test all(isapprox.(sol.delta, sol3.delta, atol=1e-4))
    end

    # invalid fix_x shape
    @test_throws ArgumentError begin
        fix_x = [true, false, true]
        odr_fit(case1..., fix_x=fix_x)
    end

    # invalid step_delta shape
    @test_throws ArgumentError begin
        step_delta = [1e-4, 1.0]
        odr_fit(case3..., step_delta=step_delta)
    end

    # invalid scale_delta shape
    @test_throws ArgumentError begin
        scale_delta = [1.0, 1.0, 1.0, 1.0]
        odr_fit(case3..., scale_delta=scale_delta)
    end

    # invalid delta0 shape
    @test_throws ArgumentError begin
        delta0 = zeros(size(case1.ydata))
        odr_fit(case3..., delta0=delta0)
    end

end # "delta0-related"

@testset "weight_x" begin

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

@testset "weight_y" begin

    # weight_y scalar
    sol = odr_fit(case1..., weight_y=1e100)
    @test all(isapprox.(sol.eps, zeros(size(sol.eps)), atol=1e-10))

    # weight_y (n,) and q=1
    weight_y = ones(size(case1.ydata))
    fix = [5, 8]
    weight_y[fix] .= 1e100
    sol = odr_fit(case1..., weight_y=weight_y)
    @test all(isapprox.(sol.eps[fix], zeros(size(sol.eps[fix])), atol=1e-10))

    # weight_y (n, q)
    weight_y = ones(size(case3.ydata))
    fix = [5, 14]
    weight_y[fix, :] .= 1e100
    sol = odr_fit(case3..., weight_y=weight_y)
    sol1 = deepcopy(sol)
    @test all(isapprox.(sol.eps[fix, :], zeros(size(sol.eps[fix, :])), atol=1e-10))

    # weight_y (n, 1, q)
    weight_y = reshape(weight_y, size(weight_y, 1), 1, size(weight_y, 2))
    sol = odr_fit(case3..., weight_y=weight_y)
    @test all(isapprox.(sol.delta, sol1.delta, atol=1e-50))
    @test all(isapprox.(sol.eps, sol1.eps, atol=1e-50))

    # weight_y (q,)
    weight_y = ones(size(case3.ydata, 2))
    fix = [1]
    weight_y[fix] .= 1e100
    sol = odr_fit(case3..., weight_y=weight_y)
    sol1 = deepcopy(sol)
    @test all(isapprox.(sol.eps[:, fix], zeros(size(sol.eps[:, fix])), atol=1e-10))

    # weight_y (1, 1, q)
    weight_y = reshape(weight_y, 1, 1, size(weight_y, 1))
    sol = odr_fit(case3..., weight_y=weight_y)
    @test all(isapprox.(sol.delta, sol1.delta, atol=1e-50))
    @test all(isapprox.(sol.eps, sol1.eps, atol=1e-50))

    # weight_y (q, q)
    q = size(case3.ydata, 2)
    weight_y = [i == j ? 1.0 : 0.0 for i in 1:q, j in 1:q]
    fix = [1]
    weight_y[fix, fix] .= 1e100
    sol = odr_fit(case3..., weight_y=weight_y)
    sol1 = deepcopy(sol)
    @test all(isapprox.(sol.eps[:, fix], zeros(size(sol.eps[:, fix])), atol=1e-10))

    # weight_y (1, q, q)
    weight_y = reshape(weight_y, 1, size(weight_y, 1), size(weight_y, 2))
    sol = odr_fit(case3..., weight_y=weight_y)
    @test all(isapprox.(sol.delta, sol1.delta, atol=1e-50))
    @test all(isapprox.(sol.eps, sol1.eps, atol=1e-50))

    # weight_y (n, q, q)
    weight_y = repeat(weight_y, size(case3.ydata, 1), 1, 1)
    sol = odr_fit(case3..., weight_y=weight_y)
    @test all(isapprox.(sol.delta, sol1.delta, atol=1e-50))
    @test all(isapprox.(sol.eps, sol1.eps, atol=1e-50))

    # weight_y has invalid shape
    weight_y = ones(1, 1, 1)
    @test_throws ArgumentError odr_fit(case3..., weight_y=weight_y)

end # "weight_y"

@testset "parameters" begin

    # maxit
    sol = odr_fit(case1..., maxit=2)
    @test sol.info == 4
    @test occursin("iteration limit", lowercase(sol.stopreason))

    # sstol
    sstol = 0.123
    sol = odr_fit(case1..., sstol=sstol)
    @test sol.info == 1
    rwork_idx = Odrpack.loc_rwork(length(case1.xdata), 1, 1, length(case1.beta0), 1, 1, true)
    @test isapprox(sol._rwork[rwork_idx.sstol+1], sstol)

    # partol
    partol = 0.456
    sol = odr_fit(case1..., partol=partol)
    @test sol.info == 2
    @test isapprox(sol._rwork[rwork_idx.partol+1], partol)

    # taufac
    taufac = 0.6969
    sol = odr_fit(case1..., taufac=taufac)
    @test sol.info == 1
    @test isapprox(sol._rwork[rwork_idx.taufac+1], taufac)

end # "parameters"

@testset "rptfile-and-errfile" begin

    # write to report file
    for (report, rptsize) in zip(["none", "short"], [0, 2600])
        rptfile = tempname()
        odr_fit(case1..., report=report, rptfile=rptfile)
        @test isfile(rptfile)
        @test abs(filesize(rptfile) - rptsize) < 200
    end

    # write to error file
    errfile = tempname()
    odr_fit(case1..., report="short", errfile=errfile)
    @test isfile(errfile) # && filesize(errfile) > 0

    # write to report and error file
    rptfile = tempname()
    errfile = tempname()
    odr_fit(case1..., report="short", rptfile=rptfile, errfile=errfile)
    @test isfile(rptfile) && filesize(rptfile) > 2500
    @test isfile(errfile) # && filesize(errfile) > 0

end # "rptfile-and-errfile"

@testset "OLS" begin
    sol1 = odr_fit(case1..., task="OLS")
    sol2 = odr_fit(case1..., weight_x=1e100)

    @test all(isapprox.(sol1.beta, sol2.beta, atol=1e-10))
    @test all(isapprox.(sol1.delta, zeros(size(sol1.delta)), atol=1e-90))
end # "OLS"

@testset "implicit-model" begin

    # "odrpack's example2"
    xdata = [0.50 -0.12;
        1.20 -0.60;
        1.60 -1.00;
        1.86 -1.40;
        2.12 -2.54;
        2.36 -3.36;
        2.44 -4.00;
        2.36 -4.75;
        2.06 -5.25;
        1.74 -5.64;
        1.34 -5.97;
        0.90 -6.32;
        -0.28 -6.44;
        -0.78 -6.44;
        -1.36 -6.41;
        -1.90 -6.25;
        -2.50 -5.88;
        -2.88 -5.50;
        -3.18 -5.24;
        -3.44 -4.86]
    ydata = fill(0.0, size(xdata, 1))
    beta0 = [-1.0, -3.0, 0.09, 0.02, 0.08]

    beta_ref = [-9.99380462E-01, -2.93104890E+00, 8.75730642E-02, 1.62299601E-02, 7.97538109E-02]

    function f!(x::Matrix{Float64}, beta::Vector{Float64}, y::Vector{Float64})
        v = @view x[:, 1]
        h = @view x[:, 2]
        y .= beta[3] * (v .- beta[1]) .^ 2 .+ 2 * beta[4] * (v .- beta[1]) .* (h .- beta[2]) .+ beta[5] * (h .- beta[2]) .^ 2 .- 1
        return nothing
    end

    sol = odr_fit(f!, xdata, ydata, beta0, task="implicit-ODR")
    @test all(isapprox.(sol.beta, beta_ref, rtol=1e-5))

end # "implicit-model"

@testset "OdrStop-exception" begin
    xdata = [1.0, 2.0, 3.0, 4.0]
    ydata = [1.0, 2.0, 3.0, 4.0]
    beta0 = [1.0, 1.0]

    function f!(x::Vector{Float64}, beta::Vector{Float64}, y::Vector{Float64})
        if beta[1] > 0
            throw(OdrStop("Oops!"))
        end
        y .= beta[1] * exp.(beta[2] * x)
        return nothing
    end

    sol = odr_fit(f!, xdata, ydata, beta0)
    @test !sol.success
    @test sol.info == 52000

end # "OdrStop-exception

@testset "jac_beta-and-jac_x" begin

    # odrpack's example5
    xdata = [0.982, 1.998, 4.978, 6.01]
    ydata = [2.7, 7.4, 148.0, 403.0]
    beta0 = [2.0, 0.5]
    bounds = ([0.0, 0.0], [10.0, 0.9])

    beta_ref = [1.63337602, 0.9]
    delta_ref = [-0.36886137, -0.31273038, 0.029287, 0.11031505]

    function f!(x::Vector{Float64}, beta::Vector{Float64}, y::Vector{Float64})
        y .= beta[1] * exp.(beta[2] * x)
        return nothing
    end

    function jac_beta!(x::Vector{Float64}, beta::Vector{Float64}, jacb::Array{Float64})
        jacb[:, 1, 1] .= exp.(beta[2] * x)
        jacb[:, 2, 1] .= beta[1] * x .* exp.(beta[2] * x)
        return nothing
    end

    function jac_x!(x::Vector{Float64}, beta::Vector{Float64}, jacx::Array{Float64})
        jacx[:, 1, 1] .= beta[1] * beta[2] * exp.(beta[2] * x)
        return nothing
    end

    # ODR without jacobian
    for diff_scheme in ["forward", "central"]
        sol = odr_fit(f!, xdata, ydata, beta0, bounds=bounds, diff_scheme=diff_scheme)
        @test all(isapprox.(sol.beta, beta_ref, rtol=1e-5))
        @test all(isapprox.(sol.delta, delta_ref, atol=1e-5))
    end

    # ODR with jacobian
    sol = odr_fit(f!, xdata, ydata, beta0, bounds=bounds,
        (jac_beta!)=jac_beta!, (jac_x!)=jac_x!)
    @test all(isapprox.(sol.beta, beta_ref, rtol=1e-6))
    @test all(isapprox.(sol.delta, delta_ref, atol=1e-6))

    # OLS with jacobian
    sol1 = odr_fit(f!, xdata, ydata, beta0, weight_x=1e100)
    sol2 = odr_fit(f!, xdata, ydata, beta0, (jac_beta!)=(jac_beta!), task="OLS")
    @test all(isapprox.(sol2.beta, sol1.beta, rtol=1e-6))
    @test all(isapprox.(sol2.delta, zeros(length(xdata))))

    # missing jac_beta
    @test_throws ArgumentError odr_fit(f!, xdata, ydata, beta0, (jac_x!)=jac_x!)

    # missing jac_x
    @test_throws ArgumentError odr_fit(f!, xdata, ydata, beta0, (jac_beta!)=jac_beta!)

    # invalid diff_scheme
    @test_throws ArgumentError odr_fit(f!, xdata, ydata, beta0, diff_scheme="invalid")

end # "jac_beta-and-jac_x"
