module Odrpack

include("OdrpackAux.jl")
using .OdrpackAux

export OdrResult, OdrStop, odr_fit

"""
    OdrResult

Results of an Orthogonal Distance Regression (ODR) computation.

# Fields
- `beta::Vector{Float64}`: Estimated parameters of the model.
- `delta::Union{Vector{Float64},Matrix{Float64}}`: Differences between the observed and fitted `x` values.
- `eps::Union{Vector{Float64},Matrix{Float64}}`: Differences between the observed and fitted `y` values.
- `xplusd::Union{Vector{Float64},Matrix{Float64}}`: Adjusted `x` values after fitting, `x + delta`.
- `yest::Union{Vector{Float64},Matrix{Float64}}`: Estimated `y` values corresponding to the fitted model, `y + eps`.
- `sd_beta::Vector{Float64}`: Standard deviations of the estimated parameters.
- `cov_beta::Matrix{Float64}`: Covariance matrix of the estimated parameters.
- `res_var::Float64`: Residual variance, indicating the variance of the residuals.
- `nfev::Int`: Number of function evaluations during the fitting process.
- `njev::Int`: Number of Jacobian evaluations during the fitting process.
- `niter::Int`: Number of iterations performed in the optimization process.
- `irank::Int`: Rank of the Jacobian matrix at the solution.
- `inv_condnum::Float64`: Inverse of the condition number of the Jacobian matrix.
- `info::Int`: Status code of the fitting process (e.g., success or failure).
- `stopreason::String`: Message indicating the reason for stopping.
- `success::Bool`: Whether the fitting process was successful.
- `sum_square::Float64`: Sum of squared residuals (including both `delta` and `eps`).
- `sum_square_delta::Float64`: Sum of squared differences between observed and fitted `x` values.
- `sum_square_eps::Float64`: Sum of squared differences between observed and fitted `y` values.
- `_iwork::Vector{Int32}`: Integer workspace array used internally by `odrpack`. Typically for advanced debugging.
- `_rwork::Vector{Float64}`: Floating-point workspace array used internally by `odrpack`. Typically for advanced debugging.
"""
struct OdrResult
    beta::Vector{Float64}
    delta::Union{Vector{Float64},Matrix{Float64}}
    eps::Union{Vector{Float64},Matrix{Float64}}
    xplusd::Union{Vector{Float64},Matrix{Float64}}
    yest::Union{Vector{Float64},Matrix{Float64}}
    sd_beta::Vector{Float64}
    cov_beta::Matrix{Float64}
    res_var::Float64
    nfev::Int
    njev::Int
    niter::Int
    irank::Int
    inv_condnum::Float64
    info::Int
    stopreason::String
    success::Bool
    sum_square::Float64
    sum_square_delta::Float64
    sum_square_eps::Float64
    _iwork::Vector{Int32}
    _rwork::Vector{Float64}
end

"""
Return a human-readable message based on the stopping condition returned by `odrpack` in 
the `info` argument of the result.

# Arguments
- `info::Int`: value of the `info` argument returned by `odrpack`. This value is used to
  determine the stopping condition.

# Returns
- `::String`: human-readable string describing the stopping condition.
"""
function get_stopreason_message(info::Integer)::String
    message = ""
    if info == 1
        message = "Sum of squares convergence."
    elseif info == 2
        message = "Parameter convergence."
    elseif info == 3
        message = "Sum of squares and parameter convergence."
    elseif info == 4
        message = "Iteration limit reached."
    elseif info >= 5
        message = "Questionable results or fatal errors detected. See report and error message."
    end
    return message
end

"""
    OdrStop <: Exception

Exception to stop the regression.

This exception can be raised in the model function or its Jacobians to
stop the regression process.
"""
struct OdrStop <: Exception
    msg::String
end
OdrStop() = OdrStop("OdrStop exception occurred.")


"""
    odr_fit(f!, xdata, ydata, beta0; kwargs...) -> OdrResult

Solve a weighted orthogonal distance regression (ODR) problem, also
known as errors-in-variables regression.

# Arguments
- `f!::Function`: Function to be fitted, with the signature `f!(x, beta, y)`.
  It must modify `y` in-place to have the same shape as `ydata`.
- `xdata::VecOrMat{Float64}`: Array of shape `(n,)` or `(n, m)` containing
  the observed values of the explanatory variable(s).
- `ydata::VecOrMat{Float64}`: Array of shape `(n,)` or `(n, q)` containing
  the observed values of the response variable(s). When the model is explicit, `ydata` must 
  contain a value for each observation. To ignore specific values (e.g., missing data), set the
  corresponding entry in `weight_y` to zero. When the model is implicit, `ydata` is not used 
  (but must be defined).
- `beta0::Vector{Float64}`: Array of shape `(np,)` with the initial guesses of the model
  parameters, within the optional bounds specified by `bounds`.

# Keywords
- `weight_x::Union{Float64,Vector{Float64},Matrix{Float64},Array{Float64,3},Nothing}=nothing`:
  Scalar or array specifying how the errors on `xdata` are to be weighted. If `weight_x` is a 
  scalar, then it is used for all data points. If `weight_x` is an array of shape `(n,)` and 
  `m==1`, then `weight_x[i]` represents the weight for `xdata[i]`. If `weight_x` is an array of
  shape `(m,)`, then it represents the diagonal of the covariant weighting matrix for all data
  points. If `weight_x` is an array of shape `(m, m)`, then it represents the full covariant 
  weighting matrix for all data points. If `weight_x` is an array of shape `(n, m)`, then 
  `weight_x[i, :]` represents the diagonal of the covariant weighting matrix for `xdata[i, :]`.
  If `weight_x` is an array of shape `(n, m, m)`, then `weight_x[i, :, :]` represents the full
  covariant weighting matrix for `xdata[i, :]`. For a comprehensive description of the options,
  refer to page 26 of the ODRPACK95 guide. By default, `weight_x` is set to one for all `xdata`
  points.
- `weight_y::Union{Float64,Vector{Float64},Matrix{Float64},Array{Float64,3},Nothing}=nothing`:
  Scalar or array specifying how the errors on `ydata` are to be weighted. If `weight_y` is a 
  scalar, then it is used for all data points. If `weight_y` is an array of shape `(n,)` and 
  `q==1`, then `weight_y[i]` represents the weight for `ydata[i]`. If `weight_y` is an array of
  shape `(q)`, then it represents the diagonal of the covariant weighting matrix for all data
  points. If `weight_y` is an array of shape `(q, q)`, then it represents the full covariant 
  weighting matrix for all data points. If `weight_y` is an array of shape `(n, q)`, then 
  `weight_y[i, :]` represents the diagonal of the covariant weighting matrix for `ydata[i, :]`.
  If `weight_y` is an array of shape `(n, q, q)`, then `weight_y[i, :, :]` represents the full
  covariant weighting matrix for `ydata[i, :]`. For a comprehensive description of the options,
  refer to page 25 of the ODRPACK95 guide. By default, `weight_y` is set to one for all `ydata`
  points.
- `bounds::Tuple{Union{Vector{Float64},Nothing},Union{Vector{Float64},Nothing}}=(nothing, nothing)`:
  Tuple of arrays with the same shape as `beta0`, specifying the lower and upper bounds of the 
  model parameters. The first array contains the lower bounds, and the second contains the upper
  bounds. By default, the bounds are set to negative and positive infinity, respectively, for 
  all elements of `beta`.
- `task::String="explicit-ODR"`:
  Specifies the regression task to be performed. `"explicit-ODR"` solves an orthogonal distance
  regression problem with an explicit model. `"implicit-ODR"` handles models defined implicitly.
  `"OLS"` performs ordinary least squares fitting.
- `fix_beta::Union{Vector{Bool},Nothing}=nothing`:
  Array with the same shape as `beta0`, specifying which elements of `beta` are to be held fixed.
  `true` means the parameter is fixed; `false` means it is adjustable. By default, all elements
  of `beta` are set to `false`. 
- `fix_x::Union{VecOrMat{Bool},Nothing}=nothing`:
  Array with the same shape as `xdata`, specifying which elements of `xdata` are to be held fixed.
  Alternatively, it can be a rank-1 array of shape `(m,)` or `(n,)`, in which case it will be 
  broadcast along the other axis. `true` means the element is fixed; `false` means it is adjustable.
  By default, in orthogonal distance regression mode, all elements of `fix_x` are set to `false`.
  In ordinary least squares mode (`task="OLS"`), all `xdata` values are automatically treated 
  as fixed.
- `jac_beta!::Union{Function,Nothing}=nothing`:
  Jacobian of the function to be fitted with respect to `beta`, with the signature `jac_beta!(x, beta, jb)`.
  It must modify the result `jb` in-place, with shape `(n, np, q)`. By default, the Jacobian is
  approximated numerically using the finite difference scheme specified by `diff_scheme`.
- `jac_x!::Union{Function,Nothing}=nothing`:
  Jacobian of the function to be fitted with respect to `x`, with the signature `jac_x!(x, beta, jx)`.
  It must modify the result `jx` in-place, with shape `(n, m, q)`. By default, the Jacobian is 
  approximated numerically using the finite difference scheme specified by `diff_scheme`.
- `delta0::Union{VecOrMat{Float64},Nothing}=nothing`:
  Array with the same shape as `xdata`, containing the initial guesses of the errors in the 
  explanatory variable. By default, `delta0` is set to zero for all elements of `xdata`.
- `diff_scheme::String="forward"`:
  Finite difference scheme used to approximate the Jacobian matrices when the user does not 
  provide `jac_beta!` and `jac_x!`. The default method is forward differences. Central 
  differences are generally more accurate but require one additional `f!` evaluation per partial
  derivative.
- `report::String="none"`:
  Specifies the level of computation reporting. `"none"` disables all output. `"short"` prints 
  a brief initial and final summary. `"long"` provides a detailed initial and final summary. 
  `"iteration"` outputs information at each iteration step in addition to the detailed summaries.
  This is useful for debugging or monitoring progress.
- `maxit::Integer=50`:
  Maximum number of allowed iterations.
- `ndigit::Union{Integer,Nothing}=nothing`: Number of reliable decimal digits in the values 
  computed by the model function `f!` and its Jacobians `jac_beta!`, and `jac_x!`. By default,
  the value is numerically determined by evaluating `f!`. 
- `taufac::Union{Float64,Nothing}=nothing`:
  Factor ranging from 0 to 1 to initialize the trust region radius. The default value is 1. 
  Reducing `taufac` may be appropriate if, at the first iteration, the computed results for the
  full Gauss-Newton step cause an overflow, or cause `beta` and/or `delta` to leave the region
  of interest. 
- `sstol::Union{Float64,Nothing}=nothing`:
  Factor ranging from 0 to 1 specifying the stopping tolerance for the sum of the squares 
  convergence. The default value is `eps()^(1/2)`, where `eps()` is the machine precision in 
  `Float64`.
- `partol::Union{Float64,Nothing}=nothing`:
  Factor ranging from 0 to 1 specifying the stopping tolerance for parameter convergence 
  (i.e., `beta` and `delta`). When the model is explicit, the default value is `eps()^(2/3)`, 
  and when the model is implicit, the default value is `eps()^(1/3)`, where `eps()` is the machine
  precision in `Float64`.
- `step_beta::Union{Vector{Float64},Nothing}=nothing`:
  Array with the same shape as `beta0` containing the _relative_ step sizes used to compute the
  finite difference derivatives with respect to the model parameters. By default, the step size
  is set internally based on the value of `ndigit` and the type of finite differences specified
  by `diff_scheme`. For additional details, refer to pages 31 and 78 of the ODRPACK95 guide.
- `step_delta::Union{VecOrMat{Float64},Nothing}=nothing`:
  Array with the same shape as `xdata`, containing the _relative_ step sizes used to compute the
  finite difference derivatives with respect to the errors in the explanatory variable. 
  Alternatively, it can be a rank-1 array of shape `(m,)` or `(n,)`, in which case it will be 
  broadcast along the other axis. By default, step size is set internally based on the value
  of `ndigit` and the type of finite differences specified by `diff_scheme`. For additional 
  details, refer to pages 31 and 78 of the ODRPACK95 guide.
- `scale_beta::Union{Vector{Float64},Nothing}=nothing`:
  Array with the same shape as `beta0` containing the scale values of the model parameters. 
  Scaling is used to improve the numerical stability of the regression, but does not affect the
  problem specification. Scaling should not be confused with the weighting matrices `weight_x` 
  and `weight_y`. By default, the scale is set internally based on the relative magnitudes of 
  `beta`. For further details, refer to pages 32 and 84 of the ODRPACK95 guide.
- `scale_delta::Union{VecOrMat{Float64},Nothing}=nothing`:
  Array with the same shape as `xdata`, containing the scale values of the errors in the 
  explanatory variable. Alternatively, it can be a rank-1 array of shape `(m,)` or `(n,)`, in 
  which case it will be broadcast along the other axis. Scaling is used to improve the numerical
  stability of the regression, but does not affect the problem specification. Scaling should 
  not be confused with the weighting matrices `weight_x` and `weight_y`. By default, the scale 
  is set internally based on the relative magnitudes of `xdata`. For further details, refer to 
  pages 32 and 85 of the ODRPACK95 guide.
- `rptfile::Union{String,Nothing}=nothing`:
  File name for storing the computation reports, as defined by `report`. By default, the reports
  are sent to standard output.
- `errfile::Union{String,Nothing}=nothing`:
  File name for storing the error reports, as defined by `report`. By default, the reports are 
  sent to standard output.

# Returns
- `OdrResult`: An object containing the results of the regression.

# References
[1] Jason W. Zwolak, Paul T. Boggs, and Layne T. Watson.
    Algorithm 869: ODRPACK95: A weighted orthogonal distance regression code 
    with bound constraints. ACM Trans. Math. Softw. 33, 4 (August 2007), 27-es.
    https://doi.org/10.1145/1268776.1268782

[2] Jason W. Zwolak, Paul T. Boggs, and Layne T. Watson. User's Reference
    Guide for ODRPACK95, 2005.
    https://github.com/HugoMVale/odrpack95/blob/main/original/Doc/guide.pdf
"""
function odr_fit(
    f!::Function,
    xdata::VecOrMat{Float64},
    ydata::VecOrMat{Float64},
    beta0::Vector{Float64};
    #
    weight_x::Union{Float64,Vector{Float64},Matrix{Float64},Array{Float64,3},Nothing}=nothing,
    weight_y::Union{Float64,Vector{Float64},Matrix{Float64},Array{Float64,3},Nothing}=nothing,
    bounds::Tuple{Union{Vector{Float64},Nothing},Union{Vector{Float64},Nothing}}=(nothing, nothing),
    task::String="explicit-ODR",
    fix_beta::Union{Vector{Bool},Nothing}=nothing,
    fix_x::Union{VecOrMat{Bool},Nothing}=nothing,
    jac_beta!::Union{Function,Nothing}=nothing,
    jac_x!::Union{Function,Nothing}=nothing,
    delta0::Union{VecOrMat{Float64},Nothing}=nothing,
    diff_scheme::String="forward",
    report::String="none",
    maxit::Integer=50,
    ndigit::Union{Integer,Nothing}=nothing,
    taufac::Union{Float64,Nothing}=nothing,
    sstol::Union{Float64,Nothing}=nothing,
    partol::Union{Float64,Nothing}=nothing,
    step_beta::Union{Vector{Float64},Nothing}=nothing,
    step_delta::Union{VecOrMat{Float64},Nothing}=nothing,
    scale_beta::Union{Vector{Float64},Nothing}=nothing,
    scale_delta::Union{VecOrMat{Float64},Nothing}=nothing,
    rptfile::Union{String,Nothing}=nothing,
    errfile::Union{String,Nothing}=nothing
)::OdrResult

    # Check xdata
    x_is_matrix = false
    m = 1
    if ndims(xdata) > 1
        x_is_matrix = true
        m = size(xdata, 2)
    end

    # Check ydata
    y_is_matrix = false
    q = 1
    if ndims(ydata) > 1
        y_is_matrix = true
        q = size(ydata, 2)
    end

    n = size(xdata, 1)
    size(ydata, 1) == n || error("The first dimension of `xdata` and `ydata` must be identical, but size(xdata)=$(size(xdata)) and size(ydata)=$(size(ydata)).")

    # Copy beta0
    np = length(beta0)
    beta = copy(beta0)

    # Check beta bounds
    lower, upper = bounds

    if lower !== nothing
        size(lower) == size(beta0) || error("The lower bound must have the same length as `beta0`.")
        all(lower .< beta0) || error("The lower bound must be less than `beta0`.")
    end

    if upper !== nothing
        size(upper) == size(beta0) || error("The upper bound must have the same length as `beta0`.")
        all(upper .> beta0) || error("The upper bound must be greater than `beta0`.")
    end

    # Check other beta-related arguments
    if fix_beta !== nothing
        size(fix_beta) == size(beta0) || error("`fix_beta` must have the same shape as `beta0`.")
    end

    if step_beta !== nothing
        size(step_beta) == size(beta0) || error("`step_beta` must have the same shape as `beta0`.")
    end

    if scale_beta !== nothing
        size(scale_beta) == size(beta0) || error("`scale_beta` must have the same shape as `beta0`.")
    end

    # Check delta0
    has_delta0 = false
    if delta0 !== nothing
        size(delta0) == size(xdata) || error("`delta0` must have the same shape as `xdata`.")
        delta = copy(delta0)
        has_delta0 = true
    else
        delta = zeros(size(xdata))
    end

    # Check fix_x
    ldifx = 1
    if fix_x !== nothing
        if size(fix_x) == size(xdata)
            ldifx = n
        elseif size(fix_x) == (m,) && m > 1 && n != m
            ldifx = 1
        elseif size(fix_x) == (n,) && m > 1 && n != m
            ldifx = n
            fix_x = repeat(fix_x, 1, m)
        else
            error("`fix_x` must either have the same shape as `xdata` or be a vector of length `m` or `n`. See page 26 of the ODRPACK95 User Guide.")
        end
    end

    # Check step_delta
    ldstpd = 1
    if step_delta !== nothing
        if size(step_delta) == size(xdata)
            ldstpd = n
        elseif size(step_delta) == (m,) && m > 1 && n != m
            ldstpd = 1
        elseif size(step_delta) == (n,) && m > 1 && n != m
            ldstpd = n
            step_delta = repeat(step_delta, 1, m)
        else
            error("`step_delta` must either have the same shape as `xdata` or be a vector of length `m` or `n`. See page 31 of the ODRPACK95 User Guide.")
        end
    end

    # Check scale_delta
    ldscld = 1
    if scale_delta !== nothing
        if size(scale_delta) == size(xdata)
            ldscld = n
        elseif size(scale_delta) == (m,) && m > 1 && n != m
            ldscld = 1
        elseif size(scale_delta) == (n,) && m > 1 && n != m
            ldscld = n
            scale_delta = repeat(scale_delta, 1, m)
        else
            error("`scale_delta` must either have the same shape as `xdata` or be a vector of length `m` or `n`. See page 32 of the ODRPACK95 User Guide.")
        end
    end

    # Check weight_x
    ldwd = 1
    ld2wd = 1
    if weight_x !== nothing
        if isa(weight_x, Real)
            ldwd = 1
            ld2wd = 1
            weight_x = fill(convert(Float64, weight_x), m)
        elseif isa(weight_x, AbstractArray)
            wx_shape = size(weight_x)
            if wx_shape == (m,)
                ldwd = 1
                ld2wd = 1
            elseif wx_shape == (m, m)
                ldwd = 1
                ld2wd = m
            elseif wx_shape == (n, m) || (wx_shape == (n,) && m == 1)
                ldwd = n
                ld2wd = 1
            elseif wx_shape in ((1, 1, m), (n, 1, m), (1, m, m), (n, m, m))
                ldwd = wx_shape[1]
                ld2wd = wx_shape[2]
            else
                error("Invalid shape for `weight_x`: $wx_shape. Expected one of " *
                      " `(m,)`, `(n,)`, `(m, m)`, `(n, m)`, " *
                      " `(1, 1, m)`, `(n, 1, m)`, `(1, m, m)`, or `(n, m, m)`. " *
                      "See page 26 of the ODRPACK95 User Guide.")
            end
        else
            error("`weight_x` must be a real number or an array.")
        end
    end

    # Check weight_y
    ldwe = 1
    ld2we = 1
    if weight_y !== nothing
        if isa(weight_y, Real)
            ldwe = 1
            ld2we = 1
            weight_y = fill(convert(Float64, weight_y), q)
        elseif isa(weight_y, AbstractArray)
            wy_shape = size(weight_y)
            if wy_shape == (q,)
                ldwe = 1
                ld2we = 1
            elseif wy_shape == (q, q)
                ldwe = 1
                ld2we = q
            elseif wy_shape == (n, q) || (wy_shape == (n,) && q == 1)
                ldwe = n
                ld2we = 1
            elseif wy_shape in ((1, 1, q), (n, 1, q), (1, q, q), (n, q, q))
                ldwe = wy_shape[1]
                ld2we = wy_shape[2]
            else
                error("Invalid shape for `weight_y`: $wy_shape. Expected one of " *
                      " `(q,)`, `(n,)`, `(q, q)`, `(n, q)`, " *
                      " `(1, 1, q)`, `(n, 1, q)`, `(1, q, q)`, or `(n, q, q)`. " *
                      "See page 26 of the ODRPACK95 User Guide.")
            end
        else
            error("`weight_y` must be a real number or an array.")
        end
    end

    # Check model jacobians
    has_jac = false
    if jac_beta! === nothing && jac_x! === nothing
        #
    elseif jac_beta! !== nothing && jac_x! !== nothing
        has_jac = true
    elseif jac_beta! !== nothing && jac_x! === nothing && task == "OLS"
        has_jac = true
    else
        error("Invalid combination of `jac_beta!` and `jac_x!`.")
    end

    # Convert fix to ifix
    ifixb = fix_beta !== nothing ? Int32.(.!fix_beta) : nothing
    ifixx = fix_x !== nothing ? Int32.(.!fix_x) : nothing

    # Set iprint flag
    iprint_mapping = Dict(
        "none" => 0,
        "short" => 1001,
        "long" => 2002,
        "iteration" => 2212
    )
    iprint = iprint_mapping[report]

    # Set job flag
    jobl = zeros(Int, 5)

    is_odr = true
    if task == "explicit-ODR"
        jobl[1] = 0
    elseif task == "implicit-ODR"
        jobl[1] = 1
    elseif task == "OLS"
        jobl[1] = 2
        is_odr = false
    else
        error("Invalid value for `task`: $(task).")
    end

    if has_jac
        jobl[2] = 2
    else
        if diff_scheme == "forward"
            jobl[2] = 0
        elseif diff_scheme == "central"
            jobl[2] = 1
        else
            error("Invalid value for `diff_scheme`: $(diff_scheme).")
        end
    end

    if has_delta0
        jobl[4] = 1
    end

    job = sum(jobl[i] * 10^(i - 1) for i in eachindex(jobl))

    # Allocate work arrays (drop restart possibility)
    lrwork, liwork = workspace_dimensions(n, m, q, np, is_odr)
    rwork = zeros(Float64, lrwork)
    iwork = zeros(Int32, liwork)

    # Create pointers for optional arguments
    ifixb_ptr = ifixb === nothing ? C_NULL : pointer(ifixb)
    ifixx_ptr = ifixx === nothing ? C_NULL : pointer(ifixx)
    lower_ptr = lower === nothing ? C_NULL : pointer(lower)
    upper_ptr = upper === nothing ? C_NULL : pointer(upper)
    weight_x_ptr = weight_x === nothing ? C_NULL : pointer(weight_x)
    weight_y_ptr = weight_y === nothing ? C_NULL : pointer(weight_y)
    step_beta_ptr = step_beta === nothing ? C_NULL : pointer(step_beta)
    step_delta_ptr = step_delta === nothing ? C_NULL : pointer(step_delta)
    scale_beta_ptr = scale_beta === nothing ? C_NULL : pointer(scale_beta)
    scale_delta_ptr = scale_delta === nothing ? C_NULL : pointer(scale_delta)
    ndigit_ptr = ndigit === nothing ? C_NULL : pointer(ndigit)
    taufac_ptr = taufac === nothing ? C_NULL : pointer(taufac)
    sstol_ptr = sstol === nothing ? C_NULL : pointer(sstol)
    partol_ptr = partol === nothing ? C_NULL : pointer(partol)

    # Open files
    lunrpt = Ref{Int32}(6)
    lunerr = Ref{Int32}(6)

    if rptfile !== nothing
        lunrpt[] = 0
        ierr = open_file(rptfile, lunrpt)
        if ierr != 0
            error("Error opening report file.")
        end
    end

    if errfile !== nothing
        if rptfile === nothing || errfile != rptfile
            lunerr[] = 0
            ierr = open_file(errfile, lunerr)
            if ierr != 0
                error("Error opening error file.")
            end
        else
            lunerr[] = lunrpt[]
        end
    end

    # Julia (C-style) callback function
    function fcn_jl!(
        n_ptr::Ptr{Cint}, m_ptr::Ptr{Cint}, q_ptr::Ptr{Cint}, np_ptr::Ptr{Cint}, ldifx_ptr::Ptr{Cint},
        beta_ptr::Ptr{Cdouble}, xplusd_ptr::Ptr{Cdouble}, ifixb_ptr::Ptr{Cint}, ifixx_ptr::Ptr{Cint},
        ideval_ptr::Ptr{Cint}, y_ptr::Ptr{Cdouble}, jacb_ptr::Ptr{Cdouble}, jacd_ptr::Ptr{Cdouble},
        istop_ptr::Ptr{Cint}
    )
        # Dereference input pointers
        n = unsafe_load(n_ptr)
        m = unsafe_load(m_ptr)
        q = unsafe_load(q_ptr)
        np = unsafe_load(np_ptr)
        ideval = unsafe_load(ideval_ptr)

        # Wrap input C-style arrays as Julia arrays
        beta = unsafe_wrap(Vector{Float64}, beta_ptr, np)

        if x_is_matrix
            xplusd = unsafe_wrap(Matrix{Float64}, xplusd_ptr, (n, m))
        else
            xplusd = unsafe_wrap(Vector{Float64}, xplusd_ptr, n)
        end

        # Evaluate model function and its Jacobians
        unsafe_store!(istop_ptr, 0)
        try
            if ideval % 10 > 0
                if y_is_matrix
                    y = unsafe_wrap(Matrix{Float64}, y_ptr, (n, q))
                else
                    y = unsafe_wrap(Vector{Float64}, y_ptr, n)
                end
                f!(xplusd, beta, y)
            elseif (div(ideval, 10)) % 10 > 0
                jacb = unsafe_wrap(Array{Float64}, jacb_ptr, (n, np, q))
                jac_beta!(xplusd, beta, jacb)
            elseif (div(ideval, 100)) % 10 > 0
                jacd = unsafe_wrap(Array{Float64}, jacd_ptr, (n, m, q))
                jac_x!(xplusd, jacd)
            else
                error("The value of `ideval` is not valid.")
            end
        catch e
            if isa(e, OdrStop)
                unsafe_store!(istop_ptr, 1)
                println("Regression stopped by OdrStop exception: $(e.msg)")
            else
                rethrow(e)
            end
        end

        return nothing
    end

    # Create a C-callable function pointer for the Julia callback function
    fcn_c = @cfunction($fcn_jl!, Cvoid, (
        Ptr{Cint},    # n
        Ptr{Cint},    # m
        Ptr{Cint},    # q
        Ptr{Cint},    # np
        Ptr{Cint},    # ldifix
        Ptr{Cdouble}, # beta
        Ptr{Cdouble}, # xplusd
        Ptr{Cint},    # ifixb
        Ptr{Cint},    # ifixx
        Ptr{Cint},    # ideval
        Ptr{Cdouble}, # f
        Ptr{Cdouble}, # fjacb
        Ptr{Cdouble}, # fjacd
        Ptr{Cint}     # istop
    ))

    # Call the Fortran function
    info = Ref{Cint}(-1)
    @ccall libodrpack.odr_long_c(
        fcn_c::Ptr{Cvoid},
        n::Ref{Cint},
        m::Ref{Cint},
        q::Ref{Cint},
        np::Ref{Cint},
        ldwe::Ref{Cint},
        ld2we::Ref{Cint},
        ldwd::Ref{Cint},
        ld2wd::Ref{Cint},
        ldifx::Ref{Cint},
        ldstpd::Ref{Cint},
        ldscld::Ref{Cint},
        lrwork::Ref{Cint},
        liwork::Ref{Cint},
        beta::Ptr{Cdouble},
        ydata::Ptr{Cdouble},
        xdata::Ptr{Cdouble},
        weight_y_ptr::Ptr{Cdouble},
        weight_x_ptr::Ptr{Cdouble},
        ifixb_ptr::Ptr{Cint},
        ifixx_ptr::Ptr{Cint},
        step_beta_ptr::Ptr{Cdouble},
        step_delta_ptr::Ptr{Cdouble},
        scale_beta_ptr::Ptr{Cdouble},
        scale_delta_ptr::Ptr{Cdouble},
        delta::Ptr{Cdouble},
        lower_ptr::Ptr{Cdouble},
        upper_ptr::Ptr{Cdouble},
        rwork::Ptr{Cdouble},
        iwork::Ptr{Cint},
        job::Ref{Cint},
        ndigit_ptr::Ptr{Cint},
        taufac_ptr::Ptr{Cdouble},
        sstol_ptr::Ptr{Cdouble},
        partol_ptr::Ptr{Cdouble},
        maxit::Ref{Cint},
        iprint::Ref{Cint},
        lunerr::Ref{Cint},
        lunrpt::Ref{Cint},
        info::Ref{Cint}
    )::Cvoid

    # Close files
    if rptfile !== nothing
        close_file(lunrpt) == 0 || error("Error closing report file.")
    end

    if errfile !== nothing && lunrpt[] != lunerr[]
        close_file(lunerr) == 0 || error("Error closing error file.")
    end

    # Indexes of integer and real work arrays
    iwidx = loc_iwork(m, q, np)
    rwidx = loc_rwork(n, m, q, np, ldwe, ld2we, is_odr)

    # Return the result

    # Extract results without messing up the original work arrays
    i0_eps = rwidx.eps
    eps = copy(rwork[(i0_eps+1):(i0_eps+n*q)])
    eps = reshape(eps, size(ydata))

    i0_sd = rwidx.sd
    sd_beta = copy(rwork[(i0_sd+1):(i0_sd+np)])

    i0_vcv = rwidx.vcv
    cov_beta = copy(rwork[(i0_vcv+1):(i0_vcv+np^2)])
    cov_beta = reshape(cov_beta, (np, np))

    # Return OdrResult   
    return OdrResult(
        beta,
        delta,
        eps,
        xdata + delta,
        ydata + eps,
        sd_beta,
        cov_beta,
        rwork[rwidx.rvar],
        iwork[iwidx.nfev],
        iwork[iwidx.njev],
        iwork[iwidx.niter],
        iwork[iwidx.irank],
        rwork[rwidx.rcond],
        Int(info[]),
        get_stopreason_message(info[]),
        info[] < 4,
        rwork[rwidx.wss],
        rwork[rwidx.wssdel],
        rwork[rwidx.wsseps],
        iwork,
        rwork
    )

end

end # module

