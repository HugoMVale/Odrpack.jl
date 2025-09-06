"""
Auxiliary functions for `odrpack`.

This module provides a Julia interface to the auxiliary routines from the Fortran
[`odrpack`](https://github.com/HugoMVale/odrpack95) library.
"""
module Auxiliary

import odrpack_jll
const lib = odrpack_jll.libodrpack95

export workspace_dimensions, loc_iwork, loc_rwork, open_file, close_file, lib, stop_message


"""
    stop_message(info) -> String

Return a human-readable message based on the stopping condition returned by `odrpack` in 
the `info` argument of the result.

# Arguments
- `info::Integer`: value of the `info` argument returned by `odrpack`. This value is used to
  determine the stopping condition.

# Returns
- `::String`: human-readable string describing the stopping condition.
"""
function stop_message(info::Integer)::String

    message = zeros(Cchar, 256)

    @ccall lib.stop_message_c(
        info::Cint,
        message::Ptr{Cchar},
        length(message)::Csize_t
    )::Cvoid

    return unsafe_string(pointer(message))

end


"""
    workspace_dimensions(n, m, q, np, isodr) -> Tuple{Int32, Int32}

Calculate the dimensions of the workspace arrays required by the underlying Fortran library.

# Arguments
- `n::Integer`: Number of observations.
- `m::Integer`: Number of columns of data in the explanatory variable.
- `q::Integer`: Number of responses per observation.
- `np::Integer`: Number of function parameters.
- `isodr::Bool`: Variable designating whether the solution is by ODR (`true`) or by OLS (`false`).

# Returns
- `Tuple{Int32, Int32}`: A tuple containing the lengths of the real and integer work arrays 
  (`lrwork`, `liwork`).
"""
function workspace_dimensions(n::Integer, m::Integer, q::Integer, np::Integer, isodr::Bool)::Tuple{Int32,Int32}

    lrwork = Ref{Int32}()
    liwork = Ref{Int32}()

    @ccall lib.workspace_dimensions_c(
        n::Ref{Cint},
        m::Ref{Cint},
        q::Ref{Cint},
        np::Ref{Cint},
        isodr::Ref{Bool},
        lrwork::Ref{Cint},
        liwork::Ref{Cint}
    )::Cvoid

    return lrwork[], liwork[]

end


struct Iworkidx
    msgb::Cint
    msgd::Cint
    ifix2::Cint
    istop::Cint
    nnzw::Cint
    npp::Cint
    idf::Cint
    job::Cint
    iprint::Cint
    lunerr::Cint
    lunrpt::Cint
    nrow::Cint
    ntol::Cint
    neta::Cint
    maxit::Cint
    niter::Cint
    nfev::Cint
    njev::Cint
    int2::Cint
    irank::Cint
    ldtt::Cint
    bound::Cint
    liwkmin::Cint
end


"""
    loc_iwork(m, q, np) -> Iworkidx

Get storage locations within the integer work space.

# Arguments
- `m::Integer`: Number of columns of data in the explanatory variable.
- `q::Integer`: Number of responses per observation.
- `np::Integer`: Number of function parameters.

# Returns
- `Iworkidx`: A structure containing the 0-based indexes of the integer work array.
"""
function loc_iwork(m::Integer, q::Integer, np::Integer)::Iworkidx

    iwi = Ref{Iworkidx}()

    @ccall lib.loc_iwork_c(
        m::Ref{Cint},
        q::Ref{Cint},
        np::Ref{Cint},
        iwi::Ref{Iworkidx}
    )::Cvoid

    return iwi[]

end


struct Rworkidx
    delta::Cint
    eps::Cint
    xplusd::Cint
    fn::Cint
    sd::Cint
    vcv::Cint
    rvar::Cint
    wss::Cint
    wssdel::Cint
    wsseps::Cint
    rcond::Cint
    eta::Cint
    olmavg::Cint
    tau::Cint
    alpha::Cint
    actrs::Cint
    pnorm::Cint
    rnorms::Cint
    prers::Cint
    partol::Cint
    sstol::Cint
    taufac::Cint
    epsmac::Cint
    beta0::Cint
    betac::Cint
    betas::Cint
    betan::Cint
    s::Cint
    ss::Cint
    ssf::Cint
    qraux::Cint
    u::Cint
    fs::Cint
    fjacb::Cint
    we1::Cint
    diff::Cint
    deltas::Cint
    deltan::Cint
    t::Cint
    tt::Cint
    omega::Cint
    fjacd::Cint
    wrk1::Cint
    wrk2::Cint
    wrk3::Cint
    wrk4::Cint
    wrk5::Cint
    wrk6::Cint
    wrk7::Cint
    lower::Cint
    upper::Cint
    lrwkmin::Cint
end


"""
    loc_rwork(n, m, q, np, ldwe, ld2we, isodr) -> Rworkidx

Get storage locations within the real work space.

# Arguments
- `n::Integer`: Number of observations.
- `m::Integer`: Number of columns of data in the explanatory variable.
- `q::Integer`: Number of responses per observation.
- `np::Integer`: Number of function parameters.
- `ldwe::Integer`: Leading dimension of the `we` array.
- `ld2we::Integer`: Second dimension of the `we` array.
- `isodr::Bool`: Indicates whether the solution is by ODR (`true`) or by OLS (`false`).

# Returns
- `Rworkidx`: A structure containing the 0-based indexes of the real work array.
"""
function loc_rwork(n::Integer, m::Integer, q::Integer, np::Integer, ldwe::Integer, ld2we::Integer, isodr::Bool)::Rworkidx

    rwi = Ref{Rworkidx}()

    @ccall lib.loc_rwork_c(
        n::Ref{Cint},
        m::Ref{Cint},
        q::Ref{Cint},
        np::Ref{Cint},
        ldwe::Ref{Cint},
        ld2we::Ref{Cint},
        isodr::Ref{Bool},
        rwi::Ref{Rworkidx}
    )::Cvoid

    return rwi[]

end


"""
    open_file(filename, lun) -> Int32

Open a new file associated with a specified logical unit number.

# Arguments
- `filename::AbstractString`: String containing the file name.
- `lun::Ref{Cint}`: Logical unit number. This value will be modified by the function.

# Returns
- `Int32`: Error code (compiler dependent). A value of `0` typically indicates success.
"""
function open_file(filename::AbstractString, lun::Ref{Int32})::Int32

    ierr = Ref{Int32}(0)

    @ccall lib.open_file(
        filename::Cstring,
        lun::Ref{Cint},
        ierr::Ref{Cint}
    )::Cvoid

    return ierr[]

end


"""
    close_file(lun) -> Int32

Close a file associated with a specified logical unit number.

# Arguments
- `lun::Ref{Int32}`: The logical unit number of the file to close.

# Returns
- `Int32`: An error code (compiler dependent). A return value of `0` typically indicates successful closure.
"""
function close_file(lun::Ref{Int32})::Int32

    ierr = Ref{Int32}(0)

    @ccall lib.close_file(
        lun::Ref{Cint},
        ierr::Ref{Cint}
    )::Cvoid

    return ierr[]

end

end # module