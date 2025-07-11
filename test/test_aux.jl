using Test
using Odrpack

@testset "workspace_dimensions" begin
    n = 10
    q = 2
    m = 3
    np = 5
    isodr = true

    lrwork, liwork = Odrpack.workspace_dimensions(n, m, q, np, isodr)

    @test lrwork == 770
    @test liwork == 46
end

@testset "loc_iwork" begin
    m, q, npar = 10, 2, 5

    iwi = Odrpack.loc_iwork(m, q, npar)

    # Check that all fields of the struct are positive integers
    for field in fieldnames(typeof(iwi))
        value = getfield(iwi, field)
        @test value ≥ 0
    end
end

@testset "loc_rwork" begin
    n = 10
    m = 2
    q = 2
    npar = 5
    ldwe = 1
    ld2we = 1
    isodr = true

    rwi = Odrpack.loc_rwork(n, m, q, npar, ldwe, ld2we, isodr)

    @test length(fieldnames(typeof(rwi))) == 52
    all_positive = all(getfield(rwi, f) >= 0 for f in fieldnames(typeof(rwi)))
    @test all_positive
end

@testset "open_file and close_file" begin
    tmpfile = tempname()
    lun = Ref{Cint}(0)

    ierr_open = Odrpack.open_file(tmpfile, lun)
    @test ierr_open == 0

    ierr_close = Odrpack.close_file(lun)
    @test ierr_close == 0

    # Cleanup temp file
    isfile(tmpfile) && rm(tmpfile)
end