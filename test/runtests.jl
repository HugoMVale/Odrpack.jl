using Test
using TestItemRunner

@run_package_tests

@testset "Odrpack Tests" begin
    include("./test_odr.jl")
end
