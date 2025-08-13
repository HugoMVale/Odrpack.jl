using Test
using TestItemRunner

@testset "Odrpack Tests" begin
    include("./test_odr.jl")
end

@run_package_tests