using Test

@testset "Odrpack Tests" begin
    include("./test_aux.jl")
    include("./test_odr.jl")
end