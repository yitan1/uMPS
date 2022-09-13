using uMPS
using Test

# using Printf
# Base.show(io::IO, f::Float64) = @printf(io, "%1.4f", f)

@testset "uMPS.jl" begin
    
end

# @testset "mps.jl" begin
    D, d = 10, 2
    phi0 = rand_singleUMPS(D, d, "UF")
    phi1, ~ = mixed_canonical(phi0)
# end

@testset "hamiltonian.jl" begin
    # op("Sp", "Spinhalf")
    h0 = spinmodel(1,1,1;s = 1)
end

@testset "groundstates.jl" begin
    h0 = spinmodel(1,1,1;s = 1)

    D = 10
    d = size(h0,1)

    phi0 = rand_singleUMPS(D,d,"UF")

    vumps(h0, phi0)

end

@testset "excitation.jl" begin
    h0 = spinmodel(1,1,1;s = 1)

    D = 10
    d = size(h0,1)

    phi0 = rand_singleUMPS(D,d,"UF")

    phi1, e = vumps(h0, phi0)

    p = 1pi   # momentum 
    N = 10    # number of excited states
    En, X = excitation(h0, p, phi1, N)
end

@testset "ground_mpo.jl" begin
    M = heisenberg() # ::MPO

    phi0 = rand_singleUMPS(D,d,"UF")

    phi1, e = vumps(M, phi0)
end
