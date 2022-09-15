using uMPS
using Test

# using Printf
# Base.show(io::IO, f::Float64) = @printf(io, "%1.4f", f)

@testset "uMPS.jl" begin
    # @test begin 2 end
end


@testset "mps.jl" begin
    @testset "random UMPS" begin
        D, d = 10, 2
        phi0 = rand_singleUMPS(D, d, "UF")
        phi1, ~ = mixed_canonical(phi0) # phi1 is mixed canonical form
        # method 2
        D, d = 10, 2
        phi3 = rand_singleUMPS(D, d, "MF")
    end
end

@testset "hamiltonian.jl" begin
    # op("Sp", "Spinhalf")
    @test h0 = spinmodel(1,1,1;s = 1)
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


using LinearAlgebra
using TensorOperations

phi0 = rand_singleUMPS(D, d, "MF")

Al,Ar = phi0[1], phi0[2]
C = phi0[4]
Ac = phi0[3]


D, d = 10, 2
phi0 = rand_singleUMPS(D, d, "UF")
phi1, ~ = mixed_canonical(phi0)
Al,Ar = phi1[1], phi1[2]
C = phi1[4]

@tensor y[:] := Al[-1,-2,1]*C[1,-3] - C[-1,1]*Ar[1,-2,-3]
norm(y)
