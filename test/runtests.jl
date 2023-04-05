using uMPS
using Test

@testset "uMPS.jl" begin
    
end

@testset "hamiltonian.jl" begin
    
end

# @testset "groundstates.jl" begin
#     h0 = spinmodel(1,1,1;s = 1)

#     D = 10
#     d = size(h0,1)

#     phi0 = randUMPS(D,d,"UF")

#     vumps(h0, phi0)

# end

@testset "excitation.jl" begin
    
end

    h0 = spinmodel(1,1,1;s = 1);

    D = 24;
    d = size(h0,1);

    phi0 = randUMPS(D,d,"UF");

    gstate , e = vumps(h0, phi0);

    M = heisenberg();
    gstate1, e1 = vumps(M, phi0);
    expectation(gstate1, h0)

    state, ~ = mixed_canonical(phi0);
    # el, fl = leftfixpoint(M, state)
    el, ~ = leftfixpoint1(M, gstate1)

# op("Sp", "Spinhalf")

# phi0 = randUMPS(10,2,"UF")

# state, e = vumps(h0, phi0)
# A = randn(5,2,5)
# Al, Ar, C = mixed_canonical(A)

# delta = 1e-10
# @tensor Ac[:] := Al[-1,-2,1]*C[1,-3]

# e0 = expectation(Al, Ar, Ac, h0)

# h = h0 - e*identitymatrix(h0)

# Lh = sumleft(Al,C, h)
# Rh = sumright(Ar, C, h)

# ~, Ac0 = eigsolve(x -> HAc(x, Al, Ar, h, Lh, Rh), Ac[:], 1, :SR; issymmetric = false, tol = delta/10)

# ~, C0 = eigsolve(x -> Hc(x, Al, Ar, h, Lh, Rh), C[:], 1, :SR; issymmetric = false, tol = delta/10)

