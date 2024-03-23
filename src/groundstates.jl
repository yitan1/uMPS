# export vumps, expectation, identitymatrix, sumleft, sumright, HAc, Hc, applyTl, getAlAr,applyTr

using Printf
# vumps for two sites hamiltonian

function vumps(h0, phi0::UMPS{:UF}; tol = 1e-10)
    state, ~ = mixed_canonical(phi0; tol = tol)

    return vumps(h0, state; tol = 1e-10)
end

function vumps(h0, state::UMPS{:MF}; tol = 1e-10)

    maxitr = 1e3
    delta = 1e-14
    println("Maximum iteration： $(maxitr)")
    @printf("initial delta: %0.4e\n", delta)

    for i = 1:maxitr 
        e = expectation(state, h0)
        # @show e
        h = h0 -  e*identitymatrix(h0)                          #shift hamiltonian
        Lh = sumleft(state, h; tol = delta/10)
        Rh = sumright(state, h; tol = delta/10)
        
        Ac = data(state)[3]
        C = data(state)[4]

        D = size(Ac,1)
        d = size(Ac,2)

        EAc, Ac0 = eigsolve(x -> HAc(x, state, h, Lh, Rh), D*d*D, 1, :SR)
        Ec, C0 = eigsolve(x -> Hc(x, state, h, Lh, Rh), D*D, 1, :SR)

        Ac = reshape(Ac0[1], size(Ac))
        C = reshape(C0[1], size(C))

        Al, Ar, tol_left, tol_right = getAlAr(Ac, C)

        # @show tol_left, tol_right
        delta = max(tol_left, tol_right)

        state = umps([Al,Ar,Ac,C], "MF")
        # @printf("EAc = %0.4e, Ec = %0.4e\n", EAc[1], Ec[1])
        @printf("i = %d, e = %0.4f, delta = %0.2e\n", i, e, delta)
        if delta < tol
            @printf("The total iteration %d times and the final error is %0.2e\nThe final energy is %0.4f", i, delta, e)
            break
        end
    end

    e = expectation(state, h0)

    state, e
end


# compute expectation of two site hamiltonian

#   1   3
#   |   |         
#     h         1 -- A -- 3
#   |   |            |
#   2   4            2

function expectation(state::UMPS{:MF}, h)
    Al = data(state)[1]
    # Ar = data(state)[2]
    Ac = data(state)[3]

    @tensor Alc[:] :=  Al[-1,-2, 1] * Ac[1,-3,-4]
    @tensor e[:] := Alc[1,3,4,2]* h[3,5,4,6]*conj(Alc)[1,5,6,2]
    e[]
end

function identitymatrix(M)
    dim = size(M,1)
    locI = diagm(ones(dim))
    
    @tensor Im[:] := locI[-1,-2]*locI[-3,-4]

    Im
end

function HAc(Ac, state::UMPS{:MF}, h, Lh, Rh)
    Al = data(state)[1]
    Ar = data(state)[2]

    Ac = reshape(Ac, size(Al))
    #diagram 1
    @tensor d1[:] := Al[4,2,1]*Ac[1,3,-3]*h[2,5,3,-2]*conj(Al)[4,5,-1]
    #diagram 2
    @tensor d2[:] := Ac[-1,2,1]*Ar[1,3,5]*h[2,-2,3,4]*conj(Ar)[-3,4,5]
    #diagram 3
    @tensor d3[:] := Lh[1,-1]*Ac[1,-2,-3]
    #diagram 3
    @tensor d4[:] := Ac[-1,-2,1]*Rh[1,-3]
    y = d1 + d2 + d3 + d4
    # y = d1 + d2

    y[:]
end

function Hc(C, state::UMPS{:MF}, h, Lh, Rh)
    Al = data(state)[1]
    Ar = data(state)[2]

    D = size(Al,1) 
    C = reshape(C, D, D)
    #diagram 1
    @tensor d1[:] := (Al[1,2,7]*conj(Al)[1,3,-1]) * h[2,3,5,6] * (Ar[8,5,4]*conj(Ar)[-2,6,4]) * C[7,8]
    # diagram 1
    @tensor d2[:] := Lh[1,-1]*C[1,-2]
    #diagram 1
    @tensor d3[:] := C[-1,1]*Rh[1,-2]
    
    y = d1 + d2 + d3
    # y = d1
    
    y[:]
end

function sumleft(state::UMPS{:MF}, h; tol = 1e-10)
    Al = data(state)[1]
    C = data(state)[4]

    D = size(Al,1)

    @tensor All[:] :=  Al[-1,-2, 1] * Al[1,-3,-4]
    @tensor HAA[:] := All[1,2,3,-1]* h[2,4,3,5]*conj(All)[1,4,5,-2]

    ##????
    # @tensor rightfp[:] := C[-1,1]*conj(C)[-2,1]
    # @tensor y[:] := HAA[-1,-2] - HAA[1,2]*rightfp[1,2]*diagm(ones(D))[-1,-2]

    y = reshape(HAA[:], D*D)

    Lh, ~ = linsolve(x -> rightapplyTM(x, state, "ll"), y; tol = tol)
    Lh = reshape(Lh, D, D)

    Lh
end

function sumright(state::UMPS{:MF}, h; tol = 1e-10)
    Ar = data(state)[2]
    C = data(state)[4]

    D = size(Ar,1)

    @tensor Arr[:] :=  Ar[-1,-2, 1] * Ar[1,-3,-4]
    @tensor HAA[:] := Arr[-1,2,3,1]* h[2,4,3,5]*conj(Arr)[-2,4,5,1]

    ##????
    # @tensor leftfp[:] :=  C[1,-1]*conj(C)[1,-2]
    # @tensor y[:] := HAA[-1,-2] - leftfp[1,2]*HAA[1,2]*diagm(ones(D))[-1,-2]

    y = reshape(HAA[:], D*D)
    Rh, ~ = linsolve(x -> leftapplyTM(x, state, "rr"), y; tol = tol)
    Rh = reshape(Rh, D, D)

    Rh
end


#########################
# x*( 1-a(T-fp) ) = y  
leftapplyTM(x, state::UMPS{:MF}, s::AbstractString, a::Float64 = 1.0) = leftapplyTM(x, state, Tag(s), a)
leftapplyTM(x, state::UMPS{:MF}, s::AbstractString, a::ComplexF64) = leftapplyTM(x, state, Tag(s), a)

# x*( 1-a(T-fp) ) = y  
rightapplyTM(x, state::UMPS{:MF}, s::AbstractString, a::Float64 = 1.0) = rightapplyTM(x, state, Tag(s), a)
rightapplyTM(x, state::UMPS{:MF}, s::AbstractString, a::ComplexF64) = rightapplyTM(x, state, Tag(s), a)

function leftapplyTM(x, state::UMPS{:MF}, t::Tag{:rr}, a)
    Ar = data(state)[2]
    C = data(state)[4]

    D = size(Ar,1)

    @tensor leftfp[:] :=  C[1,-1]*conj(C)[1,-2] # left fixed point of transfer matrix T with left gauge 
    rightfp = diagm(ones(D))        # right fixed point of transfer matrix T with right gauge

    @tensor T[:] := a*Ar[-1,1,-3]* conj(Ar)[-2,1,-4]
    x = reshape(x, D, D)
    @tensor y[:] := x[-1,-2] - T[-1,-2,1,2]*x[1,2] + a*leftfp[1,2]*x[1,2]*rightfp[-1,-2]

    y = reshape(y, D*D)
    y
end

function rightapplyTM(x, state::UMPS{:MF}, t::Tag{:ll}, a)
    Al = data(state)[1]
    # Ar = data(state)[2]
    C = data(state)[4]

    D = size(Al,1)

    leftfp = diagm(ones(D)) # left fixed point of transfer matrix T with left gauge 
    @tensor rightfp[:] := C[-1,1]*conj(C)[-2,1] # right fixed point of transfer matrix T with right gauge

    @tensor T[:] := a*Al[-1,1,-3]* conj(Al)[-2,1,-4]
    x = reshape(x, D, D)
    @tensor y[:] := x[-1,-2] - x[1,2]*T[1,2,-1,-2] + a*x[1,2]*rightfp[1, 2]*leftfp[-1,-2]

    y = reshape(y, D*D)
    y
end
#########################

###########################
function getAlAr(Ac, C)
    D = size(Ac, 1)
    d = size(Ac, 2)

    Ac_l = reshape(Ac, D*d, D)
    UAc_l, ~ = polardecomposition(Ac_l)

    Ac_r = reshape(Ac, D, D*d)
    UAc_r, ~ = polardecomposition(Ac_r) 
    
    UC, ~ = polardecomposition(C)

    Al = UAc_l*UC'
    Ar = UC'*UAc_r

    # compute tol
    tol_left = norm(Ac_l - Al*C)
    tol_right = norm(Ac_r - C*Ar)

    Al = reshape(Al, D,d,D)
    Ar = reshape(Ar, D,d,D)

    Al, Ar, tol_left, tol_right
end

function polardecomposition(A)
    U0, S, V = svd(A)
    Vt = V'

    U = U0*Vt
    P = V*diagm(S)*Vt

    U, P
end
