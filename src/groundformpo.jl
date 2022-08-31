# find ground state for mpo

function vumps(M::MPO, phi0::UMPS{:UF}; tol = 1e-10)
    state, ~ = mixed_canonical(phi0; tol = tol)

    return vumps(M, state; tol = 1e-10)
end

function vumps(M::MPO, state::UMPS{:MF}; tol = 1e-10)

    maxitr = 1e3
    delta = 1e-14

    for i = 1:maxitr 
        el, fl = leftfixpoint(M, state)
        er, fr = rightfixpoint(M, state)
        
        Ac = data(state)[3]
        C = data(state)[4]

        @tensor y[:] := fl[1,3,2]*C[1,4]*conj(C)[2,5]*fr[4,3,5]
        fl = fl/y[]

        D = size(Ac,1)
        d = size(Ac,2)

        EAc, Ac0 = eigsolve(x -> HmAc(x, state, M, fl, fr), D*d*D, 1, :SR)
        Ec, C0 = eigsolve(x -> Hmc(x, state, M, fl, fr), D*D, 1, :SR)

        Ac = reshape(Ac0[1], size(Ac))
        C = reshape(C0[1], size(C))

        Al, Ar, tol_left, tol_right = getAlAr(Ac, C)

        # @show tol_left, tol_right
        delta = max(tol_left, tol_right)

        state = umps([Al,Ar,Ac,C], "MF")

        @show i, EAc[1], Ec[1], el, er, delta
        if delta < tol
            @show i, el, delta
            break
        end
    end

    # e = expectation(state, h0)

    # state, e
    
end

export leftfixpoint, rightfixpoint
# x(Tll - ll) = x
function leftfixpoint(M::MPO, state::UMPS{:MF})
    Al = data(state)[1]
    Ml = data(M)[1]
    Mm = data(M)[2]
    D = size(Al, 1)
    dm = size(Mm, 1)

    @tensor L0[:] := Al[1,2,-1]*conj(Al)[1,3,-3]*Ml[3,-2,2]
    @tensor T[:] := Al[-1,1,-4]*Mm[-2,2,-5,1]*conj(Al)[-3,2,-6] 

    l = zeros(D,dm,D)
    l[:,dm,:] = diagm(ones(D))
    l = l/norm(l)
    @tensor ll[:] := l[-1,-2,-3]*conj(l)[-4,-5,-6] 

    # T = reshape(T, D*dm*D, D*dm*D)
    # ll = reshape(ll, D*dm*D, D*dm*D)

    val, vec = eigsolve(x -> leftapply(x, T, ll), D*dm*D, 1, :LR)
    fl0 = vec[1]
    fl = fl0 - (conj(fl0)'*l[:])*l[:]
    # @show fl[:]'*l[:]
    gammal = (fl)'*L0[:]
    fl = fl/gammal
    fl = reshape(fl, D, dm, D)
    @show fl[:]'*fl[:]
    # @show sum(leftapply(fl[:], T, ll) - fl[:])
    #compute energy
    @tensor el[:] := fl[1,2,3]*T[1,2,3,4,5,6]*l[4,5,6]

    el[], fl, val[1] 
end

function leftapply(x, T, ll)
    x1 = reshape(x, size(T)[1:3])

    @tensor y[:] := x1[1,2,3]*T[1,2,3,-1,-2,-3] - x1[1,2,3]*ll[1,2,3,-1,-2,-3]
    y = reshape(y, size(x))
    
    y
end

function rightfixpoint(M::MPO, state::UMPS{:MF})
    Ar = data(state)[2]
    Mm = data(M)[2]
    Mr = data(M)[3]
    D = size(Ar, 1)
    dm = size(Mm, 1)

    @tensor R0[:] := Ar[-1,2,1]*conj(Ar)[-3,3,1]*Mr[3,-2,2]
    @tensor T[:] := Ar[-1,1,-4]*Mm[-2,2,-5,1]*conj(Ar)[-3,2,-6] 

    r = zeros(D,dm,D)
    r[:,1,:] = diagm(ones(D))
    r = r/norm(r)
    @tensor rr[:] := r[-1,-2,-3]*conj(r)[-4,-5,-6] 

    val, vec = eigsolve(x -> rightapply(x, T, rr), D*dm*D, 1, :LR)
    fr0 = vec[1]

    fr = fr0 - (conj(fr0)'*r[:])*r[:]
    # @show fr[:]'*r[:]
    gammar = (fr)'*R0[:]
    fr = fr*gammar
    fr = reshape(fr, D,dm,D)
    @show norm(fr[:])
    # @show sum(leftapply(fr[:], T, rr) - fr[:])
    #compute energy
    @tensor er[:] := r[4,5,6]*T[4,5,6,1,2,3]*fr[1,2,3]

    er[], fr 
end

function rightapply(x, T, rr)
    x1 = reshape(x, size(T)[1:3])

    @tensor y[:] := T[-1,-2,-3,1,2,3]*x1[1,2,3] - rr[-1,-2,-3,1,2,3]*x1[1,2,3]
    y = reshape(y, size(x))
    
    y
end

function HmAc(Ac, state, M, fl, fr)
    Al = data(state)[1]
    Mm = data(M)[2]

    Ac = reshape(Ac, size(Al))

    @tensor y[:] := Ac[1,3,4]*fl[1,2,-1]*Mm[2,-2,5,3]*fr[4,5,-3]

    y[:]
end

function Hmc(C, state, M, fl, fr)
    Al = data(state)[1]
    D = size(Al,1) 

    Mm = data(M)[2]
    
    C = reshape(C, D, D)

    @tensor y[:] := fl[1,2,-1]*C[1,3]*fr[3,2,-2]

    y[:]
end