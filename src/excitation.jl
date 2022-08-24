# excitation for two sites hamiltonian 
export excitation, nullspaceforAl

function excitation(state::UMPS{:MF}, h0, p::Float64, N::Int64; tol = 1e-10)

    e = expectation(state, h0)
    @show e
    h = h0 - e*identitymatrix(h0) 

    Lh = sumleft(state, h)
    Rh = sumright(state, h)

    Al = data(state)[1] ## dim: D*d*D 
    VL = nullspaceforAl(Al)  ## dim: D*d*D(d-1)
    D = size(Al, 1)
    D2 = size(VL, 3)
    # X0 = zeros(D,D)

    En, X = eigsolve(x -> Heff(x, VL, p, state, h, Lh, Rh), D2*D, N, :SR)
    
    X = reshape.(X,D2,D)

    En, X
end

# using left gauge condition
function Heff(X, VL, p, state::UMPS{:MF}, h, Lh, Rh; tol = 1e-10)
    Al = data(state)[1]
    Ar = data(state)[2]
    D = size(Al, 1)
    D2 = size(VL, 3)

    X = reshape(X, D2, D)
    @tensor B[:] := VL[-1,-2,1]*X[1,-3]

    # LB = 0
    RB = sumrightB(B, p , state)
    L1 = sumleft1(B, p , state, h,  Lh)
    R1 = sumright1(B, p , state, h, Rh, RB)

    #compute diagram of Heff(B)
    # diagram 1
    @tensor d1[:] := B[-1,2,1]*Ar[1,3,5]*h[2,-2,3,4]*conj(Ar)[-3,4,5]
    # diagram 2
    @tensor d2[:] := B[4,2,1]*Ar[1,3,-3]*h[2,5,3,-2]*conj(Al)[4,5,-1]
    # diagram 3
    @tensor d3[:] := Al[-1,2,1]*B[1,3,5]*h[2,-2,3,4]*conj(Ar)[-3,4,5]
    # diagram 4
    @tensor d4[:] := Al[4,2,1]*B[1,3,-3]*h[2,5,3,-2]*conj(Al)[4,5,-1]
    # diagram 5
    @tensor d5[:] := B[-1,-2,1]*Rh[1,-3]
    # diagram 6
    @tensor d6[:] := Lh[1,-1]*B[1,-2,-3]
    # diagram 7
    @tensor d7[:] := L1[1,-1]*Ar[1,-2,-3]
    # diagram 8
    @tensor d8[:] := Al[-1,-2,1]*R1[1,-3]
    # diagram 9
    # LB = 0
    # diagram 10
    @tensor d10[:] := Lh[1,-1]*Al[1,-2,2]*RB[2,-3]
    # diagram 11
    # LB = 0
    # diagram 12
    # LB = 0
    # diagram 13
    @tensor d13[:] := Al[5,3,1]*Al[1,4,2]*RB[2,-3]*h[3,6,4,-2]*conj(Al)[5,6,-1]
    # diagram 14
    @tensor d14[:] := Al[-1,3,1]*Al[1,4,2]*RB[2,6]*h[3,-2,4,5]*conj(Ar)[-3,5,6]

    # Heff(B)
    HeffB = d1 + exp(-im*p)*d2 + exp(im*p)*d3 + d4 + d5 + d6 + exp(-im*p)*d7 + 
            exp(im*p)*d8 + exp(im*p)*d10 + exp(im*p)*d13 + exp(2*im*p)*d14
    
    # Heff(X)
    @tensor y[:] := HeffB[1,2,-2] * conj(VL)[1,2,-1]

    y[:]
end

function nullspaceforAl(A)
    dimA = size(A)
    A_ = permutedims(A,(3,2,1))

    VL = nullspace(reshape(A_,dimA[1],dimA[2]*dimA[3]))

    VL =reshape(VL, dimA[2],dimA[1],dimA[3]*(dimA[2]-1))
    VL = permutedims(VL,(2,1,3))

    VL
end

function sumrightB(B, p, state::UMPS{:MF}; tol = 1e-10)
    Ar = data(state)[2]
    D = size(Ar, 1)

    # coefficent of tranfermatrix
    a = exp(im*p)  

    @tensor y[:] := B[-1,1,2]*conj(Ar)[-2,1,2] 
    RB, ~ = linsolve(x -> leftapplyTM(x, state, "lr", a ), y[:]; tol = tol)

    RB = reshape(RB, D, D)
    
    RB
end

function sumleft1(B, p, state::UMPS{:MF}, h, Lh; tol = 1e-10)
    Al = data(state)[1]
    Ar = data(state)[2]

    @tensor All[:] := Al[-1,-2,1]*Al[1,-3,-4]

    @tensor y1[:] := B[1,3,-1]*Lh[1,2]*conj(Al)[2,3,-2]
    @tensor y2[:] := Al[2,3,1]*B[1,5,-1]*conj(All)[2,4,6,-2]*h[3,4,5,6]
    @tensor y3[:] := B[2,3,1]*Ar[1,5,-1]*conj(All)[2,4,6,-2]*h[3,4,5,6]

    y = y1 + y2 + exp(-im*p)*y3

    a = exp(-im*p)
    L1, ~ = linsolve(x -> rightapplyTM(x, state, "rl", a ), y[:]; tol = tol)
    L1 = reshape(L1, size(y))

    L1 
end

function sumright1(B, p, state::UMPS{:MF}, h, Rh, RB; tol = 1e-10)
    Al = data(state)[1]
    Ar = data(state)[2]

    @tensor All[:] := Al[-1,-2,1]*Al[1,-3,-4]
    @tensor Arr[:] := Ar[-1,-2,1]*Ar[1,-3,-4]

    @tensor y1[:] := B[-1,2,1]*Rh[1,3]*conj(Ar)[-2,2,3]
    @tensor y2[:] := B[-1,3,1]*Ar[1,5,2]*conj(Arr)[-2,4,6,2]*h[3,4,5,6]
    @tensor y3[:] := Al[-1,3,1]*B[1,5,2]*conj(Arr)[-2,4,6,2]*h[3,4,5,6]
    @tensor y4[:] := All[-1,3,5,1]*RB[1,2]*conj(Arr)[-2,4,6,2]*h[3,4,5,6]

    y = y1 + y2 + exp(im*p)*y3 + exp(2*im*p)*y4

    a = exp(im*p)
    R1, ~ = linsolve(x -> leftapplyTM(x, state, "lr", a ), y[:]; tol = tol)
    R1 = reshape(R1, size(y)) 

    R1
end

# x*(1 - (T - fp) ) = y  
function rightapplyTM(x, state::UMPS{:MF}, t::Tag{:rl}, a)
    Al = data(state)[1]
    Ar = data(state)[2]
    C = data(state)[4]

    D = size(Al,1)

    leftfp = conj(C') # left fixed point of transfer matrix T with left gauge 
    rightfp = C' # right fixed point of transfer matrix T with right gauge

    @tensor T[:] := a*Ar[-1,1,-3]* conj(Al)[-2,1,-4]
    x = reshape(x, D, D)
    @tensor y[:] := x[-1,-2] - x[1,2]*T[1,2,-1,-2] + x[1,2]*rightfp[1, 2]*leftfp[-1,-2]

    y = reshape(y, D*D)
    y
end

# (1 - (T - fp) )x = y  
function leftapplyTM(x, state::UMPS{:MF}, t::Tag{:lr}, a)
    Al = data(state)[1]
    Ar = data(state)[2]
    C = data(state)[4]

    D = size(Al,1)

    leftfp = conj(C) # left fixed point of transfer matrix T with left gauge 
    rightfp = C # right fixed point of transfer matrix T with right gauge

    @tensor T[:] := a*Al[-1,1,-3]* conj(Ar)[-2,1,-4]
    x = reshape(x, D, D)
    @tensor y[:] := x[-1,-2] - T[-1,-2,1,2]*x[1,2] + rightfp[-1, -2]*leftfp[1,2]*x[1,2]

    y = reshape(y, D*D)
    y
end