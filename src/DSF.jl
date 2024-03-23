function broadening(es, swk0; step = 0.1, factor = 0.05, x_max = 0)
    if x_max > 0
        x_max = x_max
    else
        x_max = ceil(maximum(es))
    end
    x = 0:step:x_max

    y =  [gauss_broad(x[i], es, swk0, factor) for i in eachindex(x)]
    
    x, y
end

function lor_broad(x, es, swk, factor)
    w = 0.0
    for i in eachindex(swk)
        w += 1/pi*factor/((x - es[i])^2 + factor^2)*swk[i]
    end

    w
end

function gauss_broad(x, es, swk, factor)
    w = 0.0
    for i in eachindex(swk)
        w += 1/sqrt(4*pi*factor)*exp(- (x - es[i])^2 /(4*factor)) *swk[i]
    end

    w
end


function dsf_all(op, X, VL, state::UMPS{:MF}, p)
    ws = fill(-1.0-1.0im, size(X,1))
    for i in eachindex(X)
        rX = reshape(X[i], size(VL,3), :)
        @tensor B[:] := VL[-1,-2,1]*rX[1,-3]
        
        ws[i] = dsf(op, B, state, p)
    end
    ws
end

function dsf(op, B, state::UMPS{:MF}, p)
    Al = data(state)[1]
    Ar = data(state)[2]
    Ac = data(state)[3]

    LB = sumLB(B, p, state)
    RB = sumRB(B, p, state)

    @tensor d1[:] := B[1,3,2]*op[3,4]*conj(Ac)[1,4,2]
    @tensor d2[:] := Al[1,2,4]*op[2,3]*conj(Al)[1,3,5]*RB[4,5]
    @tensor d3[:] := Ar[4,2,1]*op[2,3]*conj(Ar)[5,3,1]*LB[4,5]

    d = d1[] + exp(im*p)*d2[] + exp(-im*p)*d3[]

    return d
end


function sumLB(B, p, state::UMPS{:MF}; tol = 1e-10)
    Al = data(state)[1]
    Ac = data(state)[3]

    D = size(Al,1)

    a = exp(-im*p)  

    @tensor y[:] := B[1,2,-1]*conj(Ac)[1,2,-2] 

    LB, ~ = linsolve(x -> rightapplyTM(x, state, "rr", a), y; tol = tol)
    LB = reshape(LB, D, D)

    LB
end

function sumRB(B, p, state::UMPS{:MF}; tol = 1e-10)
    Al = data(state)[1]
    Ac = data(state)[3]

    D = size(Al,1)

    a = exp(-im*p)  
    @tensor y[:] := B[-1,1,2]*conj(Ac)[-2,1,2] 
    RB, ~ = linsolve(x -> leftapplyTM(x, state, "ll", a), y[:]; tol = tol)
    
    RB = reshape(RB, D, D)
    
    RB
end

#########################
function leftapplyTM(x, state::UMPS{:MF}, t::Tag{:ll}, a)
    Al = data(state)[1]
    Ar = data(state)[2]
    C = data(state)[4]

    D = size(Ar,1)

    leftfp = diagm(ones(D)) # left fixed point of transfer matrix T with left gauge 
    @tensor rightfp[:] := C[-1,1]*conj(C)[-2,1] # right fixed point of transfer matrix T with right gauge

    @tensor T[:] := a*Al[-1,1,-3]* conj(Al)[-2,1,-4]
    x = reshape(x, D, D)
    @tensor y[:] := x[-1,-2] - T[-1,-2,1,2]*x[1,2] + a*leftfp[1,2]*x[1,2]*rightfp[-1,-2]

    y = reshape(y, D*D)
    y
end

function rightapplyTM(x, state::UMPS{:MF}, t::Tag{:rr}, a)
    Al = data(state)[1]
    Ar = data(state)[2]
    C = data(state)[4]

    D = size(Al,1)

    @tensor leftfp[:] :=  C[1,-1]*conj(C)[1,-2] # left fixed point of transfer matrix T with left gauge 
    rightfp = diagm(ones(D))        # right fixed point of transfer matrix T with right gauge

    @tensor T[:] := a*Ar[-1,1,-3]* conj(Ar)[-2,1,-4]
    x = reshape(x, D, D)
    @tensor y[:] := x[-1,-2] - x[1,2]*T[1,2,-1,-2] + a*x[1,2]*rightfp[1, 2]*leftfp[-1,-2]

    y = reshape(y, D*D)
    y
end