export trans_to_single, leftfixpoint, is_identity

function vumps(M::MPO, phi::MultiUMPS{:MF}; tol = 1e-10)
    N = length(phi)

    maxitr = 1e3
    vec_ep = zeros(N,2)
    
    e = 0.0

    for i = 1:maxitr 

        vecM, vecphi = trans_to_single(M, phi) 

        el, fl = leftfixpoint(vecM, vecphi)
        er, fr = rightfixpoint(vecM, vecphi)

        Threads.@threads for n = 1:N
            Ln = leftsumN(n, phi, M, fl)
            Rn = rightsumN(n, phi, M, fr)

            Ln1 = leftsumN(n+1, phi, M, fl)

            EAc, Ac0 = eigsolve(x -> HmAc(x, M[2][n], Ln, Rn), D*d*D, 1, :SR)
            Ec, C0 = eigsolve(x -> Hmc(x, Ln1, Rn), D*D, 1, :SR)

            Ac = reshape(Ac0[1], size(Ac))
            C = reshape(C0[1], size(C))
            
            phi[n][3] = Ac
            phi[n][4] = C

        end

        Threads.@threads for n = 1:N
            Ac = phi[n][3]
            C = phi[n][4]

            preC = n==1 ? phi[N][4] : phi[n-1][4]

            Al, _ , epl, _ = getAlAr(Ac, C)
            _ , Ar, _, epr = getAlAr(Ac, preC)

            phi[n][1] = Al
            phi[n][2] = Ar

            vec_ep[n, 1] = epl
            vec_ep[n, 2] = epr
        end

        delta = maximum(vec_ep)
        @show i, el, er, delta
        if delta < tol
            e = el
            @show i, e, delta
            break
        end
    end

    phi, e
end

 
function trans_to_single(M::MPO, phi::MultiUMPS{:MF}; tol = 1e-10) 
    N = length(phi)

    vecAl = phi[1][1]
    vecAr = phi[1][2]
    vecM = M[2][1]

    C = phi[N][4]
    vecAc = [0.0]
    # @tensor VecTM[:] := Al[-1,1,-4]*M1[-2,2,-5,1]*conj(Al)[-3,2,-6] 

    for i = 2:N
        Al = phi[i][1]
        @tensor vecAl[:] := vecAl[-1,-2,1]*Al[1,-3,-4]
        dimvecAl = size(vecAl)
        vecAl = reshape(vecAl, dimvecAl[1], dimvecAl[2]*dimvecAl[3], dimvecAl[4] )

        Ar = phi[i][2]
        @tensor vecAr[:] := vecAr[-1,-2,1]*Ar[1,-3,-4]
        dimvecAr = size(vecAr)
        vecAr = reshape(vecAr, dimvecAr[1], dimvecAr[2]*dimvecAr[3], dimvecAr[4] )

        Mi = M[2][i]
        @tensor vecM[:] := vecM[-1,-2,1,-6]*Mi[1,-3,-4,-5]
        dimvecM = size(vecM)
        vecM = reshape(vecM, dimvecM[1], dimvecM[2]*dimvecM[3], dimvecM[4], dimvecM[5]*dimvecM[6])
    end


    MPO([ [0], vecM, [0] ]), umps([vecAl, vecAr, [0], C], "MF")
end

function leftsumN(N::Int, phi::MultiUMPS{:MF}, M::MPO, fl::Tensor{T, 3}) where T
    for i = 1:(N-1)
        Al = phi[i][1]
        Mm = M[2][i]

        @tensor tempT[:] := Al[-1,1,-4]*Mm[-2,2,-5,1]*conj(Al)[-3,2,-6] 
        @tensor fl[:] = fl[1,2,3]*tempT[1,2,3,-1,-2,-3]
    end

    fl
end

function rightsumN(N::Int, phi::MultiUMPS{:MF}, M::MPO, fr::Tensor{T, 3}) where T
    for i = length(phi):-1:N+1
        Ar = phi[i][2]
        Mm = M[2][i]

        @tensor tempT[:] := Ar[-1,1,-4]*Mm[-2,2,-5,1]*conj(Ar)[-3,2,-6] 
        @tensor fr[:] = tempT[-1,-2,-3,1,2,3]*fr[1,2,3]
    end
    
    fr
end
