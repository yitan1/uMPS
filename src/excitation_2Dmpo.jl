
function excitation(M::MPO, p::Float64, phi::MultiUMPS{:MF}; N::Int64 = 1, tol = 1e-10)
    
    vecM, vecphi = trans_to_single(M, phi) 
    _, fl = leftfixpoint(vecM, vecphi)
    _, fr = rightfixpoint(vecM, vecphi)

    L = length(phi)
    vecVL = Vector{Array}(undef, L) # NS -> nullspace
    for i = 1:L
        Al = phi[i][1] ## dim: D*d*D 
        vecVL[i] = nullspaceforAl(Al)  ## dim: D*d*D(d-1)
    end

    D = size(phi[1][1], 1)
    D2 = size(VL[1], 3)

    En, X = eigsolve(_x -> Heff_mpo(_x, vecVL, p, phi, M, fl, fr), L*D2*D, N, :SR)

end

function Heff_mpo(vecX, vecVL, p, phi, M, fl, fr)
    N = length(phi)  ## number of sites in unit cell
    vecX = reshape(vecX, :, N)

    y = zero(vecX) 

    Threads.@threads for i = 1:N
        # Xi = vecX[:,i] |> x -> reshape(x, D2, D)
        # @tensor Bi[:] := VLi[-1,-2,1]*Xi[1,-3]
        Bi = zero(phi[i][1])
        for j = 1:N
            Xj = vecX(:, j) |> x -> reshape(x, D2, D)
            VLj = vecVL[j]
            @tensor Bj[:] := VLj[-1,-2,1]*Xj[1,-3]
            Bi = Bi .+ Heff_ij(i, j, Bj, p, phi, M, fl, fr)
        end

        VLi = vecVL[i]
        @tensor temp[:] := Bi[1,2,-2] * conj(VLi)[1,2,-1]
        y[:,i] = temp[:]
    end
    
    y[:]
end

# Heff_ij (B_j) -> B_i
function Heff_ij(i, j, Bj, p, phi, M, fl, fr)  
    N = length(phi)

    LBj = leftsum_Bj(j, Bj, p, phi, M, fl)
    RBj = rightsum_Bj(j, Bj, p, phi, M, fr)

    # diagram 1
    # for n = 1:N
    #     d1 = fl
    # end
    # diagram 2
    # Bi
end

function leftsum_Bj(j, Bj, p, phi, M, fl)
    
end

function rightsum_Bj(j, Bj, p, phi, M, fr)
    
end