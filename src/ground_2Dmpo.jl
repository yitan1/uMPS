

function vumps(M::MPO, phi0::UMPS{:UF}; tol = 1e-10)
    state, ~ = mixed_canonical(phi0; tol = tol)

    return vumps(M, state; tol = 1e-10)
end

 
