# Matrix product operator
#     4
#     |
# 1 - M - 3
#     |
#     2
    #     3
    #     |
    #     Ml - 2
    #     |
    #     1
#     3
#     |
# 2 - Mr 
#     |
#     1
struct MPO
    data
end

data(A::MPO) = A.data

export heisenberg

function heisenberg(J = 1, Jz = 1)
    SI = op("SI", "Spinone") 
    Sx = op("Sx", "Spinone")
    Sy = op("Sy", "Spinone")
    Sz = op("Sz", "Spinone")
    Sp = op("Sp", "Spinone")
    Sm = op("Sm", "Spinone")
    d = size(SI, 1)
    mm = zeros(ComplexF64, 5, d, 5, d)
    mm[1,:,1,:] = SI
    mm[2,:,1,:] = Sp
    mm[3,:,1,:] = Sm
    mm[4,:,1,:] = Sz
    mm[5,:,2,:] = J/2*Sm
    mm[5,:,3,:] = J/2*Sp
    mm[5,:,4,:] = J*Jz*Sz
    mm[5,:,5,:] = SI
    mm = permutedims(mm, (3,2,1,4))
    ml = mm[1,:,:,:]
    mr = permutedims(mm[:,:,5,:],(2,1,3))

    H = MPO(real.([ml, mm, mr]))
end
