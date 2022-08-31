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
    SI = op("SI", "Spinhalf") 
    Sx = op("Sx", "Spinhalf")
    Sy = op("Sy", "Spinhalf")
    Sz = op("Sz", "Spinhalf")
    Sp = op("Sp", "Spinhalf")
    Sm = op("Sm", "Spinhalf")
    d = 2
    mm = zeros(ComplexF64, 5, d, 5, d)
    mm[1,:,1,:] = SI
    mm[1,:,2,:] = Sp
    mm[1,:,3,:] = Sm
    mm[1,:,4,:] = Sz
    mm[2,:,5,:] = J/2*Sm
    mm[3,:,5,:] = J/2*Sp
    mm[4,:,5,:] = J*Jz*Sz
    mm[5,:,5,:] = SI
    ml = mm[1,:,:,:]
    mr = permutedims(mm[:,:,5,:],(2,1,3))

    H = MPO([ml, mm, mr])
end
