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
    data::Vector{Tensor}
end

data(A::MPO) = A.data

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
    mm[2,:,1,:] = Sp
    mm[3,:,1,:] = Sm
    mm[4,:,1,:] = Sz
    mm[5,:,2,:] = J/2*Sm
    mm[5,:,3,:] = J/2*Sp
    mm[5,:,4,:] = J*Jz*Sz
    mm[5,:,5,:] = SI
    ml = mm[5,:,:,:]
    mr = permutedims(mm[:,:,1,:],(2,1,3))

    H = MPO([ml, mm, mr])
end
