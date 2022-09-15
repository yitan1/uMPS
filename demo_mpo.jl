using uMPS
import ITensors

Nx = 4
Ny = 3
sites = ITensors.siteinds("S=1/2",Nx*Ny)
ampo = ITensors.OpSum()

for n = 1:Nx*Ny
    iy = (n-1)÷Nx + 1
    ix = (n-1)%Nx + 1
    @show ix, iy
    if ix == Nx 
        ampo += 1.0,"Sz",n,"Sz",n-Nx+1
        ampo += 1.0,"Sx",n,"Sx",n-Nx+1
        ampo += 1.0,"Sy",n,"Sy",n-Nx+1
    else
        ampo += 1.0,"Sz",n,"Sz",n+1
        ampo += 1.0,"Sx",n,"Sx",n+1
        ampo += 1.0,"Sy",n,"Sy",n+1
    end
    if iy < Ny
        ampo += 1.0,"Sz",n,"Sz",n+Nx
        ampo += 1.0,"Sx",n,"Sx",n+Nx
        ampo += 1.0,"Sy",n,"Sy",n+Nx
    end
end

H = ITensors.MPO(ampo,sites);

M = mpo(Nx)

M[1][1] = H[1].tensor.storage.data |>  x -> reshape(x, size(H[1])) |> x -> permutedims(x, (2,1,3));
for n = 2:(3*Nx-1)
    iy = (n-1)÷Nx + 1
    ix = (n-1)%Nx + 1
    @show n,ix,iy
    M[iy][ix] = H[n].tensor.storage.data |>  x -> reshape(x, size(H[n])) |> x -> permutedims(x, (1,3,2,4))
end

M[3][Nx] = H[end].tensor.storage.data |>  x -> reshape(x, size(H[end])) |> x -> permutedims(x, (2,1,3));

d = 2
D = 10

phi0 = rand_multiUMPS(Nx::Int, D, d; form = "MF");

phi, e = vumps(M, phi0);


#########
N = length(phi0)

vecM, vecphi = trans_to_single(M, phi0); 

Mm = data(vecM)[2];
D = size(Al, 1);
dm = size(Mm, 1);

a = 14
Mm[a,:,a,:] |> sum

is_identity(Mm[a,:,a,:])

reduce(*, isapprox.(0.0, Mm[10,:,10,:];atol=1e-10))

fl = zeros(D,dm,D);
fl[:,dm,:] = diagm(ones(ComplexF64,D) .+ 0.2im);

eltype(M[2][2])