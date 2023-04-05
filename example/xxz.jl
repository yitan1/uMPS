using uMPS
using JLD2

h0 = spinmodel(1,-1,-4;s = 1/2);

D = 33;
d = size(h0,1);

phi0 = randUMPS(D,d,"UF");

state, e = vumps(h0, phi0);

println("The dimension of exicited states is $(D*D*(d-1))")
for p = 1:20
    println("The momentum k is $(p/10)pi")
    Ek, X = excitation(h0, p*pi/10, state, 30)
    jldsave("example/erg_xxz_$(p/10)pi.jld2"; Ek, X = X)
end

using CairoMakie
y = load("example/erg_xxz_0.1pi.jld2", "Ek") |> real
x = fill(pi/10, size(y))
f, ax, s = scatter(x, 2y)
for p = 1:20
    y = load("example/erg_xxz_$(p/10)pi.jld2", "Ek") |> real
    x = fill(p*pi/10, size(y))
    scatter!(x, 2y)
end
ylims!(0, 15)
ax.yticks = 0:1:15
f 