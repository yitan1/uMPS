using uMPS
using JLD2
using TensorOperations

h0 = spinmodel(1,1,1;s = 1);

D = 30;
d = size(h0,1);

phi0 = randUMPS(D,d,"UF");

state, e = vumps(h0, phi0);

dir = "example/heisenberg_s1_D30"

jldsave("$(dir)/gs.jld2"; gs = state, e0 = e)

println("The dimension of exicited states is $(D*D*(d-1))")
# for p = 1:19
    p = 0
    println("The momentum k is $(p/10)pi")
    Ek, X, VL = excitation(h0, p*pi/10, state, 200)
    jldsave("$(dir)/es_$(p/10)pi.jld2"; Ek = Ek, X = X, VL= VL)
# end

f = load("$dir/es_1.0pi.jld2")
state = load("$dir/gs.jld2", "gs");
Ek = f["Ek"]
X = f["X"]
VL = f["VL"]
op = [0 1. 0; 1. 0 1.0; 0 1. 0]/sqrt(2)
ws = dsf_all(op, X, VL, state, pi)

x, y = broadening(real(Ek), abs2.(ws))
lines(x, y)


using CairoMakie
y = load("$(dir)/es_0.0pi.jld2", "Ek") |> real
x = fill(0, size(y))
f = Figure()
ax = Axis(f[1, 1])
scatter!(ax, x, y/0.4104)
for p = 1:19
    y = load("$(dir)/es_$(p/10)pi.jld2", "Ek") |> real
    x = fill(p*pi/10, size(y))
    scatter!(ax, x, y/0.4104)
end
ylims!(0, 15)
ax.yticks = 0:1:15
f
scatter!(ax, fill(-2,size(Ek)), real(Ek)/0.4104)