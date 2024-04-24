using uMPS
using JLD2
using TensorOperations

dir = "example/hb_B1_s1_D30"

h0 = spinmodel(1,1,1, 1;s = 1);
D = 30;
d = size(h0,1);

phi0 = randUMPS(D,d,"UF");
state, e = vumps(h0, phi0);
jldsave("$(dir)/gs.jld2"; gs = state, e0 = e)

println("The dimension of exicited states is $(D*D*(d-1))")

gs = load("$(dir)/gs.jld2")
state = gs["gs"]
function optim_es(h0, n, state, N = 200)
    println("The momentum k is $(n)pi")
    Ek, X, VL = excitation(h0, n*pi, state, N)
    jldsave("$(dir)/es$(N)_$(n)pi.jld2"; Ek = Ek, X = X, VL= VL)
end

for n = 0:9
    p0 = n/10
    p = (n + 0.5)/10
    optim_es(h0, p0, state)
    optim_es(h0, p, state)
end