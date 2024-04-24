using uMPS
using JLD2
using TensorOperations

dir = "example/heisenberg_s1_D30"

h0 = spinmodel(1,1,1;s = 1);
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

# N: 30, 50, 100, 200, 300
for i in [30, 50, 100, 150, 250, 300, 350, 400]
    optim_es(h0, 1, state, i)
end

for n = 3:9
    p = (n + 0.5)/10
    optim_es(h0, p, state)
end

sx = [0 1. 0; 1. 0 1.0; 0 1. 0]/sqrt(2)
sy = [0 -1im 0; 1im 0 -1im; 0 1im 0]/sqrt(2)
sz = [1. 0 0; 0 0 0; 0 0 -1.0]
function compute_dsf_xyz_k(k, N = 200)
    println("load $dir/es$(N)_$(k)pi.jld2")

    f = load("$dir/es$(N)_$(k)pi.jld2")
    state = load("$dir/gs.jld2", "gs");
    Ek = f["Ek"]; X = f["X"]; VL = f["VL"]
    wx = dsf_all(sx, X, VL, state, k*pi)
    wy = dsf_all(sy, X, VL, state, k*pi)
    wz = dsf_all(sz, X, VL, state, k*pi)

    jldsave("$(dir)/dsf$(N)_$(k)pi.jld2"; wx = wx, wy = wy, wz = wz, Ek = Ek)
end
function get_dsf_k(k, N = 200; factor = 0.1, x_max = 8.0, b = "lor")
    dir_f = "$dir/dsf$(N)_$(k)pi.jld2"
    if !isfile(dir_f)
        println("compute $dir_f")
        compute_dsf_xyz_k(k, N)
    end
    println("load $dir_f")
    f = load(dir_f)
    wx = f["wx"]
    wy = f["wy"]
    wz = f["wz"]
    Ek = f["Ek"]
    ws = wx .+ wy .+ wz

    x, y = broadening(real(Ek), abs2.(ws), factor = factor, x_max = x_max, broad = b)
    x, y
end

using CairoMakie

xs = Float64[]
ys = Float64[]
zs = Float64[]
for i = 0:0.05:1.0
    y1, z1 = get_dsf_k(i; factor = 0.1, x_max = 8.0, b = "lor")
    y = y1
    z = z1
    x = fill(i, size(y))
    append!(xs, x)
    append!(ys, y)
    append!(zs, z)

    if i < 1.0
        append!(xs, 2 .- x)
        append!(ys, y)
        append!(zs, z)
    end
end
fig = Figure()
ax, hm = heatmap(fig[1,1], xs, ys , log10.(zs), colorrange = (-2,1.5))
cb = Colorbar(fig[0, 1], hm, height = 25, vertical = false) #height = 20
rowsize!(fig.layout, 1, Aspect(1, 0.7))\
# cb.tickalign = 1
# cb.ticks = -1:1:1
# cb.ticksize = 10
# cb.ticklabelsvisible = false
ylims!(ax, 0, 5)
fig

y, z = get_dsf_k(1.0)
lines(y,z)


f = Figure()
ax = Axis(f[1, 1])
scatter!(ax, x, y)
for p = 0:0.05:1.0
    # y = load("$(dir)/es200_$(p)pi.jld2", "Ek") |> real
    # y = y/0.4104
    # x = fill(p, size(y))
    x, y = get_dsf_k(sx, p)
    scatter!(ax, x, y)
end
for p = 0.0:0.05:1.0
    y = load("$(dir)/es200_$(p)pi.jld2", "Ek") |> real
    y = y/0.4104
    x = fill(2-p, size(y))
    scatter!(ax, x, y)
end
ylims!(0, 12)
ax.yticks = 0:1:15
f
