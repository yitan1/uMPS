# export umps, randUMPS, data, mixed_canonical

# struct Form{T}
# end
# Form(s::AbstractString) = Form{Symbol(s)}()

# Form: uniform(UF): (A)
# left(right) canonical form(L(R)F), 
# mixed canonical form(MF): (Al, Ar, Ac, C)
abstract type AbstractUMPS end

const Tensor{T,N} = Array{T,N}

mutable struct SingleUMPS{Form} <: AbstractUMPS
    data::Vector{Tensor}
end

mutable struct MultiUMPS{Form} <: AbstractUMPS
    data::Vector{SingleUMPS{Form}}
end

Base.size(A::AbstractUMPS) = size(data(A))
Base.size(::SingleUMPS) = 1
Base.length(A::AbstractUMPS) = length(data(A))

Base.getindex(A::AbstractUMPS, n) = getindex(data(A), n)

function Base.setindex!(A::AbstractUMPS, v, i::Int)
    data(A)[i] = v
end

data(A::AbstractUMPS) = A.data

# set_data!(A::SingleUMPS, data)

umps(data::Tensor{T,3}) where {T <: Number} = umps([data], "UF")

umps(data::Vector{<:Tensor}, s::AbstractString)  = SingleUMPS{Symbol(s)}(data)

umps(s::AbstractString, N::Int) = MultiUMPS{Symbol(s)}(Vector{SingleUMPS{Symbol(s)}}(undef, N))

# function randUMPS end

rand_singleUMPS(D::Int, d::Int, s::AbstractString) = rand_singleUMPS(D, d, Tag(s))

rand_singleUMPS(D::Int, d::Int, ::Tag{:UF}) = umps(rand(D,d,D))

function rand_singleUMPS(D::Int, d::Int, ::Tag{:MF})
    Ac = rand(D,d,D)
    C = diagm(rand(D))
    C = C/norm(C)
    Al, Ar = getAlAr(Ac, C)

    umps([Al, Ar, Ac, C], "MF")
end

function rand_multiUMPS(N::Int, D::Int,d::Int; form = "MF")
    phi = umps(form, N)
    Cn = diagm(rand(D)) |> x-> x/norm(x)
    preC = Cn

    for i = 1:N
        Ac = rand(D,d,D)
        if i == N
            C = Cn
        else
            C = diagm(rand(D)) |> x-> x/norm(x)
        end

        Al, _ , epl, _ = getAlAr(Ac, C)
        _ , Ar, _, epr = getAlAr(Ac, preC)

        phi[i] = umps([Al,Ar,Ac,C], form)

        preC = C
    end
    phi
end

function mixed_canonical(state::SingleUMPS{:UF}, C0 = 0; tol = 1e-10)
    A = data(state)[1]
    if C0 == 0
        C0 = diagm(rand(size(A)[3]))
        C0 = C0/norm(C0)
    end

    Al, ~, lambda = left_canonical(A, C0; tol = 1e-10)
    Ar, C, ~ = right_canonical(Al, C0; tol = 1e-10)
    
    U,C,V = svd(C)
    Vt = V'
    
    C = diagm(C)/norm(C)
    
    @tensor Al[:] = U'[-1,2]*Al[2,-2,3]*U[3,-3]
    @tensor Ar[:] = Vt[-1,2]*Ar[2,-2,3]*V[3,-3]
    @tensor Ac[:] := Al[-1,-2,1]*C[1,-3]

    state = umps([Al,Ar,Ac,C], "MF")

    state, lambda
end


function right_canonical(A::Tensor, R0::Matrix; tol = 1e-10)

    D = size(A, 1)
    
    Ar = permutedims(A, (3,2,1))
    R = permutedims(R0, (2,1))
    
    Ar, R, lambda = left_canonical(Ar, R; tol = 1e-10)
    Ar = permutedims(Ar, (3,2,1))
    R = permutedims(R, (2,1))

    Ar, R, lambda
end


function left_canonical(A::Tensor, L0::AbstractMatrix; tol = 1e-10)
    D = size(A, 1)
    d = size(A, 2)
    
    L = L0/norm(L0)
    
    ~, L = qr_pos(L);
    L = L/norm(L)
    
    Lold = L
    
    @tensor LA[:] := L[-1, 1]*A[1,-2,-3]
    LA = reshape(LA, D*d, D)
    
    Al, L = qr_pos(LA)
    Al = reshape(Al, D, d, D)
    
    lambda = norm(L)
    L = L/lambda
    
    delta = norm(L - Lold)
    
    maxiter = 1e4
    
    for i = 1:maxiter
        L = reshape(L, D*D)
        ~, L = eigsolve(x -> fixpoint_map(x, A, Al), L, 1, :LM; 
                         issymmetric = false, tol = delta/10)
        L = reshape(real(L[1]), D, D)
        
        ~, L = qr_pos(L)
        L = L/norm(L)
        
        Lold = L
        
        @tensor LA[:] := L[-1, 1]*A[1,-2,-3]
        LA = reshape(LA, D*d, D)
        Al, L = qr_pos(LA)
        Al = reshape(Al, D, d, D)
        
        lambda = norm(L)
        L = L/lambda
        
        delta = norm(L - Lold)
        
        if delta <= tol
#             @show i
            break
        end
    end
#     @show delta
    
    Al, L, lambda
end

function qr_pos(A::AbstractMatrix)
#      QR decomposition with positive diagonal entry    
    Q, R = qr(A);
    D = diagm(sign.(diag(R)))
    Q = Q*D
    R = D*R
    
    Q, R
end
    
function fixpoint_map(x::AbstractVector, A::Tensor, Al::Tensor)
    D = size(A,1)
    Deff = size(Al,1)
    x = reshape(x, Deff, D)
    @tensor y[:] := x[1,3]*conj(Al)[1,2,-1]*A[3, 2, -2]
    y = reshape(y, Deff*D)
    
    y
end