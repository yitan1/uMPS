# export umps, randUMPS, data, mixed_canonical

# struct Form{T}
# end
# Form(s::AbstractString) = Form{Symbol(s)}()

# Form: uniform(UF): (A)
# left(right) canonical form(L(R)F), 
# mixed canonical form(MF): (Al, Ar, Ac, C)
mutable struct UMPS{Form}
    data
end
umps(data, s::AbstractString) = UMPS{Symbol(s)}(data)

data(A::UMPS) = A.data

function randUMPS end

randUMPS(D::Int64, d::Int64, s::AbstractString) = randUMPS(D, d, Tag(s))

randUMPS(D::Int64, d::Int64, tag::Tag{:UF}) = UMPS{:UF}(rand(D,d,D))

function mixed_canonical(state::UMPS{:UF}, C0 = 0; tol = 1e-10)
    A = data(state)
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


function right_canonical(A, R0; tol = 1e-10)

    D = size(A, 1)
    
    Ar = permutedims(A, (3,2,1))
    R = permutedims(R0, (2,1))
    
    Ar, R, lambda = left_canonical(Ar, R; tol = 1e-10)
    Ar = permutedims(Ar, (3,2,1))
    R = permutedims(R, (2,1))

    Ar, R, lambda
end


function left_canonical(A, L0; tol = 1e-10)
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

function qr_pos(A)
#      QR decomposition with positive diagonal entry    
    Q, R = qr(A);
    D = diagm(sign.(diag(R)))
    Q = Q*D
    R = D*R
    
    Q, R
end
    
function fixpoint_map(x, A, Al)
    D = size(A,1)
    Deff = size(Al,1)
    x = reshape(x, Deff, D)
    @tensor y[:] := x[1,3]*conj(Al)[1,2,-1]*A[3, 2, -2]
    y = reshape(y, Deff*D)
    
    y
end