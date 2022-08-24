export spinmodel, op

using LinearAlgebra
using TensorOperations, KrylovKit
# two site hamiltonian

struct SiteType{T}
end
SiteType(s::AbstractString) = SiteType{Symbol(s)}()

# op -> operator
struct OpLabel{T}
end
OpLabel(s::AbstractString) = OpLabel{Symbol(s)}()

# SI = [1 0 0; 0 1 0; 1 0 0]
# Sx = [0 1.0 0; 1.0 0 1.0; 0 1.0 0]/sqrt(2);
# Sy = [0 -1im 0; 1im 0 -1im; 0 1im 0]/sqrt(2);
# Sz = [1.0 0 0; 0 0 0; 0 0 -1.0];
# Sp = real(Sx + 1im*Sy)
# Sm = real(Sx - 1im*Sy)

# spin model 
function spinmodel(a = 1, b = 1, c = 1, d = 0 ; s = 1/2)
    if s == 1/2
        SI = op("SI", "Spinhalf") 
        Sx = op("Sx", "Spinhalf")
        Sy = op("Sy", "Spinhalf")
        Sz = op("Sz", "Spinhalf")
        @tensor h[:] := a*Sx[-1,-2]*Sx[-3,-4] + b*Sy[-1,-2]*Sy[-3,-4] + c*Sz[-1,-2]*Sz[-3,-4] + 
                            d*Sx[-1,-2]*SI[-3,-4] + d*SI[-1,-2]*Sx[-3,-4]
        return h
    end
    if s == 1
        SI = op("SI", "Spinone") 
        Sx = op("Sx", "Spinone")
        Sy = op("Sy", "Spinone")
        Sz = op("Sz", "Spinone")
        @tensor h[:] := a*Sx[-1,-2]*Sx[-3,-4] + b*Sy[-1,-2]*Sy[-3,-4] + c*Sz[-1,-2]*Sz[-3,-4]
        return h
    end
end

op(label::AbstractString , st::AbstractString) = op(OpLabel(label), SiteType(st))

# operators of spinhalf
op(::OpLabel{:SI}, ::SiteType{:Spinhalf}) = [1 0; 0 1]
op(::OpLabel{:Sx}, ::SiteType{:Spinhalf}) = [0 1; 1 0]/2
op(::OpLabel{:Sy}, ::SiteType{:Spinhalf}) = [0.0 -1.0im; 1.0im 0.0]/2
op(::OpLabel{:Sz}, ::SiteType{:Spinhalf}) = [1 0; 0 -1]/2
op(::OpLabel{:Sp}, ::SiteType{:Spinhalf}) = [0 1; 0 0]
op(::OpLabel{:Sm}, ::SiteType{:Spinhalf}) = [0 0; 1 0]

# operators of spinone
op(::OpLabel{:SI}, ::SiteType{:Spinone}) = [1 0 0; 0 1 0; 1 0 0]
op(::OpLabel{:Sx}, ::SiteType{:Spinone}) = [0 1.0 0; 1.0 0 1.0; 0 1.0 0]/sqrt(2)
op(::OpLabel{:Sy}, ::SiteType{:Spinone}) = [0 -1im 0; 1im 0 -1im; 0 1im 0]/sqrt(2)
op(::OpLabel{:Sz}, ::SiteType{:Spinone}) = [1.0 0 0; 0 0 0; 0 0 -1.0]
# op(::OpLabel{:Sp}, ::SiteType{:Spinone}) = [0 1; 0 0]
# op(::OpLabel{:Sm}, ::SiteType{:Spinone}) = [0 0; 1 0]