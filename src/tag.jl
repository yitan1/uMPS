struct Tag{name} 
end

Tag(s::AbstractString) = Tag{Symbol(s)}()