# Remove once part of standard library:
import Base: return_type, show, writemime, size, strides, isassigned, getindex, convert, ntuple

immutable Generator{T,N} <: DenseArray{T,N}
    name::UTF8String
    func::Function # must return elements of type T
    itrs::NTuple{N}
    
    Generator(func::Function, itrs::NTuple{N}) = new("?? for ?? in ??s", func, itrs)
    Generator(name::String, func::Function, itrs::NTuple{N}) = new(name, func, itrs)
end

Generator(name::String, func::Function, itrs...) = Generator(name, func, itrs)
function Generator{N}(name::String, func::Function, itrs::NTuple{N})
    arg_types = map(eltype, itrs)::Type
    T = return_type(func, arg_types)
    Generator{T,N}(name, func, itrs)
end

macro generator(thunk, assignments...)
    args = map(e->e.args[1], assignments)
    itrs = map(e->e.args[2], assignments)
    itrs_str = join(map(string, assignments),", ")
    name = "($thunk for $itrs_str)"
    esc(quote
        Generator($name, ($(args...),) -> ($thunk), $(itrs...))
    end)
end

# Display without showing the contents
writemime(io::IO, ::MIME"text/plain", g::Generator) = show(io, g)
show(io::IO, g::Generator) = print(io, summary(g), ": ", g.name)

## Basic functions ##
size(g::Generator) = map(length, g.itrs)
size{T,N}(g::Generator{T,N}, d) = d > N ? 1 : length(g.itrs[d])
strides{T}(g::Generator{T,1}) = (1,)
strides{T}(g::Generator{T,2}) = (1, size(g,1))
strides{T}(g::Generator{T,3}) = (1, size(g,1), size(g,1)*size(g,2))
isassigned(g::Generator, i::Int) = 1 <= i <= length(g)

## Conversion and expansion ##
full{T,N}(g::Generator{T,N}) = convert(Array{T,N}, g)
convert{T,S,N}(::Type{Array{T}}, g::Generator{S,N}) = convert(Array{T,N},g)
function convert{T,S,N}(::Type{Array{T,N}}, g::Generator{S,N})
    A = Array(T, size(g))
    for i = 1:length(A)
        A[i] = g[i]
    end
    return A
end

# Generic case (with linear indexing)
function getindex{T,N}(g::Generator{T,N}, Idxs::Real...)
    length(Idxs) == N && return g.func(map(getindex, g.itrs, Idxs)...)
    # The intermediate case 1 < length(Idxs) < N could be handled better
    linear_idx = length(Idxs) > 1 ? sub2ind(size(g),Idxs...) : Idxs[1]
    subs = ind2sub(size(g), linear_idx)
    g.func(map(getindex, g.itrs, subs)...)
end
# Indexing: obvious easy cases
getindex{T}(g::Generator{T,1}, i1::Real)                               = g.func(g.itrs[1][i1])
getindex{T}(g::Generator{T,2}, i1::Real, i2::Real)                     = g.func(g.itrs[1][i1], g.itrs[2][i2])
getindex{T}(g::Generator{T,3}, i1::Real, i2::Real, i3::Real)           = g.func(g.itrs[1][i1], g.itrs[2][i2], g.itrs[3][i3])
getindex{T}(g::Generator{T,4}, i1::Real, i2::Real, i3::Real, i4::Real) = g.func(g.itrs[1][i1], g.itrs[2][i2], g.itrs[3][i3], g.itrs[4][i4])

# Some potential convenient uses of generators
ntuple(g::Generator) = ntuple(length(g), i -> g[i])
# Dict(g::Generator) just works as inherited from AbstractArray
