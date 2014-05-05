# Remove once part of standard library:
import Base: return_type, show, writemime, size, strides, isassigned, getindex, convert, ntuple

abstract AbstractGenerator{T,N} <: DenseArray{T,N}

immutable Generator1{T,I} <: AbstractGenerator{T,1}
    name::UTF8String
    func::Function # must return elements of type T
    itr1::I
end
immutable Generator2{T,I1,I2} <: AbstractGenerator{T,2}
    name::UTF8String
    func::Function
    itr1::I1
    itr2::I2
end
immutable NGenerator{T,N,IS} <: AbstractGenerator{T,N}
    name::UTF8String
    func::Function
    itrs::IS
end

Generator(name, func, itr1)       = Generator1(name, func, itr1)
Generator(name, func, itr1, itr2) = Generator2(name, func, itr1, itr2)
Generator(name, func, itrs...)    = NGenerator(name, func, itrs)

function Generator1{I}(name::String, func::Function, itr::I)
    T = return_type(func, (eltype(itr),))
    Generator1{T,I}(name, func, itr)
end
function Generator2{I1,I2}(name::String, func::Function, itr1::I1, itr2::I2)
    T = return_type(func, (eltype(itr1), eltype(itr2)))
    Generator2{T,I1,I2}(name, func, itr1, itr2)
end
function NGenerator{I}(name::String, func::Function, itrs::I)
    T = return_type(func, map(eltype, itrs)::Type)
    NGenerator{T,length(I),I}(name, func, itrs)
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
writemime(io::IO, ::MIME"text/plain", g::AbstractGenerator) = show(io, g)
show(io::IO, g::AbstractGenerator) = print(io, summary(g), ": ", g.name)

## Basic functions ##
size(g::NGenerator) = map(length, g.itrs)
size{T,N}(g::NGenerator{T,N}, d) = d > N ? 1 : length(g.itrs[d])
size(g::Generator1)    = (length(g.itr1),)
size(g::Generator1, d) = d > 1 ? 1 : length(g.itrs[d])
size(g::Generator2)    = (length(g.itr1), length(g.itr2))
size(g::Generator2, d) = d == 1 ? length(g.itr1) : 
                         d == 2 ? length(g.itr2) : 1

strides{T}(g::AbstractGenerator{T,1}) = (1,)
strides{T}(g::AbstractGenerator{T,2}) = (1, size(g,1))
strides{T}(g::AbstractGenerator{T,3}) = (1, size(g,1), size(g,1)*size(g,2))
isassigned(g::AbstractGenerator, i::Int) = 1 <= i <= length(g)

## Conversion and expansion ##
function full{T,N}(g::AbstractGenerator{T,N})
    A = Array(T, size(g))
    @inbounds for i = 1:length(A)
        A[i] = g[i]
    end
    return A
end
convert{T,S,N}(::Type{Array{T}}, g::AbstractGenerator{S,N}) = full(g)

# Indexing: obvious easy cases
getindex(g::Generator1, i1::Real)           = g.func(g.itr1[i1])
getindex(g::Generator2, i1::Real, i2::Real) = g.func(g.itr1[i1], g.itr2[i2])
function getindex(g::Generator2, i1::Real)
    subs = ind2sub(size(g), i1)
    g.func(g.itr1[subs[1]], g.itr2[subs[2]])
end
# Generic cases (with linear indexing)
function getindex{T,N}(g::NGenerator{T,N}, Idxs::Real...)
    length(Idxs) == N && return g.func(map(getindex, g.itrs, Idxs)...)
    # The intermediate case 1<length(Idxs)<N could be handled more efficiently
    linear_idx = length(Idxs) > 1 ? sub2ind(size(g),Idxs...) : Idxs[1]
    subs = ind2sub(size(g), linear_idx)
    g.func(map(getindex, g.itrs, subs)...)
end
getindex{T}(g::NGenerator{T,1}, i1::Real)                               = g.func(g.itrs[1][i1])
getindex{T}(g::NGenerator{T,2}, i1::Real, i2::Real)                     = g.func(g.itrs[1][i1], g.itrs[2][i2])
getindex{T}(g::NGenerator{T,3}, i1::Real, i2::Real, i3::Real)           = g.func(g.itrs[1][i1], g.itrs[2][i2], g.itrs[3][i3])
getindex{T}(g::NGenerator{T,4}, i1::Real, i2::Real, i3::Real, i4::Real) = g.func(g.itrs[1][i1], g.itrs[2][i2], g.itrs[3][i3], g.itrs[4][i4])

# Some potential convenient uses of generators
ntuple(g::AbstractGenerator) = ntuple(length(g), i -> g[i])
# Dict(g::AbstractGenerator) just works as inherited from AbstractArray
