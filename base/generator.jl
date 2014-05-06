# Remove once part of standard library:
import Base: return_type, show, writemime, size, strides, isassigned, getindex, convert, ntuple, length

abstract AbstractGenerator{T,N} <: DenseArray{T,N}

immutable Generator{T,N,IS} <: AbstractGenerator{T,N}
    name::UTF8String
    func::Function # Must return values of type T
    itrs::IS
end

Generator(name, func, itrs) = Generator(name, func, itrs, map(eltype, itrs))
function Generator{I<:Tuple}(name::String, func::Function, itrs::I, arg_types)
    T = return_type(func, arg_types)
    Generator{T,length(I),I}(name, func, itrs)
end

macro generator(thunk, assignments...)
    args = map(e->e.args[1], assignments)
    itrs = map(e->e.args[2], assignments)
    itr_str = join(map(string, assignments),", ")
    name = "($thunk for $itr_str)"
    # Build the Generator constructor call programmatically:
    # 1. Create a list of eltype calls (eltype(itr1), eltype(itr2), ...)
    type_exprs = map(sym -> Expr(:call, :eltype, sym), itrs)
    # 2. Combine with arg symbols: (sym1::eltype(itr1), sym2::eltype(itr2), ...)
    typed_args = map((sym,typ) -> Expr(symbol("::"), sym, typ), args, type_exprs)
    # 3. And create the anonymous func (with the args in a tuple expression)
    # func = Expr(symbol("->"), Expr(:tuple, typed_args...), thunk)
    func = Expr(symbol("->"), Expr(:tuple, args...), thunk)
    # 4. Finally, return the top-level Generator call
    Expr(:call, :Generator, name, func, Expr(:tuple, itrs...), Expr(:tuple, type_exprs...))
end


# Display without showing the contents
writemime(io::IO, ::MIME"text/plain", g::AbstractGenerator) = show(io, g)
show(io::IO, g::AbstractGenerator) = print(io, summary(g), ": ", g.name)

## Basic functions ##
size(g::Generator) = map(length, g.itrs)
size{T,N}(g::Generator{T,N}, d) = d > N ? 1 : length(g.itrs[d])
length(g::Generator) = prod(size(g)) # May be bigger than Int

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

# Generic cases (with linear indexing)
function getindex{T,N}(g::Generator{T,N}, Idxs::Real...)
    length(Idxs) == N && return g.func(map(getindex, g.itrs, Idxs)...)::T
    # TODO: Handle the intermediate case 1<length(Idxs)<N more efficiently
    linear_idx = length(Idxs) > 1 ? sub2ind(size(g),Idxs...) : Idxs[1]
    subs = ind2sub(size(g), linear_idx)
    g.func(map(getindex, g.itrs, subs)...)::T
end
# Indexing: obvious easy cases
getindex{T,I}(g::Generator{T,1,I}, i1::Real)                               = g.func(g.itrs[1][i1])::T
getindex{T,I}(g::Generator{T,2,I}, i1::Real, i2::Real)                     = g.func(g.itrs[1][i1], g.itrs[2][i2])::T
getindex{T,I}(g::Generator{T,3,I}, i1::Real, i2::Real, i3::Real)           = g.func(g.itrs[1][i1], g.itrs[2][i2], g.itrs[3][i3])::T
getindex{T,I}(g::Generator{T,4,I}, i1::Real, i2::Real, i3::Real, i4::Real) = g.func(g.itrs[1][i1], g.itrs[2][i2], g.itrs[3][i3], g.itrs[4][i4])::T

# Some potential convenient uses of generators
ntuple(g::AbstractGenerator) = ntuple(length(g), i -> g[i])
# Dict(g::AbstractGenerator) just works as inherited from AbstractArray
