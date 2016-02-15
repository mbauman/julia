# This file is a part of Julia. License is MIT: http://julialang.org/license
# import Base: NonSliceIndex, linearindexing, LinearFast, LinearSlow, tail, @_inline_meta, @_propagate_inbounds_meta, size, length, getindex, setindex!, AbstractCartesianIndex, similar, first, step, isempty, in, pointer, unsafe_convert, convert, to_indexes, index_shape, merge_indexes, trailingsize

typealias NonSliceIndex Union{Colon, AbstractVector}
typealias ViewIndex Union{Int, NonSliceIndex}

# The NoSlice one-element vector type keeps dimensions without losing performance
immutable NoSlice <: AbstractVector{Int}
    i::Int
end
size(::NoSlice) = (1,)
length(::NoSlice) = 1
linearindexing(::Type{NoSlice}) = LinearFast()
function getindex(N::NoSlice, i::Int)
    @_inline_meta
    @boundscheck i == 1 || throw_boundserror(N, i)
    N.i
end
function getindex(N::NoSlice, i::NoSlice)
    @_inline_meta
    @boundscheck i.i == 1 || throw_boundserror(N, i)
    N
end
getindex(N::NoSlice, ::Colon) = N
function getindex(N::NoSlice, r::Range{Int})
    @_inline_meta
    @boundscheck checkbounds(N, r)
    N
end

# ID is the intrinsic dimensionality of the SubArray -- it determines the optimal
# number of indices to use when indexing into the view, for example.
immutable SubArray{T,N,P,I,ID} <: AbstractArray{T,N}
    parent::P
    indexes::I
    dims::NTuple{N,Int}
    first_index::Int   # for linear indexing and pointer, only valid for I<:StridedIndexTuple
    strides::NTuple{ID, Int} # only valid for I<:StridedIndexTuple
end
function SubArray{P,I,N,ID}(parent::P, indexes::I, dims::NTuple{N, Int}, first_index::Int, strides::NTuple{ID, Int})
    SubArray{eltype(P), N, P, I, ID}(parent, indexes, dims, first_index, strides)
end

# SubArray does extra setup work for strided indices into fast parents
function SubArray(parent::AbstractArray, indexes::Tuple, dims::Dims)
    SubArray(linearindexing(parent), parent, indexes, dims)
end
# SubArray just wraps the indices for a LinearSlow parent
function SubArray{P, I, N}(::LinearSlow, parent::P, indexes::I, dims::NTuple{N, Int})
    SubArray(parent, indexes, dims, 0, ())
end

# But when wrapping a LinearFast array with Strided indices, we can take advantage
abstract AbstractCartesianIndex{N} # Hack for a forward declaration
typealias StridedIndexTuple Tuple{Vararg{Union{RangeIndex, NoSlice, AbstractCartesianIndex}}}
function SubArray{P, I<:StridedIndexTuple, N}(::LinearFast, parent::P, indexes::I, dims::NTuple{N, Int})
    # Pre-compute the first index and stride so we don't need to lookup the indices
    SubArray(parent, indexes, dims, compute_first_index(parent, indexes), substrides(parent, indexes))
end
function SubArray{P, I, N}(::LinearFast, parent::P, indexes::I, dims::NTuple{N, Int})
    SubArray(parent, indexes, dims, 0, ())
end

typealias ScalarIndex Union{Real, NoSlice, AbstractCartesianIndex}
typealias DroppedScalar Union{Real, AbstractCartesianIndex}

substrides(parent, I::Tuple) =
    (@_inline_meta; substrides(1, parent, 1, I))
substrides(s, parent, dim, ::Tuple{}) = ()
substrides(s, parent, dim, I::Tuple{Real, Vararg{Any}}) =
    (@_inline_meta; (substrides(s*size(parent, dim), parent, dim+1, tail(I))...))
substrides(s, parent, dim, I::Tuple{AbstractCartesianIndex, Vararg{Any}}) =
    (@_inline_meta; (substrides(s*prod(size(parent)[dim:dim+length(I[1])-1]), parent, dim+length(I[1]), tail(I))...))
substrides(s, parent, dim, I::Tuple{NoSlice, Vararg{Any}}) =
    (@_inline_meta; (s, substrides(s*size(parent, dim), parent, dim+1, tail(I))...))
substrides(s, parent, dim, I::Tuple{Colon, Vararg{Any}}) =
    (@_inline_meta; (s, substrides(s*size(parent, dim), parent, dim+1, tail(I))...))
substrides(s, parent, dim, I::Tuple{Range, Vararg{Any}}) =
    (@_inline_meta; (s*step(I[1]), substrides(s*size(parent, dim), parent, dim+1, tail(I))...))
# And we don't keep track of trailing scalar indices since they aren't needed
substrides(s, parent, dim, I::Tuple{Real, Vararg{ScalarIndex}}) = ()
substrides(s, parent, dim, I::Tuple{AbstractCartesianIndex, Vararg{ScalarIndex}}) = ()
substrides(s, parent, dim, I::Tuple{NoSlice, Vararg{ScalarIndex}}) = (s,)
substrides(s, parent, dim, I::Tuple{Colon, Vararg{ScalarIndex}}) = (s,)
substrides(s, parent, dim, I::Tuple{Range, Vararg{ScalarIndex}}) = (s*step(I[1]),)
# Colons are special: sections of repeated colons form contiguous regions that
# may be linearly indexed, but only if they are followed by scalars or one
# final UnitRange. We cannot generally express this with dispatch, but we can
# get a few common cases.
substrides(s, parent, dim, I::Tuple{Colon, Union{Colon, UnitRange}, Vararg{ScalarIndex}}) = (s,)
substrides(s, parent, dim, I::Tuple{Colon, Colon, Union{Colon, UnitRange}, Vararg{ScalarIndex}}) = (s,)
substrides(s, parent, dim, I::Tuple{Colon, Colon, Colon, Union{Colon, UnitRange}, Vararg{ScalarIndex}}) = (s,)
# Anything else is an error
substrides(s, parent, dim, I::Tuple{Any, Vararg{Any}}) = throw(ArgumentError("strides is invalid for SubArrays with indices of type $(typeof(I[1]))"))

typealias StridedArray{T,N,A<:DenseArray,I<:StridedIndexTuple} Union{DenseArray{T,N}, SubArray{T,N,A,I}}
typealias StridedVector{T,A<:DenseArray,I<:StridedIndexTuple}  Union{DenseArray{T,1}, SubArray{T,1,A,I}}
typealias StridedMatrix{T,A<:DenseArray,I<:StridedIndexTuple} Union{DenseArray{T,2}, SubArray{T,2,A,I}}
typealias StridedVecOrMat{T} Union{StridedVector{T}, StridedMatrix{T}}

# Simple utilities
size(V::SubArray) = V.dims
length(V::SubArray) = prod(V.dims)

similar(V::SubArray, T, dims::Dims) = similar(V.parent, T, dims)

parent(V::SubArray) = V.parent
parentindexes(V::SubArray) = V.indexes

parent(a::AbstractArray) = a
parentindexes(a::AbstractArray) = ntuple(i->1:size(a,i), ndims(a))

## SubArray creation
# Drops singleton dimensions (those indexed with a scalar)
function slice(A::AbstractArray, I...)
    @_inline_meta
    @boundscheck checkbounds(A, I...)
    J = to_indexes(I...)
    SubArray(A, J, index_shape(A, J...))
end

keep_leading_scalars(T::Tuple{}) = T
keep_leading_scalars(T::Tuple{Real, Vararg{Real}}) = T
keep_leading_scalars(T::Tuple{Real, Vararg{Any}}) = (@_inline_meta; (NoSlice(T[1]), keep_leading_scalars(tail(T))...))
keep_leading_scalars(T::Tuple{Any, Vararg{Any}}) = (@_inline_meta; (T[1], keep_leading_scalars(tail(T))...))

function sub(A::AbstractArray, I...)
    @_inline_meta
    @boundscheck checkbounds(A, I...)
    J = keep_leading_scalars(to_indexes(I...))
    SubArray(A, J, index_shape(A, J...))
end

# Re-indexing is the heart of a view, transforming A[i, j][x, y] to A[i[x], j[y]]
#
# Recursively look through the heads of the parent- and sub-indexes, considering
# the following cases:
# * Parent index is empty  -> ignore trailing scalars, but preserve added dimensions
# * Parent index is Any    -> re-index that with the sub-index
# * Parent index is Scalar -> that dimension was dropped, so skip the sub-index and use the index as is
#
# Furthermore, we must specially consider the case with one final sub-index,
# as it may be a linear index that spans multiple parent indexes.

# When indexing beyond the parent indices, drop all trailing scalars (they must be 1 to be inbounds)
reindex(V, idxs::Tuple{}, subidxs::Tuple{Vararg{DroppedScalar}}) = ()
# Drop any intervening scalars that are beyond the parent indices but before a nonscalar
reindex(V, idxs::Tuple{}, subidxs::Tuple{DroppedScalar, Vararg{Any}}) =
    (@_propagate_inbounds_meta; (reindex(V, idxs, tail(subidxs))...))
# And keep the nonscalar index to add the dimension
reindex(V, idxs::Tuple{}, subidxs::Tuple{Any, Vararg{Any}}) =
    (@_propagate_inbounds_meta; (subidxs[1], reindex(V, idxs, tail(subidxs))...))

reindex(V, idxs::Tuple{Any}, subidxs::Tuple{Any}) =
    (@_propagate_inbounds_meta; (idxs[1][subidxs[1]],))
reindex(V, idxs::Tuple{Any}, subidxs::Tuple{Any, Any, Vararg{Any}}) =
    (@_propagate_inbounds_meta; (idxs[1][subidxs[1]],))
reindex(V, idxs::Tuple{Any, Any, Vararg{Any}}, subidxs::Tuple{Any}) =
    (@_propagate_inbounds_meta; (merge_indexes(V, idxs, subidxs[1]),))
# As an optimization, we don't need to merge indices if all trailing indices are dropped scalars
reindex(V, idxs::Tuple{Any, DroppedScalar, Vararg{DroppedScalar}}, subidxs::Tuple{Any}) =
    (@_propagate_inbounds_meta; (idxs[1][subidxs[1]], tail(idxs)...))
reindex(V, idxs::Tuple{Any, Any, Vararg{Any}}, subidxs::Tuple{Any, Any, Vararg{Any}}) =
    (@_propagate_inbounds_meta; (idxs[1][subidxs[1]], reindex(V, tail(idxs), tail(subidxs))...))

reindex(V, idxs::Tuple{DroppedScalar}, subidxs::Tuple{Any}) = idxs
reindex(V, idxs::Tuple{DroppedScalar}, subidxs::Tuple{Any, Any, Vararg{Any}}) = idxs
reindex(V, idxs::Tuple{DroppedScalar, Any, Vararg{Any}}, subidxs::Tuple{Any}) =
    (@_propagate_inbounds_meta; (idxs[1], reindex(V, tail(idxs), subidxs)...))
reindex(V, idxs::Tuple{DroppedScalar, Any, Vararg{Any}}, subidxs::Tuple{Any, Any, Vararg{Any}}) =
    (@_propagate_inbounds_meta; (idxs[1], reindex(V, tail(idxs), subidxs)...))

# In general, we simply re-index the parent indices by the provided ones
typealias SlowSubArray{T,N,P,I} SubArray{T,N,P,I,0}
getindex(V::SlowSubArray) = (@_propagate_inbounds_meta; getindex(V, 1))
function getindex(V::SlowSubArray, I::Int...)
    @_inline_meta
    @boundscheck checkbounds(V, I...)
    @inbounds r = V.parent[reindex(V, V.indexes, I)...]
    r
end

typealias FastSubArray{T,N,P,I} SubArray{T,N,P,I,1}
getindex(V::FastSubArray) = (@_propagate_inbounds_meta; getindex(V, 1))
function getindex(V::FastSubArray, i::Int)
    @_inline_meta
    @boundscheck checkbounds(V, i)
    @inbounds r = V.parent[V.first_index + V.strides[1]*(i-1)]
    r
end
# We can avoid a multiplication if the first parent index is a Colon or UnitRange
typealias FastContiguousSubArray{T,N,P,I<:Tuple{Union{Colon, UnitRange}, Vararg{Any}}} SubArray{T,N,P,I,1}
function getindex(V::FastContiguousSubArray, i::Int)
    @_inline_meta
    @boundscheck checkbounds(V, i)
    @inbounds r = V.parent[V.first_index + i-1]
    r
end
# We need this because the ::ViewIndex... method otherwise obscures the Base fallback
function getindex(V::FastSubArray, I::Int...)
    @_inline_meta
    @boundscheck checkbounds(V, I...)
    @inbounds r = getindex(V, sub2ind(size(V), I...))
    r
end

# Now here we can get tricky: when 1 < LD < length(I), we can sometimes use linear indexing!
strides2ind(strides::Tuple, I::Tuple) = strides2ind(0, strides, I)
strides2ind(v, strides::Tuple, I::Tuple) = strides2ind(v + strides[1] * (I[1]-1), tail(strides), tail(I))
strides2ind(v, strides::Tuple, I::Tuple{}) = v
@generated function getindex{T,SN,P,SI,LD}(V::SubArray{T,SN,P,SI,LD}, I::Int...)
    N = length(I)
    if N < LD
        # Expand the last index into the appropriate number of indices
        Isplat = Expr[:(I[$d]) for d = 1:N-1]
        i = 0
        for d=N:LD
            push!(Isplat, :(s[$(i+=1)]))
        end
        sz = Expr(:tuple)
        sz.args = Expr[:(size(V, $d)) for d=N:LD]
        return quote
            # ind2sub requires all dimensions to be > 0:
            @_inline_meta
            @boundscheck checkbounds(V, I...)
            s = ind2sub($sz, to_index(I[$N]))
            @inbounds r = getindex(V, $(Isplat...))
            r
        end
    elseif N > LD
        # Reduce the trailing indices to one linear index
        # Expand the last index into the appropriate number of indices
        Isplat = Expr[:(I[$d]) for d = 1:LD-1]
        Osplat = Expr[:(I[$d]) for d = LD:N]
        sz = Expr(:tuple)
        sz.args = Expr[:(size(V, $d)) for d=LD:N-1]
        push!(sz.args, :(trailingsize(V, Val{$N})))
        return quote
            # ind2sub requires all dimensions to be > 0:
            @_inline_meta
            @boundscheck checkbounds(V, I...)
            s = sub2ind($sz, to_indexes($(Osplat...))...)
            @inbounds r = getindex(V, $(Isplat...), s)
            r
        end
    else
        return quote
            @_inline_meta
            @boundscheck checkbounds(V, I...)
            @inbounds r = V.parent[V.first_index + strides2ind(V.strides, I)]
            r
        end
    end
end

getindex{T,N}(V::SubArray{T,N}, i::ViewIndex, I::ViewIndex...) = (@_propagate_inbounds_meta; copy(slice(V, i, I...)))

setindex!(V::SubArray, x) = (@_propagate_inbounds_meta; setindex!(V, x, 1))
function setindex!{T,N}(V::SubArray{T,N}, x, I::Int...)
    @_inline_meta
    @boundscheck checkbounds(V, I...)
    @inbounds V.parent[reindex(V, V.indexes, I)...] = x
    V
end
# TODO: this could be optimized for non-slow SubArrays

# Nonscalar setindex! falls back to the defaults

function slice{T,N}(V::SubArray{T,N}, I::ViewIndex...)
    @_inline_meta
    @boundscheck checkbounds(V, I...)
    idxs = reindex(V, V.indexes, to_indexes(I...))
    SubArray(V.parent, idxs, index_shape(V.parent, idxs...))
end

function sub{T,N}(V::SubArray{T,N}, I::ViewIndex...)
    @_inline_meta
    @boundscheck checkbounds(V, I...)
    idxs = reindex(V, V.indexes, keep_leading_scalars(to_indexes(I...)))
    SubArray(V.parent, idxs, index_shape(V.parent, idxs...))
end

linearindexing(A::FastSubArray) = LinearFast()
linearindexing(A::SubArray) = LinearSlow()

getindex(::Colon, i) = to_index(i)
unsafe_getindex(::Colon, i) = to_index(i)

step(::Colon) = 1
first(::Colon) = 1
isempty(::Colon) = false
in(::Int, ::Colon) = true

# Strides are the distance between adjacent elements in a given dimension
strides{T,N,P,I<:StridedIndexTuple}(V::SubArray{T,N,P,I}) = V.strides
strides(V::SubArray) = throw(ArgumentError("strides is only valide for strided SubArrays"))

stride(V::SubArray, d::Integer) = d <= ndims(V) ? strides(V)[d] : strides(V)[end] * size(V)[end]

iscontiguous(A::SubArray) = iscontiguous(typeof(A))
iscontiguous{S<:SubArray}(::Type{S}) = false
iscontiguous{F<:FastContiguousSubArray}(::Type{F}) = true

# Fast linear SubArrays have their first index cached
first_index(V::FastSubArray) = V.first_index
first_index(V::SubArray) = first_index(V.parent, V.indexes)
function first_index(P::AbstractArray, indexes::Tuple)
    f = 1
    s = 1
    for i = 1:length(indexes)
        f += (first(indexes[i])-1)*s
        s *= size(P, i)
    end
    f
end

# Computing the first index simply steps through the indices, accumulating the
# sum of index each multiplied by the parent's stride.
# The running sum is `f`; the cumulative stride product is `s`.
compute_first_index(parent, I::Tuple) = compute_first_index(1, 1, parent, 1, I)
compute_first_index(f, s, parent, dim, I::Tuple{Real, Vararg{Any}}) =
    (@_inline_meta; compute_first_index(f + (I[1]-1)*s, s*size(parent, dim), parent, dim+1, tail(I)))
compute_first_index(f, s, parent, dim, I::Tuple{NoSlice, Vararg{Any}}) =
    (@_inline_meta; compute_first_index(f + (I[1].i-1)*s, s*size(parent, dim), parent, dim+1, tail(I)))
# Just splat out the cartesian indices and continue
compute_first_index(f, s, parent, dim, I::Tuple{AbstractCartesianIndex, Vararg{Any}}) =
    (@_inline_meta; compute_first_index(f, s, parent, dim, (I[1].I..., tail(I)...)))
compute_first_index(f, s, parent, dim, I::Tuple{Colon, Vararg{Any}}) =
    (@_inline_meta; compute_first_index(f, s*size(parent, dim), parent, dim+1, tail(I)))
compute_first_index(f, s, parent, dim, I::Tuple{Any, Vararg{Any}}) =
    (@_inline_meta; compute_first_index(f + (first(I[1])-1)*s, s*size(parent, dim), parent, dim+1, tail(I)))
compute_first_index(f, s, parent, dim, I::Tuple{}) = f


unsafe_convert{T,N,P<:Array,I<:Tuple{Vararg{Union{RangeIndex, NoSlice}}}}(::Type{Ptr{T}}, V::SubArray{T,N,P,I}) =
    pointer(V.parent) + (first_index(V)-1)*sizeof(T)

unsafe_convert{T,N,P<:Array,I<:Tuple{Vararg{Union{RangeIndex, NoSlice}}}}(::Type{Ptr{Void}}, V::SubArray{T,N,P,I}) =
    convert(Ptr{Void}, unsafe_convert(Ptr{T}, V))

pointer(V::FastSubArray, i::Int) = pointer(V.parent, V.first_index + V.strides[1]*(i-1))
pointer(V::FastContiguousSubArray, i::Int) = pointer(V.parent, V.first_index + i-1)
pointer(V::SubArray, i::Int) = pointer(V, ind2sub(size(V), i))

function pointer{T,N,P<:Array,I<:Tuple{Vararg{Union{RangeIndex, NoSlice}}}}(V::SubArray{T,N,P,I}, is::Tuple{Vararg{Int}})
    index = first_index(V)
    strds = strides(V)
    for d = 1:length(is)
        index += (is[d]-1)*strds[d]
    end
    return pointer(V.parent, index)
end

## Convert
convert{T,S,N}(::Type{Array{T,N}}, V::SubArray{S,N}) = copy!(Array(T, size(V)), V)


## Compatability
# deprecate?
function parentdims(s::SubArray)
    nd = ndims(s)
    dimindex = Array(Int, nd)
    sp = strides(s.parent)
    sv = strides(s)
    j = 1
    for i = 1:ndims(s.parent)
        r = s.indexes[i]
        if j <= nd && (isa(r,Union{Colon,Range}) ? sp[i]*step(r) : sp[i]) == sv[j]
            dimindex[j] = i
            j += 1
        end
    end
    dimindex
end
