# Performance testing

import Base: unsafe_getindex
# @inline unsafe_getindex(xs...) = Base.getindex(xs...)

function sumelt(A, n)
    s = zero(eltype(A)) + zero(eltype(A))
    for k = 1:n
        for a in A
            s += a
        end
    end
    s
end

function sumeach(A, n)
    s = zero(eltype(A)) + zero(eltype(A))
    for k = 1:n
        for I in eachindex(A)
            val = unsafe_getindex(A, I)
            s += val
        end
    end
    s
end

function sumlinear(A, n)
    s = zero(eltype(A)) + zero(eltype(A))
    for k = 1:n
        for I in 1:length(A)
            val = unsafe_getindex(A, I)
            s += val
        end
    end
    s
end
function sumcartesian(A, n)
    s = zero(eltype(A)) + zero(eltype(A))
    for k = 1:n
        for I in CartesianRange(size(A))
            val = unsafe_getindex(A, I)
            s += val
        end
    end
    s
end

function sumcolon(A, n)
    s = zero(eltype(A)) + zero(eltype(A))
    nrows = size(A, 1)
    ncols = size(A, 2)
    c = Colon()
    for k = 1:n
        @simd for i = 1:ncols
            val = unsafe_getindex(A, c, i)
            s += first(val)
        end
    end
    s
end

function sumrange(A, n)
    s = zero(eltype(A)) + zero(eltype(A))
    nrows = size(A, 1)
    ncols = size(A, 2)
    r = 1:nrows
    for k = 1:n
        @simd for i = 1:ncols
            val = unsafe_getindex(A, r, i)
            s += first(val)
        end
    end
    s
end

function sumlogical(A, n)
    s = zero(eltype(A)) + zero(eltype(A))
    nrows = size(A, 1)
    ncols = size(A, 2)
    r = falses(nrows)
    r[1:4:end] = true
    for k = 1:n
        @simd for i = 1:ncols
            val = unsafe_getindex(A, r, i)
            s += first(val)
        end
    end
    s
end

function sumvector(A, n)
    s = zero(eltype(A)) + zero(eltype(A))
    nrows = size(A, 1)
    ncols = size(A, 2)
    r = rand(1:nrows, 5)
    for k = 1:n
        @simd for i = 1:ncols
            val = unsafe_getindex(A, r, i)
            s += first(val)
        end
    end
    s
end


if !applicable(unsafe_getindex, [1 2], 1:1, 2)
    @inline Base.unsafe_getindex(A::BitArray, I1::BitArray, I2::Int) = unsafe_getindex(A, Base.to_index(I1), I2)
end

function makearrays{T}(::Type{T}, sz)
    Bit = trues(sz)
    (Bit,)
end

