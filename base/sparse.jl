# This file is a part of Julia. License is MIT: http://julialang.org/license

module SparseMatrix

using Base: Func, AddFun, OrFun, ConjFun, IdFun
using Base.Sort: Forward
using Base.LinAlg: AbstractTriangular

importall Base
importall Base.Operators
importall Base.LinAlg
import Base.promote_eltype
import Base.@get!
import Base.Broadcast.eltype_plus, Base.Broadcast.broadcast_shape

export AbstractSparseArray, AbstractSparseMatrix, AbstractSparseVector, SparseMatrixCSC,
       SparseVector,
       blkdiag, dense, droptol!, dropzeros!, etree, issparse, nonzeros, nzrange,
       rowvals, sparse, sparsevec, spdiagm, speye, spones, sprand, sprandbool, sprandn,
       spzeros, symperm, nnz

include("sparse/abstractsparse.jl")
include("sparse/sparsematrix.jl")
include("sparse/sparsevector.jl")
include("sparse/csparse.jl")

include("sparse/linalg.jl")
if Base.USE_GPL_LIBS
    include("sparse/umfpack.jl")
    include("sparse/cholmod.jl")
    include("sparse/spqr.jl")
end

doc"""
    full(S::AbstractSparseArray)

Convert a sparse matrix `S` into a dense matrix.
"""
full(::AbstractSparseArray)

doc"""
    ldltfact(::Union{SparseMatrixCSC,Symmetric{Float64,SparseMatrixCSC{Flaot64,SuiteSparse_long}},Hermitian{Complex{Float64},SparseMatrixCSC{Complex{Float64},SuiteSparse_long}}}; shift=0, perm=Int[]) -> CHOLMOD.Factor

Compute the `LDLt` factorization of a sparse symmetric or Hermitian matrix. A fill-reducing permutation is used. `F = ldltfact(A)` is most frequently used to solve systems of equations `A*x = b` with `F\b`, but also the methods `diag`, `det`, `logdet` are defined for `F`. You can also extract individual factors from `F`, using `F[:L]`. However, since pivoting is on by default, the factorization is internally represented as `A == P'*L*D*L'*P` with a permutation matrix `P`; using just `L` without accounting for `P` will give incorrect answers. To include the effects of permutation, it's typically preferable to extact "combined" factors like `PtL = F[:PtL]` (the equivalent of `P'*L`) and `LtP = F[:UP]` (the equivalent of `L'*P`). The complete list of supported factors is `:L, :PtL, :D, :UP, :U, :LD, :DU, :PtLD, :DUP`.

Setting optional `shift` keyword argument computes the factorization of `A+shift*I` instead of `A`. If the `perm` argument is nonempty, it should be a permutation of `1:size(A,1)` giving the ordering to use (instead of CHOLMOD's default AMD ordering).

The function calls the C library CHOLMOD and many other functions from the library are wrapped but not exported.
"""
ldltfact(A::SparseMatrixCSC; shift=0, perm=Int[])

doc"""
    symperm(A, p)

Return the symmetric permutation of `A`, which is `A[p,p]`. `A` should be symmetric and sparse, where only the upper triangular part of the matrix is stored. This algorithm ignores the lower triangular part of the matrix. Only the upper triangular part of the result is returned as well.
"""
symperm

doc"""
    rowvals(A)

Return a vector of the row indices of `A`, and any modifications to the returned vector will mutate `A` as well. Given the internal storage format of sparse matrices, providing access to how the row indices are stored internally can be useful in conjuction with iterating over structural nonzero values. See `nonzeros(A)` and `nzrange(A, col)`.
"""
rowvals

end # module SparseMatrix
