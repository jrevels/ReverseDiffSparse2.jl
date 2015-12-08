

# reverse-mode evaluation of an expression tree

# assumes forward_storage is already updated
# dense gradient output, assumes initialized to zero
function reverse_eval{T}(output::Vector{T},rev_storage::Vector{T},forward_storage::Vector{T},nd::Vector{NodeData},adj,const_values)

    @assert length(rev_storage) >= length(nd)
    @assert length(forward_storage) >= length(nd)

    # nd is already in order such that parents always appear before children
    # so a forward pass through nd is a backwards pass through the tree

    children_arr = rowvals(adj)

    if nd[1].nodetype == VARIABLE
        output[nd[1].index] += 1
        return # trivial case
    end

    # reverse_storage[k] is the partial derivative of the output with respect to
    # the value of node k
    rev_storage[1] = 1

    for k in 2:length(nd)
        @inbounds nod = nd[k]
        (nod.nodetype == VALUE) && continue
        # compute the value of reverse_storage[k]
        parentidx = nod.parent
        @inbounds par = nd[parentidx]
        @inbounds parentval = rev_storage[parentidx]
        op = par.index
        if par.nodetype == CALL
            if op == 1 # :+
                @inbounds rev_storage[k] = parentval
            elseif op == 2 # :-
                @inbounds siblings_idx = nzrange(adj,parentidx)
                if nod.whichchild == 1
                    @inbounds rev_storage[k] = parentval
                else
                    @inbounds rev_storage[k] = -parentval
                end
            elseif op == 3 # :*
                # dummy version for now
                @inbounds siblings_idx = nzrange(adj,parentidx)
                n_siblings = length(siblings_idx)
                if n_siblings == 2
                    otheridx = ifelse(nod.whichchild == 1, last(siblings_idx),first(siblings_idx))
                    @inbounds prod_others = forward_storage[children_arr[otheridx]]
                    @inbounds rev_storage[k] = parentval*prod_others
                else
                    @inbounds parent_val = forward_storage[parentidx]
                    if parent_val == 0.0
                        # product of all other siblings
                        prod_others = one(T)
                        for r in 1:n_siblings
                            r == nod.whichchild && continue
                            sib_idx = first(siblings_idx) + r - 1
                            @inbounds prod_others *= forward_storage[children_arr[sib_idx]]
                            prod_others == 0.0 && break
                        end
                        @inbounds rev_storage[k] = parentval*prod_others
                    else
                        @inbounds rev_storage[k] = parentval*(parent_val/forward_storage[k])
                    end
                end
            elseif op == 4 # :^
                @inbounds siblings_idx = nzrange(adj,parentidx)
                if nod.whichchild == 1 # base
                    @inbounds exponentidx = children_arr[last(siblings_idx)]
                    @inbounds exponent = forward_storage[exponentidx]
                    if exponent == 2
                        @inbounds rev_storage[k] = parentval*2*forward_storage[k]
                    else
                        rev_storage[k] = parentval*exponent*forward_storage[k]^(exponent-1)
                    end
                else
                    baseidx = children_arr[first(siblings_idx)]
                    base = forward_storage[baseidx]
                    rev_storage[k] = parentval*forward_storage[parentidx]*log(base)
                end
            elseif op == 5 # :/
                @inbounds siblings_idx = nzrange(adj,parentidx)
                if nod.whichchild == 1 # numerator
                    @inbounds denomidx = children_arr[last(siblings_idx)]
                    @inbounds denom = forward_storage[denomidx]
                    @inbounds rev_storage[k] = parentval/denom
                else # denominator
                    @inbounds numeratoridx = children_arr[first(siblings_idx)]
                    @inbounds numerator = forward_storage[numeratoridx]
                    @inbounds rev_storage[k] = -parentval*numerator*forward_storage[k]^(-2)
                end
            else
                error()
            end
        else
            @assert par.nodetype == CALLUNIVAR
            @inbounds this_value = forward_storage[k]
            @inbounds rev_storage[k] = parentval*univariate_deriv(op,this_value)
        end

        if nod.nodetype == VARIABLE
            @inbounds output[nod.index] += rev_storage[k]
        end
    end
    #@show storage

    nothing

end

export reverse_eval

switchblock = Expr(:block)
for i = 1:length(univariate_operators)
    deriv_expr = univariate_operator_deriv[i]
	ex = :(return $deriv_expr::T)
    push!(switchblock.args,i,ex)
end
switchexpr = Expr(:macrocall, Expr(:.,:Lazy,quot(symbol("@switch"))), :operator_id,switchblock)

@eval @inline function univariate_deriv{T}(operator_id,x::T)
    $switchexpr
end

function hessmat_eval!{N,T}(R::Matrix{T},
                           rev_storage::Vector{GradNumTup{N,T}},
                           forward_storage::Vector{GradNumTup{N,T}},
                           nd::Vector{NodeData},
                           adj,
                           const_values,
                           x_values::Vector{T},
                           reverse_output_vector::Vector{GradNumTup{N,T}},
                           forward_input_vector::Vector{GradNumTup{N,T}},
                           local_to_global_idx::Vector{Int})
    nevals = div(size(R, 2), N)
    last_chunk = N*nevals
    G = GradNumTup{N,T}

    # perform all evaluations requiring the full chunk-size
    for k in 1:N:last_chunk
        for r in 1:length(local_to_global_idx)
            # set up directional derivatives
            idx = local_to_global_idx[r]
            forward_input_vector[idx] = G(x_values[idx], extract_partials(G, R, r, k))
            reverse_output_vector[idx] = zero(G)
        end

        # do a forward pass
        forward_eval(forward_storage, nd, adj, const_values, forward_input_vector)

        # do a reverse pass
        reverse_eval(reverse_output_vector, rev_storage, forward_storage, nd, adj, const_values)

        # collect directional derivatives
        for c in k:(k+N-1)
            for r in 1:length(local_to_global_idx)
                idx = local_to_global_idx[r]
                R[r,c] = grad(reverse_output_vector[idx], c)
            end
        end
    end

    # do one more evaluation with whatever's chunk-size is left over
    # leftover = size(R, 2) - last_chunk
end

# returns (R[r, k], R[r, k+1], ... R[r, k+N-1])
@generated function extract_partials{N,T}(::Type{GradNumTup{N,T}}, R, r, k)
    return Expr(:tuple, [:(R[r, k+$i]) for i in 0:(N-1)]...)
end

export hessmat_eval!
