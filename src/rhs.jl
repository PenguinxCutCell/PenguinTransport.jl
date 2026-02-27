PenguinSolverCore.mass_matrix(sys::TransportSystem) = sys.M

function _source_to_reduced!(out::AbstractVector{T}, sys::TransportSystem{N,T}, src) where {N,T}
    idx_omega = sys.dof_omega.indices
    V = sys.moments.V
    # Convention: source callbacks provide physical source density, then we mass-weight by V.

    if src isa Number
        s = convert(T, src)
        @inbounds for i in eachindex(idx_omega)
            idx = idx_omega[i]
            out[i] = V[idx] * s
        end
        return out
    elseif src isa AbstractVector
        if length(src) == length(idx_omega)
            @inbounds for i in eachindex(idx_omega)
                idx = idx_omega[i]
                out[i] = V[idx] * convert(T, src[i])
            end
            return out
        elseif length(src) == sys.ops_diff.Nd
            @inbounds for i in eachindex(idx_omega)
                idx = idx_omega[i]
                out[i] = V[idx] * convert(T, src[idx])
            end
            return out
        end
        throw(DimensionMismatch("source vector has length $(length(src)); expected $(length(idx_omega)) (reduced) or $(sys.ops_diff.Nd) (full)"))
    end

    throw(ArgumentError("source callback must return scalar or vector, got $(typeof(src))"))
end

function _refresh_gamma_full!(sys::TransportSystem{N,T}, u, p, t) where {N,T}
    copyto!(sys.Tγ_full, sys.Tω_full)
    if sys.gammafun !== nothing
        payload = _evaluate_callable(sys.gammafun, sys, u, p, t)
        _set_scalar_full_or_reduced!(sys.Tγ_full, sys, payload; name="gamma field")
    end
    return sys.Tγ_full
end

function _refresh_velocity_full!(sys::TransportSystem{N,T}, u, p, t) where {N,T}
    if sys.uωfun !== nothing
        payload_omega = _evaluate_callable(sys.uωfun, sys, u, p, t)
        _set_velocity_tuple!(sys.uω_full, sys, payload_omega; name="vel_omega")
    end
    if sys.uγfun !== nothing
        payload_gamma = _evaluate_callable(sys.uγfun, sys, u, p, t)
        _set_velocity_tuple!(sys.uγ_full, sys, payload_gamma; name="vel_gamma")
    end
    return nothing
end

function PenguinSolverCore.rhs!(du, sys::TransportSystem{N,T}, u, p, t) where {N,T}
    nω = length(sys.dof_omega.indices)
    length(u) == nω || throw(DimensionMismatch("state length $(length(u)) does not match omega DOF count $nω"))
    length(du) == nω || throw(DimensionMismatch("du length $(length(du)) does not match omega DOF count $nω"))

    fill!(sys.Tω_full, zero(T))
    PenguinSolverCore.prolong!(sys.Tω_full, u, sys.dof_omega)

    _refresh_gamma_full!(sys, u, p, t)
    _refresh_velocity_full!(sys, u, p, t)

    if iszero(sys.kappa)
        CartesianOperators.convection!(
            sys.du_full,
            sys.ops_adv,
            sys.uω_full,
            sys.uγ_full,
            sys.Tω_full,
            sys.Tγ_full,
            sys.work_adv;
            scheme=sys.scheme,
            moments=sys.moments,
            embedded_bc=sys.embedded_bc,
            t=convert(T, t),
        )
    else
        adops = CartesianOperators.KernelAdvectionDiffusionOps(sys.ops_diff, sys.ops_adv, sys.kappa)
        CartesianOperators.advection_diffusion!(
            sys.du_full,
            adops,
            sys.uω_full,
            sys.uγ_full,
            sys.Tω_full,
            sys.Tγ_full,
            sys.work_diff,
            sys.work_adv;
            scheme=sys.scheme,
            moments=sys.moments,
            embedded_bc=sys.embedded_bc,
            t=convert(T, t),
        )
    end

    idx_omega = sys.dof_omega.indices
    @inbounds for i in eachindex(idx_omega)
        du[i] = sys.du_full[idx_omega[i]]
    end

    if sys.sourcefun !== nothing
        src = _evaluate_callable(sys.sourcefun, sys, u, p, t)
        _source_to_reduced!(sys.src_reduced, sys, src)
        @inbounds for i in eachindex(du)
            du[i] += sys.src_reduced[i]
        end
    end

    return du
end
