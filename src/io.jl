function full_state(sys::TransportSystem{N,T}, u_omega::AbstractVector) where {N,T}
    nω = length(sys.dof_omega.indices)
    length(u_omega) == nω ||
        throw(DimensionMismatch("reduced omega length $(length(u_omega)) does not match DOF map length $nω"))

    fill!(sys.Tω_full, zero(T))
    PenguinSolverCore.prolong!(sys.Tω_full, u_omega, sys.dof_omega)

    copyto!(sys.Tγ_full, sys.Tω_full)
    if sys.gammafun !== nothing
        payload = _evaluate_callable(sys.gammafun, sys, u_omega, nothing, zero(T))
        _set_scalar_full_or_reduced!(sys.Tγ_full, sys, payload; name="gamma field")
    end

    return copy(sys.Tω_full), copy(sys.Tγ_full)
end
