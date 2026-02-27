function _steady_probe_state(sys::TransportSystem{N,T}, u_eval) where {N,T}
    n_omega = length(sys.dof_omega.indices)
    if u_eval === nothing
        return zeros(T, n_omega)
    end
    length(u_eval) == n_omega ||
        throw(DimensionMismatch("u_eval has length $(length(u_eval)); expected $n_omega"))
    return convert(Vector{T}, u_eval)
end

function _freeze_velocity_for_steady!(
    uω_frozen::NTuple{N,Vector{T}},
    uγ_frozen::NTuple{N,Vector{T}},
    sys::TransportSystem{N,T},
    u_probe::AbstractVector{T},
    p,
    t,
) where {N,T}
    if sys.uωfun === nothing
        @inbounds for d in 1:N
            copyto!(uω_frozen[d], sys.uω_full[d])
        end
    else
        payload = _evaluate_callable(sys.uωfun, sys, u_probe, p, t)
        _set_velocity_tuple!(uω_frozen, sys, payload; name="vel_omega")
    end

    if sys.uγfun === nothing
        @inbounds for d in 1:N
            copyto!(uγ_frozen[d], sys.uγ_full[d])
        end
    else
        payload = _evaluate_callable(sys.uγfun, sys, u_probe, p, t)
        _set_velocity_tuple!(uγ_frozen, sys, payload; name="vel_gamma")
    end
    return nothing
end

function _freeze_gamma_for_steady(
    sys::TransportSystem{N,T},
    u_probe::AbstractVector{T},
    p,
    t,
) where {N,T}
    sys.gammafun === nothing && return nothing

    gamma_frozen = zeros(T, sys.ops_diff.Nd)
    payload = _evaluate_callable(sys.gammafun, sys, u_probe, p, t)
    _set_scalar_full_or_reduced!(gamma_frozen, sys, payload; name="gamma field")
    return gamma_frozen
end

function _steady_transport_raw_reduced!(
    out_omega::AbstractVector{T},
    sys::TransportSystem{N,T},
    x_omega::AbstractVector{T},
    uω_frozen::NTuple{N,Vector{T}},
    uγ_frozen::NTuple{N,Vector{T}},
    gamma_frozen::Union{Nothing,Vector{T}},
) where {N,T}
    n_omega = length(sys.dof_omega.indices)
    length(out_omega) == n_omega ||
        throw(DimensionMismatch("out_omega has length $(length(out_omega)); expected $n_omega"))
    length(x_omega) == n_omega ||
        throw(DimensionMismatch("x_omega has length $(length(x_omega)); expected $n_omega"))

    fill!(sys.Tω_full, zero(T))
    PenguinSolverCore.prolong!(sys.Tω_full, x_omega, sys.dof_omega)

    if gamma_frozen === nothing
        copyto!(sys.Tγ_full, sys.Tω_full)
    else
        copyto!(sys.Tγ_full, gamma_frozen)
    end

    if iszero(sys.kappa)
        CartesianOperators.convection!(
            sys.du_full,
            sys.ops_adv,
            uω_frozen,
            uγ_frozen,
            sys.Tω_full,
            sys.Tγ_full,
            sys.work_adv;
            scheme=sys.scheme,
            moments=sys.moments,
            embedded_bc=sys.embedded_bc,
        )
    else
        adops = CartesianOperators.KernelAdvectionDiffusionOps(sys.ops_diff, sys.ops_adv, sys.kappa)
        CartesianOperators.advection_diffusion!(
            sys.du_full,
            adops,
            uω_frozen,
            uγ_frozen,
            sys.Tω_full,
            sys.Tγ_full,
            sys.work_diff,
            sys.work_adv;
            scheme=sys.scheme,
            moments=sys.moments,
            embedded_bc=sys.embedded_bc,
        )
    end

    idx_omega = sys.dof_omega.indices
    @inbounds for i in eachindex(idx_omega)
        out_omega[i] = sys.du_full[idx_omega[i]]
    end
    return out_omega
end

"""
    steady_linear_problem(sys; p=nothing, t=0, u0=nothing, u_eval=nothing)

Construct a matrix-free steady linear problem `A*x = b` using
`LinearSolve.FunctionOperator`.

Velocities and optional `γ` override are frozen at `(u_eval, p, t)`. The affine
transport/source contributions are shifted to the right-hand side.
"""
function steady_linear_problem(
    sys::TransportSystem{N,T};
    p=nothing,
    t::Real=zero(T),
    u0=nothing,
    u_eval=nothing,
) where {N,T}
    n_omega = length(sys.dof_omega.indices)
    Nd = sys.ops_diff.Nd

    u0_vec = u0 === nothing ? zeros(T, n_omega) : convert(Vector{T}, u0)
    length(u0_vec) == n_omega ||
        throw(DimensionMismatch("u0 has length $(length(u0_vec)); expected $n_omega"))

    u_probe = _steady_probe_state(sys, u_eval)
    uω_frozen = ntuple(_ -> zeros(T, Nd), N)
    uγ_frozen = ntuple(_ -> zeros(T, Nd), N)
    _freeze_velocity_for_steady!(uω_frozen, uγ_frozen, sys, u_probe, p, t)
    gamma_frozen = _freeze_gamma_for_steady(sys, u_probe, p, t)

    b_affine = zeros(T, n_omega)
    x_zero = zeros(T, n_omega)
    _steady_transport_raw_reduced!(b_affine, sys, x_zero, uω_frozen, uγ_frozen, gamma_frozen)

    rhs = similar(u0_vec)
    @inbounds for i in eachindex(rhs)
        rhs[i] = -b_affine[i]
    end

    if sys.sourcefun !== nothing
        src = _evaluate_callable(sys.sourcefun, sys, u_probe, p, t)
        src_reduced = zeros(T, n_omega)
        _source_to_reduced!(src_reduced, sys, src)
        @inbounds for i in eachindex(rhs)
            rhs[i] -= src_reduced[i]
        end
    end

    op! = function (out, x, _u, _p, _t)
        _steady_transport_raw_reduced!(out, sys, x, uω_frozen, uγ_frozen, gamma_frozen)
        @inbounds for i in eachindex(out)
            out[i] -= b_affine[i]
        end
        return out
    end

    Aop = LinearSolve.FunctionOperator(
        op!,
        u0_vec,
        similar(u0_vec);
        isinplace=true,
        T=T,
        isconstant=true,
    )

    return LinearSolve.LinearProblem(Aop, rhs; u0=u0_vec, p=p)
end

"""
    steady_solve(sys; alg=LinearSolve.SimpleGMRES(), p=nothing, t=0, u0=nothing, u_eval=nothing, kwargs...)

Solve the steady matrix-free transport problem produced by
[`steady_linear_problem`](@ref) with `LinearSolve.solve`.
"""
function steady_solve(
    sys::TransportSystem;
    alg=LinearSolve.SimpleGMRES(),
    p=nothing,
    t=0.0,
    u0=nothing,
    u_eval=nothing,
    kwargs...,
)
    prob = steady_linear_problem(sys; p=p, t=t, u0=u0, u_eval=u_eval)
    return LinearSolve.solve(prob, alg; kwargs...)
end
