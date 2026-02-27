"""
    KappaUpdater(f)

Updater that sets `sys.kappa = f(...)` at scheduled times.
Returns `:rhs_only`.
"""
mutable struct KappaUpdater{F} <: PenguinSolverCore.AbstractUpdater
    kappa_fun::F
end

"""
    SchemeUpdater(f)

Updater that sets `sys.scheme = f(...)` (`AdvectionScheme` expected).
Returns `:rhs_only`.
"""
mutable struct SchemeUpdater{F} <: PenguinSolverCore.AbstractUpdater
    scheme_fun::F
end

"""
    VelocityUpdater(f; which=:both)

Updater for transport velocities.

- `which=:omega`: update only `uω`.
- `which=:gamma`: update only `uγ`.
- `which=:both`: update both (payload may also be named tuple with `:omega/:gamma`).

Returns `:rhs_only`.
"""
mutable struct VelocityUpdater{F} <: PenguinSolverCore.AbstractUpdater
    velocity_fun::F
    which::Symbol
end

VelocityUpdater(f; which::Symbol=:both) = VelocityUpdater{typeof(f)}(f, which)

"""
    AdvBCUpdater(f)

Updater for advection boundary conditions (`AdvBoxBC` payload).

If periodicity changes, marks operators dirty and returns `:matrix` so rebuild is
triggered. Otherwise returns `:rhs_only`.
"""
mutable struct AdvBCUpdater{F} <: PenguinSolverCore.AbstractUpdater
    bc_fun::F
end

"""
    SourceUpdater(f)

Updater for source specification. Payload can be `nothing`, a callable, or a
fixed scalar/vector value (wrapped internally).
Returns `:rhs_only`.
"""
mutable struct SourceUpdater{F} <: PenguinSolverCore.AbstractUpdater
    source_fun::F
end

function _normalize_advbc_for_system(
    sys::TransportSystem{N,T},
    value,
) where {N,T}
    ops_adv, bc_adv = _build_ops_adv(sys.moments, value)
    return ops_adv, bc_adv
end

function PenguinSolverCore.update!(upd::KappaUpdater, sys::TransportSystem{N,T}, u, p, t) where {N,T}
    payload = _evaluate_callable(upd.kappa_fun, sys, u, p, t)
    payload isa Number || throw(ArgumentError("KappaUpdater must return a scalar diffusivity, got $(typeof(payload))"))
    sys.kappa = convert(T, payload)
    return :rhs_only
end

function PenguinSolverCore.update!(upd::SchemeUpdater, sys::TransportSystem, u, p, t)
    payload = _evaluate_callable(upd.scheme_fun, sys, u, p, t)
    payload isa CartesianOperators.AdvectionScheme ||
        throw(ArgumentError("SchemeUpdater must return an AdvectionScheme, got $(typeof(payload))"))
    sys.scheme = payload
    return :rhs_only
end

function _set_velocity_payload!(
    sys::TransportSystem,
    payload;
    which::Symbol,
)
    if which === :omega
        _set_velocity_spec!(sys, payload; which=:omega)
        return :rhs_only
    elseif which === :gamma
        _set_velocity_spec!(sys, payload; which=:gamma)
        return :rhs_only
    elseif which === :both
        if payload isa NamedTuple
            has_omega = haskey(payload, :omega)
            has_gamma = haskey(payload, :gamma)
            (has_omega || has_gamma) ||
                throw(ArgumentError("VelocityUpdater NamedTuple payload must include key :omega and/or :gamma"))
            has_omega && _set_velocity_spec!(sys, payload.omega; which=:omega)
            has_gamma && _set_velocity_spec!(sys, payload.gamma; which=:gamma)
            return :rhs_only
        end

        _set_velocity_spec!(sys, payload; which=:omega)
        _set_velocity_spec!(sys, payload; which=:gamma)
        return :rhs_only
    end

    throw(ArgumentError("VelocityUpdater `which` must be :omega, :gamma, or :both; got `$which`"))
end

function PenguinSolverCore.update!(upd::VelocityUpdater, sys::TransportSystem, u, p, t)
    payload = _evaluate_callable(upd.velocity_fun, sys, u, p, t)
    return _set_velocity_payload!(sys, payload; which=upd.which)
end

function PenguinSolverCore.update!(upd::AdvBCUpdater, sys::TransportSystem{N,T}, u, p, t) where {N,T}
    payload = _evaluate_callable(upd.bc_fun, sys, u, p, t)
    ops_adv_new, bc_adv_new = _normalize_advbc_for_system(sys, payload)

    old_pattern = sys.adv_periodic
    new_pattern = _adv_periodic_pattern(bc_adv_new)

    sys.ops_adv = ops_adv_new
    sys.bc_adv = bc_adv_new
    sys.adv_periodic = new_pattern

    if old_pattern != new_pattern
        sys.ops_dirty = true
        return :matrix
    end

    return :rhs_only
end

function PenguinSolverCore.update!(upd::SourceUpdater, sys::TransportSystem, u, p, t)
    payload = _evaluate_callable(upd.source_fun, sys, u, p, t)
    if payload === nothing || payload isa Function
        sys.sourcefun = payload
    else
        fixed = payload
        sys.sourcefun = (_sys, _u, _p, _t) -> fixed
    end
    return :rhs_only
end
