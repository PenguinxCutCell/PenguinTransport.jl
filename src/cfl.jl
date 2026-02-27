@inline function _min_spacing(x::AbstractVector{T}) where {T}
    n = length(x)
    n >= 2 || return one(T)
    dxmin = typemax(T)
    @inbounds for i in 2:n
        dx = abs(x[i] - x[i - 1])
        dxmin = dx < dxmin ? dx : dxmin
    end
    return dxmin
end

@inline function _velocity_max_tuple(u::NTuple{N,<:AbstractVector}) where {N}
    return ntuple(d -> maximum(abs, u[d]; init=zero(eltype(u[d]))), N)
end

"""
    cfl_dt(sys, u; cfl=0.5, p=nothing, t=0, include_diffusion=true)

Estimate a stable explicit timestep from advective and optional diffusive limits.

- Advection: `dt <= min_d(Δx_d / max|u_d|)`.
- Diffusion (if `include_diffusion && kappa > 0`):
  `dt <= 1 / (2κ * Σ_d 1/Δx_d^2)`.

Returns `cfl * min(dt_adv, dt_diff)`, or `Inf` if no active restriction exists.
"""
function cfl_dt(
    sys::TransportSystem{N,T},
    u;
    cfl::Real=0.5,
    p=nothing,
    t::Real=zero(T),
    include_diffusion::Bool=true,
) where {N,T}
    cfl > 0 || throw(ArgumentError("cfl must be > 0"))

    # Keep velocity caches synchronized with runtime-updated fields.
    _refresh_velocity_full!(sys, u, p, t)

    dx = ntuple(d -> _min_spacing(sys.moments.xyz[d]), N)
    umax = _velocity_max_tuple(sys.uω_full)

    dt_adv = T(Inf)
    @inbounds for d in 1:N
        if umax[d] > zero(T)
            dt_adv = min(dt_adv, dx[d] / umax[d])
        end
    end

    dt_diff = T(Inf)
    if include_diffusion && sys.kappa > zero(T)
        invdx2_sum = zero(T)
        @inbounds for d in 1:N
            invdx2_sum += inv(dx[d]^2)
        end
        if invdx2_sum > zero(T)
            dt_diff = inv(T(2) * sys.kappa * invdx2_sum)
        end
    end

    dt_raw = min(dt_adv, dt_diff)
    return isfinite(dt_raw) ? convert(T, cfl) * dt_raw : T(Inf)
end
