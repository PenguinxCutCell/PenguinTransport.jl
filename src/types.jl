"""
    TransportProblem(; kappa=0, bc_diff=nothing, bc_adv=nothing, scheme=Centered(),
                       vel_omega=0, vel_gamma=vel_omega, source=nothing, embedded_inflow=nothing)

User-facing specification for scalar transport:

`M*Ṫ = C(uω, uγ; scheme)T + κΔ(Tω, Tγ) + S`.

- `kappa`: scalar diffusivity (`0` gives pure advection).
- `bc_diff`: diffusion boundary conditions (`BoxBC`); defaults to zero-Neumann.
- `bc_adv`: advection boundary conditions (`AdvBoxBC`); defaults to outflow.
- `scheme`: advection discretization (`Centered`, `Upwind1`, `MUSCL`).
- `vel_omega`, `vel_gamma`: velocity payloads (scalar/tuple/vector/function).
- `source`: optional source callback/payload.
- `embedded_inflow`: optional embedded-interface inflow trace (scalar/vector/callback or
  `CartesianOperators.AbstractEmbeddedAdvBC`), imposed only where `u⋅n < 0` on cut cells.
"""
struct TransportProblem{K,BCD,BCA,SCH,VO,VG,SRC,EMB}
    kappa::K
    bc_diff::BCD
    bc_adv::BCA
    scheme::SCH
    vel_omega::VO
    vel_gamma::VG
    source::SRC
    embedded_inflow::EMB
end

TransportProblem(kappa, bc_diff, bc_adv, scheme, vel_omega, vel_gamma) =
    TransportProblem(kappa, bc_diff, bc_adv, scheme, vel_omega, vel_gamma, nothing, nothing)

TransportProblem(kappa, bc_diff, bc_adv, scheme, vel_omega) =
    TransportProblem(kappa, bc_diff, bc_adv, scheme, vel_omega, vel_omega, nothing, nothing)

TransportProblem(; kappa=0.0,
    bc_diff=nothing,
    bc_adv=nothing,
    scheme=CartesianOperators.Centered(),
    vel_omega=0.0,
    vel_gamma=vel_omega,
    source=nothing,
    embedded_inflow=nothing,
) = TransportProblem(kappa, bc_diff, bc_adv, scheme, vel_omega, vel_gamma, source, embedded_inflow)

"""
    TransportSystem{N,T} <: PenguinSolverCore.AbstractSystem

Semidiscrete transport system built by [`build_system`](@ref).

The runtime state is reduced to active `ω` DOFs while full-length caches are kept
internally for kernel operators and boundary-condition handling.
"""
mutable struct TransportSystem{N,T} <: PenguinSolverCore.AbstractSystem
    cache::PenguinSolverCore.InvalidationCache
    updates::PenguinSolverCore.UpdateManager

    moments::CartesianGeometry.GeometricMoments{N,T}
    dof_omega::PenguinSolverCore.DofMap{Int}
    M::SparseMatrixCSC{T,Int}

    kappa::T
    scheme::CartesianOperators.AdvectionScheme
    bc_diff::CartesianOperators.BoxBC{N,T}
    bc_adv::CartesianOperators.AdvBoxBC{N,T}
    embedded_bc::CartesianOperators.AbstractEmbeddedAdvBC{T}
    adv_periodic::NTuple{N,Bool}

    ops_diff::CartesianOperators.KernelOps{N,T}
    ops_adv::CartesianOperators.KernelConvectionOps{N,T}
    work_diff::CartesianOperators.KernelWork{T}
    work_adv::CartesianOperators.KernelWork{T}

    Tω_full::Vector{T}
    Tγ_full::Vector{T}
    du_full::Vector{T}
    src_reduced::Vector{T}

    uω_full::NTuple{N,Vector{T}}
    uγ_full::NTuple{N,Vector{T}}

    uωfun::Union{Nothing,Function}
    uγfun::Union{Nothing,Function}
    sourcefun::Union{Nothing,Function}
    gammafun::Union{Nothing,Function}

    rebuild_calls::Int
    ops_dirty::Bool
end

@inline function _evaluate_callable(f, sys, u, p, t)
    if applicable(f, sys, u, p, t)
        return f(sys, u, p, t)
    elseif applicable(f, u, p, t)
        return f(u, p, t)
    elseif applicable(f, t)
        return f(t)
    elseif applicable(f)
        return f()
    end
    throw(ArgumentError("callable $(typeof(f)) must accept one of (sys, u, p, t), (u, p, t), (t), or ()"))
end

@inline _is_adv_periodic(bc::CartesianOperators.AbstractAdvBC) = bc isa CartesianOperators.AdvPeriodic

function _adv_periodic_pattern(bc_adv::CartesianOperators.AdvBoxBC{N}) where {N}
    return ntuple(d -> _is_adv_periodic(bc_adv.lo[d]) && _is_adv_periodic(bc_adv.hi[d]), N)
end

function _normalize_sourcefun(source)
    if source === nothing
        return nothing
    elseif source isa Function
        return source
    end

    fixed = source
    return (_sys, _u, _p, _t) -> fixed
end

function _set_embedded_full!(
    dest::AbstractVector{T},
    value;
    name::AbstractString,
) where {T}
    Nd = length(dest)
    if value isa Number
        fill!(dest, convert(T, value))
        return dest
    elseif value isa AbstractVector
        length(value) == Nd || throw(DimensionMismatch("$name vector has length $(length(value)); expected $Nd"))
        @inbounds for i in 1:Nd
            dest[i] = convert(T, value[i])
        end
        return dest
    end
    throw(ArgumentError("unsupported $name type $(typeof(value)); expected scalar or full vector"))
end

function _normalize_embedded_inflow(spec, moments::CartesianGeometry.GeometricMoments{N,T}) where {N,T}
    if spec === nothing
        return CartesianOperators.NoEmbeddedAdvBC(T)
    elseif spec isa CartesianOperators.AbstractEmbeddedAdvBC
        return spec
    elseif spec isa Number
        return CartesianOperators.EmbeddedInflow(convert(T, spec))
    elseif spec isa Base.RefValue{<:Real}
        return CartesianOperators.EmbeddedInflow(Ref{T}(convert(T, spec[])))
    elseif spec isa AbstractVector
        return CartesianOperators.EmbeddedInflow(T.(spec))
    elseif spec isa Function
        wrapped = function (dest::AbstractVector{T}, moms::CartesianGeometry.GeometricMoments{N,T}, t::T)
            if applicable(spec, dest, moms, t)
                spec(dest, moms, t)
                return dest
            elseif applicable(spec, moms, t)
                payload = spec(moms, t)
                return _set_embedded_full!(dest, payload; name="embedded inflow callback")
            elseif applicable(spec, t)
                payload = spec(t)
                return _set_embedded_full!(dest, payload; name="embedded inflow callback")
            elseif applicable(spec)
                payload = spec()
                return _set_embedded_full!(dest, payload; name="embedded inflow callback")
            end
            throw(ArgumentError("embedded inflow callback must accept (dest,moments,t), (moments,t), (t), or ()"))
        end
        return CartesianOperators.EmbeddedInflow(zero(T); fun=wrapped)
    end
    throw(ArgumentError("unsupported embedded_inflow type $(typeof(spec))"))
end

function _is_tuple_of_numbers(value, N::Int)
    value isa Tuple || return false
    length(value) == N || return false
    @inbounds for d in 1:N
        value[d] isa Number || return false
    end
    return true
end

function _set_scalar_full_or_reduced!(
    dest::AbstractVector{T},
    sys::TransportSystem{N,T},
    value;
    name::AbstractString,
) where {N,T}
    Nd = length(dest)
    idx_omega = sys.dof_omega.indices

    if value isa Number
        fill!(dest, convert(T, value))
        return dest
    elseif value isa AbstractVector
        if length(value) == Nd
            @inbounds for i in 1:Nd
                dest[i] = convert(T, value[i])
            end
            return dest
        elseif length(value) == length(idx_omega)
            fill!(dest, zero(T))
            @inbounds for i in eachindex(idx_omega)
                dest[idx_omega[i]] = convert(T, value[i])
            end
            return dest
        end
        throw(DimensionMismatch("$name vector has length $(length(value)); expected $Nd (full) or $(length(idx_omega)) (omega-reduced)"))
    end

    throw(ArgumentError("unsupported $name type $(typeof(value)); expected scalar or vector"))
end

function _set_velocity_component!(
    dest::AbstractVector{T},
    idx_omega::AbstractVector{Int},
    value;
    name::AbstractString,
) where {T}
    Nd = length(dest)
    if value isa Number
        fill!(dest, convert(T, value))
        return dest
    elseif value isa AbstractVector
        if length(value) == Nd
            @inbounds for i in 1:Nd
                dest[i] = convert(T, value[i])
            end
            return dest
        elseif length(value) == length(idx_omega)
            fill!(dest, zero(T))
            @inbounds for i in eachindex(idx_omega)
                dest[idx_omega[i]] = convert(T, value[i])
            end
            return dest
        end
        throw(DimensionMismatch("$name vector has length $(length(value)); expected $Nd (full) or $(length(idx_omega)) (omega-reduced)"))
    end

    throw(ArgumentError("unsupported $name component type $(typeof(value)); expected scalar or vector"))
end

function _set_velocity_tuple!(
    dest::NTuple{N,Vector{T}},
    sys::TransportSystem{N,T},
    value;
    name::AbstractString,
) where {N,T}
    idx_omega = sys.dof_omega.indices

    if value isa Number
        N == 1 || throw(ArgumentError("$name scalar is only valid in 1D; got N=$N"))
        _set_velocity_component!(dest[1], idx_omega, value; name="$name[1]")
        return dest
    end

    if N == 1 && value isa AbstractVector
        _set_velocity_component!(dest[1], idx_omega, value; name="$name[1]")
        return dest
    end

    value isa Tuple || throw(ArgumentError("$name must be scalar, vector (1D), or $N-tuple"))
    length(value) == N || throw(DimensionMismatch("$name tuple has length $(length(value)); expected $N"))
    @inbounds for d in 1:N
        _set_velocity_component!(dest[d], idx_omega, value[d]; name="$name[$d]")
    end
    return dest
end

function _set_velocity_spec!(
    sys::TransportSystem{N,T},
    spec;
    which::Symbol,
) where {N,T}
    name = which === :omega ? "vel_omega" : "vel_gamma"

    if spec === nothing
        dest = which === :omega ? sys.uω_full : sys.uγ_full
        @inbounds for d in 1:N
            fill!(dest[d], zero(T))
        end
        if which === :omega
            sys.uωfun = nothing
        else
            sys.uγfun = nothing
        end
        return nothing
    end

    if spec isa Function
        if which === :omega
            sys.uωfun = spec
        else
            sys.uγfun = spec
        end
        return nothing
    end

    if spec isa Number
        N == 1 || throw(ArgumentError("$name scalar is only valid in 1D; got N=$N"))
        val = convert(T, spec)
        fun = (_sys, _u, _p, _t) -> val
        if which === :omega
            sys.uωfun = fun
        else
            sys.uγfun = fun
        end
        return nothing
    end

    if _is_tuple_of_numbers(spec, N)
        vals = ntuple(d -> convert(T, spec[d]), N)
        fun = (_sys, _u, _p, _t) -> vals
        if which === :omega
            sys.uωfun = fun
        else
            sys.uγfun = fun
        end
        return nothing
    end

    dest = which === :omega ? sys.uω_full : sys.uγ_full
    _set_velocity_tuple!(dest, sys, spec; name=name)
    if which === :omega
        sys.uωfun = nothing
    else
        sys.uγfun = nothing
    end
    return nothing
end
