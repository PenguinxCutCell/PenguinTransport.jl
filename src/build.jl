function padded_mask(dims::NTuple{N,Int}) where {N}
    nd = prod(dims)
    mask = falses(nd)
    li = LinearIndices(dims)
    @inbounds for I in CartesianIndices(dims)
        if any(d -> I[d] == dims[d], 1:N)
            mask[li[I]] = true
        end
    end
    return mask
end

function _stencil_bc_from_adv(bc_adv::CartesianOperators.AdvBoxBC{N,T}) where {N,T}
    z = zero(T)
    lo = ntuple(d -> (bc_adv.lo[d] isa CartesianOperators.AdvPeriodic ? CartesianOperators.Periodic{T}() : CartesianOperators.Neumann{T}(z)), N)
    hi = ntuple(d -> (bc_adv.hi[d] isa CartesianOperators.AdvPeriodic ? CartesianOperators.Periodic{T}() : CartesianOperators.Neumann{T}(z)), N)
    return CartesianOperators.BoxBC(lo, hi)
end

function _build_ops_adv(moments::CartesianGeometry.GeometricMoments{N,T}, bc_adv_input) where {N,T}
    adv_tmp = CartesianOperators.kernel_convection_ops(moments; bc_adv=bc_adv_input)
    bc_adv = adv_tmp.bc_adv
    bc_stencil = _stencil_bc_from_adv(bc_adv)
    ops_base = CartesianOperators.kernel_ops(moments; bc=bc_stencil)
    ops_adv = CartesianOperators.KernelConvectionOps{N,T}(
        ops_base.A,
        ops_base.B,
        ops_base.dims,
        ops_base.Nd,
        bc_adv,
    )
    return ops_adv, bc_adv
end

"""
    build_system(moments, prob; vtol=nothing)

Build a [`TransportSystem`](@ref) from geometric moments and a [`TransportProblem`](@ref).

Active `ω` DOFs are selected using:
- material cells (`cell_type != 0`),
- positive volume above `vtol`,
- removal of padded boundary cells.

Returns a system with reduced DOF map, diagonal mass matrix, kernel operators,
and preallocated full/reduced work buffers.
"""
function build_system(
    moments::CartesianGeometry.GeometricMoments{N,T},
    prob::TransportProblem;
    vtol::Union{Nothing,Real}=nothing,
) where {N,T}
    dims = ntuple(d -> length(moments.xyz[d]), N)
    Nd = prod(dims)

    V = T.(moments.V)
    maxV = maximum(abs, V; init=zero(T))
    vtol_local = vtol === nothing ? sqrt(eps(T)) * maxV : convert(T, vtol)

    omega_material_mask = falses(Nd)
    @inbounds for i in eachindex(V)
        omega_material_mask[i] = (moments.cell_type[i] != 0) && (V[i] > vtol_local)
    end

    omega_mask = omega_material_mask .& .!padded_mask(dims)
    omega_active = findall(omega_mask)
    isempty(omega_active) && throw(ArgumentError("no active omega DOFs after masking (cell_type/V/padding)"))
    dof_omega = PenguinSolverCore.DofMap(omega_active)

    Iγ = T.(moments.interface_measure)
    maxIγ = maximum(abs, Iγ; init=zero(T))
    igamma_tol = sqrt(eps(T)) * maxIγ
    gamma_mask = omega_mask .& (Iγ .> igamma_tol)
    gamma_active = findall(gamma_mask)
    dof_gamma = PenguinSolverCore.DofMap(gamma_active)

    M = spdiagm(0 => V[dof_omega.indices])

    ops_diff = CartesianOperators.kernel_ops(moments; bc=prob.bc_diff)
    ops_adv, bc_adv = _build_ops_adv(moments, prob.bc_adv)
    bc_diff = ops_diff.bc
    embedded_bc = _normalize_embedded_inflow(prob.embedded_inflow, moments)

    prob.kappa isa Number || throw(ArgumentError("kappa must be a scalar diffusivity; got $(typeof(prob.kappa))"))
    kappa = convert(T, prob.kappa)

    prob.scheme isa CartesianOperators.AdvectionScheme ||
        throw(ArgumentError("scheme must be a CartesianOperators.AdvectionScheme; got $(typeof(prob.scheme))"))

    work_diff = CartesianOperators.KernelWork(ops_diff)
    work_adv = CartesianOperators.KernelWork(ops_adv)

    Tω_full = zeros(T, Nd)
    Tγ_full = zeros(T, Nd)
    du_full = zeros(T, Nd)
    src_reduced = zeros(T, length(dof_omega.indices))
    uω_full = ntuple(_ -> zeros(T, Nd), N)
    uγ_full = ntuple(_ -> zeros(T, Nd), N)

    sys = TransportSystem{N,T}(
        PenguinSolverCore.InvalidationCache(),
        PenguinSolverCore.UpdateManager(),
        moments,
        dof_omega,
        dof_gamma,
        M,
        kappa,
        prob.scheme,
        bc_diff,
        bc_adv,
        embedded_bc,
        _adv_periodic_pattern(bc_adv),
        ops_diff,
        ops_adv,
        work_diff,
        work_adv,
        Tω_full,
        Tγ_full,
        du_full,
        src_reduced,
        uω_full,
        uγ_full,
        nothing,
        nothing,
        _normalize_sourcefun(prob.source),
        nothing,
        0,
        false,
    )

    _set_velocity_spec!(sys, prob.vel_omega; which=:omega)
    _set_velocity_spec!(sys, prob.vel_gamma; which=:gamma)

    return sys
end
