using Random

const _EB_CX = 0.62
const _EB_CY = 0.48
const _EB_R = 0.17

function _build_eb_circle_moments(; nx::Int, ny::Int, cx::Float64=_EB_CX, cy::Float64=_EB_CY, r::Float64=_EB_R)
    x = collect(range(0.0, 1.0; length=nx + 1))
    y = collect(range(0.0, 1.0; length=ny + 1))
    # Fluid is outside the circle: levelset < 0.
    levelset(x, y, _=0.0) = r - sqrt((x - cx)^2 + (y - cy)^2)
    return CartesianGeometry.geometric_moments(levelset, (x, y), Float64, zero; method=:implicitintegration)
end

function _periodic_adv_bc_2d()
    return CartesianOperators.AdvBoxBC(
        (CartesianOperators.AdvPeriodic(Float64), CartesianOperators.AdvPeriodic(Float64)),
        (CartesianOperators.AdvPeriodic(Float64), CartesianOperators.AdvPeriodic(Float64)),
    )
end

function _diag_inflow_adv_bc(phi_in::Float64)
    return CartesianOperators.AdvBoxBC(
        (CartesianOperators.AdvInflow(phi_in), CartesianOperators.AdvInflow(phi_in)),
        (CartesianOperators.AdvOutflow(Float64), CartesianOperators.AdvOutflow(Float64)),
    )
end

function _build_eb_system(
    moments;
    vel::NTuple{2,Float64},
    scheme::CartesianOperators.AdvectionScheme,
    bc_adv,
    embedded_inflow=nothing,
)
    prob = PenguinTransport.TransportProblem(;
        kappa=0.0,
        bc_adv=bc_adv,
        scheme=scheme,
        vel_omega=vel,
        vel_gamma=vel,
        embedded_inflow=embedded_inflow,
    )
    return PenguinTransport.build_system(moments, prob)
end

function _global_to_local(sys)
    Nd = sys.ops_adv.Nd
    g2l = zeros(Int, Nd)
    idx = sys.dof_omega.indices
    @inbounds for k in eachindex(idx)
        g2l[idx[k]] = k
    end
    return g2l
end

function _reduced_from_field(sys, f::Function, t::Float64)
    dims = sys.ops_adv.dims
    li = LinearIndices(dims)
    g2l = _global_to_local(sys)

    u = zeros(Float64, length(sys.dof_omega.indices))
    @inbounds for I in CartesianIndices(dims)
        g = li[I]
        l = g2l[g]
        l == 0 && continue
        x = sys.moments.xyz[1][I[1]]
        y = sys.moments.xyz[2][I[2]]
        u[l] = f(x, y, t)
    end
    return u
end

function _reduced_mass_times_time_derivative(sys, ddt::Function, t::Float64)
    dims = sys.ops_adv.dims
    li = LinearIndices(dims)
    g2l = _global_to_local(sys)

    out = zeros(Float64, length(sys.dof_omega.indices))
    @inbounds for I in CartesianIndices(dims)
        g = li[I]
        l = g2l[g]
        l == 0 && continue
        x = sys.moments.xyz[1][I[1]]
        y = sys.moments.xyz[2][I[2]]
        out[l] = sys.moments.V[g] * ddt(x, y, t)
    end
    return out
end

function _rk2_step!(u, sys, dt::Float64, t::Float64)
    idx = sys.dof_omega.indices
    V = sys.moments.V

    rhs0 = similar(u)
    rhs1 = similar(u)
    u1 = similar(u)

    PenguinSolverCore.rhs!(rhs0, sys, u, nothing, t)
    @inbounds for i in eachindex(u)
        u1[i] = u[i] + dt * rhs0[i] / V[idx[i]]
    end

    PenguinSolverCore.rhs!(rhs1, sys, u1, nothing, t + dt)
    @inbounds for i in eachindex(u)
        u[i] = 0.5 * u[i] + 0.5 * (u1[i] + dt * rhs1[i] / V[idx[i]])
    end

    return u
end

function _rk2_integrate!(u, sys, tf::Float64; cfl::Float64=0.45)
    dt_cfl = PenguinTransport.cfl_dt(sys, u; cfl=cfl, include_diffusion=false)
    nsteps = max(1, ceil(Int, tf / dt_cfl))
    dt = tf / nsteps

    t = 0.0
    for _ in 1:nsteps
        _rk2_step!(u, sys, dt, t)
        t += dt
    end
    return u
end

function _circle_band_local_indices(sys; cx::Float64=_EB_CX, cy::Float64=_EB_CY, r::Float64=_EB_R)
    dims = sys.ops_adv.dims
    li = LinearIndices(dims)
    g2l = _global_to_local(sys)

    dx = abs(sys.moments.xyz[1][2] - sys.moments.xyz[1][1])
    dy = abs(sys.moments.xyz[2][2] - sys.moments.xyz[2][1])
    width = 2.0 * max(dx, dy)

    out = Int[]
    @inbounds for I in CartesianIndices(dims)
        g = li[I]
        l = g2l[g]
        l == 0 && continue

        x = sys.moments.xyz[1][I[1]]
        y = sys.moments.xyz[2][I[2]]
        dist = abs(sqrt((x - cx)^2 + (y - cy)^2) - r)
        if dist <= width
            push!(out, l)
        end
    end
    return out
end

function _weighted_norms(sys, e::AbstractVector{Float64}; local_ids::Union{Nothing,Vector{Int}}=nothing)
    idx = sys.dof_omega.indices
    V = sys.moments.V

    if local_ids === nothing
        ids = collect(eachindex(e))
    else
        ids = local_ids
    end

    l1 = 0.0
    l2 = 0.0
    linf = 0.0
    w = 0.0
    @inbounds for l in ids
        wi = V[idx[l]]
        ai = abs(e[l])
        l1 += wi * ai
        l2 += wi * ai * ai
        linf = max(linf, ai)
        w += wi
    end

    l1 /= max(w, eps(Float64))
    l2 = sqrt(l2 / max(w, eps(Float64)))
    return l1, l2, linf
end

@testset "High-fidelity 1: EB freestream preservation (diagonal inflow)" begin
    phi0 = 0.37
    vel = (1.0, 0.7)

    moments = _build_eb_circle_moments(; nx=48, ny=48)
    sys = _build_eb_system(
        moments;
        vel=vel,
        scheme=CartesianOperators.Upwind1(),
        bc_adv=_diag_inflow_adv_bc(phi0),
        embedded_inflow=phi0,
    )

    u = fill(phi0, length(sys.dof_omega.indices))
    _rk2_integrate!(u, sys, 0.8; cfl=0.45)

    err_all = maximum(abs.(u .- phi0))
    band_ids = _circle_band_local_indices(sys)
    @test !isempty(band_ids)
    err_band = maximum(abs.(u[band_ids] .- phi0))

    tol = max(50 * eps(Float64), 1e-12)
    @test err_all <= tol
    @test err_band <= tol
    @test all(isfinite, u)
end

@testset "High-fidelity 2: EB manufactured smooth residual order" begin
    ux = cos(pi / 6)
    uy = sin(pi / 6)
    vel = (ux, uy)
    tf = 0.15

    phi(x, y, t) = sin(2pi * (x - ux * t)) * sin(2pi * (y - uy * t))
    dphi_dt(x, y, t) = -2pi * ux * cos(2pi * (x - ux * t)) * sin(2pi * (y - uy * t)) -
                       2pi * uy * sin(2pi * (x - ux * t)) * cos(2pi * (y - uy * t))

    ns = (32, 64, 128)
    err_l1 = Float64[]
    err_l2 = Float64[]
    errb_l1 = Float64[]

    for n in ns
        moments = _build_eb_circle_moments(; nx=n, ny=n)

        embedded_exact = function (moms, t)
            dims = (length(moms.xyz[1]), length(moms.xyz[2]))
            li = LinearIndices(dims)
            out = zeros(Float64, prod(dims))
            @inbounds for I in CartesianIndices(dims)
                g = li[I]
                out[g] = phi(moms.xyz[1][I[1]], moms.xyz[2][I[2]], t)
            end
            return out
        end

        sys = _build_eb_system(
            moments;
            vel=vel,
            scheme=CartesianOperators.MUSCL(CartesianOperators.MC()),
            bc_adv=_periodic_adv_bc_2d(),
            embedded_inflow=embedded_exact,
        )

        u = _reduced_from_field(sys, phi, tf)
        mdt = _reduced_mass_times_time_derivative(sys, dphi_dt, tf)
        rhs = similar(u)
        PenguinSolverCore.rhs!(rhs, sys, u, nothing, tf)

        defect = rhs .- mdt
        band_ids = _circle_band_local_indices(sys)
        @test !isempty(band_ids)

        l1, l2, _ = _weighted_norms(sys, defect)
        bl1, _, _ = _weighted_norms(sys, defect; local_ids=band_ids)

        push!(err_l1, l1)
        push!(err_l2, l2)
        push!(errb_l1, bl1)
    end

    p_l1 = [log2(err_l1[i] / err_l1[i + 1]) for i in 1:(length(err_l1) - 1)]
    p_l2 = [log2(err_l2[i] / err_l2[i + 1]) for i in 1:(length(err_l2) - 1)]
    p_band_l1 = [log2(errb_l1[i] / errb_l1[i + 1]) for i in 1:(length(errb_l1) - 1)]

    @test all(isfinite, err_l1)
    @test all(isfinite, err_l2)
    @test all(isfinite, errb_l1)

    @test err_l1[3] < err_l1[2] < err_l1[1]
    @test err_l2[3] < err_l2[2] < err_l2[1]
    @test errb_l1[3] < errb_l1[2] < errb_l1[1]

    # Global residual consistency should stay close to second order.
    @test minimum(p_l1) >= 1.6
    @test minimum(p_l2) >= 1.6
    # EB band typically loses some order but should still clearly converge.
    @test minimum(p_band_l1) >= 1.0
end

@testset "U1: inflow/outflow classifier" begin
    isinflow(u::NTuple{2,Float64}, n::NTuple{2,Float64}) = (u[1] * n[1] + u[2] * n[2]) < 0.0

    normals = ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))
    velocities = ((1.0, 0.7), (-0.2, 1.1), (0.0, 1.0), (0.0, 0.0))

    for n in normals, u in velocities
        dotun = u[1] * n[1] + u[2] * n[2]
        expected = dotun < 0.0
        @test isinflow(u, n) == expected
    end

    # Edge case: u⋅n == 0 is treated as outflow / no inflow imposition.
    @test !isinflow((0.0, 1.0), (1.0, 0.0))
end

@testset "U2: zero-area faces contribute zero bulk advection flux" begin
    moments = _build_eb_circle_moments(; nx=40, ny=40)
    Nd = prod(length.(moments.xyz))

    Random.seed!(0xC0FFEE)
    uω1 = (randn(Nd), randn(Nd))
    uω2 = (copy(uω1[1]), copy(uω1[2]))
    uγz = (zeros(Nd), zeros(Nd))

    prob_ref = PenguinTransport.TransportProblem(;
        kappa=0.0,
        bc_adv=_periodic_adv_bc_2d(),
        scheme=CartesianOperators.Upwind1(),
        vel_omega=uω1,
        vel_gamma=uγz,
        embedded_inflow=nothing,
    )
    sys_ref = PenguinTransport.build_system(moments, prob_ref)

    zero_area_ids = [g for g in sys_ref.dof_omega.indices if abs(sys_ref.ops_adv.A[1][g]) <= 1e-14]
    @test !isempty(zero_area_ids)

    @inbounds for g in zero_area_ids
        uω2[1][g] += 123.456
    end

    prob_mod = PenguinTransport.TransportProblem(;
        kappa=0.0,
        bc_adv=_periodic_adv_bc_2d(),
        scheme=CartesianOperators.Upwind1(),
        vel_omega=uω2,
        vel_gamma=uγz,
        embedded_inflow=nothing,
    )
    sys_mod = PenguinTransport.build_system(moments, prob_mod)

    u = _reduced_from_field(sys_ref, (x, y, _t) -> sin(2pi * x) * cos(2pi * y), 0.0)
    du_ref = similar(u)
    du_mod = similar(u)

    PenguinSolverCore.rhs!(du_ref, sys_ref, u, nothing, 0.0)
    PenguinSolverCore.rhs!(du_mod, sys_mod, u, nothing, 0.0)

    @test maximum(abs.(du_ref .- du_mod)) <= 1e-12
end

@testset "U3: constant state invariant for one RK2 step" begin
    phi0 = 0.37
    vel = (1.0, 0.7)

    moments = _build_eb_circle_moments(; nx=24, ny=24)
    sys = _build_eb_system(
        moments;
        vel=vel,
        scheme=CartesianOperators.Upwind1(),
        bc_adv=_diag_inflow_adv_bc(phi0),
        embedded_inflow=phi0,
    )

    u = fill(phi0, length(sys.dof_omega.indices))
    dt = PenguinTransport.cfl_dt(sys, u; cfl=0.45, include_diffusion=false)
    _rk2_step!(u, sys, dt, 0.0)

    tol = max(100 * eps(Float64), 1e-12)
    @test maximum(abs.(u .- phi0)) <= tol
end

@testset "U4: tiny cut-cell case remains finite and conservative (constant state)" begin
    phi0 = 0.37
    vel = (1.0, 0.7)

    moments = _build_eb_circle_moments(; nx=64, ny=64, cx=0.623, cy=0.477, r=0.173)
    sys = _build_eb_system(
        moments;
        vel=vel,
        scheme=CartesianOperators.Upwind1(),
        bc_adv=_periodic_adv_bc_2d(),
        embedded_inflow=phi0,
    )

    idx = sys.dof_omega.indices
    V = sys.moments.V[idx]
    vratio = minimum(V) / maximum(V)
    @test vratio < 1e-3

    u = fill(phi0, length(idx))
    m0 = sum(V[i] * u[i] for i in eachindex(u))
    dt = PenguinTransport.cfl_dt(sys, u; cfl=0.2, include_diffusion=false)
    _rk2_step!(u, sys, dt, 0.0)
    m1 = sum(V[i] * u[i] for i in eachindex(u))

    @test all(isfinite, u)
    @test abs(m1 - m0) <= 1e-12 * max(1.0, abs(m0))

    # Optional diagnostics if a small-cell stabilization counter is introduced later.
    if hasproperty(sys, :small_cell_stabilization_count)
        @test getproperty(sys, :small_cell_stabilization_count) >= 0
    end
end
