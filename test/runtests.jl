using Test
using LinearAlgebra
using SparseArrays
using Statistics
using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinSolverCore
using PenguinBCs

function full_moments(grid)
    levelset(args...) = -1.0
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

function circle_moments(grid; r=0.22, cx=0.5, cy=0.5)
    levelset(x, y, _=0) = sqrt((x - cx)^2 + (y - cy)^2) - r
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

function omega_view(model, state)
    return state[model.layout.offsets.ω]
end

@testset "1D periodic advection schemes" begin
    grid = (0.0:0.05:1.0,)
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    nt = cap.ntotal
    x = cap.xyz[1]
    u0 = sin.(2π .* x)

    uω = (ones(nt),)
    uγ = (ones(nt),)
    bc = BorderConditions(; left=Periodic(), right=Periodic())

    m_up = TransportModelMono(cap, uω, uγ; bc_border=bc, scheme=Upwind1())
    r_up = solve_unsteady!(m_up, u0, (0.0, 1.0); dt=0.02, scheme=:BE, save_history=false)
    uf_up = omega_view(m_up, r_up.states[end])

    @test all(isfinite, uf_up)
    @test norm(uf_up, Inf) > 0

    m_c = TransportModelMono(cap, uω, uγ; bc_border=bc, scheme=Centered())
    r_c = solve_unsteady!(m_c, u0, (0.0, 1.0); dt=0.02, scheme=:BE, save_history=false)
    uf_c = omega_view(m_c, r_c.states[end])
    @test all(isfinite, uf_c)
end

@testset "Box inflow/outflow switching" begin
    grid = (0.0:0.2:1.0, 0.0:0.2:1.0)
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    nt = cap.ntotal

    bc = BorderConditions(
        ;
        left=Inflow(1.0),
        right=Inflow(0.0),
        bottom=Outflow(),
        top=Outflow(),
    )

    model_pos = TransportModelMono(cap, (ones(nt), zeros(nt)), (ones(nt), zeros(nt)); bc_border=bc, scheme=Upwind1())
    res_pos = solve_unsteady!(model_pos, zeros(nt), (0.0, 0.6); dt=0.05, scheme=:BE, save_history=false)
    u_pos = omega_view(model_pos, res_pos.states[end])

    model_neg = TransportModelMono(cap, (-ones(nt), zeros(nt)), (-ones(nt), zeros(nt)); bc_border=bc, scheme=Upwind1())
    res_neg = solve_unsteady!(model_neg, zeros(nt), (0.0, 0.6); dt=0.05, scheme=:BE, save_history=false)
    u_neg = omega_view(model_neg, res_neg.states[end])

    @test norm(u_pos - u_neg, 2) > 1e-3
end

@testset "Embedded interface inflow/outflow closure" begin
    grid = (0.0:0.1:1.0, 0.0:0.1:1.0)
    cap = assembled_capacity(circle_moments(grid); bc=0.0)
    nt = cap.ntotal
    lay = layout_mono(nt).offsets

    ncomp = ntuple(d -> [cap.n_γ[i][d] for i in 1:nt], 2)
    zω = (zeros(nt), zeros(nt))

    model_out = TransportModelMono(cap, zω, ncomp; bc_interface=3.0)
    sys_out = LinearSystem(spzeros(Float64, 2 * nt, 2 * nt), zeros(Float64, 2 * nt))
    assemble_steady_mono!(sys_out, model_out, 0.0)

    iface = findall(i -> cap.buf.Γ[i] > 0, 1:nt)
    @test !isempty(iface)
    for i in iface
        r = lay.γ[i]
        γ = cap.buf.Γ[i]
        @test sys_out.A[r, lay.γ[i]] ≈ γ atol=1e-10
        @test sys_out.A[r, lay.ω[i]] ≈ -γ atol=1e-10
    end

    model_in = TransportModelMono(cap, zω, ntuple(d -> -ncomp[d], 2); bc_interface=2.5)
    sys_in = LinearSystem(spzeros(Float64, 2 * nt, 2 * nt), zeros(Float64, 2 * nt))
    assemble_steady_mono!(sys_in, model_in, 0.0)
    for i in iface
        r = lay.γ[i]
        γ = cap.buf.Γ[i]
        @test sys_in.A[r, lay.γ[i]] ≈ γ atol=1e-10
        @test sys_in.A[r, lay.ω[i]] ≈ 0.0 atol=1e-10
        @test sys_in.b[r] ≈ γ * 2.5 atol=1e-10
    end
end

@testset "Masking and inactive-row identity" begin
    grid = (0.0:0.1:1.0, 0.0:0.1:1.0)
    cap = assembled_capacity(circle_moments(grid; r=0.35, cx=0.25, cy=0.25); bc=0.0)
    nt = cap.ntotal
    lay = layout_mono(nt).offsets
    model = TransportModelMono(cap, (ones(nt), ones(nt)), (ones(nt), ones(nt)); bc_interface=1.0)

    sys = LinearSystem(spzeros(Float64, 2 * nt, 2 * nt), zeros(Float64, 2 * nt))
    assemble_unsteady_mono!(sys, model, zeros(nt), 0.0, 0.1, :BE)

    @test all(isfinite, sys.A.nzval)
    @test all(isfinite, sys.b)

    LI = LinearIndices(cap.nnodes)
    checked = 0
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        halo = any(d -> I[d] == cap.nnodes[d], 1:2)
        inactiveω = halo || !(cap.buf.V[lin] > 0)
        inactiveγ = halo || !(cap.buf.Γ[lin] > 0)
        if inactiveω
            r = lay.ω[lin]
            @test sys.A[r, r] ≈ 1.0 atol=1e-12
            checked += 1
        end
        if inactiveγ
            r = lay.γ[lin]
            @test sys.A[r, r] ≈ 1.0 atol=1e-12
            checked += 1
        end
    end
    @test checked > 0
end

# ──────────────────────────────────────────────────────────────────
# Helpers for convergence / validation tests
# ──────────────────────────────────────────────────────────────────

"""Volume-weighted L1 error   ∑ V_i |u_i - u_exact_i| / ∑ V_i"""
function _l1_weighted_error(cap, u_num, u_exact)
    V = cap.buf.V
    nt = cap.ntotal
    num = 0.0
    den = 0.0
    LI = LinearIndices(cap.nnodes)
    N = length(cap.nnodes)
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        any(d -> I[d] == cap.nnodes[d], 1:N) && continue
        v = V[lin]
        (isfinite(v) && v > 0) || continue
        num += v * abs(u_num[lin] - u_exact[lin])
        den += v
    end
    return num / den
end

"""Volume-weighted total mass ∑ V_i * u_i (over active cells)"""
function _total_mass(cap, u)
    V = cap.buf.V
    nt = cap.ntotal
    s = 0.0
    LI = LinearIndices(cap.nnodes)
    N = length(cap.nnodes)
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        any(d -> I[d] == cap.nnodes[d], 1:N) && continue
        v = V[lin]
        (isfinite(v) && v > 0) || continue
        s += v * u[lin]
    end
    return s
end

# ──────────────────────────────────────────────────────────────────
# Convergence order test (1D periodic sine)
# ──────────────────────────────────────────────────────────────────

@testset "Convergence order: upwind ≈ 1, centered > 1.5" begin
    dxs = [0.1, 0.05, 0.025]
    cfl  = 0.5   # dt = cfl * dx

    for (label, scheme, T_end, u_exact_fn, tscheme, expected_order) in [
        # Upwind: short time to avoid excessive dissipation; CN for 2nd-order time
        ("Upwind1", Upwind1(), 0.2,
         (x, T) -> sin(2π * (x - T)),
         :CN, 0.8),
        # Centered: full period; CN + centered spatial → O(dx²)
        ("Centered", Centered(), 1.0,
         (x, T) -> sin(2π * x),  # exact after one full period
         :CN, 1.5),
    ]
        errs = Float64[]
        for dx in dxs
            grid = (0.0:dx:1.0,)
            cap = assembled_capacity(full_moments(grid); bc=0.0)
            nt = cap.ntotal
            x = cap.xyz[1]
            u0 = sin.(2π .* x)
            u_exact = [u_exact_fn(xi, T_end) for xi in x]

            uω = (ones(nt),)
            uγ = (ones(nt),)
            bc = BorderConditions(; left=Periodic(), right=Periodic())
            dt = cfl * dx

            m = TransportModelMono(cap, uω, uγ; bc_border=bc, scheme=scheme)
            r = solve_unsteady!(m, u0, (0.0, T_end); dt=dt, scheme=tscheme, save_history=false)
            uf = omega_view(m, r.states[end])
            push!(errs, _l1_weighted_error(cap, uf, u_exact))
        end

        # Compute convergence orders between successive refinements
        orders = [log(errs[i] / errs[i+1]) / log(dxs[i] / dxs[i+1]) for i in 1:length(dxs)-1]
        avg_order = mean(orders)
        @test avg_order > expected_order
    end
end
# ──────────────────────────────────────────────────────────────────
# Mass conservation (periodic advection)
# ──────────────────────────────────────────────────────────────────

@testset "Mass conservation (periodic advection)" begin
    dx = 0.05
    grid = (0.0:dx:1.0,)
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    nt = cap.ntotal
    x = cap.xyz[1]
    u0 = exp.(-100.0 .* (x .- 0.5).^2)   # Gaussian bump

    uω = (ones(nt),)
    uγ = (ones(nt),)
    bc = BorderConditions(; left=Periodic(), right=Periodic())

    for scheme_tag in (Upwind1(), Centered())
        m = TransportModelMono(cap, uω, uγ; bc_border=bc, scheme=scheme_tag)
        r = solve_unsteady!(m, u0, (0.0, 0.5); dt=0.01, scheme=:BE, save_history=true)

        mass0 = _total_mass(cap, omega_view(m, r.states[1]))
        for st in r.states
            mass_t = _total_mass(cap, omega_view(m, st))
            @test abs(mass_t - mass0) / abs(mass0) < 1e-10 # machine precision for periodic BCs
        end
    end
end

# ──────────────────────────────────────────────────────────────────
# 2D rotating Gaussian — error decreases with refinement
# ──────────────────────────────────────────────────────────────────

@testset "2D rotating Gaussian error decreases" begin
    dxs = [0.1, 0.05]
    errs = Float64[]

    for dx in dxs
        grid = (0.0:dx:1.0, 0.0:dx:1.0)
        cap = assembled_capacity(full_moments(grid); bc=0.0)
        nt = cap.ntotal
        x = [cap.C_ω[i][1] for i in 1:nt]
        y = [cap.C_ω[i][2] for i in 1:nt]

        # Solid-body rotation about (0.5, 0.5) with period T=2π
        ux = -(y .- 0.5)
        uy =  (x .- 0.5)

        sigma = 0.12
        cx0, cy0 = 0.5, 0.75
        u0 = exp.(-((x .- cx0).^2 .+ (y .- cy0).^2) ./ (2 * sigma^2))
        u_exact = copy(u0)   # exact after one full revolution T=2π

        bc = BorderConditions(; left=Periodic(), right=Periodic(), bottom=Periodic(), top=Periodic())
        T_rev = 2π
        umax = maximum(filter(isfinite, abs.(ux))) + maximum(filter(isfinite, abs.(uy)))
        dt = 0.5 * dx / (umax + 1e-12)

        m = TransportModelMono(cap, (ux, uy), (ux, uy); bc_border=bc, scheme=Upwind1())
        r = solve_unsteady!(m, u0, (0.0, T_rev); dt=dt, scheme=:BE, save_history=false)
        uf = omega_view(m, r.states[end])
        push!(errs, _l1_weighted_error(cap, uf, u_exact))
    end

    # Error should decrease when we refine
    @test errs[2] < errs[1]
end