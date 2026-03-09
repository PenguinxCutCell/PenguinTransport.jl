# ──────────────────────────────────────────────────────────────────
# Two-Phase Planar 1D Validation
#
# Validation layers in 1D with interface at x = a:
# 1) zero-source case (u1 != u2): check interface flux ratio and conservation
# 2) source in phase 1 only: check interface values/flux shift consistently
# ──────────────────────────────────────────────────────────────────

using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinBCs
using PenguinSolverCore
using LinearAlgebra

function planar_moments_left(grid; x0=0.47)
    levelset(x, y=0, z=0) = x - x0
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

function planar_moments_right(grid; x0=0.47)
    levelset(x, y=0, z=0) = x0 - x
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

function interface_flux_metrics(model::TransportModelTwoPhase, x)
    cap1 = model.cap1
    cap2 = model.cap2
    lay = model.layout
    nt = cap1.ntotal
    num_abs = 0.0
    den_abs = 0.0
    niface = 0
    for i in 1:nt
        Γ = 0.5 * (cap1.buf.Γ[i] + cap2.buf.Γ[i])
        (isfinite(Γ) && Γ > 0) || continue
        s1 = model.u1γ[1][i] * cap1.n_γ[i][1]
        s2 = model.u2γ[1][i] * cap2.n_γ[i][1]
        T1γ = x[lay.γ1[i]]
        T2γ = x[lay.γ2[i]]
        r = Γ * (s1 * T1γ + s2 * T2γ)
        num_abs += abs(r)
        den_abs += abs(Γ * s1 * T1γ) + abs(Γ * s2 * T2γ)
        niface += 1
    end
    return (rel_abs=num_abs / (den_abs + eps(Float64)), niface=niface)
end

function active_phase1_upstream_error(cap1, w1, x0, g)
    LI = LinearIndices(cap1.nnodes)
    err = 0.0
    used = 0
    for I in CartesianIndices(cap1.nnodes)
        lin = LI[I]
        I[1] == cap1.nnodes[1] && continue
        v = cap1.buf.V[lin]
        (isfinite(v) && v > 0) || continue
        x = cap1.C_ω[lin][1]
        x <= x0 - 0.1 || continue
        err = max(err, abs(w1[lin] + g)) # solver sign convention -> -g
        used += 1
    end
    return err, used
end

println("Two-phase planar 1D validation\n")

x0 = 0.47
grid = (0.0:0.025:1.0,)
cap1 = assembled_capacity(planar_moments_left(grid; x0=x0); bc=0.0)
cap2 = assembled_capacity(planar_moments_right(grid; x0=x0); bc=0.0)
nt = cap1.ntotal
u1 = 1.0
u2 = 2.0
g = 1.3

# Case A: zero sources
model0 = TransportModelTwoPhase(
    cap1, cap2,
    (u1 .* ones(nt),), (u1 .* ones(nt),),
    (u2 .* ones(nt),), (u2 .* ones(nt),);
    source1=0.0,
    source2=0.0,
    bc_border1=BorderConditions(; left=Inflow(g), right=Outflow()),
    bc_border2=BorderConditions(; left=Outflow(), right=Outflow()),
    scheme=Upwind1(),
)
sys0 = solve_steady!(model0)
lay = model0.layout
iface = findall(i -> cap1.buf.Γ[i] > 0 || cap2.buf.Γ[i] > 0, 1:nt)
i = iface[1]
s1 = model0.u1γ[1][i] * cap1.n_γ[i][1]
s2 = model0.u2γ[1][i] * cap2.n_γ[i][1]
T1γ0 = sys0.x[lay.γ1[i]]
T2γ0 = sys0.x[lay.γ2[i]]
met0 = interface_flux_metrics(model0, sys0.x)
err_up, n_up = active_phase1_upstream_error(cap1, sys0.x[lay.ω1], x0, g)

println("Case A: zero source")
println("  interface cells                    : ", met0.niface)
println("  interface flux rel. residual       : ", met0.rel_abs)
println("  ratio T2γ/T1γ                      : ", T2γ0 / T1γ0)
println("  expected ratio -(s1/s2)            : ", -(s1 / s2))
println("  upstream phase-1 |ω1 + g|_L∞       : ", err_up, " (", n_up, " cells)")
println()

# Case B: source in phase 1 only
σ = 0.4
modelσ = TransportModelTwoPhase(
    cap1, cap2,
    (u1 .* ones(nt),), (u1 .* ones(nt),),
    (u2 .* ones(nt),), (u2 .* ones(nt),);
    source1=σ,
    source2=0.0,
    bc_border1=BorderConditions(; left=Inflow(g), right=Outflow()),
    bc_border2=BorderConditions(; left=Outflow(), right=Outflow()),
    scheme=Upwind1(),
)
sysσ = solve_steady!(modelσ)
T1γσ = sysσ.x[lay.γ1[i]]
T2γσ = sysσ.x[lay.γ2[i]]
metσ = interface_flux_metrics(modelσ, sysσ.x)

println("Case B: source1 = $σ, source2 = 0")
println("  interface flux rel. residual       : ", metσ.rel_abs)
println("  ΔT1γ (with source - no source)     : ", T1γσ - T1γ0)
println("  ΔT2γ (with source - no source)     : ", T2γσ - T2γ0)

@assert met0.rel_abs < 1e-12
@assert metσ.rel_abs < 1e-12
@assert abs(T2γ0 - (-(s1 / s2)) * T1γ0) < 1e-12
@assert T1γσ > T1γ0
@assert T2γσ > T2γ0

println("\nValidation passed.")
